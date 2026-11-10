package com.argus.agi.model

import android.content.Context
import android.util.Log
import com.argus.agi.data.local.entity.CorpusChunkEntity
import com.argus.agi.data.local.entity.CorpusDocumentEntity
import com.argus.agi.sensor.DeviceGuard
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull
import java.io.Closeable
import java.io.Reader
import java.io.StringReader
import java.security.MessageDigest
import java.util.Locale
import java.util.UUID
import kotlin.math.max
import kotlin.math.min

enum class TrainingStatus { PERSISTING, QUEUED, TRAINING, COMPLETED, FAILED }

data class TrainingData(
    val id: String = UUID.randomUUID().toString(),
    val source: String,
    val title: String,
    val content: String,
    val chunkIndex: Int = 1,
    val totalChunks: Int = 1,
    val documentTitle: String? = null,
    val timestamp: Long = System.currentTimeMillis(),
    val status: TrainingStatus = TrainingStatus.QUEUED,
    val finalLoss: Float? = null,
    val stepDurationMs: Long = 0L,
    val documentId: String? = null
)

data class ActiveDocumentTraining(
    val docTitle: String,
    val totalCharacters: Int,
    val totalChunks: Int,
    val completedChunks: Int,
    val currentChunkIndex: Int,
    val latestLoss: Float? = null,
    val runningAverageLoss: Float? = null
)

class TrainingManager(
    context: Context,
    private val neuralEngine: ArgusNeuralEngine,
    private val deviceGuard: DeviceGuard,
    private val worldModel: WorldModel,
    private val trainingDispatcher: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(1),
    private val repository: com.argus.agi.data.repository.ArgusRepository? = null
) : Closeable {

    private val repo: com.argus.agi.data.repository.ArgusRepository? =
        repository ?: runCatching { com.argus.agi.data.repository.ArgusRepository(context) }.getOrNull()

    private sealed interface QueuePersistenceOperation {
        data class Save(
            val generation: Long,
            val items: List<TrainingData>
        ) : QueuePersistenceOperation

        data class Clear(
            val generation: Long
        ) : QueuePersistenceOperation

        data class Barrier(
            val completion: CompletableDeferred<Unit>
        ) : QueuePersistenceOperation
    }

    companion object {
        private const val TAG = "TrainingManager"
        const val MAX_COMPLETED_SIZE = 100
        const val MAX_LOSS_HISTORY = 100
        const val MAX_DOCUMENT_EPOCHS = 5
        private const val IDLE_POLL_MS = 500L
    }

    // StateFlow<List<TrainingData>> sebagai satu-satunya sumber kebenaran untuk queue.
    // Channel hanya berfungsi sebagai wakeup signal, bukan storage.
    private val queueLock = Any()
    private var lastScheduledTimestamp: Long = System.currentTimeMillis()
    private var queueGeneration: Long = 0L
    private val persistenceOperations =
        Channel<QueuePersistenceOperation>(capacity = Channel.UNLIMITED)

    private val _nextQueue = MutableStateFlow<List<TrainingData>>(emptyList())
    val nextQueue: StateFlow<List<TrainingData>> = _nextQueue.asStateFlow()

    private val wakeupSignal = Channel<Unit>(capacity = Channel.CONFLATED)

    private val _currentTrainingItem = MutableStateFlow<TrainingData?>(null)
    val currentTrainingItem: StateFlow<TrainingData?> = _currentTrainingItem.asStateFlow()

    private val _activeDocument = MutableStateFlow<ActiveDocumentTraining?>(null)
    val activeDocument: StateFlow<ActiveDocumentTraining?> = _activeDocument.asStateFlow()

    private val _completedQueue = MutableStateFlow<List<TrainingData>>(emptyList())
    val completedQueue: StateFlow<List<TrainingData>> = _completedQueue.asStateFlow()

    private val _lossHistory = MutableStateFlow<List<Float>>(emptyList())
    val lossHistory: StateFlow<List<Float>> = _lossHistory.asStateFlow()

    private val _isTrainingPaused = MutableStateFlow(false)
    val isTrainingPaused: StateFlow<Boolean> = _isTrainingPaused.asStateFlow()

    private val _totalStepsExecuted = MutableStateFlow(0)
    val totalStepsExecuted: StateFlow<Int> = _totalStepsExecuted.asStateFlow()

    @Volatile private var closed = false

    private val exceptionHandler = CoroutineExceptionHandler { _, t ->
        Log.e(TAG, "Unhandled exception in training scope", t)
    }
    private val scope = CoroutineScope(SupervisorJob() + trainingDispatcher + exceptionHandler)
    private var workerJob: Job? = null
    private var persistenceJob: Job? = null

    init {
        persistenceJob = scope.launch {
            for (operation in persistenceOperations) {
                processQueuePersistenceOperation(operation)
            }
        }

        val restoreGeneration = synchronized(queueLock) { queueGeneration }
        scope.launch {
            repo?.let { repository ->
                runCatching {
                    repository.recoverInterruptedTrainingItems()
                    repository.getPendingQueueSnapshot()
                }
                    .onSuccess { pending ->
                        if (pending.isNotEmpty()) {
                            synchronized(queueLock) {
                                if (
                                    restoreGeneration == queueGeneration &&
                                    _nextQueue.value.isEmpty()
                                ) {
                                    val orderedPending = pending.sortedBy { it.timestamp }
                                    _nextQueue.value = orderedPending
                                    lastScheduledTimestamp = maxOf(
                                        lastScheduledTimestamp,
                                        orderedPending.maxOf { it.timestamp }
                                    )
                                    wakeupSignal.trySend(Unit)
                                }
                            }
                        }
                    }
                    .onFailure { error ->
                        Log.e(TAG, "Failed to recover and restore persisted training queue", error)
                    }
            }
        }

        startTrainingWorker()
    }

    fun pauseTraining() { _isTrainingPaused.value = true; wakeupSignal.trySend(Unit) }
    fun resumeTraining() { _isTrainingPaused.value = false; wakeupSignal.trySend(Unit) }
    fun togglePauseTraining() {
        _isTrainingPaused.update { !it }
        wakeupSignal.trySend(Unit)
    }

    /**
     * Set force-training override SEKALI untuk menembus thermal/battery guard
     * pada langkah berikutnya. Override akan otomatis di-reset setelah satu trainStep.
     */
    fun triggerImmediateTrainingStep() {
        _isTrainingPaused.value = false
        deviceGuard.setForceTrainingOverride(true)
        wakeupSignal.trySend(Unit)
    }

    /**
     * Atomik: kosongkan queue + reset active document.
     * Item yang sedang diproses tetap diselesaikan (tidak di-cancel).
     */
    fun clearQueue() {
        synchronized(queueLock) {
            queueGeneration += 1L
            _nextQueue.value = emptyList()
            persistenceOperations.trySend(
                QueuePersistenceOperation.Clear(queueGeneration)
            )
        }
        _activeDocument.value = null
        wakeupSignal.trySend(Unit)
    }

    fun clearCompleted() { _completedQueue.value = emptyList() }

    internal suspend fun awaitQueuePersistenceIdle() {
        if (closed) return
        val barrier = CompletableDeferred<Unit>()
        persistenceOperations.send(QueuePersistenceOperation.Barrier(barrier))
        barrier.await()
    }

    // =====================================================================
    // Enqueue API
    // =====================================================================

    suspend fun enqueueDocument(
        source: String,
        title: String,
        fullText: String,
        chunkSize: Int = 380,
        overlap: Int = 50,
        epochs: Int = 1,
        language: String? = null,
        canonicalUrl: String? = null,
        rightsStatus: String = "UNVERIFIED"
    ): Int {
        require(chunkSize > overlap) {
            "chunkSize ($chunkSize) harus > overlap ($overlap) agar iterasi maju"
        }
        require(epochs in 1..MAX_DOCUMENT_EPOCHS) {
            "epochs harus berada pada rentang 1..$MAX_DOCUMENT_EPOCHS"
        }
        require(rightsStatus in setOf("UNVERIFIED", "USER_DECLARED", "LICENSE_REVIEWED")) {
            "Unsupported corpus rights status: $rightsStatus"
        }

        val clean = fullText.trim()
        if (clean.isBlank() || closed) return 0

        val processedChunks = TextModalityProcessor.process(clean, title, chunkSize, overlap)
        val chunksPerEpoch = processedChunks.size
        if (chunksPerEpoch == 0) return 0

        val documentId = sha256Utf16Be(fullText)
        val corpusDocument = CorpusDocumentEntity(
            id = documentId,
            contentFingerprint = documentId,
            fingerprintAlgorithm = "SHA-256/UTF-16BE",
            source = source,
            title = title,
            language = language,
            canonicalUrl = canonicalUrl,
            characterCount = clean.length,
            chunkCount = chunksPerEpoch,
            plannedEpochs = epochs,
            rightsStatus = rightsStatus,
            status = "READY",
            createdAtEpochMs = System.currentTimeMillis(),
            lastTrainedAtEpochMs = null,
            completedPassCount = 0
        )
        val corpusChunks = processedChunks.mapIndexed { index, chunk ->
            CorpusChunkEntity(
                id = "$documentId:${index + 1}",
                documentId = documentId,
                chunkIndex = index + 1,
                content = chunk.content,
                characterCount = chunk.content.length
            )
        }

        val activeRepository = requireNotNull(repo) {
            "Local corpus storage is unavailable; document training was not queued."
        }
        val stored = activeRepository.storeCorpusDocument(corpusDocument, corpusChunks)
        if (!stored) {
            Log.i(TAG, "Duplicate corpus document skipped: $documentId")
            return 0
        }

        val totalScheduledChunks = Math.multiplyExact(chunksPerEpoch, epochs)
        val newItems = ArrayList<TrainingData>(totalScheduledChunks)

        // Epoch-major ordering: the model sees the complete document before its next pass.
        for (epochIndex in 0 until epochs) {
            processedChunks.forEachIndexed { chunkIndex, chunk ->
                val epochTitle = if (epochs == 1) {
                    chunk.segmentTitle
                } else {
                    "${chunk.segmentTitle} [Epoch ${epochIndex + 1}/$epochs]"
                }

                newItems += TrainingData(
                    source = source,
                    title = epochTitle,
                    content = chunk.content,
                    chunkIndex = epochIndex * chunksPerEpoch + chunkIndex + 1,
                    totalChunks = totalScheduledChunks,
                    documentTitle = title,
                    status = TrainingStatus.QUEUED,
                    documentId = documentId
                )
            }
        }

        synchronized(queueLock) {
            if (closed) return 0
            val orderedItems = appendPersistingQueueItems(newItems)

            if (_activeDocument.value == null && orderedItems.isNotEmpty()) {
                _activeDocument.value = ActiveDocumentTraining(
                    docTitle = title,
                    totalCharacters = clean.length,
                    totalChunks = orderedItems.size,
                    completedChunks = 0,
                    currentChunkIndex = 1
                )
            }
            return orderedItems.size
        }
    }

    /**
     * Reads a source incrementally with a 4 KB input buffer and a sliding text window.
     *
     * Corpus chunks are retained in Room independently from the transient training queue.
     * The generated chunk strings are still accumulated in memory during this operation;
     * this is buffered ingestion, not a constant-memory disk-backed queue.
     *
     * The fingerprint is computed over the exact UTF-16 code units read from the Reader,
     * using SHA-256/UTF-16BE. This produces the same identity as enqueueDocument for the
     * same character sequence, including when surrogate pairs cross read-buffer boundaries.
     *
     * @return Pair(number of generated chunks, total characters read)
     */
    suspend fun enqueueDocumentStream(
        source: String,
        title: String,
        reader: Reader,
        chunkSize: Int = 380,
        overlap: Int = 50,
        epochs: Int = 1,
        language: String? = null,
        canonicalUrl: String? = null,
        rightsStatus: String = "UNVERIFIED"
    ): Pair<Int, Long> {
        require(chunkSize > overlap) {
            "chunkSize ($chunkSize) harus > overlap ($overlap) agar iterasi maju"
        }
        require(overlap <= chunkSize / 3) {
            "overlap ($overlap) harus <= chunkSize / 3 (${chunkSize / 3}) agar jaminan sliding window selalu maju"
        }
        require(epochs in 1..MAX_DOCUMENT_EPOCHS) {
            "epochs harus berada pada rentang 1..$MAX_DOCUMENT_EPOCHS"
        }
        require(rightsStatus in setOf("UNVERIFIED", "USER_DECLARED", "LICENSE_REVIEWED")) {
            "Unsupported corpus rights status: $rightsStatus"
        }
        if (closed) return Pair(0, 0L)

        val rawChunks = mutableListOf<String>()
        val buffer = CharArray(4096)
        val window = StringBuilder()
        val fingerprintDigest = MessageDigest.getInstance("SHA-256")
        var totalCharsRead = 0L

        while (true) {
            val readCount = reader.read(buffer)
            if (readCount < 0) break
            if (closed) {
                rawChunks.clear()
                return Pair(0, totalCharsRead)
            }

            updateUtf16BeDigest(fingerprintDigest, buffer, readCount)
            window.append(buffer, 0, readCount)
            totalCharsRead += readCount.toLong()

            while (window.length >= chunkSize) {
                val minimumBreakPosition = chunkSize / 3
                val searchText = window.substring(minimumBreakPosition, chunkSize)
                val lastNewLine = searchText.lastIndexOf('\n')
                val lastSentenceBreak = max(
                    searchText.lastIndexOf(". "),
                    max(searchText.lastIndexOf("? "), searchText.lastIndexOf("! "))
                )
                val punctuationBreak = max(lastNewLine, lastSentenceBreak)

                val splitPosition = if (punctuationBreak >= 0) {
                    minimumBreakPosition + punctuationBreak + 1
                } else {
                    val lastSpace = searchText.lastIndexOf(' ')
                    if (lastSpace >= 0) minimumBreakPosition + lastSpace + 1 else chunkSize
                }.coerceIn(1, chunkSize)

                val rawSegment = TextModalityProcessor.normalizeTypography(
                    window.substring(0, splitPosition)
                )
                val normalizedSegment = rawSegment.lines().joinToString("\n") {
                    TextModalityProcessor.formatStructuredLineIfJson(it)
                }
                if (normalizedSegment.isNotBlank()) rawChunks.add(normalizedSegment)

                val nextStart = (splitPosition - overlap).coerceAtLeast(0)
                val safeNextStart = if (nextStart == 0 && splitPosition > 0) {
                    splitPosition
                } else {
                    nextStart
                }
                val tail = window.substring(safeNextStart)
                window.setLength(0)
                window.append(tail)
            }
        }

        if (closed) {
            rawChunks.clear()
            return Pair(0, totalCharsRead)
        }

        val trailingRaw = TextModalityProcessor.normalizeTypography(window.toString().trim())
        val trailing = trailingRaw.lines().joinToString("\n") {
            TextModalityProcessor.formatStructuredLineIfJson(it)
        }
        if (trailing.isNotBlank()) rawChunks.add(trailing)

        val chunksPerEpoch = rawChunks.size
        if (chunksPerEpoch == 0) return Pair(0, totalCharsRead)

        val documentId = fingerprintDigest.digest().toHexString()
        val corpusDocument = CorpusDocumentEntity(
            id = documentId,
            contentFingerprint = documentId,
            fingerprintAlgorithm = "SHA-256/UTF-16BE",
            source = source,
            title = title,
            language = language,
            canonicalUrl = canonicalUrl,
            characterCount = totalCharsRead.coerceAtMost(Int.MAX_VALUE.toLong()).toInt(),
            chunkCount = chunksPerEpoch,
            plannedEpochs = epochs,
            rightsStatus = rightsStatus,
            status = "READY",
            createdAtEpochMs = System.currentTimeMillis(),
            lastTrainedAtEpochMs = null,
            completedPassCount = 0
        )
        val corpusChunks = rawChunks.mapIndexed { index, chunkText ->
            CorpusChunkEntity(
                id = "$documentId:${index + 1}",
                documentId = documentId,
                chunkIndex = index + 1,
                content = chunkText,
                characterCount = chunkText.length
            )
        }

        val activeRepository = requireNotNull(repo) {
            "Local corpus storage is unavailable; document training was not queued."
        }
        val stored = activeRepository.storeCorpusDocument(corpusDocument, corpusChunks)
        if (!stored) {
            Log.i(TAG, "Duplicate corpus document skipped: $documentId")
            rawChunks.clear()
            return Pair(0, totalCharsRead)
        }

        val totalScheduledChunks = Math.multiplyExact(chunksPerEpoch, epochs)
        val newItems = ArrayList<TrainingData>(totalScheduledChunks)

        for (epochIndex in 0 until epochs) {
            rawChunks.forEachIndexed { chunkIndex, chunkText ->
                val chunkTitle = if (epochs == 1) {
                    "$title [Blok ${chunkIndex + 1}/$chunksPerEpoch]"
                } else {
                    "$title [Blok ${chunkIndex + 1}/$chunksPerEpoch] [Epoch ${epochIndex + 1}/$epochs]"
                }

                newItems += TrainingData(
                    source = source,
                    title = chunkTitle,
                    content = chunkText,
                    chunkIndex = epochIndex * chunksPerEpoch + chunkIndex + 1,
                    totalChunks = totalScheduledChunks,
                    documentTitle = title,
                    status = TrainingStatus.QUEUED,
                    documentId = documentId
                )
            }
        }
        rawChunks.clear()

        synchronized(queueLock) {
            if (closed) return Pair(0, totalCharsRead)
            val orderedItems = appendPersistingQueueItems(newItems)

            if (_activeDocument.value == null && orderedItems.isNotEmpty()) {
                _activeDocument.value = ActiveDocumentTraining(
                    docTitle = title,
                    totalCharacters = totalCharsRead.coerceAtMost(Int.MAX_VALUE.toLong()).toInt(),
                    totalChunks = orderedItems.size,
                    completedChunks = 0,
                    currentChunkIndex = 1
                )
            }
            return Pair(orderedItems.size, totalCharsRead)
        }
    }

    /**
     * Re-enqueues retained corpus chunks without re-reading the original file or web page.
     * Queue clearing therefore does not delete the data needed for another training pass.
     */
    suspend fun replayCorpusDocument(
        documentId: String,
        epochs: Int = 1
    ): Int {
        require(epochs in 1..MAX_DOCUMENT_EPOCHS) {
            "epochs must be in 1..$MAX_DOCUMENT_EPOCHS."
        }
        if (closed) return 0

        val activeRepository = requireNotNull(repo) {
            "Local corpus storage is unavailable; corpus replay cannot be scheduled."
        }
        val document = activeRepository.getCorpusDocument(documentId) ?: return 0
        val chunks = activeRepository.getCorpusChunks(documentId)
        if (chunks.size != document.chunkCount || chunks.isEmpty()) {
            throw IllegalStateException(
                "Corpus document $documentId is incomplete: expected ${document.chunkCount} chunks, found ${chunks.size}."
            )
        }

        val totalScheduledChunks = Math.multiplyExact(chunks.size, epochs)
        val items = ArrayList<TrainingData>(totalScheduledChunks)
        for (epochIndex in 0 until epochs) {
            chunks.forEachIndexed { index, chunk ->
                items += TrainingData(
                    source = document.source,
                    title = "${document.title} [Replay ${index + 1}/${chunks.size}] [Epoch ${epochIndex + 1}/$epochs]",
                    content = chunk.content,
                    chunkIndex = epochIndex * chunks.size + index + 1,
                    totalChunks = totalScheduledChunks,
                    documentTitle = document.title,
                    status = TrainingStatus.QUEUED,
                    documentId = document.id
                )
            }
        }

        activeRepository.markCorpusDocumentQueued(document.id)

        synchronized(queueLock) {
            if (closed) return 0
            val orderedItems = appendPersistingQueueItems(items)
            if (_activeDocument.value == null && orderedItems.isNotEmpty()) {
                _activeDocument.value = ActiveDocumentTraining(
                    docTitle = document.title,
                    totalCharacters = document.characterCount,
                    totalChunks = orderedItems.size,
                    completedChunks = 0,
                    currentChunkIndex = 1
                )
            }
            return orderedItems.size
        }
    }

    fun enqueueData(source: String, title: String, content: String) {
        val trimmed = content.trim()
        if (trimmed.isBlank() || closed) return

        if (trimmed.length > 380) {
            scope.launch {
                runCatching { enqueueDocument(source, title, trimmed) }
                    .onFailure { error ->
                        Log.e(TAG, "Failed to persist long-form training input", error)
                    }
            }
            return
        }

        val processed = TextModalityProcessor.process(trimmed, title, chunkSize = 380, overlap = 40)
        if (processed.isEmpty()) return

        val newItems = processed.map { pChunk ->
            TrainingData(
                source = source,
                title = if (processed.size > 1) pChunk.segmentTitle else title.ifBlank { "Data Stimulus" },
                content = pChunk.content,
                status = TrainingStatus.QUEUED
            )
        }

        // Dedup atomic di bawah lock yang sama dengan dequeue
        synchronized(queueLock) {
            if (closed) return
            val current = _nextQueue.value
            val nonDuplicates = newItems.filter { item ->
                current.none { it.content == item.content && it.documentTitle == null }
            }
            if (nonDuplicates.isNotEmpty()) {
                appendPersistingQueueItems(nonDuplicates)
            }
        }
    }

    private fun sha256Utf16Be(text: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
        val bytes = text.toByteArray(Charsets.UTF_16BE)
        return digest.digest(bytes).toHexString()
    }

    private fun updateUtf16BeDigest(
        digest: MessageDigest,
        chars: CharArray,
        length: Int
    ) {
        val bytes = ByteArray(length * 2)
        for (index in 0 until length) {
            val value = chars[index].code
            bytes[index * 2] = (value ushr 8).toByte()
            bytes[index * 2 + 1] = value.toByte()
        }
        digest.update(bytes)
    }

    private fun ByteArray.toHexString(): String {
        val alphabet = "0123456789abcdef"
        return buildString(size * 2) {
            for (byte in this@toHexString) {
                val value = byte.toInt() and 0xFF
                append(alphabet[value ushr 4])
                append(alphabet[value and 0x0F])
            }
        }
    }

    // =====================================================================
    // Worker
    // =====================================================================

    private fun startTrainingWorker() {
        workerJob = scope.launch {
            while (isActive && !closed) {
                // 1. Tunggu pause lift
                if (_isTrainingPaused.value) {
                    _isTrainingPaused.first { !it || closed }
                    if (closed) break
                }

                // 2. Tunggu safety (respect force override)
                deviceGuard.safetyState.first { it.isSafeToTrain || closed }
                if (closed) break

                // Konsumsi force override segera setelah lolos guard termal/baterai
                // agar tidak membocorkan hak istimewa ini ke antrean berikutnya atau saat idle
                resetForceOverrideIfSet()

                // 3. Pop item dari queue secara atomik
                val item = popNext()
                if (item == null) {
                    // Keep document progress while the head item is still being persisted.
                    if (_nextQueue.value.isEmpty()) {
                        _currentTrainingItem.value = null
                        _activeDocument.value = null
                    }
                    withTimeoutOrNull(IDLE_POLL_MS) {
                        wakeupSignal.receiveCatching()
                    }
                    continue
                }

                // 4. Proses
                val activeItem = item.copy(status = TrainingStatus.TRAINING)
                _currentTrainingItem.value = activeItem
                updateActiveDocumentEntering(activeItem)

                try {
                    processItem(activeItem)
                } catch (t: Throwable) {
                    Log.e(TAG, "processItem gagal untuk '${item.title}'", t)
                    val failed = activeItem.copy(status = TrainingStatus.FAILED)
                    pushCompleted(failed)
                    scope.launch {
                        repo?.let { repository ->
                            runCatching { repository.markTrainingFailed(failed.id) }
                                .onFailure { error ->
                                    Log.w(TAG, "Failed to persist exception-failed training item", error)
                                }
                            failed.documentId?.let { documentId ->
                                runCatching { repository.markCorpusDocumentPartial(documentId) }
                                    .onFailure { error ->
                                        Log.w(TAG, "Failed to mark exception-failed corpus document partial", error)
                                    }
                            }
                        }
                    }
                } finally {
                    // PENTING: Jangan set _currentTrainingItem.value = null di sini!
                    // Pertahankan activeItem agar UI tidak berkedip (muncul hilang muncul hilang).
                    // Item hanya di-null-kan saat antrean benar-benar habis (item == null di atas).
                    resetForceOverrideIfSet()
                }

                // 5. Thermal throttle
                val temp = deviceGuard.safetyState.value.temperatureCelsius
                val throttleMs = when {
                    temp >= 45f -> 600L
                    temp >= 40f -> 300L
                    else -> 100L
                }
                delay(throttleMs)
            }
        }
    }

    private fun appendPersistingQueueItems(items: List<TrainingData>): List<TrainingData> {
        if (items.isEmpty()) return emptyList()

        val orderedItems = assignPersistentQueueOrder(items)
        val persistingItems = orderedItems.map { it.copy(status = TrainingStatus.PERSISTING) }
        _nextQueue.value = _nextQueue.value + persistingItems

        val operation = QueuePersistenceOperation.Save(
            generation = queueGeneration,
            items = orderedItems
        )
        val sendResult = persistenceOperations.trySend(operation)
        if (sendResult.isFailure) {
            val itemIds = orderedItems.mapTo(HashSet(orderedItems.size)) { it.id }
            _nextQueue.value = _nextQueue.value.filterNot { it.id in itemIds }
            orderedItems.forEach { item ->
                pushCompleted(item.copy(status = TrainingStatus.FAILED))
            }
            Log.e(TAG, "Training persistence worker is not accepting queue items.")
            return emptyList()
        }

        return orderedItems
    }

    private suspend fun processQueuePersistenceOperation(
        operation: QueuePersistenceOperation
    ) {
        when (operation) {
            is QueuePersistenceOperation.Clear -> {
                repo?.let { repository ->
                    runCatching { repository.clearPendingQueue() }
                        .onFailure { error ->
                            Log.e(TAG, "Failed to clear persisted training queue", error)
                        }
                }
            }

            is QueuePersistenceOperation.Barrier -> {
                operation.completion.complete(Unit)
            }

            is QueuePersistenceOperation.Save -> {
                val isCurrentGeneration = synchronized(queueLock) {
                    operation.generation == queueGeneration
                }
                if (!isCurrentGeneration) return

                val persistenceFailure = repo?.let { repository ->
                    runCatching {
                        repository.saveTrainingItems(operation.items)
                    }.exceptionOrNull()
                }

                synchronized(queueLock) {
                    if (operation.generation != queueGeneration) return@synchronized

                    val itemIds = operation.items.mapTo(HashSet(operation.items.size)) { it.id }
                    if (persistenceFailure == null) {
                        _nextQueue.value = _nextQueue.value.map { item ->
                            if (item.id in itemIds) {
                                item.copy(status = TrainingStatus.QUEUED)
                            } else {
                                item
                            }
                        }
                    } else {
                        _nextQueue.value = _nextQueue.value.filterNot { it.id in itemIds }
                        operation.items.forEach { item ->
                            pushCompleted(item.copy(status = TrainingStatus.FAILED))
                        }
                    }
                }

                if (persistenceFailure != null) {
                    Log.e(
                        TAG,
                        "Failed to persist " + operation.items.size + " training items",
                        persistenceFailure
                    )
                }
                wakeupSignal.trySend(Unit)
            }
        }
    }

    private fun assignPersistentQueueOrder(items: List<TrainingData>): List<TrainingData> {
        if (items.isEmpty()) return emptyList()

        val firstTimestamp = maxOf(System.currentTimeMillis(), lastScheduledTimestamp + 1L)
        val ordered = items.mapIndexed { index, item ->
            item.copy(timestamp = firstTimestamp + index.toLong())
        }
        lastScheduledTimestamp = firstTimestamp + ordered.lastIndex
        return ordered
    }

    private fun popNext(): TrainingData? = synchronized(queueLock) {
        val list = _nextQueue.value
        val first = list.firstOrNull() ?: return@synchronized null
        if (first.status != TrainingStatus.QUEUED) return@synchronized null

        _nextQueue.value = list.drop(1)
        first
    }

    private fun updateActiveDocumentEntering(item: TrainingData) {
        if (item.documentTitle == null || item.totalChunks <= 1) return
        _activeDocument.update { current ->
            if (current != null && current.docTitle == item.documentTitle) {
                current.copy(currentChunkIndex = item.chunkIndex)
            } else {
                // Dokumen baru mulai; ganti display tanpa merusak state dokumen sebelumnya
                ActiveDocumentTraining(
                    docTitle = item.documentTitle,
                    totalCharacters = item.content.length * item.totalChunks,
                    totalChunks = item.totalChunks,
                    completedChunks = item.chunkIndex - 1,
                    currentChunkIndex = item.chunkIndex
                )
            }
        }
    }

    private suspend fun processItem(item: TrainingData) {
        // Persist the active row as TRAINING before entering the non-interruptible optimizer
        // step. clearPendingQueue() can then safely remove only waiting rows.
        repo?.let { repository ->
            val markedStarted = runCatching {
                repository.markTrainingStarted(item.id)
            }.onFailure { error ->
                Log.w(TAG, "Failed to persist active training state for '${item.id}'", error)
            }.getOrDefault(false)

            if (!markedStarted) {
                Log.w(TAG, "Training item '${item.id}' was not marked TRAINING in Room.")
            }
        }

        if (item.chunkIndex == 1) {
            item.documentId?.let { documentId ->
                repo?.let { repository ->
                    runCatching { repository.markCorpusDocumentTraining(documentId) }
                        .onFailure { error ->
                            Log.w(TAG, "Failed to mark corpus document as training", error)
                        }
                }
            }
        }

        val startNs = System.nanoTime()
        val stepLoss = neuralEngine.trainStep(item.content)
        val durationMs = (System.nanoTime() - startNs) / 1_000_000L

        // Loss tidak valid → FAILED, bukan COMPLETED.
        if (!stepLoss.isFinite()) {
            Log.w(TAG, "trainStep return non-finite loss=$stepLoss untuk '${item.title}'")
            val failed = item.copy(status = TrainingStatus.FAILED, stepDurationMs = durationMs)
            pushCompleted(failed)
            scope.launch {
                repo?.let { repository ->
                    runCatching { repository.markTrainingFailed(failed.id) }
                        .onFailure { error ->
                            Log.w(TAG, "Failed to persist failed training item", error)
                        }
                    failed.documentId?.let { documentId ->
                        runCatching { repository.markCorpusDocumentPartial(documentId) }
                            .onFailure { error ->
                                Log.w(TAG, "Failed to mark corpus document as partial", error)
                            }
                    }
                }
            }
            return
        }

        val finished = item.copy(
            status = TrainingStatus.COMPLETED,
            finalLoss = stepLoss,
            stepDurationMs = durationMs
        )

        _lossHistory.update { (it + stepLoss).takeLast(MAX_LOSS_HISTORY) }
        _totalStepsExecuted.update { it + 1 }

        runCatching {
            worldModel.storeEntity(
                "last_training_loss",
                String.format(Locale.US, "%.4f", stepLoss),
                "On-Device Neural Core"
            )
            worldModel.storeEntity(
                "total_trained_steps",
                _totalStepsExecuted.value.toString(),
                "On-Device Neural Core"
            )
        }.onFailure { Log.w(TAG, "Gagal mencatat telemetri loss/steps", it) }

        if (item.documentTitle != null && item.totalChunks > 1) {
            var docCompleted = false
            _activeDocument.update { current ->
                if (current == null || current.docTitle != item.documentTitle) current
                else {
                    val avg = current.runningAverageLoss
                    val newAvg = if (avg != null) (avg * 0.7f + stepLoss * 0.3f) else stepLoss
                    if (item.chunkIndex >= item.totalChunks) docCompleted = true
                    current.copy(
                        completedChunks = item.chunkIndex,
                        latestLoss = stepLoss,
                        runningAverageLoss = newAvg
                    )
                }
            }
            if (docCompleted) {
                runCatching {
                    worldModel.storeEntity(
                        "doc_trained_${System.currentTimeMillis()}",
                        "Dokumen '${item.documentTitle}' selesai diasimilasi (${item.totalChunks} blok).",
                        "Document Trainer"
                    )
                }.onFailure { Log.w(TAG, "Gagal mencatat telemetri docCompleted", it) }
                _activeDocument.value = null
            }
        }

        pushCompleted(finished)
        scope.launch {
            repo?.let { repository ->
                runCatching {
                    repository.markTrainingCompleted(finished.id, stepLoss, durationMs)
                }.onFailure { error ->
                    Log.w(TAG, "Failed to persist completed training item", error)
                }

                if (
                    finished.documentId != null &&
                    finished.chunkIndex >= finished.totalChunks
                ) {
                    runCatching {
                        val document = repository.getCorpusDocument(finished.documentId)
                        if (document != null && document.chunkCount > 0) {
                            val completedPasses = finished.totalChunks / document.chunkCount
                            repository.markCorpusDocumentTrained(
                                documentId = document.id,
                                completedPassCount = completedPasses
                            )
                        }
                    }.onFailure { error ->
                        Log.w(TAG, "Failed to update completed corpus document state", error)
                    }
                }
            }
        }
    }

    private fun pushCompleted(item: TrainingData) {
        _completedQueue.update { (listOf(item) + it).take(MAX_COMPLETED_SIZE) }
    }

    private fun resetForceOverrideIfSet() {
        runCatching {
            if (deviceGuard.forceTrainingOverride.value) {
                deviceGuard.consumeForceOverride()
            }
        }.onFailure { Log.w(TAG, "Gagal reset force override", it) }
    }

    // =====================================================================
    // Chunking
    // =====================================================================

    private fun splitIntoTrainingChunks(text: String, chunkSize: Int, overlap: Int): List<String> {
        require(chunkSize > overlap) { "chunkSize harus > overlap" }
        if (text.length <= chunkSize) return listOf(text)

        val chunks = mutableListOf<String>()
        var start = 0
        while (start < text.length) {
            var end = min(start + chunkSize, text.length)
            if (end < text.length) {
                val window = text.substring(start, end)
                val lastBreak = max(
                    window.lastIndexOf('\n'),
                    max(
                        window.lastIndexOf(". "),
                        max(window.lastIndexOf("? "), window.lastIndexOf("! "))
                    )
                )
                if (lastBreak > chunkSize / 3) {
                    end = start + lastBreak + 1
                }
            }
            val sub = text.substring(start, end).trim()
            if (sub.isNotBlank()) chunks.add(sub)
            if (end >= text.length) break

            val nextStart = end - overlap
            // Jaminan maju: start harus naik minimal 1
            start = if (nextStart <= start) end else nextStart
        }
        return chunks.ifEmpty { listOf(text) }
    }

    // =====================================================================
    // Lifecycle
    // =====================================================================

    /**
     * Stops the training worker and drains queued persistence operations before shutdown.
     * An active optimizer step is allowed to finish within the shutdown timeout.
     * Pending queue items remain persisted in Room and can be restored by the next manager.
     */
    override fun close() {
        closed = true
        _isTrainingPaused.value = false
        persistenceOperations.close()
        wakeupSignal.trySend(Unit)
        resetForceOverrideIfSet()

        runBlocking {
            runCatching {
                withTimeout(2000L) {
                    workerJob?.join()
                    persistenceJob?.join()
                }
            }.onFailure {
                Log.w(TAG, "Training workers did not finish before shutdown timeout", it)
            }
        }
        scope.cancel()
    }

    fun shutdown() = close()
}