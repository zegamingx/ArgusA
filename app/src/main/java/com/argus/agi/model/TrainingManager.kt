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
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull
import java.io.Closeable
import java.io.Reader
import java.io.StringReader
import java.security.MessageDigest
import java.util.Locale
import java.util.UUID
import kotlin.math.exp
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

/** Runtime counters for this app session; they are not presented as lifetime corpus totals. */
data class TrainingSessionMetrics(
    val completedChunks: Long = 0L,
    val completedDocuments: Long = 0L,
    val inputTokens: Long = 0L,
    val predictionTargets: Long = 0L,
    val contextWindows: Long = 0L,
    val optimizerUpdates: Long = 0L,
    val reasoningExamples: Long = 0L,
    val reasoningTokens: Long = 0L,
    val vocabularySize: Int = 0,
    val vocabularyCapacity: Int = 0,
    val vocabularyTokensAdded: Int = 0,
    val tokenizerMerges: Int = 0
)

data class ValidationCorpusEvaluation(
    val corpusDocumentCount: Int,
    val evaluatedDocumentCount: Int,
    val targetTokenCount: Int,
    val meanLoss: Float,
    val perplexity: Float,
    val reasoningDocumentCount: Int = 0,
    val reasoningSequenceCount: Int = 0,
    val reasoningTargetTokenCount: Int = 0,
    val reasoningMeanLoss: Float = Float.NaN,
    val reasoningPerplexity: Float = Float.NaN
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

        data class Start(
            val itemId: String,
            val completion: CompletableDeferred<Unit>
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
        const val MAX_CORPUS_REPLAY_DOCUMENTS = 500
        const val MAX_DEFAULT_CORPUS_REPLAY_CHUNKS = 4_096
        const val MAX_CORPUS_REPLAY_CHUNKS = 50_000
        private const val MAX_TOKENIZER_SAMPLE_CHUNKS = 256
        private const val MAX_TOKENIZER_SAMPLE_CHARACTERS = 64 * 1024
        private const val MAX_STREAMED_JSON_VALUE_CHARACTERS = 256 * 1024
        private const val IDLE_POLL_MS = 500L
        private val REASONING_BLOCK_PATTERN = Regex(
            "<THINK>(.*?)</THINK>",
            RegexOption.DOT_MATCHES_ALL
        )
        private val REASONING_SEQUENCE_PATTERN = Regex(
            "<USER>.*?<THINK>.*?</THINK>",
            RegexOption.DOT_MATCHES_ALL
        )
    }

    // StateFlow<List<TrainingData>> sebagai satu-satunya sumber kebenaran untuk queue.
    // Channel hanya berfungsi sebagai wakeup signal, bukan storage.
    private val queueLock = Any()
    private val corpusSchedulingMutex = Mutex()
    private val seedSchedulingMutex = Mutex()
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

    private val preparedTokenizerDocuments = HashSet<String>()

    private val _trainingMetrics = MutableStateFlow(
        TrainingSessionMetrics(
            vocabularySize = neuralEngine.tokenizer.vocabSize,
            vocabularyCapacity = neuralEngine.config.tokenizerConfig.maxVocabularySize,
            tokenizerMerges = neuralEngine.tokenizer.mergeCount
        )
    )
    val trainingMetrics: StateFlow<TrainingSessionMetrics> = _trainingMetrics.asStateFlow()

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
        requireDeclaredRightsForWebContent(source, canonicalUrl, rightsStatus)

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
     * Ensures a durable training document has pending work without duplicating an
     * already queued, active, or fully trained document.
     *
     * This is intended for idempotent application-owned seed corpora. If the document
     * already exists but its queue was cleared or its work was interrupted, retained
     * chunks are replayed instead of trying to import the same fingerprint again.
     */
    suspend fun ensureCorpusDocumentScheduled(
        source: String,
        title: String,
        fullText: String,
        chunkSize: Int = 380,
        overlap: Int = 50,
        epochs: Int = 1
    ): Int {
        seedSchedulingMutex.lock()
        try {
            val newlyQueuedChunks = enqueueDocument(
                source = source,
                title = title,
                fullText = fullText,
                chunkSize = chunkSize,
                overlap = overlap,
                epochs = epochs
            )
            if (newlyQueuedChunks > 0 || closed) return newlyQueuedChunks

            val activeRepository = requireNotNull(repo) {
                "Local corpus storage is unavailable; seed scheduling cannot be recovered."
            }
            val fingerprint = sha256Utf16Be(fullText)
            val existingDocument = activeRepository.getCorpusDocumentByFingerprint(fingerprint)
                ?: return 0

            require(existingDocument.datasetSplit == "TRAIN") {
                "A validation-only document cannot be used as a training seed."
            }
            if (existingDocument.status == "TRAINED") return 0

            val persistedActive = existingDocument.id in
                activeRepository.getActiveCorpusTrainingDocumentIds()
            val memoryActive = synchronized(queueLock) {
                _nextQueue.value.any { it.documentId == existingDocument.id } ||
                    _currentTrainingItem.value?.documentId == existingDocument.id
            }
            if (persistedActive || memoryActive) return 0

            return replayCorpusDocument(
                documentId = existingDocument.id,
                epochs = epochs
            )
        } finally {
            seedSchedulingMutex.unlock()
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
        requireDeclaredRightsForWebContent(source, canonicalUrl, rightsStatus)
        if (closed) return Pair(0, 0L)

        val rawChunks = mutableListOf<String>()
        val buffer = CharArray(4096)
        val window = StringBuilder()
        val fingerprintDigest = MessageDigest.getInstance("SHA-256")
        var totalCharsRead = 0L

        fun appendSegment(rawSegment: String) {
            val normalizedSegment = TextModalityProcessor.normalizeTypography(rawSegment)
            if (normalizedSegment.isBlank()) return

            val formattedRecords =
                TextModalityProcessor.formatTrainingDatasetRecords(normalizedSegment)
            if (formattedRecords != null) {
                appendFormattedTrainingRecords(
                    records = formattedRecords,
                    destination = rawChunks,
                    chunkSize = chunkSize
                )
                return
            }

            val formattedText = normalizedSegment.lines().joinToString("\n") { line ->
                TextModalityProcessor.formatStructuredLineIfJson(line)
            }
            if (formattedText.isNotBlank()) rawChunks.add(formattedText)
        }

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

                var splitPosition = if (punctuationBreak >= 0) {
                    minimumBreakPosition + punctuationBreak + 1
                } else {
                    val lastSpace = searchText.lastIndexOf(' ')
                    if (lastSpace >= 0) minimumBreakPosition + lastSpace + 1 else chunkSize
                }.coerceIn(1, chunkSize)
                var structuredBoundary = false

                // Never cut a JSON object/array in the middle. JSONL records are
                // handled individually; pretty-printed values are buffered as one
                // bounded value before the dataset formatter sees them.
                var jsonStart = -1
                var lineStart = 0
                while (lineStart < splitPosition) {
                    var candidateStart = lineStart
                    while (
                        candidateStart < splitPosition &&
                        window[candidateStart] != '\n' &&
                        window[candidateStart].isWhitespace()
                    ) {
                        candidateStart++
                    }
                    if (
                        candidateStart < splitPosition &&
                        window[candidateStart] in charArrayOf('{', '[')
                    ) {
                        jsonStart = candidateStart
                        break
                    }

                    var nextLineStart = lineStart
                    while (nextLineStart < splitPosition && window[nextLineStart] != '\n') {
                        nextLineStart++
                    }
                    if (nextLineStart >= splitPosition) break
                    lineStart = nextLineStart + 1
                }

                if (jsonStart >= 0) {
                    if (jsonStart > 0) {
                        splitPosition = jsonStart
                        structuredBoundary = true
                    } else {
                        val jsonEnd = TextModalityProcessor.findCompleteJsonValueEnd(
                            text = window.toString(),
                            startIndex = jsonStart
                        )
                        if (jsonEnd == null) {
                            val bufferedCharacters = window.length - jsonStart
                            require(bufferedCharacters <= MAX_STREAMED_JSON_VALUE_CHARACTERS) {
                                "A streamed JSON training value exceeds " +
                                    "$MAX_STREAMED_JSON_VALUE_CHARACTERS characters. " +
                                    "Use JSONL with one training record per line."
                            }
                            break
                        }
                        val jsonLength = jsonEnd - jsonStart
                        require(jsonLength <= MAX_STREAMED_JSON_VALUE_CHARACTERS) {
                            "A streamed JSON training value exceeds " +
                                "$MAX_STREAMED_JSON_VALUE_CHARACTERS characters. " +
                                "Use JSONL with one training record per line."
                        }
                        splitPosition = jsonEnd
                        structuredBoundary = true
                    }
                }

                appendSegment(window.substring(0, splitPosition))

                val nextStart = if (structuredBoundary) {
                    splitPosition
                } else {
                    (splitPosition - overlap).coerceAtLeast(0)
                }
                val safeNextStart = if (!structuredBoundary && nextStart == 0 && splitPosition > 0) {
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

        val trailingRaw = window.toString().trim()
        if (trailingRaw.isNotBlank()) {
            val firstContentIndex = trailingRaw.indexOfFirst { !it.isWhitespace() }
            if (
                firstContentIndex >= 0 &&
                trailingRaw[firstContentIndex] in charArrayOf('{', '[') &&
                TextModalityProcessor.findCompleteJsonValueEnd(
                    text = trailingRaw,
                    startIndex = firstContentIndex
                ) == null
            ) {
                throw IllegalArgumentException(
                    "The streamed JSON training value is incomplete. " +
                        "Check the final JSONL record or closing JSON brackets."
                )
            }
            appendSegment(trailingRaw)
        }

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
     * Imports one complete document into the held-out validation split.
     *
     * Validation content is persisted as one source text, not split into overlapping
     * training chunks. It is never appended to the training queue. An identical document
     * already present in the training split is rejected by the repository.
     */
    suspend fun importValidationDocument(
        source: String,
        title: String,
        fullText: String,
        language: String? = null,
        canonicalUrl: String? = null,
        rightsStatus: String = "UNVERIFIED"
    ): Boolean {
        require(rightsStatus in setOf("UNVERIFIED", "USER_DECLARED", "LICENSE_REVIEWED")) {
            "Unsupported corpus rights status: $rightsStatus"
        }
        requireDeclaredRightsForWebContent(source, canonicalUrl, rightsStatus)
        val clean = fullText.trim()
        if (clean.isBlank() || closed) return false

        val normalizedText = TextModalityProcessor.normalizeTypography(clean)
            .lines()
            .joinToString("\n") { TextModalityProcessor.formatStructuredLineIfJson(it) }
        if (normalizedText.isBlank()) return false

        val documentId = sha256Utf16Be(fullText)
        val document = CorpusDocumentEntity(
            id = documentId,
            contentFingerprint = documentId,
            fingerprintAlgorithm = "SHA-256/UTF-16BE",
            source = source,
            title = title,
            language = language,
            canonicalUrl = canonicalUrl,
            characterCount = clean.length,
            chunkCount = 0,
            datasetSplit = "VALIDATION",
            validationText = normalizedText,
            plannedEpochs = 0,
            rightsStatus = rightsStatus,
            status = "READY",
            createdAtEpochMs = System.currentTimeMillis(),
            lastTrainedAtEpochMs = null,
            completedPassCount = 0
        )

        val activeRepository = requireNotNull(repo) {
            "Local corpus storage is unavailable; validation data was not imported."
        }
        return activeRepository.storeCorpusDocument(document, emptyList())
    }

    /**
     * Evaluates held-out documents as one token-weighted corpus.
     * The neural engine holds its model lock for the whole evaluation and does not update weights.
     */
    suspend fun evaluateValidationCorpus(limit: Int = 500): ValidationCorpusEvaluation {
        require(limit in 1..500) { "Validation document limit must be in 1..500." }
        val activeRepository = requireNotNull(repo) {
            "Local corpus storage is unavailable; validation cannot be evaluated."
        }
        val documents = activeRepository.getValidationCorpusDocuments(limit)
        val processedDocuments = documents.map { document ->
            val validationText = document.validationText.orEmpty()
            if (validationText.isBlank()) {
                emptyList()
            } else {
                TextModalityProcessor.process(
                    rawText = validationText,
                    documentTitle = document.title,
                    chunkSize = 380,
                    overlap = 50
                ).map { it.content }
            }
        }
        val texts = processedDocuments.flatten()
        val evaluation = neuralEngine.evaluateCorpusLoss(texts)
        val perplexity = if (evaluation.meanLoss.isFinite()) {
            exp(evaluation.meanLoss)
        } else {
            Float.NaN
        }

        // Evaluate prompt + reasoning prefixes without scoring their final answers.
        // This is a language-model fit diagnostic, not a reasoning correctness score.
        val reasoningSequences = texts.flatMap { text ->
            REASONING_SEQUENCE_PATTERN.findAll(text).map { it.value }.toList()
        }
        val reasoningEvaluation = neuralEngine.evaluateCorpusLoss(reasoningSequences)
        val reasoningPerplexity = if (reasoningEvaluation.meanLoss.isFinite()) {
            exp(reasoningEvaluation.meanLoss)
        } else {
            Float.NaN
        }
        val reasoningDocumentCount = processedDocuments.count { chunks ->
            chunks.any { it.contains(Tokenizer.THINK_START) && it.contains(Tokenizer.THINK_END) }
        }

        return ValidationCorpusEvaluation(
            corpusDocumentCount = documents.size,
            evaluatedDocumentCount = evaluation.evaluatedDocumentCount,
            targetTokenCount = evaluation.targetCount,
            meanLoss = evaluation.meanLoss,
            perplexity = perplexity,
            reasoningDocumentCount = reasoningDocumentCount,
            reasoningSequenceCount = reasoningSequences.size,
            reasoningTargetTokenCount = reasoningEvaluation.targetCount,
            reasoningMeanLoss = reasoningEvaluation.meanLoss,
            reasoningPerplexity = reasoningPerplexity
        )
    }

    private fun appendFormattedTrainingRecords(
        records: List<String>,
        destination: MutableList<String>,
        chunkSize: Int
    ) {
        val buffer = StringBuilder()

        fun flushBuffer() {
            if (buffer.isNotEmpty()) {
                destination.add(buffer.toString())
                buffer.setLength(0)
            }
        }

        for (record in records) {
            if (record.isBlank()) continue
            if (record.length > chunkSize) {
                flushBuffer()
                destination.add(record)
                continue
            }
            if (buffer.isNotEmpty() && buffer.length + record.length + 1 > chunkSize) {
                flushBuffer()
            }
            if (buffer.isNotEmpty()) buffer.append('\n')
            buffer.append(record)
        }
        flushBuffer()
    }

    /**
     * Estimates a conservative queue cap from the current app heap.
     *
     * The estimate reserves most of the heap for model tensors, activations, and the UI.
     * The 4 KiB item estimate includes the retained chunk string and queue object overhead.
     */
    private fun recommendedReplayQueueLimit(): Int {
        val runtime = Runtime.getRuntime()
        val maxHeapBytes = runtime.maxMemory().coerceAtLeast(1L)
        val allocatedBytes = runtime.totalMemory()
        val usedBytes = (allocatedBytes - runtime.freeMemory()).coerceAtLeast(0L)
        val availableHeapBytes = (maxHeapBytes - usedBytes).coerceAtLeast(0L)
        val safeBudgetBytes = min(maxHeapBytes / 16L, availableHeapBytes / 3L)
        val estimatedBytesPerItem = 4_096L
        val estimatedLimit = safeBudgetBytes / estimatedBytesPerItem

        return estimatedLimit
            .coerceIn(1L, MAX_DEFAULT_CORPUS_REPLAY_CHUNKS.toLong())
            .toInt()
    }

    /**
     * Schedules every eligible TRAIN document in stable corpus order, then applies the
     * requested deterministic replay policy. The same seed and corpus snapshot reproduce
     * the same queue order. Validation documents are excluded by the DAO and planner.
     */
    suspend fun scheduleCorpusReplay(
        seed: Long,
        epochs: Int = 1,
        policy: CorpusReplayPolicy = CorpusReplayPolicy.INTERLEAVED_DOCUMENTS,
        maxDocuments: Int = MAX_CORPUS_REPLAY_DOCUMENTS,
        maxQueuedChunks: Int = recommendedReplayQueueLimit()
    ): CorpusReplayResult {
        corpusSchedulingMutex.lock()
        try {
            require(maxDocuments in 1..MAX_CORPUS_REPLAY_DOCUMENTS) {
                "maxDocuments must be in 1..$MAX_CORPUS_REPLAY_DOCUMENTS."
            }
            require(maxQueuedChunks in 1..MAX_CORPUS_REPLAY_CHUNKS) {
                "maxQueuedChunks must be in 1..$MAX_CORPUS_REPLAY_CHUNKS."
            }
            require(epochs in 1..MAX_DOCUMENT_EPOCHS) {
                "epochs must be in 1..$MAX_DOCUMENT_EPOCHS."
            }
            if (closed) {
                return CorpusReplayResult(0, 0, epochs, seed, policy)
            }

            val activeRepository = requireNotNull(repo) {
                "Local corpus storage is unavailable; corpus replay cannot be scheduled."
            }
            val documents = activeRepository.getTrainingCorpusDocuments(maxDocuments)
            if (documents.isEmpty()) {
                return CorpusReplayResult(0, 0, epochs, seed, policy)
            }

            val requestedChunkCount = documents.sumOf { it.chunkCount.toLong() } * epochs.toLong()
            require(requestedChunkCount <= maxQueuedChunks.toLong()) {
                "Corpus replay requires $requestedChunkCount queue items, exceeding the configured limit of $maxQueuedChunks. Reduce epochs or import fewer documents per replay."
            }

            val sources = documents.map { document ->
                CorpusReplaySource(
                    document = document,
                    chunks = activeRepository.getCorpusChunks(document.id)
                )
            }
            return enqueueCorpusReplay(
                sources = sources,
                epochs = epochs,
                seed = seed,
                policy = policy,
                maxQueuedChunks = maxQueuedChunks
            )
        } finally {
            corpusSchedulingMutex.unlock()
        }
    }

    suspend fun replayCorpusDocument(
        documentId: String,
        epochs: Int = 1,
        seed: Long = 0L,
        policy: CorpusReplayPolicy = CorpusReplayPolicy.SEQUENTIAL_DOCUMENTS,
        maxQueuedChunks: Int = recommendedReplayQueueLimit()
    ): Int {
        corpusSchedulingMutex.lock()
        try {
            require(maxQueuedChunks in 1..MAX_CORPUS_REPLAY_CHUNKS) {
                "maxQueuedChunks must be in 1..$MAX_CORPUS_REPLAY_CHUNKS."
            }
            if (closed) return 0

            val activeRepository = requireNotNull(repo) {
                "Local corpus storage is unavailable; corpus replay cannot be scheduled."
            }
            val document = activeRepository.getCorpusDocument(documentId) ?: return 0
            require(document.datasetSplit == "TRAIN") {
                "Validation-only documents cannot be replayed into the training queue."
            }

            val source = CorpusReplaySource(
                document = document,
                chunks = activeRepository.getCorpusChunks(documentId)
            )
            val result = enqueueCorpusReplay(
                sources = listOf(source),
                epochs = epochs,
                seed = seed,
                policy = policy,
                maxQueuedChunks = maxQueuedChunks
            )
            return result.scheduledChunkCount
        } finally {
            corpusSchedulingMutex.unlock()
        }
    }

    private suspend fun enqueueCorpusReplay(
        sources: List<CorpusReplaySource>,
        epochs: Int,
        seed: Long,
        policy: CorpusReplayPolicy,
        maxQueuedChunks: Int
    ): CorpusReplayResult {
        require(epochs in 1..MAX_DOCUMENT_EPOCHS) {
            "epochs must be in 1..$MAX_DOCUMENT_EPOCHS."
        }
        require(sources.isNotEmpty()) { "No eligible training corpus documents were found." }

        val requestedChunkCount = sources.sumOf { it.document.chunkCount.toLong() } * epochs.toLong()
        require(requestedChunkCount in 1..maxQueuedChunks.toLong()) {
            "Corpus replay requires $requestedChunkCount queue items; configured limit is $maxQueuedChunks."
        }

        val activeRepository = requireNotNull(repo) {
            "Local corpus storage is unavailable; corpus replay cannot be scheduled."
        }
        val requestedDocumentIds = sources.map { it.document.id }.toSet()
        val activeDocumentIds = activeRepository.getActiveCorpusTrainingDocumentIds()
        val databaseConflicts = requestedDocumentIds.intersect(activeDocumentIds)
        check(databaseConflicts.isEmpty()) {
            "Corpus replay would duplicate documents that already have queued or active training work: ${databaseConflicts.joinToString()}."
        }

        val schedule = CorpusReplayScheduler.createSchedule(
            sources = sources,
            epochs = epochs,
            seed = seed,
            policy = policy
        )
        check(schedule.size.toLong() == requestedChunkCount) {
            "Replay scheduler produced an unexpected number of queue items."
        }

        val items = schedule.map { scheduled ->
            TrainingData(
                source = scheduled.document.source,
                title = buildString {
                    append(scheduled.document.title)
                    append(" [Epoch ").append(scheduled.epochNumber).append('/').append(epochs).append(']')
                    append(" [Chunk ")
                    append(scheduled.chunk.chunkIndex).append('/').append(scheduled.document.chunkCount)
                    append(']')
                },
                content = scheduled.chunk.content,
                chunkIndex = scheduled.documentChunkIndex,
                totalChunks = scheduled.documentTotalChunks,
                documentTitle = scheduled.document.title,
                status = TrainingStatus.QUEUED,
                documentId = scheduled.document.id
            )
        }

        val memoryConflicts = synchronized(queueLock) {
            val memoryActiveDocumentIds = _nextQueue.value
                .mapNotNull { it.documentId }
                .toMutableSet()
            _currentTrainingItem.value?.documentId?.let { memoryActiveDocumentIds.add(it) }
            requestedDocumentIds.intersect(memoryActiveDocumentIds)
        }
        check(memoryConflicts.isEmpty()) {
            "Corpus replay would duplicate in-memory training work: ${memoryConflicts.joinToString()}."
        }

        activeRepository.markCorpusDocumentsQueued(requestedDocumentIds)

        val orderedItems = synchronized(queueLock) {
            if (closed) {
                emptyList()
            } else {
                appendPersistingQueueItems(items).also { queued ->
                    val firstItem = queued.firstOrNull()
                    if (_activeDocument.value == null && firstItem != null) {
                        val firstDocumentId = requireNotNull(firstItem.documentId)
                        val firstDocument = sources.first {
                            it.document.id == firstDocumentId
                        }.document
                        _activeDocument.value = ActiveDocumentTraining(
                            docTitle = firstDocument.title,
                            totalCharacters = firstDocument.characterCount,
                            totalChunks = firstItem.totalChunks,
                            completedChunks = firstItem.chunkIndex - 1,
                            currentChunkIndex = firstItem.chunkIndex
                        )
                    }
                }
            }
        }

        return CorpusReplayResult(
            scheduledDocumentCount = if (orderedItems.isEmpty()) 0 else requestedDocumentIds.size,
            scheduledChunkCount = orderedItems.size,
            epochs = epochs,
            seed = seed,
            policy = policy
        )
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

    private fun requireDeclaredRightsForWebContent(
        source: String,
        canonicalUrl: String?,
        rightsStatus: String
    ) {
        val isWebContent = source.startsWith("web_article", ignoreCase = true) ||
            canonicalUrl != null
        require(!isWebContent || rightsStatus != "UNVERIFIED") {
            "Web article content must be reviewed and explicitly declared before it can be stored."
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
                    runCatching {
                        repository.clearPendingQueueAndMarkInterruptedCorpusDocumentsPartial()
                    }
                        .onFailure { error ->
                            Log.e(TAG, "Failed to clear persisted training queue", error)
                        }
                }
            }

            is QueuePersistenceOperation.Start -> {
                try {
                    repo?.let { repository ->
                        runCatching {
                            repository.markTrainingStarted(operation.itemId)
                        }.onFailure { error ->
                            Log.e(
                                TAG,
                                "Failed to persist active training state for '${operation.itemId}'",
                                error
                            )
                        }
                    }
                } finally {
                    operation.completion.complete(Unit)
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

    private suspend fun popNext(): TrainingData? {
        val startCompletion = CompletableDeferred<Unit>()
        val item = synchronized(queueLock) {
            val list = _nextQueue.value
            val first = list.firstOrNull() ?: return@synchronized null
            if (first.status != TrainingStatus.QUEUED) return@synchronized null

            val sendResult = persistenceOperations.trySend(
                QueuePersistenceOperation.Start(
                    itemId = first.id,
                    completion = startCompletion
                )
            )
            if (sendResult.isFailure) {
                Log.e(TAG, "Could not persist training start for '${first.id}'.")
                return@synchronized null
            }

            _nextQueue.value = list.drop(1)
            first
        }

        // Start is ordered in the same persistence channel as Clear. The worker waits for
        // that update so a concurrent clear cannot delete an already-active queue row.
        if (item != null) startCompletion.await()
        return item
    }

    private fun updateActiveDocumentEntering(item: TrainingData) {
        if (item.documentTitle == null) return
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

    private suspend fun prepareTokenizerForDocument(item: TrainingData) {
        val documentId = item.documentId ?: return
        if (!preparedTokenizerDocuments.add(documentId)) return

        try {
            val storedSamples = repo?.getTokenizerCorpusSample(
                documentId = documentId,
                limit = MAX_TOKENIZER_SAMPLE_CHUNKS
            ).orEmpty()

            var remainingCharacters = MAX_TOKENIZER_SAMPLE_CHARACTERS
            val corpusSample = ArrayList<String>(storedSamples.size)
            for (chunk in storedSamples) {
                if (remainingCharacters <= 0) break
                val sample = chunk.take(remainingCharacters)
                if (sample.isNotBlank()) corpusSample.add(sample)
                remainingCharacters -= sample.length
            }
            if (corpusSample.isEmpty() && item.content.isNotBlank()) {
                corpusSample.add(item.content.take(MAX_TOKENIZER_SAMPLE_CHARACTERS))
            }

            val update = neuralEngine.trainTokenizerFromCorpus(corpusSample)
            _trainingMetrics.update { current ->
                current.copy(
                    vocabularySize = update.vocabularySize,
                    vocabularyTokensAdded = current.vocabularyTokensAdded + update.tokensAdded,
                    tokenizerMerges = update.mergeCount
                )
            }
            Log.i(
                TAG,
                "Tokenizer corpus prepared for '${item.documentTitle ?: item.title}': " +
                    "${update.previousVocabularySize} -> ${update.vocabularySize} tokens, " +
                    "${update.mergeCount - update.previousMergeCount} new merges."
            )
        } catch (error: Exception) {
            preparedTokenizerDocuments.remove(documentId)
            Log.w(TAG, "Tokenizer preparation failed for document $documentId; existing vocabulary will be used.", error)
        }
    }

    private suspend fun processItem(item: TrainingData) {
        item.documentId?.let { documentId ->
            if (item.chunkIndex == 1) {
                repo?.let { repository ->
                    runCatching { repository.markCorpusDocumentTraining(documentId) }
                        .onFailure { error ->
                            Log.w(TAG, "Failed to mark corpus document as training", error)
                        }
                }
            }
            prepareTokenizerForDocument(item)
        }

        val startNs = System.nanoTime()
        val stepResult = neuralEngine.trainStepDetailed(item.content)
        val stepLoss = stepResult.loss
        val durationMs = (System.nanoTime() - startNs) / 1_000_000L

        // Invalid loss is a failed step, never a completed training item.
        if (!stepLoss.isFinite()) {
            Log.w(TAG, "trainStep returned non-finite loss=$stepLoss for '${item.title}'")
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

        val reasoningSpans = REASONING_BLOCK_PATTERN.findAll(item.content)
            .map { it.groupValues[1] }
            .toList()
        val reasoningTokenCount = reasoningSpans.sumOf { reasoningText ->
            neuralEngine.tokenizer.encode(reasoningText, autoRegister = false).size.toLong()
        }

        val finished = item.copy(
            status = TrainingStatus.COMPLETED,
            finalLoss = stepLoss,
            stepDurationMs = durationMs
        )

        _lossHistory.update { (it + stepLoss).takeLast(MAX_LOSS_HISTORY) }
        _totalStepsExecuted.update { it + 1 }

        val documentCompleted = item.documentId != null &&
            item.chunkIndex >= item.totalChunks
        _trainingMetrics.update { current ->
            current.copy(
                completedChunks = current.completedChunks + 1L,
                completedDocuments = current.completedDocuments +
                    if (documentCompleted) 1L else 0L,
                inputTokens = current.inputTokens + stepResult.inputTokenCount,
                predictionTargets = current.predictionTargets + stepResult.predictionTargetCount,
                contextWindows = current.contextWindows + stepResult.contextWindowCount,
                optimizerUpdates = current.optimizerUpdates + stepResult.optimizerUpdateCount,
                reasoningExamples = current.reasoningExamples + reasoningSpans.size.toLong(),
                reasoningTokens = current.reasoningTokens + reasoningTokenCount,
                vocabularySize = stepResult.vocabularySize
            )
        }

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
        }.onFailure { Log.w(TAG, "Failed to record training telemetry.", it) }

        if (item.documentTitle != null) {
            var documentCompletedInUi = false
            _activeDocument.update { current ->
                if (current == null || current.docTitle != item.documentTitle) {
                    current
                } else {
                    val previousAverage = current.runningAverageLoss
                    val updatedAverage = if (previousAverage != null) {
                        previousAverage * 0.7f + stepLoss * 0.3f
                    } else {
                        stepLoss
                    }
                    if (documentCompleted) documentCompletedInUi = true
                    current.copy(
                        completedChunks = item.chunkIndex,
                        latestLoss = stepLoss,
                        runningAverageLoss = updatedAverage
                    )
                }
            }
            if (documentCompletedInUi) {
                runCatching {
                    worldModel.storeEntity(
                        "doc_trained_${System.currentTimeMillis()}",
                        "Document '${item.documentTitle}' completed assimilation (${item.totalChunks} training items).",
                        "Document Trainer"
                    )
                }.onFailure { Log.w(TAG, "Failed to record document completion telemetry.", it) }
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