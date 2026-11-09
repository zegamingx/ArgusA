package com.argus.agi.model

import android.content.Context
import android.util.Log
import com.argus.agi.sensor.DeviceGuard
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
import java.util.Locale
import java.util.UUID
import kotlin.math.max
import kotlin.math.min

enum class TrainingStatus { QUEUED, TRAINING, COMPLETED, FAILED }

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
    val stepDurationMs: Long = 0L
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

    companion object {
        private const val TAG = "TrainingManager"
        const val MAX_COMPLETED_SIZE = 100
        const val MAX_LOSS_HISTORY = 100
        private const val IDLE_POLL_MS = 500L
    }

    // StateFlow<List<TrainingData>> sebagai satu-satunya sumber kebenaran untuk queue.
    // Channel hanya berfungsi sebagai wakeup signal, bukan storage.
    private val queueLock = Any()

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

    init {
        scope.launch {
            repo?.let { r ->
                runCatching {
                    val pending = r.getPendingQueueSnapshot()
                    if (pending.isNotEmpty()) {
                        synchronized(queueLock) {
                            val current = _nextQueue.value
                            if (current.isEmpty()) {
                                _nextQueue.value = pending
                                wakeupSignal.trySend(Unit)
                            }
                        }
                    }
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
            _nextQueue.value = emptyList()
        }
        _activeDocument.value = null
        scope.launch {
            repo?.let { runCatching { it.clearPendingQueue() } }
        }
    }

    fun clearCompleted() { _completedQueue.value = emptyList() }

    // =====================================================================
    // Enqueue API
    // =====================================================================

    fun enqueueDocument(
        source: String,
        title: String,
        fullText: String,
        chunkSize: Int = 380,
        overlap: Int = 50
    ): Int {
        require(chunkSize > overlap) {
            "chunkSize ($chunkSize) harus > overlap ($overlap) agar iterasi maju"
        }
        val clean = fullText.trim()
        if (clean.isBlank() || closed) return 0

        val processedChunks = TextModalityProcessor.process(clean, title, chunkSize, overlap)
        val total = processedChunks.size
        if (total == 0) return 0

        val newItems = processedChunks.mapIndexed { idx, pChunk ->
            TrainingData(
                source = source,
                title = pChunk.segmentTitle,
                content = pChunk.content,
                chunkIndex = idx + 1,
                totalChunks = total,
                documentTitle = title,
                status = TrainingStatus.QUEUED
            )
        }

        synchronized(queueLock) {
            if (closed) return 0
            val current = _nextQueue.value
            _nextQueue.value = current + newItems
            scope.launch {
                repo?.let { runCatching { it.saveTrainingItems(newItems) } }
            }
            if (_activeDocument.value == null && newItems.isNotEmpty()) {
                _activeDocument.value = ActiveDocumentTraining(
                    docTitle = title,
                    totalCharacters = clean.length,
                    totalChunks = newItems.size,
                    completedChunks = 0,
                    currentChunkIndex = 1
                )
            }
            wakeupSignal.trySend(Unit)
            return newItems.size
        }
    }

    /**
     * Memproses dokumen besar secara streaming tanpa memuat seluruh teks ke heap RAM sekaligus.
     * Menggunakan buffer 4KB dan sliding window lokal berukuran chunkSize (~380 karakter),
     * membaca seluruh file sampai tuntas (EOF) tanpa batasan blok arbitrer.
     *
     * @return Pair(jumlahChunkDihasilkan, totalKarakterTerbaca)
     */
    fun enqueueDocumentStream(
        source: String,
        title: String,
        reader: Reader,
        chunkSize: Int = 380,
        overlap: Int = 50
    ): Pair<Int, Long> {
        require(chunkSize > overlap) {
            "chunkSize ($chunkSize) harus > overlap ($overlap) agar iterasi maju"
        }
        require(overlap <= chunkSize / 3) {
            "overlap ($overlap) harus <= chunkSize / 3 (${chunkSize / 3}) agar jaminan sliding window selalu maju"
        }
        if (closed) return Pair(0, 0L)

        val rawChunks = mutableListOf<String>()
        val buf = CharArray(4096)
        val window = StringBuilder()
        var totalCharsRead = 0L

        var readCount: Int
        while (reader.read(buf).also { readCount = it } != -1) {
            if (closed) break
            window.append(buf, 0, readCount)
            totalCharsRead += readCount

            while (window.length >= chunkSize) {
                var splitIdx = -1
                val minBreakPos = chunkSize / 3
                val searchStr = window.substring(minBreakPos, chunkSize)

                val nl = searchStr.lastIndexOf('\n')
                val dot = searchStr.lastIndexOf(". ")
                val qm = searchStr.lastIndexOf("? ")
                val em = searchStr.lastIndexOf("! ")
                val punct = max(nl, max(dot, max(qm, em)))

                if (punct >= 0) {
                    splitIdx = minBreakPos + punct + 1
                } else {
                    val sp = searchStr.lastIndexOf(' ')
                    if (sp >= 0) splitIdx = minBreakPos + sp + 1
                }

                val cutPos = if (splitIdx in 1..chunkSize) splitIdx else chunkSize
                val rawSegment = TextModalityProcessor.normalizeTypography(window.substring(0, cutPos))
                val segment = rawSegment.lines().joinToString("\n") { TextModalityProcessor.formatStructuredLineIfJson(it) }
                if (segment.isNotBlank()) {
                    rawChunks.add(segment)
                }

                val nextStart = (cutPos - overlap).coerceAtLeast(0)
                val safeNextStart = if (nextStart == 0 && cutPos > 0) cutPos else nextStart
                val tail = window.substring(safeNextStart)
                window.setLength(0)
                window.append(tail)
            }
        }

        val trailingRaw = TextModalityProcessor.normalizeTypography(window.toString().trim())
        val trailing = trailingRaw.lines().joinToString("\n") { TextModalityProcessor.formatStructuredLineIfJson(it) }
        if (trailing.isNotBlank()) {
            rawChunks.add(trailing)
        }

        val total = rawChunks.size
        if (total == 0) return Pair(0, totalCharsRead)

        val newItems = rawChunks.mapIndexed { idx, chunkText ->
            TrainingData(
                source = source,
                title = "$title [Blok ${idx + 1}/$total]",
                content = chunkText,
                chunkIndex = idx + 1,
                totalChunks = total,
                documentTitle = title,
                status = TrainingStatus.QUEUED
            )
        }

        synchronized(queueLock) {
            if (closed) return Pair(0, totalCharsRead)
            val current = _nextQueue.value
            _nextQueue.value = current + newItems
            scope.launch {
                repo?.let { runCatching { it.saveTrainingItems(newItems) } }
            }
            if (_activeDocument.value == null && newItems.isNotEmpty()) {
                _activeDocument.value = ActiveDocumentTraining(
                    docTitle = title,
                    totalCharacters = if (totalCharsRead > 0) totalCharsRead.toInt() else newItems.sumOf { it.content.length },
                    totalChunks = newItems.size,
                    completedChunks = 0,
                    currentChunkIndex = 1
                )
            }
            wakeupSignal.trySend(Unit)
            return Pair(newItems.size, totalCharsRead)
        }
    }

    fun enqueueData(source: String, title: String, content: String) {
        val trimmed = content.trim()
        if (trimmed.isBlank() || closed) return

        if (trimmed.length > 380) {
            enqueueDocument(source, title, trimmed)
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
                _nextQueue.value = current + nonDuplicates
                scope.launch {
                    repo?.let { runCatching { it.saveTrainingItems(nonDuplicates) } }
                }
            }
        }
        wakeupSignal.trySend(Unit)
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
                    // Antrean selesai: barulah set item aktif dan corpus aktif ke null
                    _currentTrainingItem.value = null
                    _activeDocument.value = null
                    // Idle: tunggu wakeup, tapi berkala cek pause/close
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
                        repo?.let { runCatching { it.markTrainingFailed(failed.id) } }
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

    private fun popNext(): TrainingData? = synchronized(queueLock) {
        val list = _nextQueue.value
        if (list.isEmpty()) null
        else {
            _nextQueue.value = list.subList(1, list.size).toList()
            list[0]
        }
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

    private fun processItem(item: TrainingData) {
        val startNs = System.nanoTime()
        val stepLoss = neuralEngine.trainStep(item.content)
        val durationMs = (System.nanoTime() - startNs) / 1_000_000L

        // Loss tidak valid → FAILED, bukan COMPLETED
        if (!stepLoss.isFinite()) {
            Log.w(TAG, "trainStep return non-finite loss=$stepLoss untuk '${item.title}'")
            val failed = item.copy(status = TrainingStatus.FAILED, stepDurationMs = durationMs)
            pushCompleted(failed)
            scope.launch {
                repo?.let { runCatching { it.markTrainingFailed(failed.id) } }
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
                // Corpus Aktif sudah selesai: bersihkan dari UI agar tidak terus muncul!
                _activeDocument.value = null
            }
        }

        pushCompleted(finished)
        scope.launch {
            repo?.let { runCatching { it.markTrainingCompleted(finished.id, stepLoss, durationMs) } }
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
     * Menutup manager. Item yang sedang diproses tidak di-cancel paksa — worker
     * akan menyelesaikan step yang sedang berjalan lalu berhenti.
     * Item yang masih pending DI-DROP karena shutdown. Panggil [clearQueue]
     * sebelum [close] jika ingin memastikan queue kosong diketahui user.
     */
    override fun close() {
        closed = true
        wakeupSignal.trySend(Unit)
        resetForceOverrideIfSet()
        // Beri waktu bagi worker untuk menyelesaikan step yang sedang berjalan
        runBlocking {
            runCatching {
                withTimeout(1500L) {
                    workerJob?.join()
                }
            }.onFailure { Log.w(TAG, "Worker join timeout saat close", it) }
        }
        scope.cancel()
    }

    fun shutdown() = close()
}