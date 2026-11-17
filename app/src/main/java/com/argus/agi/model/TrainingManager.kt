package com.argus.agi.model

import android.app.ActivityManager
import android.content.Context
import android.util.Log
import com.argus.agi.data.local.entity.CorpusChunkEntity
import com.argus.agi.data.local.entity.CorpusDocumentEntity
import com.argus.agi.sensor.DeviceGuard
import kotlinx.coroutines.CancellationException
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
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull
import java.io.Closeable
import java.io.File
import java.io.Reader
import java.io.StringReader
import java.security.MessageDigest
import java.util.Locale
import java.util.UUID
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

enum class TrainingStatus {
    PERSISTING,
    QUEUED,
    TRAINING,
    OPTIMIZER_COMMITTED,
    COMPLETED,
    FAILED
}

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

    private val repositoryInitialization =
        runCatching { repository ?: com.argus.agi.data.repository.ArgusRepository(context) }
    private val repo = repositoryInitialization.getOrNull()
    private val activityManager =
        context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
    private val validationFilesDirectory = context.applicationContext.filesDir
    private val trainingCommitJournalFile =
        File(validationFilesDirectory, TrainingCommitJournal.FILE_NAME)
    private val conversationalValidationPersistence =
        ConversationalValidationPersistence(validationFilesDirectory)

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
            val completion: CompletableDeferred<Result<Unit>>
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

    private val workerShutdown = MutableStateFlow(false)

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
    private val oomRecoveryItemIds = HashSet<String>()
    private val oomRetryCounts = HashMap<String, Int>()

    @Volatile
    private var activeBatchBaselineOptimizerStep: Int? = null

    @Volatile
    private var activeBatchRoomCommitted = false

    private data class TrainingMemorySnapshot(
        val availableHeapBytes: Long,
        val maximumHeapBytes: Long,
        val availableSystemMemoryBytes: Long,
        val isSystemLowMemory: Boolean
    )

    private fun captureTrainingMemorySnapshot(): TrainingMemorySnapshot {
        val runtime = Runtime.getRuntime()
        val maximumHeapBytes = runtime.maxMemory().coerceAtLeast(0L)
        val availableHeapBytes = (
            runtime.freeMemory() +
                (maximumHeapBytes - runtime.totalMemory()).coerceAtLeast(0L)
            ).coerceAtLeast(0L)
        val manager = activityManager
        if (manager == null) {
            return TrainingMemorySnapshot(
                availableHeapBytes = availableHeapBytes,
                maximumHeapBytes = maximumHeapBytes,
                availableSystemMemoryBytes = Long.MAX_VALUE,
                isSystemLowMemory = false
            )
        }

        val memoryInfo = ActivityManager.MemoryInfo()
        manager.getMemoryInfo(memoryInfo)
        return TrainingMemorySnapshot(
            availableHeapBytes = availableHeapBytes,
            maximumHeapBytes = maximumHeapBytes,
            availableSystemMemoryBytes =
                (memoryInfo.availMem - memoryInfo.threshold).coerceAtLeast(0L),
            isSystemLowMemory = memoryInfo.lowMemory
        )
    }

    private val _trainingMetrics = MutableStateFlow(
        TrainingSessionMetrics(
            vocabularySize = neuralEngine.tokenizer.vocabSize,
            vocabularyCapacity = neuralEngine.config.tokenizerConfig.maxVocabularySize,
            tokenizerMerges = neuralEngine.tokenizer.mergeCount
        )
    )
    val trainingMetrics: StateFlow<TrainingSessionMetrics> = _trainingMetrics.asStateFlow()

    private val _conversationalValidationState = MutableStateFlow(
        conversationalValidationPersistence.loadState()
    )
    val conversationalValidationState: StateFlow<ConversationValidationState> =
        _conversationalValidationState.asStateFlow()
    private val conversationalValidationMutex = Mutex()
    private var manualValidationJob: Job? = null
    private val validationRunDocumentIds = LinkedHashSet<String>()
    private val validationRunFailedDocumentIds = LinkedHashSet<String>()

    @Volatile private var closed = false

    private val exceptionHandler = CoroutineExceptionHandler { _, t ->
        Log.e(TAG, "Unhandled exception in training scope", t)
    }
    private val scope = CoroutineScope(SupervisorJob() + trainingDispatcher + exceptionHandler)
    private var workerJob: Job? = null
    private var persistenceJob: Job? = null

    init {
        persistenceJob = scope.launch {
            var activeOperation: QueuePersistenceOperation? = null
            try {
                for (operation in persistenceOperations) {
                    activeOperation = operation
                    try {
                        processQueuePersistenceOperation(operation)
                    } catch (cancelled: CancellationException) {
                        failPendingPersistenceOperation(operation, cancelled)
                        throw cancelled
                    } catch (error: Exception) {
                        failPendingPersistenceOperation(operation, error)
                        Log.e(TAG, "Training queue persistence operation failed unexpectedly.", error)
                    } finally {
                        activeOperation = null
                    }
                }
            } finally {
                val stopped = CancellationException(
                    "Training queue persistence worker stopped before all operations completed."
                )
                activeOperation?.let { operation ->
                    failPendingPersistenceOperation(operation, stopped)
                }
                while (true) {
                    val pending = persistenceOperations.tryReceive().getOrNull() ?: break
                    failPendingPersistenceOperation(pending, stopped)
                }
            }
        }

        val restoreGeneration = synchronized(queueLock) { queueGeneration }
        scope.launch {
            repo?.let { repository ->
                runSuspendCatching {
                    reconcilePersistedTrainingCommit(repository)
                    repository.recoverInterruptedTrainingItems()
                    val committed = repository.getOptimizerCommittedTrainingItemsSnapshot()
                    val pending = repository.getPendingQueueSnapshot()
                    (committed.sortedBy { it.timestamp } + pending.sortedBy { it.timestamp })
                }
                    .onSuccess { restoredItems ->
                        synchronized(queueLock) {
                            if (
                                restoreGeneration == queueGeneration &&
                                _nextQueue.value.isEmpty()
                            ) {
                                _nextQueue.value = restoredItems
                                restoredItems.maxOfOrNull { it.timestamp }?.let { latest ->
                                    lastScheduledTimestamp = maxOf(lastScheduledTimestamp, latest)
                                }
                                wakeupSignal.trySend(Unit)
                            }
                        }
                    }
                    .onFailure { error ->
                        _isTrainingPaused.value = true
                        Log.e(
                            TAG,
                            "Training is paused because its durable optimizer-commit state could not be reconciled.",
                            error
                        )
                    }
            }
        }

        if (repo == null) {
            Log.e(
                TAG,
                "Training persistence is unavailable; queued work will not be reported as durable.",
                repositoryInitialization.exceptionOrNull()
            )
        }
        startTrainingWorker()
    }

    fun pauseTraining() { _isTrainingPaused.value = true; wakeupSignal.trySend(Unit) }
    fun resumeTraining() {
        synchronized(queueLock) {
            oomRetryCounts.clear()
        }
        _isTrainingPaused.value = false
        wakeupSignal.trySend(Unit)
    }

    fun togglePauseTraining() {
        val resumed = _isTrainingPaused.value
        if (resumed) {
            synchronized(queueLock) {
                oomRetryCounts.clear()
            }
        }
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
            val committedItems = _nextQueue.value.filter {
                it.status == TrainingStatus.OPTIMIZER_COMMITTED
            }
            _nextQueue.value = committedItems
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
        require(chunkSize > 0) { "chunkSize harus > 0." }
        require(overlap >= 0) { "overlap tidak boleh negatif." }
        require(overlap < chunkSize) {
            "overlap ($overlap) harus < chunkSize ($chunkSize) agar iterasi maju."
        }
        require(epochs in 1..MAX_DOCUMENT_EPOCHS) {
            "epochs harus berada pada rentang 1..$MAX_DOCUMENT_EPOCHS"
        }
        require(rightsStatus in setOf("UNVERIFIED", "USER_DECLARED", "LICENSE_REVIEWED")) {
            "Unsupported corpus rights status: $rightsStatus"
        }
        requireDeclaredRightsForWebContent(source, canonicalUrl, rightsStatus)

        if (fullText.isBlank() || closed) return 0

        val processedChunks = TextModalityProcessor.process(
            rawText = fullText,
            documentTitle = title,
            chunkSize = chunkSize,
            overlap = overlap,
            tokenizer = neuralEngine.tokenizer
        )
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
            characterCount = fullText.length,
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
                    totalCharacters = fullText.length,
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
    /**
     * Reads incrementally but retains the complete source transcript until modality
     * detection and token-aware processing have finished. Splitting before modality
     * detection can turn partial conversation roles into unrelated prose and damage SFT.
     * JSONL records remain atomic in TextModalityProcessor.
     *
     * chunkSize and overlap are token counts from the active neural-engine tokenizer.
     * Model context-window construction remains the engine's responsibility.
     *
     * @return Pair(number of generated chunks, total UTF-16 characters read)
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
        require(chunkSize > 0) { "chunkSize must be positive." }
        require(overlap >= 0) { "overlap cannot be negative." }
        require(overlap < chunkSize) {
            "overlap ($overlap) must be less than chunkSize ($chunkSize)."
        }
        require(overlap <= chunkSize / 3) {
            "overlap ($overlap) must be <= one third of chunkSize (${chunkSize / 3})."
        }
        require(epochs in 1..MAX_DOCUMENT_EPOCHS) {
            "epochs must be in 1..$MAX_DOCUMENT_EPOCHS."
        }
        require(rightsStatus in setOf("UNVERIFIED", "USER_DECLARED", "LICENSE_REVIEWED")) {
            "Unsupported corpus rights status: $rightsStatus"
        }
        requireDeclaredRightsForWebContent(source, canonicalUrl, rightsStatus)
        if (closed) return Pair(0, 0L)

        val sourceText = StringBuilder()
        val buffer = CharArray(4096)
        val fingerprintDigest = MessageDigest.getInstance("SHA-256")
        var totalCharsRead = 0L

        while (true) {
            val readCount = reader.read(buffer)
            if (readCount < 0) break
            if (readCount == 0) continue
            if (closed) {
                sourceText.setLength(0)
                return Pair(0, totalCharsRead)
            }

            updateUtf16BeDigest(fingerprintDigest, buffer, readCount)
            sourceText.append(buffer, 0, readCount)
            totalCharsRead += readCount.toLong()
        }

        if (closed) {
            sourceText.setLength(0)
            return Pair(0, totalCharsRead)
        }

        val completeTranscript = sourceText.toString()
        if (completeTranscript.isBlank()) return Pair(0, totalCharsRead)

        validateStreamedJsonValueSizes(completeTranscript)

        val processedChunks = TextModalityProcessor.process(
            rawText = completeTranscript,
            documentTitle = title,
            chunkSize = chunkSize,
            overlap = overlap,
            tokenizer = neuralEngine.tokenizer
        )
        val rawChunks = processedChunks.map(ProcessedChunk::content)
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
            return Pair(0, totalCharsRead)
        }

        val totalScheduledChunks = Math.multiplyExact(chunksPerEpoch, epochs)
        val newItems = ArrayList<TrainingData>(totalScheduledChunks)
        for (epochIndex in 0 until epochs) {
            rawChunks.forEachIndexed { chunkIndex, chunkText ->
                val chunkTitle = if (epochs == 1) {
                    "$title [Block ${chunkIndex + 1}/$chunksPerEpoch]"
                } else {
                    "$title [Block ${chunkIndex + 1}/$chunksPerEpoch] [Epoch ${epochIndex + 1}/$epochs]"
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

    private fun validateStreamedJsonValueSizes(text: String) {
        fun skipWhitespace(start: Int): Int {
            var index = start
            while (index < text.length && text[index].isWhitespace()) index++
            return index
        }

        var cursor = skipWhitespace(0)
        if (cursor >= text.length) return

        if (text[cursor] == '[') {
            val end = TextModalityProcessor.findCompleteJsonValueEnd(text, cursor)
                ?: throw IllegalArgumentException(
                    "The streamed JSON array is incomplete. Check its closing brackets."
                )
            require(end - cursor <= MAX_STREAMED_JSON_VALUE_CHARACTERS) {
                "A streamed JSON training array exceeds " +
                    "$MAX_STREAMED_JSON_VALUE_CHARACTERS characters. " +
                    "Use JSONL with one training record per line."
            }
            require(skipWhitespace(end) == text.length) {
                "Unexpected content follows the streamed JSON array."
            }
            return
        }

        if (text[cursor] != '{') return

        while (cursor < text.length) {
            cursor = skipWhitespace(cursor)
            if (cursor >= text.length || text[cursor] != '{') return

            val end = TextModalityProcessor.findCompleteJsonValueEnd(text, cursor)
                ?: throw IllegalArgumentException(
                    "A streamed JSON training record is incomplete. " +
                        "Check the final JSONL record or closing JSON braces."
                )
            require(end - cursor <= MAX_STREAMED_JSON_VALUE_CHARACTERS) {
                "A streamed JSON training record exceeds " +
                    "$MAX_STREAMED_JSON_VALUE_CHARACTERS characters. " +
                    "Use smaller JSONL records."
            }
            cursor = end
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
        if (fullText.isBlank() || closed) return false

        // Store the exact source document. The validation evaluator applies the
        // modality serializer once, immediately before tokenization.
        val documentId = sha256Utf16Be(fullText)
        val document = CorpusDocumentEntity(
            id = documentId,
            contentFingerprint = documentId,
            fingerprintAlgorithm = "SHA-256/UTF-16BE",
            source = source,
            title = title,
            language = language,
            canonicalUrl = canonicalUrl,
            characterCount = fullText.length,
            chunkCount = 0,
            datasetSplit = "VALIDATION",
            validationText = fullText,
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
        // Keep one complete validation sequence per source document. The neural
        // engine owns context-windowing so overlapping ingestion chunks cannot
        // duplicate prediction targets or distort corpus-level loss weighting.
        val processedDocuments = documents.map { document ->
            TextModalityProcessor.formatTrainingDocument(document.validationText.orEmpty())
        }
        val texts = processedDocuments.filter { it.isNotBlank() }
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
        val reasoningDocumentCount = texts.count { text ->
            text.contains(Tokenizer.THINK_START) && text.contains(Tokenizer.THINK_END)
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

    /**
     * Runs a hand-authored conversational SFT evaluation suite against the live model.
     * Answers are generated deterministically and are never queued or persisted as
     * training examples. Reference facts stay independent from model outputs.
     */
    suspend fun evaluateConversationalValidation(
        cases: List<ConversationValidationCase> = ConversationValidationSuite.cases,
        thresholds: ConversationalValidationThresholds = ConversationalValidationThresholds()
    ): ConversationalValidationReport {
        return ConversationalValidationEvaluator.evaluate(
            cases = cases,
            thresholds = thresholds
        ) { conversationPrompt ->
            neuralEngine.generate(
                prompt = conversationPrompt,
                maxNewTokens = 80,
                temperature = 0f,
                includeReasoningMarkers = false,
                reasoningMode = false
            )
        }
    }

    /**
     * Returns an explicit candidate eligibility decision. It does not replace a
     * checkpoint or equate lower training loss with conversational improvement.
     */
    suspend fun assessConversationalCandidate(
        baseline: ConversationalValidationReport,
        cases: List<ConversationValidationCase> = ConversationValidationSuite.cases,
        thresholds: ConversationalValidationThresholds = ConversationalValidationThresholds()
    ): Pair<ConversationalValidationReport, ConversationPromotionDecision> {
        val candidate = evaluateConversationalValidation(cases, thresholds)
        return candidate to ConversationalValidationGate.assessPromotion(
            baseline = baseline,
            candidate = candidate
        )
    }

    fun retryConversationalValidation() {
        resumeTraining()
    }

    /**
     * Runs the fixed evaluation suite on the accepted model without training or promotion.
     * Manual evaluation is blocked while any training item is active or queued.
     */
    fun runConversationalValidation() {
        if (manualValidationJob?.isActive == true) return
        manualValidationJob = scope.launch {
            conversationalValidationMutex.lock()
            try {
                if (_currentTrainingItem.value != null || _nextQueue.value.isNotEmpty()) {
                    _conversationalValidationState.update { current ->
                        current.copy(
                            message = "Wait until the training queue is empty before running a manual conversational evaluation."
                        )
                    }
                    return@launch
                }

                setConversationalValidationStatus(
                    ConversationValidationStatus.EVALUATING,
                    "Evaluating the current local model against the fixed conversational suite."
                )
                val activeCheckpoint = File(validationFilesDirectory, "argus_model_weights.bin")
                if ((!activeCheckpoint.isFile || activeCheckpoint.length() < 4L) &&
                    !neuralEngine.saveCheckpoint()
                ) {
                    throw IllegalStateException("The current model checkpoint could not be saved.")
                }
                check(conversationalValidationPersistence.ensureAcceptedCheckpoint(activeCheckpoint)) {
                    "The accepted checkpoint could not be secured."
                }

                val currentBaseline = conversationalValidationPersistence.loadState().baselineReport
                if (currentBaseline == null || !ConversationValidationContract.isCurrent(currentBaseline)) {
                    check(restoreAcceptedConversationCheckpoint()) {
                        "The accepted checkpoint could not be restored before baseline evaluation."
                    }
                    val baseline = evaluateConversationalValidation()
                    check(ConversationValidationContract.isCurrent(baseline)) {
                        "The conversational baseline was produced with an unexpected evaluation contract."
                    }
                    check(conversationalValidationPersistence.saveBaseline(baseline)) {
                        "The conversational baseline could not be persisted."
                    }
                } else {
                    val report = evaluateConversationalValidation()
                    check(
                        conversationalValidationPersistence.recordCandidate(
                            report = report,
                            decision = null,
                            status = ConversationValidationStatus.BASELINE_READY,
                            message = "Manual evaluation completed. No training or checkpoint promotion was performed."
                        )
                    ) { "The conversational evaluation report could not be persisted." }
                }
                _conversationalValidationState.value = conversationalValidationPersistence.loadState()
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Throwable) {
                Log.e(TAG, "Manual conversational validation failed.", error)
                setConversationalValidationStatus(
                    ConversationValidationStatus.FAILED,
                    "Conversational evaluation failed: ${error.message}"
                )
            } finally {
                conversationalValidationMutex.unlock()
            }
        }
    }

    private fun restoreAcceptedConversationCheckpoint(): Boolean {
        val activeCheckpoint = File(validationFilesDirectory, "argus_model_weights.bin")
        return conversationalValidationPersistence.restoreAcceptedCheckpoint(activeCheckpoint) &&
            neuralEngine.loadCheckpointIfExists()
    }

    private suspend fun prepareConversationalTrainingRun() {
        conversationalValidationMutex.lock()
        try {
            setConversationalValidationStatus(
                ConversationValidationStatus.PREPARING_BASELINE,
                "Securing the accepted checkpoint and preparing a fixed conversational baseline."
            )

            val activeCheckpoint = File(validationFilesDirectory, "argus_model_weights.bin")
            if ((!activeCheckpoint.isFile || activeCheckpoint.length() < 4L) &&
                !neuralEngine.saveCheckpoint()
            ) {
                throw IllegalStateException("The current model checkpoint could not be saved.")
            }
            check(conversationalValidationPersistence.ensureAcceptedCheckpoint(activeCheckpoint)) {
                "The accepted checkpoint could not be secured."
            }

            val storedBaseline = conversationalValidationPersistence.loadState().baselineReport
            if (storedBaseline == null || !ConversationValidationContract.isCurrent(storedBaseline)) {
                check(restoreAcceptedConversationCheckpoint()) {
                    "The accepted checkpoint could not be restored before baseline evaluation."
                }
                val baseline = evaluateConversationalValidation()
                check(ConversationValidationContract.isCurrent(baseline)) {
                    "The conversational baseline was produced with an unexpected evaluation contract."
                }
                check(conversationalValidationPersistence.saveBaseline(baseline)) {
                    "The conversational baseline could not be persisted."
                }
            }

            setConversationalValidationStatus(
                ConversationValidationStatus.TRAINING,
                "Training is running against a secured conversational baseline. Evaluation will begin after the complete queue drains."
            )
        } finally {
            conversationalValidationMutex.unlock()
        }
    }

    private suspend fun completeConversationalTrainingRun(hadFailures: Boolean) {
        conversationalValidationMutex.lock()
        try {
            if (hadFailures) {
                check(restoreAcceptedConversationCheckpoint()) {
                    "The previous accepted checkpoint could not be restored after a failed training item."
                }

                val successfulDocumentIds =
                    validationRunDocumentIds - validationRunFailedDocumentIds
                var replayedDocumentCount = 0
                if (successfulDocumentIds.isNotEmpty()) {
                    val activeRepository = requireNotNull(repo) {
                        "Local corpus storage is unavailable; successful documents cannot be replayed after rollback."
                    }
                    val memoryActiveDocumentIds = _nextQueue.value
                        .mapNotNull(TrainingData::documentId)
                        .toSet()
                    val persistedActiveDocumentIds =
                        activeRepository.getActiveCorpusTrainingDocumentIds()
                    val replayDocumentIds = successfulDocumentIds -
                        memoryActiveDocumentIds - persistedActiveDocumentIds

                    if (replayDocumentIds.isNotEmpty()) {
                        activeRepository.markCorpusDocumentsQueued(replayDocumentIds)
                        for (documentId in replayDocumentIds.sorted()) {
                            val document = activeRepository.getCorpusDocument(documentId)
                                ?: throw IllegalStateException(
                                    "Training corpus document disappeared during rollback recovery: $documentId"
                                )
                            val replayedChunks = replayCorpusDocument(
                                documentId = documentId,
                                epochs = document.plannedEpochs
                            )
                            check(replayedChunks > 0) {
                                "Successful training document could not be re-queued after checkpoint rollback: $documentId"
                            }
                            replayedDocumentCount++
                        }
                    }
                }

                val rollbackMessage = buildString {
                    append("At least one training item failed. The candidate run was rejected and the previous accepted checkpoint was restored.")
                    if (replayedDocumentCount > 0) {
                        append(" Re-queued ")
                        append(replayedDocumentCount)
                        append(" previously successful document(s) because their weight updates were rolled back.")
                    }
                    if (validationRunFailedDocumentIds.isNotEmpty()) {
                        append(" Failed documents remain marked partial for explicit recovery.")
                    }
                }
                check(
                    conversationalValidationPersistence.recordStatus(
                        ConversationValidationStatus.ROLLED_BACK,
                        rollbackMessage,
                        clearDecision = true
                    )
                ) { "The rollback result could not be persisted." }
                _conversationalValidationState.value = conversationalValidationPersistence.loadState()
                return
            }

            setConversationalValidationStatus(
                ConversationValidationStatus.EVALUATING,
                "The training queue is complete. Evaluating factual coverage, instruction compliance, multi-turn consistency, and contradictions."
            )
            check(neuralEngine.saveCheckpoint()) {
                "The trained candidate checkpoint could not be saved for evaluation."
            }

            val baseline = conversationalValidationPersistence.loadState().baselineReport
                ?: throw IllegalStateException("The accepted conversational baseline is missing.")
            val (candidate, decision) = assessConversationalCandidate(baseline)

            if (decision.accepted) {
                check(
                    conversationalValidationPersistence.promoteCheckpoint(
                        activeCheckpoint = File(validationFilesDirectory, "argus_model_weights.bin"),
                        report = candidate,
                        decision = decision
                    )
                ) { "The improved checkpoint could not be committed as the new best-known model." }
                _conversationalValidationState.value = conversationalValidationPersistence.loadState()
            } else {
                check(restoreAcceptedConversationCheckpoint()) {
                    "The candidate failed validation and the previous accepted checkpoint could not be restored."
                }
                check(
                    conversationalValidationPersistence.recordCandidate(
                        report = candidate,
                        decision = decision,
                        status = ConversationValidationStatus.ROLLED_BACK,
                        message = "${decision.reason} The previous accepted checkpoint was restored."
                    )
                ) { "The rejected candidate report could not be persisted." }
                _conversationalValidationState.value = conversationalValidationPersistence.loadState()
            }
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Throwable) {
            Log.e(TAG, "Conversational checkpoint evaluation or promotion failed.", error)
            val restored = runCatching { restoreAcceptedConversationCheckpoint() }.getOrDefault(false)
            val status = if (restored) {
                ConversationValidationStatus.ROLLED_BACK
            } else {
                ConversationValidationStatus.FAILED
            }
            val message = if (restored) {
                "Conversational evaluation failed: ${error.message}. The previous accepted checkpoint was restored."
            } else {
                "Conversational evaluation failed and the previous accepted checkpoint could not be restored: ${error.message}"
            }
            conversationalValidationPersistence.recordStatus(status, message, clearDecision = true)
            _conversationalValidationState.value = conversationalValidationPersistence.loadState().copy(
                status = status,
                message = message,
                updatedAtEpochMs = System.currentTimeMillis()
            )
        } finally {
            conversationalValidationMutex.unlock()
        }
    }

    private fun setConversationalValidationStatus(
        status: ConversationValidationStatus,
        message: String
    ) {
        conversationalValidationPersistence.recordStatus(status, message)
        _conversationalValidationState.update { current ->
            current.copy(
                status = status,
                message = message,
                updatedAtEpochMs = System.currentTimeMillis()
            )
        }
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
        if (content.isBlank() || closed) return

        if (neuralEngine.tokenizer.encode(content, autoRegister = false).size > 380) {
            scope.launch {
                runSuspendCatching { enqueueDocument(source, title, content) }
                    .onFailure { error ->
                        Log.e(TAG, "Failed to persist long-form training input", error)
                    }
            }
            return
        }

        val processed = TextModalityProcessor.process(
            rawText = content,
            documentTitle = title,
            chunkSize = 380,
            overlap = 40,
            tokenizer = neuralEngine.tokenizer
        )
        if (processed.isEmpty()) return

        val newItems = processed.map { pChunk ->
            TrainingData(
                source = source,
                title = if (processed.size > 1) pChunk.segmentTitle else title.ifBlank { "Data Stimulus" },
                content = pChunk.content,
                status = TrainingStatus.QUEUED
            )
        }

        synchronized(queueLock) {
            if (closed) return
            val current = _nextQueue.value
            val nonDuplicates = newItems.filter { item ->
                current.none { it.content == item.content && it.documentTitle == null }
            }
            if (nonDuplicates.isNotEmpty()) appendPersistingQueueItems(nonDuplicates)
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

    private suspend fun reconcilePersistedTrainingCommit(
        repository: com.argus.agi.data.repository.ArgusRepository
    ) {
        val snapshot = TrainingCommitJournal.read(trainingCommitJournalFile) ?: return
        val currentOptimizerStep = neuralEngine.optimizerStep

        if (!snapshot.isCommitted) {
            check(currentOptimizerStep == snapshot.baselineOptimizerStep) {
                "A prepared training journal does not match the restored optimizer checkpoint."
            }
            TrainingCommitJournal.delete(trainingCommitJournalFile)
            return
        }

        val committedOptimizerStep = requireNotNull(snapshot.committedOptimizerStep)
        if (currentOptimizerStep < committedOptimizerStep) {
            val committedIds = repository.getOptimizerCommittedTrainingItemsSnapshot()
                .mapTo(HashSet()) { it.id }
            check(snapshot.entries.none { it.itemId in committedIds }) {
                "Room marks a batch committed, but the restored checkpoint is older than its optimizer commit."
            }
            // The write-ahead journal reached COMMITTED, but the following checkpoint
            // replacement did not. The only durable model state is the saved baseline.
            TrainingCommitJournal.delete(trainingCommitJournalFile)
            return
        }

        val committedItems = ArrayList<TrainingData>(snapshot.entries.size)
        for (entry in snapshot.entries) {
            val existing = repository.getTrainingItem(entry.itemId)
                ?: throw IllegalStateException(
                    "The committed training journal references a missing queue item: ${entry.itemId}."
                )
            if (
                existing.status == TrainingStatus.COMPLETED ||
                existing.status == TrainingStatus.FAILED
            ) {
                continue
            }
            val result = requireNotNull(entry.result) {
                "A committed journal entry is missing its training result."
            }
            committedItems.add(
                existing.copy(
                    status = TrainingStatus.OPTIMIZER_COMMITTED,
                    finalLoss = result.loss,
                    stepDurationMs = entry.stepDurationMs
                )
            )
        }
        if (committedItems.isNotEmpty()) {
            repository.markTrainingItemsOptimizerCommitted(committedItems)
        }
    }

    private suspend fun deleteTrainingCommitJournalWhenResolved() {
        val snapshot = TrainingCommitJournal.read(trainingCommitJournalFile) ?: return
        if (!snapshot.isCommitted) return
        val repository = repo ?: return
        val pendingCommittedIds = repository.getOptimizerCommittedTrainingItemsSnapshot()
            .asSequence()
            .map(TrainingData::id)
            .toHashSet()
        if (snapshot.entries.none { entry -> entry.itemId in pendingCommittedIds }) {
            TrainingCommitJournal.delete(trainingCommitJournalFile)
        }
    }

    /**
     * Requeues a pre-commit retry ahead of ordinary queued work. The earlier timestamp
     * is persisted so the same priority survives an app restart.
     */
    private fun prependPersistingQueueItems(items: List<TrainingData>): List<TrainingData> {
        if (items.isEmpty()) return emptyList()
        synchronized(queueLock) {
            val existingIds = _nextQueue.value.mapTo(HashSet()) { it.id }
            val uniqueItems = items.filterNot { it.id in existingIds }
            if (uniqueItems.isEmpty()) return emptyList()

            val earliestExistingTimestamp = _nextQueue.value.minOfOrNull { it.timestamp }
                ?: (System.currentTimeMillis() + 1L)
            val firstTimestamp = (earliestExistingTimestamp - uniqueItems.size.toLong())
                .coerceAtLeast(0L)
            val orderedItems = uniqueItems.mapIndexed { index, item ->
                item.copy(
                    timestamp = firstTimestamp + index.toLong(),
                    status = TrainingStatus.QUEUED
                )
            }
            val persistingItems = orderedItems.map { item ->
                item.copy(status = TrainingStatus.PERSISTING)
            }
            _nextQueue.value = persistingItems + _nextQueue.value

            val sendResult = persistenceOperations.trySend(
                QueuePersistenceOperation.Save(
                    generation = queueGeneration,
                    items = orderedItems
                )
            )
            if (sendResult.isFailure) {
                val itemIds = orderedItems.mapTo(HashSet(orderedItems.size)) { it.id }
                _nextQueue.value = _nextQueue.value.filterNot { it.id in itemIds }
                Log.e(TAG, "Training persistence worker rejected the high-priority OOM retry batch.")
                return emptyList()
            }
            return orderedItems
        }
    }

    private fun prependCommittedQueueItems(items: List<TrainingData>) {
        if (items.isEmpty()) return
        synchronized(queueLock) {
            val queuedIds = _nextQueue.value.mapTo(HashSet()) { it.id }
            val additions = items
                .filterNot { it.id in queuedIds }
                .map { it.copy(status = TrainingStatus.OPTIMIZER_COMMITTED) }
                .sortedBy { it.timestamp }
            if (additions.isNotEmpty()) {
                _nextQueue.value = additions + _nextQueue.value
                wakeupSignal.trySend(Unit)
            }
        }
    }

    private fun startTrainingWorker() {
        workerJob = scope.launch {
            try {
                if (conversationalValidationPersistence.hasAcceptedCheckpoint()) {
                    check(restoreAcceptedConversationCheckpoint()) {
                        "The last accepted conversational checkpoint could not be restored."
                    }
                }
            } catch (error: Throwable) {
                Log.e(TAG, "Failed to restore the accepted conversational checkpoint.", error)
                _isTrainingPaused.value = true
                setConversationalValidationStatus(
                    ConversationValidationStatus.FAILED,
                    "Training is paused because the last accepted checkpoint could not be restored: ${error.message}"
                )
            }

            var validationRunActive = false
            var validationRunHadFailures = false

            while (isActive && !closed) {
                if (_isTrainingPaused.value) {
                    _isTrainingPaused.first { !it || closed }
                    if (closed) break
                }

                combine(deviceGuard.safetyState, workerShutdown) { safety, shutdownRequested ->
                    safety.isSafeToTrain || shutdownRequested
                }.first { canContinue -> canContinue }
                if (closed) break

                resetForceOverrideIfSet()
                deviceGuard.refreshState()

                val firstQueuedItem = _nextQueue.value.firstOrNull()
                if (!validationRunActive && firstQueuedItem?.status == TrainingStatus.QUEUED) {
                    validationRunDocumentIds.clear()
                    validationRunFailedDocumentIds.clear()
                    try {
                        prepareConversationalTrainingRun()
                        validationRunActive = true
                        validationRunHadFailures = false
                    } catch (cancelled: CancellationException) {
                        throw cancelled
                    } catch (error: Throwable) {
                        Log.e(TAG, "Could not prepare the conversational training baseline.", error)
                        _isTrainingPaused.value = true
                        setConversationalValidationStatus(
                            ConversationValidationStatus.FAILED,
                            "Training is paused because its conversational baseline could not be secured: ${error.message}"
                        )
                        continue
                    }
                }

                val safetySnapshot = deviceGuard.safetyState.value
                val memorySnapshot = captureTrainingMemorySnapshot()
                val thermalBatchLimit = DeviceAdaptiveRuntimePolicy.trainingBatchLimit(
                    configuredBatchSize = neuralEngine.config.trainingBatchSize,
                    thermalStatus = safetySnapshot.thermalStatus,
                    availableHeapBytes = memorySnapshot.availableHeapBytes,
                    maximumHeapBytes = memorySnapshot.maximumHeapBytes,
                    availableSystemMemoryBytes = memorySnapshot.availableSystemMemoryBytes,
                    systemLowMemory = memorySnapshot.isSystemLowMemory,
                    isPowerSaveMode = safetySnapshot.isPowerSaveMode,
                    isCharging = safetySnapshot.isCharging,
                    batteryPct = safetySnapshot.batteryPct,
                    memoryRecovery = consecutiveOomRecoveries > 0
                )
                if (thermalBatchLimit == 0) {
                    // Severe thermal states are normally blocked by DeviceGuard above.
                    // Re-enter the safety wait instead of spinning or starting another batch.
                    continue
                }

                val runtimeContextLength = DeviceAdaptiveRuntimePolicy.trainingContextLimit(
                    configuredContextLength = neuralEngine.config.maxSeqLen,
                    availableHeapBytes = memorySnapshot.availableHeapBytes,
                    maximumHeapBytes = memorySnapshot.maximumHeapBytes,
                    dModel = neuralEngine.config.dModel,
                    feedForwardSize = neuralEngine.config.dFf,
                    vocabularySize = neuralEngine.tokenizer.vocabSize,
                    layerCount = neuralEngine.config.layerCount,
                    headCount = neuralEngine.config.nHeads,
                    thermalStatus = safetySnapshot.thermalStatus,
                    availableSystemMemoryBytes = memorySnapshot.availableSystemMemoryBytes,
                    systemLowMemory = memorySnapshot.isSystemLowMemory,
                    isPowerSaveMode = safetySnapshot.isPowerSaveMode,
                    isCharging = safetySnapshot.isCharging,
                    memoryRecovery = consecutiveOomRecoveries > 0
                )

                val firstItem = popNext()
                if (firstItem == null) {
                    if (_nextQueue.value.isEmpty()) {
                        _currentTrainingItem.value = null
                        _activeDocument.value = null
                        if (validationRunActive) {
                            completeConversationalTrainingRun(validationRunHadFailures)
                            validationRunActive = false
                            validationRunHadFailures = false
                        }
                    }
                    withTimeoutOrNull(IDLE_POLL_MS) {
                        wakeupSignal.receiveCatching()
                    }
                    continue
                }

                val batchItems = ArrayList<TrainingData>(thermalBatchLimit)
                batchItems += firstItem
                while (batchItems.size < thermalBatchLimit) {
                    val nextItem = popNext() ?: break
                    batchItems += nextItem
                }

                val activeItems = batchItems.map { it.copy(status = TrainingStatus.TRAINING) }
                _currentTrainingItem.value = activeItems.first()
                updateActiveDocumentEntering(activeItems.first())

                try {
                    if (!processBatch(activeItems, runtimeContextLength)) {
                        validationRunHadFailures = true
                    } else {
                        consecutiveOomRecoveries = 0
                    }
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (outOfMemory: OutOfMemoryError) {
                    Log.e(
                        TAG,
                        "Training micro-batch hit memory pressure; preserving queued samples and reducing the next workload.",
                        outOfMemory
                    )
                    val residencyReleased = runCatching {
                        neuralEngine.releaseVulkanTensorResidencyForMemoryPressure()
                    }.getOrDefault(false)
                    val checkpointSaved = runCatching {
                        neuralEngine.saveCheckpoint()
                    }.getOrDefault(false)
                    val requeued = appendPersistingQueueItems(
                        activeItems.map { item -> item.copy(status = TrainingStatus.QUEUED) }
                    )
                    _currentTrainingItem.value = null
                    _activeDocument.value = null

                    if (requeued.isEmpty()) {
                        _isTrainingPaused.value = true
                        Log.e(TAG, "OOM recovery could not return the active samples to the durable queue.")
                    } else {
                        consecutiveOomRecoveries++
                        if (!residencyReleased || !checkpointSaved || consecutiveOomRecoveries > 1) {
                            _isTrainingPaused.value = true
                            Log.e(
                                TAG,
                                "Training paused after OOM recovery: residencyReleased=$residencyReleased, " +
                                    "checkpointSaved=$checkpointSaved, retries=$consecutiveOomRecoveries."
                            )
                        } else {
                            Log.w(
                                TAG,
                                "OOM recovery saved the model and requeued ${requeued.size} samples; " +
                                    "retrying once with batch 1 and a reduced context."
                            )
                        }
                    }
                } catch (error: Throwable) {
                    validationRunHadFailures = true
                    activeItems.mapNotNull(TrainingData::documentId)
                        .forEach(validationRunFailedDocumentIds::add)
                    Log.e(
                        TAG,
                        "Training micro-batch failed (${activeItems.size} samples).",
                        error
                    )
                    activeItems.forEach { item ->
                        markTrainingItemFailed(item, error.message)
                    }
                } finally {
                    resetForceOverrideIfSet()
                }

                val currentSafety = deviceGuard.safetyState.value
                val throttleMs = DeviceAdaptiveRuntimePolicy.trainingBatchDelayMillis(
                    thermalStatus = currentSafety.thermalStatus,
                    temperatureCelsius = currentSafety.temperatureCelsius,
                    batteryPct = currentSafety.batteryPct,
                    isCharging = currentSafety.isCharging,
                    isPowerSaveMode = currentSafety.isPowerSaveMode
                )
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

    private fun failPendingPersistenceOperation(
        operation: QueuePersistenceOperation,
        error: Throwable
    ) {
        when (operation) {
            is QueuePersistenceOperation.Start -> {
                operation.completion.complete(Result.failure(error))
            }
            is QueuePersistenceOperation.Barrier -> {
                operation.completion.completeExceptionally(error)
            }
            is QueuePersistenceOperation.Clear,
            is QueuePersistenceOperation.Save -> Unit
        }
    }

    private suspend fun processQueuePersistenceOperation(
        operation: QueuePersistenceOperation
    ) {
        when (operation) {
            is QueuePersistenceOperation.Clear -> {
                runSuspendCatching {
                    val repository = repo ?: throw IllegalStateException(
                        "Training queue persistence is unavailable.",
                        repositoryInitialization.exceptionOrNull()
                    )
                    repository.clearPendingQueueAndMarkInterruptedCorpusDocumentsPartial()
                }.onFailure { error ->
                    Log.e(TAG, "Failed to clear persisted training queue", error)
                }
            }

            is QueuePersistenceOperation.Start -> {
                val result = runSuspendCatching {
                    val repository = repo ?: throw IllegalStateException(
                        "Training queue persistence is unavailable.",
                        repositoryInitialization.exceptionOrNull()
                    )
                    check(repository.markTrainingStarted(operation.itemId)) {
                        "Training queue item '${operation.itemId}' was not marked as TRAINING in Room."
                    }
                }
                result.onFailure { error ->
                    Log.e(
                        TAG,
                        "Failed to persist active training state for '${operation.itemId}'",
                        error
                    )
                }
                operation.completion.complete(result)
            }

            is QueuePersistenceOperation.Barrier -> {
                operation.completion.complete(Unit)
            }

            is QueuePersistenceOperation.Save -> {
                val isCurrentGeneration = synchronized(queueLock) {
                    operation.generation == queueGeneration
                }
                if (!isCurrentGeneration) return

                val persistenceFailure = runSuspendCatching {
                    val repository = repo ?: throw IllegalStateException(
                        "Training queue persistence is unavailable.",
                        repositoryInitialization.exceptionOrNull()
                    )
                    repository.saveTrainingItems(operation.items)
                }.exceptionOrNull()

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

    /**
     * Reserves the next item only after Room confirms its TRAINING transition.
     * Internal visibility allows persistence-contract tests to exercise the same path
     * used by the worker without starting the optimizer loop.
     */
    internal suspend fun popNext(): TrainingData? {
        val startCompletion = CompletableDeferred<Result<Unit>>()
        val reservation = synchronized(queueLock) {
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
            first to queueGeneration
        } ?: return null

        val (item, reservationGeneration) = reservation

        // Start is ordered in the same persistence channel as Clear. Training must not
        // begin until Room confirms that the queue row is now TRAINING.
        val persistenceResult = startCompletion.await()
        if (persistenceResult.isFailure) {
            synchronized(queueLock) {
                // A clear that happened while the Room update was pending must win.
                // Never resurrect a queue item from an older generation.
                if (
                    !closed &&
                    queueGeneration == reservationGeneration &&
                    _nextQueue.value.none { queued -> queued.id == item.id }
                ) {
                    _nextQueue.value = listOf(item) + _nextQueue.value
                }
            }
            Log.e(
                TAG,
                "Training item '${item.id}' was returned to the queue because its start state was not persisted.",
                persistenceResult.exceptionOrNull()
            )
            return null
        }
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
        } catch (cancelled: CancellationException) {
            preparedTokenizerDocuments.remove(documentId)
            throw cancelled
        } catch (error: Exception) {
            preparedTokenizerDocuments.remove(documentId)
            Log.w(TAG, "Tokenizer preparation failed for document $documentId; existing vocabulary will be used.", error)
        }
    }

    private suspend fun processBatch(
        items: List<TrainingData>,
        runtimeContextLength: Int
    ): Boolean {
        require(items.isNotEmpty()) { "A training micro-batch cannot be empty." }
        items.mapNotNull(TrainingData::documentId)
            .forEach(validationRunDocumentIds::add)

        for (item in items) {
            item.documentId?.let { documentId ->
                if (item.chunkIndex == 1) {
                    repo?.let { activeRepository ->
                        runSuspendCatching {
                            activeRepository.markCorpusDocumentTraining(documentId)
                        }.onFailure { error ->
                            Log.w(TAG, "Failed to mark corpus document as training", error)
                        }
                    }
                }
                prepareTokenizerForDocument(item)
            }
        }

        val startNs = System.nanoTime()
        val batchResult = neuralEngine.trainBatchDetailed(
            texts = items.map(TrainingData::content),
            runtimeContextLength = runtimeContextLength
        )
        val elapsedMs = (System.nanoTime() - startNs) / 1_000_000L
        require(batchResult.samples.size == items.size) {
            "Neural engine returned ${batchResult.samples.size} results for ${items.size} samples."
        }
        val perItemDurationMs = (elapsedMs / items.size).coerceAtLeast(1L)

        var allItemsSucceeded = true
        for ((item, stepResult) in items.zip(batchResult.samples)) {
            try {
                if (!processItem(item, stepResult, perItemDurationMs)) {
                    allItemsSucceeded = false
                    item.documentId?.let(validationRunFailedDocumentIds::add)
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: OutOfMemoryError) {
                throw error
            } catch (error: Throwable) {
                allItemsSucceeded = false
                item.documentId?.let(validationRunFailedDocumentIds::add)
                Log.e(TAG, "Failed to finalize training item '${item.title}'.", error)
                markTrainingItemFailed(item, error.message)
            }
        }
        return allItemsSucceeded
    }

    private fun markTrainingItemFailed(item: TrainingData, reason: String?) {
        val failed = item.copy(status = TrainingStatus.FAILED)
        pushCompleted(failed)
        if (!reason.isNullOrBlank()) {
            Log.w(TAG, "Training item '${item.title}' failed: $reason")
        }
        scope.launch {
            repo?.let { activeRepository ->
                runSuspendCatching {
                    activeRepository.markTrainingFailed(failed.id)
                }.onFailure { error ->
                    Log.w(TAG, "Failed to persist failed training item", error)
                }
                failed.documentId?.let { documentId ->
                    runSuspendCatching {
                        activeRepository.markCorpusDocumentPartial(documentId)
                    }.onFailure { error ->
                        Log.w(TAG, "Failed to mark failed corpus document partial", error)
                    }
                }
            }
        }
    }

    private suspend fun processItem(
        item: TrainingData,
        stepResult: NeuralTrainingStepResult,
        durationMs: Long
    ): Boolean {
        updateActiveDocumentEntering(item)
        val stepLoss = stepResult.loss
        // Invalid loss is a failed step, never a completed training item.
        if (!stepLoss.isFinite()) {
            Log.w(TAG, "trainBatchDetailed returned non-finite loss=$stepLoss for '${item.title}'")
            markTrainingItemFailed(item, "The model produced a non-finite loss.")
            return false
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
                runSuspendCatching {
                    repository.markTrainingCompleted(finished.id, stepLoss, durationMs)
                }.onFailure { error ->
                    Log.w(TAG, "Failed to persist completed training item", error)
                }

                if (
                    finished.documentId != null &&
                    finished.chunkIndex >= finished.totalChunks
                ) {
                    runSuspendCatching {
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
        return true
    }

    /**
     * Preserve coroutine cancellation while converting ordinary persistence failures
     * into Result values for logging and queue recovery.
     */
    private suspend fun <T> runSuspendCatching(
        block: suspend () -> T
    ): Result<T> {
        return try {
            Result.success(block())
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Throwable) {
            Result.failure(error)
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
    // Lifecycle
    // =====================================================================

    /**
     * Stops the training worker and drains queued persistence operations before shutdown.
     * An active optimizer step is allowed to finish within the shutdown timeout.
     * Pending queue items remain persisted in Room and can be restored by the next manager.
     */
    override fun close() {
        if (closed) return
        closed = true
        workerShutdown.value = true
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