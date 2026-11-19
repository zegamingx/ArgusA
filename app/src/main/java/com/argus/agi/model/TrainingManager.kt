package com.argus.agi.model

import android.app.ActivityManager
import android.content.Context
import android.util.Log
import com.argus.agi.data.local.entity.CorpusChunkEntity
import com.argus.agi.data.local.entity.CorpusDocumentEntity
import com.argus.agi.data.local.entity.CorpusSourceProvenance
import com.argus.agi.data.local.entity.TrainingReplaySessionEntity
import com.argus.agi.runtime.DeviceAdaptiveRuntimePolicy
import com.argus.agi.runtime.RuntimeConfigurationPlanner
import com.argus.agi.runtime.RuntimeDeferReason
import com.argus.agi.runtime.RuntimeModelConstraints
import com.argus.agi.runtime.RuntimeResourceSnapshot
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
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.BufferedReader
import java.io.Closeable
import java.io.InputStreamReader
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.Reader
import java.io.StringReader
import java.nio.charset.StandardCharsets
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
    val documentId: String? = null,
    val replaySessionId: String? = null
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
    private val ingestionStagingDirectory = context.applicationContext.cacheDir
    private val trainingCommitJournalFile =
        File(validationFilesDirectory, TrainingCommitJournal.FILE_NAME)
    private val conversationalValidationPersistence =
        ConversationalValidationPersistence(validationFilesDirectory)

    private sealed interface QueuePersistenceOperation {
        data class Save(
            val generation: Long,
            val items: List<TrainingData>,
            val replaySession: TrainingReplaySessionEntity? = null
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
        // Import remains bounded even when the supplied Reader is unbounded.
        const val MAX_STREAMED_DOCUMENT_CHARACTERS = 2 * 1024 * 1024
        const val MAX_STREAMED_DOCUMENT_CHUNKS = 8_192
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

        if (repo == null) {
            Log.e(
                TAG,
                "Training persistence is unavailable; queued work will not be reported as durable.",
                repositoryInitialization.exceptionOrNull()
            )
        }
        startTrainingWorker()
    }

    fun pauseTraining() {
        _isTrainingPaused.value = true
        wakeupSignal.trySend(Unit)
    }

    private fun resumeAfterCommittedQueueRefresh(forceTrainingOnce: Boolean = false) {
        scope.launch {
            val repository = repo
            if (repository == null) {
                _isTrainingPaused.value = true
                Log.e(TAG, "Cannot resume training because durable queue storage is unavailable.")
                return@launch
            }
            try {
                awaitQueuePersistenceIdle()
                // Recover a prepared/committed journal before moving any interrupted
                // TRAINING row back to QUEUED. A committed optimizer step must first
                // receive its checkpoint; a partial pre-commit step must roll back.
                reconcilePersistedTrainingCommit(repository)
                repository.recoverInterruptedTrainingItems()
                // OOM may have disabled or suspended the native device even when the
                // host checkpoint was recovered. Re-arm the optional runtime before work.
                neuralEngine.resumeVulkanTensorResidencyAfterForeground()
                val committedItems = repository.getOptimizerCommittedTrainingItemsSnapshot()
                val pendingItems = repository.getPendingQueueSnapshot()
                val databaseIds = (committedItems + pendingItems).mapTo(HashSet()) { it.id }
                val inMemoryItems = synchronized(queueLock) {
                    _nextQueue.value.filterNot { item -> item.id in databaseIds }
                }
                val restoredItems = committedItems.sortedBy { it.timestamp } +
                    pendingItems.sortedBy { it.timestamp } +
                    inMemoryItems.map { item -> item.copy(status = TrainingStatus.QUEUED) }
                synchronized(queueLock) {
                    _nextQueue.value = restoredItems.distinctBy(TrainingData::id)
                    restoredItems.maxOfOrNull { it.timestamp }?.let { latest ->
                        lastScheduledTimestamp = maxOf(lastScheduledTimestamp, latest)
                    }
                }
                _currentTrainingItem.value = null
                _activeDocument.value = null
                synchronized(queueLock) {
                    oomRetryCounts.clear()
                }
                if (forceTrainingOnce) {
                    deviceGuard.setForceTrainingOverride(true)
                }
                _isTrainingPaused.value = false
                wakeupSignal.trySend(Unit)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Throwable) {
                Log.e(TAG, "Cannot resume training until checkpoint and queue recovery is complete.", error)
                _isTrainingPaused.value = true
            }
        }
    }

    fun resumeTraining() {
        resumeAfterCommittedQueueRefresh()
    }

    fun togglePauseTraining() {
        if (_isTrainingPaused.value) {
            resumeTraining()
        } else {
            pauseTraining()
        }
    }

    /**
     * Set force-training override SEKALI untuk menembus thermal/battery guard
     * pada langkah berikutnya. Override akan otomatis di-reset setelah satu trainStep.
     */
    fun triggerImmediateTrainingStep() {
        resumeAfterCommittedQueueRefresh(forceTrainingOnce = true)
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
        rightsStatus: String = "UNVERIFIED",
        sourceProvenance: CorpusSourceProvenance = CorpusSourceProvenance()
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
        sourceProvenance.validateFor(rightsStatus)
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
        val stored = activeRepository.storeCorpusDocument(corpusDocument, corpusChunks, sourceProvenance)
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
     * Ingests a Reader through a bounded disk spool instead of growing a transcript buffer
     * while the caller is still producing input. UTF-16 code units are staged verbatim so
     * SHA-256/UTF-16BE remains identical to enqueueDocument, even across surrogate boundaries.
     *
     * JSONL is processed one complete object per line from the spool. ChatML, ShareGPT,
     * reasoning and legacy plain-text documents retain document-level parser semantics.
     * A live-heap bound and conservative queue-capacity bound are enforced before parsing.
     *
     * @return Pair(number of generated queue chunks, total UTF-16 characters read)
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
        rightsStatus: String = "UNVERIFIED",
        sourceProvenance: CorpusSourceProvenance = CorpusSourceProvenance()
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
        sourceProvenance.validateFor(rightsStatus)
        requireDeclaredRightsForWebContent(source, canonicalUrl, rightsStatus)
        if (closed) return Pair(0, 0L)

        val availableHeapBytes = captureTrainingMemorySnapshot().availableHeapBytes
        val maximumRawChunks = MAX_STREAMED_DOCUMENT_CHUNKS / epochs
        val effectiveChunkStride = (chunkSize - overlap).coerceAtLeast(1)
        val queueBoundedCharacterLimit = (
            maximumRawChunks.toLong() * effectiveChunkStride.toLong() / 4L
        ).coerceIn(1L, MAX_STREAMED_DOCUMENT_CHARACTERS.toLong()).toInt()
        val streamCharacterLimit = DeviceAdaptiveRuntimePolicy.streamedDocumentCharacterLimit(
            availableHeapBytes = availableHeapBytes,
            hardLimitCharacters = queueBoundedCharacterLimit
        )
        require(streamCharacterLimit > 0) {
            "There is not enough available app-heap headroom for a streamed document import."
        }

        check(
            ingestionStagingDirectory.isDirectory ||
                ingestionStagingDirectory.mkdirs()
        ) {
            "Temporary ingestion storage is unavailable."
        }

        val stagingFile = File(
            ingestionStagingDirectory,
            "argus-ingest-${UUID.randomUUID()}.utf16be"
        )
        val inputBuffer = CharArray(4096)
        val encodedBuffer = ByteArray(inputBuffer.size * 2)
        val fingerprintDigest = MessageDigest.getInstance("SHA-256")
        var totalCharsRead = 0L

        try {
            BufferedOutputStream(FileOutputStream(stagingFile)).use { output ->
                while (true) {
                    currentCoroutineContext().ensureActive()
                    var readCount = reader.read(inputBuffer)
                    if (readCount == 0) {
                        // Reader implementations should make progress for a non-empty buffer,
                        // but handle a non-conforming zero-length read without spinning forever.
                        val singleCharacter = reader.read()
                        if (singleCharacter < 0) break
                        inputBuffer[0] = singleCharacter.toChar()
                        readCount = 1
                    }
                    if (readCount < 0) break
                    if (closed) return Pair(0, totalCharsRead)

                    val nextTotal = Math.addExact(totalCharsRead, readCount.toLong())
                    require(nextTotal <= streamCharacterLimit.toLong()) {
                        "Streamed document exceeds the current safe limit of " +
                            "$streamCharacterLimit UTF-16 characters. " +
                            "Split the source into smaller documents at a record or paragraph boundary."
                    }

                    updateUtf16BeDigest(fingerprintDigest, inputBuffer, readCount)
                    for (index in 0 until readCount) {
                        val value = inputBuffer[index].code
                        encodedBuffer[index * 2] = (value ushr 8).toByte()
                        encodedBuffer[index * 2 + 1] = value.toByte()
                    }
                    output.write(encodedBuffer, 0, readCount * 2)
                    totalCharsRead = nextTotal
                }
            }

            if (closed) return Pair(0, totalCharsRead)
            if (totalCharsRead == 0L) return Pair(0, 0L)

            val streamedJsonlChunks = processStagedJsonLinesIfValid(
                file = stagingFile,
                documentTitle = title,
                chunkSize = chunkSize,
                overlap = overlap
            )
            val processedChunks = streamedJsonlChunks ?: run {
                // Single-object ChatML, legacy plain text and other formats retain
                // their document-level parser semantics under the dynamic heap cap.
                val completeTranscript = readUtf16BeStagingFile(
                    file = stagingFile,
                    expectedCharacterCount = totalCharsRead.toInt()
                )
                if (completeTranscript.isBlank()) return Pair(0, totalCharsRead)

                validateStreamedJsonValueSizes(completeTranscript)
                TextModalityProcessor.process(
                    rawText = completeTranscript,
                    documentTitle = title,
                    chunkSize = chunkSize,
                    overlap = overlap,
                    tokenizer = neuralEngine.tokenizer
                )
            }
            val rawChunks = processedChunks.map(ProcessedChunk::content)
            val chunksPerEpoch = rawChunks.size
            if (chunksPerEpoch == 0) return Pair(0, totalCharsRead)

            val totalScheduledChunks = Math.multiplyExact(chunksPerEpoch, epochs)
            require(totalScheduledChunks <= MAX_STREAMED_DOCUMENT_CHUNKS) {
                "Streamed document would create $totalScheduledChunks queue chunks, " +
                    "exceeding the safe limit of $MAX_STREAMED_DOCUMENT_CHUNKS. " +
                    "Use a larger chunk size or split the source."
            }

            val documentId = fingerprintDigest.digest().toHexString()
            val corpusDocument = CorpusDocumentEntity(
                id = documentId,
                contentFingerprint = documentId,
                fingerprintAlgorithm = "SHA-256/UTF-16BE",
                source = source,
                title = title,
                language = language,
                canonicalUrl = canonicalUrl,
                characterCount = totalCharsRead.toInt(),
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
            val stored = activeRepository.storeCorpusDocument(
                corpusDocument,
                corpusChunks,
                sourceProvenance
            )
            if (!stored) {
                Log.i(TAG, "Duplicate corpus document skipped: $documentId")
                return Pair(0, totalCharsRead)
            }

            val newItems = ArrayList<TrainingData>(totalScheduledChunks)
            for (epochIndex in 0 until epochs) {
                rawChunks.forEachIndexed { chunkIndex, chunkText ->
                    val chunkTitle = if (epochs == 1) {
                        "$title [Block ${chunkIndex + 1}/$chunksPerEpoch]"
                    } else {
                        "$title [Block ${chunkIndex + 1}/$chunksPerEpoch] " +
                            "[Epoch ${epochIndex + 1}/$epochs]"
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
                        totalCharacters = totalCharsRead.toInt(),
                        totalChunks = orderedItems.size,
                        completedChunks = 0,
                        currentChunkIndex = 1
                    )
                }
                return Pair(orderedItems.size, totalCharsRead)
            }
        } finally {
            if (stagingFile.exists() && !stagingFile.delete()) {
                Log.w(TAG, "Could not remove temporary streamed-ingestion file: ${stagingFile.name}")
            }
        }
    }

    /**
     * Processes JSONL one complete record per line. A non-JSONL file returns null so the
     * caller can use the document-level parser without committing partial results.
     *
     * The returned list is bounded by MAX_STREAMED_DOCUMENT_CHUNKS. Each JSONObject is
     * parsed and released before the next source line is read; conversation objects stay
     * atomic because TextModalityProcessor receives the full JSON object for each line.
     */
    private suspend fun processStagedJsonLinesIfValid(
        file: File,
        documentTitle: String,
        chunkSize: Int,
        overlap: Int
    ): List<ProcessedChunk>? {
        val processed = ArrayList<ProcessedChunk>()
        var foundRecord = false

        BufferedReader(
            InputStreamReader(
                FileInputStream(file),
                StandardCharsets.UTF_16BE
            )
        ).use { reader ->
            while (true) {
                currentCoroutineContext().ensureActive()
                val line = reader.readLine() ?: break
                val record = line.removePrefix("\uFEFF").trim()
                if (record.isEmpty()) continue
                if (!record.startsWith('{')) return null
                require(record.length <= MAX_STREAMED_JSON_VALUE_CHARACTERS) {
                    "A streamed JSONL record exceeds " +
                        "$MAX_STREAMED_JSON_VALUE_CHARACTERS characters. " +
                        "Split the record before importing it."
                }

                validateStreamedJsonValueSizes(record)
                val isJsonObject = runCatching {
                    org.json.JSONObject(record)
                }.isSuccess
                if (!isJsonObject) return null
                require(processed.size < MAX_STREAMED_DOCUMENT_CHUNKS) {
                    "JSONL document exceeds the safe limit of " +
                        "$MAX_STREAMED_DOCUMENT_CHUNKS processed records."
                }

                foundRecord = true
                val nextChunks = TextModalityProcessor.process(
                    rawText = record,
                    documentTitle = documentTitle,
                    chunkSize = chunkSize,
                    overlap = overlap,
                    tokenizer = neuralEngine.tokenizer
                )
                val nextSize = Math.addExact(processed.size, nextChunks.size)
                require(nextSize <= MAX_STREAMED_DOCUMENT_CHUNKS) {
                    "JSONL document exceeds the safe limit of " +
                        "$MAX_STREAMED_DOCUMENT_CHUNKS processed records."
                }
                processed.addAll(nextChunks)
            }
        }

        return processed.takeIf { foundRecord }
    }

    /**
     * Rehydrates a bounded UTF-16BE spool in fixed-size blocks. Reading code units rather
     * than using a charset decoder preserves even isolated surrogate code units exactly.
     */
    private fun readUtf16BeStagingFile(
        file: File,
        expectedCharacterCount: Int
    ): String {
        require(expectedCharacterCount in 1..MAX_STREAMED_DOCUMENT_CHARACTERS)
        val result = StringBuilder(expectedCharacterCount)
        val byteBuffer = ByteArray(8 * 1024)
        val charBuffer = CharArray(byteBuffer.size / 2)
        var charactersRead = 0
        var pendingHighByte = -1

        BufferedInputStream(FileInputStream(file)).use { input ->
            while (true) {
                val byteCount = input.read(byteBuffer)
                if (byteCount < 0) break
                if (byteCount == 0) continue

                var inputIndex = 0
                var charCount = 0
                if (pendingHighByte >= 0) {
                    val lowByte = byteBuffer[inputIndex].toInt() and 0xFF
                    charBuffer[charCount++] = ((pendingHighByte shl 8) or lowByte).toChar()
                    pendingHighByte = -1
                    inputIndex++
                }

                while (inputIndex + 1 < byteCount) {
                    val high = byteBuffer[inputIndex].toInt() and 0xFF
                    val low = byteBuffer[inputIndex + 1].toInt() and 0xFF
                    charBuffer[charCount++] = ((high shl 8) or low).toChar()
                    inputIndex += 2
                }

                if (inputIndex < byteCount) {
                    pendingHighByte = byteBuffer[inputIndex].toInt() and 0xFF
                }
                if (charCount > 0) {
                    result.append(charBuffer, 0, charCount)
                    charactersRead = Math.addExact(charactersRead, charCount)
                }
            }
        }

        check(pendingHighByte < 0) {
            "Temporary streamed-ingestion data ended in the middle of a UTF-16 code unit."
        }
        check(charactersRead == expectedCharacterCount) {
            "Temporary streamed-ingestion character count does not match the source stream."
        }
        return result.toString()
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
        rightsStatus: String = "UNVERIFIED",
        sourceProvenance: CorpusSourceProvenance = CorpusSourceProvenance()
    ): Boolean {
        require(rightsStatus in setOf("UNVERIFIED", "USER_DECLARED", "LICENSE_REVIEWED")) {
            "Unsupported corpus rights status: $rightsStatus"
        }
        sourceProvenance.validateFor(rightsStatus)
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
        return activeRepository.storeCorpusDocument(document, emptyList(), sourceProvenance)
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
        val replaySessionId = UUID.randomUUID().toString()
        val replaySession = TrainingReplaySessionEntity(
            id = replaySessionId,
            sourceSnapshotFingerprint = CorpusReplayScheduler.sourceSnapshotFingerprint(sources),
            seed = seed,
            policy = policy.name,
            epochs = epochs,
            schedulerFormatVersion = CorpusReplayScheduler.SCHEDULER_FORMAT_VERSION,
            scheduledDocumentCount = requestedDocumentIds.size,
            scheduledChunkCount = schedule.size
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
                documentId = scheduled.document.id,
                replaySessionId = replaySessionId
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

        val orderedItems = synchronized(queueLock) {
            if (closed) {
                emptyList()
            } else {
                appendPersistingQueueItems(items, replaySession).also { queued ->
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

        if (neuralEngine.tokenizer.encodeToIds(content, autoRegister = false).size > 380) {
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

    private fun requireJournalCompatibility(
        snapshot: TrainingCommitJournalSnapshot,
        recoveryPhase: String
    ) {
        val activeModelState = neuralEngine.snapshotModelState()
        if (snapshot.modelIdentityFingerprint == null) {
            Log.w(
                TAG,
                "Legacy training journal has no model identity fingerprint; preserving its prior optimizer-step recovery path."
            )
        } else {
            check(
                TrainingCommitJournal.matchesModelIdentity(
                    snapshot,
                    activeModelState.identity.fingerprint
                )
            ) {
                "Training journal model identity mismatch during $recoveryPhase. " +
                    "The checkpoint was preserved and queue recovery was stopped."
            }
        }
        if (snapshot.optimizerConfigFingerprint == null) {
            Log.w(
                TAG,
                "Legacy training journal has no optimizer configuration fingerprint; compatibility is unverified."
            )
        } else {
            check(
                TrainingCommitJournal.matchesOptimizerConfiguration(
                    snapshot,
                    activeModelState.optimizerConfiguration.fingerprint
                )
            ) {
                "Training journal optimizer configuration mismatch during $recoveryPhase. " +
                    "The checkpoint was preserved and queue recovery was stopped."
            }
            check(
                TrainingCommitJournal.allowsCheckpointForRecovery(
                    snapshot = snapshot,
                    checkpointFormat = activeModelState.checkpointFormat,
                    checkpointVerified = activeModelState.checkpointVerified,
                    checkpointOptimizerConfigFingerprint =
                        activeModelState.checkpointOptimizerConfigurationFingerprint
                )
            ) {
                "Training journal optimizer configuration cannot be reconciled with the verified checkpoint during $recoveryPhase. " +
                    "The checkpoint was preserved and queue recovery was stopped."
            }
            if (activeModelState.checkpointFormat == ModelCheckpointFormat.ARG5) {
                Log.w(
                    TAG,
                    "Journal v3 is being recovered from a verified ARG5 checkpoint without embedded optimizer identity; checkpoint optimizer compatibility remains unverified."
                )
            }
        }
        val expectedCheckpointFingerprint = snapshot.checkpointSha256Fingerprint
        if (expectedCheckpointFingerprint != null) {
            val actualCheckpointFingerprint = neuralEngine.checkpointFileSha256Fingerprint()
            check(
                TrainingCommitJournal.matchesCheckpointFileFingerprint(
                    snapshot,
                    actualCheckpointFingerprint
                )
            ) {
                "Training journal checkpoint-file SHA-256 mismatch during $recoveryPhase. " +
                    "The checkpoint was preserved and queue recovery was stopped."
            }
        }
    }

    private suspend fun reconcilePersistedTrainingCommit(
        repository: com.argus.agi.data.repository.ArgusRepository
    ) {
        val snapshot = TrainingCommitJournal.read(trainingCommitJournalFile) ?: return
        val currentOptimizerStep = neuralEngine.optimizerStep

        if (!snapshot.isCommitted) {
            // A CPU update can finish mutating host tensors immediately before an OOM
            // in its timing/reconciliation tail, while adamStep is still at baseline.
            // Always restore the full checkpoint; optimizerStep equality alone does not
            // prove that every parameter and AdamW moment is unchanged.
            check(neuralEngine.releaseVulkanTensorResidencyForMemoryPressure()) {
                "Vulkan training state could not be released before PREPARED-batch rollback."
            }
            check(
                neuralEngine.loadCheckpointIfExists() &&
                    neuralEngine.optimizerStep == snapshot.baselineOptimizerStep
            ) {
                "The prepared training batch could not be rolled back to its exact optimizer baseline."
            }
            requireJournalCompatibility(snapshot, "PREPARED-batch rollback")
            neuralEngine.resumeVulkanTensorResidencyAfterForeground()
            TrainingCommitJournal.delete(trainingCommitJournalFile)
            activeBatchBaselineOptimizerStep = null
            activeBatchRoomCommitted = false
            return
        }

        val committedOptimizerStep = requireNotNull(snapshot.committedOptimizerStep)
        if (!TrainingCommitJournal.isExactCommittedCheckpoint(snapshot, currentOptimizerStep)) {
            check(TrainingCommitJournal.isExactBaselineCheckpoint(snapshot, currentOptimizerStep)) {
                "The model checkpoint optimizer step $currentOptimizerStep does not match " +
                    "the journal baseline ${snapshot.baselineOptimizerStep} or committed " +
                    "step $committedOptimizerStep."
            }
            val committedIds = repository.getOptimizerCommittedTrainingItemsSnapshot()
                .mapTo(HashSet()) { it.id }
            check(snapshot.entries.none { it.itemId in committedIds }) {
                "Room marks a batch committed, but the restored checkpoint is only its baseline."
            }
            // COMMITTED reached disk before its model checkpoint. A restart therefore
            // restored the baseline; reload that complete state rather than trusting only
            // adamStep, then let Room's TRAINING rows return to QUEUED.
            check(neuralEngine.releaseVulkanTensorResidencyForMemoryPressure()) {
                "Vulkan training state could not be released before committed-journal rollback."
            }
            check(
                neuralEngine.loadCheckpointIfExists() &&
                    TrainingCommitJournal.isExactBaselineCheckpoint(
                        snapshot,
                        neuralEngine.optimizerStep
                    )
            ) {
                "The committed batch could not be rolled back to its exact saved baseline."
            }
            requireJournalCompatibility(snapshot, "COMMITTED-batch baseline rollback")
            neuralEngine.resumeVulkanTensorResidencyAfterForeground()
            TrainingCommitJournal.delete(trainingCommitJournalFile)
            activeBatchBaselineOptimizerStep = null
            activeBatchRoomCommitted = false
            return
        }

        requireJournalCompatibility(snapshot, "COMMITTED-batch recovery")

        // Room must never publish COMMITTED until the exact model and optimizer
        // state represented by this journal is durably saved. A checkpoint that is
        // ahead of the journal is an inconsistency, not a valid committed transaction.
        if (snapshot.checkpointSha256Fingerprint == null) {
            // Older committed journals are upgraded only when both semantic identities
            // exist; v1/v2 remain readable without manufacturing missing fingerprints.
            check(neuralEngine.saveCheckpoint()) {
                "The committed optimizer checkpoint could not be secured during recovery."
            }
            if (
                snapshot.modelIdentityFingerprint != null &&
                snapshot.optimizerConfigFingerprint != null
            ) {
                val checkpointFingerprint = neuralEngine.checkpointFileSha256Fingerprint()
                TrainingCommitJournal.writeCommitted(
                    file = trainingCommitJournalFile,
                    baselineOptimizerStep = snapshot.baselineOptimizerStep,
                    committedOptimizerStep = committedOptimizerStep,
                    entries = snapshot.entries,
                    modelIdentityFingerprint = snapshot.modelIdentityFingerprint,
                    optimizerConfigFingerprint = snapshot.optimizerConfigFingerprint,
                    checkpointSha256Fingerprint = checkpointFingerprint
                )
                val checkpointBoundJournal = requireNotNull(
                    TrainingCommitJournal.read(trainingCommitJournalFile)
                ) {
                    "The recovered checkpoint-bound journal could not be read back."
                }
                check(
                    checkpointBoundJournal.checkpointSha256Fingerprint == checkpointFingerprint
                ) {
                    "The recovered checkpoint file identity was not persisted in the journal."
                }
                requireJournalCompatibility(
                    checkpointBoundJournal,
                    "committed-batch recovery after checkpoint write"
                )
            }
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
        } else {
            // Every journal entry is already COMPLETED or FAILED. Do not let a stale
            // journal keep an accepted validation run looking interrupted forever.
            TrainingCommitJournal.delete(trainingCommitJournalFile)
            activeBatchBaselineOptimizerStep = null
            activeBatchRoomCommitted = false
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

    private suspend fun ensureLiveBatchCommitPersisted(
        activeItems: MutableList<TrainingData>
    ): Boolean {
        if (activeBatchRoomCommitted) return true
        val snapshot = TrainingCommitJournal.read(trainingCommitJournalFile) ?: return false
        if (!snapshot.isCommitted) return false

        if (!TrainingCommitJournal.isExactCommittedCheckpoint(snapshot, neuralEngine.optimizerStep)) {
            return false
        }
        requireJournalCompatibility(snapshot, "live committed-batch recovery")
        check(neuralEngine.saveCheckpoint()) {
            "The optimizer committed the batch, but its checkpoint could not be secured."
        }

        val repository = requireNotNull(repo) {
            "The committed batch cannot be finalized without durable queue storage."
        }
        val entriesById = snapshot.entries.associateBy(TrainingCommitJournalEntry::itemId)
        val committedItems = ArrayList<TrainingData>(activeItems.size)
        for (index in activeItems.indices) {
            val item = activeItems[index]
            val entry = entriesById[item.id] ?: continue
            val result = requireNotNull(entry.result) {
                "The committed training journal is missing a sample result."
            }
            val committed = item.copy(
                status = TrainingStatus.OPTIMIZER_COMMITTED,
                finalLoss = result.loss,
                stepDurationMs = entry.stepDurationMs
            )
            activeItems[index] = committed
            committedItems.add(committed)
        }
        if (committedItems.isNotEmpty()) {
            repository.markTrainingItemsOptimizerCommitted(committedItems)
        }
        activeBatchRoomCommitted = true
        return true
    }

    private suspend fun handleTrainingOutOfMemory(
        activeItems: MutableList<TrainingData>,
        error: OutOfMemoryError
    ) {
        Log.e(
            TAG,
            "Training batch hit memory pressure; recovering from its durable optimizer transaction.",
            error
        )

        val residencyReleased = runCatching {
            neuralEngine.releaseVulkanTensorResidencyForMemoryPressure()
        }.getOrDefault(false)
        if (!residencyReleased) {
            _isTrainingPaused.value = true
            Log.e(TAG, "Training paused because Vulkan state could not be safely released after OOM.")
            return
        }

        val committed = runCatching {
            ensureLiveBatchCommitPersisted(activeItems)
        }.onFailure { recoveryError ->
            Log.e(TAG, "Could not secure the committed optimizer transaction after OOM.", recoveryError)
        }.getOrDefault(false)

        if (committed) {
            runCatching {
                neuralEngine.resumeVulkanTensorResidencyAfterForeground()
                val repository = requireNotNull(repo)
                val activeIds = activeItems.mapTo(HashSet()) { it.id }
                val pendingCommitted = repository
                    .getOptimizerCommittedTrainingItemsSnapshot()
                    .filter { item -> item.id in activeIds }
                prependCommittedQueueItems(pendingCommitted)
            }.onFailure { recoveryError ->
                Log.e(TAG, "Committed training finalization could not be returned to the queue.", recoveryError)
            }
            _currentTrainingItem.value = null
            _activeDocument.value = null
            _isTrainingPaused.value = true
            Log.w(
                TAG,
                "The optimizer commit was preserved. Training is paused so the remaining committed items can be finalized without retraining."
            )
            return
        }

        val journal = runCatching {
            TrainingCommitJournal.read(trainingCommitJournalFile)
        }.onFailure { journalError ->
            Log.e(TAG, "Training commit journal could not be read during OOM recovery.", journalError)
        }.getOrNull()
        val baselineStep = activeBatchBaselineOptimizerStep
            ?: journal?.baselineOptimizerStep

        val repository = repo
        val committedIdsInRoom = if (repository != null) {
            runCatching {
                repository.getOptimizerCommittedTrainingItemsSnapshot()
                    .mapTo(HashSet()) { it.id }
            }.getOrElse { roomError ->
                Log.e(TAG, "Could not verify Room commit state during OOM recovery.", roomError)
                _isTrainingPaused.value = true
                return
            }
        } else {
            emptySet()
        }
        if (activeItems.any { it.id in committedIdsInRoom }) {
            _isTrainingPaused.value = true
            Log.e(
                TAG,
                "Room already contains optimizer-committed items, but the corresponding model checkpoint could not be verified. The queue was left untouched."
            )
            return
        }

        val rolledBack = if (baselineStep == null) {
            // OOM happened before the prepared journal was created and before the
            // optimizer was invoked. No model transaction needs to be replayed.
            true
        } else {
            runCatching {
                val restored = neuralEngine.loadCheckpointIfExists() &&
                    neuralEngine.optimizerStep == baselineStep
                if (restored && journal != null) {
                    requireJournalCompatibility(journal, "OOM batch rollback")
                }
                restored
            }.getOrDefault(false)
        }
        if (!rolledBack) {
            _isTrainingPaused.value = true
            Log.e(
                TAG,
                "Training paused because the model could not be restored to its pre-batch optimizer checkpoint."
            )
            return
        }

        runCatching {
            TrainingCommitJournal.delete(trainingCommitJournalFile)
            neuralEngine.resumeVulkanTensorResidencyAfterForeground()
        }.onFailure { recoveryError ->
            _isTrainingPaused.value = true
            Log.e(TAG, "Training rollback completed but Vulkan/journal recovery did not finish.", recoveryError)
            return
        }

        activeBatchBaselineOptimizerStep = null
        activeBatchRoomCommitted = false
        val requeued = prependPersistingQueueItems(
            activeItems.map { item -> item.copy(status = TrainingStatus.QUEUED) }
        )
        _currentTrainingItem.value = null
        _activeDocument.value = null

        if (requeued.isEmpty()) {
            _isTrainingPaused.value = true
            Log.e(TAG, "OOM recovery could not persist the batch at the front of the queue.")
            return
        }

        val shouldPauseForRepeatedOom = synchronized(queueLock) {
            requeued.any { item ->
                val retryCount = (oomRetryCounts[item.id] ?: 0) + 1
                oomRetryCounts[item.id] = retryCount
                oomRecoveryItemIds.add(item.id)
                retryCount > 1
            }
        }
        if (shouldPauseForRepeatedOom) {
            _isTrainingPaused.value = true
            Log.e(TAG, "The same training item hit OOM more than once. It was preserved for explicit user retry.")
        } else {
            Log.w(
                TAG,
                "The uncommitted batch was rolled back and moved to the front of the queue. Its retry is constrained to batch 1 and reduced context."
            )
        }
    }

    private suspend fun handleCommittedFinalizationOutOfMemory(
        item: TrainingData,
        error: OutOfMemoryError
    ) {
        Log.e(TAG, "Finalizing an already committed training item hit OOM; the item will not be retrained.", error)
        val released = runCatching {
            neuralEngine.releaseVulkanTensorResidencyForMemoryPressure()
        }.getOrDefault(false)
        if (!released) {
            _isTrainingPaused.value = true
            return
        }
        runCatching {
            neuralEngine.resumeVulkanTensorResidencyAfterForeground()
            val repository = requireNotNull(repo)
            val pending = repository.getOptimizerCommittedTrainingItemsSnapshot()
                .filter { committed -> committed.id == item.id }
            prependCommittedQueueItems(pending)
        }.onFailure { recoveryError ->
            Log.e(TAG, "Committed item could not be returned to finalization queue.", recoveryError)
        }
        _currentTrainingItem.value = null
        _activeDocument.value = null
        _isTrainingPaused.value = true
    }

    private suspend fun processCommittedItem(item: TrainingData): Boolean {
        val journalEntry = runCatching {
            val snapshot = TrainingCommitJournal.read(trainingCommitJournalFile)
            if (snapshot?.isCommitted == true) {
                requireJournalCompatibility(snapshot, "committed-item finalization")
            }
            snapshot
                ?.takeIf(TrainingCommitJournalSnapshot::isCommitted)
                ?.entries
                ?.firstOrNull { entry -> entry.itemId == item.id }
        }.getOrElse { error ->
            Log.e(TAG, "Could not read committed result for '${item.id}'.", error)
            throw error
        }
        val result = journalEntry?.result ?: NeuralTrainingStepResult(
            loss = item.finalLoss ?: Float.NaN,
            inputTokenCount = 0,
            predictionTargetCount = 0,
            contextWindowCount = 0,
            optimizerUpdateCount = 0,
            vocabularySize = neuralEngine.tokenizer.vocabSize
        )
        val duration = journalEntry?.stepDurationMs ?: item.stepDurationMs
        val completed = processItem(item, result, duration)
        deleteTrainingCommitJournalWhenResolved()
        synchronized(queueLock) {
            oomRecoveryItemIds.remove(item.id)
            oomRetryCounts.remove(item.id)
        }
        return completed
    }

    private suspend fun deleteTrainingCommitJournalWhenResolved() {
        val snapshot = TrainingCommitJournal.read(trainingCommitJournalFile) ?: return
        if (!snapshot.isCommitted) return
        val repository = repo ?: return
        val pendingIds = repository.getOptimizerCommittedTrainingItemsSnapshot()
            .mapTo(HashSet()) { it.id }
        if (snapshot.entries.none { entry -> entry.itemId in pendingIds }) {
            TrainingCommitJournal.delete(trainingCommitJournalFile)
            activeBatchBaselineOptimizerStep = null
            activeBatchRoomCommitted = false
        }
    }

    private fun startTrainingWorker() {
        workerJob = scope.launch {
            var validationRunActive = false
            var validationRunHadFailures = false
            try {
                // Finish in-memory queue writes before examining persisted state.
                awaitQueuePersistenceIdle()
                val repository = requireNotNull(repo) {
                    "Training cannot start without durable local queue storage."
                }
                val persistedCommitted = repository.getOptimizerCommittedTrainingItemsSnapshot()
                val interruptedCandidateRun =
                    conversationalValidationPersistence.hasInterruptedCandidateRun()
                val preserveCandidateCheckpoint = interruptedCandidateRun ||
                    trainingCommitJournalFile.isFile ||
                    persistedCommitted.isNotEmpty()

                // An interrupted candidate must keep the exact model checkpoint that
                // corresponds to its journal. Restoring the older accepted model here
                // would invalidate Room's committed rows without replaying their work.
                if (
                    !preserveCandidateCheckpoint &&
                    conversationalValidationPersistence.hasAcceptedCheckpoint()
                ) {
                    check(restoreAcceptedConversationCheckpoint()) {
                        "The last accepted conversational checkpoint could not be restored."
                    }
                }

                run {
                    reconcilePersistedTrainingCommit(repository)
                    repository.recoverInterruptedTrainingItems()
                    val committed = repository.getOptimizerCommittedTrainingItemsSnapshot()
                    val pending = repository.getPendingQueueSnapshot()
                    val databaseIds = (committed + pending).mapTo(HashSet()) { it.id }
                    val inMemoryItems = synchronized(queueLock) {
                        _nextQueue.value.filterNot { item -> item.id in databaseIds }
                    }
                    val restoredItems = committed.sortedBy { it.timestamp } +
                        pending.sortedBy { it.timestamp } +
                        inMemoryItems.map { item -> item.copy(status = TrainingStatus.QUEUED) }
                    synchronized(queueLock) {
                        _nextQueue.value = restoredItems.distinctBy(TrainingData::id)
                        restoredItems.maxOfOrNull { it.timestamp }?.let { latest ->
                            lastScheduledTimestamp = maxOf(lastScheduledTimestamp, latest)
                        }
                    }
                }
                val hasRestoredCommittedItems = repository
                    .getOptimizerCommittedTrainingItemsSnapshot()
                    .isNotEmpty()
                validationRunActive = interruptedCandidateRun || hasRestoredCommittedItems
                if (
                    validationRunActive &&
                    conversationalValidationPersistence.loadState().baselineReport == null
                ) {
                    throw IllegalStateException(
                        "An interrupted candidate exists without a durable conversational baseline."
                    )
                }
                if (validationRunActive) {
                    setConversationalValidationStatus(
                        ConversationValidationStatus.TRAINING,
                        "Resuming the interrupted local candidate from its last durable optimizer transaction."
                    )
                }
                wakeupSignal.trySend(Unit)
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Throwable) {
                Log.e(TAG, "Training startup recovery failed.", error)
                _isTrainingPaused.value = true
                setConversationalValidationStatus(
                    ConversationValidationStatus.FAILED,
                    "Training is paused because startup checkpoint/queue recovery failed: ${error.message}"
                )
                return@launch
            }

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
                if (
                    !validationRunActive &&
                    firstQueuedItem?.status == TrainingStatus.QUEUED
                ) {
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

                _currentTrainingItem.value = firstItem
                updateActiveDocumentEntering(firstItem)
                try {
                    if (firstItem.status == TrainingStatus.OPTIMIZER_COMMITTED) {
                        try {
                            if (!processCommittedItem(firstItem)) {
                                validationRunHadFailures = true
                            }
                        } catch (cancelled: CancellationException) {
                            throw cancelled
                        } catch (outOfMemory: OutOfMemoryError) {
                            handleCommittedFinalizationOutOfMemory(firstItem, outOfMemory)
                        } catch (error: Throwable) {
                            Log.e(
                                TAG,
                                "Committed training item '${firstItem.id}' could not be finalized.",
                                error
                            )
                            prependCommittedQueueItems(listOf(firstItem))
                            _isTrainingPaused.value = true
                        }
                    } else {
                        check(firstItem.status == TrainingStatus.QUEUED) {
                            "The training worker received an unsupported queue status: ${firstItem.status}."
                        }

                        val memoryRecovery = synchronized(queueLock) {
                            firstItem.id in oomRecoveryItemIds
                        }
                        val safetySnapshot = deviceGuard.safetyState.value
                        val memorySnapshot = captureTrainingMemorySnapshot()
                        val runtimeConfiguration = RuntimeConfigurationPlanner.create(
                            backend = neuralEngine.runtimeComputeBackend,
                            resources = RuntimeResourceSnapshot(
                                availableAppHeapBytes = memorySnapshot.availableHeapBytes,
                                maximumAppHeapBytes = memorySnapshot.maximumHeapBytes,
                                availableSystemMemoryBytes = memorySnapshot.availableSystemMemoryBytes,
                                systemLowMemory = memorySnapshot.isSystemLowMemory,
                                thermalStatus = safetySnapshot.thermalStatus,
                                temperatureCelsius = safetySnapshot.temperatureCelsius,
                                isPowerSaveMode = safetySnapshot.isPowerSaveMode,
                                isCharging = safetySnapshot.isCharging,
                                batteryPct = safetySnapshot.batteryPct,
                                memoryRecovery = memoryRecovery
                            ),
                            model = RuntimeModelConstraints(
                                configuredBatchSize = neuralEngine.config.trainingBatchSize,
                                configuredContextLength = neuralEngine.config.maxSeqLen,
                                dModel = neuralEngine.config.dModel,
                                feedForwardSize = neuralEngine.config.dFf,
                                vocabularySize = neuralEngine.tokenizer.vocabSize,
                                layerCount = neuralEngine.config.layerCount,
                                headCount = neuralEngine.config.nHeads
                            )
                        )

                        if (runtimeConfiguration.deferReason == RuntimeDeferReason.THERMAL_LIMIT) {
                            prependPersistingQueueItems(listOf(firstItem))
                            continue
                        }
                        if (!runtimeConfiguration.canTrain) {
                            // Even the minimum legal context exceeds current live
                            // workspace headroom. Return the claim to durable QUEUED
                            // state without touching model/checkpoint/optimizer state.
                            prependPersistingQueueItems(listOf(firstItem))
                            _currentTrainingItem.value = null
                            _activeDocument.value = null
                            Log.i(
                                TAG,
                                "Training deferred: minimum context does not fit current memory headroom."
                            )
                            delay(IDLE_POLL_MS)
                            continue
                        }

                        val batchItems = ArrayList<TrainingData>(
                            runtimeConfiguration.effectiveTrainingBatchSize
                        )
                        batchItems += firstItem
                        while (batchItems.size < runtimeConfiguration.effectiveTrainingBatchSize) {
                            val nextItem = popNext(expectedStatus = TrainingStatus.QUEUED) ?: break
                            batchItems += nextItem
                        }
                        val activeItems = batchItems
                            .map { item -> item.copy(status = TrainingStatus.TRAINING) }
                            .toMutableList()
                        _currentTrainingItem.value = activeItems.first()
                        updateActiveDocumentEntering(activeItems.first())

                        try {
                            if (!processBatch(activeItems, runtimeConfiguration.effectiveContextLength)) {
                                validationRunHadFailures = true
                            } else {
                                synchronized(queueLock) {
                                    activeItems.forEach { item ->
                                        oomRecoveryItemIds.remove(item.id)
                                        oomRetryCounts.remove(item.id)
                                    }
                                }
                            }
                        } catch (cancelled: CancellationException) {
                            throw cancelled
                        } catch (outOfMemory: OutOfMemoryError) {
                            handleTrainingOutOfMemory(activeItems, outOfMemory)
                        } catch (error: Throwable) {
                            Log.e(
                                TAG,
                                "Training batch failed (${activeItems.size} samples).",
                                error
                            )
                            val committed = runCatching {
                                ensureLiveBatchCommitPersisted(activeItems)
                            }.onFailure { recoveryError ->
                                Log.e(TAG, "Could not reconcile optimizer state after a training failure.", recoveryError)
                            }.getOrDefault(false)

                            if (committed || activeBatchRoomCommitted) {
                                val repository = repo
                                val pendingCommitted = repository.getOptimizerCommittedTrainingItemsSnapshot()
                                    .filter { pending -> activeItems.any { it.id == pending.id } }
                                    .orEmpty()
                                prependCommittedQueueItems(pendingCommitted)
                                _isTrainingPaused.value = true
                            } else {
                                val journal = runCatching {
                                    TrainingCommitJournal.read(trainingCommitJournalFile)
                                }.getOrNull()
                                if (journal != null) {
                                    val released = runCatching {
                                        neuralEngine.releaseVulkanTensorResidencyForMemoryPressure()
                                    }.getOrDefault(false)
                                    val baseline = activeBatchBaselineOptimizerStep
                                        ?: journal.baselineOptimizerStep
                                    val rolledBack = released && runCatching {
                                        val restored = neuralEngine.loadCheckpointIfExists() &&
                                            neuralEngine.optimizerStep == baseline
                                        if (restored) {
                                            requireJournalCompatibility(journal, "failed-batch rollback")
                                        }
                                        restored
                                    }.getOrDefault(false)
                                    if (!rolledBack) {
                                        _isTrainingPaused.value = true
                                        Log.e(TAG, "Training paused because the failed batch could not be rolled back to its baseline.")
                                    } else {
                                        TrainingCommitJournal.delete(trainingCommitJournalFile)
                                        neuralEngine.resumeVulkanTensorResidencyAfterForeground()
                                        activeBatchBaselineOptimizerStep = null
                                        activeBatchRoomCommitted = false
                                        activeItems.forEach { item ->
                                            markTrainingItemFailed(item, error.message)
                                        }
                                        activeItems.mapNotNull(TrainingData::documentId)
                                            .forEach(validationRunFailedDocumentIds::add)
                                    }
                                } else {
                                    activeItems.forEach { item ->
                                        markTrainingItemFailed(item, error.message)
                                    }
                                    activeItems.mapNotNull(TrainingData::documentId)
                                        .forEach(validationRunFailedDocumentIds::add)
                                }
                                validationRunHadFailures = true
                            }
                        }
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

    private fun appendPersistingQueueItems(
        items: List<TrainingData>,
        replaySession: TrainingReplaySessionEntity? = null
    ): List<TrainingData> {
        if (items.isEmpty()) return emptyList()

        val orderedItems = assignPersistentQueueOrder(items)
        val persistingItems = orderedItems.map { it.copy(status = TrainingStatus.PERSISTING) }
        _nextQueue.value = _nextQueue.value + persistingItems

        val operation = QueuePersistenceOperation.Save(
            generation = queueGeneration,
            items = orderedItems,
            replaySession = replaySession
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
                    val replaySession = operation.replaySession
                    if (replaySession == null) {
                        repository.saveTrainingItems(operation.items)
                    } else {
                        repository.saveCorpusReplaySession(replaySession, operation.items)
                    }
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
    internal suspend fun popNext(
        expectedStatus: TrainingStatus? = null
    ): TrainingData? {
        val startCompletion = CompletableDeferred<Result<Unit>>()
        val reservation = synchronized(queueLock) {
            val list = _nextQueue.value
            val first = list.firstOrNull() ?: return@synchronized null
            if (expectedStatus != null && first.status != expectedStatus) {
                return@synchronized null
            }

            if (first.status == TrainingStatus.OPTIMIZER_COMMITTED) {
                _nextQueue.value = list.drop(1)
                return@synchronized Triple(first, null, queueGeneration)
            }
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
            Triple(first, startCompletion, queueGeneration)
        } ?: return null

        val item = reservation.first
        val completion = reservation.second ?: return item
        val reservationGeneration = reservation.third

        // A committed item is already durable and must go directly to finalization.
        // A queued item is not allowed to train until Room confirms TRAINING.
        val persistenceResult = completion.await()
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
        } catch (error: OutOfMemoryError) {
            preparedTokenizerDocuments.remove(documentId)
            throw error
        } catch (error: Exception) {
            preparedTokenizerDocuments.remove(documentId)
            Log.w(TAG, "Tokenizer preparation failed for document $documentId; existing vocabulary will be used.", error)
        }
    }

    private suspend fun processBatch(
        items: MutableList<TrainingData>,
        runtimeContextLength: Int
    ): Boolean {
        require(items.isNotEmpty()) { "A training micro-batch cannot be empty." }
        val repository = requireNotNull(repo) {
            "Durable training storage is required before optimizer updates can begin."
        }
        items.mapNotNull(TrainingData::documentId)
            .forEach(validationRunDocumentIds::add)

        for (item in items) {
            item.documentId?.let { documentId ->
                if (item.chunkIndex == 1) {
                    try {
                        repository.markCorpusDocumentTraining(documentId)
                    } catch (cancelled: CancellationException) {
                        throw cancelled
                    } catch (error: OutOfMemoryError) {
                        throw error
                    } catch (error: Throwable) {
                        Log.w(TAG, "Failed to mark corpus document as training", error)
                    }
                }
                prepareTokenizerForDocument(item)
            }
        }

        // Save the exact model/tokenizer/optimizer baseline before preparing the
        // transaction journal. Managed training disables engine auto-checkpointing,
        // so this file remains the rollback point until the journal and Room agree.
        check(neuralEngine.saveCheckpoint()) {
            "The pre-batch model checkpoint could not be secured."
        }
        val baselineModelState = neuralEngine.snapshotModelState()
        val baselineModelIdentityFingerprint = baselineModelState.identity.fingerprint
        val baselineOptimizerConfigFingerprint = baselineModelState.optimizerConfiguration.fingerprint
        val baselineOptimizerStep = baselineModelState.optimizerStep
        activeBatchBaselineOptimizerStep = baselineOptimizerStep
        activeBatchRoomCommitted = false
        TrainingCommitJournal.writePrepared(
            file = trainingCommitJournalFile,
            baselineOptimizerStep = baselineOptimizerStep,
            itemIds = items.map(TrainingData::id),
            modelIdentityFingerprint = baselineModelIdentityFingerprint,
            optimizerConfigFingerprint = baselineOptimizerConfigFingerprint
        )

        val startNs = System.nanoTime()
        val batchResult = neuralEngine.trainBatchDetailed(
            texts = items.map(TrainingData::content),
            runtimeContextLength = runtimeContextLength,
            autoCheckpoint = false
        )
        val elapsedMs = (System.nanoTime() - startNs) / 1_000_000L
        require(batchResult.samples.size == items.size) {
            "Neural engine returned ${batchResult.samples.size} results for ${items.size} samples."
        }
        val perItemDurationMs = (elapsedMs / items.size).coerceAtLeast(1L)

        val commitEntries = items.zip(batchResult.samples).map { (item, result) ->
            TrainingCommitJournalEntry(
                itemId = item.id,
                result = result,
                stepDurationMs = perItemDurationMs
            )
        }
        TrainingCommitJournal.writeCommitted(
            file = trainingCommitJournalFile,
            baselineOptimizerStep = baselineOptimizerStep,
            committedOptimizerStep = neuralEngine.optimizerStep,
            entries = commitEntries,
            modelIdentityFingerprint = baselineModelIdentityFingerprint,
            optimizerConfigFingerprint = baselineOptimizerConfigFingerprint
        )
        val committedJournal = requireNotNull(
            TrainingCommitJournal.read(trainingCommitJournalFile)
        ) {
            "The committed training journal could not be read back."
        }
        requireJournalCompatibility(committedJournal, "live batch commit before checkpoint write")

        // The write-ahead COMMITTED marker is durable before the model checkpoint.
        // Room is updated only after that checkpoint is atomically secured.
        check(neuralEngine.saveCheckpoint()) {
            "The committed optimizer state could not be checkpointed."
        }
        val committedCheckpointFingerprint = neuralEngine.checkpointFileSha256Fingerprint()
        TrainingCommitJournal.writeCommitted(
            file = trainingCommitJournalFile,
            baselineOptimizerStep = baselineOptimizerStep,
            committedOptimizerStep = neuralEngine.optimizerStep,
            entries = commitEntries,
            modelIdentityFingerprint = baselineModelIdentityFingerprint,
            optimizerConfigFingerprint = baselineOptimizerConfigFingerprint,
            checkpointSha256Fingerprint = committedCheckpointFingerprint
        )
        val checkpointBoundJournal = requireNotNull(
            TrainingCommitJournal.read(trainingCommitJournalFile)
        ) {
            "The checkpoint-bound training journal could not be read back."
        }
        check(
            checkpointBoundJournal.checkpointSha256Fingerprint == committedCheckpointFingerprint
        ) {
            "The committed checkpoint SHA-256 was not persisted in the journal."
        }
        requireJournalCompatibility(
            checkpointBoundJournal,
            "live batch commit after checkpoint write"
        )

        val committedItems = items.zip(commitEntries).map { (item, entry) ->
            val result = requireNotNull(entry.result)
            item.copy(
                status = TrainingStatus.OPTIMIZER_COMMITTED,
                finalLoss = result.loss,
                stepDurationMs = entry.stepDurationMs
            )
        }
        repository.markTrainingItemsOptimizerCommitted(committedItems)
        committedItems.forEachIndexed { index, item -> items[index] = item }
        activeBatchRoomCommitted = true

        var allItemsSucceeded = true
        for ((item, entry) in items.zip(commitEntries)) {
            val result = requireNotNull(entry.result)
            if (!processItem(item, result, entry.stepDurationMs)) {
                allItemsSucceeded = false
                item.documentId?.let(validationRunFailedDocumentIds::add)
            }
            synchronized(queueLock) {
                oomRecoveryItemIds.remove(item.id)
                oomRetryCounts.remove(item.id)
            }
        }

        deleteTrainingCommitJournalWhenResolved()
        if (!trainingCommitJournalFile.exists()) {
            activeBatchBaselineOptimizerStep = null
            activeBatchRoomCommitted = false
        }
        return allItemsSucceeded
    }

    private suspend fun markTrainingItemFailed(item: TrainingData, reason: String?) {
        val repository = requireNotNull(repo) {
            "A failed training item cannot be finalized without durable queue storage."
        }
        val existing = repository.getTrainingItem(item.id)
        if (existing?.status == TrainingStatus.COMPLETED ||
            existing?.status == TrainingStatus.FAILED) {
            return
        }

        val failed = item.copy(status = TrainingStatus.FAILED)
        repository.markTrainingFailed(failed.id)
        failed.documentId?.let { documentId ->
            repository.markCorpusDocumentPartial(documentId)
        }
        pushCompleted(failed)
        if (!reason.isNullOrBlank()) {
            Log.w(TAG, "Training item '${item.title}' failed: $reason")
        }
    }

    private suspend fun processItem(
        item: TrainingData,
        stepResult: NeuralTrainingStepResult,
        durationMs: Long
    ): Boolean {
        val repository = requireNotNull(repo) {
            "Training finalization requires durable local storage."
        }
        val persistedItem = repository.getTrainingItem(item.id)
            ?: throw IllegalStateException(
                "Training item '${item.id}' disappeared before finalization."
            )
        when (persistedItem.status) {
            TrainingStatus.COMPLETED -> return true
            TrainingStatus.FAILED -> return false
            TrainingStatus.OPTIMIZER_COMMITTED -> Unit
            else -> throw IllegalStateException(
                "Training item '${item.id}' cannot be finalized from status ${persistedItem.status}."
            )
        }

        updateActiveDocumentEntering(item)
        val stepLoss = stepResult.loss
        // Invalid loss is a failed step, never a completed training item.
        if (!stepLoss.isFinite()) {
            Log.w(TAG, "trainBatchDetailed returned non-finite loss=$stepLoss for '${item.title}'")
            markTrainingItemFailed(item, "The model produced a non-finite loss.")
            return false
        }

        // Perform allocation-heavy finalization before the Room compare-and-set. If OOM
        // occurs here, the durable row remains OPTIMIZER_COMMITTED and will be finalized
        // again without another optimizer update.
        val reasoningSpans = REASONING_BLOCK_PATTERN.findAll(item.content)
            .map { it.groupValues[1] }
            .toList()
        val reasoningTokenCount = reasoningSpans.sumOf { reasoningText ->
            neuralEngine.tokenizer.encodeToIds(reasoningText, autoRegister = false).size.toLong()
        }
        val finished = item.copy(
            status = TrainingStatus.COMPLETED,
            finalLoss = stepLoss,
            stepDurationMs = durationMs
        )

        val completionClaimed = repository.markTrainingCompleted(
            finished.id,
            stepLoss,
            durationMs
        )
        if (!completionClaimed) {
            val currentItem = repository.getTrainingItem(finished.id)
                ?: throw IllegalStateException(
                    "Training item '${finished.id}' disappeared during completion."
                )
            check(currentItem.status == TrainingStatus.COMPLETED) {
                "Training item '${finished.id}' left OPTIMIZER_COMMITTED without a valid completion claim."
            }
            return true
        }

        // The Room status is durable before transient UI and telemetry updates. If one of
        // those updates hits OOM, recovery sees COMPLETED and will not reapply metrics or
        // re-run the model on this sample.
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
        if (
            finished.documentId != null &&
            finished.chunkIndex >= finished.totalChunks
        ) {
            try {
                val document = repository.getCorpusDocument(finished.documentId)
                if (document != null && document.chunkCount > 0) {
                    val completedPasses = finished.totalChunks / document.chunkCount
                    repository.markCorpusDocumentTrained(
                        documentId = document.id,
                        completedPassCount = completedPasses
                    )
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: OutOfMemoryError) {
                throw error
            } catch (error: Throwable) {
                Log.w(TAG, "Failed to update completed corpus document state", error)
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
