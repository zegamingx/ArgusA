package com.argus.agi.model

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.argus.agi.data.local.ArgusDatabase
import com.argus.agi.data.repository.ArgusRepository
import com.argus.agi.sensor.DeviceGuard
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import java.security.MessageDigest
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class TrainingManagerPersistenceTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun screenTrainingRequiresConsentAndCanBeClearedWithoutCreatingReplayCorpus() = runBlocking {
        withTrainingManager { database, repository, manager ->
            manager.pauseTraining()
            val privacyPreferences = context.getSharedPreferences(
                "argus_excluded_apps",
                Context.MODE_PRIVATE
            )
            privacyPreferences.edit()
                .putBoolean("screen_perception_enabled", true)
                .putBoolean("screen_training_enabled", false)
                .commit()

            manager.enqueueScreenData(
                packageName = "com.example.reader",
                title = "Screen capture",
                content = "A sufficiently long screen sample that must never enter the document replay corpus."
            )
            awaitPersistence(manager)
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())
            assertEquals(0, repository.getCorpusDocumentCount())

            privacyPreferences.edit()
                .putBoolean("screen_training_enabled", true)
                .commit()
            manager.enqueueScreenData(
                packageName = "com.example.reader",
                title = "Screen capture",
                content = "A screen sample without any per-app Read or Train grant."
            )
            awaitPersistence(manager)
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())

            privacyPreferences.edit()
                .putStringSet("allowed_screen_packages", setOf("com.example.reader"))
                .commit()
            manager.enqueueScreenData(
                packageName = "com.example.reader",
                title = "Screen capture",
                content = "A screen sample with Read permission but no Train permission."
            )
            awaitPersistence(manager)
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())

            privacyPreferences.edit()
                .putStringSet("allowed_screen_training_packages", setOf("com.example.reader"))
                .commit()
            manager.enqueueScreenData(
                packageName = "com.example.reader",
                title = "Screen capture",
                content = "A sufficiently long screen sample that requires separate training consent."
            )
            awaitPersistence(manager)

            val queuedItems = repository.getPendingQueueSnapshot()
            assertEquals(1, queuedItems.size)
            assertTrue(queuedItems.single().source.startsWith("ScreenTrainingV2 (com.example.reader)"))
            assertEquals(0, repository.getCorpusDocumentCount())

            privacyPreferences.edit()
                .putBoolean("screen_training_enabled", false)
                .commit()
            val deletion = manager.clearScreenDerivedData()

            assertEquals(1, deletion.trainingItemsDeleted)
            assertEquals(0, deletion.corpusDocumentsDeleted)
            assertEquals(0, deletion.screenMemoriesDeleted)
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())
            assertTrue(manager.nextQueue.value.none { it.source.startsWith("ScreenTrainingV2 (") })
        }
    }

    @Test
    fun alreadyPersistedTrainingStartIsAcceptedIdempotently() = runBlocking {
        withTrainingManager { _, repository, manager ->
            val activeItem = TrainingData(
                source = "persistence-test",
                title = "Already persisted active state",
                content = "The Room transition may have committed before its caller was cancelled.",
                status = TrainingStatus.TRAINING
            )
            repository.saveTrainingItem(activeItem)

            assertTrue(repository.markTrainingStarted(activeItem.id))
            assertTrue(repository.markTrainingStarted(activeItem.id))
            assertEquals(
                TrainingStatus.TRAINING,
                repository.getTrainingItem(activeItem.id)?.status
            )

            val completedItem = activeItem.copy(
                id = "completed-persistence-test",
                status = TrainingStatus.COMPLETED
            )
            repository.saveTrainingItem(completedItem)
            assertFalse(repository.markTrainingStarted(completedItem.id))
        }
    }

    @Test
    fun failedStartPersistenceRestoresItemWithoutTraining() = runBlocking {
        withTrainingManager { database, repository, manager ->
            manager.pauseTraining()
            manager.enqueueData(
                source = "persistence-test",
                title = "Start persistence failure",
                content = "A short queue item used to verify durable training state."
            )
            awaitPersistence(manager)

            val queuedItem = manager.nextQueue.value.single()
            assertEquals(TrainingStatus.QUEUED, queuedItem.status)
            assertEquals(queuedItem.id, repository.getPendingQueueSnapshot().single().id)

            database.close()

            val nextItem = withTimeout(5_000L) {
                manager.popNext()
            }
            assertNull(
                "A Room start-state failure must resolve instead of leaving popNext suspended.",
                nextItem
            )
            assertEquals(listOf(queuedItem.id), manager.nextQueue.value.map(TrainingData::id))
            assertEquals(TrainingStatus.QUEUED, manager.nextQueue.value.single().status)
            assertEquals(0, manager.totalStepsExecuted.value)
        }
    }

    @Test
    fun failedSavePersistenceDoesNotLeavePhantomQueuedItem() = runBlocking {
        withTrainingManager { database, _, manager ->
            manager.pauseTraining()
            database.close()

            manager.enqueueData(
                source = "persistence-test",
                title = "Save persistence failure",
                content = "A short queue item whose Room insert is expected to fail."
            )
            awaitPersistence(manager)

            assertTrue(manager.nextQueue.value.isEmpty())
            assertEquals(1, manager.completedQueue.value.size)
            assertEquals(TrainingStatus.FAILED, manager.completedQueue.value.single().status)
            assertEquals(0, manager.totalStepsExecuted.value)
        }
    }

    @Test
    fun resumeRestartsTheTrainingWorkerAfterARecoverableStartupFailure() = runBlocking {
        val startupAttempts = AtomicInteger(0)
        withTrainingManager(
            onWorkerStartup = {
                if (startupAttempts.incrementAndGet() == 1) {
                    throw IllegalStateException("Injected one-time training worker startup failure.")
                }
            }
        ) { _, repository, manager ->
            withTimeout(10_000L) {
                while (!manager.isTrainingPaused.value) {
                    delay(20L)
                }
            }
            assertEquals(1, startupAttempts.get())
            assertEquals(
                TrainingRecoveryPhase.BLOCKED,
                manager.trainingRecoveryStatus.value.phase
            )

            manager.resumeTraining()
            withTimeout(10_000L) {
                while (
                    manager.isTrainingPaused.value ||
                    startupAttempts.get() < 2 ||
                    manager.trainingRecoveryStatus.value.phase != TrainingRecoveryPhase.READY
                ) {
                    delay(20L)
                }
            }
            assertEquals(2, startupAttempts.get())
            assertEquals(
                TrainingRecoveryPhase.READY,
                manager.trainingRecoveryStatus.value.phase
            )

            manager.pauseTraining()
            manager.enqueueData(
                source = "worker-restart-test",
                title = "Resume after worker recovery",
                content = "This queue item verifies that a restarted worker resumes durable local training."
            )
            awaitPersistence(manager)
            val queuedItem = manager.nextQueue.value.single()

            manager.resumeTraining()
            awaitTrainingStatus(repository, queuedItem.id, TrainingStatus.COMPLETED)
            assertEquals(1, manager.totalStepsExecuted.value)
        }
    }

    @Test
    fun concurrentResumeRequestIsRetriedAfterAnEarlierRecoveryAttemptFails() = runBlocking {
        val startupAttempts = AtomicInteger(0)
        val firstResumeAttemptEnteredStartup = CompletableDeferred<Unit>()
        val releaseFirstResumeAttempt = CompletableDeferred<Unit>()

        withTrainingManager(
            onWorkerStartup = {
                when (startupAttempts.incrementAndGet()) {
                    1 -> throw IllegalStateException(
                        "Injected initial training worker startup failure."
                    )
                    2 -> {
                        firstResumeAttemptEnteredStartup.complete(Unit)
                        releaseFirstResumeAttempt.await()
                        throw IllegalStateException(
                            "Injected transient failure during the first Resume attempt."
                        )
                    }
                    else -> Unit
                }
            }
        ) { _, _, manager ->
            withTimeout(10_000L) {
                while (!manager.isTrainingPaused.value) {
                    delay(20L)
                }
            }
            assertEquals(1, startupAttempts.get())

            manager.resumeTraining()
            withTimeout(10_000L) {
                firstResumeAttemptEnteredStartup.await()
            }

            // This request arrives while the first recovery owns the resume mutex.
            // It must queue rather than disappear.
            manager.resumeTraining()
            releaseFirstResumeAttempt.complete(Unit)

            withTimeout(10_000L) {
                while (startupAttempts.get() < 3 || manager.isTrainingPaused.value) {
                    delay(20L)
                }
            }

            assertEquals(3, startupAttempts.get())
            assertFalse(manager.isTrainingPaused.value)
        }
    }

    @Test
    fun optimizerCommittedRowsSurviveInterruptedTrainingRecovery() = runBlocking {
        withTrainingManager { _, repository, manager ->
            manager.pauseTraining()
            val trainingItem = TrainingData(
                source = "persistence-test",
                title = "Committed optimizer transaction",
                content = "The optimizer already committed this item.",
                status = TrainingStatus.TRAINING
            )
            repository.saveTrainingItem(trainingItem)

            repository.markTrainingItemsOptimizerCommitted(
                listOf(
                    trainingItem.copy(
                        status = TrainingStatus.OPTIMIZER_COMMITTED,
                        finalLoss = 0.375f,
                        stepDurationMs = 120L
                    )
                )
            )
            repository.recoverInterruptedTrainingItems()

            assertTrue(repository.getPendingQueueSnapshot().isEmpty())
            val committed = repository.getOptimizerCommittedTrainingItemsSnapshot()
            assertEquals(1, committed.size)
            assertEquals(trainingItem.id, committed.single().id)
            assertEquals(TrainingStatus.OPTIMIZER_COMMITTED, committed.single().status)
            assertEquals(0.375f, requireNotNull(committed.single().finalLoss), 0f)
            assertEquals(120L, committed.single().stepDurationMs)
        }
    }

    @Test
    fun committedItemCanOnlyBeCompletedOnce() = runBlocking {
        withTrainingManager { _, repository, manager ->
            manager.pauseTraining()
            val item = TrainingData(
                source = "persistence-test",
                title = "Single completion claim",
                content = "A committed row must only transition to completed once.",
                status = TrainingStatus.TRAINING
            )
            repository.saveTrainingItem(item)
            repository.markTrainingItemsOptimizerCommitted(
                listOf(
                    item.copy(
                        status = TrainingStatus.OPTIMIZER_COMMITTED,
                        finalLoss = 0.25f,
                        stepDurationMs = 75L
                    )
                )
            )

            assertTrue(repository.markTrainingCompleted(item.id, 0.25f, 75L))
            assertFalse(repository.markTrainingCompleted(item.id, 0.25f, 75L))

            val persisted = requireNotNull(repository.getTrainingItem(item.id))
            assertEquals(TrainingStatus.COMPLETED, persisted.status)
            assertEquals(0.25f, requireNotNull(persisted.finalLoss), 0f)
            assertEquals(75L, persisted.stepDurationMs)
        }
    }

    @Test
    fun injectedFailuresAtEveryTrainingTransactionBoundaryPreserveTheCorrectState() = runBlocking {
        val cases = listOf(
            TrainingCommitBoundary.PREPARED_JOURNAL_DURABLE to TrainingStatus.FAILED,
            TrainingCommitBoundary.OPTIMIZER_UPDATE_APPLIED to TrainingStatus.FAILED,
            TrainingCommitBoundary.COMMITTED_JOURNAL_DURABLE to TrainingStatus.OPTIMIZER_COMMITTED,
            TrainingCommitBoundary.CHECKPOINT_FILE_REPLACED to TrainingStatus.OPTIMIZER_COMMITTED,
            TrainingCommitBoundary.CHECKPOINT_BOUND_JOURNAL_DURABLE to TrainingStatus.OPTIMIZER_COMMITTED,
            TrainingCommitBoundary.ROOM_OPTIMIZER_STATE_COMMITTED to TrainingStatus.OPTIMIZER_COMMITTED,
            TrainingCommitBoundary.QUEUE_ITEM_FINALIZED to TrainingStatus.COMPLETED
        )

        for ((boundary, expectedStatus) in cases) {
            val failureInjected = CompletableDeferred<Unit>()
            withTrainingManager(
                onCommitBoundary = { reachedBoundary ->
                    if (reachedBoundary == boundary && failureInjected.complete(Unit)) {
                        throw IllegalStateException("Injected fault after $boundary")
                    }
                }
            ) { _, repository, manager ->
                manager.pauseTraining()
                manager.enqueueData(
                    source = "transaction-fault-injection",
                    title = "Fault boundary $boundary",
                    content = "A deterministic sample used to verify durable optimizer transaction recovery."
                )
                awaitPersistence(manager)
                val item = manager.nextQueue.value.single()

                manager.triggerImmediateTrainingStep()
                withTimeout(30_000L) { failureInjected.await() }
                awaitTrainingStatus(repository, item.id, expectedStatus)

                val journalFile = File(context.filesDir, TrainingCommitJournal.FILE_NAME)
                val journal = TrainingCommitJournal.read(journalFile)
                when (expectedStatus) {
                    TrainingStatus.FAILED -> assertNull(
                        "A pre-commit fault must roll back and remove its PREPARED journal.",
                        journal
                    )
                    TrainingStatus.OPTIMIZER_COMMITTED -> {
                        val committed = requireNotNull(journal) {
                            "A durable optimizer commit must retain its recovery journal."
                        }
                        assertTrue(committed.isCommitted)
                        assertFalse(committed.checkpointSha256Fingerprint.isNullOrBlank())
                        assertEquals(
                            sha256Fingerprint(File(context.filesDir, CHECKPOINT_FILE_NAME)),
                            committed.checkpointSha256Fingerprint
                        )
                        assertEquals(
                            item.id,
                            repository.getOptimizerCommittedTrainingItemsSnapshot().single().id
                        )
                    }
                    TrainingStatus.COMPLETED -> {
                        val committed = requireNotNull(journal) {
                            "A completed item keeps the commit journal until batch finalization resolves."
                        }
                        assertTrue(committed.isCommitted)
                        assertEquals(
                            sha256Fingerprint(File(context.filesDir, CHECKPOINT_FILE_NAME)),
                            committed.checkpointSha256Fingerprint
                        )
                        assertEquals(
                            TrainingStatus.COMPLETED,
                            requireNotNull(repository.getTrainingItem(item.id)).status
                        )
                    }
                    else -> throw AssertionError("Unexpected injected-boundary expectation: $expectedStatus")
                }
            }
        }
    }

    private fun sha256Fingerprint(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(file.readBytes())
        return digest.joinToString(separator = "") { byte ->
            "%02x".format(Locale.ROOT, byte.toInt() and 0xFF)
        }
    }

    private suspend fun awaitTrainingStatus(
        repository: ArgusRepository,
        itemId: String,
        expectedStatus: TrainingStatus
    ) {
        withTimeout(30_000L) {
            while (repository.getTrainingItem(itemId)?.status != expectedStatus) {
                delay(20L)
            }
        }
    }

    private suspend fun awaitPersistence(manager: TrainingManager) {
        withTimeout(5_000L) {
            manager.awaitQueuePersistenceIdle()
        }
    }

    private suspend fun withTrainingManager(
        onCommitBoundary: (TrainingCommitBoundary) -> Unit = {},
        onWorkerStartup: suspend () -> Unit = {},
        test: suspend (
            ArgusDatabase,
            ArgusRepository,
            TrainingManager
        ) -> Unit
    ) {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()
        val journalFile = File(context.filesDir, TrainingCommitJournal.FILE_NAME)
        val previousJournal = journalFile.takeIf(File::isFile)?.readBytes()
        val validationMetadataFile = File(context.filesDir, "argus_conversation_quality.json")
        val previousValidationMetadata = validationMetadataFile.takeIf(File::isFile)?.readBytes()
        val previousAcceptedCheckpoints = context.filesDir.listFiles()
            .orEmpty()
            .filter { it.name.startsWith("argus_accepted_") && it.name.endsWith(".bin") }
            .associate { it.name to it.readBytes() }
        val database = Room.inMemoryDatabaseBuilder(
            context,
            ArgusDatabase::class.java
        ).build()
        val repository = ArgusRepository(context, database)
        val trainingDispatcher = Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "ArgusTrainingManagerPersistenceTest").apply {
                isDaemon = true
            }
        }.asCoroutineDispatcher()
        var manager: TrainingManager? = null

        try {
            checkpoint.delete()
            validationMetadataFile.delete()
            context.filesDir.listFiles()
                .orEmpty()
                .filter { it.name.startsWith("argus_accepted_") && it.name.endsWith(".bin") }
                .forEach { acceptedCheckpoint ->
                    check(acceptedCheckpoint.delete()) {
                        "An existing accepted checkpoint could not be isolated for the test."
                    }
                }
            check(!journalFile.exists() || journalFile.delete()) {
                "The previous training transaction journal could not be cleared for the test."
            }
            File(context.filesDir, "${TrainingCommitJournal.FILE_NAME}.tmp").delete()
            val engine = ArgusNeuralEngine.createBlocking(
                context,
                EngineConfig(
                    dModel = 16,
                    nHeads = 4,
                    dFf = 32,
                    maxSeqLen = 32,
                    loraRank = 2,
                    tokenizerConfig = TokenizerConfig(
                        maxVocabularySize = 512,
                        maxTokenLength = 32
                    )
                )
            )
            check(engine.saveCheckpoint()) {
                "The deterministic validation fixture checkpoint could not be saved."
            }
            val validationPersistence = ConversationalValidationPersistence(context.filesDir)
            check(validationPersistence.ensureAcceptedCheckpoint(checkpoint)) {
                "The accepted checkpoint fixture could not be secured."
            }
            val baselineReport = ConversationalValidationEvaluator.evaluate(
                cases = ConversationValidationSuite.cases
            ) { "Deterministic baseline fixture answer." }
            check(ConversationValidationContract.isCurrent(baselineReport)) {
                "The deterministic validation baseline does not match the active contract."
            }
            check(validationPersistence.saveBaseline(baselineReport)) {
                "The deterministic validation baseline fixture could not be persisted."
            }
            val deviceGuard = DeviceGuard(context)
            val worldModel = WorldModel(context, repository)
            val activeManager = TrainingManager(
                context = context,
                neuralEngine = engine,
                deviceGuard = deviceGuard,
                worldModel = worldModel,
                trainingDispatcher = trainingDispatcher,
                repository = repository,
                onCommitBoundary = onCommitBoundary,
                onWorkerStartup = onWorkerStartup
            )
            manager = activeManager
            test(database, repository, activeManager)
        } finally {
            try {
                manager?.close()
            } finally {
                runCatching { database.close() }
                trainingDispatcher.close()
            }
            if (previousCheckpoint == null) {
                checkpoint.delete()
            } else {
                checkpoint.writeBytes(previousCheckpoint)
            }
            File(context.filesDir, "$CHECKPOINT_FILE_NAME.tmp").delete()
            if (previousJournal == null) {
                journalFile.delete()
            } else {
                journalFile.writeBytes(previousJournal)
            }
            File(context.filesDir, "${TrainingCommitJournal.FILE_NAME}.tmp").delete()
            context.filesDir.listFiles()
                .orEmpty()
                .filter { it.name.startsWith("argus_accepted_") && it.name.endsWith(".bin") }
                .forEach { acceptedCheckpoint -> acceptedCheckpoint.delete() }
            previousAcceptedCheckpoints.forEach { (fileName, contents) ->
                File(context.filesDir, fileName).writeBytes(contents)
            }
            if (previousValidationMetadata == null) {
                validationMetadataFile.delete()
            } else {
                validationMetadataFile.writeBytes(previousValidationMetadata)
            }
        }
    }

    private companion object {
        const val CHECKPOINT_FILE_NAME = "argus_model_weights.bin"
    }
}
