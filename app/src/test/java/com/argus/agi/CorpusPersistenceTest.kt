package com.argus.agi

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import android.content.Context
import com.argus.agi.data.local.ArgusDatabase
import com.argus.agi.data.repository.ArgusRepository
import com.argus.agi.model.ArgusNeuralEngine
import com.argus.agi.model.CorpusReplayPolicy
import com.argus.agi.model.EngineConfig
import com.argus.agi.model.TrainingManager
import com.argus.agi.model.TrainingData
import com.argus.agi.model.WorldModel
import com.argus.agi.sensor.DeviceGuard
import java.io.StringReader
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class CorpusPersistenceTest {



    @Test
    fun analyticalBackpropGradientsMatchCentralFiniteDifferences() {
        val context: Context = ApplicationProvider.getApplicationContext()
        val checkpointFile = java.io.File(context.filesDir, "argus_model_weights.bin")
        val previousCheckpoint = checkpointFile.takeIf { it.isFile }?.readBytes()

        try {
            checkpointFile.delete()
            val engine = ArgusNeuralEngine.createBlocking(
                context = context,
                config = EngineConfig(
                    dModel = 48,
                    nHeads = 4,
                    dFf = 128,
                    maxSeqLen = 32,
                    loraRank = 2
                )
            )
            val sample = "Argus learns local language"
            val initialTrainingLoss = engine.trainStep(sample)
            assertTrue("The setup step must produce a finite loss.", initialTrainingLoss.isFinite())
            assertTrue("The diagnostic baseline checkpoint must be writable.", engine.saveCheckpoint())
            val baselineCheckpoint = checkpointFile.readBytes()
            val lossOnlyEvaluation = engine.evaluateLoss(sample)
            val parameterGroups = listOf(
                "wHead",
                "wEmbed",
                "wDown",
                "wUp",
                "wGate",
                "wQ",
                "wK",
                "wV",
                "wO",
                "loraQa",
                "loraQb",
                "loraVa",
                "loraVb"
            )

            parameterGroups.forEach { parameterName ->
                val result = engine.checkGradient(
                    text = sample,
                    parameterName = parameterName,
                    parameterIndex = -1,
                    epsilon = 1e-2f
                )
                assertEquals(
                    "Loss-only evaluation must match the loss used by full backprop.",
                    lossOnlyEvaluation,
                    result.meanLoss,
                    1e-5f
                )
                val magnitude = maxOf(
                    kotlin.math.abs(result.analyticalGradient),
                    kotlin.math.abs(result.numericalGradient)
                )
                val tolerance = 2e-3f + 0.05f * magnitude

                assertTrue(
                    "$parameterName selected gradient should be informative: $result",
                    kotlin.math.abs(result.analyticalGradient) > 1e-8f
                )
                assertTrue(
                    "$parameterName analytical/numerical mismatch: $result",
                    result.absoluteError <= tolerance
                )
            }

            assertTrue("The diagnostic checkpoint must remain writable.", engine.saveCheckpoint())
            assertArrayEquals(
                "Gradient diagnostics must not mutate parameters, optimizer state, or tokenizer state.",
                baselineCheckpoint,
                checkpointFile.readBytes()
            )
        } finally {
            if (previousCheckpoint == null) {
                checkpointFile.delete()
            } else {
                checkpointFile.writeBytes(previousCheckpoint)
            }
        }
    }

    @Test
    fun validationDocumentsStayHeldOutAndEvaluateWithoutChangingWeights() = runTest {
        val context: Context = ApplicationProvider.getApplicationContext()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            ArgusDatabase::class.java
        ).allowMainThreadQueries().build()
        val repository = ArgusRepository(context, database)
        val engine = ArgusNeuralEngine.createBlocking(context)
        val manager = TrainingManager(
            context = context,
            neuralEngine = engine,
            deviceGuard = DeviceGuard(context),
            worldModel = WorldModel(context),
            repository = repository
        )

        try {
            manager.pauseTraining()
            assertTrue("Baseline checkpoint must be writable.", engine.saveCheckpoint())
            val checkpoint = java.io.File(context.filesDir, "argus_model_weights.bin").readBytes()
            val validationText = (
                "Held out documents must remain separate from the training corpus. " +
                    "Evaluation computes next token cross entropy without changing model weights. "
                ).repeat(20)

            assertTrue(
                manager.importValidationDocument(
                    source = "Validation test",
                    title = "Held-out document",
                    fullText = validationText,
                    language = "en"
                )
            )
            val shorterValidationText =
                "A second independent held-out document tests corpus-level target weighting. ".repeat(5)
            assertTrue(
                manager.importValidationDocument(
                    source = "Validation test",
                    title = "Short held-out document",
                    fullText = shorterValidationText,
                    language = "en"
                )
            )

            val heldOut = repository.getCorpusDocuments()
                .single { it.validationText == validationText.trim() }
            assertEquals("VALIDATION", heldOut.datasetSplit)
            assertEquals(validationText.trim(), heldOut.validationText)
            assertEquals(0, heldOut.chunkCount)
            assertTrue(repository.getCorpusChunks(heldOut.id).isEmpty())
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())

            val firstDocumentLoss = engine.evaluateLossWithTargetCount(validationText)
            val secondDocumentLoss = engine.evaluateLossWithTargetCount(shorterValidationText)
            val expectedTargetCount =
                firstDocumentLoss.targetCount + secondDocumentLoss.targetCount
            val expectedWeightedLoss = (
                firstDocumentLoss.meanLoss * firstDocumentLoss.targetCount +
                    secondDocumentLoss.meanLoss * secondDocumentLoss.targetCount
                ) / expectedTargetCount.toFloat()

            val evaluation = manager.evaluateValidationCorpus()
            assertEquals(2, evaluation.corpusDocumentCount)
            assertEquals(2, evaluation.evaluatedDocumentCount)
            assertEquals(expectedTargetCount, evaluation.targetTokenCount)
            assertEquals(expectedWeightedLoss, evaluation.meanLoss, 1e-5f)
            assertTrue(evaluation.targetTokenCount > 0)
            assertTrue(evaluation.meanLoss.isFinite())
            assertTrue(evaluation.perplexity.isFinite())

            assertTrue(
                "Validation evaluation must not mutate the model checkpoint.",
                checkpoint.contentEquals(
                    java.io.File(context.filesDir, "argus_model_weights.bin").readBytes()
                )
            )

            val replayFailure = runCatching {
                manager.replayCorpusDocument(heldOut.id, epochs = 1)
            }.exceptionOrNull()
            assertTrue("Validation documents must never be replayed for training.", replayFailure != null)
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())

            val directQueueFailure = runCatching {
                repository.saveTrainingItem(
                    TrainingData(
                        source = "Repository boundary test",
                        title = "Invalid validation queue row",
                        content = validationText,
                        documentId = heldOut.id
                    )
                )
            }.exceptionOrNull()
            assertTrue(
                "Repository must reject validation document queue references.",
                directQueueFailure != null
            )
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())

            val trainingCollision = runCatching {
                manager.enqueueDocument(
                    source = "Attempted training duplicate",
                    title = "Same held-out content",
                    fullText = validationText,
                    chunkSize = 180,
                    overlap = 30,
                    epochs = 1
                )
            }.exceptionOrNull()
            assertTrue(
                "Identical validation content must not be imported into the training split.",
                trainingCollision != null
            )
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())
        } finally {
            manager.close()
            database.close()
        }
    }


    @Test
    fun seededCorpusReplayIsRepeatableAndPreservesPerDocumentChunkOrder() = runTest {
        val context: Context = ApplicationProvider.getApplicationContext()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            ArgusDatabase::class.java
        ).allowMainThreadQueries().build()
        val repository = ArgusRepository(context, database)
        val manager = TrainingManager(
            context = context,
            neuralEngine = ArgusNeuralEngine.createBlocking(context),
            deviceGuard = DeviceGuard(context),
            worldModel = WorldModel(context),
            repository = repository
        )

        try {
            manager.pauseTraining()
            val documents = listOf(
                "Research document alpha preserves its own source sequence. ".repeat(24),
                "Research document beta contributes another ordered token stream. ".repeat(19),
                "Research document gamma is interleaved only at chunk boundaries. ".repeat(16)
            )

            documents.forEachIndexed { index, text ->
                val queued = manager.enqueueDocument(
                    source = "Replay scheduler test",
                    title = "Replay document ${index + 1}",
                    fullText = text,
                    chunkSize = 170,
                    overlap = 25,
                    epochs = 1
                )
                assertTrue("Each source document must produce more than one chunk.", queued > 1)
            }
            manager.awaitQueuePersistenceIdle()
            manager.clearQueue()
            manager.awaitQueuePersistenceIdle()

            val trainingDocuments = repository.getTrainingCorpusDocuments()
            assertEquals(3, trainingDocuments.size)
            val expectedChunkCount =
                trainingDocuments.sumOf { it.chunkCount.toLong() } * 2L

            val overLimitFailure = runCatching {
                manager.scheduleCorpusReplay(
                    seed = 42L,
                    epochs = 2,
                    policy = CorpusReplayPolicy.INTERLEAVED_DOCUMENTS,
                    maxQueuedChunks = 1
                )
            }.exceptionOrNull()
            assertTrue("An oversized replay must be rejected before queue insertion.", overLimitFailure != null)
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())

            val firstRun = manager.scheduleCorpusReplay(
                seed = 42L,
                epochs = 2,
                policy = CorpusReplayPolicy.INTERLEAVED_DOCUMENTS
            )
            assertEquals(3, firstRun.scheduledDocumentCount)
            assertEquals(expectedChunkCount.toInt(), firstRun.scheduledChunkCount)
            manager.awaitQueuePersistenceIdle()

            val firstQueue = repository.getPendingQueueSnapshot()
            assertEquals(expectedChunkCount.toInt(), firstQueue.size)
            val firstOrder = firstQueue.map { item ->
                Triple(
                    requireNotNull(item.documentId),
                    item.chunkIndex,
                    item.content
                )
            }

            val orderedByDocument = firstQueue.groupBy { requireNotNull(it.documentId) }
            assertEquals(3, orderedByDocument.size)
            orderedByDocument.forEach { (documentId, documentItems) ->
                assertEquals(
                    "Each document must retain a contiguous chunk order over all epochs.",
                    (1..documentItems.size).toList(),
                    documentItems.map { it.chunkIndex }
                )
                val retainedChunks = repository.getCorpusChunks(documentId)
                    .sortedBy { it.chunkIndex }
                    .map { it.content }
                val expectedDocumentContent = buildList {
                    repeat(2) { addAll(retainedChunks) }
                }
                assertEquals(
                    "Each epoch must replay the retained document chunks in their original order.",
                    expectedDocumentContent,
                    documentItems.map { it.content }
                )
            }
            assertTrue(
                "Interleaved replay must mix at least two document streams.",
                firstQueue.zipWithNext().any { (first, second) ->
                    first.documentId != second.documentId
                }
            )

            val duplicateRunFailure = runCatching {
                manager.scheduleCorpusReplay(
                    seed = 42L,
                    epochs = 2,
                    policy = CorpusReplayPolicy.INTERLEAVED_DOCUMENTS
                )
            }.exceptionOrNull()
            assertTrue(
                "A corpus document with active queue work must not be scheduled twice.",
                duplicateRunFailure != null
            )
            assertEquals(firstQueue.size, repository.getPendingQueueSnapshot().size)

            manager.clearQueue()
            manager.awaitQueuePersistenceIdle()
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())

            val repeatedRun = manager.scheduleCorpusReplay(
                seed = 42L,
                epochs = 2,
                policy = CorpusReplayPolicy.INTERLEAVED_DOCUMENTS
            )
            manager.awaitQueuePersistenceIdle()
            val repeatedQueue = repository.getPendingQueueSnapshot()
            val repeatedOrder = repeatedQueue.map { item ->
                Triple(
                    requireNotNull(item.documentId),
                    item.chunkIndex,
                    item.content
                )
            }

            assertEquals(firstRun, repeatedRun)
            assertEquals("The same seed and corpus snapshot must reproduce the exact queue order.", firstOrder, repeatedOrder)
        } finally {
            manager.close()
            database.close()
        }
    }

    @Test
    fun retainedCorpusSurvivesQueueClearAndCanBeReplayed() = runTest {
        val context: Context = ApplicationProvider.getApplicationContext()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            ArgusDatabase::class.java
        ).allowMainThreadQueries().build()
        val repository = ArgusRepository(context, database)
        val engine = ArgusNeuralEngine.createBlocking(context)
        val manager = TrainingManager(
            context = context,
            neuralEngine = engine,
            deviceGuard = DeviceGuard(context),
            worldModel = WorldModel(context),
            repository = repository
        )

        try {
            manager.pauseTraining()
            val text = (
                "Argus retains imported documents independently from the active training queue. " +
                    "Clearing pending work must not destroy the source chunks required for replay. "
                ).repeat(18)

            val queuedCount = manager.enqueueDocument(
                source = "Corpus persistence test",
                title = "Retained document",
                fullText = text,
                chunkSize = 180,
                overlap = 30,
                epochs = 2
            )

            assertTrue("Initial import must schedule training chunks.", queuedCount > 0)
            manager.awaitQueuePersistenceIdle()

            val documents = repository.getCorpusDocuments()
            assertEquals("Exactly one corpus document must be retained.", 1, documents.size)
            val document = documents.single()
            val storedChunks = repository.getCorpusChunks(document.id)
            assertEquals(document.chunkCount, storedChunks.size)
            assertEquals(1, repository.getCorpusSources(document.id).size)
            assertEquals(queuedCount, repository.getPendingQueueSnapshot().size)

            val duplicateCount = manager.enqueueDocument(
                source = "Second import source",
                title = "Same text under another title",
                fullText = text,
                chunkSize = 180,
                overlap = 30,
                epochs = 1
            )
            assertEquals("Identical content must not be queued twice.", 0, duplicateCount)
            assertEquals(1, repository.getCorpusDocumentCount())
            assertEquals(2, repository.getCorpusSources(document.id).size)

            val streamDuplicate = manager.enqueueDocumentStream(
                source = "Stream duplicate source",
                title = "Stream duplicate",
                reader = StringReader(text),
                chunkSize = 180,
                overlap = 30,
                epochs = 1
            )
            assertEquals("Stream and string ingestion must share the same fingerprint.", 0, streamDuplicate.first)
            assertEquals(1, repository.getCorpusDocumentCount())
            assertEquals(3, repository.getCorpusSources(document.id).size)

            val unverifiedArticleImport = manager.enqueueDocument(
                source = "web_article|source=reviewed-page",
                title = "Article source",
                fullText = text,
                canonicalUrl = "https://example.org/article",
                rightsStatus = "UNVERIFIED"
            )
            assertEquals(0, unverifiedArticleImport)

            val declaredArticleImport = manager.enqueueDocument(
                source = "web_article|source=reviewed-page|rights_declared_by_user=true",
                title = "Article source",
                fullText = text,
                canonicalUrl = "https://example.org/article",
                rightsStatus = "USER_DECLARED"
            )
            assertEquals(0, declaredArticleImport)

            val articleSource = repository.getCorpusSources(document.id)
                .single { it.canonicalUrl == "https://example.org/article" }
            assertEquals("USER_DECLARED", articleSource.rightsStatus)
            assertEquals(4, repository.getCorpusSources(document.id).size)

            manager.enqueueDocument(
                source = "web_article|source=reviewed-page",
                title = "Article source",
                fullText = text,
                canonicalUrl = "https://example.org/article",
                rightsStatus = "UNVERIFIED"
            )
            val sourceAfterUnverifiedReimport = repository.getCorpusSources(document.id)
                .single { it.canonicalUrl == "https://example.org/article" }
            assertEquals(
                "A less authoritative repeat import must not downgrade a prior rights declaration.",
                "USER_DECLARED",
                sourceAfterUnverifiedReimport.rightsStatus
            )

            val activeQueueItem = repository.getPendingQueueSnapshot().first()
            assertTrue(activeQueueItem.chunkIndex < activeQueueItem.totalChunks)
            assertTrue(repository.markTrainingStarted(activeQueueItem.id))

            manager.clearQueue()
            manager.awaitQueuePersistenceIdle()

            assertTrue(
                "Queue clearing must remove waiting work.",
                repository.getPendingQueueSnapshot().isEmpty()
            )
            assertEquals(
                "An already active training row must survive queue clearing.",
                com.argus.agi.model.TrainingStatus.TRAINING,
                repository.getTrainingItem(activeQueueItem.id)?.status
            )
            assertEquals(
                "A document with remaining chunks removed from the queue must become partial.",
                "PARTIAL",
                repository.getCorpusDocument(document.id)?.status
            )

            assertEquals(
                "Interrupted active training rows must be re-queued on recovery.",
                1,
                repository.recoverInterruptedTrainingItems()
            )
            val recoveredItems = repository.getPendingQueueSnapshot()
            assertEquals(1, recoveredItems.size)
            assertEquals(activeQueueItem.id, recoveredItems.single().id)

            manager.clearQueue()
            manager.awaitQueuePersistenceIdle()
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())

            assertEquals(
                "Corpus data must survive queue clearing.",
                document.chunkCount,
                repository.getCorpusChunks(document.id).size
            )

            val replayedCount = manager.replayCorpusDocument(document.id, epochs = 2)
            assertEquals(document.chunkCount * 2, replayedCount)
            manager.awaitQueuePersistenceIdle()

            val replayedItems = repository.getPendingQueueSnapshot()
            assertEquals(replayedCount, replayedItems.size)
            assertTrue(replayedItems.all { it.documentId == document.id })
            assertEquals(
                "Replay must preserve canonical corpus chunk text.",
                storedChunks.first().content,
                replayedItems.first().content
            )
        } finally {
            manager.close()
            database.close()
        }
    }
}
