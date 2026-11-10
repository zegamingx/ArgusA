package com.argus.agi

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import android.content.Context
import com.argus.agi.data.local.ArgusDatabase
import com.argus.agi.data.repository.ArgusRepository
import com.argus.agi.model.ArgusNeuralEngine
import com.argus.agi.model.TrainingManager
import com.argus.agi.model.TrainingData
import com.argus.agi.model.WorldModel
import com.argus.agi.sensor.DeviceGuard
import java.io.StringReader
import kotlinx.coroutines.test.runTest
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
