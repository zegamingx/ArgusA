package com.argus.agi

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import android.content.Context
import com.argus.agi.data.local.ArgusDatabase
import com.argus.agi.data.repository.ArgusRepository
import com.argus.agi.model.ArgusNeuralEngine
import com.argus.agi.model.TrainingManager
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

            manager.clearQueue()
            manager.awaitQueuePersistenceIdle()

            assertTrue("Queue clearing must remove pending work.", repository.getPendingQueueSnapshot().isEmpty())
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
