package com.argus.agi.data.repository

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.argus.agi.data.local.ArgusDatabase
import com.argus.agi.data.local.entity.CorpusChunkEntity
import com.argus.agi.data.local.entity.CorpusDocumentEntity
import com.argus.agi.data.local.entity.TrainingReplaySessionEntity
import com.argus.agi.model.TrainingData
import com.argus.agi.model.TrainingStatus
import com.argus.agi.model.WorldEntity
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ArgusRepositoryReplaySessionTest {

    private lateinit var database: ArgusDatabase
    private lateinit var repository: ArgusRepository

    @Before
    fun setUp() {
        val context: Context = ApplicationProvider.getApplicationContext()
        database = Room.inMemoryDatabaseBuilder(
            context,
            ArgusDatabase::class.java
        ).allowMainThreadQueries().build()
        repository = ArgusRepository(context, database)
    }

    @After
    fun tearDown() {
        database.close()
    }

    @Test
    fun persistsReplayDescriptorQueueRowsAndReadyStateTogether() = runBlocking {
        val document = corpusDocument("document-a", "a")
        database.corpusDao().insertDocument(document)
        database.corpusDao().insertChunks(
            listOf(
                CorpusChunkEntity(
                    id = "chunk-a-1",
                    documentId = document.id,
                    chunkIndex = 1,
                    content = "local training corpus",
                    characterCount = 21
                )
            )
        )
        val session = replaySession("session-a", scheduledDocuments = 1, scheduledChunks = 1)
        val item = trainingItem(
            id = "item-a",
            documentId = document.id,
            sessionId = session.id
        )

        repository.saveCorpusReplaySession(session, listOf(item))

        assertEquals(session, database.trainingReplaySessionDao().getSession(session.id))
        val persistedItem = database.trainingQueueDao().getItem(item.id)
        assertEquals(session.id, persistedItem?.replaySessionId)
        assertEquals(TrainingStatus.QUEUED.name, persistedItem?.status)
        assertEquals("READY", database.corpusDao().getDocument(document.id)?.status)
    }

    @Test
    fun missingSourceDocumentDoesNotPersistAnyPartOfReplaySession() = runBlocking {
        val existingDocument = corpusDocument("document-existing", "c")
        database.corpusDao().insertDocument(existingDocument)
        database.corpusDao().insertChunks(
            listOf(
                CorpusChunkEntity(
                    id = "chunk-existing-1",
                    documentId = existingDocument.id,
                    chunkIndex = 1,
                    content = "existing retained content",
                    characterCount = 26
                )
            )
        )
        val session = replaySession("session-invalid", scheduledDocuments = 2, scheduledChunks = 2)
        val items = listOf(
            trainingItem("item-existing", existingDocument.id, session.id),
            trainingItem("item-missing", "document-missing", session.id)
        )

        val failure = runCatching {
            repository.saveCorpusReplaySession(session, items)
        }.exceptionOrNull()

        assertTrue(failure is IllegalArgumentException)
        assertNull(database.trainingReplaySessionDao().getSession(session.id))
        assertTrue(database.trainingQueueDao().getPendingQueueSnapshot().isEmpty())
        assertEquals("IMPORTED", database.corpusDao().getDocument(existingDocument.id)?.status)
    }

    @Test
    fun failureDuringReadyTransitionRollsBackSessionAndQueueRows() = runBlocking {
        val document = corpusDocument("document-trigger", "e")
        database.corpusDao().insertDocument(document)
        database.corpusDao().insertChunks(
            listOf(
                CorpusChunkEntity(
                    id = "chunk-trigger-1",
                    documentId = document.id,
                    chunkIndex = 1,
                    content = "transaction rollback sample",
                    characterCount = 28
                )
            )
        )
        database.openHelper.writableDatabase.execSQL(
            """
            CREATE TRIGGER abort_replay_ready_update
            BEFORE UPDATE OF status ON corpus_documents
            WHEN NEW.status = 'READY'
            BEGIN
                SELECT RAISE(ABORT, 'forced replay transaction failure');
            END
            """.trimIndent()
        )
        val session = replaySession("session-trigger", scheduledDocuments = 1, scheduledChunks = 1)
        val item = trainingItem("item-trigger", document.id, session.id)

        val failure = runCatching {
            repository.saveCorpusReplaySession(session, listOf(item))
        }.exceptionOrNull()

        assertTrue(failure != null)
        assertNull(database.trainingReplaySessionDao().getSession(session.id))
        assertNull(database.trainingQueueDao().getItem(item.id))
        assertTrue(database.trainingQueueDao().getPendingQueueSnapshot().isEmpty())
        assertEquals("IMPORTED", database.corpusDao().getDocument(document.id)?.status)
    }

    @Test
    fun optimizerCommitBatchRollsBackAllRowsWhenOneStatusUpdateFails() = runBlocking {
        val first = trainingItem(
            id = "optimizer-first",
            documentId = null,
            sessionId = null
        )
        val second = trainingItem(
            id = "optimizer-second",
            documentId = null,
            sessionId = null
        )
        repository.saveTrainingItems(listOf(first, second))
        database.openHelper.writableDatabase.execSQL(
            """
            CREATE TRIGGER abort_second_optimizer_commit
            BEFORE UPDATE OF status ON training_items
            WHEN NEW.id = 'optimizer-second'
                AND NEW.status = 'OPTIMIZER_COMMITTED'
            BEGIN
                SELECT RAISE(ABORT, 'forced optimizer commit persistence failure');
            END
            """.trimIndent()
        )
        val committedItems = listOf(first, second).map { item ->
            item.copy(
                status = TrainingStatus.OPTIMIZER_COMMITTED,
                finalLoss = 0.25f,
                stepDurationMs = 10L
            )
        }

        val failure = runCatching {
            repository.markTrainingItemsOptimizerCommitted(committedItems)
        }.exceptionOrNull()

        assertTrue(failure != null)
        assertEquals(
            TrainingStatus.QUEUED,
            requireNotNull(repository.getTrainingItem(first.id)).status
        )
        assertEquals(
            TrainingStatus.QUEUED,
            requireNotNull(repository.getTrainingItem(second.id)).status
        )
    }

    @Test
    fun clearScreenDerivedDataDeletesScreenOnlyRecordsAndPreservesSharedCorpusSources() = runBlocking {
        val screenOnly = screenCorpusDocument(
            id = "screen-only-document",
            fingerprint = "a".repeat(64),
            source = "Screen (com.example.private)"
        )
        val screenOnlyChunk = screenCorpusChunk("screen-only-document", "screen-only-chunk")
        assertTrue(
            repository.storeCorpusDocument(
                screenOnly,
                listOf(screenOnlyChunk)
            )
        )

        val sharedScreenDocument = screenCorpusDocument(
            id = "shared-screen-document",
            fingerprint = "b".repeat(64),
            source = "Screen (com.example.reader)"
        )
        val sharedChunk = screenCorpusChunk("shared-screen-document", "shared-screen-chunk")
        assertTrue(
            repository.storeCorpusDocument(
                sharedScreenDocument,
                listOf(sharedChunk)
            )
        )
        assertTrue(
            !repository.storeCorpusDocument(
                sharedScreenDocument.copy(source = "user-approved-document"),
                listOf(sharedChunk)
            )
        )

        val screenQueueItem = TrainingData(
            id = "screen-queue-item",
            source = "Screen (com.example.private)",
            title = "Screen sample",
            content = "Private screen text"
        )
        val retainedQueueItem = TrainingData(
            id = "retained-queue-item",
            source = "user-approved-document",
            title = "Approved document",
            content = "User-approved document text"
        )
        repository.saveTrainingItems(listOf(screenQueueItem, retainedQueueItem))
        repository.saveMemories(
            listOf(
                WorldEntity("active_screen_app", "com.example.private", "Accessibility Perception"),
                WorldEntity("last_screen_content", "Private screen content", "Screen OCR/Text Parser"),
                WorldEntity("user_fact", "Retained local memory", "User")
            )
        )

        val result = repository.clearScreenDerivedData()

        assertEquals(1, result.trainingItemsDeleted)
        assertEquals(1, result.corpusDocumentsDeleted)
        assertEquals(2, result.screenMemoriesDeleted)

        assertNull(repository.getCorpusDocument("screen-only-document"))
        assertEquals(
            "shared-screen-document",
            repository.getCorpusDocument("shared-screen-document")?.id
        )
        assertEquals(
            listOf("user-approved-document"),
            repository.getCorpusSources("shared-screen-document").map { it.source }
        )
        assertEquals(1, repository.getCorpusChunks("shared-screen-document").size)

        assertNull(repository.getTrainingItem("screen-queue-item"))
        assertEquals(
            "user-approved-document",
            requireNotNull(repository.getTrainingItem("retained-queue-item")).source
        )

        val memories = repository.getMemoriesSnapshot().associateBy(WorldEntity::key)
        assertTrue("active screen app memory must be deleted", "active_screen_app" !in memories)
        assertTrue("screen content memory must be deleted", "last_screen_content" !in memories)
        assertEquals("Retained local memory", memories["user_fact"]?.value)
    }

    @Test
    fun clearScreenDerivedDataRefusesOptimizerCommittedScreenItemsWithoutPartialDeletion() = runBlocking {
        val screenItem = TrainingData(
            id = "screen-committed-item",
            source = "Screen (com.example.private)",
            title = "Committed screen sample",
            content = "Screen text whose optimizer transaction is committed",
            status = TrainingStatus.OPTIMIZER_COMMITTED,
            finalLoss = 0.25f,
            stepDurationMs = 12L
        )
        repository.saveTrainingItem(screenItem)
        repository.saveMemory(
            WorldEntity("last_screen_content", "Keep until commit is finalized", "Screen OCR/Text Parser")
        )

        val failure = runCatching {
            repository.clearScreenDerivedData()
        }.exceptionOrNull()

        assertTrue(failure is IllegalStateException)
        assertEquals(
            TrainingStatus.OPTIMIZER_COMMITTED,
            requireNotNull(repository.getTrainingItem(screenItem.id)).status
        )
        assertTrue(
            repository.getMemoriesSnapshot().any { it.key == "last_screen_content" }
        )
    }

    private fun screenCorpusDocument(
        id: String,
        fingerprint: String,
        source: String
    ) = CorpusDocumentEntity(
        id = id,
        contentFingerprint = fingerprint,
        fingerprintAlgorithm = "SHA-256/UTF-16BE",
        source = source,
        title = id,
        language = "en",
        canonicalUrl = null,
        characterCount = 20,
        chunkCount = 1,
        datasetSplit = "TRAIN",
        validationText = null,
        plannedEpochs = 1,
        rightsStatus = "UNVERIFIED",
        status = "IMPORTED",
        createdAtEpochMs = 300L,
        lastTrainedAtEpochMs = null,
        completedPassCount = 0
    )

    private fun screenCorpusChunk(documentId: String, chunkId: String) =
        CorpusChunkEntity(
            id = chunkId,
            documentId = documentId,
            chunkIndex = 1,
            content = "A retained screen-derived corpus text sample",
            characterCount = 44
        )

    private fun corpusDocument(id: String, fingerprintCharacter: String) =
        CorpusDocumentEntity(
            id = id,
            contentFingerprint = fingerprintCharacter.repeat(64),
            fingerprintAlgorithm = "SHA-256",
            source = "local-import",
            title = id,
            language = "id",
            canonicalUrl = null,
            characterCount = 21,
            chunkCount = 1,
            datasetSplit = "TRAIN",
            validationText = null,
            plannedEpochs = 1,
            rightsStatus = "USER_DECLARED",
            status = "IMPORTED",
            createdAtEpochMs = 100L,
            lastTrainedAtEpochMs = null,
            completedPassCount = 0
        )

    private fun replaySession(
        id: String,
        scheduledDocuments: Int,
        scheduledChunks: Int
    ) = TrainingReplaySessionEntity(
        id = id,
        sourceSnapshotFingerprint = "d".repeat(64),
        seed = 42L,
        policy = "INTERLEAVED_DOCUMENTS",
        epochs = 2,
        schedulerFormatVersion = 1,
        scheduledDocumentCount = scheduledDocuments,
        scheduledChunkCount = scheduledChunks,
        createdAtEpochMs = 200L
    )

    private fun trainingItem(id: String, documentId: String?, sessionId: String?) =
        TrainingData(
            id = id,
            source = "local-import",
            title = id,
            content = "local training corpus",
            documentId = documentId,
            replaySessionId = sessionId
        )
}
