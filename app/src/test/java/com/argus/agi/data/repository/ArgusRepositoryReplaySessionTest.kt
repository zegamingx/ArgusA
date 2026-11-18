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

    private fun trainingItem(id: String, documentId: String, sessionId: String) =
        TrainingData(
            id = id,
            source = "local-import",
            title = id,
            content = "local training corpus",
            documentId = documentId,
            replaySessionId = sessionId
        )
}
