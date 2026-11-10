package com.argus.agi.data.repository

import android.content.Context
import android.net.Uri
import androidx.room.withTransaction
import com.argus.agi.data.local.ArgusDatabase
import com.argus.agi.data.local.entity.ChatMessageEntity
import com.argus.agi.data.local.entity.CorpusChunkEntity
import com.argus.agi.data.local.entity.CorpusDocumentEntity
import com.argus.agi.data.local.entity.TrainingItemEntity
import com.argus.agi.data.local.entity.WorldMemoryEntity
import com.argus.agi.model.TrainingData
import com.argus.agi.model.TrainingStatus
import com.argus.agi.model.WorldEntity
import com.argus.agi.ui.ChatMessage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.withContext

class ArgusRepository(
    context: Context,
    database: ArgusDatabase = ArgusDatabase.getInstance(context)
) {

    private val db = database
    private val chatMessageDao = db.chatMessageDao()
    private val worldMemoryDao = db.worldMemoryDao()
    private val trainingQueueDao = db.trainingQueueDao()
    private val corpusDao = db.corpusDao()

    // =========================================================================
    // Chat Messages
    // =========================================================================

    val allChatMessages: Flow<List<ChatMessage>> = chatMessageDao.getAllMessages().map { entities ->
        entities.map { it.toDomain() }
    }

    suspend fun getChatMessagesSnapshot(): List<ChatMessage> = withContext(Dispatchers.IO) {
        chatMessageDao.getAllMessagesSnapshot().map { it.toDomain() }
    }

    suspend fun saveChatMessage(message: ChatMessage) = withContext(Dispatchers.IO) {
        chatMessageDao.insertMessage(message.toEntity())
    }

    suspend fun saveChatMessages(messages: List<ChatMessage>) = withContext(Dispatchers.IO) {
        chatMessageDao.insertMessages(messages.map { it.toEntity() })
    }

    suspend fun clearChatMessages() = withContext(Dispatchers.IO) {
        chatMessageDao.clearAll()
    }

    // =========================================================================
    // World Model Memories
    // =========================================================================

    val allMemories: Flow<List<WorldEntity>> = worldMemoryDao.getAllMemories().map { entities ->
        entities.map { it.toDomain() }
    }

    suspend fun getMemoriesSnapshot(): List<WorldEntity> = withContext(Dispatchers.IO) {
        worldMemoryDao.getAllMemoriesSnapshot().map { it.toDomain() }
    }

    suspend fun saveMemory(entity: WorldEntity) = withContext(Dispatchers.IO) {
        worldMemoryDao.insertMemory(entity.toEntity())
    }

    suspend fun saveMemories(entities: List<WorldEntity>) = withContext(Dispatchers.IO) {
        worldMemoryDao.insertMemories(entities.map { it.toEntity() })
    }

    suspend fun deleteMemory(key: String) = withContext(Dispatchers.IO) {
        worldMemoryDao.deleteMemory(key)
    }

    suspend fun pruneExpiredMemories(ttlMs: Long) = withContext(Dispatchers.IO) {
        val threshold = System.currentTimeMillis() - ttlMs
        worldMemoryDao.deleteExpired(threshold)
    }

    // =========================================================================
    // Training Queue & History
    // =========================================================================

    val pendingTrainingQueue: Flow<List<TrainingData>> = trainingQueueDao.getPendingQueue().map { entities ->
        entities.map { it.toDomain() }
    }

    suspend fun getPendingQueueSnapshot(): List<TrainingData> = withContext(Dispatchers.IO) {
        trainingQueueDao.getPendingQueueSnapshot().map { it.toDomain() }
    }

    suspend fun saveTrainingItem(item: TrainingData) = withContext(Dispatchers.IO) {
        trainingQueueDao.insertItem(item.toEntity())
    }

    suspend fun saveTrainingItems(items: List<TrainingData>) = withContext(Dispatchers.IO) {
        trainingQueueDao.insertItems(items.map { it.toEntity() })
    }

    suspend fun markTrainingCompleted(id: String, loss: Float, durationMs: Long) = withContext(Dispatchers.IO) {
        trainingQueueDao.updateStatus(id, "COMPLETED", loss, durationMs)
    }

    suspend fun markTrainingFailed(id: String) = withContext(Dispatchers.IO) {
        trainingQueueDao.updateStatus(id, "FAILED", null, 0L)
    }

    suspend fun clearPendingQueue() = withContext(Dispatchers.IO) {
        trainingQueueDao.clearPendingQueue()
    }

    // =========================================================================
    // Persistent Local Corpus
    // =========================================================================

    val corpusDocuments: Flow<List<CorpusDocumentEntity>> =
        corpusDao.observeDocuments(limit = 500)

    suspend fun storeCorpusDocument(
        document: CorpusDocumentEntity,
        chunks: List<CorpusChunkEntity>
    ): Boolean = withContext(Dispatchers.IO) {
        require(chunks.size == document.chunkCount) {
            "Corpus chunk count does not match document metadata."
        }
        require(chunks.all { it.documentId == document.id }) {
            "Every corpus chunk must reference the document being stored."
        }

        db.withTransaction {
            val insertedRowId = corpusDao.insertDocument(document)
            if (insertedRowId == -1L) {
                false
            } else {
                corpusDao.insertChunks(chunks)
                true
            }
        }
    }

    suspend fun getCorpusDocument(documentId: String): CorpusDocumentEntity? =
        withContext(Dispatchers.IO) {
            corpusDao.getDocument(documentId)
        }

    suspend fun getCorpusDocumentByFingerprint(fingerprint: String): CorpusDocumentEntity? =
        withContext(Dispatchers.IO) {
            corpusDao.getDocumentByFingerprint(fingerprint)
        }

    suspend fun getCorpusDocuments(limit: Int = 500): List<CorpusDocumentEntity> =
        withContext(Dispatchers.IO) {
            require(limit > 0) { "limit must be positive." }
            corpusDao.getDocuments(limit)
        }

    suspend fun getCorpusDocumentCount(): Int = withContext(Dispatchers.IO) {
        corpusDao.getDocumentCount()
    }

    suspend fun getCorpusChunks(documentId: String): List<CorpusChunkEntity> =
        withContext(Dispatchers.IO) {
            corpusDao.getChunks(documentId)
        }

    suspend fun markCorpusDocumentTraining(documentId: String) =
        withContext(Dispatchers.IO) {
            corpusDao.updateTrainingState(
                documentId = documentId,
                status = "TRAINING",
                trainedAtEpochMs = null,
                completedPassCount = 0
            )
        }

    suspend fun markCorpusDocumentPartial(documentId: String) =
        withContext(Dispatchers.IO) {
            corpusDao.updateTrainingState(
                documentId = documentId,
                status = "PARTIAL",
                trainedAtEpochMs = null,
                completedPassCount = 0
            )
        }

    suspend fun markCorpusDocumentTrained(
        documentId: String,
        completedPassCount: Int
    ) = withContext(Dispatchers.IO) {
        require(completedPassCount >= 0) { "completedPassCount cannot be negative." }
        corpusDao.markTrainedIfNoChunkFailed(
            documentId = documentId,
            trainedAtEpochMs = System.currentTimeMillis(),
            completedPassCount = completedPassCount
        )
    }

    suspend fun markCorpusDocumentQueued(documentId: String) =
        withContext(Dispatchers.IO) {
            corpusDao.updateTrainingState(
                documentId = documentId,
                status = "READY",
                trainedAtEpochMs = null,
                completedPassCount = 0
            )
        }

    suspend fun deleteCorpusDocument(documentId: String): Boolean =
        withContext(Dispatchers.IO) {
            corpusDao.deleteDocument(documentId) > 0
        }

    // =========================================================================
    // Converters
    // =========================================================================

    private fun ChatMessageEntity.toDomain() = ChatMessage(
        id = id,
        isUser = isUser,
        text = text,
        thinkReasoning = thinkReasoning,
        attachedFileName = attachedFileName,
        attachedImageUri = attachedImageUri?.let { Uri.parse(it) },
        timestamp = timestamp
    )

    private fun ChatMessage.toEntity() = ChatMessageEntity(
        id = id,
        isUser = isUser,
        text = text,
        thinkReasoning = thinkReasoning,
        attachedFileName = attachedFileName,
        attachedImageUri = attachedImageUri?.toString(),
        timestamp = timestamp
    )

    private fun WorldMemoryEntity.toDomain() = WorldEntity(
        key = key,
        value = value,
        source = source,
        timestamp = timestamp
    )

    private fun WorldEntity.toEntity() = WorldMemoryEntity(
        key = key,
        value = value,
        source = source,
        timestamp = timestamp
    )

    private fun TrainingItemEntity.toDomain() = TrainingData(
        id = id,
        source = source,
        title = title,
        content = content,
        chunkIndex = chunkIndex,
        totalChunks = totalChunks,
        documentTitle = documentTitle,
        timestamp = timestamp,
        status = runCatching { TrainingStatus.valueOf(status) }.getOrDefault(TrainingStatus.QUEUED),
        finalLoss = finalLoss,
        stepDurationMs = stepDurationMs,
        documentId = documentId
    )

    private fun TrainingData.toEntity() = TrainingItemEntity(
        id = id,
        source = source,
        title = title,
        content = content,
        chunkIndex = chunkIndex,
        totalChunks = totalChunks,
        documentTitle = documentTitle,
        timestamp = timestamp,
        status = status.name,
        finalLoss = finalLoss,
        stepDurationMs = stepDurationMs,
        documentId = documentId
    )
}
