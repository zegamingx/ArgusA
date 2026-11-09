package com.argus.agi.data.repository

import android.content.Context
import android.net.Uri
import com.argus.agi.data.local.ArgusDatabase
import com.argus.agi.data.local.entity.ChatMessageEntity
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

class ArgusRepository(context: Context) {

    private val db = ArgusDatabase.getInstance(context)
    private val chatMessageDao = db.chatMessageDao()
    private val worldMemoryDao = db.worldMemoryDao()
    private val trainingQueueDao = db.trainingQueueDao()

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
        stepDurationMs = stepDurationMs
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
        stepDurationMs = stepDurationMs
    )
}
