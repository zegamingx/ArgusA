package com.argus.agi.data.repository

import android.content.Context
import android.net.Uri
import androidx.room.withTransaction
import com.argus.agi.data.local.ArgusDatabase
import com.argus.agi.data.local.entity.ChatMessageEntity
import com.argus.agi.data.local.entity.CorpusChunkEntity
import com.argus.agi.data.local.entity.CorpusDocumentEntity
import com.argus.agi.data.local.entity.CorpusSourceEntity
import com.argus.agi.data.local.entity.CorpusSourceProvenance
import com.argus.agi.data.local.entity.TrainingItemEntity
import com.argus.agi.data.local.entity.TrainingReplaySessionEntity
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
    private val trainingReplaySessionDao = db.trainingReplaySessionDao()
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

    suspend fun getOptimizerCommittedTrainingItemsSnapshot(): List<TrainingData> =
        withContext(Dispatchers.IO) {
            trainingQueueDao.getOptimizerCommittedItemsSnapshot().map { it.toDomain() }
        }

    /**
     * Persists the optimizer commit before queue finalization. Room commits the entire
     * micro-batch atomically so an interrupted finalizer can never retrain a committed item.
     */
    suspend fun markTrainingItemsOptimizerCommitted(items: List<TrainingData>) =
        withContext(Dispatchers.IO) {
            require(items.isNotEmpty() && items.size <= 8) {
                "An optimizer commit must contain between one and eight training items."
            }
            require(items.all { it.status == TrainingStatus.OPTIMIZER_COMMITTED }) {
                "Every committed training item must carry OPTIMIZER_COMMITTED status."
            }
            db.withTransaction {
                for (item in items) {
                    val existing = trainingQueueDao.getItem(item.id)
                        ?: throw IllegalStateException(
                            "Committed training item '${item.id}' is missing from Room."
                        )
                    if (existing.status == "COMPLETED" || existing.status == "FAILED") {
                        continue
                    }
                    val updated = trainingQueueDao.markOptimizerCommitted(
                        id = item.id,
                        loss = item.finalLoss,
                        duration = item.stepDurationMs
                    )
                    check(updated == 1) {
                        "Training item '${item.id}' could not be persisted as optimizer-committed."
                    }
                }
            }
        }

    suspend fun getActiveCorpusTrainingDocumentIds(): Set<String> =
        withContext(Dispatchers.IO) {
            trainingQueueDao.getActiveCorpusDocumentIds().toSet()
        }

    suspend fun saveTrainingItem(item: TrainingData) = withContext(Dispatchers.IO) {
        db.withTransaction {
            validateTrainingDocumentReferences(listOf(item))
            trainingQueueDao.insertItem(item.toEntity())
        }
    }

    suspend fun saveTrainingItems(items: List<TrainingData>) = withContext(Dispatchers.IO) {
        db.withTransaction {
            validateTrainingDocumentReferences(items)
            trainingQueueDao.insertItems(items.map { it.toEntity() })
        }
    }

    /**
     * Persists a materialized corpus replay as one database transaction.
     * Session provenance, every queue row, and corpus READY state either all commit or none do.
     */
    suspend fun saveCorpusReplaySession(
        session: TrainingReplaySessionEntity,
        items: List<TrainingData>
    ) = withContext(Dispatchers.IO) {
        require(items.isNotEmpty()) {
            "A corpus replay session must contain queue items."
        }
        require(session.scheduledChunkCount == items.size) {
            "Replay session chunk count does not match its queue item count."
        }
        require(items.all { it.replaySessionId == session.id }) {
            "Every replay queue item must reference the session being persisted."
        }
        val documentIds = items.mapNotNull(TrainingData::documentId).distinct()
        require(documentIds.size == session.scheduledDocumentCount) {
            "Replay session document count does not match its queue items."
        }
        require(items.all { it.documentId != null }) {
            "Corpus replay queue items must reference retained corpus documents."
        }

        db.withTransaction {
            validateTrainingDocumentReferences(items)
            trainingReplaySessionDao.insert(session)
            trainingQueueDao.insertItems(items.map { it.toEntity() })
            documentIds.forEach { documentId ->
                check(
                    corpusDao.updateTrainingState(
                        documentId = documentId,
                        status = "READY",
                        trainedAtEpochMs = null,
                        completedPassCount = 0
                    ) == 1
                ) {
                    "Corpus document could not be marked READY for replay: $documentId"
                }
            }
        }
    }

    /**
     * A persisted queue row may reference a retained TRAIN document only.
     * Validation documents are never eligible for optimizer updates, even if a caller
     * bypasses TrainingManager and writes directly through the repository.
     */
    private suspend fun validateTrainingDocumentReferences(items: List<TrainingData>) {
        items.mapNotNull { it.documentId }.distinct().forEach { documentId ->
            val document = corpusDao.getDocument(documentId)
                ?: throw IllegalArgumentException(
                    "Training queue references an unknown corpus document: $documentId"
                )
            require(document.datasetSplit == "TRAIN") {
                "Validation-only corpus documents cannot be inserted into the training queue."
            }
        }
    }

    suspend fun markTrainingCompleted(
        id: String,
        loss: Float,
        durationMs: Long
    ): Boolean = withContext(Dispatchers.IO) {
        val updated = trainingQueueDao.markOptimizerCommittedCompleted(
            id = id,
            loss = loss,
            duration = durationMs
        )
        updated == 1
    }

    suspend fun markTrainingStarted(id: String): Boolean = withContext(Dispatchers.IO) {
        trainingQueueDao.markTrainingStarted(id) > 0
    }

    suspend fun getTrainingItem(id: String): TrainingData? = withContext(Dispatchers.IO) {
        trainingQueueDao.getItem(id)?.toDomain()
    }

    suspend fun recoverInterruptedTrainingItems(): Int = withContext(Dispatchers.IO) {
        trainingQueueDao.recoverInterruptedTrainingItems()
    }

    suspend fun markTrainingFailed(id: String) = withContext(Dispatchers.IO) {
        trainingQueueDao.updateStatus(id, "FAILED", null, 0L)
    }

    suspend fun clearPendingQueue() = withContext(Dispatchers.IO) {
        trainingQueueDao.clearPendingQueue()
    }

    /**
     * Clears waiting work and records corpus documents whose replay was interrupted.
     * A sole active final chunk is allowed to finish and mark its document TRAINED.
     */
    suspend fun clearPendingQueueAndMarkInterruptedCorpusDocumentsPartial(): Set<String> =
        withContext(Dispatchers.IO) {
            db.withTransaction {
                val activeItems = trainingQueueDao.getActiveCorpusTrainingItems()
                val itemsByDocument = activeItems
                    .filter { it.documentId != null }
                    .groupBy { requireNotNull(it.documentId) }

                val partialDocumentIds = itemsByDocument
                    .filterValues { documentItems ->
                        val canFinishDocument = documentItems.size == 1 &&
                            documentItems.single().status in setOf("TRAINING", "OPTIMIZER_COMMITTED") &&
                            documentItems.single().chunkIndex >= documentItems.single().totalChunks
                        !canFinishDocument
                    }
                    .keys
                    .toSet()

                trainingQueueDao.clearPendingQueue()
                partialDocumentIds.forEach { documentId ->
                    corpusDao.updateTrainingState(
                        documentId = documentId,
                        status = "PARTIAL",
                        trainedAtEpochMs = null,
                        completedPassCount = 0
                    )
                }
                partialDocumentIds
            }
        }

    // =========================================================================
    // Persistent Local Corpus
    // =========================================================================

    val corpusDocuments: Flow<List<CorpusDocumentEntity>> =
        corpusDao.observeDocuments(limit = 500)

    suspend fun storeCorpusDocument(
        document: CorpusDocumentEntity,
        chunks: List<CorpusChunkEntity>,
        sourceProvenance: CorpusSourceProvenance = CorpusSourceProvenance()
    ): Boolean = withContext(Dispatchers.IO) {
        sourceProvenance.validateFor(document.rightsStatus)
        val isWebContent = document.source.startsWith("web_article", ignoreCase = true) ||
            document.canonicalUrl != null
        require(!isWebContent || document.rightsStatus != "UNVERIFIED") {
            "Web article content must be reviewed and explicitly declared before it can be stored."
        }
        require(document.datasetSplit == "TRAIN" || document.datasetSplit == "VALIDATION") {
            "Unsupported corpus dataset split: ${document.datasetSplit}"
        }
        require(chunks.size == document.chunkCount) {
            "Corpus chunk count does not match document metadata."
        }
        require(chunks.all { it.documentId == document.id }) {
            "Every corpus chunk must reference the document being stored."
        }
        when (document.datasetSplit) {
            "TRAIN" -> require(document.validationText == null && document.chunkCount > 0) {
                "Training documents must contain retained chunks and no validation payload."
            }
            "VALIDATION" -> require(
                !document.validationText.isNullOrBlank() &&
                    document.chunkCount == 0 &&
                    chunks.isEmpty()
            ) {
                "Validation documents must retain source text and must not contain training chunks."
            }
        }

        db.withTransaction {
            val insertedRowId = corpusDao.insertDocument(document)
            val storedDocumentId = if (insertedRowId == -1L) {
                val existing = corpusDao.getDocumentByFingerprint(document.contentFingerprint)
                    ?: error("Corpus fingerprint conflict has no matching document.")
                require(existing.datasetSplit == document.datasetSplit) {
                    "Identical content already belongs to the ${existing.datasetSplit} corpus; cross-split reuse is blocked to prevent leakage."
                }
                existing.id
            } else {
                if (chunks.isNotEmpty()) corpusDao.insertChunks(chunks)
                document.id
            }

            val sourceIdentity = document.canonicalUrl ?: document.source
            val sourceId = sha256Utf16Be("${storedDocumentId}\u0000$sourceIdentity")
            corpusDao.insertSource(
                CorpusSourceEntity(
                    id = sourceId,
                    documentId = storedDocumentId,
                    source = document.source,
                    canonicalUrl = document.canonicalUrl,
                    language = document.language,
                    rightsStatus = document.rightsStatus,
                    importedAtEpochMs = document.createdAtEpochMs,
                    licenseIdentifier = sourceProvenance.licenseIdentifier,
                    licenseName = sourceProvenance.licenseName,
                    licenseUrl = sourceProvenance.licenseUrl,
                    attributionText = sourceProvenance.attributionText,
                    rightsEvidenceUrl = sourceProvenance.rightsEvidenceUrl,
                    userDeclarationText = sourceProvenance.userDeclarationText,
                    userDeclaredAtEpochMs = sourceProvenance.userDeclaredAtEpochMs,
                    reviewedAtEpochMs = sourceProvenance.reviewedAtEpochMs
                )
            )
            corpusDao.updateSourceMetadata(
                sourceId = sourceId,
                source = document.source,
                canonicalUrl = document.canonicalUrl,
                language = document.language,
                rightsStatus = document.rightsStatus,
                importedAtEpochMs = document.createdAtEpochMs,
                licenseIdentifier = sourceProvenance.licenseIdentifier,
                licenseName = sourceProvenance.licenseName,
                licenseUrl = sourceProvenance.licenseUrl,
                attributionText = sourceProvenance.attributionText,
                rightsEvidenceUrl = sourceProvenance.rightsEvidenceUrl,
                userDeclarationText = sourceProvenance.userDeclarationText,
                userDeclaredAtEpochMs = sourceProvenance.userDeclaredAtEpochMs,
                reviewedAtEpochMs = sourceProvenance.reviewedAtEpochMs
            )
            check(corpusDao.refreshDocumentRightsStatus(storedDocumentId) == 1) {
                "The corpus document's aggregate rights status could not be synchronized with its sources."
            }
            insertedRowId != -1L
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

    suspend fun getValidationCorpusDocuments(limit: Int = 500): List<CorpusDocumentEntity> =
        withContext(Dispatchers.IO) {
            require(limit in 1..500) { "Validation document limit must be in 1..500." }
            corpusDao.getValidationDocuments(limit)
        }

    suspend fun getTrainingCorpusDocuments(limit: Int = 500): List<CorpusDocumentEntity> =
        withContext(Dispatchers.IO) {
            require(limit in 1..500) { "Training document limit must be in 1..500." }
            corpusDao.getTrainingDocuments(limit)
        }

    suspend fun getCorpusChunks(documentId: String): List<CorpusChunkEntity> =
        withContext(Dispatchers.IO) {
            corpusDao.getChunks(documentId)
        }

    suspend fun getTokenizerCorpusSample(documentId: String, limit: Int): List<String> =
        withContext(Dispatchers.IO) {
            require(limit in 1..512) { "Tokenizer corpus sample limit must be in 1..512." }
            corpusDao.getTokenizerCorpusSample(documentId, limit)
        }

    suspend fun getCorpusSources(documentId: String): List<CorpusSourceEntity> =
        withContext(Dispatchers.IO) {
            corpusDao.getSources(documentId)
        }

    suspend fun markCorpusDocumentTraining(documentId: String) =
        withContext(Dispatchers.IO) {
            corpusDao.markTrainingIfNotPartial(documentId)
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

    suspend fun markCorpusDocumentsQueued(documentIds: Collection<String>) =
        withContext(Dispatchers.IO) {
            val stableIds = documentIds.distinct()
            db.withTransaction {
                stableIds.forEach { documentId ->
                    corpusDao.updateTrainingState(
                        documentId = documentId,
                        status = "READY",
                        trainedAtEpochMs = null,
                        completedPassCount = 0
                    )
                }
            }
        }

    suspend fun deleteCorpusDocument(documentId: String): Boolean =
        withContext(Dispatchers.IO) {
            corpusDao.deleteDocument(documentId) > 0
        }

    private fun sha256Utf16Be(value: String): String {
        val digest = java.security.MessageDigest.getInstance("SHA-256")
            .digest(value.toByteArray(Charsets.UTF_16BE))
        val alphabet = "0123456789abcdef"
        return buildString(digest.size * 2) {
            digest.forEach { byte ->
                val item = byte.toInt() and 0xFF
                append(alphabet[item ushr 4])
                append(alphabet[item and 0x0F])
            }
        }
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
        documentId = documentId,
        replaySessionId = replaySessionId
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
        documentId = documentId,
        replaySessionId = replaySessionId
    )
}
