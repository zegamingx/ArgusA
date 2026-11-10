package com.argus.agi.data.local.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.argus.agi.data.local.entity.CorpusChunkEntity
import com.argus.agi.data.local.entity.CorpusDocumentEntity
import com.argus.agi.data.local.entity.CorpusSourceEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface CorpusDao {

    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertDocument(document: CorpusDocumentEntity): Long

    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insertChunks(chunks: List<CorpusChunkEntity>)

    @Insert(onConflict = OnConflictStrategy.IGNORE)
    suspend fun insertSource(source: CorpusSourceEntity): Long

    @Query(
        "UPDATE corpus_sources SET source = :source, canonicalUrl = :canonicalUrl, " +
            "language = :language, rightsStatus = :rightsStatus, " +
            "importedAtEpochMs = :importedAtEpochMs WHERE id = :sourceId"
    )
    suspend fun updateSourceMetadata(
        sourceId: String,
        source: String,
        canonicalUrl: String?,
        language: String?,
        rightsStatus: String,
        importedAtEpochMs: Long
    ): Int

    @Query("SELECT * FROM corpus_sources WHERE documentId = :documentId ORDER BY importedAtEpochMs ASC")
    suspend fun getSources(documentId: String): List<CorpusSourceEntity>

    @Query("SELECT * FROM corpus_documents WHERE id = :documentId LIMIT 1")
    suspend fun getDocument(documentId: String): CorpusDocumentEntity?

    @Query("SELECT * FROM corpus_documents WHERE contentFingerprint = :fingerprint LIMIT 1")
    suspend fun getDocumentByFingerprint(fingerprint: String): CorpusDocumentEntity?

    @Query("SELECT * FROM corpus_documents ORDER BY createdAtEpochMs DESC LIMIT :limit")
    fun observeDocuments(limit: Int): Flow<List<CorpusDocumentEntity>>

    @Query("SELECT * FROM corpus_documents ORDER BY createdAtEpochMs DESC LIMIT :limit")
    suspend fun getDocuments(limit: Int): List<CorpusDocumentEntity>

    @Query("SELECT COUNT(*) FROM corpus_documents")
    suspend fun getDocumentCount(): Int

    @Query("SELECT * FROM corpus_chunks WHERE documentId = :documentId ORDER BY chunkIndex ASC")
    suspend fun getChunks(documentId: String): List<CorpusChunkEntity>

    @Query(
        "UPDATE corpus_documents SET status = :status, " +
            "lastTrainedAtEpochMs = :trainedAtEpochMs, " +
            "completedPassCount = :completedPassCount " +
            "WHERE id = :documentId"
    )
    suspend fun updateTrainingState(
        documentId: String,
        status: String,
        trainedAtEpochMs: Long?,
        completedPassCount: Int
    ): Int

    @Query(
        "UPDATE corpus_documents SET status = 'TRAINED', " +
            "lastTrainedAtEpochMs = :trainedAtEpochMs, " +
            "completedPassCount = :completedPassCount " +
            "WHERE id = :documentId AND status != 'PARTIAL'"
    )
    suspend fun markTrainedIfNoChunkFailed(
        documentId: String,
        trainedAtEpochMs: Long,
        completedPassCount: Int
    ): Int

    @Query("DELETE FROM corpus_documents WHERE id = :documentId")
    suspend fun deleteDocument(documentId: String): Int
}
