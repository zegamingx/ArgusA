package com.argus.agi.data.local.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.argus.agi.data.local.entity.CorpusChunkEntity
import com.argus.agi.data.local.entity.CorpusDocumentEntity
import com.argus.agi.data.local.entity.CorpusSourceEntity
import kotlinx.coroutines.flow.Flow

/**
 * Bounded projection used by tokenizer preparation. Never hydrate the complete corpus row.
 */
data class TokenizerCorpusSample(
    val chunkIndex: Int,
    val content: String
)

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
            "language = :language, rightsStatus = CASE " +
            "WHEN rightsStatus = 'LICENSE_REVIEWED' THEN rightsStatus " +
            "WHEN :rightsStatus = 'LICENSE_REVIEWED' THEN :rightsStatus " +
            "WHEN rightsStatus = 'USER_DECLARED' AND :rightsStatus = 'UNVERIFIED' THEN rightsStatus " +
            "ELSE :rightsStatus END, " +
            "importedAtEpochMs = :importedAtEpochMs, " +
            "licenseIdentifier = COALESCE(:licenseIdentifier, licenseIdentifier), " +
            "licenseName = COALESCE(:licenseName, licenseName), " +
            "licenseUrl = COALESCE(:licenseUrl, licenseUrl), " +
            "attributionText = COALESCE(:attributionText, attributionText), " +
            "rightsEvidenceUrl = COALESCE(:rightsEvidenceUrl, rightsEvidenceUrl), " +
            "userDeclarationText = COALESCE(:userDeclarationText, userDeclarationText), " +
            "userDeclaredAtEpochMs = COALESCE(:userDeclaredAtEpochMs, userDeclaredAtEpochMs), " +
            "reviewedAtEpochMs = COALESCE(:reviewedAtEpochMs, reviewedAtEpochMs) " +
            "WHERE id = :sourceId"
    )
    suspend fun updateSourceMetadata(
        sourceId: String,
        source: String,
        canonicalUrl: String?,
        language: String?,
        rightsStatus: String,
        importedAtEpochMs: Long,
        licenseIdentifier: String?,
        licenseName: String?,
        licenseUrl: String?,
        attributionText: String?,
        rightsEvidenceUrl: String?,
        userDeclarationText: String?,
        userDeclaredAtEpochMs: Long?,
        reviewedAtEpochMs: Long?
    ): Int

    @Query(
        """
        UPDATE corpus_documents
        SET rightsStatus = CASE
            WHEN EXISTS (
                SELECT 1 FROM corpus_sources
                WHERE documentId = :documentId AND rightsStatus = 'LICENSE_REVIEWED'
            ) THEN 'LICENSE_REVIEWED'
            WHEN EXISTS (
                SELECT 1 FROM corpus_sources
                WHERE documentId = :documentId AND rightsStatus = 'USER_DECLARED'
            ) THEN 'USER_DECLARED'
            ELSE 'UNVERIFIED'
        END
        WHERE id = :documentId
        """
    )
    suspend fun refreshDocumentRightsStatus(documentId: String): Int

    @Query("SELECT * FROM corpus_sources WHERE documentId = :documentId ORDER BY importedAtEpochMs ASC")
    suspend fun getSources(documentId: String): List<CorpusSourceEntity>

    @Query(
        "SELECT id FROM corpus_documents WHERE source LIKE :sourcePattern " +
            "UNION SELECT DISTINCT documentId FROM corpus_sources WHERE source LIKE :sourcePattern"
    )
    suspend fun getDocumentIdsBySourcePattern(sourcePattern: String): List<String>

    @Query("DELETE FROM corpus_sources WHERE documentId = :documentId AND source LIKE :sourcePattern")
    suspend fun deleteSourcesBySourcePattern(
        documentId: String,
        sourcePattern: String
    ): Int

    @Query(
        "UPDATE corpus_documents SET source = :source, canonicalUrl = :canonicalUrl, " +
            "language = :language WHERE id = :documentId"
    )
    suspend fun updateDocumentPrimarySource(
        documentId: String,
        source: String,
        canonicalUrl: String?,
        language: String?
    ): Int

    @Query("SELECT * FROM corpus_documents WHERE id = :documentId LIMIT 1")
    suspend fun getDocument(documentId: String): CorpusDocumentEntity?

    @Query("SELECT id FROM corpus_documents WHERE status = 'IMPORTING' ORDER BY createdAtEpochMs ASC")
    suspend fun getImportingDocumentIds(): List<String>

    @Query("UPDATE corpus_documents SET status = 'READY' WHERE id = :documentId AND status = 'IMPORTING'")
    suspend fun completeStreamedImport(documentId: String): Int

    @Query("SELECT * FROM corpus_chunks WHERE documentId = :documentId AND chunkIndex = :chunkIndex LIMIT 1")
    suspend fun getChunk(documentId: String, chunkIndex: Int): CorpusChunkEntity?


    @Query("SELECT * FROM corpus_documents WHERE contentFingerprint = :fingerprint LIMIT 1")
    suspend fun getDocumentByFingerprint(fingerprint: String): CorpusDocumentEntity?

    @Query("SELECT * FROM corpus_documents ORDER BY createdAtEpochMs DESC LIMIT :limit")
    fun observeDocuments(limit: Int): Flow<List<CorpusDocumentEntity>>

    @Query("SELECT * FROM corpus_documents ORDER BY createdAtEpochMs DESC LIMIT :limit")
    suspend fun getDocuments(limit: Int): List<CorpusDocumentEntity>

    @Query("SELECT COUNT(*) FROM corpus_documents")
    suspend fun getDocumentCount(): Int

    @Query(
        "SELECT * FROM corpus_documents " +
            "WHERE datasetSplit = 'VALIDATION' AND validationText IS NOT NULL " +
            "ORDER BY createdAtEpochMs ASC LIMIT :limit"
    )
    suspend fun getValidationDocuments(limit: Int): List<CorpusDocumentEntity>

    @Query(
        "SELECT * FROM corpus_documents " +
            "WHERE datasetSplit = 'TRAIN' AND chunkCount > 0 " +
            "ORDER BY createdAtEpochMs ASC, id ASC LIMIT :limit"
    )
    suspend fun getTrainingDocuments(limit: Int): List<CorpusDocumentEntity>

    @Query("SELECT * FROM corpus_chunks WHERE documentId = :documentId ORDER BY chunkIndex ASC")
    suspend fun getChunks(documentId: String): List<CorpusChunkEntity>

    @Query(
        """
        SELECT chunkIndex, substr(content, 1, :maxCharacters) AS content
        FROM corpus_chunks
        WHERE documentId = :documentId AND chunkIndex > :afterChunkIndex
        ORDER BY chunkIndex ASC
        LIMIT 1
        """
    )
    suspend fun getNextTokenizerCorpusSample(
        documentId: String,
        afterChunkIndex: Int,
        maxCharacters: Int
    ): TokenizerCorpusSample?

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
        "UPDATE corpus_documents SET status = 'TRAINING', " +
            "lastTrainedAtEpochMs = NULL, completedPassCount = 0 " +
            "WHERE id = :documentId AND status != 'PARTIAL'"
    )
    suspend fun markTrainingIfNotPartial(documentId: String): Int

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
