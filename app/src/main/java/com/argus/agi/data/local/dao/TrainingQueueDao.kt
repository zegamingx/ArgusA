package com.argus.agi.data.local.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.argus.agi.data.local.entity.TrainingItemEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface TrainingQueueDao {
    @Query("SELECT * FROM training_items WHERE status = 'QUEUED' ORDER BY timestamp ASC")
    fun getPendingQueue(): Flow<List<TrainingItemEntity>>

    @Query("SELECT * FROM training_items WHERE status = 'QUEUED' ORDER BY timestamp ASC")
    suspend fun getPendingQueueSnapshot(): List<TrainingItemEntity>

    @Query(
        "SELECT * FROM training_items WHERE status = 'OPTIMIZER_COMMITTED' ORDER BY timestamp ASC"
    )
    suspend fun getOptimizerCommittedItemsSnapshot(): List<TrainingItemEntity>

    @Query(
        "SELECT DISTINCT documentId FROM training_items " +
            "WHERE documentId IS NOT NULL AND status IN ('QUEUED', 'TRAINING', 'OPTIMIZER_COMMITTED')"
    )
    suspend fun getActiveCorpusDocumentIds(): List<String>

    @Query(
        "SELECT * FROM training_items " +
            "WHERE documentId IS NOT NULL AND status IN ('QUEUED', 'TRAINING', 'OPTIMIZER_COMMITTED')"
    )
    suspend fun getActiveCorpusTrainingItems(): List<TrainingItemEntity>

    @Query("SELECT * FROM training_items WHERE status = 'COMPLETED' ORDER BY timestamp DESC LIMIT :limit")
    fun getCompletedItems(limit: Int = 50): Flow<List<TrainingItemEntity>>

    @Query("SELECT * FROM training_items WHERE status = 'COMPLETED' ORDER BY timestamp DESC LIMIT :limit")
    suspend fun getCompletedItemsSnapshot(limit: Int = 50): List<TrainingItemEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertItem(item: TrainingItemEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertItems(items: List<TrainingItemEntity>)

    @Query("UPDATE training_items SET status = :status, finalLoss = :loss, stepDurationMs = :duration WHERE id = :id")
    suspend fun updateStatus(id: String, status: String, loss: Float?, duration: Long)

    @Query(
        "UPDATE training_items SET status = 'OPTIMIZER_COMMITTED', " +
            "finalLoss = :loss, stepDurationMs = :duration " +
            "WHERE id = :id AND status IN ('TRAINING', 'QUEUED', 'OPTIMIZER_COMMITTED')"
    )
    suspend fun markOptimizerCommitted(id: String, loss: Float?, duration: Long): Int

    @Query(
        "UPDATE training_items SET status = 'COMPLETED', finalLoss = :loss, " +
            "stepDurationMs = :duration WHERE id = :id AND status = 'OPTIMIZER_COMMITTED'"
    )
    suspend fun markOptimizerCommittedCompleted(
        id: String,
        loss: Float,
        duration: Long
    ): Int

    @Query("SELECT * FROM training_items WHERE id = :id LIMIT 1")
    suspend fun getItem(id: String): TrainingItemEntity?

    @Query("UPDATE training_items SET status = 'TRAINING' WHERE id = :id AND status = 'QUEUED'")
    suspend fun markTrainingStarted(id: String): Int

    /**
     * TrainingManager reconciles its checksummed optimizer-commit journal against the
     * model checkpoint before calling this method. Remaining TRAINING rows are therefore
     * pre-commit or legacy rows and can safely return to QUEUED for replay.
     */
    @Query("UPDATE training_items SET status = 'QUEUED' WHERE status = 'TRAINING'")
    suspend fun recoverInterruptedTrainingItems(): Int

    @Query("DELETE FROM training_items WHERE id = :id")
    suspend fun deleteItem(id: String)

    @Query("DELETE FROM training_items WHERE status = 'QUEUED'")
    suspend fun clearPendingQueue()

    @Query("DELETE FROM training_items")
    suspend fun clearAll()
}
