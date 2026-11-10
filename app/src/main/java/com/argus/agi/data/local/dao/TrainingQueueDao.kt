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
        "SELECT DISTINCT documentId FROM training_items " +
            "WHERE documentId IS NOT NULL AND status IN ('QUEUED', 'TRAINING')"
    )
    suspend fun getActiveCorpusDocumentIds(): List<String>

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

    @Query("SELECT * FROM training_items WHERE id = :id LIMIT 1")
    suspend fun getItem(id: String): TrainingItemEntity?

    @Query("UPDATE training_items SET status = 'TRAINING' WHERE id = :id AND status = 'QUEUED'")
    suspend fun markTrainingStarted(id: String): Int

    /**
     * A process can die after a queue row is marked TRAINING but before completion is
     * persisted. Requeue those rows on startup. This is at-least-once recovery: a chunk
     * may be optimized again if the model checkpoint was saved before the process died.
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
