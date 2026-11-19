package com.argus.agi.data.local.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.argus.agi.data.local.entity.WorldMemoryEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface WorldMemoryDao {
    @Query("SELECT * FROM world_memories ORDER BY timestamp DESC")
    fun getAllMemories(): Flow<List<WorldMemoryEntity>>

    @Query("SELECT * FROM world_memories ORDER BY timestamp DESC")
    suspend fun getAllMemoriesSnapshot(): List<WorldMemoryEntity>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertMemory(memory: WorldMemoryEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertMemories(memories: List<WorldMemoryEntity>)

    @Query("DELETE FROM world_memories WHERE `key` = :key")
    suspend fun deleteMemory(key: String)

    @Query("DELETE FROM world_memories WHERE timestamp < :threshold")
    suspend fun deleteExpired(threshold: Long)

    @Query(
        "DELETE FROM world_memories WHERE `key` IN ('active_screen_app', 'last_screen_content') " +
            "OR source IN ('Accessibility Perception', 'Screen OCR/Text Parser')"
    )
    suspend fun deleteScreenContextMemories(): Int

    @Query("DELETE FROM world_memories")
    suspend fun clearAll()
}
