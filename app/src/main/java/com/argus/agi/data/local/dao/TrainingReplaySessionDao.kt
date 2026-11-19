package com.argus.agi.data.local.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import com.argus.agi.data.local.entity.TrainingReplaySessionEntity

@Dao
interface TrainingReplaySessionDao {

    @Insert(onConflict = OnConflictStrategy.ABORT)
    suspend fun insert(session: TrainingReplaySessionEntity)

    @Query("SELECT * FROM training_replay_sessions WHERE id = :sessionId LIMIT 1")
    suspend fun getSession(sessionId: String): TrainingReplaySessionEntity?
}
