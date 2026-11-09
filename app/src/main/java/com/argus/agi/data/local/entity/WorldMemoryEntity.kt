package com.argus.agi.data.local.entity

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "world_memories")
data class WorldMemoryEntity(
    @PrimaryKey
    val key: String,
    val value: String,
    val source: String,
    val timestamp: Long = System.currentTimeMillis()
)
