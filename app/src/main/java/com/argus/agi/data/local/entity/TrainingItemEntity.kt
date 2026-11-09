package com.argus.agi.data.local.entity

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "training_items")
data class TrainingItemEntity(
    @PrimaryKey
    val id: String,
    val source: String,
    val title: String,
    val content: String,
    val chunkIndex: Int = 1,
    val totalChunks: Int = 1,
    val documentTitle: String? = null,
    val timestamp: Long = System.currentTimeMillis(),
    val status: String = "QUEUED",
    val finalLoss: Float? = null,
    val stepDurationMs: Long = 0L
)
