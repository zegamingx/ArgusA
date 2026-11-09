package com.argus.agi.data.local.entity

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "chat_messages")
data class ChatMessageEntity(
    @PrimaryKey
    val id: String,
    val isUser: Boolean,
    val text: String,
    val thinkReasoning: String? = null,
    val attachedFileName: String? = null,
    val attachedImageUri: String? = null,
    val timestamp: Long = System.currentTimeMillis()
)
