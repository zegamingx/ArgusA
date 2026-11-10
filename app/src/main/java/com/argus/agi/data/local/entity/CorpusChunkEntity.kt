package com.argus.agi.data.local.entity

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Normalized training text retained independently from transient queue state.
 */
@Entity(
    tableName = "corpus_chunks",
    foreignKeys = [
        ForeignKey(
            entity = CorpusDocumentEntity::class,
            parentColumns = ["id"],
            childColumns = ["documentId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [
        Index(value = ["documentId"]),
        Index(value = ["documentId", "chunkIndex"], unique = true)
    ]
)
data class CorpusChunkEntity(
    @PrimaryKey
    val id: String,
    val documentId: String,
    val chunkIndex: Int,
    val content: String,
    val characterCount: Int
)
