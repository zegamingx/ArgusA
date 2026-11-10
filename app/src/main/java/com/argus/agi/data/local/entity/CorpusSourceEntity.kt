package com.argus.agi.data.local.entity

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Provenance record for one origin of a deduplicated corpus document.
 * A single content fingerprint may therefore retain multiple source references.
 */
@Entity(
    tableName = "corpus_sources",
    foreignKeys = [
        ForeignKey(
            entity = CorpusDocumentEntity::class,
            parentColumns = ["id"],
            childColumns = ["documentId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index(value = ["documentId"])]
)
data class CorpusSourceEntity(
    @PrimaryKey
    val id: String,
    val documentId: String,
    val source: String,
    val canonicalUrl: String?,
    val language: String?,
    val rightsStatus: String,
    val importedAtEpochMs: Long
)
