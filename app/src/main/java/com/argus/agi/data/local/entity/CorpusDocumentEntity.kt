package com.argus.agi.data.local.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * A durable, device-local source document. Training queue rows are disposable work items;
 * this record and its CorpusChunkEntity rows are the retained corpus used for replay.
 */
@Entity(
    tableName = "corpus_documents",
    indices = [
        Index(value = ["contentFingerprint"], unique = true),
        Index(value = ["createdAtEpochMs"])
    ]
)
data class CorpusDocumentEntity(
    @PrimaryKey
    val id: String,
    val contentFingerprint: String,
    val fingerprintAlgorithm: String,
    val source: String,
    val title: String,
    val language: String?,
    val canonicalUrl: String?,
    val characterCount: Int,
    val chunkCount: Int,
    val datasetSplit: String = "TRAIN",
    val validationText: String? = null,
    val plannedEpochs: Int,
    val rightsStatus: String,
    val status: String,
    val createdAtEpochMs: Long,
    val lastTrainedAtEpochMs: Long?,
    val completedPassCount: Int
)
