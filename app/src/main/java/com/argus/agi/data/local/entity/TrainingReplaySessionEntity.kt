package com.argus.agi.data.local.entity

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Immutable provenance for one fully materialized corpus replay plan.
 *
 * Queue rows carry [id] as replaySessionId and remain the authoritative per-item cursor.
 * This descriptor records enough information to audit the plan's source and ordering policy.
 */
@Entity(
    tableName = "training_replay_sessions",
    indices = [Index(value = ["createdAtEpochMs"])]
)
data class TrainingReplaySessionEntity(
    @PrimaryKey
    val id: String,
    val sourceSnapshotFingerprint: String,
    val seed: Long,
    val policy: String,
    val epochs: Int,
    val schedulerFormatVersion: Int,
    val scheduledDocumentCount: Int,
    val scheduledChunkCount: Int,
    val createdAtEpochMs: Long = System.currentTimeMillis()
) {
    init {
        require(id.isNotBlank()) { "Replay session ID cannot be blank." }
        require(sourceSnapshotFingerprint.length == 64 &&
            sourceSnapshotFingerprint.all { character ->
                character in '0'..'9' || character in 'a'..'f'
            }
        ) {
            "Replay source snapshot fingerprint must be a lowercase SHA-256 value."
        }
        require(policy.isNotBlank()) { "Replay policy cannot be blank." }
        require(epochs > 0) { "Replay epochs must be positive." }
        require(schedulerFormatVersion > 0) { "Scheduler format version must be positive." }
        require(scheduledDocumentCount > 0) { "Replay document count must be positive." }
        require(scheduledChunkCount > 0) { "Replay chunk count must be positive." }
    }
}
