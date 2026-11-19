package com.argus.agi.model

/**
 * Transaction boundaries in the TrainingManager batch commit/finalization path.
 *
 * The observer is a no-op in production and lets persistence tests inject failures
 * immediately after a mutation or durable transition.
 */
internal enum class TrainingCommitBoundary {
    PREPARED_JOURNAL_DURABLE,
    OPTIMIZER_UPDATE_APPLIED,
    COMMITTED_JOURNAL_DURABLE,
    CHECKPOINT_FILE_REPLACED,
    CHECKPOINT_BOUND_JOURNAL_DURABLE,
    ROOM_OPTIMIZER_STATE_COMMITTED,
    QUEUE_ITEM_FINALIZED
}
