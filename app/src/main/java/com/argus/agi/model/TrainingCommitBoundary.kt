package com.argus.agi.model

/**
 * Durable boundaries of the TrainingManager batch transaction.
 *
 * The observer is a no-op in production and can be used by persistence tests to
 * inject failures immediately after each durable transition.
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
