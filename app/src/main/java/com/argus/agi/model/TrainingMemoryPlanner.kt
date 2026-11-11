package com.argus.agi.model

/**
 * Lower-bound accounting for full-parameter FP32 training with Adam.
 *
 * The 16-byte baseline accounts for FP32 parameters, FP32 gradients, and two FP32 Adam
 * moment buffers. It deliberately excludes activations, temporary tensors, tokenizer data,
 * allocator fragmentation, checkpoint buffers, and runtime overhead. It is therefore a
 * lower bound, never a "fits in RAM" guarantee.
 */
object TrainingMemoryPlanner {

    const val BASELINE_BYTES_PER_PARAMETER: Long = 16L

    /**
     * Adds one FP32 gradient-accumulation buffer to the ordinary weights, gradients,
     * and two Adam moments used by the current serial micro-batch implementation.
     */
    const val ACCUMULATED_BYTES_PER_PARAMETER: Long = 20L

    fun minimumAccumulatedTrainingStateBytes(parameterCount: Long): Long {
        require(parameterCount >= 0L) { "parameterCount cannot be negative." }
        if (parameterCount == 0L) return 0L
        if (parameterCount > Long.MAX_VALUE / ACCUMULATED_BYTES_PER_PARAMETER) {
            return Long.MAX_VALUE
        }
        return parameterCount * ACCUMULATED_BYTES_PER_PARAMETER
    }

    fun minimumTrainingStateBytes(parameterCount: Long): Long {
        require(parameterCount >= 0L) { "parameterCount cannot be negative." }
        if (parameterCount == 0L) return 0L
        if (parameterCount > Long.MAX_VALUE / BASELINE_BYTES_PER_PARAMETER) {
            return Long.MAX_VALUE
        }
        return parameterCount * BASELINE_BYTES_PER_PARAMETER
    }
}
