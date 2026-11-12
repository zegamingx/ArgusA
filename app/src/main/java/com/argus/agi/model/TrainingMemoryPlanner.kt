package com.argus.agi.model

/**
 * Lower-bound accounting for FP16 model parameters with FP32 AdamW state.
 *
 * The 14-byte baseline accounts for a 2-byte FP16 parameter, a 4-byte gradient,
 * and two 4-byte FP32 Adam moments. Serial micro-batch accumulation adds one
 * further 4-byte FP32 gradient buffer, bringing the state floor to 18 bytes per
 * parameter. Activations, scratch tensors, tokenizer data, allocator fragmentation,
 * checkpoint buffers, and runtime overhead are additional.
 */
object TrainingMemoryPlanner {

    const val BASELINE_BYTES_PER_PARAMETER: Long = 14L

    /** Adds one FP32 gradient-accumulation buffer to the FP16/FP32 training state. */
    const val ACCUMULATED_BYTES_PER_PARAMETER: Long = 18L

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
