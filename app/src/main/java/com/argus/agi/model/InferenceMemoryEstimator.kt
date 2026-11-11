package com.argus.agi.model

import kotlin.math.ceil
import kotlin.math.max

/**
 * Conservative, metadata-aware footprint estimate for a future quantized inference loader.
 * It does not allocate memory and must not be treated as proof that a device can load a model.
 */
enum class InferenceQuantization(val estimatedBytesPerParameter: Double) {
    Q4_K_M(0.625),
    Q5_K_M(0.75),
    Q8_0(1.0625),
    F16(2.0)
}

data class InferenceModelShape(
    val parameterCount: Long,
    val layerCount: Int,
    val contextTokens: Int,
    val keyValueHeadCount: Int,
    val headDimension: Int,
    val keyValueBytesPerElement: Int = 2
) {
    init {
        require(parameterCount > 0L) { "parameterCount must be positive." }
        require(layerCount > 0) { "layerCount must be positive." }
        require(contextTokens > 0) { "contextTokens must be positive." }
        require(keyValueHeadCount > 0) { "keyValueHeadCount must be positive." }
        require(headDimension > 0) { "headDimension must be positive." }
        require(keyValueBytesPerElement > 0) {
            "keyValueBytesPerElement must be positive."
        }
    }
}

data class InferenceMemoryEstimate(
    val weightBytes: Long,
    val keyValueCacheBytes: Long?,
    val runtimeWorkspaceBytes: Long,
    val totalRequiredBytes: Long?
) {
    val hasCompleteModelMetadata: Boolean
        get() = keyValueCacheBytes != null && totalRequiredBytes != null
}

object InferenceMemoryEstimator {

    private const val MIB = 1024L * 1024L
    private const val MINIMUM_WORKSPACE_BYTES = 256L * MIB
    private const val WORKSPACE_WEIGHT_FRACTION = 0.05

    /**
     * Estimates resident weight storage, optional KV cache, and runtime workspace.
     * Pass null shape metadata only when the GGUF metadata has not yet been inspected.
     */
    fun estimate(
        parameterCount: Long,
        quantization: InferenceQuantization,
        shape: InferenceModelShape? = null
    ): InferenceMemoryEstimate {
        require(parameterCount > 0L) { "parameterCount must be positive." }
        if (shape != null) {
            require(shape.parameterCount == parameterCount) {
                "The model shape parameter count must match the requested parameter count."
            }
        }

        val weightBytes = safeCeilToLong(
            parameterCount.toDouble() * quantization.estimatedBytesPerParameter
        )
        val workspaceBytes = max(
            MINIMUM_WORKSPACE_BYTES,
            safeCeilToLong(weightBytes.toDouble() * WORKSPACE_WEIGHT_FRACTION)
        )

        val cacheBytes = shape?.let(::estimateKeyValueCacheBytes)
        val totalBytes = cacheBytes?.let {
            safeAdd(safeAdd(weightBytes, it), workspaceBytes)
        }

        return InferenceMemoryEstimate(
            weightBytes = weightBytes,
            keyValueCacheBytes = cacheBytes,
            runtimeWorkspaceBytes = workspaceBytes,
            totalRequiredBytes = totalBytes
        )
    }

    /**
     * Leaves at least the requested fraction of currently available system memory unused.
     * Returns null when KV metadata is missing, because a safe total cannot be established.
     */
    fun fitsWithinAvailableMemory(
        estimate: InferenceMemoryEstimate,
        availableRamBytes: Long,
        maximumUtilization: Double = 0.8
    ): Boolean? {
        require(availableRamBytes >= 0L) { "availableRamBytes cannot be negative." }
        require(maximumUtilization.isFinite() && maximumUtilization in 0.1..0.9) {
            "maximumUtilization must be between 0.1 and 0.9."
        }
        val totalRequiredBytes = estimate.totalRequiredBytes ?: return null
        val usableBudget = (availableRamBytes.toDouble() * maximumUtilization).toLong()
        return totalRequiredBytes <= usableBudget
    }

    private fun estimateKeyValueCacheBytes(shape: InferenceModelShape): Long {
        var bytes = shape.contextTokens.toLong()
        bytes = safeMultiply(bytes, shape.layerCount.toLong())
        bytes = safeMultiply(bytes, shape.keyValueHeadCount.toLong())
        bytes = safeMultiply(bytes, shape.headDimension.toLong())
        bytes = safeMultiply(bytes, 2L)
        return safeMultiply(bytes, shape.keyValueBytesPerElement.toLong())
    }

    private fun safeCeilToLong(value: Double): Long {
        if (!value.isFinite() || value >= Long.MAX_VALUE.toDouble()) return Long.MAX_VALUE
        return ceil(value).toLong().coerceAtLeast(0L)
    }

    private fun safeMultiply(first: Long, second: Long): Long {
        if (first == 0L || second == 0L) return 0L
        if (first > Long.MAX_VALUE / second) return Long.MAX_VALUE
        return first * second
    }

    private fun safeAdd(first: Long, second: Long): Long =
        if (first > Long.MAX_VALUE - second) Long.MAX_VALUE else first + second
}
