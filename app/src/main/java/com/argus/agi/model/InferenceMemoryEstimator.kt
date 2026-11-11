package com.argus.agi.model

import kotlin.math.ceil
import kotlin.math.max

/**
 * Conservative, metadata-aware footprint estimate for a future quantized inference loader.
 * It does not allocate memory and must not be treated as proof that a device can load a model.
 */
enum class ArgusWeightPrecision(val estimatedBytesPerParameter: Double) {
    INT4(0.5),
    INT5(0.625),
    INT8(1.0),
    F16(2.0)
}

data class InferenceModelShape(
    val parameterCount: Long,
    val layerCount: Int,
    val contextTokens: Int,
    val keyValueHeadCount: Int,
    val headDimension: Int,
    val keyValueBytesPerElement: Int = 2,
    val fixedPositionBufferElements: Long = 0L,
    val fixedPositionBytesPerElement: Int = 4
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
        require(fixedPositionBufferElements >= 0L) {
            "fixedPositionBufferElements cannot be negative."
        }
        require(fixedPositionBytesPerElement > 0) {
            "fixedPositionBytesPerElement must be positive."
        }
    }
}

data class InferenceMemoryEstimate(
    val weightBytes: Long,
    val fixedModelBufferBytes: Long,
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
     * Pass null shape metadata when the native model loader has not inspected its architecture yet.
     */
    /**
     * Estimates an Argus architecture directly from its validated native configuration.
     * The current attention layout uses equal query and key/value head counts.
     */
    fun estimate(
        architecture: ArgusArchitectureConfig,
        quantization: ArgusWeightPrecision,
        keyValueBytesPerElement: Int = 2,
        fixedPositionBytesPerElement: Int = 4
    ): InferenceMemoryEstimate {
        require(keyValueBytesPerElement > 0) {
            "keyValueBytesPerElement must be positive."
        }
        require(fixedPositionBytesPerElement > 0) {
            "fixedPositionBytesPerElement must be positive."
        }

        val parameterBreakdown = ArgusArchitectureParameterEstimator.estimate(architecture)
        val parameterCount = parameterBreakdown.totalTrainableParameters
        val shape = InferenceModelShape(
            parameterCount = parameterCount,
            layerCount = architecture.layerCount,
            contextTokens = architecture.contextLength,
            keyValueHeadCount = architecture.attentionHeadCount,
            headDimension = architecture.headDimension,
            keyValueBytesPerElement = keyValueBytesPerElement,
            fixedPositionBufferElements = parameterBreakdown.fixedPositionBufferElements,
            fixedPositionBytesPerElement = fixedPositionBytesPerElement
        )
        return estimate(
            parameterCount = parameterCount,
            quantization = quantization,
            shape = shape
        )
    }

    fun estimate(
        parameterCount: Long,
        quantization: ArgusWeightPrecision,
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

        val fixedBufferBytes = shape?.let {
            safeMultiply(
                it.fixedPositionBufferElements,
                it.fixedPositionBytesPerElement.toLong()
            )
        } ?: 0L
        val cacheBytes = shape?.let(::estimateKeyValueCacheBytes)
        val totalBytes = cacheBytes?.let {
            val modelBytes = safeAdd(weightBytes, fixedBufferBytes)
            safeAdd(safeAdd(modelBytes, it), workspaceBytes)
        }

        return InferenceMemoryEstimate(
            weightBytes = weightBytes,
            fixedModelBufferBytes = fixedBufferBytes,
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
