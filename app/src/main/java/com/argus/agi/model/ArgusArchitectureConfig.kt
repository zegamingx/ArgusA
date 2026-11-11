package com.argus.agi.model

/**
 * Stable, Argus-owned description of a neural architecture.
 *
 * This is an architecture contract and accounting model. It does not imply that
 * every described topology is executable by the current legacy engine.
 */
enum class ArgusAttentionKind {
    CAUSAL_MULTI_HEAD
}

enum class ArgusFeedForwardKind {
    SWIGLU
}

enum class ArgusPositionEncodingKind {
    FIXED_SINUSOIDAL_TABLE,
    ROTARY
}

enum class ArgusNormalizationKind {
    LAYER_NORM
}

data class ArgusArchitectureConfig(
    val schemaVersion: Int = CURRENT_SCHEMA_VERSION,
    val architectureId: String = DEFAULT_ARCHITECTURE_ID,
    val layerCount: Int,
    val hiddenSize: Int,
    val attentionHeadCount: Int,
    val feedForwardSize: Int,
    val vocabularySize: Int,
    val contextLength: Int,
    val adapterRank: Int = 0,
    val attentionKind: ArgusAttentionKind = ArgusAttentionKind.CAUSAL_MULTI_HEAD,
    val feedForwardKind: ArgusFeedForwardKind = ArgusFeedForwardKind.SWIGLU,
    val positionEncodingKind: ArgusPositionEncodingKind =
        ArgusPositionEncodingKind.FIXED_SINUSOIDAL_TABLE,
    val normalizationKind: ArgusNormalizationKind = ArgusNormalizationKind.LAYER_NORM,
    val tokenEmbeddingsAreTied: Boolean = false
) {
    init {
        require(schemaVersion == CURRENT_SCHEMA_VERSION) {
            "Unsupported Argus architecture schema version: $schemaVersion."
        }
        require(architectureId.isNotBlank()) {
            "Architecture ID must not be blank."
        }
        require(layerCount > 0) {
            "Layer count must be positive."
        }
        require(hiddenSize > 0) {
            "Hidden size must be positive."
        }
        require(attentionHeadCount > 0 && hiddenSize % attentionHeadCount == 0) {
            "Attention head count must divide hidden size evenly."
        }
        require(feedForwardSize > 0) {
            "Feed-forward size must be positive."
        }
        require(vocabularySize >= MINIMUM_VOCABULARY_SIZE) {
            "Vocabulary must include all canonical special-token IDs."
        }
        require(contextLength >= 2) {
            "Context length must be at least two tokens."
        }
        require(adapterRank in 0..hiddenSize) {
            "Adapter rank must be between zero and hidden size."
        }
    }

    val headDimension: Int
        get() = hiddenSize / attentionHeadCount

    companion object {
        const val CURRENT_SCHEMA_VERSION = 1
        const val DEFAULT_ARCHITECTURE_ID = "argus.transformer.swiglu.v1"
        const val MINIMUM_VOCABULARY_SIZE = 9
    }
}

/**
 * Maps the current single-block engine configuration into the architecture contract.
 * The legacy engine has exactly one Transformer block until its forward, backward,
 * optimizer, cache, and checkpoint paths are migrated to a block collection.
 */
fun EngineConfig.toArgusArchitectureConfig(
    vocabularySize: Int,
    layerCount: Int = this.layerCount
): ArgusArchitectureConfig = ArgusArchitectureConfig(
    layerCount = layerCount,
    hiddenSize = dModel,
    attentionHeadCount = nHeads,
    feedForwardSize = dFf,
    vocabularySize = vocabularySize,
    contextLength = maxSeqLen,
    adapterRank = loraRank
)

/**
 * Logical parameter and resident-buffer accounting for an Argus architecture.
 *
 * Position tables are resident tensors but are not trainable parameters in the
 * current fixed-sinusoidal implementation.
 */
data class ArgusParameterBreakdown(
    val tokenEmbeddingParameters: Long,
    val outputProjectionParameters: Long,
    val attentionParameters: Long,
    val feedForwardParameters: Long,
    val adapterParameters: Long,
    val fixedPositionBufferElements: Long
) {
    val totalTrainableParameters: Long
        get() = checkedSum(
            tokenEmbeddingParameters,
            outputProjectionParameters,
            attentionParameters,
            feedForwardParameters,
            adapterParameters
        )

    val totalResidentModelElements: Long
        get() = Math.addExact(totalTrainableParameters, fixedPositionBufferElements)

    private fun checkedSum(vararg values: Long): Long =
        values.fold(0L) { total, value -> Math.addExact(total, value) }
}

object ArgusArchitectureParameterEstimator {

    /**
     * Counts the current Argus Transformer/SwiGLU layout:
     * - one input embedding matrix and an untied output projection by default;
     * - Q, K, V, and output attention projections per layer;
     * - gate, up, and down SwiGLU projections per layer;
     * - optional LoRA A/B matrices for Q and V per layer;
     * - a fixed sinusoidal position table when selected.
     *
     * Linear projections currently have no bias vectors and normalization has no
     * learned affine vectors, matching the implemented legacy engine.
     */
    fun estimate(config: ArgusArchitectureConfig): ArgusParameterBreakdown {
        val hidden = config.hiddenSize.toLong()
        val vocabulary = config.vocabularySize.toLong()
        val layers = config.layerCount.toLong()
        val feedForward = config.feedForwardSize.toLong()
        val rank = config.adapterRank.toLong()

        val embeddingParameters = multiply(vocabulary, hidden)
        val outputProjectionParameters =
            if (config.tokenEmbeddingsAreTied) 0L else embeddingParameters

        val attentionPerLayer = multiply(4L, multiply(hidden, hidden))
        val attentionParameters = multiply(layers, attentionPerLayer)

        val feedForwardPerLayer = multiply(3L, multiply(hidden, feedForward))
        val feedForwardParameters = multiply(layers, feedForwardPerLayer)

        val adaptersPerLayer = multiply(4L, multiply(hidden, rank))
        val adapterParameters = multiply(layers, adaptersPerLayer)

        val fixedPositionBufferElements = when (config.positionEncodingKind) {
            ArgusPositionEncodingKind.FIXED_SINUSOIDAL_TABLE ->
                multiply(config.contextLength.toLong(), hidden)
            ArgusPositionEncodingKind.ROTARY -> 0L
        }

        return ArgusParameterBreakdown(
            tokenEmbeddingParameters = embeddingParameters,
            outputProjectionParameters = outputProjectionParameters,
            attentionParameters = attentionParameters,
            feedForwardParameters = feedForwardParameters,
            adapterParameters = adapterParameters,
            fixedPositionBufferElements = fixedPositionBufferElements
        )
    }

    private fun multiply(left: Long, right: Long): Long =
        Math.multiplyExact(left, right)
}
