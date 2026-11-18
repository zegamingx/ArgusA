package com.argus.agi.model

import java.nio.charset.StandardCharsets
import java.security.MessageDigest

/**
 * Versioned training-update semantics used by the Kotlin and Vulkan AdamW paths.
 *
 * Policy IDs are part of the fingerprint because changing update granularity,
 * normalization, clipping order, or decay groups changes the meaning of the saved
 * Adam moments even when the scalar hyperparameters remain unchanged.
 */
internal data class OptimizerConfig(
    val schemaVersion: Int = 1,
    val algorithmId: String = "adamw",
    val learningRateScheduleId: String = "inverse_time_decay_v1",
    val gradientNormalizationPolicyId: String = "mean_by_valid_targets_v1",
    val gradientClippingPolicyId: String = "global_l2_after_normalization_epsilon_1e-6_v1",
    val weightDecayPolicyId: String = "transformer_projection_weights_only_v1",
    val baseLearningRate: Float = 0.005f,
    val learningRateDecay: Float = 0.0002f,
    val weightDecay: Float = 0.01f,
    val beta1: Float = 0.9f,
    val beta2: Float = 0.999f,
    val epsilon: Float = 1.0e-8f,
    val maximumGradientNorm: Float = 1.0f
) {
    init {
        require(schemaVersion > 0)
        requireStableIdentifier(algorithmId, "optimizer algorithm")
        requireStableIdentifier(learningRateScheduleId, "learning-rate schedule")
        requireStableIdentifier(gradientNormalizationPolicyId, "gradient normalization policy")
        requireStableIdentifier(gradientClippingPolicyId, "gradient clipping policy")
        requireStableIdentifier(weightDecayPolicyId, "weight-decay policy")
        require(baseLearningRate.isFinite() && baseLearningRate > 0f)
        require(learningRateDecay.isFinite() && learningRateDecay >= 0f)
        require(weightDecay.isFinite() && weightDecay >= 0f)
        require(beta1.isFinite() && beta1 >= 0f && beta1 < 1f)
        require(beta2.isFinite() && beta2 >= 0f && beta2 < 1f)
        require(epsilon.isFinite() && epsilon > 0f)
        require(maximumGradientNorm.isFinite() && maximumGradientNorm > 0f)
    }

    /** Stable SHA-256 identity; float values use raw IEEE-754 bits, not decimal text. */
    val fingerprint: String by lazy(LazyThreadSafetyMode.PUBLICATION) {
        val canonical = buildString {
            append("ARGUS-OPTIMIZER-CONFIG\n")
            append(schemaVersion).append('\n')
            append(algorithmId).append('\n')
            append(learningRateScheduleId).append('\n')
            append(gradientNormalizationPolicyId).append('\n')
            append(gradientClippingPolicyId).append('\n')
            append(weightDecayPolicyId).append('\n')
            append(baseLearningRate.toStableBits()).append('\n')
            append(learningRateDecay.toStableBits()).append('\n')
            append(weightDecay.toStableBits()).append('\n')
            append(beta1.toStableBits()).append('\n')
            append(beta2.toStableBits()).append('\n')
            append(epsilon.toStableBits()).append('\n')
            append(maximumGradientNorm.toStableBits())
        }
        MessageDigest.getInstance("SHA-256")
            .digest(canonical.toByteArray(StandardCharsets.UTF_8))
            .joinToString(separator = "") { byte ->
                "%02x".format(java.util.Locale.ROOT, byte.toInt() and 0xFF)
            }
    }

    val identity: OptimizerConfigIdentity
        get() = OptimizerConfigIdentity(schemaVersion, algorithmId, fingerprint)

    private fun requireStableIdentifier(value: String, label: String) {
        require(value.isNotBlank() && value.length <= 96 && value.all { character ->
            character in 'a'..'z' || character in 'A'..'Z' || character in '0'..'9' ||
                character == '_' || character == '-' || character == '.'
        }) {
            "$label ID must be a non-empty stable ASCII identifier."
        }
    }

    private fun Float.toStableBits(): String =
        toRawBits().toUInt().toString(16).padStart(8, '0')

    companion object {
        val DEFAULT = OptimizerConfig()
    }
}

internal data class OptimizerConfigIdentity(
    val schemaVersion: Int,
    val algorithmId: String,
    val fingerprint: String
) {
    init {
        require(schemaVersion > 0)
        require(algorithmId.isNotBlank())
        require(fingerprint.length == 64 && fingerprint.all { character ->
            character in '0'..'9' || character in 'a'..'f'
        })
    }
}
