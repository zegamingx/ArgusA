package com.argus.agi.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class OptimizerConfigTest {
    @Test
    fun defaultConfigurationHasStableSha256Identity() {
        val fingerprint = OptimizerConfig.DEFAULT.fingerprint
        assertEquals(64, fingerprint.length)
        assertEquals(fingerprint, OptimizerConfig.DEFAULT.copy().fingerprint)
        assertEquals(fingerprint, OptimizerConfig.DEFAULT.identity.fingerprint)
    }

    @Test
    fun eachOptimizerSemanticChangeChangesTheFingerprint() {
        val fingerprint = OptimizerConfig.DEFAULT.fingerprint
        val variants = listOf(
            OptimizerConfig.DEFAULT.copy(schemaVersion = 2),
            OptimizerConfig.DEFAULT.copy(algorithmId = "adamw_variant"),
            OptimizerConfig.DEFAULT.copy(learningRateScheduleId = "constant_v1"),
            OptimizerConfig.DEFAULT.copy(gradientNormalizationPolicyId = "mean_per_sample_v1"),
            OptimizerConfig.DEFAULT.copy(gradientClippingPolicyId = "per_tensor_l2_v1"),
            OptimizerConfig.DEFAULT.copy(weightDecayPolicyId = "all_weights_v2"),
            OptimizerConfig.DEFAULT.copy(baseLearningRate = 0.006f),
            OptimizerConfig.DEFAULT.copy(learningRateDecay = 0.0003f),
            OptimizerConfig.DEFAULT.copy(weightDecay = 0.02f),
            OptimizerConfig.DEFAULT.copy(beta1 = 0.91f),
            OptimizerConfig.DEFAULT.copy(beta2 = 0.998f),
            OptimizerConfig.DEFAULT.copy(epsilon = 1.0e-7f),
            OptimizerConfig.DEFAULT.copy(maximumGradientNorm = 0.9f),
            OptimizerConfig.DEFAULT.copy(gradientNormEpsilon = 2.0e-6f)
        )
        variants.forEach { changed -> assertNotEquals(fingerprint, changed.fingerprint) }
    }

    @Test
    fun rejectsInvalidOptimizerHyperparameters() {
        val invalid = listOf(
            runCatching { OptimizerConfig.DEFAULT.copy(baseLearningRate = 0f) },
            runCatching { OptimizerConfig.DEFAULT.copy(learningRateDecay = Float.NaN) },
            runCatching { OptimizerConfig.DEFAULT.copy(weightDecay = -0.1f) },
            runCatching { OptimizerConfig.DEFAULT.copy(beta1 = 1f) },
            runCatching { OptimizerConfig.DEFAULT.copy(beta2 = Float.POSITIVE_INFINITY) },
            runCatching { OptimizerConfig.DEFAULT.copy(epsilon = 0f) },
            runCatching { OptimizerConfig.DEFAULT.copy(maximumGradientNorm = -1f) },
            runCatching { OptimizerConfig.DEFAULT.copy(gradientNormEpsilon = Float.NaN) }
        )
        invalid.forEach { result -> assertNotNull(result.exceptionOrNull()) }
    }
}
