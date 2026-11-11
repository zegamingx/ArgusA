package com.argus.agi.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class DeviceModelCapacityPolicyTest {

    @Test
    fun preservesUserDefinedRamAnchors() {
        val expected = listOf(
            1.0 to (0.5 to 1.0),
            2.0 to (1.5 to 2.5),
            3.0 to (2.5 to 3.5),
            4.0 to (3.0 to 5.0),
            5.0 to (4.0 to 6.0),
            6.0 to (5.0 to 7.0),
            7.0 to (6.0 to 8.0),
            8.0 to (7.0 to 9.0),
            10.0 to (9.0 to 12.0),
            12.0 to (12.0 to 14.0),
            16.0 to (18.0 to 21.0),
            24.0 to (28.0 to 32.0),
            32.0 to (38.0 to 42.0)
        )

        expected.forEach { (ramGb, range) ->
            val recommendation = DeviceModelCapacityPolicy.recommend(ramGb)
            assertEquals(range.first, requireNotNull(recommendation.minimumParametersBillions), 0.0001)
            assertEquals(range.second, requireNotNull(recommendation.maximumParametersBillions), 0.0001)
            assertFalse(recommendation.isExtrapolated)
        }
    }

    @Test
    fun interpolatesBetweenDefinedAnchors() {
        val recommendation = DeviceModelCapacityPolicy.recommend(9.0)

        assertEquals(8.0, requireNotNull(recommendation.minimumParametersBillions), 0.0001)
        assertEquals(10.5, requireNotNull(recommendation.maximumParametersBillions), 0.0001)
        assertFalse(recommendation.isExtrapolated)
    }

    @Test
    fun extrapolatesAboveThirtyTwoGbUsingRequestedGrowth() {
        val recommendation = DeviceModelCapacityPolicy.recommend(33.0)

        assertEquals(39.3, requireNotNull(recommendation.minimumParametersBillions), 0.0001)
        assertEquals(43.5, requireNotNull(recommendation.maximumParametersBillions), 0.0001)
        assertTrue(recommendation.isExtrapolated)
    }

    @Test
    fun returnsNoRecommendationBelowOneGbOrForInvalidMemory() {
        assertNull(DeviceModelCapacityPolicy.recommend(0.9).minimumParametersBillions)
        assertNull(DeviceModelCapacityPolicy.recommend(Double.NaN).maximumParametersBillions)
    }
}

class InferenceMemoryEstimatorTest {

    @Test
    fun estimatesQuantizedWeightsAndLeavesKvCacheUnknownWithoutShapeMetadata() {
        val estimate = InferenceMemoryEstimator.estimate(
            parameterCount = 1_000_000_000L,
            quantization = ArgusWeightPrecision.INT4
        )

        assertEquals(625_000_000L, estimate.weightBytes)
        assertEquals(0L, estimate.fixedModelBufferBytes)
        assertNull(estimate.keyValueCacheBytes)
        assertNull(estimate.totalRequiredBytes)
        assertFalse(estimate.hasCompleteModelMetadata)
    }

    @Test
    fun estimatesKvCacheFromModelMetadata() {
        val estimate = InferenceMemoryEstimator.estimate(
            parameterCount = 1_000_000_000L,
            quantization = ArgusWeightPrecision.INT4,
            shape = InferenceModelShape(
                parameterCount = 1_000_000_000L,
                layerCount = 32,
                contextTokens = 2_048,
                keyValueHeadCount = 8,
                headDimension = 128,
                keyValueBytesPerElement = 2
            )
        )

        assertEquals(268_435_456L, estimate.keyValueCacheBytes)
        assertTrue(requireNotNull(estimate.totalRequiredBytes) > estimate.weightBytes)
        assertTrue(estimate.hasCompleteModelMetadata)
    }

    @Test
    fun derivesWeightKvAndFixedPositionMemoryFromArgusArchitecture() {
        val architecture = ArgusArchitectureConfig(
            layerCount = 2,
            hiddenSize = 32,
            attentionHeadCount = 4,
            feedForwardSize = 64,
            vocabularySize = 128,
            contextLength = 32,
            adapterRank = 2
        )
        val estimate = InferenceMemoryEstimator.estimate(
            architecture = architecture,
            quantization = ArgusWeightPrecision.INT4
        )

        val parameterBreakdown = ArgusArchitectureParameterEstimator.estimate(architecture)
        assertEquals(29_184L, parameterBreakdown.totalTrainableParameters)
        assertEquals(14_592L, estimate.weightBytes)
        assertEquals(4_096L, estimate.fixedModelBufferBytes)
        assertEquals(8_192L, estimate.keyValueCacheBytes)
        assertTrue(requireNotNull(estimate.totalRequiredBytes) > estimate.weightBytes)
        assertTrue(estimate.hasCompleteModelMetadata)
    }

    @Test
    fun refusesToClaimFitWhenKvCacheMetadataIsUnknown() {
        val estimate = InferenceMemoryEstimator.estimate(
            parameterCount = 1_000_000_000L,
            quantization = ArgusWeightPrecision.INT4
        )

        assertNull(
            InferenceMemoryEstimator.fitsWithinAvailableMemory(
                estimate = estimate,
                availableRamBytes = 8L * 1024L * 1024L * 1024L
            )
        )
    }
}

class TrainingMemoryPlannerTest {

    @Test
    fun reportsSixteenBytePerParameterTrainingStateFloor() {
        assertEquals(
            16_000_000_000L,
            TrainingMemoryPlanner.minimumTrainingStateBytes(1_000_000_000L)
        )
    }

    @Test
    fun saturatesInsteadOfOverflowing() {
        assertEquals(
            Long.MAX_VALUE,
            TrainingMemoryPlanner.minimumTrainingStateBytes(Long.MAX_VALUE)
        )
    }
}
