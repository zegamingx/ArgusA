package com.argus.agi.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ArgusDeviceModelPolicyTest {

    @Test
    fun selectsFourModelFamiliesAndBoundedBatchSizesFromDeviceCapacity() {
        val compact = ArgusDeviceModelPolicy.select(
            totalRamGb = 4,
            maxHeapMb = 256L,
            availableRamGb = 0.6f,
            cpuCores = 4
        )
        val balanced = ArgusDeviceModelPolicy.select(
            totalRamGb = 6,
            maxHeapMb = 256L,
            availableRamGb = 0.8f,
            cpuCores = 6
        )
        val performance = ArgusDeviceModelPolicy.select(
            totalRamGb = 8,
            maxHeapMb = 384L,
            availableRamGb = 1.7f,
            cpuCores = 8
        )
        val flagship = ArgusDeviceModelPolicy.select(
            totalRamGb = 12,
            maxHeapMb = 512L,
            availableRamGb = 3.0f,
            cpuCores = 8
        )

        assertEquals(ArgusDeviceModelTier.COMPACT, compact.tier)
        assertEquals(1, compact.trainingBatchSize)
        assertEquals(ArgusDeviceModelTier.BALANCED, balanced.tier)
        assertEquals(2, balanced.trainingBatchSize)
        assertEquals(ArgusDeviceModelTier.PERFORMANCE, performance.tier)
        assertEquals(4, performance.trainingBatchSize)
        assertEquals(ArgusDeviceModelTier.FLAGSHIP, flagship.tier)
        assertEquals(8, flagship.trainingBatchSize)
    }

    @Test
    fun clampsActiveParameterTargetToDeviceMemoryAndSevenBFamilyCeiling() {
        val profile = ArgusDeviceModelPolicy.select(
            totalRamGb = 12,
            maxHeapMb = 512L,
            availableRamGb = 3.0f,
            cpuCores = 8
        )

        assertTrue(profile.targetParameterCount >= 1_000_000L)
        assertTrue(profile.targetParameterCount <= profile.memorySafeParameterLimit)
        assertTrue(profile.targetParameterCount <= 7_000_000_000L)
    }

    @Test
    fun selectsAnExecutableModelWithAtLeastOneMillionParameters() {
        val config = EngineConfig.forDevice(
            totalRamGb = 4,
            maxHeapMb = 256L,
            availableRamGb = 0.6f,
            cpuCores = 4
        )
        val architecture = ArgusArchitectureConfig(
            layerCount = config.layerCount,
            hiddenSize = config.dModel,
            attentionHeadCount = config.nHeads,
            feedForwardSize = config.dFf,
            vocabularySize = TokenizerConfig.MINIMUM_VOCAB_SIZE,
            contextLength = config.maxSeqLen,
            adapterRank = config.loraRank
        )
        val parameterCount = ArgusArchitectureParameterEstimator
            .estimate(architecture)
            .totalTrainableParameters

        assertTrue(parameterCount >= 1_000_000L)
        assertEquals(ArgusDeviceModelTier.COMPACT, config.modelTier)
        assertEquals(1, config.trainingBatchSize)
    }

    @Test
    fun rejectsDeviceMemoryBudgetsBelowTheMinimumExecutableModel() {
        val failure = runCatching {
            ArgusDeviceModelPolicy.select(
                totalRamGb = 2,
                maxHeapMb = 64L,
                availableRamGb = 0.1f,
                cpuCores = 4
            )
        }.exceptionOrNull()

        assertTrue(failure is IllegalArgumentException)
    }

    @Test
    fun neverReturnsTrainingBatchOutsideOneThroughEight() {
        val profiles = listOf(
            ArgusDeviceModelPolicy.select(2, 128L, 0.3f, 4),
            ArgusDeviceModelPolicy.select(6, 256L, 0.8f, 6),
            ArgusDeviceModelPolicy.select(8, 384L, 1.7f, 8),
            ArgusDeviceModelPolicy.select(12, 512L, 3.0f, 8)
        )

        profiles.forEach { profile ->
            assertTrue(profile.trainingBatchSize in 1..8)
        }
    }
}
