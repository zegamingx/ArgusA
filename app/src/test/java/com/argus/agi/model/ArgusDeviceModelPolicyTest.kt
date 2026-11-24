package com.argus.agi.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ArgusDeviceModelPolicyTest {

    @Test
    fun devicesWithoutAdvertisedVulkanStartWithCompactTier() {
        val profile = ArgusDeviceModelPolicy.select(
            totalRamGb = 12,
            maxHeapMb = 512L,
            availableRamGb = 3.0f,
            cpuCores = 8,
            vulkanSupported = false
        )

        assertEquals(ArgusDeviceModelTier.COMPACT, profile.tier)
        assertEquals(1, profile.trainingBatchSize)
        assertEquals(1_024, profile.tier.vocabularyCapacity)
        assertTrue(profile.targetParameterCount <= profile.tier.maximumFamilyParameters)
    }

    @Test
    fun selectsFourModelFamiliesAndBoundedBatchSizesFromDeviceCapacity() {
        val compact = ArgusDeviceModelPolicy.select(4, 256L, 0.6f, 4)
        val balanced = ArgusDeviceModelPolicy.select(6, 256L, 0.8f, 6)
        val performance = ArgusDeviceModelPolicy.select(8, 384L, 1.7f, 8)
        val flagship = ArgusDeviceModelPolicy.select(12, 512L, 3.0f, 8)

        assertEquals(ArgusDeviceModelTier.COMPACT, compact.tier)
        assertEquals(1, compact.trainingBatchSize)
        assertEquals(1_024, compact.tier.vocabularyCapacity)

        assertEquals(ArgusDeviceModelTier.BALANCED, balanced.tier)
        assertEquals(2, balanced.trainingBatchSize)
        assertEquals(2_048, balanced.tier.vocabularyCapacity)

        assertEquals(ArgusDeviceModelTier.PERFORMANCE, performance.tier)
        assertEquals(4, performance.trainingBatchSize)
        assertEquals(4_096, performance.tier.vocabularyCapacity)

        assertEquals(ArgusDeviceModelTier.FLAGSHIP, flagship.tier)
        assertEquals(8, flagship.trainingBatchSize)
        assertEquals(8_192, flagship.tier.vocabularyCapacity)
    }

    @Test
    fun clampsActiveParameterTargetToDeviceMemoryAndFamilyCeiling() {
        val profile = ArgusDeviceModelPolicy.select(12, 512L, 3.0f, 8)

        assertTrue(
            profile.targetParameterCount >= ArgusDeviceModelProfile.MINIMUM_EXECUTABLE_PARAMETER_BUDGET
        )
        assertTrue(profile.targetParameterCount <= profile.memorySafeParameterLimit)
        assertTrue(profile.targetParameterCount <= profile.tier.maximumFamilyParameters)
        assertTrue(profile.memorySafeParameterLimit >= profile.targetParameterCount)
    }

    @Test
    fun heapByteConversionOverflowFailsAsIllegalArgumentInsteadOfArithmetic() {
        val failure = runCatching {
            ArgusDeviceModelPolicy.select(12, Long.MAX_VALUE, 1.0f, 8)
        }.exceptionOrNull()

        assertTrue(failure is IllegalArgumentException)
    }

    @Test
    fun rejectsDeviceMemoryBudgetsBelowTheMinimumExecutableModel() {
        val failure = runCatching {
            ArgusDeviceModelPolicy.select(2, 64L, 0.1f, 4)
        }.exceptionOrNull()

        assertTrue(failure is IllegalArgumentException)
    }

    @Test
    fun everyProfileUsesTheNewFp16WeightAndFp32StateMemoryContract() {
        assertEquals(
            14_000_000L,
            TrainingMemoryPlanner.minimumTrainingStateBytes(1_000_000L)
        )
        assertEquals(
            18_000_000L,
            TrainingMemoryPlanner.minimumAccumulatedTrainingStateBytes(1_000_000L)
        )
    }

    @Test
    fun neverReturnsTrainingBatchOutsideOneThroughEight() {
        val profiles = listOf(
            ArgusDeviceModelPolicy.select(4, 256L, 0.6f, 4),
            ArgusDeviceModelPolicy.select(6, 256L, 0.8f, 6),
            ArgusDeviceModelPolicy.select(8, 384L, 1.7f, 8),
            ArgusDeviceModelPolicy.select(12, 512L, 3.0f, 8)
        )

        profiles.forEach { profile ->
            assertTrue(profile.trainingBatchSize in 1..8)
        }
    }
}
