package com.argus.agi.model

import android.os.PowerManager
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DeviceAdaptiveRuntimePolicyTest {

    @Test
    fun streamedDocumentImportLimitUsesLiveHeapAndHardMaximum() {
        assertEquals(
            2_048,
            DeviceAdaptiveRuntimePolicy.streamedDocumentCharacterLimit(
                availableHeapBytes = 49_152L,
                hardLimitCharacters = 2 * 1024 * 1024
            )
        )
        assertEquals(
            2 * 1024 * 1024,
            DeviceAdaptiveRuntimePolicy.streamedDocumentCharacterLimit(
                availableHeapBytes = Long.MAX_VALUE,
                hardLimitCharacters = 2 * 1024 * 1024
            )
        )
        assertEquals(
            0,
            DeviceAdaptiveRuntimePolicy.streamedDocumentCharacterLimit(
                availableHeapBytes = 0L,
                hardLimitCharacters = 2 * 1024 * 1024
            )
        )
    }

    @Test
    fun deviceLocalHeadroomIgnoresHostOnlyMemoryHeaps() {
        val headroom = DeviceAdaptiveRuntimePolicy.deviceLocalHeapHeadroomBytes(
            budgets = listOf(1_000L, 2_000L),
            usages = listOf(250L, 700L),
            heapFlags = listOf(0, 0x1)
        )

        assertEquals(listOf(1_300L), headroom)
    }

    @Test
    fun legacyDriversWithoutHeapFlagsUseConservativeAllHeapHeadroom() {
        val headroom = DeviceAdaptiveRuntimePolicy.deviceLocalHeapHeadroomBytes(
            budgets = listOf(1_000L, 2_000L),
            usages = listOf(600L, 1_000L),
            heapFlags = emptyList()
        )

        assertEquals(listOf(400L, 1_000L), headroom)
    }

    @Test
    fun tensorArenaBudgetUsesLiveHeapDeviceLocalAndDescriptorLimits() {
        val mebibyte = 1024L * 1024L
        val budget = DeviceAdaptiveRuntimePolicy.calculateTensorArenaBudget(
            maximumAppHeapBytes = 1_024L * mebibyte,
            availableAppHeapBytes = 128L * mebibyte,
            deviceLocalHeadroomBytes = 120L * mebibyte,
            maximumStorageBufferRangeBytes = 512L * mebibyte
        )

        assertEquals(40L * mebibyte, budget.totalBytes)
        assertEquals(budget.totalBytes / 12L, budget.modelBytes)
        assertEquals(budget.totalBytes / 6L, budget.activationBytes)
        assertEquals(budget.totalBytes / 3L, budget.optimizerBytes)
    }

    @Test
    fun tensorArenaFallsBackToLiveAppHeapWhenGpuBudgetIsUnavailable() {
        val mebibyte = 1024L * 1024L
        val budget = DeviceAdaptiveRuntimePolicy.calculateTensorArenaBudget(
            maximumAppHeapBytes = 1_024L * mebibyte,
            availableAppHeapBytes = 128L * mebibyte,
            deviceLocalHeadroomBytes = null,
            maximumStorageBufferRangeBytes = 0L
        )

        assertEquals(64L * mebibyte, budget.totalBytes)
    }

    @Test
    fun adaptiveBatchAndContextRespectMemoryThermalAndPowerConstraints() {
        val batchSize = DeviceAdaptiveRuntimePolicy.trainingBatchLimit(
            configuredBatchSize = 8,
            thermalStatus = PowerManager.THERMAL_STATUS_LIGHT,
            availableHeapBytes = 256L * 1024L * 1024L,
            maximumHeapBytes = 1_024L * 1024L * 1024L,
            availableSystemMemoryBytes = 2L * 1024L * 1024L * 1024L
        )
        assertEquals(4, batchSize)

        val memoryLimitedBatch = DeviceAdaptiveRuntimePolicy.trainingBatchLimit(
            configuredBatchSize = 8,
            thermalStatus = PowerManager.THERMAL_STATUS_LIGHT,
            availableHeapBytes = 128L * 1024L * 1024L,
            maximumHeapBytes = 1_024L * 1024L * 1024L,
            availableSystemMemoryBytes = 128L * 1024L * 1024L,
            systemLowMemory = true
        )
        assertEquals(1, memoryLimitedBatch)

        val context = DeviceAdaptiveRuntimePolicy.trainingContextLimit(
            configuredContextLength = 128,
            availableHeapBytes = 128L * 1024L * 1024L,
            maximumHeapBytes = 1_024L * 1024L * 1024L,
            dModel = 128,
            feedForwardSize = 512,
            vocabularySize = 2_048,
            layerCount = 2,
            headCount = 4,
            thermalStatus = PowerManager.THERMAL_STATUS_MODERATE,
            availableSystemMemoryBytes = 2L * 1024L * 1024L * 1024L
        )
        assertTrue(context in 32..96)
    }

    @Test
    fun returnsNoTrainingContextWhenEvenMinimumWindowExceedsLiveWorkspaceBudget() {
        val contextLimit = DeviceAdaptiveRuntimePolicy.trainingContextLimit(
            configuredContextLength = 128,
            availableHeapBytes = 512L * 1024L,
            maximumHeapBytes = 512L * 1024L * 1024L,
            dModel = 128,
            feedForwardSize = 512,
            vocabularySize = 2_048,
            layerCount = 2,
            headCount = 4,
            thermalStatus = PowerManager.THERMAL_STATUS_LIGHT,
            availableSystemMemoryBytes = 2L * 1024L * 1024L * 1024L
        )

        assertEquals(
            "The worker must defer training rather than force a context that does not fit.",
            0,
            contextLimit
        )
    }

    @Test
    fun keepsTheMinimumLegalContextWhenItsWorkspaceFitsTheCurrentBudget() {
        val contextLimit = DeviceAdaptiveRuntimePolicy.trainingContextLimit(
            configuredContextLength = 128,
            availableHeapBytes = 64L * 1024L * 1024L,
            maximumHeapBytes = 512L * 1024L * 1024L,
            dModel = 128,
            feedForwardSize = 512,
            vocabularySize = 2_048,
            layerCount = 2,
            headCount = 4,
            thermalStatus = PowerManager.THERMAL_STATUS_LIGHT,
            availableSystemMemoryBytes = 2L * 1024L * 1024L * 1024L,
            memoryRecovery = true
        )

        assertEquals(64, contextLimit)
    }

    @Test
    fun trainingWorkspaceEstimateGrowsWithContextAndArchitecture() {
        val short = DeviceAdaptiveRuntimePolicy.estimateTrainingWindowWorkspaceBytes(
            contextLength = 32,
            dModel = 128,
            feedForwardSize = 512,
            vocabularySize = 2_048,
            layerCount = 1,
            headCount = 4
        )
        val long = DeviceAdaptiveRuntimePolicy.estimateTrainingWindowWorkspaceBytes(
            contextLength = 64,
            dModel = 128,
            feedForwardSize = 512,
            vocabularySize = 2_048,
            layerCount = 1,
            headCount = 4
        )
        val deeper = DeviceAdaptiveRuntimePolicy.estimateTrainingWindowWorkspaceBytes(
            contextLength = 32,
            dModel = 128,
            feedForwardSize = 512,
            vocabularySize = 2_048,
            layerCount = 2,
            headCount = 4
        )

        assertTrue(long > short)
        assertTrue(deeper > short)
    }

    @Test
    fun adaptiveBatchAndContextReduceLoadForPowerSavingAndRecovery() {
        assertEquals(
            4,
            DeviceAdaptiveRuntimePolicy.trainingBatchLimit(
                configuredBatchSize = 8,
                thermalStatus = PowerManager.THERMAL_STATUS_LIGHT,
                isPowerSaveMode = true,
                isCharging = false,
                batteryPct = 18
            )
        )
        assertEquals(
            1,
            DeviceAdaptiveRuntimePolicy.trainingBatchLimit(
                configuredBatchSize = 8,
                thermalStatus = PowerManager.THERMAL_STATUS_LIGHT,
                memoryRecovery = true
            )
        )
        assertEquals(
            64,
            DeviceAdaptiveRuntimePolicy.trainingContextLimit(
                configuredContextLength = 128,
                availableHeapBytes = 800L * 1024L * 1024L,
                maximumHeapBytes = 1_024L * 1024L * 1024L,
                dModel = 128,
                feedForwardSize = 512,
                vocabularySize = 2_048,
                layerCount = 1,
                headCount = 4,
                thermalStatus = PowerManager.THERMAL_STATUS_LIGHT,
                memoryRecovery = true
            )
        )
        assertEquals(
            500L,
            DeviceAdaptiveRuntimePolicy.trainingBatchDelayMillis(
                thermalStatus = PowerManager.THERMAL_STATUS_LIGHT,
                temperatureCelsius = 35.0f,
                batteryPct = 20,
                isCharging = false,
                isPowerSaveMode = true
            )
        )
    }

    @Test
    fun thermalStatusReducesTrainingBatchAndSevereStatusPausesIt() {
        assertEquals(
            8,
            DeviceAdaptiveRuntimePolicy.trainingBatchLimit(
                configuredBatchSize = 8,
                thermalStatus = PowerManager.THERMAL_STATUS_LIGHT
            )
        )
        assertEquals(
            4,
            DeviceAdaptiveRuntimePolicy.trainingBatchLimit(
                configuredBatchSize = 8,
                thermalStatus = PowerManager.THERMAL_STATUS_MODERATE
            )
        )
        assertEquals(
            0,
            DeviceAdaptiveRuntimePolicy.trainingBatchLimit(
                configuredBatchSize = 8,
                thermalStatus = PowerManager.THERMAL_STATUS_SEVERE
            )
        )
    }
}
