package com.argus.agi.model

import android.os.PowerManager
import org.junit.Assert.assertEquals
import org.junit.Test

class DeviceAdaptiveRuntimePolicyTest {

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
