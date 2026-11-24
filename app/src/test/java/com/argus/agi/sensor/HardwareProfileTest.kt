package com.argus.agi.sensor

import com.argus.agi.model.ArgusDeviceModelPolicy
import com.argus.agi.model.ArgusDeviceModelTier
import org.junit.Assert.assertEquals
import org.junit.Test

class HardwareProfileTest {

    private fun gibBytes(gib: Double): Long = (gib * 1024.0 * 1024.0 * 1024.0).toLong()

    @Test
    fun marketedCapacityRoundsToNearestGibInsteadOfTruncating() {
        // An 8 GB device reports only ~7.5 GiB of usable totalMem because firmware,
        // the kernel and the GPU reserve the rest; truncation classified it as 7 GB.
        assertEquals(8, HardwareProfile.totalRamGbFromBytes(gibBytes(7.56)))
        assertEquals(8, HardwareProfile.totalRamGbFromBytes(gibBytes(7.5)))
        assertEquals(12, HardwareProfile.totalRamGbFromBytes(gibBytes(11.6)))
        assertEquals(4, HardwareProfile.totalRamGbFromBytes(gibBytes(3.6)))
    }

    @Test
    fun belowHalfGibBoundaryStillRoundsDown() {
        assertEquals(7, HardwareProfile.totalRamGbFromBytes(gibBytes(7.4)))
        assertEquals(6, HardwareProfile.totalRamGbFromBytes(gibBytes(6.49)))
    }

    @Test
    fun unknownTotalMemoryKeepsTheSafeFallback() {
        assertEquals(8, HardwareProfile.totalRamGbFromBytes(0L))
        assertEquals(8, HardwareProfile.totalRamGbFromBytes(-1L))
    }

    @Test
    fun eightGbDeviceWithReservedMemoryReachesPerformanceTier() {
        val totalRamGb = HardwareProfile.totalRamGbFromBytes(gibBytes(7.56))

        val profile = ArgusDeviceModelPolicy.select(
            totalRamGb = totalRamGb,
            maxHeapMb = 384L,
            availableRamGb = 1.7f,
            cpuCores = 8
        )

        assertEquals(ArgusDeviceModelTier.PERFORMANCE, profile.tier)
        assertEquals(4, profile.trainingBatchSize)
        assertEquals(4_096, profile.tier.vocabularyCapacity)
    }
}
