package com.argus.agi.runtime

import android.os.PowerManager
import com.argus.agi.backend.ComputeBackend
import com.argus.agi.backend.ComputeBackendKind
import com.argus.agi.device.DeviceProfile
import com.argus.agi.model.VulkanDeviceCapabilities
import com.argus.agi.model.toDeviceProfile
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RuntimeConfigurationPlannerTest {

    @Test
    fun keepsVulkanPrimaryAndRequiresSupportedEnabledAndVerifiedFp16() {
        val profile = deviceProfile(shaderFloat16Supported = true, shaderFloat16Enabled = true)
        val configuration = RuntimeConfigurationPlanner.create(
            backend = FakeComputeBackend(profile, isReady = true, isFp16ComputeReady = true),
            resources = healthyResources(),
            model = modelConstraints()
        )

        assertEquals(RuntimeComputeBackend.VULKAN, configuration.computeBackend)
        assertTrue(configuration.fp16ComputeUsable)
        assertEquals(2, configuration.effectiveTrainingBatchSize)
        assertEquals(128, configuration.effectiveContextLength)
        assertNull(configuration.deferReason)
        assertEquals("Mali-G57", configuration.deviceName)
        assertEquals("1.3.278", configuration.effectiveVulkanApiVersion)
        assertEquals(256, configuration.dispatchLimits?.maxComputeWorkGroupInvocations)
    }

    @Test
    fun doesNotEnableFp16WhenPhysicalDeviceFeatureWasNotEnabled() {
        val profile = deviceProfile(shaderFloat16Supported = true, shaderFloat16Enabled = false)
        val configuration = RuntimeConfigurationPlanner.create(
            backend = FakeComputeBackend(profile, isReady = true, isFp16ComputeReady = true),
            resources = healthyResources(),
            model = modelConstraints()
        )

        assertFalse(configuration.fp16ComputeUsable)
        assertEquals(RuntimeComputeBackend.VULKAN, configuration.computeBackend)
    }

    @Test
    fun doesNotEnableFp16WhenKernelSelfTestHasNotPassed() {
        val profile = deviceProfile(shaderFloat16Supported = true, shaderFloat16Enabled = true)
        val configuration = RuntimeConfigurationPlanner.create(
            backend = FakeComputeBackend(profile, isReady = true, isFp16ComputeReady = false),
            resources = healthyResources(),
            model = modelConstraints()
        )

        assertFalse(configuration.fp16ComputeUsable)
    }

    @Test
    fun reportsCpuOnlyAsTheExplicitFallbackWhenVulkanIsUnavailable() {
        val configuration = RuntimeConfigurationPlanner.create(
            backend = FakeComputeBackend(null, isReady = false, isFp16ComputeReady = false),
            resources = healthyResources(),
            model = modelConstraints()
        )

        assertEquals(RuntimeComputeBackend.CPU_FALLBACK, configuration.computeBackend)
        assertFalse(configuration.fp16ComputeUsable)
        assertNull(configuration.dispatchLimits)
    }

    @Test
    fun blocksTrainingWhenThermalLimitIsSevere() {
        val configuration = RuntimeConfigurationPlanner.create(
            backend = FakeComputeBackend(deviceProfile(), isReady = true, isFp16ComputeReady = false),
            resources = healthyResources().copy(thermalStatus = PowerManager.THERMAL_STATUS_SEVERE),
            model = modelConstraints()
        )

        assertEquals(0, configuration.effectiveTrainingBatchSize)
        assertEquals(0, configuration.effectiveContextLength)
        assertEquals(RuntimeDeferReason.THERMAL_LIMIT, configuration.deferReason)
        assertFalse(configuration.canTrain)
    }

    @Test
    fun defersTrainingWhenEvenTheMinimumContextDoesNotFitMemory() {
        val configuration = RuntimeConfigurationPlanner.create(
            backend = FakeComputeBackend(deviceProfile(), isReady = true, isFp16ComputeReady = false),
            resources = healthyResources().copy(
                availableAppHeapBytes = 2L * MEBIBYTE,
                availableSystemMemoryBytes = 128L * MEBIBYTE
            ),
            model = modelConstraints()
        )

        assertEquals(RuntimeDeferReason.INSUFFICIENT_MEMORY, configuration.deferReason)
        assertFalse(configuration.canTrain)
        assertEquals(0, configuration.effectiveContextLength)
    }

    @Test
    fun runtimeLimitsDoNotRewriteConfiguredModelDimensions() {
        val model = modelConstraints()
        val configuration = RuntimeConfigurationPlanner.create(
            backend = FakeComputeBackend(deviceProfile(), isReady = true, isFp16ComputeReady = false),
            resources = healthyResources().copy(
                availableAppHeapBytes = 200L * MEBIBYTE,
                maximumAppHeapBytes = 512L * MEBIBYTE
            ),
            model = model
        )

        assertEquals(128, configuration.configuredContextLength)
        assertEquals(128, model.configuredContextLength)
        assertEquals(128, configuration.effectiveContextLength)
    }

    @Test
    fun tensorArenaBudgetUsesTheDeviceLocalHeapAndDescriptorLimit() {
        val profile = deviceProfile(
            deviceLocalBudgetBytes = 60L * MEBIBYTE,
            deviceLocalUsageBytes = 0L,
            maxStorageBufferRangeBytes = 8L * MEBIBYTE
        )
        val budget = RuntimeConfigurationPlanner.calculateTensorArenaBudget(
            deviceProfile = profile,
            maximumAppHeapBytes = 400L * MEBIBYTE,
            availableAppHeapBytes = 200L * MEBIBYTE
        )

        assertEquals(8L * MEBIBYTE, budget.totalBytes)
    }

    private class FakeComputeBackend(
        profile: DeviceProfile?,
        override val isReady: Boolean,
        override val isFp16ComputeReady: Boolean
    ) : ComputeBackend {
        override val kind: ComputeBackendKind = ComputeBackendKind.VULKAN
        override val deviceProfile: StateFlow<DeviceProfile?> =
            MutableStateFlow(profile)

        override fun initializeAfterModelReady(): Boolean = isReady

        override fun resetForRetry() = Unit

        override fun synchronizeAuthoritativeState(): Boolean = true

        override fun releaseDeviceResourcesForMemoryPressure(): Boolean = true

        override fun resumeDeviceResourcesAfterForeground() = Unit
    }

    private fun healthyResources(): RuntimeResourceSnapshot = RuntimeResourceSnapshot(
        availableAppHeapBytes = 400L * MEBIBYTE,
        maximumAppHeapBytes = 512L * MEBIBYTE,
        availableSystemMemoryBytes = 2L * GIBIBYTE,
        systemLowMemory = false,
        thermalStatus = PowerManager.THERMAL_STATUS_NONE,
        temperatureCelsius = 30.0f,
        isPowerSaveMode = false,
        isCharging = true,
        batteryPct = 100,
        memoryRecovery = false
    )

    private fun modelConstraints(): RuntimeModelConstraints = RuntimeModelConstraints(
        configuredBatchSize = 2,
        configuredContextLength = 128,
        dModel = 128,
        feedForwardSize = 512,
        vocabularySize = 2_048,
        layerCount = 1,
        headCount = 4
    )

    private fun deviceProfile(
        shaderFloat16Supported: Boolean = false,
        shaderFloat16Enabled: Boolean = false,
        deviceLocalBudgetBytes: Long = 60L * MEBIBYTE,
        deviceLocalUsageBytes: Long = 0L,
        maxStorageBufferRangeBytes: Long = 1024L * MEBIBYTE
    ) = VulkanDeviceCapabilities.fromJson(
        """
        {
          "deviceName": "Mali-G57",
          "apiVersionMajor": 1,
          "apiVersionMinor": 3,
          "apiVersionPatch": 278,
          "effectiveApiVersionMajor": 1,
          "effectiveApiVersionMinor": 3,
          "effectiveApiVersionPatch": 278,
          "maxComputeWorkGroupInvocations": 256,
          "maxComputeWorkGroupSizeX": 256,
          "maxComputeWorkGroupSizeY": 256,
          "maxComputeWorkGroupSizeZ": 64,
          "maxComputeWorkGroupCountX": 65535,
          "maxComputeWorkGroupCountY": 65535,
          "maxComputeWorkGroupCountZ": 65535,
          "maxComputeSharedMemoryBytes": 32768,
          "maxStorageBufferRangeBytes": $maxStorageBufferRangeBytes,
          "shaderFloat16": $shaderFloat16Supported,
          "shaderFloat16FeatureSupported": $shaderFloat16Supported,
          "shaderFloat16FeatureEnabled": $shaderFloat16Enabled,
          "memoryHeapCount": 1,
          "memoryBudgetExtensionEnabled": true,
          "memoryBudgetQueryAvailable": true,
          "memoryBudgetHeapCount": 1,
          "memoryHeapBudgetBytes": [$deviceLocalBudgetBytes],
          "memoryHeapUsageBytes": [$deviceLocalUsageBytes],
          "memoryHeapFlags": [1]
        }
        """.trimIndent()
    ).toDeviceProfile()

    private companion object {
        const val MEBIBYTE = 1024L * 1024L
        const val GIBIBYTE = 1024L * 1024L * 1024L
    }
}
