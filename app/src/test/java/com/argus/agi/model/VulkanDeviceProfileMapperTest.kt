package com.argus.agi.model

import com.argus.agi.device.VulkanApiVersion
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class VulkanDeviceProfileMapperTest {

    @Test
    fun mapsIdentityLimitsFeaturesAndDeviceLocalMemoryBudget() {
        val profile = VulkanDeviceCapabilities.fromJson(
            """
            {
              "deviceName": "Mali-G57",
              "vendorId": 5045,
              "deviceId": 87,
              "deviceType": 1,
              "apiVersionMajor": 1,
              "apiVersionMinor": 3,
              "apiVersionPatch": 278,
              "effectiveApiVersionMajor": 1,
              "effectiveApiVersionMinor": 3,
              "effectiveApiVersionPatch": 278,
              "driverName": "ARM",
              "driverInfo": "49.1.0",
              "driverVersionRaw": 490100,
              "deviceIdentityPropertiesAvailable": true,
              "deviceUUID": "device-uuid",
              "driverUUID": "driver-uuid",
              "pipelineCacheUUID": "cache-uuid",
              "maxComputeWorkGroupInvocations": 256,
              "maxComputeWorkGroupSizeX": 256,
              "maxComputeWorkGroupSizeY": 256,
              "maxComputeWorkGroupSizeZ": 64,
              "maxComputeWorkGroupCountX": 65535,
              "maxComputeWorkGroupCountY": 65535,
              "maxComputeWorkGroupCountZ": 65535,
              "maxComputeSharedMemoryBytes": 32768,
              "maxStorageBufferRangeBytes": 1073741824,
              "subgroupSize": 32,
              "subgroupSupportedStages": 32,
              "subgroupSupportedOperations": 20,
              "subgroupSizeControl": true,
              "subgroupSizeControlEnabled": true,
              "shaderFloat16": true,
              "shaderFloat16FeatureSupported": true,
              "shaderFloat16FeatureEnabled": false,
              "shaderInt8": true,
              "shaderInt8FeatureEnabled": true,
              "storageBuffer16BitAccess": true,
              "storageBuffer16BitAccessEnabled": true,
              "storageBuffer8BitAccess": true,
              "storageBuffer8BitAccessEnabled": false,
              "shaderIntegerDotProduct": true,
              "shaderIntegerDotProductEnabled": false,
              "integerDotProduct8BitSignedAccelerated": true,
              "memoryPropertiesAvailable": true,
              "memoryHeapCount": 2,
              "memoryBudgetExtensionSupported": true,
              "memoryBudgetExtensionEnabled": true,
              "memoryBudgetQueryAvailable": true,
              "memoryBudgetHeapCount": 2,
              "memoryHeapBudgetBytes": [1000, 2000],
              "memoryHeapUsageBytes": [400, 500],
              "memoryHeapFlags": [1, 0],
              "synchronization2Supported": true,
              "synchronization2Enabled": true
            }
            """.trimIndent()
        ).toDeviceProfile()

        assertEquals("Mali-G57", profile.identity.deviceName)
        assertEquals(5045, profile.identity.vendorId)
        assertEquals(VulkanApiVersion(1, 3, 278), profile.identity.apiVersion)
        assertEquals(VulkanApiVersion(1, 3, 278), profile.identity.effectiveApiVersion)
        assertEquals(1073741824L, profile.limits.maxStorageBufferRangeBytes)
        assertEquals(listOf(600L), profile.memory.deviceLocalMemoryBudgetHeadroomBytes)

        assertTrue(profile.features.computeSubgroupsSupported)
        assertTrue(profile.features.computeSubgroupArithmeticSupported)
        assertTrue(profile.features.computeSubgroupShuffleSupported)
        assertTrue(profile.features.subgroupSizeControl.isUsable)
        assertTrue(profile.features.shaderFloat16.supported)
        assertFalse(profile.features.shaderFloat16.enabled)
        assertFalse(profile.features.shaderFloat16.isUsable)
        assertTrue(profile.features.shaderInt8.isUsable)
        assertTrue(profile.features.acceleratedInt8DotProduct)
        assertFalse(profile.features.integerDotProduct.isUsable)
        assertTrue(profile.features.synchronization2.isUsable)
    }

    @Test
    fun rejectsIncompleteHeapArraysInsteadOfPublishingPartialBudget() {
        val profile = VulkanDeviceCapabilities.fromJson(
            """
            {
              "memoryHeapCount": 2,
              "memoryBudgetExtensionEnabled": true,
              "memoryBudgetQueryAvailable": true,
              "memoryBudgetHeapCount": 2,
              "memoryHeapBudgetBytes": [1000, 2000],
              "memoryHeapUsageBytes": [400],
              "memoryHeapFlags": [1, 0]
            }
            """.trimIndent()
        ).toDeviceProfile()

        assertFalse(profile.memory.heapMetadataConsistent)
        assertTrue(profile.memory.memoryBudgetHeadroomBytes.isEmpty())
        assertTrue(profile.memory.deviceLocalMemoryBudgetHeadroomBytes.isEmpty())
    }

    @Test
    fun doesNotUseMemoryBudgetWhenExtensionWasNotEnabled() {
        val profile = VulkanDeviceCapabilities.fromJson(
            """
            {
              "memoryHeapCount": 1,
              "memoryBudgetExtensionSupported": true,
              "memoryBudgetExtensionEnabled": false,
              "memoryBudgetQueryAvailable": true,
              "memoryBudgetHeapCount": 1,
              "memoryHeapBudgetBytes": [1000],
              "memoryHeapUsageBytes": [400],
              "memoryHeapFlags": [1]
            }
            """.trimIndent()
        ).toDeviceProfile()

        assertTrue(profile.memory.heaps.isNotEmpty())
        assertTrue(profile.memory.deviceLocalMemoryBudgetHeadroomBytes.isEmpty())
    }
}
