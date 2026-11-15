package com.argus.agi.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class VulkanDeviceCapabilitiesTest {

    @Test
    fun parsesGpuTimestampPerformanceMetrics() {
        val metrics = VulkanGpuPerformanceMetrics.fromJson(
            """
            {
              "timestampProfilingAvailable": true,
              "timestampedDispatchCount": 12,
              "lastAdamwGpuTimeNanoseconds": 48000,
              "lastLinearGpuTimeNanoseconds": 9200,
              "lastFp16LinearGpuTimeNanoseconds": 6100,
              "totalGpuTimeNanoseconds": 90000
            }
            """
        )

        assertTrue(metrics.timestampProfilingAvailable)
        assertEquals(12L, metrics.timestampedDispatchCount)
        assertEquals(48000L, metrics.lastAdamwGpuTimeNanoseconds)
        assertEquals(9200L, metrics.lastLinearGpuTimeNanoseconds)
        assertEquals(6100L, metrics.lastFp16LinearGpuTimeNanoseconds)
        assertEquals(90000L, metrics.totalGpuTimeNanoseconds)
    }

    @Test
    fun missingGpuTimestampMetricsRemainSafelyUnavailable() {
        val metrics = VulkanGpuPerformanceMetrics.fromJson("{}")

        assertFalse(metrics.timestampProfilingAvailable)
        assertEquals(0L, metrics.timestampedDispatchCount)
        assertEquals(0L, metrics.totalGpuTimeNanoseconds)
    }


    @Test
    fun memoryBudgetRemainsOptionalWhenExtensionIsUnavailable() {
        val capabilities = VulkanDeviceCapabilities.fromJson(
            """
            {
              "deviceName": "Mali-G57",
              "memoryPropertiesAvailable": true,
              "memoryHeapCount": 1,
              "memoryTypeCount": 2,
              "hostVisibleMemoryTypeCount": 1,
              "hostVisibleDeviceLocalMemoryTypeCount": 1,
              "memoryBudgetExtensionSupported": false,
              "memoryBudgetExtensionEnabled": false,
              "memoryBudgetQueryAvailable": false,
              "memoryBudgetHeapCount": 0,
              "memoryHeapBudgetBytes": [],
              "memoryHeapUsageBytes": []
            }
            """
        )

        assertFalse(capabilities.memoryBudgetExtensionSupported)
        assertFalse(capabilities.memoryBudgetExtensionEnabled)
        assertFalse(capabilities.memoryBudgetQueryAvailable)
        assertTrue(capabilities.memoryHeapBudgetBytes.isEmpty())
        assertTrue(capabilities.memoryHeapUsageBytes.isEmpty())
        assertTrue(capabilities.memoryBudgetHeadroomBytes.isEmpty())
    }

    @Test
    fun parsesSubgroupPrecisionAndAccelerationFlags() {
        val capabilities = VulkanDeviceCapabilities.fromJson(
            """
            {
              "deviceName": "Mali-G57",
              "vendorId": 0,
              "deviceId": 123,
              "deviceType": 1,
              "memoryPropertiesAvailable": true,
              "memoryHeapCount": 1,
              "memoryTypeCount": 3,
              "hostVisibleMemoryTypeCount": 2,
              "hostVisibleDeviceLocalMemoryTypeCount": 1,
              "memoryBudgetExtensionSupported": true,
              "memoryBudgetExtensionEnabled": true,
              "memoryBudgetQueryAvailable": true,
              "memoryBudgetHeapCount": 2,
              "memoryHeapBudgetBytes": [1000, 2000],
              "memoryHeapUsageBytes": [250, 700],
              "apiVersionMajor": 1,
              "apiVersionMinor": 3,
              "apiVersionPatch": 278,
              "driverName": "Mali",
              "driverInfo": "49.1.0",
              "driverVersionRaw": 123456,
              "maxComputeWorkGroupInvocations": 512,
              "maxComputeWorkGroupSizeX": 512,
              "maxComputeWorkGroupSizeY": 512,
              "maxComputeWorkGroupSizeZ": 64,
              "maxComputeWorkGroupCountX": 65535,
              "maxComputeWorkGroupCountY": 65535,
              "maxComputeWorkGroupCountZ": 65535,
              "maxComputeSharedMemoryBytes": 32768,
              "maxStorageBufferRangeBytes": 134217728,
              "subgroupSize": 16,
              "subgroupSupportedStages": 32,
              "subgroupSupportedOperations": 511,
              "computeQueueTimestampValidBits": 64,
              "timestampPeriodNanoseconds": 1.0,
              "timestampComputeAndGraphics": true,
              "computeTimestampQueriesSupported": true,
              "effectiveApiVersionMajor": 1,
              "effectiveApiVersionMinor": 3,
              "effectiveApiVersionPatch": 278,
              "subgroupSizeControl": true,
              "computeFullSubgroups": true,
              "minSubgroupSize": 4,
              "maxSubgroupSize": 32,
              "shaderFloat16": true,
              "shaderFloat16FeatureSupported": true,
              "shaderFloat16FeatureEnabled": true,
              "shaderInt8": true,
              "storageBuffer16BitAccess": true,
              "shaderIntegerDotProduct": true,
              "integerDotProduct8BitSignedAccelerated": true,
              "integerDotProduct4x8BitPackedSignedAccelerated": true,
              "cooperativeMatrixExtension": false,
              "cooperativeMatrix": false,
              "cooperativeMatrixTypes": []
            }
            """
        )

        assertEquals("Mali-G57", capabilities.deviceName)
        assertTrue(capabilities.memoryPropertiesAvailable)
        assertEquals(1, capabilities.memoryHeapCount)
        assertEquals(3, capabilities.memoryTypeCount)
        assertEquals(2, capabilities.hostVisibleMemoryTypeCount)
        assertEquals(1, capabilities.hostVisibleDeviceLocalMemoryTypeCount)
        assertTrue(capabilities.memoryBudgetExtensionSupported)
        assertTrue(capabilities.memoryBudgetExtensionEnabled)
        assertTrue(capabilities.memoryBudgetQueryAvailable)
        assertEquals(2, capabilities.memoryBudgetHeapCount)
        assertEquals(listOf(1000L, 2000L), capabilities.memoryHeapBudgetBytes)
        assertEquals(listOf(250L, 700L), capabilities.memoryHeapUsageBytes)
        assertEquals(listOf(750L, 1300L), capabilities.memoryBudgetHeadroomBytes)
        assertEquals("1.3.278", capabilities.apiVersion)
        assertEquals(16, capabilities.subgroupSize)
        assertTrue(capabilities.supportsComputeSubgroups)
        assertTrue(capabilities.supportsComputeSubgroupArithmetic)
        assertTrue(capabilities.supportsComputeSubgroupReduction)
        assertTrue(capabilities.supportsComputeSubgroupShuffle)
        assertTrue(capabilities.supportsComputeSubgroupShuffleRelative)
        assertTrue(capabilities.supportsComputeTimestampQueries)
        assertEquals(64, capabilities.computeQueueTimestampValidBits)
        assertEquals(1.3, capabilities.effectiveApiVersionMajor +
            capabilities.effectiveApiVersionMinor / 10.0, 0.0001)
        assertTrue(capabilities.shaderFloat16)
        assertTrue(capabilities.shaderFloat16FeatureSupported)
        assertTrue(capabilities.shaderFloat16FeatureEnabled)
        assertTrue(capabilities.shaderInt8)
        assertTrue(capabilities.hasAcceleratedInt8DotProduct)
        assertFalse(capabilities.hasUsableCooperativeMatrixConfiguration)
    }

    @Test
    fun cooperativeMatrixIsUsableOnlyWhenComputeStageAndConfigurationsExist() {
        val capabilities = VulkanDeviceCapabilities.fromJson(
            """
            {
              "deviceName": "Adreno",
              "apiVersionMajor": 1,
              "apiVersionMinor": 4,
              "apiVersionPatch": 351,
              "cooperativeMatrixExtension": true,
              "cooperativeMatrix": true,
              "cooperativeMatrixSupportedStages": 32,
              "cooperativeMatrixEnabled": true,
              "cooperativeMatrixTypes": [
                {
                  "mSize": 16,
                  "nSize": 16,
                  "kSize": 16,
                  "aType": 1,
                  "bType": 1,
                  "cType": 1,
                  "resultType": 1,
                  "saturatingAccumulation": false,
                  "scope": 3
                }
              ]
            }
            """
        )

        assertTrue(capabilities.hasSupportedCooperativeMatrixConfiguration)
        assertTrue(capabilities.hasUsableCooperativeMatrixConfiguration)
        assertEquals(16, capabilities.cooperativeMatrixTypes.single().mSize)
    }

    @Test
    fun cooperativeMatrixWithoutComputeSupportIsNotSelected() {
        val capabilities = VulkanDeviceCapabilities.fromJson(
            """
            {
              "deviceName": "Unknown",
              "cooperativeMatrix": true,
              "cooperativeMatrixSupportedStages": 16,
              "cooperativeMatrixTypes": [
                {
                  "mSize": 16,
                  "nSize": 16,
                  "kSize": 16,
                  "aType": 1,
                  "bType": 1,
                  "cType": 1,
                  "resultType": 1,
                  "saturatingAccumulation": false,
                  "scope": 3
                }
              ]
            }
            """
        )

        assertFalse(capabilities.hasUsableCooperativeMatrixConfiguration)
    }
    @Test
    fun reportedCapabilitiesDoNotImplyLogicalDeviceFeaturesAreEnabled() {
        val capabilities = VulkanDeviceCapabilities.fromJson(
            """
            {
              "deviceName": "Mali-G57",
              "apiVersionMajor": 1,
              "apiVersionMinor": 2,
              "effectiveApiVersionMajor": 1,
              "effectiveApiVersionMinor": 2,
              "subgroupSize": 16,
              "subgroupSupportedStages": 32,
              "subgroupSupportedOperations": 20,
              "shaderIntegerDotProduct": true,
              "integerDotProduct8BitSignedAccelerated": true,
              "cooperativeMatrix": true,
              "cooperativeMatrixSupportedStages": 32,
              "cooperativeMatrixTypes": [
                {
                  "mSize": 16,
                  "nSize": 16,
                  "kSize": 16,
                  "aType": 1,
                  "bType": 1,
                  "cType": 1,
                  "resultType": 1,
                  "saturatingAccumulation": false,
                  "scope": 3
                }
              ],
              "computeQueueTimestampValidBits": 0,
              "timestampPeriodNanoseconds": 1.0
            }
            """
        )

        assertTrue(capabilities.supportsComputeSubgroupArithmetic)
        assertTrue(capabilities.supportsComputeSubgroupReduction)
        assertTrue(capabilities.supportsComputeSubgroupShuffle)
        assertFalse(capabilities.canUseIntegerDotProductKernel)
        assertTrue(capabilities.hasAcceleratedInt8DotProduct)
        assertTrue(capabilities.hasSupportedCooperativeMatrixConfiguration)
        assertFalse(capabilities.hasUsableCooperativeMatrixConfiguration)
        assertFalse(capabilities.supportsComputeTimestampQueries)
        assertFalse(capabilities.shaderIntegerDotProductEnabled)
        assertFalse(capabilities.cooperativeMatrixEnabled)
    }

    @Test
    fun timestampProfilingRequiresValidBitsOnTheSelectedComputeQueue() {
        val capabilities = VulkanDeviceCapabilities.fromJson(
            """
            {
              "deviceName": "Unknown",
              "subgroupSize": 16,
              "subgroupSupportedStages": 32,
              "computeQueueTimestampValidBits": 0,
              "computeTimestampQueriesSupported": false,
              "timestampPeriodNanoseconds": 1.0
            }
            """
        )

        assertFalse(capabilities.supportsComputeTimestampQueries)
    }


    @Test
    fun parsesRepresentativeAndroidGpuProfilesWithoutVendorSpecificAssumptions() {
        val maliG57 = VulkanDeviceCapabilities.fromJson(
            """
            {
              "deviceName": "Mali-G57",
              "vendorId": 5045,
              "deviceId": 2425415696,
              "apiVersionMajor": 1,
              "apiVersionMinor": 3,
              "apiVersionPatch": 278,
              "driverName": "ARM Proprietary",
              "driverInfo": "v1.r49p1",
              "maxComputeWorkGroupInvocations": 512,
              "maxComputeSharedMemoryBytes": 32768,
              "maxStorageBufferRangeBytes": 2147483648,
              "subgroupSize": 16,
              "subgroupSupportedStages": 32,
              "shaderFloat16": true,
              "storageBuffer16BitAccess": true,
              "shaderIntegerDotProduct": true,
              "integerDotProduct8BitUnsignedAccelerated": true,
              "integerDotProduct8BitSignedAccelerated": true,
              "integerDotProduct4x8BitPackedUnsignedAccelerated": true,
              "integerDotProduct4x8BitPackedSignedAccelerated": true
            }
            """
        )
        val adreno830 = VulkanDeviceCapabilities.fromJson(
            """
            {
              "deviceName": "Adreno 830",
              "vendorId": 20803,
              "deviceId": 1145378817,
              "apiVersionMajor": 1,
              "apiVersionMinor": 3,
              "apiVersionPatch": 284,
              "driverName": "Qualcomm Technologies Inc. Adreno Vulkan Driver",
              "driverInfo": "Driver Build 72c15dcb23",
              "maxComputeWorkGroupInvocations": 1024,
              "maxComputeSharedMemoryBytes": 32768,
              "maxStorageBufferRangeBytes": 2147483647,
              "subgroupSize": 64,
              "subgroupSupportedStages": 32,
              "shaderIntegerDotProduct": true,
              "integerDotProduct8BitUnsignedAccelerated": false,
              "integerDotProduct8BitSignedAccelerated": false,
              "integerDotProduct4x8BitPackedUnsignedAccelerated": true,
              "integerDotProduct4x8BitPackedSignedAccelerated": true,
              "integerDotProduct4x8BitPackedMixedSignednessAccelerated": true,
              "cooperativeMatrixExtension": true,
              "cooperativeMatrix": true,
              "cooperativeMatrixSupportedStages": 32
            }
            """
        )
        val xclipse940 = VulkanDeviceCapabilities.fromJson(
            """
            {
              "deviceName": "Samsung Xclipse 940",
              "vendorId": 5197,
              "deviceId": 39846400,
              "apiVersionMajor": 1,
              "apiVersionMinor": 3,
              "apiVersionPatch": 304,
              "driverName": "Samsung Proprietary driver",
              "driverInfo": "SPAL Driver version 25.3.3",
              "maxComputeWorkGroupInvocations": 1024,
              "maxComputeSharedMemoryBytes": 32768,
              "maxStorageBufferRangeBytes": 4294967295,
              "subgroupSize": 64,
              "subgroupSupportedStages": 32,
              "shaderIntegerDotProduct": true,
              "integerDotProduct8BitUnsignedAccelerated": true,
              "integerDotProduct8BitSignedAccelerated": true,
              "integerDotProduct4x8BitPackedUnsignedAccelerated": true,
              "integerDotProduct4x8BitPackedSignedAccelerated": true
            }
            """
        )
        val maliG925 = VulkanDeviceCapabilities.fromJson(
            """
            {
              "deviceName": "Mali-G925-Immortalis MC12",
              "vendorId": 5045,
              "deviceId": 3627028496,
              "apiVersionMajor": 1,
              "apiVersionMinor": 3,
              "apiVersionPatch": 278,
              "driverName": "ARM Proprietary",
              "driverInfo": "v1.r49p1",
              "maxComputeWorkGroupInvocations": 1024,
              "maxComputeSharedMemoryBytes": 32768,
              "maxStorageBufferRangeBytes": 2147483648,
              "subgroupSize": 16,
              "subgroupSupportedStages": 32,
              "shaderIntegerDotProduct": true,
              "integerDotProduct8BitUnsignedAccelerated": true,
              "integerDotProduct8BitSignedAccelerated": true,
              "cooperativeMatrixExtension": true,
              "cooperativeMatrix": true,
              "cooperativeMatrixSupportedStages": 32
            }
            """
        )

        assertEquals(5045, maliG57.vendorId)
        assertEquals("90910010", maliG57.deviceId.toUInt().toString(16))
        assertEquals("1.3.278", maliG57.apiVersion)
        assertEquals(16, maliG57.subgroupSize)
        assertEquals(512, maliG57.maxComputeWorkGroupInvocations)
        assertTrue(maliG57.hasAcceleratedInt8DotProduct)
        assertFalse(maliG57.canUseIntegerDotProductKernel)

        assertEquals(20803, adreno830.vendorId)
        assertEquals("44050001", adreno830.deviceId.toUInt().toString(16))
        assertEquals("1.3.284", adreno830.apiVersion)
        assertEquals(64, adreno830.subgroupSize)
        assertEquals(1024, adreno830.maxComputeWorkGroupInvocations)
        assertTrue(adreno830.hasAcceleratedInt8DotProduct)
        assertFalse(adreno830.canUseIntegerDotProductKernel)
        assertFalse(adreno830.hasUsableCooperativeMatrixConfiguration)

        assertEquals(5197, xclipse940.vendorId)
        assertEquals("2600200", xclipse940.deviceId.toUInt().toString(16))
        assertEquals("1.3.304", xclipse940.apiVersion)
        assertEquals(64, xclipse940.subgroupSize)
        assertEquals(1024, xclipse940.maxComputeWorkGroupInvocations)
        assertEquals(4294967295L, xclipse940.maxStorageBufferRangeBytes)
        assertTrue(xclipse940.hasAcceleratedInt8DotProduct)
        assertFalse(xclipse940.canUseIntegerDotProductKernel)

        assertEquals(5045, maliG925.vendorId)
        assertEquals("d8300010", maliG925.deviceId.toUInt().toString(16))
        assertEquals("1.3.278", maliG925.apiVersion)
        assertEquals(16, maliG925.subgroupSize)
        assertEquals(1024, maliG925.maxComputeWorkGroupInvocations)
        assertTrue(maliG925.hasAcceleratedInt8DotProduct)
        assertFalse(maliG925.canUseIntegerDotProductKernel)
        assertFalse(maliG925.hasUsableCooperativeMatrixConfiguration)
    }

}
