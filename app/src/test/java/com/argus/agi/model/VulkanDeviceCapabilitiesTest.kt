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
    fun rejectsPartialCapabilityPayloadMissingCoreFields() {
        val nameOnly = runCatching {
            VulkanDeviceCapabilities.fromJson("""{ "deviceName": "Mali-G57" }""")
        }.exceptionOrNull()
        assertTrue(nameOnly is IllegalArgumentException)

        val empty = runCatching {
            VulkanDeviceCapabilities.fromJson("{}")
        }.exceptionOrNull()
        assertTrue(empty is IllegalArgumentException)
    }

    @Test
    fun rejectsCapabilityPayloadWithZeroCoreLimits() {
        val failure = runCatching {
            VulkanDeviceCapabilities.fromJson(
                """
                {
                  "deviceName": "Mali-G57",
                  "apiVersionMajor": 1,
                  "apiVersionMinor": 3,
                  "apiVersionPatch": 278,
                  "maxComputeWorkGroupInvocations": 512,
                  "maxComputeWorkGroupSizeX": 512,
                  "maxComputeWorkGroupSizeY": 512,
                  "maxComputeWorkGroupSizeZ": 64,
                  "maxComputeWorkGroupCountX": 65535,
                  "maxComputeWorkGroupCountY": 65535,
                  "maxComputeWorkGroupCountZ": 65535,
                  "maxComputeSharedMemoryBytes": 32768,
                  "maxStorageBufferRangeBytes": 0
                }
                """
            )
        }.exceptionOrNull()

        assertTrue(failure is IllegalArgumentException)
    }

    @Test
    fun parsesUint32MaxWorkGroupCountsFromMaliDevices() {
        // Regression: Mali-G57 (Infinix X6725C) reports maxComputeWorkGroupCount
        // as UINT32_MAX (4294967295) on every axis and a deviceId above
        // Int.MAX_VALUE. optInt truncated those to negative values, the
        // core-field validation then rejected the whole payload, and Vulkan
        // capability parsing failed on device. The values must clamp to
        // Int.MAX_VALUE ("effectively unlimited" / opaque identifier).
        val capabilities = VulkanDeviceCapabilities.fromJson(
            """
            {
              "deviceName": "Mali-G57",
              "deviceId": 2425421840,
              "apiVersionMajor": 1,
              "apiVersionMinor": 3,
              "apiVersionPatch": 278,
              "maxComputeWorkGroupInvocations": 512,
              "maxComputeWorkGroupSizeX": 512,
              "maxComputeWorkGroupSizeY": 512,
              "maxComputeWorkGroupSizeZ": 64,
              "maxComputeWorkGroupCountX": 4294967295,
              "maxComputeWorkGroupCountY": 4294967295,
              "maxComputeWorkGroupCountZ": 4294967295,
              "maxComputeSharedMemoryBytes": 32768,
              "maxStorageBufferRangeBytes": 2147483648
            }
            """
        )

        assertEquals(Int.MAX_VALUE, capabilities.maxComputeWorkGroupCountX)
        assertEquals(Int.MAX_VALUE, capabilities.maxComputeWorkGroupCountY)
        assertEquals(Int.MAX_VALUE, capabilities.maxComputeWorkGroupCountZ)
        // deviceId is an opaque bit-pattern identifier, not a numeric limit:
        // its bits must survive intact (rendered as hex identity).
        assertEquals("90910010", capabilities.deviceId.toUInt().toString(16))
        assertEquals(2147483648L, capabilities.maxStorageBufferRangeBytes)
    }

    @Test
    fun rejectsCapabilityPayloadWithBlankDeviceName() {
        val failure = runCatching {
            VulkanDeviceCapabilities.fromJson(
                """
                {
                  "deviceName": " ",
                  "apiVersionMajor": 1,
                  "apiVersionMinor": 3,
                  "apiVersionPatch": 278,
                  "maxComputeWorkGroupInvocations": 512,
                  "maxComputeWorkGroupSizeX": 512,
                  "maxComputeWorkGroupSizeY": 512,
                  "maxComputeWorkGroupSizeZ": 64,
                  "maxComputeWorkGroupCountX": 65535,
                  "maxComputeWorkGroupCountY": 65535,
                  "maxComputeWorkGroupCountZ": 65535,
                  "maxComputeSharedMemoryBytes": 32768,
                  "maxStorageBufferRangeBytes": 134217728
                }
                """
            )
        }.exceptionOrNull()

        assertTrue(failure is IllegalArgumentException)
    }

    @Test
    fun memoryBudgetRemainsOptionalWhenExtensionIsUnavailable() {
        val capabilities = VulkanDeviceCapabilities.fromJson(
            """
            {
              "deviceName": "Mali-G57",
              "apiVersionMajor": 1,
              "apiVersionMinor": 3,
              "apiVersionPatch": 278,
              "maxComputeWorkGroupInvocations": 512,
              "maxComputeWorkGroupSizeX": 512,
              "maxComputeWorkGroupSizeY": 512,
              "maxComputeWorkGroupSizeZ": 64,
              "maxComputeWorkGroupCountX": 65535,
              "maxComputeWorkGroupCountY": 65535,
              "maxComputeWorkGroupCountZ": 65535,
              "maxComputeSharedMemoryBytes": 32768,
              "maxStorageBufferRangeBytes": 134217728,
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
              "memoryHeapCount": 2,
              "memoryTypeCount": 3,
              "hostVisibleMemoryTypeCount": 2,
              "hostVisibleDeviceLocalMemoryTypeCount": 1,
              "memoryBudgetExtensionSupported": true,
              "memoryBudgetExtensionEnabled": true,
              "memoryBudgetQueryAvailable": true,
              "memoryBudgetHeapCount": 2,
              "memoryHeapBudgetBytes": [1000, 2000],
              "memoryHeapUsageBytes": [250, 700],
              "memoryHeapFlags": [0, 1],
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
        assertEquals(2, capabilities.memoryHeapCount)
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
        assertEquals(listOf(0, 1), capabilities.memoryHeapFlags)
        assertEquals(listOf(1300L), capabilities.deviceLocalMemoryBudgetHeadroomBytes)
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
              "maxComputeWorkGroupInvocations": 512,
              "maxComputeWorkGroupSizeX": 512,
              "maxComputeWorkGroupSizeY": 512,
              "maxComputeWorkGroupSizeZ": 64,
              "maxComputeWorkGroupCountX": 65535,
              "maxComputeWorkGroupCountY": 65535,
              "maxComputeWorkGroupCountZ": 65535,
              "maxComputeSharedMemoryBytes": 32768,
              "maxStorageBufferRangeBytes": 134217728,
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
              "apiVersionMajor": 1,
              "apiVersionMinor": 3,
              "apiVersionPatch": 278,
              "maxComputeWorkGroupInvocations": 512,
              "maxComputeWorkGroupSizeX": 512,
              "maxComputeWorkGroupSizeY": 512,
              "maxComputeWorkGroupSizeZ": 64,
              "maxComputeWorkGroupCountX": 65535,
              "maxComputeWorkGroupCountY": 65535,
              "maxComputeWorkGroupCountZ": 65535,
              "maxComputeSharedMemoryBytes": 32768,
              "maxStorageBufferRangeBytes": 134217728,
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
              "apiVersionPatch": 278,
              "maxComputeWorkGroupInvocations": 512,
              "maxComputeWorkGroupSizeX": 512,
              "maxComputeWorkGroupSizeY": 512,
              "maxComputeWorkGroupSizeZ": 64,
              "maxComputeWorkGroupCountX": 65535,
              "maxComputeWorkGroupCountY": 65535,
              "maxComputeWorkGroupCountZ": 65535,
              "maxComputeSharedMemoryBytes": 32768,
              "maxStorageBufferRangeBytes": 134217728,
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
        assertFalse(capabilities.synchronization2ExtensionSupported)
        assertFalse(capabilities.synchronization2Supported)
        assertFalse(capabilities.synchronization2Enabled)
    }

    @Test
    fun timestampProfilingRequiresValidBitsOnTheSelectedComputeQueue() {
        val capabilities = VulkanDeviceCapabilities.fromJson(
            """
            {
              "deviceName": "Unknown",
              "apiVersionMajor": 1,
              "apiVersionMinor": 3,
              "apiVersionPatch": 278,
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
              "deviceId": 2425421840,
              "apiVersionMajor": 1,
              "apiVersionMinor": 3,
              "apiVersionPatch": 278,
              "driverName": "ARM Proprietary",
              "driverInfo": "v1.r49p1",
              "maxComputeWorkGroupInvocations": 512,
              "maxComputeWorkGroupSizeX": 512,
              "maxComputeWorkGroupSizeY": 512,
              "maxComputeWorkGroupSizeZ": 64,
              "maxComputeWorkGroupCountX": 65535,
              "maxComputeWorkGroupCountY": 65535,
              "maxComputeWorkGroupCountZ": 65535,
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
              "deviceId": 1141178369,
              "apiVersionMajor": 1,
              "apiVersionMinor": 3,
              "apiVersionPatch": 284,
              "driverName": "Qualcomm Technologies Inc. Adreno Vulkan Driver",
              "driverInfo": "Driver Build 72c15dcb23",
              "maxComputeWorkGroupInvocations": 1024,
              "maxComputeWorkGroupSizeX": 512,
              "maxComputeWorkGroupSizeY": 512,
              "maxComputeWorkGroupSizeZ": 64,
              "maxComputeWorkGroupCountX": 65535,
              "maxComputeWorkGroupCountY": 65535,
              "maxComputeWorkGroupCountZ": 65535,
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
              "maxComputeWorkGroupSizeX": 512,
              "maxComputeWorkGroupSizeY": 512,
              "maxComputeWorkGroupSizeZ": 64,
              "maxComputeWorkGroupCountX": 65535,
              "maxComputeWorkGroupCountY": 65535,
              "maxComputeWorkGroupCountZ": 65535,
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
              "deviceId": 3627024400,
              "apiVersionMajor": 1,
              "apiVersionMinor": 3,
              "apiVersionPatch": 278,
              "driverName": "ARM Proprietary",
              "driverInfo": "v1.r49p1",
              "maxComputeWorkGroupInvocations": 1024,
              "maxComputeWorkGroupSizeX": 512,
              "maxComputeWorkGroupSizeY": 512,
              "maxComputeWorkGroupSizeZ": 64,
              "maxComputeWorkGroupCountX": 65535,
              "maxComputeWorkGroupCountY": 65535,
              "maxComputeWorkGroupCountZ": 65535,
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



    @Test
    fun parsesSamsungGalaxyA34MaliG68ShaderCoreProfileWithoutEnablingVendorKernels() {
        val capabilities = VulkanDeviceCapabilities.fromJson(
            """
            {
              "deviceName": "Mali-G68 MC4",
              "vendorId": 5045,
              "deviceId": 2449739792,
              "apiVersionMajor": 1,
              "apiVersionMinor": 3,
              "apiVersionPatch": 278,
              "driverName": "Mali-G68 MC4",
              "driverInfo": "v1.r49p1-03bet0.3666fcaacb504f548df1f5efc729b42d",
              "deviceIdentityPropertiesAvailable": true,
              "deviceUUID": "10100492010000000000000000000000",
              "driverUUID": "4E02E2B0CEBEF9B79A6FD34809684BF3",
              "pipelineCacheUUID": "68586441A476998CBA24892B10AD89AE",
              "maxComputeWorkGroupInvocations": 512,
              "maxComputeWorkGroupSizeX": 512,
              "maxComputeWorkGroupSizeY": 512,
              "maxComputeWorkGroupSizeZ": 512,
              "maxComputeWorkGroupCountX": 65535,
              "maxComputeWorkGroupCountY": 65535,
              "maxComputeWorkGroupCountZ": 65535,
              "maxComputeSharedMemoryBytes": 32768,
              "maxStorageBufferRangeBytes": 2147483648,
              "subgroupSize": 16,
              "subgroupSupportedStages": 48,
              "subgroupSupportedOperations": 511,
              "subgroupSizeControl": true,
              "computeFullSubgroups": true,
              "minSubgroupSize": 16,
              "maxSubgroupSize": 16,
              "maxComputeWorkgroupSubgroups": 32,
              "requiredSubgroupSizeStages": 48,
              "timestampPeriodNanoseconds": 76.9231,
              "timestampComputeAndGraphics": true,
              "shaderFloat16": true,
              "shaderInt8": true,
              "storageBuffer16BitAccess": true,
              "storageBuffer8BitAccess": true,
              "shaderIntegerDotProduct": true,
              "integerDotProduct8BitUnsignedAccelerated": true,
              "integerDotProduct8BitSignedAccelerated": true,
              "integerDotProduct8BitMixedSignednessAccelerated": false,
              "integerDotProduct4x8BitPackedUnsignedAccelerated": true,
              "integerDotProduct4x8BitPackedSignedAccelerated": true,
              "integerDotProduct4x8BitPackedMixedSignednessAccelerated": false,
              "shaderFloat16FeatureSupported": true,
              "shaderFloat16FeatureEnabled": true,
              "shaderInt8FeatureEnabled": false,
              "storageBuffer16BitAccessEnabled": false,
              "storageBuffer8BitAccessEnabled": false,
              "shaderIntegerDotProductEnabled": false,
              "cooperativeMatrixExtension": false,
              "synchronization2ExtensionSupported": false,
              "synchronization2Supported": true,
              "synchronization2Enabled": true,
              "armTensorsExtensionSupported": false,
              "armShaderCorePropertiesExtensionSupported": true,
              "armShaderCoreBuiltinsExtensionSupported": true,
              "armShaderCoreBuiltinsFeatureQueryAvailable": true,
              "armShaderCoreBuiltinsFeatureSupported": true,
              "armShaderCoreBuiltinsFeatureEnabled": false,
              "armShaderCoreCount": 4,
              "armShaderWarpsPerCore": 32,
              "armShaderFmaRate": 32,
              "armShaderPixelRate": 2,
              "armShaderTexelRate": 4
            }
            """
        )

        assertEquals("Mali-G68 MC4", capabilities.deviceName)
        assertEquals(5045, capabilities.vendorId)
        assertEquals("92041010", capabilities.deviceId.toUInt().toString(16))
        assertEquals("1.3.278", capabilities.apiVersion)
        assertEquals("v1.r49p1-03bet0.3666fcaacb504f548df1f5efc729b42d", capabilities.driverInfo)
        assertEquals("10100492010000000000000000000000", capabilities.deviceUuid)
        assertEquals("4E02E2B0CEBEF9B79A6FD34809684BF3", capabilities.driverUuid)
        assertEquals("68586441A476998CBA24892B10AD89AE", capabilities.pipelineCacheUuid)

        assertEquals(512, capabilities.maxComputeWorkGroupInvocations)
        assertEquals(32768L, capabilities.maxComputeSharedMemoryBytes)
        assertEquals(2147483648L, capabilities.maxStorageBufferRangeBytes)
        assertEquals(16, capabilities.subgroupSize)
        assertTrue(capabilities.supportsComputeSubgroups)
        assertTrue(capabilities.supportsComputeSubgroupArithmetic)
        assertTrue(capabilities.supportsComputeSubgroupShuffle)

        assertTrue(capabilities.shaderFloat16FeatureSupported)
        assertTrue(capabilities.shaderFloat16FeatureEnabled)
        assertTrue(capabilities.shaderInt8)
        assertFalse(capabilities.shaderInt8FeatureEnabled)
        assertTrue(capabilities.storageBuffer16BitAccess)
        assertFalse(capabilities.storageBuffer16BitAccessEnabled)
        assertTrue(capabilities.storageBuffer8BitAccess)
        assertFalse(capabilities.storageBuffer8BitAccessEnabled)
        assertTrue(capabilities.hasAcceleratedInt8DotProduct)
        assertFalse(capabilities.canUseIntegerDotProductKernel)
        assertFalse(capabilities.hasUsableCooperativeMatrixConfiguration)
        assertFalse(capabilities.synchronization2ExtensionSupported)
        assertTrue(capabilities.synchronization2Supported)
        assertTrue(capabilities.synchronization2Enabled)
        assertFalse(capabilities.armTensorsExtensionSupported)

        assertTrue(capabilities.hasArmShaderCoreTopology)
        assertTrue(capabilities.hasArmShaderCorePerformanceProfile)
        assertEquals(4, capabilities.armShaderCoreCount)
        assertEquals(32, capabilities.armShaderWarpsPerCore)
        assertEquals(32, capabilities.armShaderFmaRate)
        assertEquals(2, capabilities.armShaderPixelRate)
        assertEquals(4, capabilities.armShaderTexelRate)
        assertFalse(capabilities.armShaderCoreBuiltinsFeatureEnabled)
    }

    @Test
    fun parsesMaliG2UltraArmTensorAndIdentityProfileWithoutEnablingIt() {
        val capabilities = VulkanDeviceCapabilities.fromJson(
            """
            {
              "deviceName": "Mali-G2-Ultra-NX MC12",
              "vendorId": 5045,
              "deviceId": 4042293250,
              "deviceType": 1,
              "apiVersionMajor": 1,
              "apiVersionMinor": 4,
              "apiVersionPatch": 349,
              "effectiveApiVersionMajor": 1,
              "effectiveApiVersionMinor": 4,
              "effectiveApiVersionPatch": 0,
              "driverName": "Mali-G2-Ultra-NX MC12",
              "driverInfo": "v1.r56p0-18eac0.99766b76a7d9937554caf805b4a9a6ca",
              "deviceIdentityPropertiesAvailable": true,
              "deviceUUID": "00080F00010000000000000000000000",
              "driverUUID": "295140483411D0A9C450125E67D65489",
              "pipelineCacheUUID": "16E03C7F6B12BD8DC035C2564FC780F1",
              "maxComputeWorkGroupInvocations": 1024,
              "maxComputeWorkGroupSizeX": 1024,
              "maxComputeWorkGroupSizeY": 1024,
              "maxComputeWorkGroupSizeZ": 1024,
              "maxComputeWorkGroupCountX": 65535,
              "maxComputeWorkGroupCountY": 65535,
              "maxComputeWorkGroupCountZ": 65535,
              "maxComputeSharedMemoryBytes": 32768,
              "maxStorageBufferRangeBytes": 2147483648,
              "subgroupSize": 16,
              "subgroupSupportedStages": 49,
              "subgroupSupportedOperations": 511,
              "subgroupSizeControl": true,
              "minSubgroupSize": 16,
              "maxSubgroupSize": 16,
              "maxComputeWorkgroupSubgroups": 64,
              "timestampPeriodNanoseconds": 0.999001,
              "timestampComputeAndGraphics": true,
              "computeQueueTimestampValidBits": 64,
              "computeTimestampQueriesSupported": true,
              "shaderIntegerDotProduct": true,
              "integerDotProduct8BitUnsignedAccelerated": true,
              "integerDotProduct8BitSignedAccelerated": true,
              "integerDotProduct4x8BitPackedUnsignedAccelerated": true,
              "integerDotProduct4x8BitPackedSignedAccelerated": true,
              "cooperativeMatrixExtension": true,
              "cooperativeMatrix": true,
              "cooperativeMatrixSupportedStages": 32,
              "cooperativeMatrixTypes": [],
              "synchronization2ExtensionSupported": false,
              "synchronization2Supported": true,
              "synchronization2Enabled": true,
              "armTensorsExtensionSupported": true,
              "armTensorsFeatureQueryAvailable": true,
              "armTensorsFeatureSupported": true,
              "armTensorNonPackedSupported": true,
              "armTensorShaderAccessSupported": true,
              "armTensorDynamicIndexingSupported": true,
              "armTensorNonUniformIndexingSupported": false,
              "armTensorUpdateAfterBindSupported": false,
              "armTensorDeviceFeaturesEnabled": false,
              "shaderTensorSupportedStages": 49,
              "armTensorMaxDimensionCount": 4,
              "armTensorMaxElements": 4294967295,
              "armTensorMaxPerDimensionElements": 4294967294,
              "armTensorMaxStride": 4294967295,
              "armTensorMaxSizeBytes": 4294967294,
              "armTensorMaxShaderAccessArrayLength": 16,
              "armTensorMaxShaderAccessSizeBytes": 16,
              "armTensorMaxDescriptorSetStorageTensors": 500000,
              "armTensorMaxPerStageDescriptorSetStorageTensors": 500000,
              "armTensorNonUniformIndexingNative": false
            }
            """
        )

        assertEquals("Mali-G2-Ultra-NX MC12", capabilities.deviceName)
        assertEquals(5045, capabilities.vendorId)
        assertEquals("f0f08002", capabilities.deviceId.toUInt().toString(16))
        assertEquals("1.4.349", capabilities.apiVersion)
        assertEquals(1, capabilities.effectiveApiVersionMajor)
        assertEquals(4, capabilities.effectiveApiVersionMinor)
        assertTrue(capabilities.synchronization2Supported)
        assertTrue(capabilities.synchronization2Enabled)
        assertEquals("00080F00010000000000000000000000", capabilities.deviceUuid)
        assertEquals("295140483411D0A9C450125E67D65489", capabilities.driverUuid)
        assertEquals("16E03C7F6B12BD8DC035C2564FC780F1", capabilities.pipelineCacheUuid)
        assertTrue(capabilities.deviceIdentityPropertiesAvailable)

        assertEquals(1024, capabilities.maxComputeWorkGroupInvocations)
        assertEquals(32768L, capabilities.maxComputeSharedMemoryBytes)
        assertEquals(2147483648L, capabilities.maxStorageBufferRangeBytes)
        assertEquals(16, capabilities.subgroupSize)
        assertEquals(16, capabilities.minSubgroupSize)
        assertEquals(16, capabilities.maxSubgroupSize)
        assertTrue(capabilities.supportsComputeSubgroups)
        assertTrue(capabilities.supportsComputeTimestampQueries)
        assertEquals(0.999001, capabilities.timestampPeriodNanoseconds, 0.000001)

        assertTrue(capabilities.hasAcceleratedInt8DotProduct)
        assertFalse(capabilities.canUseIntegerDotProductKernel)
        assertTrue(capabilities.cooperativeMatrix)
        assertFalse(capabilities.hasUsableCooperativeMatrixConfiguration)

        assertTrue(capabilities.armTensorsExtensionSupported)
        assertTrue(capabilities.armTensorsFeatureQueryAvailable)
        assertTrue(capabilities.armTensorsFeatureSupported)
        assertTrue(capabilities.armTensorNonPackedSupported)
        assertTrue(capabilities.armTensorShaderAccessSupported)
        assertTrue(capabilities.armTensorDynamicIndexingSupported)
        assertFalse(capabilities.armTensorNonUniformIndexingSupported)
        assertFalse(capabilities.armTensorUpdateAfterBindSupported)
        assertEquals(49, capabilities.shaderTensorSupportedStages)
        assertEquals(4, capabilities.armTensorMaxDimensionCount)
        assertEquals(4294967295L, capabilities.armTensorMaxElements)
        assertEquals(4294967294L, capabilities.armTensorMaxPerDimensionElements)
        assertEquals(4294967295L, capabilities.armTensorMaxStride)
        assertEquals(4294967294L, capabilities.armTensorMaxSizeBytes)
        assertEquals(16, capabilities.armTensorMaxShaderAccessArrayLength)
        assertEquals(16, capabilities.armTensorMaxShaderAccessSizeBytes)
        assertEquals(500000, capabilities.armTensorMaxDescriptorSetStorageTensors)
        assertEquals(500000, capabilities.armTensorMaxPerStageDescriptorSetStorageTensors)
        assertFalse(capabilities.armTensorNonUniformIndexingNative)
        assertTrue(capabilities.supportsComputeArmTensors)
        assertFalse(capabilities.armTensorDeviceFeaturesEnabled)
        assertFalse(capabilities.canUseArmTensorCompute)
    }

}
