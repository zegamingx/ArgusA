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
    fun parsesSubgroupPrecisionAndAccelerationFlags() {
        val capabilities = VulkanDeviceCapabilities.fromJson(
            """
            {
              "deviceName": "Mali-G57",
              "vendorId": 0,
              "deviceId": 123,
              "deviceType": 1,
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
              "subgroupSizeControl": true,
              "computeFullSubgroups": true,
              "minSubgroupSize": 4,
              "maxSubgroupSize": 32,
              "shaderFloat16": true,
              "shaderInt8": true,
              "storageBuffer16BitAccess": true,
              "shaderIntegerDotProduct": true,
              "integerDotProduct8BitSignedAccelerated": true,
              "integerDotProduct4x8BitPackedSignedAccelerated": true,
              "cooperativeMatrixExtension": false,
              "cooperativeMatrix": false,
              "cooperativeMatrixTypes": []
            }
            """.replace("ARM_PLACEHOLDER", "0")
        )

        assertEquals("Mali-G57", capabilities.deviceName)
        assertEquals("1.3.278", capabilities.apiVersion)
        assertEquals(16, capabilities.subgroupSize)
        assertTrue(capabilities.supportsComputeSubgroups)
        assertTrue(capabilities.shaderFloat16)
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
}
