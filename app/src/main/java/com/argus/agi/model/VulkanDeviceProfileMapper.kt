package com.argus.agi.model

import com.argus.agi.device.DeviceFeatureState
import com.argus.agi.device.DeviceFeatures
import com.argus.agi.device.DeviceIdentity
import com.argus.agi.device.DeviceLimits
import com.argus.agi.device.DeviceMemoryHeap
import com.argus.agi.device.DeviceMemoryProfile
import com.argus.agi.device.DeviceProfile
import com.argus.agi.device.VulkanApiVersion

/**
 * Adapts the existing native Vulkan capability report to the stable device contract.
 *
 * The legacy report remains the native/JNI parsing boundary while consumers migrate
 * from the model package to device/runtime contracts incrementally.
 */
internal fun VulkanDeviceCapabilities.toDeviceProfile(): DeviceProfile {
    val declaredHeapCount = memoryBudgetHeapCount.takeIf { count -> count > 0 }
        ?: memoryHeapCount
    val heapMetadataConsistent =
        memoryHeapBudgetBytes.size == memoryHeapUsageBytes.size &&
            (memoryHeapFlags.isEmpty() || memoryHeapFlags.size == memoryHeapBudgetBytes.size) &&
            (declaredHeapCount == 0 || declaredHeapCount == memoryHeapBudgetBytes.size)

    val heaps = if (heapMetadataConsistent) {
        memoryHeapBudgetBytes.indices.map { index ->
            DeviceMemoryHeap(
                budgetBytes = memoryHeapBudgetBytes[index],
                usageBytes = memoryHeapUsageBytes[index],
                flags = memoryHeapFlags.getOrNull(index)
            )
        }
    } else {
        emptyList()
    }

    return DeviceProfile(
        identity = DeviceIdentity(
            deviceName = deviceName,
            vendorId = vendorId,
            deviceId = deviceId,
            deviceType = deviceType,
            apiVersion = VulkanApiVersion(
                major = apiVersionMajor,
                minor = apiVersionMinor,
                patch = apiVersionPatch
            ),
            effectiveApiVersion = VulkanApiVersion(
                major = effectiveApiVersionMajor,
                minor = effectiveApiVersionMinor,
                patch = effectiveApiVersionPatch
            ),
            driverName = driverName,
            driverInfo = driverInfo,
            driverVersionRaw = driverVersionRaw,
            identityPropertiesAvailable = deviceIdentityPropertiesAvailable,
            deviceUuid = deviceUuid,
            driverUuid = driverUuid,
            pipelineCacheUuid = pipelineCacheUuid
        ),
        limits = DeviceLimits(
            maxComputeWorkGroupInvocations = maxComputeWorkGroupInvocations,
            maxComputeWorkGroupSizeX = maxComputeWorkGroupSizeX,
            maxComputeWorkGroupSizeY = maxComputeWorkGroupSizeY,
            maxComputeWorkGroupSizeZ = maxComputeWorkGroupSizeZ,
            maxComputeWorkGroupCountX = maxComputeWorkGroupCountX,
            maxComputeWorkGroupCountY = maxComputeWorkGroupCountY,
            maxComputeWorkGroupCountZ = maxComputeWorkGroupCountZ,
            maxComputeSharedMemoryBytes = maxComputeSharedMemoryBytes,
            maxStorageBufferRangeBytes = maxStorageBufferRangeBytes
        ),
        memory = DeviceMemoryProfile(
            memoryPropertiesAvailable = memoryPropertiesAvailable,
            memoryHeapCount = memoryHeapCount,
            memoryTypeCount = memoryTypeCount,
            hostVisibleMemoryTypeCount = hostVisibleMemoryTypeCount,
            hostVisibleDeviceLocalMemoryTypeCount = hostVisibleDeviceLocalMemoryTypeCount,
            budgetExtensionSupported = memoryBudgetExtensionSupported,
            budgetExtensionEnabled = memoryBudgetExtensionEnabled,
            budgetQueryAvailable = memoryBudgetQueryAvailable,
            budgetHeapCount = memoryBudgetHeapCount,
            heapMetadataConsistent = heapMetadataConsistent,
            heaps = heaps
        ),
        features = DeviceFeatures(
            computeSubgroupsSupported = supportsComputeSubgroups,
            computeSubgroupArithmeticSupported = supportsComputeSubgroupArithmetic,
            computeSubgroupReductionSupported = supportsComputeSubgroupReduction,
            computeSubgroupShuffleSupported = supportsComputeSubgroupShuffle,
            computeSubgroupShuffleRelativeSupported = supportsComputeSubgroupShuffleRelative,
            subgroupSizeControl = DeviceFeatureState(
                supported = subgroupSizeControl,
                enabled = subgroupSizeControlEnabled
            ),
            shaderFloat16 = DeviceFeatureState(
                supported = shaderFloat16 || shaderFloat16FeatureSupported,
                enabled = shaderFloat16FeatureEnabled
            ),
            shaderInt8 = DeviceFeatureState(
                supported = shaderInt8,
                enabled = shaderInt8FeatureEnabled
            ),
            storageBuffer16BitAccess = DeviceFeatureState(
                supported = storageBuffer16BitAccess,
                enabled = storageBuffer16BitAccessEnabled
            ),
            storageBuffer8BitAccess = DeviceFeatureState(
                supported = storageBuffer8BitAccess,
                enabled = storageBuffer8BitAccessEnabled
            ),
            integerDotProduct = DeviceFeatureState(
                supported = shaderIntegerDotProduct,
                enabled = shaderIntegerDotProductEnabled
            ),
            acceleratedInt8DotProduct = hasAcceleratedInt8DotProduct,
            cooperativeMatrix = DeviceFeatureState(
                supported = hasSupportedCooperativeMatrixConfiguration,
                enabled = cooperativeMatrixEnabled
            ),
            armTensors = DeviceFeatureState(
                supported = supportsComputeArmTensors,
                enabled = armTensorDeviceFeaturesEnabled
            ),
            armShaderCoreBuiltins = DeviceFeatureState(
                supported = armShaderCoreBuiltinsExtensionSupported &&
                    armShaderCoreBuiltinsFeatureQueryAvailable &&
                    armShaderCoreBuiltinsFeatureSupported,
                enabled = armShaderCoreBuiltinsFeatureEnabled
            ),
            computeTimestampQueriesSupported = supportsComputeTimestampQueries,
            synchronization2 = DeviceFeatureState(
                supported = synchronization2Supported,
                enabled = synchronization2Enabled
            )
        )
    )
}
