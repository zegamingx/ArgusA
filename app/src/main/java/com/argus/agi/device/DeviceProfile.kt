package com.argus.agi.device

/**
 * Immutable snapshot of the physical compute device and the Vulkan limits/features
 * reported for it. This is device metadata only: it does not imply that Argus has a
 * compiled, self-tested kernel for any reported feature.
 */
internal data class VulkanApiVersion(
    val major: Int,
    val minor: Int,
    val patch: Int
)

internal data class DeviceIdentity(
    val deviceName: String,
    val vendorId: Int,
    val deviceId: Int,
    val deviceType: Int,
    val apiVersion: VulkanApiVersion,
    val effectiveApiVersion: VulkanApiVersion,
    val driverName: String,
    val driverInfo: String,
    val driverVersionRaw: Long,
    val identityPropertiesAvailable: Boolean,
    val deviceUuid: String,
    val driverUuid: String,
    val pipelineCacheUuid: String
)

internal data class DeviceLimits(
    val maxComputeWorkGroupInvocations: Int,
    val maxComputeWorkGroupSizeX: Int,
    val maxComputeWorkGroupSizeY: Int,
    val maxComputeWorkGroupSizeZ: Int,
    val maxComputeWorkGroupCountX: Int,
    val maxComputeWorkGroupCountY: Int,
    val maxComputeWorkGroupCountZ: Int,
    val maxComputeSharedMemoryBytes: Long,
    val maxStorageBufferRangeBytes: Long
)

/**
 * Feature support and logical-device enablement are separate Vulkan facts.
 */
internal data class DeviceFeatureState(
    val supported: Boolean,
    val enabled: Boolean
) {
    val isUsable: Boolean
        get() = supported && enabled
}

internal data class DeviceFeatures(
    val computeSubgroupsSupported: Boolean,
    val computeSubgroupArithmeticSupported: Boolean,
    val computeSubgroupReductionSupported: Boolean,
    val computeSubgroupShuffleSupported: Boolean,
    val computeSubgroupShuffleRelativeSupported: Boolean,
    val subgroupSizeControl: DeviceFeatureState,
    val shaderFloat16: DeviceFeatureState,
    val shaderInt8: DeviceFeatureState,
    val storageBuffer16BitAccess: DeviceFeatureState,
    val storageBuffer8BitAccess: DeviceFeatureState,
    val integerDotProduct: DeviceFeatureState,
    val acceleratedInt8DotProduct: Boolean,
    val cooperativeMatrix: DeviceFeatureState,
    val armTensors: DeviceFeatureState,
    val armShaderCoreBuiltins: DeviceFeatureState,
    val computeTimestampQueriesSupported: Boolean,
    val synchronization2: DeviceFeatureState
)

internal data class DeviceMemoryHeap(
    val budgetBytes: Long,
    val usageBytes: Long,
    val flags: Int?
) {
    val headroomBytes: Long
        get() {
            val safeBudget = budgetBytes.coerceAtLeast(0L)
            val safeUsage = usageBytes.coerceAtLeast(0L)
            return (safeBudget - safeUsage).coerceAtLeast(0L)
        }

    val isDeviceLocal: Boolean?
        get() = flags?.let { value -> (value and DEVICE_LOCAL_HEAP_FLAG) != 0 }

    private companion object {
        const val DEVICE_LOCAL_HEAP_FLAG = 0x1
    }
}

internal data class DeviceMemoryProfile(
    val memoryPropertiesAvailable: Boolean,
    val memoryHeapCount: Int,
    val memoryTypeCount: Int,
    val hostVisibleMemoryTypeCount: Int,
    val hostVisibleDeviceLocalMemoryTypeCount: Int,
    val budgetExtensionSupported: Boolean,
    val budgetExtensionEnabled: Boolean,
    val budgetQueryAvailable: Boolean,
    val budgetHeapCount: Int,
    val heapMetadataConsistent: Boolean,
    val heaps: List<DeviceMemoryHeap>
) {
    private val hasUsableBudgetQuery: Boolean
        get() = budgetExtensionEnabled && budgetQueryAvailable && heapMetadataConsistent

    val memoryBudgetHeadroomBytes: List<Long>
        get() = if (hasUsableBudgetQuery) {
            heaps.map(DeviceMemoryHeap::headroomBytes)
        } else {
            emptyList()
        }

    /**
     * Prefer device-local heaps. If a driver omits heap flags, use all queried heaps
     * conservatively rather than guessing which heap is device-local.
     */
    val deviceLocalMemoryBudgetHeadroomBytes: List<Long>
        get() {
            if (!hasUsableBudgetQuery || heaps.isEmpty()) return emptyList()

            val heapFlagsAvailable = heaps.all { heap -> heap.flags != null }
            val selectedHeaps = if (heapFlagsAvailable) {
                heaps.filter { heap -> heap.isDeviceLocal == true }
            } else {
                heaps
            }
            return selectedHeaps.map(DeviceMemoryHeap::headroomBytes)
        }
}

/**
 * Stable, backend-independent view of device facts used by runtime planning.
 *
 * Kernel verification remains the responsibility of the compute backend. A
 * reported feature is never treated as proof that a corresponding kernel exists.
 */
internal data class DeviceProfile(
    val identity: DeviceIdentity,
    val limits: DeviceLimits,
    val memory: DeviceMemoryProfile,
    val features: DeviceFeatures
)
