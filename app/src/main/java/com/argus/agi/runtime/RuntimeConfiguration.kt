package com.argus.agi.runtime

import com.argus.agi.device.DeviceProfile

/**
 * Live resource and thermal observations. This is a per-planning snapshot, not
 * persisted model metadata and must be refreshed before each training batch.
 */
internal data class RuntimeResourceSnapshot(
    val availableAppHeapBytes: Long,
    val maximumAppHeapBytes: Long,
    val availableSystemMemoryBytes: Long,
    val systemLowMemory: Boolean,
    val thermalStatus: Int,
    val temperatureCelsius: Float,
    val isPowerSaveMode: Boolean,
    val isCharging: Boolean,
    val batteryPct: Int,
    val memoryRecovery: Boolean
)

/**
 * Read-only dimensions copied from the active model configuration. Runtime planning
 * may reduce work performed per step, but it may never mutate these model dimensions.
 */
internal data class RuntimeModelConstraints(
    val configuredBatchSize: Int,
    val configuredContextLength: Int,
    val dModel: Int,
    val feedForwardSize: Int,
    val vocabularySize: Int,
    val layerCount: Int,
    val headCount: Int
) {
    init {
        require(configuredBatchSize in 1..8)
        require(configuredContextLength >= 2)
        require(dModel > 0 && feedForwardSize > 0 && vocabularySize > 0)
        require(layerCount > 0 && headCount > 0)
        require(dModel % headCount == 0)
    }
}

internal enum class RuntimeComputeBackend {
    VULKAN,
    CPU_FALLBACK
}

internal enum class RuntimeDeferReason {
    THERMAL_LIMIT,
    INSUFFICIENT_MEMORY
}

internal data class RuntimeDispatchLimits(
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

internal data class RuntimeConfiguration(
    val computeBackend: RuntimeComputeBackend,
    val deviceName: String?,
    val effectiveVulkanApiVersion: String?,
    val dispatchLimits: RuntimeDispatchLimits?,
    val fp16ComputeUsable: Boolean,
    val configuredTrainingBatchSize: Int,
    val effectiveTrainingBatchSize: Int,
    val configuredContextLength: Int,
    val effectiveContextLength: Int,
    val trainingBatchDelayMillis: Long,
    val tensorArenaBudget: DeviceAdaptiveRuntimePolicy.TensorArenaBudget,
    val deferReason: RuntimeDeferReason?
) {
    val canTrain: Boolean
        get() = deferReason == null &&
            effectiveTrainingBatchSize > 0 &&
            effectiveContextLength >= 2
}

/**
 * Produces one immutable runtime plan from actual device facts and current resource
 * observations. It deliberately receives no mutable model or checkpoint object.
 */
internal object RuntimeConfigurationPlanner {

    fun create(
        deviceProfile: DeviceProfile?,
        resources: RuntimeResourceSnapshot,
        model: RuntimeModelConstraints,
        vulkanRuntimeReady: Boolean,
        fp16KernelVerified: Boolean
    ): RuntimeConfiguration {
        val batchLimit = DeviceAdaptiveRuntimePolicy.trainingBatchLimit(
            configuredBatchSize = model.configuredBatchSize,
            thermalStatus = resources.thermalStatus,
            availableHeapBytes = resources.availableAppHeapBytes,
            maximumHeapBytes = resources.maximumAppHeapBytes,
            availableSystemMemoryBytes = resources.availableSystemMemoryBytes,
            systemLowMemory = resources.systemLowMemory,
            isPowerSaveMode = resources.isPowerSaveMode,
            isCharging = resources.isCharging,
            batteryPct = resources.batteryPct,
            memoryRecovery = resources.memoryRecovery
        )

        val contextLimit = if (batchLimit == 0) {
            0
        } else {
            DeviceAdaptiveRuntimePolicy.trainingContextLimit(
                configuredContextLength = model.configuredContextLength,
                availableHeapBytes = resources.availableAppHeapBytes,
                maximumHeapBytes = resources.maximumAppHeapBytes,
                dModel = model.dModel,
                feedForwardSize = model.feedForwardSize,
                vocabularySize = model.vocabularySize,
                layerCount = model.layerCount,
                headCount = model.headCount,
                thermalStatus = resources.thermalStatus,
                availableSystemMemoryBytes = resources.availableSystemMemoryBytes,
                systemLowMemory = resources.systemLowMemory,
                isPowerSaveMode = resources.isPowerSaveMode,
                isCharging = resources.isCharging,
                memoryRecovery = resources.memoryRecovery
            )
        }

        val deferReason = when {
            batchLimit == 0 -> RuntimeDeferReason.THERMAL_LIMIT
            contextLimit == 0 -> RuntimeDeferReason.INSUFFICIENT_MEMORY
            else -> null
        }

        val dispatchLimits = deviceProfile?.limits?.let { limits ->
            RuntimeDispatchLimits(
                maxComputeWorkGroupInvocations = limits.maxComputeWorkGroupInvocations,
                maxComputeWorkGroupSizeX = limits.maxComputeWorkGroupSizeX,
                maxComputeWorkGroupSizeY = limits.maxComputeWorkGroupSizeY,
                maxComputeWorkGroupSizeZ = limits.maxComputeWorkGroupSizeZ,
                maxComputeWorkGroupCountX = limits.maxComputeWorkGroupCountX,
                maxComputeWorkGroupCountY = limits.maxComputeWorkGroupCountY,
                maxComputeWorkGroupCountZ = limits.maxComputeWorkGroupCountZ,
                maxComputeSharedMemoryBytes = limits.maxComputeSharedMemoryBytes,
                maxStorageBufferRangeBytes = limits.maxStorageBufferRangeBytes
            )
        }

        return RuntimeConfiguration(
            computeBackend = if (vulkanRuntimeReady) {
                RuntimeComputeBackend.VULKAN
            } else {
                RuntimeComputeBackend.CPU_FALLBACK
            },
            deviceName = deviceProfile?.identity?.deviceName,
            effectiveVulkanApiVersion = deviceProfile?.identity?.effectiveApiVersion?.let { version ->
                "${version.major}.${version.minor}.${version.patch}"
            },
            dispatchLimits = dispatchLimits,
            fp16ComputeUsable = vulkanRuntimeReady &&
                fp16KernelVerified &&
                deviceProfile?.features?.shaderFloat16?.isUsable == true,
            configuredTrainingBatchSize = model.configuredBatchSize,
            effectiveTrainingBatchSize = batchLimit,
            configuredContextLength = model.configuredContextLength,
            effectiveContextLength = contextLimit,
            trainingBatchDelayMillis = DeviceAdaptiveRuntimePolicy.trainingBatchDelayMillis(
                thermalStatus = resources.thermalStatus,
                temperatureCelsius = resources.temperatureCelsius,
                batteryPct = resources.batteryPct,
                isCharging = resources.isCharging,
                isPowerSaveMode = resources.isPowerSaveMode
            ),
            tensorArenaBudget = calculateTensorArenaBudget(
                deviceProfile = deviceProfile,
                maximumAppHeapBytes = resources.maximumAppHeapBytes,
                availableAppHeapBytes = resources.availableAppHeapBytes
            ),
            deferReason = deferReason
        )
    }

    fun calculateTensorArenaBudget(
        deviceProfile: DeviceProfile?,
        maximumAppHeapBytes: Long,
        availableAppHeapBytes: Long
    ): DeviceAdaptiveRuntimePolicy.TensorArenaBudget {
        return DeviceAdaptiveRuntimePolicy.calculateTensorArenaBudget(
            maximumAppHeapBytes = maximumAppHeapBytes,
            availableAppHeapBytes = availableAppHeapBytes,
            deviceLocalHeadroomBytes = deviceProfile
                ?.memory
                ?.deviceLocalMemoryBudgetHeadroomBytes
                ?.minOrNull(),
            maximumStorageBufferRangeBytes = deviceProfile
                ?.limits
                ?.maxStorageBufferRangeBytes
                ?: 0L
        )
    }
}
