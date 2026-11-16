package com.argus.agi.model

import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class VulkanInitializationProgress(
    val status: String = "Waiting for model readiness",
    val driverInfo: String = "Vulkan driver detection pending",
    val isInitializing: Boolean = false,
    val isVerified: Boolean = false,
    val isCpuFallback: Boolean = false
)

data class VulkanGpuPerformanceMetrics(
    val timestampProfilingAvailable: Boolean = false,
    val timestampedDispatchCount: Long = 0L,
    val lastAdamwGpuTimeNanoseconds: Long = 0L,
    val lastLinearGpuTimeNanoseconds: Long = 0L,
    val lastFp16LinearGpuTimeNanoseconds: Long = 0L,
    val totalGpuTimeNanoseconds: Long = 0L
) {
    companion object {
        fun fromJson(payload: String): VulkanGpuPerformanceMetrics {
            val json = org.json.JSONObject(payload)
            return VulkanGpuPerformanceMetrics(
                timestampProfilingAvailable = json.optBoolean("timestampProfilingAvailable"),
                timestampedDispatchCount = json.optLong("timestampedDispatchCount"),
                lastAdamwGpuTimeNanoseconds = json.optLong("lastAdamwGpuTimeNanoseconds"),
                lastLinearGpuTimeNanoseconds = json.optLong("lastLinearGpuTimeNanoseconds"),
                lastFp16LinearGpuTimeNanoseconds =
                    json.optLong("lastFp16LinearGpuTimeNanoseconds"),
                totalGpuTimeNanoseconds = json.optLong("totalGpuTimeNanoseconds")
            )
        }
    }
}


internal class VulkanTensorAllocation internal constructor(
    internal val nativeHandle: Long,
    val byteCount: Long
)

internal data class VulkanTensorArenaMetrics(
    val capacityBytes: Long = 0L,
    val allocatedBytes: Long = 0L,
    val activeAllocationCount: Int = 0,
    val deviceLocalMemory: Boolean = false
) {
    companion object {
        fun fromJson(payload: String): VulkanTensorArenaMetrics {
            val json = org.json.JSONObject(payload)
            return VulkanTensorArenaMetrics(
                capacityBytes = json.optLong("capacityBytes"),
                allocatedBytes = json.optLong("allocatedBytes"),
                activeAllocationCount = json.optInt("activeAllocationCount"),
                deviceLocalMemory = json.optBoolean("deviceLocalMemory")
            )
        }
    }
}

data class VulkanCooperativeMatrixType(
    val mSize: Int,
    val nSize: Int,
    val kSize: Int,
    val aType: Int,
    val bType: Int,
    val cType: Int,
    val resultType: Int,
    val saturatingAccumulation: Boolean,
    val scope: Int
)

data class VulkanDeviceCapabilities(
    val deviceName: String,
    val vendorId: Int,
    val deviceId: Int,
    val deviceType: Int,
    val apiVersionMajor: Int,
    val apiVersionMinor: Int,
    val apiVersionPatch: Int,
    val driverName: String,
    val driverInfo: String,
    val driverVersionRaw: Long,
    val deviceIdentityPropertiesAvailable: Boolean,
    val deviceUuid: String,
    val driverUuid: String,
    val pipelineCacheUuid: String,
    val maxComputeWorkGroupInvocations: Int,
    val maxComputeWorkGroupSizeX: Int,
    val maxComputeWorkGroupSizeY: Int,
    val maxComputeWorkGroupSizeZ: Int,
    val maxComputeWorkGroupCountX: Int,
    val maxComputeWorkGroupCountY: Int,
    val maxComputeWorkGroupCountZ: Int,
    val maxComputeSharedMemoryBytes: Long,
    val maxStorageBufferRangeBytes: Long,
    val subgroupSize: Int,
    val subgroupSupportedStages: Int,
    val subgroupSupportedOperations: Int,
    val subgroupQuadOperationsInAllStages: Boolean,
    val subgroupSizeControl: Boolean,
    val computeFullSubgroups: Boolean,
    val minSubgroupSize: Int,
    val maxSubgroupSize: Int,
    val maxComputeWorkgroupSubgroups: Int,
    val requiredSubgroupSizeStages: Int,
    val shaderFloat16: Boolean,
    val shaderInt8: Boolean,
    val storageBuffer16BitAccess: Boolean,
    val storageBuffer8BitAccess: Boolean,
    val shaderIntegerDotProduct: Boolean,
    val integerDotProduct8BitUnsignedAccelerated: Boolean,
    val integerDotProduct8BitSignedAccelerated: Boolean,
    val integerDotProduct8BitMixedSignednessAccelerated: Boolean,
    val integerDotProduct4x8BitPackedUnsignedAccelerated: Boolean,
    val integerDotProduct4x8BitPackedSignedAccelerated: Boolean,
    val integerDotProduct4x8BitPackedMixedSignednessAccelerated: Boolean,
    val cooperativeMatrixExtension: Boolean,
    val cooperativeMatrix: Boolean,
    val cooperativeMatrixRobustBufferAccess: Boolean,
    val cooperativeMatrixSupportedStages: Int,
    val cooperativeMatrixTypes: List<VulkanCooperativeMatrixType>,
    val armTensorsExtensionSupported: Boolean,
    val armTensorsFeatureQueryAvailable: Boolean,
    val armTensorsFeatureSupported: Boolean,
    val armTensorNonPackedSupported: Boolean,
    val armTensorShaderAccessSupported: Boolean,
    val armTensorDynamicIndexingSupported: Boolean,
    val armTensorNonUniformIndexingSupported: Boolean,
    val armTensorUpdateAfterBindSupported: Boolean,
    val armTensorDeviceFeaturesEnabled: Boolean,
    val shaderTensorSupportedStages: Int,
    val armTensorMaxDimensionCount: Int,
    val armTensorMaxElements: Long,
    val armTensorMaxPerDimensionElements: Long,
    val armTensorMaxStride: Long,
    val armTensorMaxSizeBytes: Long,
    val armTensorMaxShaderAccessArrayLength: Int,
    val armTensorMaxShaderAccessSizeBytes: Int,
    val armTensorMaxDescriptorSetStorageTensors: Int,
    val armTensorMaxPerStageDescriptorSetStorageTensors: Int,
    val armTensorNonUniformIndexingNative: Boolean,
    val armShaderCorePropertiesExtensionSupported: Boolean,
    val armShaderCoreBuiltinsExtensionSupported: Boolean,
    val armShaderCoreBuiltinsFeatureQueryAvailable: Boolean,
    val armShaderCoreBuiltinsFeatureSupported: Boolean,
    val armShaderCoreBuiltinsFeatureEnabled: Boolean,
    val armShaderCoreCount: Int,
    val armShaderWarpsPerCore: Int,
    val armShaderFmaRate: Int,
    val armShaderPixelRate: Int,
    val armShaderTexelRate: Int,
    val effectiveApiVersionMajor: Int,
    val effectiveApiVersionMinor: Int,
    val effectiveApiVersionPatch: Int,
    val timestampPeriodNanoseconds: Double,
    val timestampComputeAndGraphics: Boolean,
    val computeQueueTimestampValidBits: Int,
    val computeTimestampQueriesSupported: Boolean,
    val shaderFloat16FeatureSupported: Boolean,
    val shaderFloat16FeatureEnabled: Boolean,
    val shaderInt8FeatureEnabled: Boolean,
    val storageBuffer16BitAccessEnabled: Boolean,
    val storageBuffer8BitAccessEnabled: Boolean,
    val subgroupSizeControlEnabled: Boolean,
    val shaderIntegerDotProductEnabled: Boolean,
    val cooperativeMatrixEnabled: Boolean,
    val synchronization2ExtensionSupported: Boolean,
    val synchronization2Supported: Boolean,
    val synchronization2Enabled: Boolean,
    val memoryPropertiesAvailable: Boolean,
    val memoryHeapCount: Int,
    val memoryTypeCount: Int,
    val hostVisibleMemoryTypeCount: Int,
    val hostVisibleDeviceLocalMemoryTypeCount: Int,
    val memoryBudgetExtensionSupported: Boolean,
    val memoryBudgetExtensionEnabled: Boolean,
    val memoryBudgetQueryAvailable: Boolean,
    val memoryBudgetHeapCount: Int,
    val memoryHeapBudgetBytes: List<Long>,
    val memoryHeapUsageBytes: List<Long>
) {
    val apiVersion: String
        get() = "$apiVersionMajor.$apiVersionMinor.$apiVersionPatch"

    val memoryBudgetHeadroomBytes: List<Long>
        get() = memoryHeapBudgetBytes.zip(memoryHeapUsageBytes) { budget, usage ->
            (budget - usage).coerceAtLeast(0L)
        }

    val supportsComputeSubgroups: Boolean
        get() = subgroupSize > 0 && (subgroupSupportedStages and COMPUTE_SHADER_STAGE_BIT) != 0

    val supportsComputeSubgroupArithmetic: Boolean
        get() = supportsComputeSubgroups &&
            (subgroupSupportedOperations and SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0

    val supportsComputeSubgroupReduction: Boolean
        get() = supportsComputeSubgroupArithmetic

    val supportsComputeSubgroupShuffle: Boolean
        get() = supportsComputeSubgroups &&
            (subgroupSupportedOperations and SUBGROUP_FEATURE_SHUFFLE_BIT) != 0

    val supportsComputeSubgroupShuffleRelative: Boolean
        get() = supportsComputeSubgroups &&
            (subgroupSupportedOperations and SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) != 0

    val supportsComputeTimestampQueries: Boolean
        get() = computeTimestampQueriesSupported &&
            computeQueueTimestampValidBits > 0 && timestampPeriodNanoseconds > 0.0

    val hasAcceleratedInt8DotProduct: Boolean
        get() = shaderIntegerDotProduct &&
            (integerDotProduct8BitSignedAccelerated ||
                integerDotProduct8BitUnsignedAccelerated ||
                integerDotProduct4x8BitPackedSignedAccelerated ||
                integerDotProduct4x8BitPackedUnsignedAccelerated)

    val canUseIntegerDotProductKernel: Boolean
        get() = shaderIntegerDotProduct && shaderIntegerDotProductEnabled &&
            hasAcceleratedInt8DotProduct

    val hasSupportedCooperativeMatrixConfiguration: Boolean
        get() = cooperativeMatrix &&
            (cooperativeMatrixSupportedStages and COMPUTE_SHADER_STAGE_BIT) != 0 &&
            cooperativeMatrixTypes.isNotEmpty()

    val hasUsableCooperativeMatrixConfiguration: Boolean
        get() = cooperativeMatrixEnabled && hasSupportedCooperativeMatrixConfiguration

    val supportsComputeArmTensors: Boolean
        get() = armTensorsExtensionSupported &&
            armTensorsFeatureQueryAvailable &&
            armTensorsFeatureSupported &&
            armTensorShaderAccessSupported &&
            (shaderTensorSupportedStages and COMPUTE_SHADER_STAGE_BIT) != 0

    val canUseArmTensorCompute: Boolean
        get() = supportsComputeArmTensors && armTensorDeviceFeaturesEnabled

    val hasArmShaderCoreTopology: Boolean
        get() = armShaderCoreBuiltinsExtensionSupported &&
            armShaderCoreBuiltinsFeatureQueryAvailable &&
            armShaderCoreBuiltinsFeatureSupported &&
            armShaderCoreCount > 0 &&
            armShaderWarpsPerCore > 0

    val hasArmShaderCorePerformanceProfile: Boolean
        get() = armShaderCorePropertiesExtensionSupported && armShaderFmaRate > 0

    companion object {
        private const val COMPUTE_SHADER_STAGE_BIT = 0x00000020
        private const val SUBGROUP_FEATURE_ARITHMETIC_BIT = 0x00000004
        private const val SUBGROUP_FEATURE_SHUFFLE_BIT = 0x00000010
        private const val SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT = 0x00000020

        fun fromJson(payload: String): VulkanDeviceCapabilities {
            val json = org.json.JSONObject(payload)
            val matrixArray = json.optJSONArray("cooperativeMatrixTypes")
            val matrixTypes = buildList {
                if (matrixArray != null) {
                    for (index in 0 until matrixArray.length()) {
                        val matrix = matrixArray.optJSONObject(index) ?: continue
                        add(
                            VulkanCooperativeMatrixType(
                                mSize = matrix.optInt("mSize"),
                                nSize = matrix.optInt("nSize"),
                                kSize = matrix.optInt("kSize"),
                                aType = matrix.optInt("aType"),
                                bType = matrix.optInt("bType"),
                                cType = matrix.optInt("cType"),
                                resultType = matrix.optInt("resultType"),
                                saturatingAccumulation = matrix.optBoolean("saturatingAccumulation"),
                                scope = matrix.optInt("scope")
                            )
                        )
                    }
                }
            }

            return VulkanDeviceCapabilities(
                deviceName = json.optString("deviceName", "Unknown Vulkan GPU"),
                vendorId = json.optInt("vendorId"),
                deviceId = json.optInt("deviceId"),
                deviceType = json.optInt("deviceType"),
                apiVersionMajor = json.optInt("apiVersionMajor"),
                apiVersionMinor = json.optInt("apiVersionMinor"),
                apiVersionPatch = json.optInt("apiVersionPatch"),
                driverName = json.optString("driverName"),
                driverInfo = json.optString("driverInfo"),
                driverVersionRaw = json.optLong("driverVersionRaw"),
                deviceIdentityPropertiesAvailable = json.optBoolean("deviceIdentityPropertiesAvailable"),
                deviceUuid = json.optString("deviceUUID"),
                driverUuid = json.optString("driverUUID"),
                pipelineCacheUuid = json.optString("pipelineCacheUUID"),
                maxComputeWorkGroupInvocations = json.optInt("maxComputeWorkGroupInvocations"),
                maxComputeWorkGroupSizeX = json.optInt("maxComputeWorkGroupSizeX"),
                maxComputeWorkGroupSizeY = json.optInt("maxComputeWorkGroupSizeY"),
                maxComputeWorkGroupSizeZ = json.optInt("maxComputeWorkGroupSizeZ"),
                maxComputeWorkGroupCountX = json.optInt("maxComputeWorkGroupCountX"),
                maxComputeWorkGroupCountY = json.optInt("maxComputeWorkGroupCountY"),
                maxComputeWorkGroupCountZ = json.optInt("maxComputeWorkGroupCountZ"),
                maxComputeSharedMemoryBytes = json.optLong("maxComputeSharedMemoryBytes"),
                maxStorageBufferRangeBytes = json.optLong("maxStorageBufferRangeBytes"),
                subgroupSize = json.optInt("subgroupSize"),
                subgroupSupportedStages = json.optInt("subgroupSupportedStages"),
                subgroupSupportedOperations = json.optInt("subgroupSupportedOperations"),
                subgroupQuadOperationsInAllStages = json.optBoolean("subgroupQuadOperationsInAllStages"),
                subgroupSizeControl = json.optBoolean("subgroupSizeControl"),
                computeFullSubgroups = json.optBoolean("computeFullSubgroups"),
                minSubgroupSize = json.optInt("minSubgroupSize"),
                maxSubgroupSize = json.optInt("maxSubgroupSize"),
                maxComputeWorkgroupSubgroups = json.optInt("maxComputeWorkgroupSubgroups"),
                requiredSubgroupSizeStages = json.optInt("requiredSubgroupSizeStages"),
                shaderFloat16 = json.optBoolean("shaderFloat16"),
                shaderInt8 = json.optBoolean("shaderInt8"),
                storageBuffer16BitAccess = json.optBoolean("storageBuffer16BitAccess"),
                storageBuffer8BitAccess = json.optBoolean("storageBuffer8BitAccess"),
                shaderIntegerDotProduct = json.optBoolean("shaderIntegerDotProduct"),
                integerDotProduct8BitUnsignedAccelerated =
                    json.optBoolean("integerDotProduct8BitUnsignedAccelerated"),
                integerDotProduct8BitSignedAccelerated =
                    json.optBoolean("integerDotProduct8BitSignedAccelerated"),
                integerDotProduct8BitMixedSignednessAccelerated =
                    json.optBoolean("integerDotProduct8BitMixedSignednessAccelerated"),
                integerDotProduct4x8BitPackedUnsignedAccelerated =
                    json.optBoolean("integerDotProduct4x8BitPackedUnsignedAccelerated"),
                integerDotProduct4x8BitPackedSignedAccelerated =
                    json.optBoolean("integerDotProduct4x8BitPackedSignedAccelerated"),
                integerDotProduct4x8BitPackedMixedSignednessAccelerated =
                    json.optBoolean("integerDotProduct4x8BitPackedMixedSignednessAccelerated"),
                cooperativeMatrixExtension = json.optBoolean("cooperativeMatrixExtension"),
                cooperativeMatrix = json.optBoolean("cooperativeMatrix"),
                cooperativeMatrixRobustBufferAccess =
                    json.optBoolean("cooperativeMatrixRobustBufferAccess"),
                cooperativeMatrixSupportedStages = json.optInt("cooperativeMatrixSupportedStages"),
                cooperativeMatrixTypes = matrixTypes,
                armTensorsExtensionSupported = json.optBoolean("armTensorsExtensionSupported"),
                armTensorsFeatureQueryAvailable = json.optBoolean("armTensorsFeatureQueryAvailable"),
                armTensorsFeatureSupported = json.optBoolean("armTensorsFeatureSupported"),
                armTensorNonPackedSupported = json.optBoolean("armTensorNonPackedSupported"),
                armTensorShaderAccessSupported = json.optBoolean("armTensorShaderAccessSupported"),
                armTensorDynamicIndexingSupported = json.optBoolean("armTensorDynamicIndexingSupported"),
                armTensorNonUniformIndexingSupported =
                    json.optBoolean("armTensorNonUniformIndexingSupported"),
                armTensorUpdateAfterBindSupported =
                    json.optBoolean("armTensorUpdateAfterBindSupported"),
                armTensorDeviceFeaturesEnabled = json.optBoolean("armTensorDeviceFeaturesEnabled"),
                shaderTensorSupportedStages = json.optInt("shaderTensorSupportedStages"),
                armTensorMaxDimensionCount = json.optInt("armTensorMaxDimensionCount"),
                armTensorMaxElements = json.optLong("armTensorMaxElements"),
                armTensorMaxPerDimensionElements = json.optLong("armTensorMaxPerDimensionElements"),
                armTensorMaxStride = json.optLong("armTensorMaxStride"),
                armTensorMaxSizeBytes = json.optLong("armTensorMaxSizeBytes"),
                armTensorMaxShaderAccessArrayLength =
                    json.optInt("armTensorMaxShaderAccessArrayLength"),
                armTensorMaxShaderAccessSizeBytes =
                    json.optInt("armTensorMaxShaderAccessSizeBytes"),
                armTensorMaxDescriptorSetStorageTensors =
                    json.optInt("armTensorMaxDescriptorSetStorageTensors"),
                armTensorMaxPerStageDescriptorSetStorageTensors =
                    json.optInt("armTensorMaxPerStageDescriptorSetStorageTensors"),
                armTensorNonUniformIndexingNative =
                    json.optBoolean("armTensorNonUniformIndexingNative"),
                armShaderCorePropertiesExtensionSupported =
                    json.optBoolean("armShaderCorePropertiesExtensionSupported"),
                armShaderCoreBuiltinsExtensionSupported =
                    json.optBoolean("armShaderCoreBuiltinsExtensionSupported"),
                armShaderCoreBuiltinsFeatureQueryAvailable =
                    json.optBoolean("armShaderCoreBuiltinsFeatureQueryAvailable"),
                armShaderCoreBuiltinsFeatureSupported =
                    json.optBoolean("armShaderCoreBuiltinsFeatureSupported"),
                armShaderCoreBuiltinsFeatureEnabled =
                    json.optBoolean("armShaderCoreBuiltinsFeatureEnabled"),
                armShaderCoreCount = json.optInt("armShaderCoreCount"),
                armShaderWarpsPerCore = json.optInt("armShaderWarpsPerCore"),
                armShaderFmaRate = json.optInt("armShaderFmaRate"),
                armShaderPixelRate = json.optInt("armShaderPixelRate"),
                armShaderTexelRate = json.optInt("armShaderTexelRate"),
                effectiveApiVersionMajor = json.optInt(
                    "effectiveApiVersionMajor",
                    json.optInt("apiVersionMajor")
                ),
                effectiveApiVersionMinor = json.optInt(
                    "effectiveApiVersionMinor",
                    json.optInt("apiVersionMinor")
                ),
                effectiveApiVersionPatch = json.optInt(
                    "effectiveApiVersionPatch",
                    json.optInt("apiVersionPatch")
                ),
                timestampPeriodNanoseconds = json.optDouble("timestampPeriodNanoseconds"),
                timestampComputeAndGraphics = json.optBoolean("timestampComputeAndGraphics"),
                computeQueueTimestampValidBits = json.optInt("computeQueueTimestampValidBits"),
                computeTimestampQueriesSupported = json.optBoolean(
                    "computeTimestampQueriesSupported",
                    json.optInt("computeQueueTimestampValidBits") > 0
                ),
                shaderFloat16FeatureSupported = json.optBoolean("shaderFloat16FeatureSupported"),
                shaderFloat16FeatureEnabled = json.optBoolean("shaderFloat16FeatureEnabled"),
                shaderInt8FeatureEnabled = json.optBoolean("shaderInt8FeatureEnabled"),
                storageBuffer16BitAccessEnabled = json.optBoolean("storageBuffer16BitAccessEnabled"),
                storageBuffer8BitAccessEnabled = json.optBoolean("storageBuffer8BitAccessEnabled"),
                subgroupSizeControlEnabled = json.optBoolean("subgroupSizeControlEnabled"),
                shaderIntegerDotProductEnabled = json.optBoolean("shaderIntegerDotProductEnabled"),
                cooperativeMatrixEnabled = json.optBoolean("cooperativeMatrixEnabled"),
                synchronization2ExtensionSupported =
                    json.optBoolean("synchronization2ExtensionSupported"),
                synchronization2Supported = json.optBoolean("synchronization2Supported"),
                synchronization2Enabled = json.optBoolean("synchronization2Enabled"),
                memoryPropertiesAvailable = json.optBoolean("memoryPropertiesAvailable"),
                memoryHeapCount = json.optInt("memoryHeapCount"),
                memoryTypeCount = json.optInt("memoryTypeCount"),
                hostVisibleMemoryTypeCount = json.optInt("hostVisibleMemoryTypeCount"),
                hostVisibleDeviceLocalMemoryTypeCount =
                    json.optInt("hostVisibleDeviceLocalMemoryTypeCount"),
                memoryBudgetExtensionSupported =
                    json.optBoolean("memoryBudgetExtensionSupported"),
                memoryBudgetExtensionEnabled =
                    json.optBoolean("memoryBudgetExtensionEnabled"),
                memoryBudgetQueryAvailable =
                    json.optBoolean("memoryBudgetQueryAvailable"),
                memoryBudgetHeapCount = json.optInt("memoryBudgetHeapCount"),
                memoryHeapBudgetBytes = json.optJSONArray("memoryHeapBudgetBytes")?.let { array ->
                    List(array.length()) { index -> array.optLong(index) }
                } ?: emptyList(),
                memoryHeapUsageBytes = json.optJSONArray("memoryHeapUsageBytes")?.let { array ->
                    List(array.length()) { index -> array.optLong(index) }
                } ?: emptyList()
            )
        }
    }
}

/**
 * Optional Vulkan Compute backend for batched dense operations and AdamW updates.
 *
 * Dense forward and backward operations use Vulkan when the runtime passes its
 * numerical self-check and the workload is large enough.
 * All callers retain a numerically equivalent Kotlin fallback.
 */
internal object VulkanTrainingBackend {

    private const val TAG = "VulkanTrainingBackend"
    private const val MIN_GPU_UPDATE_ELEMENTS = 49_152L
    private const val MIN_LINEAR_OPERATION_COUNT = 262_144L

    @Volatile
    private var disabledAfterRuntimeFailure = false

    @Volatile
    private var _currentBackend = "Kotlin CPU"

    @Volatile
    private var fp16RuntimeReady = false

    @Volatile
    private var fp16DisabledAfterRuntimeFailure = false

    @Volatile
    private var nativeReady = false

    @Volatile
    private var initializationAttempted = false

    private val _runtimeStatus =
        MutableStateFlow("Kotlin CPU — waiting for model readiness")
    val runtimeStatus: StateFlow<String> = _runtimeStatus.asStateFlow()

    private val _initializationProgress = MutableStateFlow(VulkanInitializationProgress())
    val initializationProgress: StateFlow<VulkanInitializationProgress> =
        _initializationProgress.asStateFlow()

    private val _deviceCapabilities = MutableStateFlow<VulkanDeviceCapabilities?>(null)
    val deviceCapabilities: StateFlow<VulkanDeviceCapabilities?> =
        _deviceCapabilities.asStateFlow()

    private val _gpuPerformanceMetrics = MutableStateFlow(VulkanGpuPerformanceMetrics())
    val gpuPerformanceMetrics: StateFlow<VulkanGpuPerformanceMetrics> =
        _gpuPerformanceMetrics.asStateFlow()

    private val _tensorArenaMetrics = MutableStateFlow(VulkanTensorArenaMetrics())
    val tensorArenaMetrics: StateFlow<VulkanTensorArenaMetrics> =
        _tensorArenaMetrics.asStateFlow()

    private val _modelTensorResidencyBudgetBytes = MutableStateFlow(0L)
    val modelTensorResidencyBudgetBytes: StateFlow<Long> =
        _modelTensorResidencyBudgetBytes.asStateFlow()

    private val _activationTensorResidencyBudgetBytes = MutableStateFlow(0L)
    internal val activationTensorResidencyBudgetBytes: StateFlow<Long> =
        _activationTensorResidencyBudgetBytes.asStateFlow()

    private val tensorResidencyRegistry = VulkanTensorResidencyRegistry(
        allocate = { byteCount -> allocateGpuTensor(byteCount) },
        upload = { allocation, bytes -> uploadGpuTensor(allocation, bytes) },
        release = { allocation -> releaseGpuTensor(allocation) },
        maxResidentBytes = { _modelTensorResidencyBudgetBytes.value }
    )

    private val _modelTensorResidencyReady = MutableStateFlow(false)
    val modelTensorResidencyReady: StateFlow<Boolean> =
        _modelTensorResidencyReady.asStateFlow()

    @Volatile
    private var residentActivationLinearReady = false
    @Volatile
    private var residentAttentionReady = false
    private val activationTensorRegistry = VulkanActivationTensorRegistry(
        allocate = { byteCount -> allocateGpuTensor(byteCount) },
        upload = { allocation, bytes -> uploadGpuTensor(allocation, bytes) },
        download = { allocation, bytes -> downloadGpuTensor(allocation, bytes) },
        release = { allocation -> releaseGpuTensor(allocation) },
        dispatch = { input, weights, output, rows, inputColumns, outputColumns, transposeWeights ->
            tryApplyResidentLinear(
                input, weights, output, rows, inputColumns, outputColumns, transposeWeights
            )
        },
        maxResidentBytes = { _activationTensorResidencyBudgetBytes.value },
        dispatchLinearSet = { input, weights, output, rowsPerBatch, inputColumns, outputColumns, transposeWeights ->
            tryApplyResidentLinearBatch(
                input,
                weights,
                output,
                rowsPerBatch,
                inputColumns,
                outputColumns,
                transposeWeights
            )
        },
        dispatchAttention = { qkv, scores, attentionOutput, projectedOutput, normalizedOutput,
            residual, outputWeights, sequenceLength, modelWidth, headCount, headDimension ->
            tryApplyResidentAttention(
                qkv, scores, attentionOutput, projectedOutput, normalizedOutput, residual,
                outputWeights, sequenceLength, modelWidth, headCount, headDimension
            )
        }
    )
    internal val residentActivations: VulkanActivationTensorRegistry
        get() = activationTensorRegistry

    private var lastModelParameterInventory: List<VulkanModelParameterTensor> = emptyList()
    private var residencySuspended = false
    private var residencyBlockedAfterFailure = false
    private var residencyRebuildPending = false
    private var runtimeRestartPending = false

    private fun invalidateTensorResidency(releaseAllocations: Boolean): Boolean {
        _modelTensorResidencyReady.value = false
        val modelReleased = tensorResidencyRegistry.invalidate(releaseAllocations)
        val activationsReleased = activationTensorRegistry.invalidate(releaseAllocations)
        val released = modelReleased && activationsReleased
        if (!released && releaseAllocations && nativeReady) {
            val arenaCleared = runCatching { nativeClearTensorArena() }.getOrDefault(false)
            if (arenaCleared) {
                tensorResidencyRegistry.invalidate(releaseAllocations = false)
                activationTensorRegistry.invalidate(releaseAllocations = false)
                refreshTensorArenaMetrics()
                return true
            }

            disabledAfterRuntimeFailure = true
            _currentBackend = "Kotlin CPU"
            _runtimeStatus.value =
                "Kotlin CPU fallback — Vulkan tensor-arena release failed"
            runCatching { destroyNativeRuntime() }
                .onFailure { error ->
                    Log.e(TAG, "Vulkan runtime teardown after an arena-release failure failed.", error)
                }
        }
        return released
    }

    internal val hasAttemptedInitialization: Boolean
        get() = initializationAttempted

    val currentBackend: String
        get() = _currentBackend

    /**
     * Status inspection is deliberately side-effect free. Vulkan must never be loaded
     * as a side effect of rendering a screen or building the foreground notification.
     */
    val isRuntimeAvailable: Boolean
        get() = nativeReady && !disabledAfterRuntimeFailure

    val isFp16ComputeAvailable: Boolean
        get() = isRuntimeAvailable && fp16RuntimeReady && !fp16DisabledAfterRuntimeFailure

    internal val isResidentActivationLinearAvailable: Boolean
        get() = isRuntimeAvailable && residentActivationLinearReady

    internal val isResidentAttentionAvailable: Boolean
        get() = isRuntimeAvailable && residentActivationLinearReady && residentAttentionReady

    /**
     * Initialize Vulkan only after ArgusNeuralEngine has constructed or restored its tensors.
     * Driver verification is isolated from application cold-start; failure keeps Kotlin CPU active.
     */
    private fun publishInitializationProgress(
        status: String,
        driverInfo: String,
        isInitializing: Boolean,
        isVerified: Boolean = false,
        isCpuFallback: Boolean = false
    ) {
        _initializationProgress.value = VulkanInitializationProgress(
            status = status,
            driverInfo = driverInfo,
            isInitializing = isInitializing,
            isVerified = isVerified,
            isCpuFallback = isCpuFallback
        )
    }

    private fun refreshGpuPerformanceMetrics() {
        if (!nativeReady) return
        runCatching {
            VulkanGpuPerformanceMetrics.fromJson(nativeGetGpuPerformanceMetrics())
        }.onSuccess { metrics ->
            _gpuPerformanceMetrics.value = metrics
        }.onFailure { error ->
            Log.d(TAG, "GPU timestamp metrics could not be read.", error)
        }
    }

    private fun destroyNativeRuntime() {
        try {
            nativeDestroy()
        } finally {
            nativeReady = false
            fp16RuntimeReady = false
            residentActivationLinearReady = false
            residentAttentionReady = false
            invalidateTensorResidency(releaseAllocations = false)
            _tensorArenaMetrics.value = VulkanTensorArenaMetrics()
        }
    }

    private fun releaseNativeRuntimeAfterFailure(nativeLibraryLoaded: Boolean) {
        nativeReady = false
        fp16RuntimeReady = false
        if (nativeLibraryLoaded) {
            runCatching { destroyNativeRuntime() }
                .onFailure { error ->
                    Log.w(TAG, "Failed to release a partially initialized Vulkan runtime.", error)
                }
        } else {
            invalidateTensorResidency(releaseAllocations = false)
        }
    }


    private fun refreshTensorArenaMetrics() {
        if (!nativeReady) return
        runCatching {
            VulkanTensorArenaMetrics.fromJson(nativeGetTensorArenaMetrics())
        }.onSuccess { metrics ->
            _tensorArenaMetrics.value = metrics
        }.onFailure { error ->
            Log.d(TAG, "Vulkan tensor-arena metrics could not be read.", error)
        }
    }

    private fun calculateTotalTensorResidencyBudgetBytes(
        capabilities: VulkanDeviceCapabilities?
    ): Long {
        val maximumAppHeap = Runtime.getRuntime().maxMemory().coerceAtLeast(0L)
        val appBudget = maximumAppHeap / 4L
        val reportedHeapHeadroom = capabilities?.memoryBudgetHeadroomBytes
            ?.filter { it > 0L }
            ?.minOrNull()
        val gpuBudget = reportedHeapHeadroom?.div(3L) ?: appBudget
        val descriptorRange = capabilities?.maxStorageBufferRangeBytes
            ?.takeIf { it > 0L } ?: Long.MAX_VALUE
        return minOf(appBudget, gpuBudget, descriptorRange).coerceAtLeast(0L)
    }

    private fun calculateModelTensorResidencyBudgetBytes(
        capabilities: VulkanDeviceCapabilities?
    ): Long = calculateTotalTensorResidencyBudgetBytes(capabilities) * 3L / 4L

    private fun calculateActivationTensorResidencyBudgetBytes(
        capabilities: VulkanDeviceCapabilities?
    ): Long = calculateTotalTensorResidencyBudgetBytes(capabilities) / 4L

    /**
     * Allocates persistent Vulkan storage. New contents are undefined until uploaded
     * or initialized by a future GPU operation; this API does not change model ownership.
     */
    @Synchronized
    fun allocateGpuTensor(byteCount: Long): VulkanTensorAllocation? {
        if (!isRuntimeAvailable || byteCount <= 0L) return null
        val handle = try {
            nativeAllocateTensor(byteCount)
        } catch (error: LinkageError) {
            Log.e(TAG, "Vulkan tensor allocation entry point is unavailable.", error)
            return null
        } catch (error: RuntimeException) {
            Log.e(TAG, "Vulkan tensor allocation failed.", error)
            return null
        }
        if (handle == 0L) return null
        refreshTensorArenaMetrics()
        return VulkanTensorAllocation(handle, byteCount)
    }

    @Synchronized
    fun uploadGpuTensor(allocation: VulkanTensorAllocation, bytes: ByteArray): Boolean {
        if (!isRuntimeAvailable || bytes.size.toLong() != allocation.byteCount) return false
        val uploaded = try {
            nativeUploadTensor(allocation.nativeHandle, bytes)
        } catch (error: LinkageError) {
            Log.e(TAG, "Vulkan tensor upload entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Vulkan tensor upload failed.", error)
            false
        }
        if (uploaded) refreshTensorArenaMetrics()
        return uploaded
    }

    @Synchronized
    fun downloadGpuTensor(allocation: VulkanTensorAllocation, destination: ByteArray): Boolean {
        if (!isRuntimeAvailable || destination.size.toLong() != allocation.byteCount) return false
        val downloaded = try {
            nativeDownloadTensor(allocation.nativeHandle, destination)
        } catch (error: LinkageError) {
            Log.e(TAG, "Vulkan tensor readback entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Vulkan tensor readback failed.", error)
            false
        }
        if (downloaded) refreshTensorArenaMetrics()
        return downloaded
    }

    @Synchronized
    fun releaseGpuTensor(allocation: VulkanTensorAllocation): Boolean {
        if (!isRuntimeAvailable) return false
        val released = try {
            nativeReleaseTensor(allocation.nativeHandle)
        } catch (error: LinkageError) {
            Log.e(TAG, "Vulkan tensor release entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Vulkan tensor release failed.", error)
            false
        }
        if (released) refreshTensorArenaMetrics()
        return released
    }

    /**
     * Prepares the complete model parameter inventory in persistent Tensor Arena
     * allocations. The registry remains a read-only mirror of Kotlin-owned weights.
     */
    private fun refreshModelTensorResidencyBudget() {
        if (!nativeReady) return
        runCatching {
            VulkanDeviceCapabilities.fromJson(nativeGetDeviceCapabilities())
        }.onSuccess { capabilities ->
            _deviceCapabilities.value = capabilities
            _modelTensorResidencyBudgetBytes.value =
                calculateModelTensorResidencyBudgetBytes(capabilities)
            _activationTensorResidencyBudgetBytes.value =
                calculateActivationTensorResidencyBudgetBytes(capabilities)
            Log.i(
                TAG,
                "Refreshed FP16 model-residency budget: " +
                    "${_modelTensorResidencyBudgetBytes.value} bytes."
            )
        }.onFailure { error ->
            Log.w(
                TAG,
                "Could not refresh Vulkan memory-budget telemetry; retaining the last verified budget.",
                error
            )
        }
    }

    @Synchronized
    fun synchronizeModelTensorResidency(
        tensors: List<VulkanModelParameterTensor>
    ): Boolean {
        lastModelParameterInventory = tensors.toList()
        residencyRebuildPending = false
        _modelTensorResidencyReady.value = false
        if (residencySuspended ||
            residencyBlockedAfterFailure ||
            (!isFp16ComputeAvailable && !isResidentActivationLinearAvailable)) {
            return false
        }

        val synchronized = try {
            tensorResidencyRegistry.synchronizeAll(tensors)
        } catch (error: OutOfMemoryError) {
            Log.w(TAG, "Full model tensor residency exceeded available memory.", error)
            false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Full model tensor residency synchronization failed.", error)
            false
        }

        if (!synchronized) {
            invalidateTensorResidency(releaseAllocations = true)
            residencyBlockedAfterFailure = true
            Log.w(
                TAG,
                "Complete FP16 model residency was not established within the device memory budget; " +
                    "persistent FP16 dispatch is disabled until the next foreground recovery. " +
                    "The verified FP32 Vulkan/CPU fallback remains active."
            )
        } else {
            residencyBlockedAfterFailure = false
            Log.i(
                TAG,
                "Resident and revision-verified ${tensors.size} model parameter tensors " +
                    "(${tensorResidencyRegistry.residentByteCount()} bytes)."
            )
        }
        refreshTensorArenaMetrics()
        _modelTensorResidencyReady.value = synchronized
        return synchronized
    }

    /**
     * Drop rebuildable GPU replicas when the app is backgrounded. Kotlin model weights,
     * optimizer moments and checkpoints remain authoritative and untouched.
     */
    @Synchronized
    fun releaseModelTensorResidencyForMemoryPressure(): Boolean {
        residencySuspended = true
        residencyRebuildPending = false
        _modelTensorResidencyReady.value = false

        if (!isRuntimeAvailable) {
            invalidateTensorResidency(releaseAllocations = false)
            return false
        }

        val cleared = try {
            nativeClearTensorArena()
        } catch (error: LinkageError) {
            Log.e(TAG, "Vulkan arena-clear entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Vulkan arena clear failed during memory pressure.", error)
            false
        }

        if (cleared) {
            invalidateTensorResidency(releaseAllocations = false)
            refreshTensorArenaMetrics()
            Log.i(TAG, "Released the complete Vulkan tensor arena for background memory pressure.")
            return true
        }

        invalidateTensorResidency(releaseAllocations = true)
        runtimeRestartPending = true
        disabledAfterRuntimeFailure = true
        _currentBackend = "Kotlin CPU"
        _runtimeStatus.value =
            "Kotlin CPU fallback — Vulkan arena could not be returned under memory pressure"
        runCatching { destroyNativeRuntime() }
            .onFailure { error ->
                Log.e(TAG, "Vulkan teardown after memory-pressure release failure failed.", error)
            }
        refreshTensorArenaMetrics()
        return false
    }

    /** Re-enable a single full-inventory rebuild after the app returns to the foreground. */
    @Synchronized
    fun resumeModelTensorResidencyAfterForeground() {
        if (runtimeRestartPending) {
            runtimeRestartPending = false
            resetRuntimeForRetry()
            if (!initializeAfterModelReady()) {
                residencySuspended = true
                return
            }
            residencySuspended = false
            residencyBlockedAfterFailure = false
            residencyRebuildPending = lastModelParameterInventory.isNotEmpty()
            return
        }
        if (!residencySuspended) return
        refreshModelTensorResidencyBudget()
        residencySuspended = false
        residencyBlockedAfterFailure = false
        residencyRebuildPending = lastModelParameterInventory.isNotEmpty()
    }

    private fun ensureModelTensorResidencyReady(): Boolean {
        if (_modelTensorResidencyReady.value) return true
        if (residencySuspended || !residencyRebuildPending) return false

        val inventory = lastModelParameterInventory
        residencyRebuildPending = false
        if (inventory.isEmpty()) return false
        return synchronizeModelTensorResidency(inventory)
    }

    /**
     * Acquires a revision-synchronized FP16 model tensor for the resident FP32-activation
     * compute graph. This path does not require shaderFloat16 support because the resident
     * linear shader unpacks the stored binary16 words to FP32 before accumulation.
     */
    @Synchronized
    internal fun acquireResidentModelParameterTensor(
        parameters: Float16ParameterArray
    ): VulkanTensorAllocation? {
        if (!isResidentActivationLinearAvailable || !ensureModelTensorResidencyReady()) {
            return null
        }
        return try {
            tensorResidencyRegistry.acquire(parameters)
        } catch (error: OutOfMemoryError) {
            Log.w(TAG, "Resident model parameter synchronization exceeded available memory.", error)
            null
        } catch (error: RuntimeException) {
            Log.e(TAG, "Resident model parameter synchronization failed.", error)
            null
        }
    }


    /**
     * Runs a set of projections that share one input activation (Q/K/V in the
     * Transformer forward pass). Inputs are uploaded once, weights are acquired from
     * the revision-verified model registry, and all outputs remain resident until the
     * single explicit readback after the complete dispatch.
     *
     * Output arrays are committed only after dispatch, readback, and temporary
     * allocation release succeed, so callers can safely use their normal fallback.
     */
    @Synchronized
    internal fun tryApplyResidentProjectionSet(
        inputRows: Array<FloatArray>,
        weightTensors: Array<Float16ParameterArray>,
        outputRows: Array<Array<FloatArray>>,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean = false
    ): Boolean {
        if (!isResidentActivationLinearAvailable ||
            inputRows.isEmpty() ||
            weightTensors.isEmpty() ||
            weightTensors.size != outputRows.size ||
            inputColumns <= 0 ||
            outputColumns <= 0 ||
            inputRows.any { it.size != inputColumns } ||
            outputRows.any { batch ->
                batch.size != inputRows.size || batch.any { it.size != outputColumns }
            }) {
            return false
        }

        val expectedWeightElements = runCatching {
            Math.multiplyExact(inputColumns, outputColumns)
        }.getOrNull() ?: return false
        if (weightTensors.any { it.size != expectedWeightElements }) return false
        if (!ensureModelTensorResidencyReady()) return false

        val residentWeights = ArrayList<VulkanTensorAllocation>(weightTensors.size)
        for (parameters in weightTensors) {
            val allocation = acquireResidentModelParameterTensor(parameters) ?: return false
            residentWeights.add(allocation)
        }

        val temporaryActivations = ArrayList<VulkanActivationTensor>(2)
        var stagedOutputs: Array<Array<FloatArray>>? = null
        var dispatchAndReadbackSucceeded = false
        var cleanupSucceeded = true

        try {
            val inputTensor = activationTensorRegistry.allocate(inputRows.size, inputColumns)
            if (inputTensor != null) {
                temporaryActivations.add(inputTensor)
                if (activationTensorRegistry.upload(inputTensor, inputRows)) {
                    val outputRowCount = runCatching {
                        Math.multiplyExact(inputRows.size, weightTensors.size)
                    }.getOrNull()
                    val outputTensor = outputRowCount?.let { rows ->
                        activationTensorRegistry.allocate(rows, outputColumns)
                    }
                    if (outputTensor != null) {
                        temporaryActivations.add(outputTensor)
                        if (activationTensorRegistry.applyLinearSet(
                                inputTensor,
                                residentWeights,
                                outputTensor,
                                transposeWeights
                            )
                        ) {
                            val packedOutput = activationTensorRegistry.download(outputTensor)
                            if (packedOutput != null && packedOutput.size == outputRowCount) {
                                stagedOutputs = Array(weightTensors.size) { batch ->
                                    Array(inputRows.size) { row ->
                                        packedOutput[batch * inputRows.size + row]
                                    }
                                }
                                dispatchAndReadbackSucceeded = true
                            }
                        }
                    }
                }
            }
        } catch (error: OutOfMemoryError) {
            Log.w(TAG, "Resident projection-set activation allocation exceeded the GPU/host budget.", error)
        } catch (error: RuntimeException) {
            Log.e(TAG, "Resident projection-set dispatch failed before output commit.", error)
        } finally {
            for (tensor in temporaryActivations.asReversed()) {
                if (!activationTensorRegistry.release(tensor)) cleanupSucceeded = false
            }
        }

        if (!cleanupSucceeded) {
            Log.w(TAG, "Resident projection-set temporary activation cleanup was incomplete; using fallback.")
            return false
        }
        if (!dispatchAndReadbackSucceeded) return false

        val completedOutputs = stagedOutputs ?: return false
        for (batch in outputRows.indices) {
            for (row in inputRows.indices) {
                System.arraycopy(
                    completedOutputs[batch][row],
                    0,
                    outputRows[batch][row],
                    0,
                    outputColumns
                )
            }
        }
        return true
    }


    /**
     * Executes M3B–M3D as one resident Vulkan attention graph.
     * Q/K/V are packed into one upload; score, causal softmax, weighted V,
     * W_O projection and residual run in one command buffer. Probabilities and
     * attention outputs remain available as caches for the existing backward pass.
     * Host arrays are committed only after every dispatch, readback and release succeeds.
     */
    @Synchronized
    internal fun tryApplyResidentAttention(
        queryRows: Array<FloatArray>,
        keyRows: Array<FloatArray>,
        valueRows: Array<FloatArray>,
        residualRows: Array<FloatArray>,
        outputWeights: Float16ParameterArray,
        attentionWeights: Array<Array<FloatArray>>,
        attentionOutputRows: Array<FloatArray>,
        projectedOutputRows: Array<FloatArray>,
        normalizedOutputRows: Array<FloatArray>,
        modelWidth: Int,
        headCount: Int
    ): Boolean {
        val sequenceLength = queryRows.size
        if (!isResidentAttentionAvailable ||
            sequenceLength <= 0 ||
            modelWidth <= 0 ||
            headCount <= 0 ||
            modelWidth % headCount != 0 ||
            keyRows.size != sequenceLength ||
            valueRows.size != sequenceLength ||
            residualRows.size != sequenceLength ||
            attentionOutputRows.size != sequenceLength ||
            projectedOutputRows.size != sequenceLength ||
            normalizedOutputRows.size != sequenceLength ||
            attentionWeights.size != headCount ||
            queryRows.any { it.size != modelWidth } ||
            keyRows.any { it.size != modelWidth } ||
            valueRows.any { it.size != modelWidth } ||
            residualRows.any { it.size != modelWidth } ||
            attentionOutputRows.any { it.size != modelWidth } ||
            projectedOutputRows.any { it.size != modelWidth } ||
            normalizedOutputRows.any { it.size != modelWidth } ||
            attentionWeights.any { head ->
                head.size != sequenceLength || head.any { it.size != sequenceLength }
            }) {
            return false
        }

        val weightElementCount = runCatching {
            Math.multiplyExact(modelWidth, modelWidth)
        }.getOrNull() ?: return false
        if (outputWeights.size != weightElementCount || !ensureModelTensorResidencyReady()) {
            return false
        }
        val residentWeights = acquireResidentModelParameterTensor(outputWeights) ?: return false

        val packedQkvRows = try {
            Array(Math.multiplyExact(sequenceLength, 3)) { index ->
                when {
                    index < sequenceLength -> queryRows[index]
                    index < sequenceLength * 2 -> keyRows[index - sequenceLength]
                    else -> valueRows[index - sequenceLength * 2]
                }
            }
        } catch (error: ArithmeticException) {
            return false
        } catch (error: OutOfMemoryError) {
            return false
        }

        val temporary = ArrayList<VulkanActivationTensor>(6)
        var stagedScores: Array<FloatArray>? = null
        var stagedAttention: Array<FloatArray>? = null
        var stagedProjected: Array<FloatArray>? = null
        var stagedNormalized: Array<FloatArray>? = null
        var completed = false
        var cleanupSucceeded = true

        try {
            val qkv = activationTensorRegistry.allocate(packedQkvRows.size, modelWidth)
            if (qkv != null) {
                temporary.add(qkv)
                val residual = activationTensorRegistry.allocate(sequenceLength, modelWidth)
                if (residual != null) {
                    temporary.add(residual)
                    val scoreRows = runCatching {
                        Math.multiplyExact(sequenceLength, headCount)
                    }.getOrNull()
                    val scores = scoreRows?.let { rows ->
                        activationTensorRegistry.allocate(rows, sequenceLength)
                    }
                    if (scores != null) {
                        temporary.add(scores)
                        val attention = activationTensorRegistry.allocate(sequenceLength, modelWidth)
                        if (attention != null) {
                            temporary.add(attention)
                            val projected = activationTensorRegistry.allocate(sequenceLength, modelWidth)
                            if (projected != null) {
                                temporary.add(projected)
                                val normalized = activationTensorRegistry.allocate(sequenceLength, modelWidth)
                                if (normalized != null) {
                                    temporary.add(normalized)
                                    if (activationTensorRegistry.upload(qkv, packedQkvRows) &&
                                        activationTensorRegistry.upload(residual, residualRows) &&
                                        activationTensorRegistry.applyAttention(
                                            qkv = qkv,
                                            scores = scores,
                                            attentionOutput = attention,
                                            projectedOutput = projected,
                                            normalizedOutput = normalized,
                                            residual = residual,
                                            outputWeights = residentWeights,
                                            sequenceLength = sequenceLength,
                                            modelWidth = modelWidth,
                                            headCount = headCount,
                                            headDimension = modelWidth / headCount
                                        )
                                    ) {
                                        stagedScores = activationTensorRegistry.download(scores)
                                        stagedAttention = activationTensorRegistry.download(attention)
                                        stagedProjected = activationTensorRegistry.download(projected)
                                        stagedNormalized = activationTensorRegistry.download(normalized)
                                        completed =
                                            stagedScores != null &&
                                            stagedAttention != null &&
                                            stagedProjected != null &&
                                            stagedNormalized != null
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } catch (error: OutOfMemoryError) {
            Log.w(TAG, "Resident attention workspace exceeded the activation memory budget.", error)
        } catch (error: RuntimeException) {
            Log.e(TAG, "Resident attention graph failed before host-output commit.", error)
        } finally {
            for (tensor in temporary.asReversed()) {
                if (!activationTensorRegistry.release(tensor)) cleanupSucceeded = false
            }
        }

        if (!cleanupSucceeded) {
            Log.w(TAG, "Resident attention tensor release failed; rebuilding residency before retry.")
            invalidateTensorResidency(releaseAllocations = true)
            residencyRebuildPending = lastModelParameterInventory.isNotEmpty()
            return false
        }
        if (!completed) return false

        val scores = stagedScores ?: return false
        val attention = stagedAttention ?: return false
        val projected = stagedProjected ?: return false
        val normalized = stagedNormalized ?: return false
        if (scores.size != headCount * sequenceLength ||
            attention.size != sequenceLength ||
            projected.size != sequenceLength ||
            normalized.size != sequenceLength) {
            return false
        }

        for (head in 0 until headCount) {
            for (query in 0 until sequenceLength) {
                System.arraycopy(
                    scores[head * sequenceLength + query],
                    0,
                    attentionWeights[head][query],
                    0,
                    sequenceLength
                )
            }
        }
        for (row in 0 until sequenceLength) {
            System.arraycopy(attention[row], 0, attentionOutputRows[row], 0, modelWidth)
            System.arraycopy(projected[row], 0, projectedOutputRows[row], 0, modelWidth)
            System.arraycopy(normalized[row], 0, normalizedOutputRows[row], 0, modelWidth)
        }
        return true
    }

    /** Reset failed native state before the user explicitly retries Vulkan initialization. */
    @Synchronized
    fun resetRuntimeForRetry() {
        if (initializationAttempted) {
            runCatching { nativeDestroy() }
                .onFailure { error ->
                    Log.w(TAG, "Vulkan native cleanup during runtime retry failed.", error)
                }
        }
        nativeReady = false
        fp16RuntimeReady = false
        fp16DisabledAfterRuntimeFailure = false
        disabledAfterRuntimeFailure = false
        initializationAttempted = false
        residencySuspended = false
        residencyBlockedAfterFailure = false
        residencyRebuildPending = false
        runtimeRestartPending = false
        invalidateTensorResidency(releaseAllocations = false)
        _modelTensorResidencyBudgetBytes.value = 0L
        _deviceCapabilities.value = null
        _gpuPerformanceMetrics.value = VulkanGpuPerformanceMetrics()
        _tensorArenaMetrics.value = VulkanTensorArenaMetrics()
        _currentBackend = "Kotlin CPU"
        _runtimeStatus.value = "Kotlin CPU — Vulkan retry requested"
        publishInitializationProgress(
            status = "Waiting for model readiness",
            driverInfo = "Vulkan driver detection pending",
            isInitializing = false
        )
    }

    @Synchronized
    fun initializeAfterModelReady(): Boolean {
        if (isRuntimeAvailable) {
            publishInitializationProgress(
                status = "Vulkan runtime verified",
                driverInfo = _initializationProgress.value.driverInfo,
                isInitializing = false,
                isVerified = true
            )
            return true
        }
        if (initializationAttempted) return false

        residentActivationLinearReady = false
        residentAttentionReady = false
        initializationAttempted = true
        _deviceCapabilities.value = null
        _runtimeStatus.value = "Preparing Vulkan after model initialization…"
        publishInitializationProgress(
            status = "Initializing Vulkan",
            driverInfo = "Detecting Vulkan compute driver...",
            isInitializing = true
        )

        var detectedDriverInfo = "Vulkan driver unavailable"
        var nativeLibraryLoaded = false
        return try {
            System.loadLibrary("argus_training")
            nativeLibraryLoaded = true
            invalidateTensorResidency(releaseAllocations = false)
            val initialized = nativeInitialize()
            if (initialized) {
                detectedDriverInfo = runCatching { nativeGetDriverInfo() }
                    .getOrDefault("Vulkan driver information unavailable")
                val detectedCapabilities = runCatching {
                    VulkanDeviceCapabilities.fromJson(nativeGetDeviceCapabilities())
                }.onFailure { error ->
                    Log.w(TAG, "Vulkan runtime started but capability metadata could not be parsed.", error)
                }.getOrNull()
                _deviceCapabilities.value = detectedCapabilities
                _modelTensorResidencyBudgetBytes.value =
                    calculateModelTensorResidencyBudgetBytes(detectedCapabilities)
                _activationTensorResidencyBudgetBytes.value =
                    calculateActivationTensorResidencyBudgetBytes(detectedCapabilities)
                Log.i(
                    TAG,
                    "FP16 model-residency budget: ${_modelTensorResidencyBudgetBytes.value} bytes."
                )
                if (detectedCapabilities != null) {
                    Log.i(
                        TAG,
                        "GPU capabilities: ${detectedCapabilities.deviceName}; " +
                            "Vulkan ${detectedCapabilities.apiVersion}; " +
                            "subgroup ${detectedCapabilities.subgroupSize}; " +
                            "FP16 ${detectedCapabilities.shaderFloat16}; " +
                            "INT8 dot ${detectedCapabilities.hasAcceleratedInt8DotProduct}; " +
                            "cooperative matrix ${detectedCapabilities.hasUsableCooperativeMatrixConfiguration}."
                    )
                }
                publishInitializationProgress(
                    status = "Verifying Vulkan compute kernels",
                    driverInfo = detectedDriverInfo,
                    isInitializing = true
                )
            }

            val verified = initialized && runCatching {
                verifyNativeKernels()
            }.getOrDefault(false)

            residentAttentionReady = false
            residentActivationLinearReady = verified && runCatching {
                verifyResidentActivationLinearKernel() &&
                    verifyResidentActivationLinearBatchKernel()
            }.getOrDefault(false)
            residentAttentionReady = verified && residentActivationLinearReady && runCatching {
                verifyResidentAttentionKernel()
            }.getOrDefault(false)
            if (verified && !residentAttentionReady) {
                Log.w(
                    TAG,
                    "Resident causal-attention self-check failed; the existing Kotlin attention fallback remains active."
                )
            }
            if (verified && !residentActivationLinearReady) {
                Log.w(
                    TAG,
                    "Resident activation-linear kernel self-check failed; existing verified Vulkan/CPU paths remain active."
                )
            }

            if (verified && runCatching { nativeSupportsFp16() }.getOrDefault(false)) {
                publishInitializationProgress(
                    status = "Verifying FP16 acceleration",
                    driverInfo = detectedDriverInfo,
                    isInitializing = true
                )
                fp16RuntimeReady = runCatching {
                    verifyNativeFp16Kernel()
                }.getOrDefault(false)
                if (!fp16RuntimeReady) {
                    Log.w(
                        TAG,
                        "Vulkan FP16 kernel self-check failed; the verified FP32 Vulkan path remains available."
                    )
                }
            }

            if (!verified) {
                releaseNativeRuntimeAfterFailure(nativeLibraryLoaded)
            }

            nativeReady = verified
            if (verified) {
                refreshGpuPerformanceMetrics()
                _currentBackend = "Kotlin CPU"
                _runtimeStatus.value =
                    "Vulkan runtime verified — eligible operations can use the GPU"
                publishInitializationProgress(
                    status = "Vulkan runtime verified",
                    driverInfo = detectedDriverInfo,
                    isInitializing = false,
                    isVerified = true
                )
                Log.i(
                    TAG,
                    "Vulkan Compute initialized and passed linear, batched-linear, and AdamW numerical self-checks."
                )
            } else {
                disabledAfterRuntimeFailure = true
                _currentBackend = "Kotlin CPU"
                _runtimeStatus.value = "Kotlin CPU fallback — Vulkan self-check was unsuccessful"
                publishInitializationProgress(
                    status = "Vulkan unavailable — CPU fallback",
                    driverInfo = detectedDriverInfo,
                    isInitializing = false,
                    isCpuFallback = true
                )
                Log.w(TAG, "Vulkan is unavailable or failed verification; Kotlin CPU remains active.")
            }
            verified
        } catch (error: LinkageError) {
            releaseNativeRuntimeAfterFailure(nativeLibraryLoaded)
            disabledAfterRuntimeFailure = true
            _currentBackend = "Kotlin CPU"
            _runtimeStatus.value = "Kotlin CPU fallback — native Vulkan library unavailable"
            publishInitializationProgress(
                status = "Vulkan loader unavailable — CPU fallback",
                driverInfo = "Vulkan driver unavailable",
                isInitializing = false,
                isCpuFallback = true
            )
            Log.i(TAG, "Native Vulkan library or Vulkan loader is unavailable.", error)
            false
        } catch (error: SecurityException) {
            releaseNativeRuntimeAfterFailure(nativeLibraryLoaded)
            disabledAfterRuntimeFailure = true
            _currentBackend = "Kotlin CPU"
            _runtimeStatus.value = "Kotlin CPU fallback — native Vulkan access was blocked"
            publishInitializationProgress(
                status = "Vulkan access blocked — CPU fallback",
                driverInfo = detectedDriverInfo,
                isInitializing = false,
                isCpuFallback = true
            )
            Log.w(TAG, "Native Vulkan library loading was blocked.", error)
            false
        } catch (error: OutOfMemoryError) {
            releaseNativeRuntimeAfterFailure(nativeLibraryLoaded)
            disabledAfterRuntimeFailure = true
            _currentBackend = "Kotlin CPU"
            _runtimeStatus.value = "Kotlin CPU fallback — insufficient memory for Vulkan initialization"
            publishInitializationProgress(
                status = "Vulkan memory allocation failed — CPU fallback",
                driverInfo = detectedDriverInfo,
                isInitializing = false,
                isCpuFallback = true
            )
            Log.w(TAG, "Vulkan initialization exceeded the available memory budget; keeping CPU fallback.", error)
            false
        } catch (error: RuntimeException) {
            releaseNativeRuntimeAfterFailure(nativeLibraryLoaded)
            disabledAfterRuntimeFailure = true
            _currentBackend = "Kotlin CPU"
            _runtimeStatus.value = "Kotlin CPU fallback — Vulkan initialization failed"
            publishInitializationProgress(
                status = "Vulkan initialization failed — CPU fallback",
                driverInfo = detectedDriverInfo,
                isInitializing = false,
                isCpuFallback = true
            )
            Log.w(TAG, "Native Vulkan initialization failed.", error)
            false
        }
    }


    private fun verifyNativeKernels(): Boolean {
        val input = floatArrayOf(
            1f, 2f, 3f,
            4f, 5f, 6f
        )
        val normalWeights = floatArrayOf(
            1f, 2f,
            3f, 4f,
            5f, 6f
        )
        val normalOutput = FloatArray(4)
        if (!nativeApplyLinear(
                input,
                normalWeights,
                normalOutput,
                2,
                1,
                2,
                3,
                2,
                false
            )) {
            return false
        }

        val expectedOutput = floatArrayOf(22f, 28f, 49f, 64f)
        if (!approximatelyEqual(normalOutput, expectedOutput, 1e-4f)) {
            return false
        }

        val transposedWeights = floatArrayOf(
            1f, 3f, 5f,
            2f, 4f, 6f
        )
        val transposedOutput = FloatArray(4)
        if (!nativeApplyLinear(
                input,
                transposedWeights,
                transposedOutput,
                2,
                1,
                2,
                3,
                2,
                true
            ) ||
            !approximatelyEqual(transposedOutput, expectedOutput, 1e-4f)) {
            return false
        }

        if (!verifyBatchedLinearKernel()) {
            return false
        }

        val parameters = floatArrayOf(1f, 2f)
        val gradients = floatArrayOf(0.1f, -0.2f)
        val firstMoments = FloatArray(2)
        val secondMoments = FloatArray(2)
        val adamwUpdated = nativeApplyAdamWUpdate(
            parameterGroups = arrayOf(parameters),
            gradientGroups = arrayOf(gradients),
            firstMomentGroups = arrayOf(firstMoments),
            secondMomentGroups = arrayOf(secondMoments),
            weightDecayByGroup = floatArrayOf(0f),
            learningRate = 0.01f,
            beta1 = 0.9f,
            beta2 = 0.999f,
            biasCorrection1 = 0.1f,
            biasCorrection2 = 0.001f,
            epsilon = 1e-8f,
            gradientClipFactor = 1f
        )

        if (!adamwUpdated ||
            !approximatelyEqual(parameters, floatArrayOf(0.99f, 2.01f), 1e-4f) ||
            !approximatelyEqual(firstMoments, floatArrayOf(0.01f, -0.02f), 1e-5f) ||
            !approximatelyEqual(secondMoments, floatArrayOf(0.00001f, 0.00004f), 1e-7f)) {
            return false
        }

        // Weight-decay groups are intentionally interleaved. The native
        // optimizer sorts them into contiguous GPU segments and must still
        // commit every result to its original Java tensor reference.
        val groupedParameterReference = arrayOf(
            floatArrayOf(1.0f, 2.0f),
            floatArrayOf(3.0f, 4.0f),
            floatArrayOf(-1.0f, 0.5f)
        )
        val groupedParameters = Array(groupedParameterReference.size) {
            groupedParameterReference[it].copyOf()
        }
        val groupedGradients = arrayOf(
            floatArrayOf(0.1f, -0.2f),
            floatArrayOf(0.3f, -0.4f),
            floatArrayOf(0.5f, -0.6f)
        )
        val groupedFirstMoments = Array(groupedParameters.size) { FloatArray(2) }
        val groupedSecondMoments = Array(groupedParameters.size) { FloatArray(2) }
        val groupedDecay = floatArrayOf(0.01f, 0.0f, 0.01f)
        val groupedUpdated = nativeApplyAdamWUpdate(
            parameterGroups = groupedParameters,
            gradientGroups = groupedGradients,
            firstMomentGroups = groupedFirstMoments,
            secondMomentGroups = groupedSecondMoments,
            weightDecayByGroup = groupedDecay,
            learningRate = 0.01f,
            beta1 = 0.9f,
            beta2 = 0.999f,
            biasCorrection1 = 0.1f,
            biasCorrection2 = 0.001f,
            epsilon = 1e-8f,
            gradientClipFactor = 1f
        )
        if (!groupedUpdated) return false

        for (groupIndex in groupedParameters.indices) {
            val expectedParameters = FloatArray(2)
            val expectedFirstMoments = FloatArray(2)
            val expectedSecondMoments = FloatArray(2)
            for (elementIndex in 0 until 2) {
                val gradient = groupedGradients[groupIndex][elementIndex]
                val firstMoment = 0.1f * gradient
                val secondMoment = 0.001f * gradient * gradient
                val correctedFirst = firstMoment / 0.1f
                val correctedSecond = secondMoment / 0.001f
                val parameter = groupedParameterReference[groupIndex][elementIndex]
                val update = correctedFirst /
                    (kotlin.math.sqrt(correctedSecond) + 1e-8f) +
                    groupedDecay[groupIndex] * parameter

                expectedParameters[elementIndex] = parameter - 0.01f * update
                expectedFirstMoments[elementIndex] = firstMoment
                expectedSecondMoments[elementIndex] = secondMoment
            }

            if (!approximatelyEqual(groupedParameters[groupIndex], expectedParameters, 1e-4f) ||
                !approximatelyEqual(groupedFirstMoments[groupIndex], expectedFirstMoments, 1e-5f) ||
                !approximatelyEqual(groupedSecondMoments[groupIndex], expectedSecondMoments, 1e-7f)) {
                return false
            }
        }
        return true
    }


    /**
     * Verifies the first M3 resident activation operation against an FP32 Kotlin reference.
     * Activations remain FP32 in the arena; only the authoritative model matrix is FP16.
     */
    private fun verifyResidentActivationLinearKernel(): Boolean {
        val rows = 9
        val inputColumns = 19
        val outputColumns = 17
        val input = FloatArray(rows * inputColumns) { index ->
            ((index % 23) - 11) * 0.03125f
        }
        val fullPrecisionWeights = FloatArray(inputColumns * outputColumns) { index ->
            ((index % 19) - 9) * 0.0625f
        }
        val fp16Weights = Float16ParameterArray.fromFloatArray(fullPrecisionWeights)
        val normalWeights = fp16Weights.toShortArray()
        val transposedWeights = ShortArray(normalWeights.size)
        for (inner in 0 until inputColumns) {
            for (column in 0 until outputColumns) {
                transposedWeights[column * inputColumns + inner] =
                    normalWeights[inner * outputColumns + column]
            }
        }

        fun expected(weights: ShortArray, transpose: Boolean): FloatArray {
            val result = FloatArray(rows * outputColumns)
            for (row in 0 until rows) {
                for (column in 0 until outputColumns) {
                    var sum = 0f
                    for (inner in 0 until inputColumns) {
                        val weightIndex = if (transpose) {
                            column * inputColumns + inner
                        } else {
                            inner * outputColumns + column
                        }
                        sum += input[row * inputColumns + inner] *
                            Float16Codec.toFloat(weights[weightIndex])
                    }
                    result[row * outputColumns + column] = sum
                }
            }
            return result
        }

        var inputHandle = 0L
        var weightHandle = 0L
        var outputHandle = 0L
        try {
            val inputBytes = encodeFloatArray(input)
            val inputByteCount = inputBytes.size.toLong()
            val weightBytes = fp16Weights.toLittleEndianByteArray()
            val outputByteCount = Math.multiplyExact(
                (rows * outputColumns).toLong(),
                Float.SIZE_BYTES.toLong()
            )
            inputHandle = nativeAllocateTensor(inputByteCount)
            weightHandle = nativeAllocateTensor(weightBytes.size.toLong())
            outputHandle = nativeAllocateTensor(outputByteCount)
            if (inputHandle == 0L || weightHandle == 0L || outputHandle == 0L) {
                return false
            }
            if (!nativeUploadTensor(inputHandle, inputBytes) ||
                !nativeUploadTensor(weightHandle, weightBytes)) {
                return false
            }

            fun verify(weights: ShortArray, transpose: Boolean): Boolean {
                if (transpose) {
                    val transposedParameterBytes = ByteArray(
                        Math.multiplyExact(weights.size, Short.SIZE_BYTES)
                    )
                    val transposedBuffer = ByteBuffer.wrap(transposedParameterBytes)
                        .order(ByteOrder.LITTLE_ENDIAN)
                    for (value in weights) transposedBuffer.putShort(value)
                    if (!nativeUploadTensor(weightHandle, transposedParameterBytes)) return false
                } else if (!nativeUploadTensor(weightHandle, fp16Weights.toLittleEndianByteArray())) {
                    return false
                }

                if (!nativeApplyResidentLinear(
                        inputHandle,
                        weightHandle,
                        outputHandle,
                        rows,
                        inputColumns,
                        outputColumns,
                        transpose
                    )) {
                    return false
                }
                val outputBytes = ByteArray(outputByteCount.toInt())
                if (!nativeDownloadTensor(outputHandle, outputBytes)) return false
                return approximatelyEqual(
                    decodeFloatArray(outputBytes),
                    expected(weights, transpose),
                    1e-4f
                )
            }

            return verify(normalWeights, transpose = false) &&
                verify(transposedWeights, transpose = true)
        } catch (error: LinkageError) {
            Log.e(TAG, "Resident activation-linear self-check entry point is unavailable.", error)
            return false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Resident activation-linear self-check failed.", error)
            return false
        } catch (error: OutOfMemoryError) {
            Log.w(TAG, "Resident activation-linear self-check exceeded available memory.", error)
            return false
        } finally {
            if (outputHandle != 0L) runCatching { nativeReleaseTensor(outputHandle) }
            if (weightHandle != 0L) runCatching { nativeReleaseTensor(weightHandle) }
            if (inputHandle != 0L) runCatching { nativeReleaseTensor(inputHandle) }
        }
    }


    /**
     * Verifies one shared FP32 input against three independently resident FP16 matrices.
     * Both matrix orientations are checked, including partial output tiles and a second K tile.
     */
    private fun verifyResidentActivationLinearBatchKernel(): Boolean {
        val batchCount = 3
        val rows = 9
        val inputColumns = 19
        val outputColumns = 17
        val input = FloatArray(rows * inputColumns) { index ->
            ((index % 23) - 11) * 0.03125f
        }
        val packedMatrices = Array(batchCount) { batch ->
            Float16ParameterArray.fromFloatArray(
                FloatArray(inputColumns * outputColumns) { index ->
                    (((index * (batch + 3)) % 29) - 14) * 0.015625f
                }
            ).toShortArray()
        }
        val transposedMatrices = Array(batchCount) { batch ->
            ShortArray(packedMatrices[batch].size).also { transposed ->
                for (inner in 0 until inputColumns) {
                    for (column in 0 until outputColumns) {
                        transposed[column * inputColumns + inner] =
                            packedMatrices[batch][inner * outputColumns + column]
                    }
                }
            }
        }

        fun encodeHalf(values: ShortArray): ByteArray {
            val bytes = ByteArray(Math.multiplyExact(values.size, Short.SIZE_BYTES))
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            for (value in values) buffer.putShort(value)
            return bytes
        }

        fun expected(transposeWeights: Boolean): FloatArray {
            val result = FloatArray(batchCount * rows * outputColumns)
            for (batch in 0 until batchCount) {
                val weights = if (transposeWeights) transposedMatrices[batch] else packedMatrices[batch]
                for (row in 0 until rows) {
                    for (column in 0 until outputColumns) {
                        var sum = 0f
                        for (inner in 0 until inputColumns) {
                            val weightIndex = if (transposeWeights) {
                                column * inputColumns + inner
                            } else {
                                inner * outputColumns + column
                            }
                            sum += input[row * inputColumns + inner] *
                                Float16Codec.toFloat(weights[weightIndex])
                        }
                        result[(batch * rows + row) * outputColumns + column] = sum
                    }
                }
            }
            return result
        }

        var inputHandle = 0L
        var outputHandle = 0L
        val weightHandles = LongArray(batchCount)
        try {
            val inputBytes = encodeFloatArray(input)
            val outputByteCount = Math.multiplyExact(
                Math.multiplyExact(batchCount, Math.multiplyExact(rows, outputColumns)).toLong(),
                Float.SIZE_BYTES.toLong()
            )
            inputHandle = nativeAllocateTensor(inputBytes.size.toLong())
            outputHandle = nativeAllocateTensor(outputByteCount)
            if (inputHandle == 0L || outputHandle == 0L) return false

            for (batch in 0 until batchCount) {
                val weightBytes = encodeHalf(packedMatrices[batch])
                weightHandles[batch] = nativeAllocateTensor(weightBytes.size.toLong())
                if (weightHandles[batch] == 0L ||
                    !nativeUploadTensor(weightHandles[batch], weightBytes)
                ) {
                    return false
                }
            }
            if (!nativeUploadTensor(inputHandle, inputBytes)) return false

            fun verifyOrientation(transposeWeights: Boolean): Boolean {
                if (transposeWeights) {
                    for (batch in 0 until batchCount) {
                        if (!nativeUploadTensor(
                                weightHandles[batch],
                                encodeHalf(transposedMatrices[batch])
                            )
                        ) return false
                    }
                }

                if (!nativeApplyResidentLinearBatch(
                        inputHandle,
                        weightHandles,
                        outputHandle,
                        rows,
                        inputColumns,
                        outputColumns,
                        transposeWeights
                    )
                ) return false

                val outputElementCount = Math.multiplyExact(
                    batchCount,
                    Math.multiplyExact(rows, outputColumns)
                )
                val outputBytes = ByteArray(
                    Math.multiplyExact(outputElementCount, Float.SIZE_BYTES)
                )
                if (!nativeDownloadTensor(outputHandle, outputBytes)) return false
                return approximatelyEqual(
                    decodeFloatArray(outputBytes),
                    expected(transposeWeights),
                    1e-4f
                )
            }

            return verifyOrientation(transposeWeights = false) &&
                verifyOrientation(transposeWeights = true)
        } catch (error: LinkageError) {
            Log.e(TAG, "Resident activation batch self-check entry point is unavailable.", error)
            return false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Resident activation batch self-check failed.", error)
            return false
        } catch (error: OutOfMemoryError) {
            Log.w(TAG, "Resident activation batch self-check exceeded available memory.", error)
            return false
        } finally {
            for (handle in weightHandles) {
                if (handle != 0L) runCatching { nativeReleaseTensor(handle) }
            }
            if (outputHandle != 0L) runCatching { nativeReleaseTensor(outputHandle) }
            if (inputHandle != 0L) runCatching { nativeReleaseTensor(inputHandle) }
        }
    }


    private fun tryApplyResidentAttention(
        qkv: VulkanTensorAllocation,
        scores: VulkanTensorAllocation,
        attentionOutput: VulkanTensorAllocation,
        projectedOutput: VulkanTensorAllocation,
        normalizedOutput: VulkanTensorAllocation,
        residual: VulkanTensorAllocation,
        outputWeights: VulkanTensorAllocation,
        sequenceLength: Int,
        modelWidth: Int,
        headCount: Int,
        headDimension: Int
    ): Boolean {
        if (!isRuntimeAvailable || !residentAttentionReady ||
            sequenceLength <= 0 || modelWidth <= 0 ||
            headCount <= 0 || headDimension <= 0) {
            return false
        }

        val sequenceModelElements = runCatching {
            Math.multiplyExact(sequenceLength.toLong(), modelWidth.toLong())
        }.getOrNull() ?: return false
        val qkvBytes = runCatching {
            Math.multiplyExact(Math.multiplyExact(sequenceModelElements, 3L), Float.SIZE_BYTES.toLong())
        }.getOrNull() ?: return false
        val scoreElements = runCatching {
            Math.multiplyExact(
                Math.multiplyExact(headCount.toLong(), sequenceLength.toLong()),
                sequenceLength.toLong()
            )
        }.getOrNull() ?: return false
        val scoreBytes = runCatching {
            Math.multiplyExact(scoreElements, Float.SIZE_BYTES.toLong())
        }.getOrNull() ?: return false
        val activationBytes = runCatching {
            Math.multiplyExact(sequenceModelElements, Float.SIZE_BYTES.toLong())
        }.getOrNull() ?: return false
        val weightBytes = runCatching {
            Math.multiplyExact(
                Math.multiplyExact(modelWidth.toLong(), modelWidth.toLong()),
                Short.SIZE_BYTES.toLong()
            )
        }.getOrNull() ?: return false

        if (qkv.byteCount < qkvBytes || scores.byteCount < scoreBytes ||
            attentionOutput.byteCount < activationBytes ||
            projectedOutput.byteCount < activationBytes ||
            normalizedOutput.byteCount < activationBytes ||
            residual.byteCount < activationBytes ||
            outputWeights.byteCount < weightBytes) {
            return false
        }

        val applied = try {
            nativeRunResidentAttention(
                qkv.nativeHandle,
                scores.nativeHandle,
                attentionOutput.nativeHandle,
                projectedOutput.nativeHandle,
                normalizedOutput.nativeHandle,
                residual.nativeHandle,
                outputWeights.nativeHandle,
                sequenceLength,
                modelWidth,
                headCount,
                headDimension
            )
        } catch (error: LinkageError) {
            Log.e(TAG, "Resident Vulkan attention JNI entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Resident Vulkan attention graph dispatch failed.", error)
            false
        }
        if (applied) {
            refreshGpuPerformanceMetrics()
        } else {
            residentAttentionReady = false
            Log.w(TAG, "Resident Vulkan attention/LayerNorm dispatch failed; M3B–M3E fall back to Kotlin.")
        }
        return applied
    }


    private fun verifyResidentAttentionKernel(): Boolean {
        val sequenceLength = 4
        val modelWidth = 72
        val headCount = 2
        val headDimension = modelWidth / headCount
        val query = Array(sequenceLength) { row ->
            if (row == 0) FloatArray(modelWidth) else
                FloatArray(modelWidth) { column -> (((row * 7 + column * 3) % 17) - 8) * 0.0625f }
        }
        val key = Array(sequenceLength) { row ->
            if (row == 0) FloatArray(modelWidth) else
                FloatArray(modelWidth) { column -> (((row * 5 + column * 11) % 19) - 9) * 0.046875f }
        }
        val value = Array(sequenceLength) { row ->
            if (row == 0) FloatArray(modelWidth) else
                FloatArray(modelWidth) { column -> (((row * 13 + column * 2) % 23) - 11) * 0.03125f }
        }
        val residual = Array(sequenceLength) { row ->
            if (row == 0) FloatArray(modelWidth) else
                FloatArray(modelWidth) { column -> (((row * 3 + column * 5) % 13) - 6) * 0.025f }
        }
        val outputWeight = Float16ParameterArray.fromFloatArray(
            FloatArray(modelWidth * modelWidth) { index -> (((index * 7) % 29) - 14) * 0.0078125f }
        )
        val outputWeightValues = outputWeight.toFloatArray()
        val expectedScores = Array(headCount * sequenceLength) { FloatArray(sequenceLength) }
        val expectedAttention = Array(sequenceLength) { FloatArray(modelWidth) }
        val expectedProjected = Array(sequenceLength) { FloatArray(modelWidth) }
        val expectedNormalized = Array(sequenceLength) { FloatArray(modelWidth) }
        val scale = 1.0f / kotlin.math.sqrt(headDimension.toFloat())

        for (head in 0 until headCount) {
            val offset = head * headDimension
            for (queryIndex in 0 until sequenceLength) {
                var maximum = -Float.MAX_VALUE
                for (keyIndex in 0..queryIndex) {
                    var dot = 0f
                    for (dimension in 0 until headDimension) {
                        dot += query[queryIndex][offset + dimension] * key[keyIndex][offset + dimension]
                    }
                    val score = dot * scale
                    expectedScores[head * sequenceLength + queryIndex][keyIndex] = score
                    maximum = max(maximum, score)
                }
                var denominator = 0f
                for (keyIndex in 0..queryIndex) {
                    val probability = exp(
                        expectedScores[head * sequenceLength + queryIndex][keyIndex] - maximum
                    )
                    expectedScores[head * sequenceLength + queryIndex][keyIndex] = probability
                    denominator += probability
                }
                for (keyIndex in 0..queryIndex) {
                    expectedScores[head * sequenceLength + queryIndex][keyIndex] /= denominator
                }
                for (keyIndex in queryIndex + 1 until sequenceLength) {
                    expectedScores[head * sequenceLength + queryIndex][keyIndex] = 0f
                }
                for (dimension in 0 until headDimension) {
                    var result = 0f
                    for (keyIndex in 0..queryIndex) {
                        result += expectedScores[head * sequenceLength + queryIndex][keyIndex] *
                            value[keyIndex][offset + dimension]
                    }
                    expectedAttention[queryIndex][offset + dimension] = result
                }
            }
        }

        for (row in 0 until sequenceLength) {
            for (column in 0 until modelWidth) {
                var result = 0f
                for (inner in 0 until modelWidth) {
                    result += expectedAttention[row][inner] *
                        outputWeightValues[inner * modelWidth + column]
                }
                expectedProjected[row][column] = result + residual[row][column]
            }
            var mean = 0f
            for (column in 0 until modelWidth) mean += expectedProjected[row][column]
            mean /= modelWidth
            var variance = 0f
            for (column in 0 until modelWidth) {
                val difference = expectedProjected[row][column] - mean
                variance += difference * difference
            }
            variance /= modelWidth
            val inverseStandardDeviation = 1.0f / kotlin.math.sqrt(variance + 1e-5f)
            for (column in 0 until modelWidth) {
                expectedNormalized[row][column] =
                    (expectedProjected[row][column] - mean) * inverseStandardDeviation
            }
        }

        var qkvHandle = 0L
        var scoreHandle = 0L
        var attentionOutputHandle = 0L
        var projectedOutputHandle = 0L
        var normalizedOutputHandle = 0L
        var residualHandle = 0L
        var outputWeightHandle = 0L
        try {
            val qkvValues = FloatArray(sequenceLength * modelWidth * 3)
            var cursor = 0
            for (matrix in arrayOf(query, key, value)) {
                for (row in matrix) {
                    row.copyInto(qkvValues, cursor)
                    cursor += row.size
                }
            }
            val residualValues = FloatArray(sequenceLength * modelWidth)
            for (row in 0 until sequenceLength) {
                residual[row].copyInto(residualValues, row * modelWidth)
            }
            val qkvBytes = encodeFloatArray(qkvValues)
            val residualBytes = encodeFloatArray(residualValues)
            val weightBytes = outputWeight.toLittleEndianByteArray()
            val scoreByteCount = Math.multiplyExact(
                Math.multiplyExact(headCount, sequenceLength * sequenceLength),
                Float.SIZE_BYTES
            )
            val activationByteCount = Math.multiplyExact(
                Math.multiplyExact(sequenceLength, modelWidth),
                Float.SIZE_BYTES
            )

            qkvHandle = nativeAllocateTensor(qkvBytes.size.toLong())
            scoreHandle = nativeAllocateTensor(scoreByteCount.toLong())
            attentionOutputHandle = nativeAllocateTensor(activationByteCount.toLong())
            projectedOutputHandle = nativeAllocateTensor(activationByteCount.toLong())
            normalizedOutputHandle = nativeAllocateTensor(activationByteCount.toLong())
            residualHandle = nativeAllocateTensor(residualBytes.size.toLong())
            outputWeightHandle = nativeAllocateTensor(weightBytes.size.toLong())
            if (qkvHandle == 0L || scoreHandle == 0L ||
                attentionOutputHandle == 0L || projectedOutputHandle == 0L ||
                normalizedOutputHandle == 0L || residualHandle == 0L || outputWeightHandle == 0L) {
                return false
            }

            if (!nativeUploadTensor(qkvHandle, qkvBytes) ||
                !nativeUploadTensor(residualHandle, residualBytes) ||
                !nativeUploadTensor(outputWeightHandle, weightBytes) ||
                !nativeRunResidentAttention(
                    qkvHandle, scoreHandle, attentionOutputHandle, projectedOutputHandle,
                    normalizedOutputHandle, residualHandle, outputWeightHandle,
                    sequenceLength, modelWidth, headCount, headDimension
                )) {
                return false
            }

            val actualScoresBytes = ByteArray(scoreByteCount)
            val actualAttentionBytes = ByteArray(activationByteCount)
            val actualProjectedBytes = ByteArray(activationByteCount)
            val actualNormalizedBytes = ByteArray(activationByteCount)
            if (!nativeDownloadTensor(scoreHandle, actualScoresBytes) ||
                !nativeDownloadTensor(attentionOutputHandle, actualAttentionBytes) ||
                !nativeDownloadTensor(projectedOutputHandle, actualProjectedBytes) ||
                !nativeDownloadTensor(normalizedOutputHandle, actualNormalizedBytes)) {
                return false
            }

            val expectedScoreValues = FloatArray(headCount * sequenceLength * sequenceLength)
            for (row in expectedScores.indices) {
                expectedScores[row].copyInto(expectedScoreValues, row * sequenceLength)
            }
            val expectedAttentionValues = FloatArray(sequenceLength * modelWidth)
            val expectedProjectedValues = FloatArray(sequenceLength * modelWidth)
            val expectedNormalizedValues = FloatArray(sequenceLength * modelWidth)
            for (row in 0 until sequenceLength) {
                expectedAttention[row].copyInto(expectedAttentionValues, row * modelWidth)
                expectedProjected[row].copyInto(expectedProjectedValues, row * modelWidth)
                expectedNormalized[row].copyInto(expectedNormalizedValues, row * modelWidth)
            }
            return approximatelyEqual(decodeFloatArray(actualScoresBytes), expectedScoreValues, 2e-4f) &&
                approximatelyEqual(decodeFloatArray(actualAttentionBytes), expectedAttentionValues, 2e-4f) &&
                approximatelyEqual(decodeFloatArray(actualProjectedBytes), expectedProjectedValues, 2e-4f) &&
                approximatelyEqual(decodeFloatArray(actualNormalizedBytes), expectedNormalizedValues, 2e-4f)
        } catch (error: LinkageError) {
            Log.e(TAG, "Resident attention self-check JNI entry point is unavailable.", error)
            return false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Resident attention self-check failed.", error)
            return false
        } catch (error: OutOfMemoryError) {
            Log.w(TAG, "Resident attention self-check exceeded available memory.", error)
            return false
        } finally {
            if (outputWeightHandle != 0L) runCatching { nativeReleaseTensor(outputWeightHandle) }
            if (residualHandle != 0L) runCatching { nativeReleaseTensor(residualHandle) }
            if (normalizedOutputHandle != 0L) runCatching { nativeReleaseTensor(normalizedOutputHandle) }
            if (projectedOutputHandle != 0L) runCatching { nativeReleaseTensor(projectedOutputHandle) }
            if (attentionOutputHandle != 0L) runCatching { nativeReleaseTensor(attentionOutputHandle) }
            if (scoreHandle != 0L) runCatching { nativeReleaseTensor(scoreHandle) }
            if (qkvHandle != 0L) runCatching { nativeReleaseTensor(qkvHandle) }
        }
    }

    private fun encodeFloatArray(values: FloatArray): ByteArray {
        val byteCount = Math.multiplyExact(values.size, Float.SIZE_BYTES)
        val buffer = ByteBuffer.allocate(byteCount).order(ByteOrder.LITTLE_ENDIAN)
        for (value in values) {
            if (!value.isFinite()) {
                throw IllegalArgumentException("Resident activation inputs must be finite.")
            }
            buffer.putFloat(value)
        }
        return buffer.array()
    }

    private fun decodeFloatArray(bytes: ByteArray): FloatArray {
        require(bytes.size % Float.SIZE_BYTES == 0) {
            "Resident activation output must contain complete FP32 elements."
        }
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(bytes.size / Float.SIZE_BYTES) {
            buffer.float
        }
    }

    /**
     * Compares one non-square, multi-batch Vulkan dispatch with a Kotlin CPU reference.
     * 72 outputs deliberately cross the 64-invocation workgroup boundary.
     */
    private fun verifyBatchedLinearKernel(): Boolean {
        // Exercise partial 8x16 output tiles, multiple batches, and a K
        // dimension that requires a second 16-wide shared-memory tile.
        val batchCount = 3
        val rowsPerBatch = 9
        val inputColumns = 19
        val outputColumns = 19
        val rowCount = batchCount * rowsPerBatch
        val input = FloatArray(rowCount * inputColumns) { index ->
            ((index % 17) - 8) * 0.125f
        }
        val weightsPerBatch = inputColumns * outputColumns
        val weights = FloatArray(batchCount * weightsPerBatch) { index ->
            ((index % 13) - 6) * 0.0625f
        }
        val transposedWeights = FloatArray(weights.size)
        for (batchIndex in 0 until batchCount) {
            val batchOffset = batchIndex * weightsPerBatch
            for (inner in 0 until inputColumns) {
                for (column in 0 until outputColumns) {
                    transposedWeights[batchOffset + column * inputColumns + inner] =
                        weights[batchOffset + inner * outputColumns + column]
                }
            }
        }

        fun verify(weightValues: FloatArray, transposeWeights: Boolean): Boolean {
            val expected = FloatArray(rowCount * outputColumns)
            for (batchIndex in 0 until batchCount) {
                val batchInputOffset = batchIndex * rowsPerBatch * inputColumns
                val batchWeightOffset = batchIndex * weightsPerBatch
                for (row in 0 until rowsPerBatch) {
                    val inputRowOffset = batchInputOffset + row * inputColumns
                    val outputRowOffset =
                        (batchIndex * rowsPerBatch + row) * outputColumns
                    for (column in 0 until outputColumns) {
                        var sum = 0f
                        for (inner in 0 until inputColumns) {
                            val weightIndex = if (transposeWeights) {
                                batchWeightOffset + column * inputColumns + inner
                            } else {
                                batchWeightOffset + inner * outputColumns + column
                            }
                            sum += input[inputRowOffset + inner] * weightValues[weightIndex]
                        }
                        expected[outputRowOffset + column] = sum
                    }
                }
            }

            val actual = FloatArray(expected.size)
            return nativeApplyLinear(
                input,
                weightValues,
                actual,
                rows = rowCount,
                batchCount = batchCount,
                rowsPerBatch = rowsPerBatch,
                inputColumns = inputColumns,
                outputColumns = outputColumns,
                transposeWeights = transposeWeights
            ) && approximatelyEqual(actual, expected, 1e-4f)
        }

        return verify(weights, transposeWeights = false) &&
            verify(transposedWeights, transposeWeights = true)
    }

    private fun approximatelyEqual(
        actual: FloatArray,
        expected: FloatArray,
        tolerance: Float
    ): Boolean {
        if (actual.size != expected.size) return false
        for (index in actual.indices) {
            if (!actual[index].isFinite() ||
                kotlin.math.abs(actual[index] - expected[index]) > tolerance) {
                return false
            }
        }
        return true
    }

    private fun verifyNativeFp16Kernel(): Boolean {
        // Exercise partial 8x16 tiles, a second K tile, independent batches,
        // odd packed lengths, both matrix layouts, and both persistent paths.
        val batchCount = 3
        val rowsPerBatch = 9
        val inputColumns = 19
        val outputColumns = 19
        val rowCount = batchCount * rowsPerBatch
        val input = FloatArray(rowCount * inputColumns) { index ->
            ((index % 17) - 8) * 0.125f
        }
        val sourceWeights = FloatArray(
            batchCount * inputColumns * outputColumns
        ) { index ->
            ((index % 13) - 6) * 0.0625f
        }
        val weights = Float16ParameterArray.fromFloatArray(sourceWeights).toShortArray()
        val transposedWeights = ShortArray(weights.size)
        val weightBatchSize = inputColumns * outputColumns

        for (batchIndex in 0 until batchCount) {
            val batchOffset = batchIndex * weightBatchSize
            for (inner in 0 until inputColumns) {
                for (column in 0 until outputColumns) {
                    transposedWeights[batchOffset + column * inputColumns + inner] =
                        weights[batchOffset + inner * outputColumns + column]
                }
            }
        }

        fun calculateExpected(
            weightValues: ShortArray,
            transposeWeights: Boolean
        ): FloatArray {
            val expected = FloatArray(rowCount * outputColumns)
            for (batchIndex in 0 until batchCount) {
                val inputBatchOffset = batchIndex * rowsPerBatch * inputColumns
                val weightBatchOffset = batchIndex * weightBatchSize
                for (row in 0 until rowsPerBatch) {
                    val inputRowOffset = inputBatchOffset + row * inputColumns
                    val outputRowOffset =
                        (batchIndex * rowsPerBatch + row) * outputColumns

                    for (column in 0 until outputColumns) {
                        var sum = 0f
                        for (inner in 0 until inputColumns) {
                            val inputValue = Float16Codec.toFloat(
                                Float16Codec.fromFiniteFloat(
                                    input[inputRowOffset + inner]
                                )
                            )
                            val weightIndex = if (transposeWeights) {
                                weightBatchOffset + column * inputColumns + inner
                            } else {
                                weightBatchOffset + inner * outputColumns + column
                            }
                            val weightValue = Float16Codec.toFloat(weightValues[weightIndex])
                            val halfProduct = Float16Codec.toFloat(
                                Float16Codec.fromFiniteFloat(inputValue * weightValue)
                            )
                            sum += halfProduct
                        }
                        expected[outputRowOffset + column] = sum
                    }
                }
            }
            return expected
        }

        fun calculateExpectedSingleMatrix(
            weightValues: ShortArray,
            transposeWeights: Boolean
        ): FloatArray {
            val expected = FloatArray(rowCount * outputColumns)
            for (row in 0 until rowCount) {
                val inputRowOffset = row * inputColumns
                val outputRowOffset = row * outputColumns
                for (column in 0 until outputColumns) {
                    var sum = 0f
                    for (inner in 0 until inputColumns) {
                        val inputValue = Float16Codec.toFloat(
                            Float16Codec.fromFiniteFloat(input[inputRowOffset + inner])
                        )
                        val weightIndex = if (transposeWeights) {
                            column * inputColumns + inner
                        } else {
                            inner * outputColumns + column
                        }
                        val weightValue = Float16Codec.toFloat(weightValues[weightIndex])
                        val halfProduct = Float16Codec.toFloat(
                            Float16Codec.fromFiniteFloat(inputValue * weightValue)
                        )
                        sum += halfProduct
                    }
                    expected[outputRowOffset + column] = sum
                }
            }
            return expected
        }

        fun verifyStaging(
            weightValues: ShortArray,
            transposeWeights: Boolean
        ): Boolean {
            val expected = calculateExpected(weightValues, transposeWeights)
            val actual = FloatArray(expected.size)
            return nativeApplyFp16Linear(
                input = input,
                weights = weightValues,
                output = actual,
                rows = rowCount,
                batchCount = batchCount,
                rowsPerBatch = rowsPerBatch,
                inputColumns = inputColumns,
                outputColumns = outputColumns,
                transposeWeights = transposeWeights
            ) && approximatelyEqual(actual, expected, 1e-3f)
        }

        fun verifyPersistentBatched(
            weightValues: ShortArray,
            transposeWeights: Boolean
        ): Boolean {
            val expected = calculateExpected(weightValues, transposeWeights)
            val parameters = Array(batchCount) { batchIndex ->
                val offset = batchIndex * weightBatchSize
                val batchValues = FloatArray(weightBatchSize) { index ->
                    Float16Codec.toFloat(weightValues[offset + index])
                }
                Float16ParameterArray.fromFloatArray(batchValues)
            }
            val handles = LongArray(batchCount)

            try {
                // Allocate each matrix separately. Their arena offsets are not
                // assumed to be contiguous; the kernel must use the offset table.
                for (batchIndex in parameters.indices) {
                    val byteCount = Math.multiplyExact(
                        parameters[batchIndex].size.toLong(),
                        Short.SIZE_BYTES.toLong()
                    )
                    val handle = nativeAllocateTensor(byteCount)
                    if (handle == 0L) return false
                    handles[batchIndex] = handle
                }

                for (batchIndex in parameters.indices) {
                    if (!nativeUploadTensor(
                            handles[batchIndex],
                            parameters[batchIndex].toLittleEndianByteArray()
                        )
                    ) {
                        return false
                    }
                }

                val actual = FloatArray(expected.size)
                return nativeApplyFp16LinearWithArena(
                    input = input,
                    weightHandles = handles,
                    output = actual,
                    rows = rowCount,
                    batchCount = batchCount,
                    rowsPerBatch = rowsPerBatch,
                    inputColumns = inputColumns,
                    outputColumns = outputColumns,
                    transposeWeights = transposeWeights
                ) && approximatelyEqual(actual, expected, 1e-3f)
            } catch (error: LinkageError) {
                Log.e(TAG, "Persistent batched FP16 Vulkan self-check entry point is unavailable.", error)
                return false
            } catch (error: RuntimeException) {
                Log.e(TAG, "Persistent batched FP16 Vulkan self-check failed.", error)
                return false
            } catch (error: OutOfMemoryError) {
                Log.w(TAG, "Persistent batched FP16 Vulkan self-check exceeded available memory.", error)
                return false
            } finally {
                for (handle in handles) {
                    if (handle != 0L) {
                        runCatching { nativeReleaseTensor(handle) }
                    }
                }
            }
        }

        fun verifyPersistentSingleMatrix(
            weightValues: ShortArray,
            transposeWeights: Boolean
        ): Boolean {
            val matrixValues = weightValues.copyOfRange(0, weightBatchSize)
            val parameters = Float16ParameterArray.fromFloatArray(
                FloatArray(weightBatchSize) { index ->
                    Float16Codec.toFloat(matrixValues[index])
                }
            )
            val expected = calculateExpectedSingleMatrix(matrixValues, transposeWeights)
            var handle = 0L

            try {
                val byteCount = Math.multiplyExact(
                    parameters.size.toLong(),
                    Short.SIZE_BYTES.toLong()
                )
                handle = nativeAllocateTensor(byteCount)
                if (handle == 0L) return false

                if (!nativeUploadTensor(handle, parameters.toLittleEndianByteArray())) {
                    return false
                }

                val actual = FloatArray(expected.size)
                return nativeApplyFp16LinearWithArena(
                    input = input,
                    weightHandles = longArrayOf(handle),
                    output = actual,
                    rows = rowCount,
                    batchCount = 1,
                    rowsPerBatch = rowCount,
                    inputColumns = inputColumns,
                    outputColumns = outputColumns,
                    transposeWeights = transposeWeights
                ) && approximatelyEqual(actual, expected, 1e-3f)
            } catch (error: LinkageError) {
                Log.e(TAG, "Persistent single-matrix FP16 Vulkan self-check entry point is unavailable.", error)
                return false
            } catch (error: RuntimeException) {
                Log.e(TAG, "Persistent single-matrix FP16 Vulkan self-check failed.", error)
                return false
            } catch (error: OutOfMemoryError) {
                Log.w(TAG, "Persistent single-matrix FP16 Vulkan self-check exceeded available memory.", error)
                return false
            } finally {
                if (handle != 0L) {
                    runCatching { nativeReleaseTensor(handle) }
                }
            }
        }

        return verifyStaging(weights, transposeWeights = false) &&
            verifyStaging(transposedWeights, transposeWeights = true) &&
            verifyPersistentBatched(weights, transposeWeights = false) &&
            verifyPersistentBatched(transposedWeights, transposeWeights = true) &&
            verifyPersistentSingleMatrix(weights, transposeWeights = false) &&
            verifyPersistentSingleMatrix(transposedWeights, transposeWeights = true)
    }

    /**
     * Uses a persistent GPU replica for a single FP16 model matrix. Kotlin remains
     * authoritative; the registry uploads only after the model tensor is mutated.
     */

    @Synchronized
    internal fun tryApplyResidentLinear(
        input: VulkanTensorAllocation,
        weights: VulkanTensorAllocation,
        output: VulkanTensorAllocation,
        rows: Int,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean = false
    ): Boolean {
        if (!isRuntimeAvailable ||
            !residentActivationLinearReady ||
            rows <= 0 ||
            inputColumns <= 0 ||
            outputColumns <= 0) {
            return false
        }

        val requiredInputBytes = runCatching {
            Math.multiplyExact(
                Math.multiplyExact(rows.toLong(), inputColumns.toLong()),
                Float.SIZE_BYTES.toLong()
            )
        }.getOrNull() ?: return false
        val requiredOutputBytes = runCatching {
            Math.multiplyExact(
                Math.multiplyExact(rows.toLong(), outputColumns.toLong()),
                Float.SIZE_BYTES.toLong()
            )
        }.getOrNull() ?: return false
        val requiredWeightBytes = runCatching {
            Math.multiplyExact(
                Math.multiplyExact(inputColumns.toLong(), outputColumns.toLong()),
                Short.SIZE_BYTES.toLong()
            )
        }.getOrNull() ?: return false
        if (input.byteCount < requiredInputBytes ||
            weights.byteCount < requiredWeightBytes ||
            output.byteCount < requiredOutputBytes) {
            return false
        }

        val applied = try {
            nativeApplyResidentLinear(
                input.nativeHandle,
                weights.nativeHandle,
                output.nativeHandle,
                rows,
                inputColumns,
                outputColumns,
                transposeWeights
            )
        } catch (error: LinkageError) {
            Log.e(TAG, "Resident Vulkan linear JNI entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Resident Vulkan linear dispatch failed.", error)
            false
        }
        if (applied) {
            refreshGpuPerformanceMetrics()
        } else {
            residentActivationLinearReady = false
            Log.w(TAG, "Resident activation-linear dispatch failed; this M3 path is disabled for this runtime.")
        }
        return applied
    }


    private fun tryApplyResidentLinearBatch(
        input: VulkanTensorAllocation,
        weights: List<VulkanTensorAllocation>,
        output: VulkanTensorAllocation,
        rowsPerBatch: Int,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean
    ): Boolean {
        if (!isRuntimeAvailable ||
            !residentActivationLinearReady ||
            weights.size < 2 ||
            rowsPerBatch <= 0 ||
            inputColumns <= 0 ||
            outputColumns <= 0) {
            return false
        }

        val inputByteCount = runCatching {
            Math.multiplyExact(
                Math.multiplyExact(rowsPerBatch.toLong(), inputColumns.toLong()),
                Float.SIZE_BYTES.toLong()
            )
        }.getOrNull() ?: return false
        val outputByteCount = runCatching {
            Math.multiplyExact(
                Math.multiplyExact(
                    Math.multiplyExact(rowsPerBatch.toLong(), weights.size.toLong()),
                    outputColumns.toLong()
                ),
                Float.SIZE_BYTES.toLong()
            )
        }.getOrNull() ?: return false
        val weightByteCount = runCatching {
            Math.multiplyExact(
                Math.multiplyExact(inputColumns.toLong(), outputColumns.toLong()),
                Short.SIZE_BYTES.toLong()
            )
        }.getOrNull() ?: return false

        if (input.byteCount < inputByteCount ||
            output.byteCount < outputByteCount ||
            weights.any { it.byteCount < weightByteCount }) {
            return false
        }

        val applied = try {
            nativeApplyResidentLinearBatch(
                input.nativeHandle,
                LongArray(weights.size) { index -> weights[index].nativeHandle },
                output.nativeHandle,
                rowsPerBatch,
                inputColumns,
                outputColumns,
                transposeWeights
            )
        } catch (error: LinkageError) {
            Log.e(TAG, "Resident Vulkan batched-linear JNI entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Resident Vulkan batched-linear dispatch failed.", error)
            false
        }
        if (applied) {
            refreshGpuPerformanceMetrics()
        } else {
            residentActivationLinearReady = false
            Log.w(TAG, "Resident Vulkan batched-linear dispatch failed; this resident M3 path is disabled for this runtime.")
        }
        return applied
    }

    @Synchronized
    fun tryApplyFp16Linear(
        inputRows: Array<FloatArray>,
        weights: Float16ParameterArray,
        outputRows: Array<FloatArray>,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean = false
    ): Boolean {
        require(inputRows.isNotEmpty()) { "FP16 linear input must contain at least one row." }
        require(inputColumns > 0 && outputColumns > 0) {
            "FP16 linear dimensions must be positive."
        }
        require(outputRows.size == inputRows.size) {
            "FP16 linear output row count must match its input."
        }
        val requiredWeightElements = inputColumns.toLong() * outputColumns.toLong()
        if (requiredWeightElements > weights.size.toLong()) return false

        for (row in inputRows.indices) {
            require(inputRows[row].size == inputColumns) {
                "FP16 input row $row has an unexpected width."
            }
            require(outputRows[row].size == outputColumns) {
                "FP16 output row $row has an unexpected width."
            }
        }

        val rowCount = inputRows.size
        val inputElements = rowCount.toLong() * inputColumns.toLong()
        val outputElements = rowCount.toLong() * outputColumns.toLong()
        if (inputElements > Int.MAX_VALUE.toLong() ||
            outputElements > Int.MAX_VALUE.toLong()) {
            return false
        }
        val operationCount = runCatching {
            Math.multiplyExact(inputElements, outputColumns.toLong())
        }.getOrNull() ?: return false
        if (!isFp16ComputeAvailable ||
            operationCount < MIN_LINEAR_OPERATION_COUNT ||
            !ensureModelTensorResidencyReady()) {
            return false
        }

        val flattenedInput: FloatArray
        val flattenedOutput: FloatArray
        try {
            flattenedInput = FloatArray(inputElements.toInt())
            flattenedOutput = FloatArray(outputElements.toInt())
        } catch (error: OutOfMemoryError) {
            Log.w(TAG, "Not enough heap memory to stage a persistent-weight Vulkan linear operation.", error)
            return false
        }
        for (row in inputRows.indices) {
            System.arraycopy(
                inputRows[row],
                0,
                flattenedInput,
                row * inputColumns,
                inputColumns
            )
        }

        val weightAllocation = tensorResidencyRegistry.acquire(weights) ?: return false
        val updated = try {
            nativeApplyFp16LinearWithArena(
                input = flattenedInput,
                weightHandles = longArrayOf(weightAllocation.nativeHandle),
                output = flattenedOutput,
                rows = rowCount,
                batchCount = 1,
                rowsPerBatch = rowCount,
                inputColumns = inputColumns,
                outputColumns = outputColumns,
                transposeWeights = transposeWeights
            )
        } catch (error: LinkageError) {
            Log.e(TAG, "Persistent-weight Vulkan FP16 entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Persistent-weight Vulkan FP16 dispatch failed.", error)
            false
        }

        if (!updated) {
            fp16DisabledAfterRuntimeFailure = true
            fp16RuntimeReady = false
            invalidateTensorResidency(releaseAllocations = true)
            Log.w(TAG, "Persistent-weight FP16 dispatch failed; using the Kotlin CPU path.")
            return false
        }

        for (row in outputRows.indices) {
            System.arraycopy(
                flattenedOutput,
                row * outputColumns,
                outputRows[row],
                0,
                outputColumns
            )
        }
        refreshGpuPerformanceMetrics()
        _currentBackend = "Vulkan Compute (FP16 persistent model weights)"
        return true
    }

    @Synchronized
    fun tryApplyFp16BatchedLinear(
        inputBatches: Array<Array<FloatArray>>,
        weightBatches: Array<Float16ParameterArray>,
        outputBatches: Array<Array<FloatArray>>,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean = false
    ): Boolean {
        require(inputBatches.isNotEmpty()) {
            "FP16 batched linear input must contain at least one batch."
        }
        require(inputColumns > 0 && outputColumns > 0) {
            "FP16 batched linear dimensions must be positive."
        }

        val batchCount = inputBatches.size
        require(weightBatches.size == batchCount && outputBatches.size == batchCount) {
            "FP16 batched linear inputs, weights, and outputs must have the same batch count."
        }

        val rowsPerBatch = inputBatches[0].size
        require(rowsPerBatch > 0) {
            "Every FP16 batched linear input must contain at least one row."
        }

        val requiredWeightElementsLong = Math.multiplyExact(
            inputColumns.toLong(),
            outputColumns.toLong()
        )
        if (requiredWeightElementsLong > Int.MAX_VALUE.toLong()) return false
        val requiredWeightElements = requiredWeightElementsLong.toInt()

        for (batchIndex in 0 until batchCount) {
            require(inputBatches[batchIndex].size == rowsPerBatch) {
                "FP16 batched linear inputs must have an identical row count."
            }
            require(outputBatches[batchIndex].size == rowsPerBatch) {
                "FP16 batched linear outputs must match the input row count."
            }
            if (weightBatches[batchIndex].size < requiredWeightElements) return false

            for (row in 0 until rowsPerBatch) {
                require(inputBatches[batchIndex][row].size == inputColumns) {
                    "FP16 input row $row in batch $batchIndex has an unexpected width."
                }
                require(outputBatches[batchIndex][row].size == outputColumns) {
                    "FP16 output row $row in batch $batchIndex has an unexpected width."
                }
            }
        }

        val totalRowsLong = batchCount.toLong() * rowsPerBatch.toLong()
        if (totalRowsLong > Int.MAX_VALUE.toLong()) return false

        val inputElementsLong = runCatching {
            Math.multiplyExact(totalRowsLong, inputColumns.toLong())
        }.getOrNull() ?: return false
        val outputElementsLong = runCatching {
            Math.multiplyExact(totalRowsLong, outputColumns.toLong())
        }.getOrNull() ?: return false
        val totalWeightElementsLong = runCatching {
            Math.multiplyExact(batchCount.toLong(), requiredWeightElementsLong)
        }.getOrNull() ?: return false
        if (inputElementsLong > Int.MAX_VALUE.toLong() ||
            outputElementsLong > Int.MAX_VALUE.toLong() ||
            totalWeightElementsLong > Int.MAX_VALUE.toLong()) {
            return false
        }

        val operationCount = runCatching {
            Math.multiplyExact(
                Math.multiplyExact(totalRowsLong, inputColumns.toLong()),
                outputColumns.toLong()
            )
        }.getOrNull() ?: return false
        if (!isFp16ComputeAvailable ||
            operationCount < MIN_LINEAR_OPERATION_COUNT ||
            !ensureModelTensorResidencyReady()) {
            return false
        }

        val flattenedInput: FloatArray
        val flattenedOutput: FloatArray
        try {
            flattenedInput = FloatArray(inputElementsLong.toInt())
            flattenedOutput = FloatArray(outputElementsLong.toInt())
        } catch (error: OutOfMemoryError) {
            Log.w(TAG, "Not enough heap memory to stage FP16 batched linear activations.", error)
            return false
        }

        val inputBatchSize = rowsPerBatch * inputColumns
        for (batchIndex in 0 until batchCount) {
            val inputBatchOffset = batchIndex * inputBatchSize
            for (row in 0 until rowsPerBatch) {
                System.arraycopy(
                    inputBatches[batchIndex][row],
                    0,
                    flattenedInput,
                    inputBatchOffset + row * inputColumns,
                    inputColumns
                )
            }
        }

        // The registry keeps one GPU allocation per authoritative Kotlin tensor.
        // Unchanged revisions reuse their allocation without another weight upload.
        val weightHandles = LongArray(batchCount)
        for (batchIndex in 0 until batchCount) {
            val allocation = tensorResidencyRegistry.acquire(weightBatches[batchIndex])
                ?: return false
            weightHandles[batchIndex] = allocation.nativeHandle
        }

        val updated = try {
            nativeApplyFp16LinearWithArena(
                input = flattenedInput,
                weightHandles = weightHandles,
                output = flattenedOutput,
                rows = totalRowsLong.toInt(),
                batchCount = batchCount,
                rowsPerBatch = rowsPerBatch,
                inputColumns = inputColumns,
                outputColumns = outputColumns,
                transposeWeights = transposeWeights
            )
        } catch (error: LinkageError) {
            Log.e(TAG, "Persistent batched FP16 Vulkan entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Persistent batched FP16 Vulkan dispatch failed.", error)
            false
        }

        if (!updated) {
            fp16DisabledAfterRuntimeFailure = true
            fp16RuntimeReady = false
            invalidateTensorResidency(releaseAllocations = true)
            Log.w(
                TAG,
                "Persistent batched FP16 dispatch failed; retaining the verified FP32 Vulkan/CPU fallback."
            )
            return false
        }

        val outputBatchSize = rowsPerBatch * outputColumns
        for (batchIndex in 0 until batchCount) {
            val outputBatchOffset = batchIndex * outputBatchSize
            for (row in 0 until rowsPerBatch) {
                System.arraycopy(
                    flattenedOutput,
                    outputBatchOffset + row * outputColumns,
                    outputBatches[batchIndex][row],
                    0,
                    outputColumns
                )
            }
        }

        refreshGpuPerformanceMetrics()
        _currentBackend = "Vulkan Compute (FP16 persistent batched model weights)"
        return true
    }

    @Synchronized
    fun tryApplyLinear(
        inputRows: Array<FloatArray>,
        weights: FloatArray,
        outputRows: Array<FloatArray>,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean = false
    ): Boolean = tryApplyBatchedLinear(
        inputBatches = arrayOf(inputRows),
        weightBatches = arrayOf(weights),
        outputBatches = arrayOf(outputRows),
        inputColumns = inputColumns,
        outputColumns = outputColumns,
        transposeWeights = transposeWeights
    )

    /**
     * Runs independent per-batch dense products in one Vulkan dispatch. Each batch
     * has its own contiguous input matrix and weight tensor (one batch per attention head).
     */
    @Synchronized
    fun tryApplyBatchedLinear(
        inputBatches: Array<Array<FloatArray>>,
        weightBatches: Array<FloatArray>,
        outputBatches: Array<Array<FloatArray>>,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean = false
    ): Boolean {
        require(inputBatches.isNotEmpty()) { "Batched linear input must contain at least one batch." }
        require(inputColumns > 0 && outputColumns > 0) {
            "Batched linear dimensions must be positive."
        }
        val batchCount = inputBatches.size
        require(weightBatches.size == batchCount && outputBatches.size == batchCount) {
            "Batched linear inputs, weights, and outputs must have the same batch count."
        }
        val rowsPerBatch = inputBatches[0].size
        require(rowsPerBatch > 0) { "Every batched linear input must contain at least one row." }
        val requiredWeightElements = inputColumns.toLong() * outputColumns.toLong()
        require(requiredWeightElements <= Int.MAX_VALUE.toLong()) {
            "Batched linear weight tensor exceeds the supported array size."
        }

        for (batchIndex in 0 until batchCount) {
            require(inputBatches[batchIndex].size == rowsPerBatch) {
                "Batched linear inputs must have an identical row count."
            }
            require(outputBatches[batchIndex].size == rowsPerBatch) {
                "Batched linear outputs must match the input row count."
            }
            require(weightBatches[batchIndex].size >= requiredWeightElements.toInt()) {
                "Batched linear weight tensor $batchIndex is smaller than its dimensions."
            }
            for (row in 0 until rowsPerBatch) {
                require(inputBatches[batchIndex][row].size == inputColumns) {
                    "Batched linear input row $row in batch $batchIndex has an unexpected width."
                }
                require(outputBatches[batchIndex][row].size == outputColumns) {
                    "Batched linear output row $row in batch $batchIndex has an unexpected width."
                }
            }
        }

        val totalRowsLong = batchCount.toLong() * rowsPerBatch.toLong()
        if (totalRowsLong > Int.MAX_VALUE.toLong()) return false

        val inputElementCountLong = totalRowsLong * inputColumns.toLong()
        val weightElementCountLong = batchCount.toLong() * requiredWeightElements
        val outputElementCountLong = totalRowsLong * outputColumns.toLong()
        if (inputElementCountLong > Int.MAX_VALUE.toLong() ||
            weightElementCountLong > Int.MAX_VALUE.toLong() ||
            outputElementCountLong > Int.MAX_VALUE.toLong()) {
            return false
        }

        val operationCount = totalRowsLong *
            inputColumns.toLong() *
            outputColumns.toLong()
        if (!isRuntimeAvailable ||
            disabledAfterRuntimeFailure ||
            operationCount < MIN_LINEAR_OPERATION_COUNT) {
            return false
        }

        val flattenedInput: FloatArray
        val flattenedWeights: FloatArray
        val flattenedOutput: FloatArray
        try {
            flattenedInput = FloatArray(inputElementCountLong.toInt())
            flattenedWeights = FloatArray(weightElementCountLong.toInt())
            flattenedOutput = FloatArray(outputElementCountLong.toInt())
        } catch (error: OutOfMemoryError) {
            Log.w(TAG, "Not enough heap memory to stage a Vulkan batched-linear operation.", error)
            return false
        }

        val weightBatchSize = requiredWeightElements.toInt()
        for (batchIndex in 0 until batchCount) {
            val inputBatchOffset = batchIndex * rowsPerBatch * inputColumns
            for (row in 0 until rowsPerBatch) {
                System.arraycopy(
                    inputBatches[batchIndex][row],
                    0,
                    flattenedInput,
                    inputBatchOffset + row * inputColumns,
                    inputColumns
                )
            }
            System.arraycopy(
                weightBatches[batchIndex],
                0,
                flattenedWeights,
                batchIndex * weightBatchSize,
                weightBatchSize
            )
        }

        val updated = try {
            nativeApplyLinear(
                input = flattenedInput,
                weights = flattenedWeights,
                output = flattenedOutput,
                rows = totalRowsLong.toInt(),
                batchCount = batchCount,
                rowsPerBatch = rowsPerBatch,
                inputColumns = inputColumns,
                outputColumns = outputColumns,
                transposeWeights = transposeWeights
            )
        } catch (error: LinkageError) {
            Log.e(TAG, "Vulkan batched-linear entry point is unavailable; disabling the backend.", error)
            false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Vulkan batched-linear dispatch failed; disabling the backend.", error)
            false
        }

        if (!updated) {
            disabledAfterRuntimeFailure = true
            _runtimeStatus.value = "Kotlin CPU fallback — Vulkan linear dispatch failed"
            _currentBackend = "Kotlin CPU"
            runCatching { destroyNativeRuntime() }
                .onFailure { error ->
                    Log.w(TAG, "Failed to release Vulkan after a batched-linear dispatch failure.", error)
                }
            Log.w(TAG, "Vulkan batched-linear dispatch was not completed; using the Kotlin CPU path.")
            return false
        }

        val outputBatchSize = rowsPerBatch * outputColumns
        for (batchIndex in 0 until batchCount) {
            val outputBatchOffset = batchIndex * outputBatchSize
            for (row in 0 until rowsPerBatch) {
                System.arraycopy(
                    flattenedOutput,
                    outputBatchOffset + row * outputColumns,
                    outputBatches[batchIndex][row],
                    0,
                    outputColumns
                )
            }
        }
        refreshGpuPerformanceMetrics()
        _currentBackend = "Vulkan Compute (Batched Linear)"
        return true
    }

    /**
     * Adapts binary16 model parameters to the FP32 Vulkan AdamW kernel.
     *
     * Gradient accumulation and AdamW state remain FP32. Only the active
     * parameter prefix is staged; successful results are quantized back to
     * binary16 once. Returning false leaves the complete update to the CPU path.
     */
    @Synchronized
    fun tryApplyFp16AdamWUpdate(
        parameters: Array<Float16ParameterArray>,
        gradients: Array<FloatArray>,
        firstMoments: Array<FloatArray>,
        secondMoments: Array<FloatArray>,
        weightDecayByGroup: FloatArray,
        learningRate: Float,
        beta1: Float,
        beta2: Float,
        biasCorrection1: Float,
        biasCorrection2: Float,
        epsilon: Float,
        gradientClipFactor: Float
    ): Boolean {
        require(parameters.isNotEmpty()) {
            "AdamW requires at least one parameter group."
        }
        require(
            parameters.size == gradients.size &&
                parameters.size == firstMoments.size &&
                parameters.size == secondMoments.size &&
                parameters.size == weightDecayByGroup.size
        ) {
            "AdamW tensor and weight-decay group counts must match."
        }
        require(learningRate.isFinite() && learningRate >= 0f) {
            "AdamW learning rate must be finite and non-negative."
        }
        require(beta1.isFinite() && beta1 >= 0f && beta1 < 1f &&
            beta2.isFinite() && beta2 >= 0f && beta2 < 1f) {
            "AdamW beta values must be finite and in [0, 1)."
        }
        require(biasCorrection1.isFinite() && biasCorrection1 > 0f &&
            biasCorrection2.isFinite() && biasCorrection2 > 0f) {
            "AdamW bias corrections must be finite and positive."
        }
        require(epsilon.isFinite() && epsilon > 0f) {
            "AdamW epsilon must be finite and positive."
        }
        require(gradientClipFactor.isFinite() &&
            gradientClipFactor >= 0f && gradientClipFactor <= 1f) {
            "AdamW gradient clipping factor must be in [0, 1]."
        }

        var totalElements = 0L
        var stagingElements = 0L
        for (index in parameters.indices) {
            val parameterSize = parameters[index].size
            val activeSize = gradients[index].size
            require(parameterSize > 0 && activeSize > 0 && activeSize <= parameterSize) {
                "AdamW active gradient prefix is invalid for parameter group $index."
            }
            require(
                firstMoments[index].size == parameterSize &&
                    secondMoments[index].size == parameterSize
            ) {
                "FP32 AdamW moments must match parameter capacity for group $index."
            }
            require(weightDecayByGroup[index].isFinite() &&
                weightDecayByGroup[index] >= 0f) {
                "AdamW weight decay must be finite and non-negative for group $index."
            }
            totalElements = Math.addExact(totalElements, activeSize.toLong())
            // Parameter, first-moment, and second-moment staging must all be
            // isolated from model state until every GPU result is validated.
            stagingElements = Math.addExact(
                stagingElements,
                Math.multiplyExact(activeSize.toLong(), 3L)
            )
        }

        if (!isRuntimeAvailable || disabledAfterRuntimeFailure ||
            totalElements < MIN_GPU_UPDATE_ELEMENTS) {
            _currentBackend = "Kotlin CPU"
            return false
        }

        val runtime = Runtime.getRuntime()
        val availableHeapBytes = runtime.freeMemory() +
            (runtime.maxMemory() - runtime.totalMemory()).coerceAtLeast(0L)
        val stagingBytes = Math.multiplyExact(stagingElements, Float.SIZE_BYTES.toLong())
        if (stagingBytes > availableHeapBytes / 2L) {
            Log.d(TAG, "Vulkan AdamW staging exceeds the safe available heap budget; retaining CPU fallback.")
            return false
        }

        return try {
            val stagedParameters = Array(parameters.size) { index ->
                FloatArray(gradients[index].size) { element ->
                    parameters[index][element]
                }
            }
            val stagedFirstMoments = Array(firstMoments.size) { index ->
                firstMoments[index].copyOf(gradients[index].size)
            }
            val stagedSecondMoments = Array(secondMoments.size) { index ->
                secondMoments[index].copyOf(gradients[index].size)
            }

            val updated = tryApplyAdamWUpdate(
                parameters = stagedParameters,
                gradients = gradients,
                firstMoments = stagedFirstMoments,
                secondMoments = stagedSecondMoments,
                weightDecayByGroup = weightDecayByGroup,
                learningRate = learningRate,
                beta1 = beta1,
                beta2 = beta2,
                biasCorrection1 = biasCorrection1,
                biasCorrection2 = biasCorrection2,
                epsilon = epsilon,
                gradientClipFactor = gradientClipFactor
            )
            if (!updated) return false

            // Validate the complete device result before committing FP16 weights.
            for (group in stagedParameters.indices) {
                require(stagedParameters[group].all(Float::isFinite)) {
                    "Vulkan AdamW produced non-finite parameters in group $group."
                }
                require(stagedFirstMoments[group].all(Float::isFinite) &&
                    stagedSecondMoments[group].all(Float::isFinite)) {
                    "Vulkan AdamW produced non-finite optimizer state in group $group."
                }
                for (value in stagedParameters[group]) {
                    Float16Codec.fromFiniteFloat(value)
                }
            }

            for (index in parameters.indices) {
                val activeSize = stagedParameters[index].size
                for (element in 0 until activeSize) {
                    parameters[index][element] = stagedParameters[index][element]
                }
                if (stagedFirstMoments[index] !== firstMoments[index]) {
                    System.arraycopy(
                        stagedFirstMoments[index], 0,
                        firstMoments[index], 0,
                        activeSize
                    )
                }
                if (stagedSecondMoments[index] !== secondMoments[index]) {
                    System.arraycopy(
                        stagedSecondMoments[index], 0,
                        secondMoments[index], 0,
                        activeSize
                    )
                }
            }
            true
        } catch (error: OutOfMemoryError) {
            Log.w(TAG, "Not enough heap memory to stage FP16 parameters for Vulkan AdamW; retaining CPU fallback.", error)
            false
        } catch (error: IllegalArgumentException) {
            Log.e(TAG, "Vulkan AdamW returned invalid staged tensors; retaining CPU fallback.", error)
            false
        }
    }

    @Synchronized
    fun tryApplyAdamWUpdate(
        parameters: Array<FloatArray>,
        gradients: Array<FloatArray>,
        firstMoments: Array<FloatArray>,
        secondMoments: Array<FloatArray>,
        weightDecayByGroup: FloatArray,
        learningRate: Float,
        beta1: Float,
        beta2: Float,
        biasCorrection1: Float,
        biasCorrection2: Float,
        epsilon: Float,
        gradientClipFactor: Float
    ): Boolean {
        require(parameters.isNotEmpty()) { "AdamW requires at least one parameter group." }
        require(parameters.size == gradients.size) {
            "Parameter and gradient group counts must match."
        }
        require(parameters.size == firstMoments.size) {
            "Parameter and first-moment group counts must match."
        }
        require(parameters.size == secondMoments.size) {
            "Parameter and second-moment group counts must match."
        }
        require(parameters.size == weightDecayByGroup.size) {
            "Weight-decay configuration must match the parameter groups."
        }

        var totalElements = 0L
        for (index in parameters.indices) {
            val groupSize = parameters[index].size
            require(groupSize > 0) {
                "AdamW parameter groups must not be empty."
            }
            require(
                gradients[index].size == groupSize &&
                    firstMoments[index].size == groupSize &&
                    secondMoments[index].size == groupSize
            ) {
                "AdamW tensor sizes must match exactly for parameter group $index."
            }
            totalElements = Math.addExact(totalElements, groupSize.toLong())
        }

        if (!isRuntimeAvailable ||
            disabledAfterRuntimeFailure ||
            totalElements < MIN_GPU_UPDATE_ELEMENTS) {
            _currentBackend = "Kotlin CPU"
            return false
        }

        val updated = try {
            nativeApplyAdamWUpdate(
                parameterGroups = parameters,
                gradientGroups = gradients,
                firstMomentGroups = firstMoments,
                secondMomentGroups = secondMoments,
                weightDecayByGroup = weightDecayByGroup,
                learningRate = learningRate,
                beta1 = beta1,
                beta2 = beta2,
                biasCorrection1 = biasCorrection1,
                biasCorrection2 = biasCorrection2,
                epsilon = epsilon,
                gradientClipFactor = gradientClipFactor
            )
        } catch (error: LinkageError) {
            Log.e(TAG, "Vulkan optimizer entry point is unavailable; disabling the backend.", error)
            false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Vulkan optimizer dispatch failed; disabling the backend.", error)
            false
        }

        if (updated) {
            refreshGpuPerformanceMetrics()
            _currentBackend = "Vulkan Compute (AdamW)"
            return true
        }

        disabledAfterRuntimeFailure = true
        _runtimeStatus.value = "Kotlin CPU fallback — Vulkan optimizer dispatch failed"
        _currentBackend = "Kotlin CPU"
        runCatching { destroyNativeRuntime() }
            .onFailure { error ->
                Log.w(TAG, "Failed to release the Vulkan runtime after a dispatch failure.", error)
            }
        Log.w(TAG, "Vulkan AdamW dispatch was not completed; using the Kotlin CPU optimizer.")
        return false
    }

    private external fun nativeInitialize(): Boolean

    private external fun nativeGetDriverInfo(): String

    private external fun nativeGetDeviceCapabilities(): String

    private external fun nativeGetGpuPerformanceMetrics(): String

    private external fun nativeSupportsFp16(): Boolean

    private external fun nativeApplyFp16Linear(
        input: FloatArray,
        weights: ShortArray,
        output: FloatArray,
        rows: Int,
        batchCount: Int,
        rowsPerBatch: Int,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean
    ): Boolean

    private external fun nativeApplyFp16LinearWithArena(
        input: FloatArray,
        weightHandles: LongArray,
        output: FloatArray,
        rows: Int,
        batchCount: Int,
        rowsPerBatch: Int,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean
    ): Boolean

    private external fun nativeApplyResidentLinear(
        inputHandle: Long,
        weightHandle: Long,
        outputHandle: Long,
        rows: Int,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean
    ): Boolean

    private external fun nativeApplyResidentLinearBatch(
        inputHandle: Long,
        weightHandles: LongArray,
        outputHandle: Long,
        rowsPerBatch: Int,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean
    ): Boolean

    private external fun nativeRunResidentAttention(
        qkvHandle: Long,
        scoreHandle: Long,
        attentionOutputHandle: Long,
        projectedOutputHandle: Long,
        normalizedOutputHandle: Long,
        residualHandle: Long,
        outputWeightHandle: Long,
        sequenceLength: Int,
        modelWidth: Int,
        headCount: Int,
        headDimension: Int
    ): Boolean

    private external fun nativeApplyLinear(
        input: FloatArray,
        weights: FloatArray,
        output: FloatArray,
        rows: Int,
        batchCount: Int,
        rowsPerBatch: Int,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean
    ): Boolean

    private external fun nativeApplyAdamWUpdate(
        parameterGroups: Array<FloatArray>,
        gradientGroups: Array<FloatArray>,
        firstMomentGroups: Array<FloatArray>,
        secondMomentGroups: Array<FloatArray>,
        weightDecayByGroup: FloatArray,
        learningRate: Float,
        beta1: Float,
        beta2: Float,
        biasCorrection1: Float,
        biasCorrection2: Float,
        epsilon: Float,
        gradientClipFactor: Float
    ): Boolean

    private external fun nativeAllocateTensor(byteCount: Long): Long

    private external fun nativeUploadTensor(tensorHandle: Long, source: ByteArray): Boolean

    private external fun nativeDownloadTensor(
        tensorHandle: Long,
        destination: ByteArray
    ): Boolean

    private external fun nativeReleaseTensor(tensorHandle: Long): Boolean

    private external fun nativeGetTensorArenaMetrics(): String

    private external fun nativeDestroy()

    private external fun nativeClearTensorArena(): Boolean
}
