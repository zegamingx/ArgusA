package com.argus.agi.model

import android.util.Log
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


internal data class VulkanTensorAllocation internal constructor(
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

    private fun releaseNativeRuntimeAfterFailure(nativeLibraryLoaded: Boolean) {
        nativeReady = false
        fp16RuntimeReady = false
        if (nativeLibraryLoaded) {
            runCatching { nativeDestroy() }
                .onFailure { error ->
                    Log.w(TAG, "Failed to release a partially initialized Vulkan runtime.", error)
                }
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
        // Cover partial 8x16 tiles, a second K tile, multiple independent
        // batches, odd packed-input and weight lengths, and both weight layouts.
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

        fun verify(weightValues: ShortArray, transposeWeights: Boolean): Boolean {
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

        return verify(weights, transposeWeights = false) &&
            verify(transposedWeights, transposeWeights = true)
    }

    @Synchronized
    fun tryApplyFp16Linear(
        inputRows: Array<FloatArray>,
        weights: Float16ParameterArray,
        outputRows: Array<FloatArray>,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean = false
    ): Boolean = tryApplyFp16BatchedLinear(
        inputBatches = arrayOf(inputRows),
        weightBatches = arrayOf(weights.toShortArray()),
        outputBatches = arrayOf(outputRows),
        inputColumns = inputColumns,
        outputColumns = outputColumns,
        transposeWeights = transposeWeights
    )

    @Synchronized
    fun tryApplyFp16BatchedLinear(
        inputBatches: Array<Array<FloatArray>>,
        weightBatches: Array<ShortArray>,
        outputBatches: Array<Array<FloatArray>>,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean = false
    ): Boolean {
        require(inputBatches.isNotEmpty()) { "FP16 batched linear input must contain at least one batch." }
        require(inputColumns > 0 && outputColumns > 0) {
            "FP16 batched linear dimensions must be positive."
        }
        val batchCount = inputBatches.size
        require(weightBatches.size == batchCount && outputBatches.size == batchCount) {
            "FP16 batched linear inputs, weights, and outputs must have the same batch count."
        }
        val rowsPerBatch = inputBatches[0].size
        require(rowsPerBatch > 0) { "Every FP16 batched linear input must contain at least one row." }
        val requiredWeightElementsLong =
            inputColumns.toLong() * outputColumns.toLong()
        if (requiredWeightElementsLong > Int.MAX_VALUE.toLong()) return false
        val requiredWeightElements = requiredWeightElementsLong.toInt()

        for (batchIndex in 0 until batchCount) {
            require(inputBatches[batchIndex].size == rowsPerBatch) {
                "FP16 batched linear inputs must have an identical row count."
            }
            require(outputBatches[batchIndex].size == rowsPerBatch) {
                "FP16 batched linear outputs must match the input row count."
            }
            // Some model tensors are allocated to their maximum capacity while a
            // training pass only uses an active prefix (for example, wHead when the
            // tokenizer vocabulary has not filled its configured capacity).
            // The staging copy below intentionally copies only this active matrix.
            if (weightBatches[batchIndex].size < requiredWeightElements) {
                return false
            }
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

        val inputElementsLong = totalRowsLong * inputColumns.toLong()
        val weightElementsLong = batchCount.toLong() * requiredWeightElements.toLong()
        val outputElementsLong = totalRowsLong * outputColumns.toLong()
        if (inputElementsLong > Int.MAX_VALUE.toLong() ||
            weightElementsLong > Int.MAX_VALUE.toLong() ||
            outputElementsLong > Int.MAX_VALUE.toLong()) {
            return false
        }

        val operationCount = totalRowsLong *
            inputColumns.toLong() *
            outputColumns.toLong()
        if (!isFp16ComputeAvailable ||
            operationCount < MIN_LINEAR_OPERATION_COUNT) {
            return false
        }

        val flattenedInput: FloatArray
        val flattenedWeights: ShortArray
        val flattenedOutput: FloatArray
        try {
            flattenedInput = FloatArray(inputElementsLong.toInt())
            flattenedWeights = ShortArray(weightElementsLong.toInt())
            flattenedOutput = FloatArray(outputElementsLong.toInt())
        } catch (error: OutOfMemoryError) {
            Log.w(TAG, "Not enough heap memory to stage a Vulkan FP16 linear operation.", error)
            return false
        }

        val inputBatchSize = rowsPerBatch * inputColumns
        val weightBatchSize = requiredWeightElements
        for (batchIndex in 0 until batchCount) {
            val inputOffset = batchIndex * inputBatchSize
            for (row in 0 until rowsPerBatch) {
                System.arraycopy(
                    inputBatches[batchIndex][row],
                    0,
                    flattenedInput,
                    inputOffset + row * inputColumns,
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
            nativeApplyFp16Linear(
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
            Log.e(TAG, "Vulkan FP16 linear entry point is unavailable; disabling FP16 dispatch.", error)
            false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Vulkan FP16 linear dispatch failed; disabling FP16 dispatch.", error)
            false
        }

        if (!updated) {
            fp16DisabledAfterRuntimeFailure = true
            fp16RuntimeReady = false
            Log.w(TAG, "Vulkan FP16 dispatch was not completed; the verified FP32 Vulkan/CPU path remains available.")
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
        _currentBackend = "Vulkan Compute (FP16 multiply / FP32 accumulation)"
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
            runCatching { nativeDestroy() }
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
        runCatching { nativeDestroy() }
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
}
