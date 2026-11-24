package com.argus.agi.model

import com.argus.agi.backend.ComputeBackend
import com.argus.agi.backend.ComputeBackendKind
import com.argus.agi.device.DeviceProfile
import com.argus.agi.runtime.AdaptiveComputeOperation
import com.argus.agi.runtime.DeviceAdaptiveRuntimePolicy
import com.argus.agi.runtime.RuntimeConfigurationPlanner
import com.argus.agi.runtime.AdaptiveComputePerformanceMonitor

import android.util.Log
import com.argus.agi.diagnostics.ArgusDiagnosticBridge
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

internal class VulkanResidentGradientAccumulator internal constructor(
    internal var active: VulkanTensorAllocation,
    internal var spare: VulkanTensorAllocation,
    internal val staging: VulkanTensorAllocation,
    val elementCount: Int,
    internal val tensorSizes: IntArray
) {
    internal val pendingReleaseAllocations = ArrayList<VulkanTensorAllocation>()
    internal var activeReleased: Boolean = false
    internal var spareReleased: Boolean = false
    internal var stagingReleased: Boolean = false
    internal var released: Boolean = false
}

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
    val memoryHeapUsageBytes: List<Long>,
    val memoryHeapFlags: List<Int> = emptyList()
) {
    val apiVersion: String
        get() = "$apiVersionMajor.$apiVersionMinor.$apiVersionPatch"

    val memoryBudgetHeadroomBytes: List<Long>
        get() = memoryHeapBudgetBytes.zip(memoryHeapUsageBytes) { budget, usage ->
            (budget - usage).coerceAtLeast(0L)
        }

    /**
     * Headroom for device-local heaps only. Older drivers may not expose heap flags;
     * in that case retain the conservative all-heap calculation for compatibility.
     */
    val deviceLocalMemoryBudgetHeadroomBytes: List<Long>
        get() = DeviceAdaptiveRuntimePolicy.deviceLocalHeapHeadroomBytes(
            budgets = memoryHeapBudgetBytes,
            usages = memoryHeapUsageBytes,
            heapFlags = memoryHeapFlags
        )

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

        /**
         * Reads an unsigned 32-bit Vulkan limit/identifier without signed
         * truncation. org.json tokenizes values above Int.MAX_VALUE as Long and
         * optInt would truncate them (UINT32_MAX became -1, which the core-field
         * validation below then rejected, breaking capability parsing entirely
         * on devices such as Mali-G57 that report UINT32_MAX work-group counts).
         * Clamping to Int.MAX_VALUE preserves the meaning of a numeric limit:
         * effectively unlimited. (Opaque identifiers such as deviceId are not
         * read through this helper; their bits are preserved as-is.)
         */
        private fun org.json.JSONObject.optUint32Clamped(name: String): Int =
            optLong(name).coerceIn(0L, Int.MAX_VALUE.toLong()).toInt()

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

            val capabilities = VulkanDeviceCapabilities(
                deviceName = json.optString("deviceName", "Unknown Vulkan GPU"),
                vendorId = json.optInt("vendorId"),
                // deviceId is an opaque identifier whose bits are the identity
                // (rendered as hex via toUInt); preserve the low 32 bits exactly
                // instead of clamping like the numeric limits below.
                deviceId = json.optLong("deviceId").toInt(),
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
                maxComputeWorkGroupCountX = json.optUint32Clamped("maxComputeWorkGroupCountX"),
                maxComputeWorkGroupCountY = json.optUint32Clamped("maxComputeWorkGroupCountY"),
                maxComputeWorkGroupCountZ = json.optUint32Clamped("maxComputeWorkGroupCountZ"),
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
                } ?: emptyList(),
                memoryHeapFlags = json.optJSONArray("memoryHeapFlags")?.let { array ->
                    List(array.length()) { index -> array.optInt(index) }
                } ?: emptyList()
            )

            // A partial payload must fail instead of zero-filling a plausible-looking
            // capability object: downstream code treats a zero limit as "no limit", so
            // silently accepting missing core identity/limit fields is undiagnosable.
            // Optional feature/extension fields keep their opt* defaults. Both call
            // sites parse inside runCatching, so throwing here surfaces as "capability
            // metadata unavailable" and retains the last verified state.
            val missingCoreFields = buildList {
                if (!json.has("deviceName") || capabilities.deviceName.isBlank()) {
                    add("deviceName")
                }
                if (!json.has("apiVersionMajor") || !json.has("apiVersionMinor") ||
                    !json.has("apiVersionPatch") || capabilities.apiVersionMajor < 1
                ) {
                    add("apiVersion")
                }
                if (capabilities.maxComputeWorkGroupInvocations <= 0) {
                    add("maxComputeWorkGroupInvocations")
                }
                if (capabilities.maxComputeWorkGroupSizeX <= 0 ||
                    capabilities.maxComputeWorkGroupSizeY <= 0 ||
                    capabilities.maxComputeWorkGroupSizeZ <= 0
                ) {
                    add("maxComputeWorkGroupSize")
                }
                if (capabilities.maxComputeWorkGroupCountX <= 0 ||
                    capabilities.maxComputeWorkGroupCountY <= 0 ||
                    capabilities.maxComputeWorkGroupCountZ <= 0
                ) {
                    add("maxComputeWorkGroupCount")
                }
                if (capabilities.maxComputeSharedMemoryBytes <= 0L) {
                    add("maxComputeSharedMemoryBytes")
                }
                if (capabilities.maxStorageBufferRangeBytes <= 0L) {
                    add("maxStorageBufferRangeBytes")
                }
            }
            require(missingCoreFields.isEmpty()) {
                "Vulkan device capability payload is missing or has invalid core fields: " +
                    missingCoreFields.joinToString(", ")
            }
            return capabilities
        }
    }
}

/**
 * Optional Vulkan Compute backend for batched dense operations and AdamW updates.
 *
 * Dense forward and backward operations use Vulkan whenever runtime readiness,
 * enabled device capabilities, kernel availability, and resource checks permit it.
 * Kotlin execution is reserved for unavailable, unsupported, or failed Vulkan paths.
 */
internal object VulkanTrainingBackend : ComputeBackend {

    private const val TAG = "VulkanTrainingBackend"

    override val kind: ComputeBackendKind = ComputeBackendKind.VULKAN
    private const val MEMORY_BUDGET_REFRESH_INTERVAL_NANOS = 5_000_000_000L

    internal fun recordCpuLinearTiming(operationCount: Long, elapsedNanoseconds: Long) {
        AdaptiveComputePerformanceMonitor.recordCpu(
            AdaptiveComputeOperation.LINEAR,
            operationCount,
            elapsedNanoseconds
        )
    }

    internal fun recordCpuAdamWTiming(operationCount: Long, elapsedNanoseconds: Long) {
        AdaptiveComputePerformanceMonitor.recordCpu(
            AdaptiveComputeOperation.ADAMW,
            operationCount,
            elapsedNanoseconds
        )
    }

    internal fun recordGpuAdamWTiming(operationCount: Long, elapsedNanoseconds: Long) {
        AdaptiveComputePerformanceMonitor.recordGpu(
            AdaptiveComputeOperation.ADAMW,
            operationCount,
            elapsedNanoseconds
        )
    }

    internal const val BACKWARD_CROSS_ENTROPY = 0
    internal const val BACKWARD_SWIGLU = 1
    internal const val BACKWARD_LAYER_NORM = 2
    internal const val BACKWARD_CAUSAL_SOFTMAX = 3
    internal const val BACKWARD_EMBEDDING_GRADIENT = 4
    internal const val BACKWARD_GRADIENT_ACCUMULATE = 5
    internal const val BACKWARD_GRADIENT_SCALE = 6
    internal const val BACKWARD_GRADIENT_NORM_PARTIAL = 7
    internal const val BACKWARD_GRADIENT_NORM_FINAL = 8
    internal const val BACKWARD_GRADIENT_NORMALIZE_CLIP = 10

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

    private val _deviceProfile = MutableStateFlow<DeviceProfile?>(null)
    override val deviceProfile: StateFlow<DeviceProfile?> = _deviceProfile.asStateFlow()

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

    private val _optimizerStateResidencyBudgetBytes = MutableStateFlow(0L)

    private val tensorResidencyRegistry = VulkanTensorResidencyRegistry(
        allocate = { byteCount -> allocateGpuTensor(byteCount) },
        upload = { allocation, bytes -> uploadGpuTensor(allocation, bytes) },
        release = { allocation -> releaseGpuTensor(allocation) },
        download = { allocation, bytes -> downloadGpuTensor(allocation, bytes) },
        maxResidentBytes = { _modelTensorResidencyBudgetBytes.value }
    )

    private val optimizerStateResidencyRegistry = VulkanOptimizerStateRegistry(
        allocate = { byteCount -> allocateGpuTensor(byteCount) },
        upload = { allocation, bytes -> uploadGpuTensor(allocation, bytes) },
        release = { allocation -> releaseGpuTensor(allocation) },
        download = { allocation, bytes -> downloadGpuTensor(allocation, bytes) },
        maxResidentBytes = { _optimizerStateResidencyBudgetBytes.value }
    )

    private val _modelTensorResidencyReady = MutableStateFlow(false)
    val modelTensorResidencyReady: StateFlow<Boolean> =
        _modelTensorResidencyReady.asStateFlow()

    @Volatile
    private var residentActivationLinearReady = false
    @Volatile
    private var residentAttentionReady = false
    @Volatile
    private var residentSwiGluReady = false
    @Volatile
    private var residentResidualLayerNormReady = false

    @Volatile
    private var residentBackwardReady = false
    @Volatile
    private var residentGradientAccumulatorReady = false

    @Volatile
    private var residentAdamwReady = false

    @Volatile
    private var lastMemoryBudgetRefreshNanos = 0L

    internal val isResidentAdamWAvailable: Boolean
        get() = isRuntimeAvailable && residentActivationLinearReady && residentAdamwReady
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
        },
        dispatchSwiGlu = { gate, up, output, rows, columns ->
            dispatchResidentSwiGlu(gate, up, output, rows, columns)
        },
        dispatchResidualLayerNorm = { down, residual, projected, normalized, rows, columns ->
            dispatchResidentResidualLayerNorm(
                down, residual, projected, normalized, rows, columns
            )
        }
    )
    internal val residentActivations: VulkanActivationTensorRegistry
        get() = activationTensorRegistry

    private var lastModelParameterInventory: List<VulkanModelParameterTensor> = emptyList()

    /**
     * Live resident gradient accumulators. Their arena allocations are owned by no
     * residency registry, so they are tracked here to let residency invalidation and
     * runtime teardown release or retire them together with the three registries.
     * Every access happens under the backend monitor (all accumulator entry points and
     * every invalidateTensorResidency call site are @Synchronized on this singleton).
     */
    private val liveResidentGradientAccumulators =
        mutableSetOf<VulkanResidentGradientAccumulator>()

    private var residencySuspended = false
    private var residencyBlockedAfterFailure = false
    private var residencyRebuildPending = false
    private var runtimeRestartPending = false
    private var residentGradientRecoveryPending = false
    private var modelParameterMirrorReliable = true
    private var optimizerStateMirrorReliable = true

    private fun invalidateTensorResidency(releaseAllocations: Boolean): Boolean {
        if (releaseAllocations && nativeReady && !synchronizeTrainingStateMirrors()) {
            ArgusDiagnosticBridge.error(TAG,
                "Tensor Arena release was refused because model or optimizer state could not be recovered."
            )
            return false
        }
        _modelTensorResidencyReady.value = false
        val modelReleased = tensorResidencyRegistry.invalidate(releaseAllocations)
        val activationsReleased = activationTensorRegistry.invalidate(releaseAllocations)
        val optimizerStatesReleased =
            optimizerStateResidencyRegistry.invalidate(releaseAllocations)
        // The resident gradient accumulator holds arena allocations outside every
        // registry, so it must be discarded with them. This point is only reached after
        // the mirror-synchronization guard above has verified host recovery of model
        // and optimizer state; the accumulator itself is gradient scratch for the
        // current accumulation window, is never mirrored or checkpointed separately,
        // and any caller still holding it observes a clean `released` failure on its
        // next accumulate/finalize/download call instead of dispatching on arena
        // handles that invalidation has already freed.
        val gradientsReleased =
            discardTrackedResidentGradientAccumulators(releaseAllocations)
        val released = modelReleased && activationsReleased && optimizerStatesReleased &&
            gradientsReleased
        if (!released && releaseAllocations && nativeReady) {
            val arenaCleared = runCatching { nativeClearTensorArena() }.getOrDefault(false)
            if (arenaCleared) {
                tensorResidencyRegistry.invalidate(releaseAllocations = false)
                activationTensorRegistry.invalidate(releaseAllocations = false)
                optimizerStateResidencyRegistry.invalidate(releaseAllocations = false)
                // The arena clear freed every remaining accumulator allocation.
                discardTrackedResidentGradientAccumulators(releaseAllocations = false)
                refreshTensorArenaMetrics()
                return true
            }

            disabledAfterRuntimeFailure = true
            _currentBackend = "Kotlin CPU"
            _runtimeStatus.value =
                "Kotlin CPU fallback — Vulkan tensor-arena release failed"
            runCatching { destroyNativeRuntime() }
                .onFailure { error ->
                    ArgusDiagnosticBridge.error(TAG, "Vulkan runtime teardown after an arena-release failure failed.", error)
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

    override val isReady: Boolean
        get() = isRuntimeAvailable

    val isFp16ComputeAvailable: Boolean
        get() = isRuntimeAvailable && fp16RuntimeReady && !fp16DisabledAfterRuntimeFailure

    override val isFp16ComputeReady: Boolean
        get() = isFp16ComputeAvailable

    internal val isResidentActivationLinearAvailable: Boolean
        get() = isRuntimeAvailable && residentActivationLinearReady

    internal val isResidentAttentionAvailable: Boolean
        get() = isRuntimeAvailable && residentActivationLinearReady && residentAttentionReady

    internal val isResidentSwiGluAvailable: Boolean
        get() = isRuntimeAvailable && residentSwiGluReady

    internal val isResidentFeedForwardAvailable: Boolean
        get() = isRuntimeAvailable &&
            residentActivationLinearReady &&
            residentSwiGluReady &&
            residentResidualLayerNormReady

    internal val isResidentBackwardAvailable: Boolean
        get() = isRuntimeAvailable && residentBackwardReady

    internal val isResidentGradientAccumulatorAvailable: Boolean
        get() = isResidentBackwardAvailable && residentGradientAccumulatorReady

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

    private fun destroyNativeRuntime(): Boolean {
        if (nativeReady && !synchronizeTrainingStateMirrors()) {
            disabledAfterRuntimeFailure = true
            _currentBackend = "Vulkan disabled — GPU training state recovery pending"
            _runtimeStatus.value =
                "GPU model/optimizer state could not be read back; Vulkan memory is retained for recovery"
            publishInitializationProgress(
                status = "GPU state recovery required",
                driverInfo = _initializationProgress.value.driverInfo,
                isInitializing = false,
                isCpuFallback = false
            )
            ArgusDiagnosticBridge.error(TAG,
                "Vulkan teardown was postponed because model or optimizer state could not be recovered."
            )
            return false
        }
        return try {
            nativeDestroy()
            nativeReady = false
            fp16RuntimeReady = false
            residentActivationLinearReady = false
            residentAttentionReady = false
            residentSwiGluReady = false
            residentResidualLayerNormReady = false
            residentBackwardReady = false
            residentGradientAccumulatorReady = false
            residentAdamwReady = false
            disabledAfterRuntimeFailure = true
            _currentBackend = "Kotlin CPU"
            _runtimeStatus.value = "Kotlin CPU fallback — Vulkan runtime is no longer available"
            publishInitializationProgress(
                status = "Vulkan unavailable — CPU fallback",
                driverInfo = _initializationProgress.value.driverInfo,
                isInitializing = false,
                isCpuFallback = true
            )
            invalidateTensorResidency(releaseAllocations = false)
            _tensorArenaMetrics.value = VulkanTensorArenaMetrics()
            true
        } catch (error: RuntimeException) {
            disabledAfterRuntimeFailure = true
            _currentBackend = "Kotlin CPU — Vulkan teardown failed after state recovery"
            _runtimeStatus.value = "Vulkan teardown failed; recovered host state is retained"
            ArgusDiagnosticBridge.error(TAG, "Vulkan native runtime destruction failed.", error)
            false
        } catch (error: LinkageError) {
            disabledAfterRuntimeFailure = true
            _currentBackend = "Kotlin CPU — Vulkan teardown entry point unavailable"
            _runtimeStatus.value = "Vulkan teardown failed; recovered host state is retained"
            ArgusDiagnosticBridge.error(TAG, "Vulkan native runtime destruction entry point is unavailable.", error)
            false
        }
    }

    private fun releaseNativeRuntimeAfterFailure(nativeLibraryLoaded: Boolean) {
        nativeReady = false
        fp16RuntimeReady = false
        if (nativeLibraryLoaded) {
            runCatching { destroyNativeRuntime() }
                .onFailure { error ->
                    ArgusDiagnosticBridge.warning(TAG, "Failed to release a partially initialized Vulkan runtime.", error)
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
        deviceProfile: DeviceProfile?
    ): Long {
        val runtime = Runtime.getRuntime()
        val maximumAppHeapBytes = runtime.maxMemory().coerceAtLeast(0L)
        val availableAppHeapBytes = (
            runtime.freeMemory() +
                (maximumAppHeapBytes - runtime.totalMemory()).coerceAtLeast(0L)
            ).coerceAtLeast(0L)
        return RuntimeConfigurationPlanner.calculateTensorArenaBudget(
            deviceProfile = deviceProfile,
            maximumAppHeapBytes = maximumAppHeapBytes,
            availableAppHeapBytes = availableAppHeapBytes
        ).totalBytes
    }

    /**
     * Atomic AdamW replacement reserves active and candidate allocations:
     * FP16 weights 2 + candidate weights 2, active/candidate FP32 moments 16,
     * and a transient FP32 gradient 4 bytes per parameter. The bounded arena is
     * split into 1/12 active model, 1/3 active optimizer state, and 1/6 activation/
     * gradient capacity; the remaining equal shares hold both candidate sets.
     */
    private fun calculateModelTensorResidencyBudgetBytes(
        deviceProfile: DeviceProfile?
    ): Long = calculateTotalTensorResidencyBudgetBytes(deviceProfile) / 12L

    private fun calculateActivationTensorResidencyBudgetBytes(
        deviceProfile: DeviceProfile?
    ): Long = calculateTotalTensorResidencyBudgetBytes(deviceProfile) / 6L

    private fun calculateOptimizerStateResidencyBudgetBytes(
        deviceProfile: DeviceProfile?
    ): Long = calculateTotalTensorResidencyBudgetBytes(deviceProfile) / 3L

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
            ArgusDiagnosticBridge.error(TAG, "Vulkan tensor allocation entry point is unavailable.", error)
            return null
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Vulkan tensor allocation failed.", error)
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
            ArgusDiagnosticBridge.error(TAG, "Vulkan tensor upload entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Vulkan tensor upload failed.", error)
            false
        }
        if (uploaded) refreshTensorArenaMetrics()
        return uploaded
    }

    @Synchronized
    fun downloadGpuTensor(allocation: VulkanTensorAllocation, destination: ByteArray): Boolean {
        if (!nativeReady || destination.size.toLong() != allocation.byteCount) return false
        val downloaded = try {
            nativeDownloadTensor(allocation.nativeHandle, destination)
        } catch (error: LinkageError) {
            ArgusDiagnosticBridge.error(TAG, "Vulkan tensor readback entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Vulkan tensor readback failed.", error)
            false
        }
        if (downloaded) refreshTensorArenaMetrics()
        return downloaded
    }

    @Synchronized
    fun releaseGpuTensor(allocation: VulkanTensorAllocation): Boolean {
        if (!nativeReady) return false
        val released = try {
            nativeReleaseTensor(allocation.nativeHandle)
        } catch (error: LinkageError) {
            ArgusDiagnosticBridge.error(TAG, "Vulkan tensor release entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Vulkan tensor release failed.", error)
            false
        }
        if (released) refreshTensorArenaMetrics()
        return released
    }

    /** Synchronizes one host parameter only when its resident GPU copy is newer. */
    @Synchronized
    internal fun synchronizeModelParameterTensor(
        parameters: Float16ParameterArray
    ): Boolean {
        if (modelParameterMirrorReliable) return true
        if (!nativeReady) return false
        return try {
            val synchronized = tensorResidencyRegistry.synchronizeHostMirror(parameters)
            if (synchronized && tensorResidencyRegistry.areHostMirrorsCurrent()) {
                modelParameterMirrorReliable = true
            }
            synchronized
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Model-parameter recovery readback exceeded the available heap.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Model-parameter recovery readback failed.", error)
            false
        }
    }

    /** Synchronizes only the host tensors required by a CPU fallback operation. */
    @Synchronized
    internal fun synchronizeModelParameterTensors(
        parameters: List<Float16ParameterArray>
    ): Boolean {
        if (parameters.isEmpty() || modelParameterMirrorReliable) return true
        if (!nativeReady) return false
        return try {
            for (parameter in parameters) {
                if (!tensorResidencyRegistry.synchronizeHostMirror(parameter)) return false
            }
            if (tensorResidencyRegistry.areHostMirrorsCurrent()) {
                modelParameterMirrorReliable = true
            }
            true
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Selected model-parameter readback exceeded the available heap.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Selected model-parameter readback failed.", error)
            false
        }
    }

    /** Synchronizes the complete model only for inference, checkpointing, or runtime recovery. */
    @Synchronized
    internal fun synchronizeModelParameterMirrors(): Boolean {
        if (modelParameterMirrorReliable) return true
        if (!nativeReady) return false
        return try {
            val synchronized = tensorResidencyRegistry.synchronizeHostMirrors()
            if (synchronized) modelParameterMirrorReliable = true
            synchronized
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Model-parameter recovery readback exceeded the available heap.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Model-parameter recovery readback failed.", error)
            false
        }
    }

    /** Synchronizes persistent FP32 moments into the checkpoint/CPU recovery mirror. */
    @Synchronized
    internal fun synchronizeOptimizerStateMirrors(): Boolean {
        if (optimizerStateMirrorReliable) return true
        if (!nativeReady) return false
        return try {
            val synchronized = optimizerStateResidencyRegistry.synchronizeHostMirrors()
            if (synchronized) optimizerStateMirrorReliable = true
            synchronized
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Optimizer-state recovery readback exceeded the available heap.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Optimizer-state recovery readback failed.", error)
            false
        }
    }

    /** Both host mirrors must be current before CPU fallback, checkpointing, or teardown. */
    @Synchronized
    internal fun synchronizeTrainingStateMirrors(): Boolean =
        synchronizeModelParameterMirrors() && synchronizeOptimizerStateMirrors()

    override fun synchronizeAuthoritativeState(): Boolean =
        synchronizeTrainingStateMirrors()

    override fun releaseDeviceResourcesForMemoryPressure(): Boolean =
        releaseModelTensorResidencyForMemoryPressure()

    override fun resumeDeviceResourcesAfterForeground() =
        resumeModelTensorResidencyAfterForeground()

    /** Drops optimizer replicas only after the exact GPU moments have been recovered. */
    @Synchronized
    internal fun invalidateOptimizerStateResidency(): Boolean {
        if (nativeReady && !synchronizeTrainingStateMirrors()) {
            ArgusDiagnosticBridge.error(TAG, "Cannot invalidate GPU optimizer state because host recovery was not verified.")
            return false
        }
        if (!modelParameterMirrorReliable || !optimizerStateMirrorReliable) {
            ArgusDiagnosticBridge.error(TAG, "CPU fallback is blocked because a host training-state mirror is stale.")
            return false
        }
        val released = optimizerStateResidencyRegistry.invalidate(
            releaseAllocations = nativeReady
        )
        if (!released && nativeReady) {
            ArgusDiagnosticBridge.warning(TAG, "Optimizer-state release failed; rebuilding the optional Vulkan runtime.")
            runCatching { destroyNativeRuntime() }
                .onFailure { error ->
                    ArgusDiagnosticBridge.error(TAG, "Vulkan teardown after optimizer-state release failure failed.", error)
                }
            return false
        }
        return true
    }

    /**
     * Prepares the complete model parameter inventory in persistent Tensor Arena
     * allocations. The registry remains a read-only mirror of Kotlin-owned weights.
     */
    private fun refreshModelTensorResidencyBudget(force: Boolean = false) {
        if (!nativeReady) return
        val nowNanos = System.nanoTime()
        if (!force &&
            nowNanos - lastMemoryBudgetRefreshNanos < MEMORY_BUDGET_REFRESH_INTERVAL_NANOS) {
            return
        }
        runCatching {
            VulkanDeviceCapabilities.fromJson(nativeGetDeviceCapabilities())
        }.onSuccess { capabilities ->
            val detectedProfile = capabilities.toDeviceProfile()
            _deviceCapabilities.value = capabilities
            _deviceProfile.value = detectedProfile
            _modelTensorResidencyBudgetBytes.value =
                calculateModelTensorResidencyBudgetBytes(detectedProfile)
            _activationTensorResidencyBudgetBytes.value =
                calculateActivationTensorResidencyBudgetBytes(detectedProfile)
            _optimizerStateResidencyBudgetBytes.value =
                calculateOptimizerStateResidencyBudgetBytes(detectedProfile)
            lastMemoryBudgetRefreshNanos = nowNanos
            Log.i(
                TAG,
                "Refreshed live Vulkan residency budgets: model=" +
                    "${_modelTensorResidencyBudgetBytes.value}, activation=" +
                    "${_activationTensorResidencyBudgetBytes.value}, optimizer=" +
                    "${_optimizerStateResidencyBudgetBytes.value} bytes."
            )
        }.onFailure { error ->
            // Stamp the attempt even on failure: a persistent parse/refresh
            // failure must not retry (and log) on every training step. The
            // throttle interval becomes the failure backoff.
            lastMemoryBudgetRefreshNanos = nowNanos
            ArgusDiagnosticBridge.warning(TAG,
                "Could not refresh Vulkan memory-budget telemetry; retaining the last verified budget.",
                error
            )
        }
    }

    @Synchronized
    fun synchronizeModelTensorResidency(
        tensors: List<VulkanModelParameterTensor>
    ): Boolean {
        refreshModelTensorResidencyBudget()
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
            ArgusDiagnosticBridge.warning(TAG, "Full model tensor residency exceeded available memory.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Full model tensor residency synchronization failed.", error)
            false
        }

        if (!synchronized) {
            invalidateTensorResidency(releaseAllocations = true)
            residencyBlockedAfterFailure = true
            ArgusDiagnosticBridge.warning(TAG,
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
        val wasResidencyReady = _modelTensorResidencyReady.value
        residencySuspended = true
        residencyRebuildPending = false
        _modelTensorResidencyReady.value = false

        if (!isRuntimeAvailable) {
            // A runtime may be disabled for dispatch while native handles are still alive
            // because an earlier GPU readback/teardown was postponed. Never drop those
            // handles until the host mirrors are verified or the entire arena is cleared.
            if (nativeReady) {
                if (!synchronizeTrainingStateMirrors()) {
                    residencySuspended = false
                    _modelTensorResidencyReady.value = wasResidencyReady
                    ArgusDiagnosticBridge.error(TAG,
                        "Memory-pressure release was postponed; disabled Vulkan still owns unrecovered model or optimizer state."
                    )
                    return false
                }
                val disabledRuntimeArenaCleared = runCatching {
                    nativeClearTensorArena()
                }.getOrDefault(false)
                if (disabledRuntimeArenaCleared) {
                    invalidateTensorResidency(releaseAllocations = false)
                    refreshTensorArenaMetrics()
                    return true
                }
                val teardownCompleted = destroyNativeRuntime()
                if (teardownCompleted) {
                    refreshTensorArenaMetrics()
                    return true
                }
                residencySuspended = true
                residencyBlockedAfterFailure = true
                residencyRebuildPending = false
                runtimeRestartPending = true
                _modelTensorResidencyReady.value = false
                ArgusDiagnosticBridge.error(TAG,
                    "Memory-pressure release was postponed; recovered host state is safe, but Vulkan teardown must be retried before residency resumes."
                )
                return false
            }
            // With no native device alive, the successfully recovered host arrays are
            // authoritative and no native allocation handles remain to be preserved.
            invalidateTensorResidency(releaseAllocations = false)
            // No native Vulkan runtime means there are no device allocations left to
            // release. The host model and optimizer state remain authoritative.
            return true
        }

        if (!synchronizeTrainingStateMirrors()) {
            residencySuspended = false
            _modelTensorResidencyReady.value = wasResidencyReady
            ArgusDiagnosticBridge.error(TAG, "Memory-pressure release was postponed because model or optimizer state could not be recovered.")
            return false
        }

        val cleared = try {
            nativeClearTensorArena()
        } catch (error: LinkageError) {
            ArgusDiagnosticBridge.error(TAG, "Vulkan arena-clear entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Vulkan arena clear failed during memory pressure.", error)
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
                ArgusDiagnosticBridge.error(TAG, "Vulkan teardown after memory-pressure release failure failed.", error)
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
        refreshModelTensorResidencyBudget(force = true)
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
            ArgusDiagnosticBridge.warning(TAG, "Resident model parameter synchronization exceeded available memory.", error)
            null
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident model parameter synchronization failed.", error)
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
        val operationCount = runCatching {
            Math.multiplyExact(
                Math.multiplyExact(inputRows.size.toLong(), inputColumns.toLong()),
                Math.multiplyExact(outputColumns.toLong(), weightTensors.size.toLong())
            )
        }.getOrNull() ?: return false
        if (!ensureModelTensorResidencyReady()) {
            return false
        }
        val gpuStartNanos = System.nanoTime()

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
            ArgusDiagnosticBridge.warning(TAG, "Resident projection-set activation allocation exceeded the GPU/host budget.", error)
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident projection-set dispatch failed before output commit.", error)
        } finally {
            for (tensor in temporaryActivations.asReversed()) {
                if (!activationTensorRegistry.release(tensor)) cleanupSucceeded = false
            }
        }

        if (!cleanupSucceeded) {
            ArgusDiagnosticBridge.warning(TAG, "Resident projection-set temporary activation cleanup was incomplete; using fallback.")
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
        AdaptiveComputePerformanceMonitor.recordGpu(
            AdaptiveComputeOperation.LINEAR,
            operationCount,
            System.nanoTime() - gpuStartNanos
        )
        return true
    }


    /**
     * Executes element-wise SwiGLU on resident Vulkan activations.
     * Gate, up, and output remain host-visible for the current backward pass.
     */
    @Synchronized
    internal fun tryApplyResidentSwiGlu(
        gateRows: Array<FloatArray>,
        upRows: Array<FloatArray>,
        outputRows: Array<FloatArray>
    ): Boolean {
        if (!isResidentSwiGluAvailable) return false
        val rowCount = gateRows.size
        if (rowCount <= 0 || upRows.size != rowCount || outputRows.size != rowCount) return false
        val columnCount = gateRows[0].size
        if (columnCount <= 0 || gateRows.any { it.size != columnCount } ||
            upRows.any { it.size != columnCount } || outputRows.any { it.size != columnCount }) {
            return false
        }

        val temporary = ArrayList<VulkanActivationTensor>(3)
        var stagedOutput: Array<FloatArray>? = null
        var cleanupSucceeded = true
        try {
            val gate = activationTensorRegistry.allocate(rowCount, columnCount)
            if (gate != null) {
                temporary.add(gate)
                val up = activationTensorRegistry.allocate(rowCount, columnCount)
                if (up != null) {
                    temporary.add(up)
                    val output = activationTensorRegistry.allocate(rowCount, columnCount)
                    if (output != null) {
                        temporary.add(output)
                        if (activationTensorRegistry.upload(gate, gateRows) &&
                            activationTensorRegistry.upload(up, upRows) &&
                            activationTensorRegistry.applySwiGlu(gate, up, output)) {
                            stagedOutput = activationTensorRegistry.download(output)
                        }
                    }
                }
            }
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Resident SwiGLU workspace exceeded the activation memory budget.", error)
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident SwiGLU failed before host-output commit.", error)
        } finally {
            for (tensor in temporary.asReversed()) {
                if (!activationTensorRegistry.release(tensor)) cleanupSucceeded = false
            }
        }

        if (!cleanupSucceeded) {
            ArgusDiagnosticBridge.warning(TAG, "Resident SwiGLU tensor release failed; rebuilding Tensor Arena residency.")
            invalidateTensorResidency(releaseAllocations = true)
            residencyRebuildPending = lastModelParameterInventory.isNotEmpty()
            return false
        }
        val completedOutput = stagedOutput ?: return false
        for (row in outputRows.indices) {
            System.arraycopy(completedOutput[row], 0, outputRows[row], 0, columnCount)
        }
        refreshGpuPerformanceMetrics()
        _currentBackend = "Vulkan Compute (Resident SwiGLU)"
        return true
    }

    /**
     * Runs W_gate, W_up, SwiGLU, W_down, residual addition, and LayerNorm as a
     * resident Tensor Arena chain. Backward caches are downloaded only after success.
     */
    @Synchronized
    internal fun tryApplyResidentFeedForward(
        inputRows: Array<FloatArray>,
        gateWeights: Float16ParameterArray,
        upWeights: Float16ParameterArray,
        downWeights: Float16ParameterArray,
        gateRows: Array<FloatArray>,
        upRows: Array<FloatArray>,
        activationRows: Array<FloatArray>,
        projectedOutputRows: Array<FloatArray>,
        normalizedOutputRows: Array<FloatArray>
    ): Boolean {
        if (!isResidentFeedForwardAvailable || inputRows.isEmpty()) return false
        val rowCount = inputRows.size
        val inputColumns = inputRows[0].size
        if (inputColumns <= 0 || inputRows.any { it.size != inputColumns }) return false
        if (gateWeights.size % inputColumns != 0) return false
        val feedForwardColumns = gateWeights.size / inputColumns
        if (feedForwardColumns <= 0) return false

        val expectedWeightElements = runCatching {
            Math.multiplyExact(inputColumns, feedForwardColumns)
        }.getOrNull() ?: return false
        if (gateWeights.size != expectedWeightElements ||
            upWeights.size != expectedWeightElements ||
            downWeights.size != expectedWeightElements) {
            return false
        }

        val outputShapesMatch = gateRows.size == rowCount &&
            upRows.size == rowCount &&
            activationRows.size == rowCount &&
            projectedOutputRows.size == rowCount &&
            normalizedOutputRows.size == rowCount &&
            gateRows.all { it.size == feedForwardColumns } &&
            upRows.all { it.size == feedForwardColumns } &&
            activationRows.all { it.size == feedForwardColumns } &&
            projectedOutputRows.all { it.size == inputColumns } &&
            normalizedOutputRows.all { it.size == inputColumns }
        if (!outputShapesMatch || !ensureModelTensorResidencyReady()) return false

        val residentGateWeights = acquireResidentModelParameterTensor(gateWeights) ?: return false
        val residentUpWeights = acquireResidentModelParameterTensor(upWeights) ?: return false
        val residentDownWeights = acquireResidentModelParameterTensor(downWeights) ?: return false

        val temporary = ArrayList<VulkanActivationTensor>(7)
        var stagedGate: Array<FloatArray>? = null
        var stagedUp: Array<FloatArray>? = null
        var stagedActivation: Array<FloatArray>? = null
        var stagedProjected: Array<FloatArray>? = null
        var stagedNormalized: Array<FloatArray>? = null
        var executionSucceeded = false
        var cleanupSucceeded = true

        try {
            executionSucceeded = run feedForward@ {
                val input = activationTensorRegistry.allocate(rowCount, inputColumns)
                    ?: return@feedForward false
                temporary.add(input)
                val gate = activationTensorRegistry.allocate(rowCount, feedForwardColumns)
                    ?: return@feedForward false
                temporary.add(gate)
                val up = activationTensorRegistry.allocate(rowCount, feedForwardColumns)
                    ?: return@feedForward false
                temporary.add(up)
                val activation = activationTensorRegistry.allocate(rowCount, feedForwardColumns)
                    ?: return@feedForward false
                temporary.add(activation)
                val down = activationTensorRegistry.allocate(rowCount, inputColumns)
                    ?: return@feedForward false
                temporary.add(down)
                val projected = activationTensorRegistry.allocate(rowCount, inputColumns)
                    ?: return@feedForward false
                temporary.add(projected)
                val normalized = activationTensorRegistry.allocate(rowCount, inputColumns)
                    ?: return@feedForward false
                temporary.add(normalized)

                if (!activationTensorRegistry.upload(input, inputRows) ||
                    !activationTensorRegistry.applyLinear(input, residentGateWeights, gate) ||
                    !activationTensorRegistry.applyLinear(input, residentUpWeights, up) ||
                    !activationTensorRegistry.applySwiGlu(gate, up, activation) ||
                    !activationTensorRegistry.applyLinear(activation, residentDownWeights, down) ||
                    !activationTensorRegistry.applyResidualLayerNorm(down, input, projected, normalized)) {
                    return@feedForward false
                }

                stagedGate = activationTensorRegistry.download(gate)
                stagedUp = activationTensorRegistry.download(up)
                stagedActivation = activationTensorRegistry.download(activation)
                stagedProjected = activationTensorRegistry.download(projected)
                stagedNormalized = activationTensorRegistry.download(normalized)
                stagedGate != null &&
                    stagedUp != null &&
                    stagedActivation != null &&
                    stagedProjected != null &&
                    stagedNormalized != null
            }
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Resident feed-forward workspace exceeded the activation memory budget.", error)
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident feed-forward graph failed before host-output commit.", error)
        } finally {
            for (tensor in temporary.asReversed()) {
                if (!activationTensorRegistry.release(tensor)) cleanupSucceeded = false
            }
        }

        if (!cleanupSucceeded) {
            ArgusDiagnosticBridge.warning(TAG, "Resident feed-forward cleanup failed; invalidating Tensor Arena residency.")
            invalidateTensorResidency(releaseAllocations = true)
            residencyRebuildPending = lastModelParameterInventory.isNotEmpty()
            return false
        }
        if (!executionSucceeded) return false

        val completedGate = stagedGate ?: return false
        val completedUp = stagedUp ?: return false
        val completedActivation = stagedActivation ?: return false
        val completedProjected = stagedProjected ?: return false
        val completedNormalized = stagedNormalized ?: return false
        for (row in 0 until rowCount) {
            System.arraycopy(completedGate[row], 0, gateRows[row], 0, feedForwardColumns)
            System.arraycopy(completedUp[row], 0, upRows[row], 0, feedForwardColumns)
            System.arraycopy(completedActivation[row], 0, activationRows[row], 0, feedForwardColumns)
            System.arraycopy(completedProjected[row], 0, projectedOutputRows[row], 0, inputColumns)
            System.arraycopy(completedNormalized[row], 0, normalizedOutputRows[row], 0, inputColumns)
        }
        refreshGpuPerformanceMetrics()
        _currentBackend = "Vulkan Compute (Resident Feed-forward block)"
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
            ArgusDiagnosticBridge.warning(TAG, "Resident attention workspace exceeded the activation memory budget.", error)
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident attention graph failed before host-output commit.", error)
        } finally {
            for (tensor in temporary.asReversed()) {
                if (!activationTensorRegistry.release(tensor)) cleanupSucceeded = false
            }
        }

        if (!cleanupSucceeded) {
            ArgusDiagnosticBridge.warning(TAG, "Resident attention tensor release failed; rebuilding residency before retry.")
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

    /** Reset failed native state before an explicit backend initialization retry. */
    override fun resetForRetry() {
        resetRuntimeForRetry()
    }

    /** Retained Vulkan-specific entry point for internal recovery orchestration. */
    @Synchronized
    fun resetRuntimeForRetry() {
        if (nativeReady && !synchronizeTrainingStateMirrors()) {
            ArgusDiagnosticBridge.error(TAG, "Vulkan retry was postponed because model or optimizer state could not be recovered.")
            return
        }
        if (!modelParameterMirrorReliable || !optimizerStateMirrorReliable) {
            ArgusDiagnosticBridge.error(TAG, "Vulkan retry was refused because a host training-state mirror is stale.")
            return
        }
        if (initializationAttempted) {
            runCatching { nativeDestroy() }
                .onFailure { error ->
                    ArgusDiagnosticBridge.warning(TAG, "Vulkan native cleanup during runtime retry failed.", error)
                }
        }
        nativeReady = false
        fp16RuntimeReady = false
        fp16DisabledAfterRuntimeFailure = false
        residentActivationLinearReady = false
        residentAttentionReady = false
        residentSwiGluReady = false
        residentResidualLayerNormReady = false
        residentBackwardReady = false
        residentGradientAccumulatorReady = false
        residentAdamwReady = false
        disabledAfterRuntimeFailure = false
        initializationAttempted = false
        residencySuspended = false
        residencyBlockedAfterFailure = false
        residencyRebuildPending = false
        runtimeRestartPending = false
        residentGradientRecoveryPending = false
        invalidateTensorResidency(releaseAllocations = false)
        _modelTensorResidencyBudgetBytes.value = 0L
        _activationTensorResidencyBudgetBytes.value = 0L
        _optimizerStateResidencyBudgetBytes.value = 0L
        _deviceCapabilities.value = null
        _deviceProfile.value = null
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
    override fun initializeAfterModelReady(): Boolean {
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
        if (!modelParameterMirrorReliable || !optimizerStateMirrorReliable) {
            ArgusDiagnosticBridge.error(TAG, "Vulkan initialization is blocked because the host training-state mirror is stale.")
            return false
        }

        residentActivationLinearReady = false
        residentAttentionReady = false
        residentSwiGluReady = false
        residentResidualLayerNormReady = false
        residentBackwardReady = false
        residentGradientAccumulatorReady = false
        residentAdamwReady = false
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
                    ArgusDiagnosticBridge.warning(TAG, "Vulkan runtime started but capability metadata could not be parsed.", error)
                }.getOrNull()
                val detectedProfile = detectedCapabilities?.toDeviceProfile()
                _deviceCapabilities.value = detectedCapabilities
                _deviceProfile.value = detectedProfile
                _modelTensorResidencyBudgetBytes.value =
                    calculateModelTensorResidencyBudgetBytes(detectedProfile)
                _activationTensorResidencyBudgetBytes.value =
                    calculateActivationTensorResidencyBudgetBytes(detectedProfile)
                _optimizerStateResidencyBudgetBytes.value =
                    calculateOptimizerStateResidencyBudgetBytes(detectedProfile)
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
            nativeReady = verified

            residentAttentionReady = false
            residentSwiGluReady = false
            residentResidualLayerNormReady = false
            residentActivationLinearReady = verified && runCatching {
                verifyResidentActivationLinearKernel() &&
                    verifyResidentActivationLinearBatchKernel()
            }.getOrDefault(false)
            residentAttentionReady = verified && residentActivationLinearReady && runCatching {
                verifyResidentAttentionKernel()
            }.getOrDefault(false)
            residentSwiGluReady = verified && runCatching {
                verifyResidentSwiGluKernel()
            }.getOrDefault(false)
            residentResidualLayerNormReady =
                verified && residentAttentionReady && runCatching {
                    verifyResidentResidualLayerNormKernel()
                }.getOrDefault(false)

            residentBackwardReady = verified && runCatching {
                verifyResidentBackwardKernels()
            }.getOrDefault(false)
            residentGradientAccumulatorReady = verified && residentBackwardReady && runCatching {
                verifyResidentGradientAccumulator()
            }.getOrDefault(false)
            residentAdamwReady = verified && residentActivationLinearReady && runCatching {
                verifyResidentAdamwKernel()
            }.getOrDefault(false)
            if (verified && residentActivationLinearReady && !residentAdamwReady) {
                ArgusDiagnosticBridge.warning(TAG,
                    "Resident FP16 AdamW self-check failed; staged optimizer and CPU fallback remain available."
                )
            }
            if (verified && residentBackwardReady && !residentGradientAccumulatorReady) {
                ArgusDiagnosticBridge.warning(TAG,
                    "Resident gradient accumulation/norm/clipping self-check failed; the Kotlin accumulation fallback remains active."
                )
            }
            if (verified && residentAttentionReady && !residentResidualLayerNormReady) {
                ArgusDiagnosticBridge.warning(TAG,
                    "Resident residual LayerNorm self-check failed; the existing feed-forward fallback remains active."
                )
            }
            if (verified && !residentSwiGluReady) {
                ArgusDiagnosticBridge.warning(TAG,
                    "Resident SwiGLU self-check failed; the Kotlin activation fallback remains active."
                )
            }
            if (verified && !residentAttentionReady) {
                ArgusDiagnosticBridge.warning(TAG,
                    "Resident causal-attention self-check failed; the existing Kotlin attention fallback remains active."
                )
            }
            if (verified && !residentActivationLinearReady) {
                ArgusDiagnosticBridge.warning(TAG,
                    "Resident activation-linear kernel self-check failed; existing verified Vulkan/CPU paths remain active."
                )
            }

            if (verified && !residentBackwardReady) {
                ArgusDiagnosticBridge.warning(TAG,
                    "Resident backward numerical self-check failed; the verified Kotlin backward path remains active."
                )
            }

            if (nativeReady && runCatching { nativeSupportsFp16() }.getOrDefault(false)) {
                publishInitializationProgress(
                    status = "Verifying FP16 acceleration",
                    driverInfo = detectedDriverInfo,
                    isInitializing = true
                )
                fp16RuntimeReady = runCatching {
                    verifyNativeFp16Kernel()
                }.getOrDefault(false)
                if (!fp16RuntimeReady) {
                    ArgusDiagnosticBridge.warning(TAG,
                        "Vulkan FP16 kernel self-check failed; the verified FP32 Vulkan path remains available."
                    )
                }
            }

            // Optional kernel checks can tear down the native device when a temporary
            // allocation cannot be released. Never republish a destroyed runtime.
            val runtimeIntact = verified && nativeReady
            if (!runtimeIntact) {
                releaseNativeRuntimeAfterFailure(nativeLibraryLoaded)
            }

            nativeReady = runtimeIntact
            if (runtimeIntact) {
                refreshGpuPerformanceMetrics()
                _currentBackend = "Kotlin CPU"
                _runtimeStatus.value = when {
                    residentAttentionReady && residentSwiGluReady && residentResidualLayerNormReady ->
                        "Vulkan runtime verified — resident attention and complete feed-forward kernels passed self-check"
                    residentAttentionReady && residentSwiGluReady ->
                        "Vulkan runtime verified — attention and SwiGLU passed; feed-forward residual LayerNorm uses Kotlin fallback"
                    residentAttentionReady ->
                        "Vulkan runtime verified — attention passed; feed-forward operations use Kotlin fallback"
                    else ->
                        "Vulkan runtime verified — attention and feed-forward paths use Kotlin fallback"
                }
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
                ArgusDiagnosticBridge.warning(TAG, "Vulkan is unavailable or failed verification; Kotlin CPU remains active.")
            }
            runtimeIntact
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
            ArgusDiagnosticBridge.warning(TAG, "Native Vulkan library loading was blocked.", error)
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
            ArgusDiagnosticBridge.warning(TAG, "Vulkan initialization exceeded the available memory budget; keeping CPU fallback.", error)
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
            ArgusDiagnosticBridge.warning(TAG, "Native Vulkan initialization failed.", error)
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
            ArgusDiagnosticBridge.error(TAG, "Resident activation-linear self-check entry point is unavailable.", error)
            return false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident activation-linear self-check failed.", error)
            return false
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Resident activation-linear self-check exceeded available memory.", error)
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
            ArgusDiagnosticBridge.error(TAG, "Resident activation batch self-check entry point is unavailable.", error)
            return false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident activation batch self-check failed.", error)
            return false
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Resident activation batch self-check exceeded available memory.", error)
            return false
        } finally {
            for (handle in weightHandles) {
                if (handle != 0L) runCatching { nativeReleaseTensor(handle) }
            }
            if (outputHandle != 0L) runCatching { nativeReleaseTensor(outputHandle) }
            if (inputHandle != 0L) runCatching { nativeReleaseTensor(inputHandle) }
        }
    }


    private fun dispatchResidentSwiGlu(
        gate: VulkanTensorAllocation,
        up: VulkanTensorAllocation,
        output: VulkanTensorAllocation,
        rows: Int,
        columns: Int
    ): Boolean {
        if (!isRuntimeAvailable || !residentSwiGluReady || rows <= 0 || columns <= 0) return false
        val applied = try {
            nativeRunResidentSwiGlu(
                gate.nativeHandle, up.nativeHandle, output.nativeHandle, rows, columns
            )
        } catch (error: LinkageError) {
            ArgusDiagnosticBridge.error(TAG, "Resident Vulkan SwiGLU JNI entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident Vulkan SwiGLU dispatch failed.", error)
            false
        }
        if (!applied) {
            residentSwiGluReady = false
            _runtimeStatus.value =
                "Vulkan runtime verified — resident SwiGLU dispatch failed; this operation uses Kotlin fallback"
            ArgusDiagnosticBridge.warning(TAG, "Resident Vulkan SwiGLU dispatch failed; Kotlin activation fallback remains active.")
        }
        return applied
    }

    private fun dispatchResidentResidualLayerNorm(
        down: VulkanTensorAllocation,
        residual: VulkanTensorAllocation,
        projected: VulkanTensorAllocation,
        normalized: VulkanTensorAllocation,
        rows: Int,
        columns: Int
    ): Boolean {
        if (!isRuntimeAvailable ||
            !residentResidualLayerNormReady ||
            rows <= 0 ||
            columns <= 0) {
            return false
        }
        val applied = try {
            nativeRunResidentResidualLayerNorm(
                down.nativeHandle,
                residual.nativeHandle,
                projected.nativeHandle,
                normalized.nativeHandle,
                rows,
                columns
            )
        } catch (error: LinkageError) {
            ArgusDiagnosticBridge.error(TAG, "Resident residual LayerNorm JNI entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident residual LayerNorm dispatch failed.", error)
            false
        }
        if (!applied) {
            residentResidualLayerNormReady = false
            ArgusDiagnosticBridge.warning(TAG,
                "Resident residual LayerNorm dispatch failed; the complete feed-forward " +
                    "GPU chain is disabled for this runtime."
            )
        }
        return applied
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
            ArgusDiagnosticBridge.error(TAG, "Resident Vulkan attention JNI entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident Vulkan attention graph dispatch failed.", error)
            false
        }
        if (applied) {
            refreshGpuPerformanceMetrics()
            _currentBackend = "Vulkan Compute (Resident attention + LayerNorm)"
        } else {
            residentAttentionReady = false
            _currentBackend = "Kotlin CPU (Resident attention + LayerNorm fallback)"
            _runtimeStatus.value =
                "Vulkan runtime verified — resident attention/LayerNorm dispatch failed; this path uses Kotlin fallback"
            ArgusDiagnosticBridge.warning(TAG, "Resident Vulkan attention/LayerNorm dispatch failed; M3B–M3E fall back to Kotlin.")
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
                    maximum = maxOf(maximum, score)
                }
                var denominator = 0f
                for (keyIndex in 0..queryIndex) {
                    val probability = kotlin.math.exp(
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
            ArgusDiagnosticBridge.error(TAG, "Resident attention self-check JNI entry point is unavailable.", error)
            return false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident attention self-check failed.", error)
            return false
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Resident attention self-check exceeded available memory.", error)
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

    /**
     * Verifies residual addition and population-variance LayerNorm at non-power-of-two
     * width 73. The first two rows exercise zero-variance and constant-value stability.
     */
    private fun verifyResidentResidualLayerNormKernel(): Boolean {
        val rows = 3
        val columns = 73
        val elementCount = Math.multiplyExact(rows, columns)
        val down = FloatArray(elementCount) { index ->
            val row = index / columns
            val column = index % columns
            when (row) {
                0 -> ((column % 11) - 5) * 0.125f
                1 -> ((column % 7) - 3) * 0.25f
                else -> ((column * 3 % 17) - 8) * 0.0625f
            }
        }
        val residual = FloatArray(elementCount) { index ->
            val row = index / columns
            val column = index % columns
            when (row) {
                0 -> -down[index]
                1 -> 3.25f - down[index]
                else -> ((column * 5 % 19) - 9) * 0.0625f
            }
        }
        val expectedProjected = FloatArray(elementCount) { index -> down[index] + residual[index] }
        val expectedNormalized = FloatArray(elementCount)
        for (row in 0 until rows) {
            val base = row * columns
            var sum = 0f
            for (column in 0 until columns) sum += expectedProjected[base + column]
            val mean = sum / columns.toFloat()
            var varianceSum = 0f
            for (column in 0 until columns) {
                val difference = expectedProjected[base + column] - mean
                varianceSum += difference * difference
            }
            val inverseStandardDeviation =
                1.0f / kotlin.math.sqrt(varianceSum / columns.toFloat() + 1.0e-5f)
            for (column in 0 until columns) {
                expectedNormalized[base + column] =
                    (expectedProjected[base + column] - mean) * inverseStandardDeviation
            }
        }

        var downHandle = 0L
        var residualHandle = 0L
        var projectedHandle = 0L
        var normalizedHandle = 0L
        var operationSucceeded = false
        var cleanupSucceeded = true
        try {
            operationSucceeded = run selfCheck@ {
                val downBytes = encodeFloatArray(down)
                val residualBytes = encodeFloatArray(residual)
                val projectedBytes = ByteArray(Math.multiplyExact(elementCount, Float.SIZE_BYTES))
                val normalizedBytes = ByteArray(Math.multiplyExact(elementCount, Float.SIZE_BYTES))
                downHandle = nativeAllocateTensor(downBytes.size.toLong())
                residualHandle = nativeAllocateTensor(residualBytes.size.toLong())
                projectedHandle = nativeAllocateTensor(projectedBytes.size.toLong())
                normalizedHandle = nativeAllocateTensor(normalizedBytes.size.toLong())
                if (downHandle == 0L || residualHandle == 0L ||
                    projectedHandle == 0L || normalizedHandle == 0L) {
                    return@selfCheck false
                }
                if (!nativeUploadTensor(downHandle, downBytes) ||
                    !nativeUploadTensor(residualHandle, residualBytes) ||
                    !nativeRunResidentResidualLayerNorm(
                        downHandle,
                        residualHandle,
                        projectedHandle,
                        normalizedHandle,
                        rows,
                        columns
                    ) ||
                    !nativeDownloadTensor(projectedHandle, projectedBytes) ||
                    !nativeDownloadTensor(normalizedHandle, normalizedBytes)) {
                    return@selfCheck false
                }
                approximatelyEqual(expectedProjected, decodeFloatArray(projectedBytes), 1.0e-4f) &&
                    approximatelyEqual(expectedNormalized, decodeFloatArray(normalizedBytes), 1.0e-4f)
            }
        } catch (error: LinkageError) {
            ArgusDiagnosticBridge.error(TAG, "Resident residual LayerNorm self-check JNI entry point is unavailable.", error)
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident residual LayerNorm numerical self-check failed.", error)
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Resident residual LayerNorm self-check exceeded available memory.", error)
        } finally {
            if (normalizedHandle != 0L &&
                !runCatching { nativeReleaseTensor(normalizedHandle) }.getOrDefault(false)) {
                cleanupSucceeded = false
            }
            if (projectedHandle != 0L &&
                !runCatching { nativeReleaseTensor(projectedHandle) }.getOrDefault(false)) {
                cleanupSucceeded = false
            }
            if (residualHandle != 0L &&
                !runCatching { nativeReleaseTensor(residualHandle) }.getOrDefault(false)) {
                cleanupSucceeded = false
            }
            if (downHandle != 0L &&
                !runCatching { nativeReleaseTensor(downHandle) }.getOrDefault(false)) {
                cleanupSucceeded = false
            }
        }
        if (!cleanupSucceeded) {
            ArgusDiagnosticBridge.warning(TAG, "Resident residual LayerNorm self-check could not release every temporary tensor.")
            runCatching { nativeClearTensorArena() }
        }
        return operationSucceeded && cleanupSucceeded
    }

    private fun verifyResidentSwiGluKernel(): Boolean {
        val rows = 3
        val columns = 73
        val elementCount = Math.multiplyExact(rows, columns)
        val gate = FloatArray(elementCount) { index -> (((index * 7) % 31) - 15) * 0.25f }
        val up = FloatArray(elementCount) { index -> (((index * 11) % 23) - 11) * 0.125f }
        val expected = FloatArray(elementCount) { index ->
            val sigmoid = 1.0f / (1.0f + kotlin.math.exp(-gate[index]))
            (gate[index] * sigmoid) * up[index]
        }
        var gateHandle = 0L
        var upHandle = 0L
        var outputHandle = 0L
        var verified = false
        var cleanupSucceeded = true
        try {
            val gateBytes = encodeFloatArray(gate)
            val upBytes = encodeFloatArray(up)
            val outputBytes = ByteArray(Math.multiplyExact(elementCount, Float.SIZE_BYTES))
            gateHandle = nativeAllocateTensor(gateBytes.size.toLong())
            upHandle = nativeAllocateTensor(upBytes.size.toLong())
            outputHandle = nativeAllocateTensor(outputBytes.size.toLong())
            if (gateHandle == 0L || upHandle == 0L || outputHandle == 0L) return false
            if (!nativeUploadTensor(gateHandle, gateBytes) || !nativeUploadTensor(upHandle, upBytes) ||
                !nativeRunResidentSwiGlu(gateHandle, upHandle, outputHandle, rows, columns) ||
                !nativeDownloadTensor(outputHandle, outputBytes)) return false
            verified = approximatelyEqual(expected, decodeFloatArray(outputBytes), 1.0e-4f)
        } catch (error: LinkageError) {
            ArgusDiagnosticBridge.error(TAG, "Resident SwiGLU self-check entry point is unavailable.", error)
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident SwiGLU self-check failed.", error)
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Resident SwiGLU self-check exceeded available memory.", error)
        } finally {
            if (outputHandle != 0L && !runCatching { nativeReleaseTensor(outputHandle) }.getOrDefault(false)) cleanupSucceeded = false
            if (upHandle != 0L && !runCatching { nativeReleaseTensor(upHandle) }.getOrDefault(false)) cleanupSucceeded = false
            if (gateHandle != 0L && !runCatching { nativeReleaseTensor(gateHandle) }.getOrDefault(false)) cleanupSucceeded = false
        }
        if (!cleanupSucceeded) ArgusDiagnosticBridge.warning(TAG, "Resident SwiGLU self-check could not release every temporary tensor.")
        return verified && cleanupSucceeded
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
     * Dispatches one verified FP32 backward kernel. Every input and output gets an
     * isolated arena allocation; caller-owned arrays are committed only after all
     * readbacks are finite and every temporary allocation has been released.
     */
    @Synchronized
    internal fun tryRunResidentBackward(
        operation: Int,
        rows: Int,
        columns: Int,
        sequenceLength: Int,
        inputs: Array<FloatArray?>,
        outputs: Array<FloatArray>,
        scale: Float = 1.0f,
        epsilon: Float = 1.0e-5f
    ): Boolean {
        if (!isRuntimeAvailable ||
            inputs.size != 3 ||
            outputs.isEmpty() ||
            outputs.size > 2 ||
            rows <= 0 ||
            columns <= 0 ||
            !scale.isFinite() ||
            !epsilon.isFinite() ||
            epsilon <= 0.0f ||
            inputs[0] == null) {
            return false
        }

        val temporary = ArrayList<VulkanTensorAllocation>(inputs.count { it != null } + outputs.size)
        val outputBytes = ArrayList<ByteArray>(outputs.size)
        var dispatched = false
        var cleanupSucceeded = true
        val stagedOutputs = arrayOfNulls<FloatArray>(outputs.size)

        try {
            var requiredBytes = 0L
            for (input in inputs) {
                if (input != null) {
                    requiredBytes = Math.addExact(
                        requiredBytes,
                        Math.multiplyExact(input.size.toLong(), Float.SIZE_BYTES.toLong())
                    )
                }
            }
            for (output in outputs) {
                requiredBytes = Math.addExact(
                    requiredBytes,
                    Math.multiplyExact(output.size.toLong(), Float.SIZE_BYTES.toLong())
                )
            }
            val budget = _activationTensorResidencyBudgetBytes.value
            if (budget > 0L && requiredBytes > budget) return false

            val inputAllocations = arrayOfNulls<VulkanTensorAllocation>(3)
            for (index in inputs.indices) {
                val values = inputs[index] ?: continue
                if (values.isEmpty()) return false
                val bytes = encodeFloatArray(values)
                val allocation = allocateGpuTensor(bytes.size.toLong()) ?: return false
                temporary.add(allocation)
                if (!uploadGpuTensor(allocation, bytes)) return false
                inputAllocations[index] = allocation
            }

            val outputAllocations = ArrayList<VulkanTensorAllocation>(outputs.size)
            for (output in outputs) {
                if (output.isEmpty()) return false
                val byteCount = Math.multiplyExact(output.size, Float.SIZE_BYTES)
                val allocation = allocateGpuTensor(byteCount.toLong()) ?: return false
                temporary.add(allocation)
                outputAllocations.add(allocation)
                outputBytes.add(ByteArray(byteCount))
            }

            val input0 = inputAllocations[0] ?: return false
            dispatched = nativeRunResidentBackward(
                operation = operation,
                input0Handle = input0.nativeHandle,
                input1Handle = inputAllocations[1]?.nativeHandle ?: 0L,
                input2Handle = inputAllocations[2]?.nativeHandle ?: 0L,
                output0Handle = outputAllocations[0].nativeHandle,
                output1Handle = outputAllocations.getOrNull(1)?.nativeHandle ?: 0L,
                rows = rows,
                columns = columns,
                sequenceLength = sequenceLength,
                scale = scale,
                epsilon = epsilon
            )
            if (!dispatched) return false

            for (index in outputAllocations.indices) {
                if (!downloadGpuTensor(outputAllocations[index], outputBytes[index])) {
                    return false
                }
                val values = decodeFloatArray(outputBytes[index])
                if (values.any { !it.isFinite() } || values.size != outputs[index].size) {
                    ArgusDiagnosticBridge.warning(TAG, "Resident backward kernel returned invalid FP32 output.")
                    return false
                }
                stagedOutputs[index] = values
            }
        } catch (error: LinkageError) {
            ArgusDiagnosticBridge.error(TAG, "Resident backward JNI entry point is unavailable.", error)
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident backward dispatch failed validation or execution.", error)
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Resident backward workspace exceeded the available activation budget.", error)
        } finally {
            for (allocation in temporary.asReversed()) {
                if (!runCatching { releaseGpuTensor(allocation) }.getOrDefault(false)) {
                    cleanupSucceeded = false
                }
            }
        }

        if (!cleanupSucceeded) {
            ArgusDiagnosticBridge.warning(TAG, "Resident backward dispatch could not release every temporary tensor.")
            runCatching { destroyNativeRuntime() }
                .onFailure { error ->
                    ArgusDiagnosticBridge.error(TAG, "Failed to tear down Vulkan after a backward workspace leak.", error)
                }
            return false
        }
        if (!dispatched) return false

        for (index in outputs.indices) {
            val result = stagedOutputs[index] ?: return false
            System.arraycopy(result, 0, outputs[index], 0, result.size)
        }
        runCatching { refreshGpuPerformanceMetrics() }
        _currentBackend = "Vulkan Compute (resident backward)"
        return true
    }

    @Synchronized
    internal fun createResidentGradientAccumulator(
        gradientTensors: Array<FloatArray>
    ): VulkanResidentGradientAccumulator? {
        if (!isResidentBackwardAvailable || gradientTensors.isEmpty() ||
            gradientTensors.any { it.isEmpty() }) {
            return null
        }

        var active: VulkanTensorAllocation? = null
        var spare: VulkanTensorAllocation? = null
        var staging: VulkanTensorAllocation? = null
        var committed = false
        try {
            val elementCount = gradientTensors.fold(0L) { total, tensor ->
                Math.addExact(total, tensor.size.toLong())
            }
            if (elementCount <= 0L || elementCount > Int.MAX_VALUE.toLong()) return null
            val byteCount = Math.multiplyExact(elementCount, Float.SIZE_BYTES.toLong())
            if (byteCount > Int.MAX_VALUE.toLong()) return null

            val budget = _activationTensorResidencyBudgetBytes.value
            if (budget > 0L && byteCount > budget / 3L) return null

            val activeAllocation = allocateGpuTensor(byteCount) ?: return null
            active = activeAllocation
            val spareAllocation = allocateGpuTensor(byteCount) ?: return null
            spare = spareAllocation
            val stagingAllocation = allocateGpuTensor(byteCount) ?: return null
            staging = stagingAllocation

            val zeros = ByteArray(byteCount.toInt())
            if (!uploadGpuTensor(activeAllocation, zeros) ||
                !uploadGpuTensor(spareAllocation, zeros)) {
                return null
            }

            val accumulator = VulkanResidentGradientAccumulator(
                active = activeAllocation,
                spare = spareAllocation,
                staging = stagingAllocation,
                elementCount = elementCount.toInt(),
                tensorSizes = IntArray(gradientTensors.size) { index ->
                    gradientTensors[index].size
                }
            )
            committed = true
            liveResidentGradientAccumulators.add(accumulator)
            return accumulator
        } catch (error: LinkageError) {
            ArgusDiagnosticBridge.error(TAG, "Resident gradient accumulator JNI entry point is unavailable.", error)
            return null
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Resident gradient accumulator exceeded the available memory budget.", error)
            return null
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident gradient accumulator allocation failed.", error)
            return null
        } finally {
            if (!committed) {
                var cleanupSucceeded = true
                val stagingToRelease = staging
                if (stagingToRelease != null && !releaseGpuTensor(stagingToRelease)) {
                    cleanupSucceeded = false
                }
                val spareToRelease = spare
                if (spareToRelease != null && !releaseGpuTensor(spareToRelease)) {
                    cleanupSucceeded = false
                }
                val activeToRelease = active
                if (activeToRelease != null && !releaseGpuTensor(activeToRelease)) {
                    cleanupSucceeded = false
                }
                if (!cleanupSucceeded && isRuntimeAvailable) {
                    ArgusDiagnosticBridge.error(TAG, "Partial gradient accumulator cleanup failed; disabling the optional Vulkan runtime.")
                    runCatching { destroyNativeRuntime() }
                }
            }
        }
    }

    @Synchronized
    internal fun accumulateResidentGradients(
        accumulator: VulkanResidentGradientAccumulator,
        gradientTensors: Array<FloatArray>,
        targetWeight: Int
    ): Boolean {
        if (!isResidentBackwardAvailable ||
            accumulator.released ||
            targetWeight <= 0) {
            return false
        }
        val flattened = flattenGradientTensors(gradientTensors, accumulator.tensorSizes) ?: return false
        if (flattened.size != accumulator.elementCount) return false

        val committed = try {
            val inputBytes = encodeFloatArray(flattened)
            if (!uploadGpuTensor(accumulator.staging, inputBytes)) {
                false
            } else {
                val updated = nativeRunResidentBackward(
                    operation = BACKWARD_GRADIENT_ACCUMULATE,
                    input0Handle = accumulator.staging.nativeHandle,
                    input1Handle = accumulator.active.nativeHandle,
                    input2Handle = 0L,
                    output0Handle = accumulator.spare.nativeHandle,
                    output1Handle = 0L,
                    rows = accumulator.elementCount,
                    columns = 1,
                    sequenceLength = 0,
                    scale = targetWeight.toFloat(),
                    epsilon = 1.0e-5f
                )
                if (updated) {
                    val previous = accumulator.active
                    accumulator.active = accumulator.spare
                    accumulator.spare = previous
                }
                updated
            }
        } catch (error: LinkageError) {
            ArgusDiagnosticBridge.error(TAG, "Resident gradient-accumulation JNI entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident gradient accumulation failed.", error)
            false
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Resident gradient accumulation exceeded available memory.", error)
            false
        }

        // Once the buffer swap is committed, optional metrics must not turn success into
        // a failure signal that would cause the caller to append the same sample twice.
        if (committed) runCatching { refreshGpuPerformanceMetrics() }
        return committed
    }

    @Synchronized
    internal fun downloadResidentGradientAccumulator(
        accumulator: VulkanResidentGradientAccumulator,
        destinationTensors: Array<FloatArray>
    ): Boolean {
        if (accumulator.released) return false
        return try {
            val bytes = ByteArray(Math.multiplyExact(accumulator.elementCount, Float.SIZE_BYTES))
            if (!downloadGpuTensor(accumulator.active, bytes)) return false
            val values = decodeFloatArray(bytes)
            if (values.any { !it.isFinite() }) return false
            if (!hasGradientTensorShapes(destinationTensors, accumulator.tensorSizes)) return false
            if (!releaseResidentGradientAccumulator(accumulator)) return false
            copyGradientTensors(values, destinationTensors, accumulator.tensorSizes)
            true
        } catch (error: LinkageError) {
            ArgusDiagnosticBridge.error(TAG, "Resident gradient accumulator download JNI entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident gradient accumulator recovery failed.", error)
            false
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Resident gradient accumulator recovery exceeded available memory.", error)
            false
        }
    }

    @Synchronized
    internal fun discardResidentGradientAccumulator(
        accumulator: VulkanResidentGradientAccumulator
    ) {
        if (releaseResidentGradientAccumulator(accumulator)) return
        if (isRuntimeAvailable) {
            ArgusDiagnosticBridge.warning(TAG, "Resident gradient accumulator cleanup failed; disabling the optional Vulkan runtime.")
            runCatching { destroyNativeRuntime() }
                .onFailure { error ->
                    ArgusDiagnosticBridge.error(TAG, "Vulkan teardown after accumulator cleanup failure failed.", error)
                }
        } else {
            ArgusDiagnosticBridge.warning(TAG,
                "Resident gradient accumulator cleanup could not run while the Vulkan runtime is unavailable; retiring its bookkeeping."
            )
        }
        // The native allocations are either destroyed with the runtime above or belong
        // to a runtime that is no longer reachable, so the accumulator must never be
        // reused or retried: always retire it in bookkeeping, even when the native
        // release could not run.
        markResidentGradientAccumulatorReleased(accumulator)
    }

    @Synchronized
    internal fun finalizeResidentGradientAccumulator(
        accumulator: VulkanResidentGradientAccumulator,
        destinationTensors: Array<FloatArray>,
        targetCount: Int,
        maximumGradientNorm: Float,
        retainResidentBuffer: Boolean = false
    ): Boolean {
        if (!isResidentBackwardAvailable ||
            accumulator.released ||
            targetCount <= 0 ||
            !maximumGradientNorm.isFinite() ||
            maximumGradientNorm <= 0.0f ||
            !hasGradientTensorShapes(destinationTensors, accumulator.tensorSizes)) {
            return false
        }

        var partial: VulkanTensorAllocation? = null
        var norm: VulkanTensorAllocation? = null
        var stagedResult: FloatArray? = null
        var completed = false
        var cleanupSucceeded = true
        try {
            val partialCount = 1 + (accumulator.elementCount - 1) / 4096
            val partialAllocation = allocateGpuTensor(
                Math.multiplyExact(partialCount, Float.SIZE_BYTES).toLong()
            ) ?: return false
            partial = partialAllocation
            val normAllocation = allocateGpuTensor(Float.SIZE_BYTES.toLong()) ?: return false
            norm = normAllocation

            if (!nativeRunResidentBackward(
                    operation = BACKWARD_GRADIENT_NORM_PARTIAL,
                    input0Handle = accumulator.active.nativeHandle,
                    input1Handle = 0L,
                    input2Handle = 0L,
                    output0Handle = partialAllocation.nativeHandle,
                    output1Handle = 0L,
                    rows = accumulator.elementCount,
                    columns = 1,
                    sequenceLength = 0,
                    scale = 1.0f,
                    epsilon = 1.0e-5f
                )) {
                return false
            }
            if (!nativeRunResidentBackward(
                    operation = BACKWARD_GRADIENT_NORM_FINAL,
                    input0Handle = partialAllocation.nativeHandle,
                    input1Handle = 0L,
                    input2Handle = 0L,
                    output0Handle = normAllocation.nativeHandle,
                    output1Handle = 0L,
                    rows = partialCount,
                    columns = 1,
                    sequenceLength = 0,
                    scale = 1.0f,
                    epsilon = 1.0e-5f
                )) {
                return false
            }

            // The final shader reads the GPU-reduced squared norm and computes
            // global clipping without transferring norm state to Kotlin.
            if (!nativeRunResidentBackward(
                    operation = BACKWARD_GRADIENT_NORMALIZE_CLIP,
                    input0Handle = accumulator.active.nativeHandle,
                    input1Handle = normAllocation.nativeHandle,
                    input2Handle = 0L,
                    output0Handle = accumulator.spare.nativeHandle,
                    output1Handle = 0L,
                    rows = accumulator.elementCount,
                    columns = 1,
                    sequenceLength = 0,
                    scale = 1.0f / targetCount.toFloat(),
                    epsilon = maximumGradientNorm
                )) {
                return false
            }

            // Resident AdamW validates every consumed gradient in its first GPU pass.
            // Keep the normalized/clipped tensor resident instead of reading the complete
            // gradient inventory back to Kotlin when the optimizer will consume it on GPU.
            if (!retainResidentBuffer) {
                val resultBytes = ByteArray(
                    Math.multiplyExact(accumulator.elementCount, Float.SIZE_BYTES)
                )
                if (!downloadGpuTensor(accumulator.spare, resultBytes)) return false
                val result = decodeFloatArray(resultBytes)
                if (result.any { !it.isFinite() }) return false
                stagedResult = result
            }
            completed = true
        } catch (error: LinkageError) {
            ArgusDiagnosticBridge.error(TAG, "Resident gradient finalization JNI entry point is unavailable.", error)
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident gradient finalization failed.", error)
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Resident gradient finalization exceeded available memory.", error)
        } finally {
            val partialToRelease = partial
            if (partialToRelease != null &&
                !runCatching { releaseGpuTensor(partialToRelease) }.getOrDefault(false)) {
                accumulator.pendingReleaseAllocations.add(partialToRelease)
                cleanupSucceeded = false
            }
            val normToRelease = norm
            if (normToRelease != null &&
                !runCatching { releaseGpuTensor(normToRelease) }.getOrDefault(false)) {
                accumulator.pendingReleaseAllocations.add(normToRelease)
                cleanupSucceeded = false
            }
            if (completed && cleanupSucceeded) {
                if (retainResidentBuffer) {
                    val previousActive = accumulator.active
                    accumulator.active = accumulator.spare
                    accumulator.spare = previousActive
                } else if (!releaseResidentGradientAccumulator(accumulator)) {
                    cleanupSucceeded = false
                }
            }
        }
        if (!completed || !cleanupSucceeded) return false
        if (retainResidentBuffer) return true
        val result = stagedResult ?: return false
        copyGradientTensors(result, destinationTensors, accumulator.tensorSizes)
        return true
    }

    private fun flattenGradientTensors(
        tensors: Array<FloatArray>,
        expectedSizes: IntArray
    ): FloatArray? {
        if (tensors.size != expectedSizes.size) return null
        val elementCount = expectedSizes.fold(0L) { total, size ->
            if (size <= 0) return null
            Math.addExact(total, size.toLong())
        }
        if (elementCount > Int.MAX_VALUE.toLong()) return null
        return try {
            FloatArray(elementCount.toInt()).also { result ->
                var offset = 0
                for (index in tensors.indices) {
                    val tensor = tensors[index]
                    val expectedSize = expectedSizes[index]
                    if (tensor.isEmpty() || tensor.size > expectedSize ||
                        tensor.any { !it.isFinite() }) {
                        return null
                    }
                    // A growing vocabulary may leave the active output-head gradient
                    // shorter than its persistent optimizer capacity. Pad the unused suffix.
                    System.arraycopy(tensor, 0, result, offset, tensor.size)
                    offset += expectedSize
                }
            }
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Could not flatten gradient tensors for Vulkan accumulation.", error)
            null
        }
    }

    private fun hasGradientTensorShapes(
        tensors: Array<FloatArray>,
        expectedSizes: IntArray
    ): Boolean =
        tensors.size == expectedSizes.size &&
            tensors.indices.all { index -> tensors[index].size == expectedSizes[index] }

    private fun copyGradientTensors(
         values: FloatArray,
        destinations: Array<FloatArray>,
        expectedSizes: IntArray
    ) {
        require(destinations.size == expectedSizes.size) {
            "Gradient destination tensor count does not match the resident accumulator."
        }
        val expectedCount = expectedSizes.fold(0L) { total, size ->
            Math.addExact(total, size.toLong())
        }
        require(expectedCount == values.size.toLong()) {
            "Gradient destination element count does not match the resident accumulator."
        }
        var offset = 0
        for (index in destinations.indices) {
            val destination = destinations[index]
            require(destination.size == expectedSizes[index]) {
                "Gradient destination shape does not match the resident accumulator at index $index."
            }
            System.arraycopy(values, offset, destination, 0, destination.size)
            offset += destination.size
        }
    }

    /**
     * Retires an accumulator in bookkeeping without touching native memory. Only valid
     * when its arena allocations are already gone (arena cleared, runtime destroyed or
     * unavailable); the `released` guards then fail any later use cleanly.
     */
    private fun markResidentGradientAccumulatorReleased(
        accumulator: VulkanResidentGradientAccumulator
    ) {
        accumulator.activeReleased = true
        accumulator.spareReleased = true
        accumulator.stagingReleased = true
        accumulator.pendingReleaseAllocations.clear()
        accumulator.released = true
        liveResidentGradientAccumulators.remove(accumulator)
    }

    /**
     * Discards every tracked resident gradient accumulator alongside the residency
     * registries. With [releaseAllocations] each allocation is released natively and a
     * failure is reported so the caller can escalate to an arena clear or teardown;
     * without it the native allocations are already gone and only the bookkeeping is
     * retired.
     */
    private fun discardTrackedResidentGradientAccumulators(
        releaseAllocations: Boolean
    ): Boolean {
        if (liveResidentGradientAccumulators.isEmpty()) return true
        var allReleased = true
        val snapshot = liveResidentGradientAccumulators.toList()
        for (accumulator in snapshot) {
            if (releaseAllocations) {
                if (!releaseResidentGradientAccumulator(accumulator)) {
                    allReleased = false
                }
            } else {
                markResidentGradientAccumulatorReleased(accumulator)
            }
        }
        return allReleased
    }

    private fun releaseResidentGradientAccumulator(
        accumulator: VulkanResidentGradientAccumulator
    ): Boolean {
        if (accumulator.released) {
            liveResidentGradientAccumulators.remove(accumulator)
            return true
        }

        while (accumulator.pendingReleaseAllocations.isNotEmpty()) {
            val pending = accumulator.pendingReleaseAllocations.first()
            if (!releaseGpuTensor(pending)) return false
            accumulator.pendingReleaseAllocations.removeAt(0)
        }

        if (!accumulator.stagingReleased) {
            accumulator.stagingReleased = releaseGpuTensor(accumulator.staging)
            if (!accumulator.stagingReleased) return false
        }
        if (!accumulator.spareReleased) {
            accumulator.spareReleased = releaseGpuTensor(accumulator.spare)
            if (!accumulator.spareReleased) return false
        }
        if (!accumulator.activeReleased) {
            accumulator.activeReleased = releaseGpuTensor(accumulator.active)
            if (!accumulator.activeReleased) return false
        }

        accumulator.released = true
        liveResidentGradientAccumulators.remove(accumulator)
        return true
    }

    private fun verifyResidentAdamwKernel(): Boolean {
        val initialParameters = arrayOf(
            floatArrayOf(1.0f, -2.0f, 0.5f, -0.25f, 0.125f),
            floatArrayOf(-0.75f, 0.375f, 1.25f, -1.5f, 0.875f, -0.25f)
        )
        // Active gradient prefixes deliberately differ from persistent tensor capacity.
        val initialGradients = arrayOf(
            floatArrayOf(0.1f, -0.2f, 0.05f, 0.025f),
            floatArrayOf(-0.03f, 0.04f, 0.07f)
        )
        val initialFirstMoments = arrayOf(
            floatArrayOf(0.01f, -0.02f, 0.03f, -0.04f, 0.005f),
            floatArrayOf(-0.01f, 0.02f, -0.03f, 0.04f, -0.005f, 0.006f)
        )
        val initialSecondMoments = arrayOf(
            floatArrayOf(0.001f, 0.002f, 0.003f, 0.004f, 0.0005f),
            floatArrayOf(0.0005f, 0.0004f, 0.0007f, 0.0009f, 0.0002f, 0.0003f)
        )
        val weightDecayByGroup = floatArrayOf(0.0f, 0.013f)
        val parameterBytes = Array(initialParameters.size) { index ->
            Float16ParameterArray.fromFloatArray(initialParameters[index]).toLittleEndianByteArray()
        }
        val firstMomentBytes = Array(initialFirstMoments.size) { index ->
            encodeFloatArray(initialFirstMoments[index])
        }
        val secondMomentBytes = Array(initialSecondMoments.size) { index ->
            encodeFloatArray(initialSecondMoments[index])
        }
        val activeElementCounts = IntArray(initialGradients.size) { index ->
            initialGradients[index].size
        }
        val totalGradientElements = initialParameters.fold(0) { total, parameter ->
            Math.addExact(total, parameter.size)
        }
        val flattenedGradients = FloatArray(totalGradientElements)
        var gradientCursor = 0
        for (group in initialGradients.indices) {
            val gradient = initialGradients[group]
            System.arraycopy(gradient, 0, flattenedGradients, gradientCursor, gradient.size)
            // Match the M4 accumulator layout: one padded segment per full parameter tensor.
            gradientCursor = Math.addExact(gradientCursor, initialParameters[group].size)
        }
        val gradientBytes = encodeFloatArray(flattenedGradients)

        val parameterAllocations = ArrayList<VulkanTensorAllocation>(initialParameters.size)
        val candidateParameterAllocations =
            ArrayList<VulkanTensorAllocation>(initialParameters.size)
        val firstMomentAllocations = ArrayList<VulkanTensorAllocation>(initialParameters.size)
        val secondMomentAllocations = ArrayList<VulkanTensorAllocation>(initialParameters.size)
        val candidateFirstMomentAllocations =
            ArrayList<VulkanTensorAllocation>(initialParameters.size)
        val candidateSecondMomentAllocations =
            ArrayList<VulkanTensorAllocation>(initialParameters.size)
        var gradientAllocation: VulkanTensorAllocation? = null
        var statusAllocation: VulkanTensorAllocation? = null
        var verified = false

        try {
            for (group in initialParameters.indices) {
                parameterAllocations.add(
                    allocateGpuTensor(parameterBytes[group].size.toLong()) ?: return false
                )
                candidateParameterAllocations.add(
                    allocateGpuTensor(parameterBytes[group].size.toLong()) ?: return false
                )
                val momentBytes = firstMomentBytes[group].size.toLong()
                firstMomentAllocations.add(allocateGpuTensor(momentBytes) ?: return false)
                secondMomentAllocations.add(allocateGpuTensor(secondMomentBytes[group].size.toLong()) ?: return false)
                candidateFirstMomentAllocations.add(allocateGpuTensor(momentBytes) ?: return false)
                candidateSecondMomentAllocations.add(
                    allocateGpuTensor(secondMomentBytes[group].size.toLong()) ?: return false
                )
            }
            val gradientTensor = allocateGpuTensor(gradientBytes.size.toLong()) ?: return false
            gradientAllocation = gradientTensor
            val statusTensor = allocateGpuTensor(Int.SIZE_BYTES.toLong()) ?: return false
            statusAllocation = statusTensor

            for (group in initialParameters.indices) {
                if (!uploadGpuTensor(parameterAllocations[group], parameterBytes[group]) ||
                    !uploadGpuTensor(firstMomentAllocations[group], firstMomentBytes[group]) ||
                    !uploadGpuTensor(secondMomentAllocations[group], secondMomentBytes[group])) {
                    return false
                }
            }
            if (!uploadGpuTensor(gradientTensor, gradientBytes) ||
                !uploadGpuTensor(statusTensor, ByteArray(Int.SIZE_BYTES))) {
                return false
            }

            val learningRate = 0.01f
            val beta1 = 0.9f
            val beta2 = 0.999f
            val biasCorrection1 = 0.1f
            val biasCorrection2 = 0.001f
            val epsilon = 1.0e-8f
            val gradientClipFactor = 0.75f

            fun dispatch(): Boolean = nativeApplyResidentAdamWUpdate(
                parameterHandles = LongArray(parameterAllocations.size) { index ->
                    parameterAllocations[index].nativeHandle
                },
                candidateParameterHandles = LongArray(candidateParameterAllocations.size) { index ->
                    candidateParameterAllocations[index].nativeHandle
                },
                gradientHandle = gradientTensor.nativeHandle,
                firstMomentHandles = LongArray(firstMomentAllocations.size) { index ->
                    firstMomentAllocations[index].nativeHandle
                },
                secondMomentHandles = LongArray(secondMomentAllocations.size) { index ->
                    secondMomentAllocations[index].nativeHandle
                },
                candidateFirstMomentHandles = LongArray(candidateFirstMomentAllocations.size) { index ->
                    candidateFirstMomentAllocations[index].nativeHandle
                },
                candidateSecondMomentHandles = LongArray(candidateSecondMomentAllocations.size) { index ->
                    candidateSecondMomentAllocations[index].nativeHandle
                },
                activeElementCounts = activeElementCounts,
                weightDecayByGroup = weightDecayByGroup,
                learningRate = learningRate,
                beta1 = beta1,
                beta2 = beta2,
                biasCorrection1 = biasCorrection1,
                biasCorrection2 = biasCorrection2,
                epsilon = epsilon,
                gradientClipFactor = gradientClipFactor,
                statusHandle = statusTensor.nativeHandle
            )

            if (!dispatch()) return false

            val statusBytes = ByteArray(Int.SIZE_BYTES)
            if (!downloadGpuTensor(statusTensor, statusBytes) ||
                ByteBuffer.wrap(statusBytes).order(ByteOrder.LITTLE_ENDIAN).int != 0) {
                return false
            }

            val expectedParameters = Array(initialParameters.size) { group ->
                initialParameters[group].copyOf()
            }
            val expectedFirstMoments = Array(initialParameters.size) { group ->
                initialFirstMoments[group].copyOf()
            }
            val expectedSecondMoments = Array(initialParameters.size) { group ->
                initialSecondMoments[group].copyOf()
            }
            for (group in initialParameters.indices) {
                for (element in initialGradients[group].indices) {
                    val gradient = initialGradients[group][element] * gradientClipFactor
                    val firstMoment =
                        beta1 * initialFirstMoments[group][element] + (1.0f - beta1) * gradient
                    val secondMoment =
                        beta2 * initialSecondMoments[group][element] +
                            (1.0f - beta2) * gradient * gradient
                    val correctedFirst = firstMoment / biasCorrection1
                    val correctedSecond = secondMoment / biasCorrection2
                    val update = correctedFirst / (kotlin.math.sqrt(correctedSecond) + epsilon) +
                        weightDecayByGroup[group] * initialParameters[group][element]
                    expectedFirstMoments[group][element] = firstMoment
                    expectedSecondMoments[group][element] = secondMoment
                    expectedParameters[group][element] = Float16Codec.toFloat(
                        Float16Codec.fromFiniteFloat(
                            initialParameters[group][element] - learningRate * update
                        )
                    )
                }
            }

            val validCandidateParameterBytes = Array(initialParameters.size) { group ->
                ByteArray(parameterBytes[group].size)
            }
            val validCandidateFirstBytes = Array(initialParameters.size) { group ->
                ByteArray(firstMomentBytes[group].size)
            }
            val validCandidateSecondBytes = Array(initialParameters.size) { group ->
                ByteArray(secondMomentBytes[group].size)
            }
            for (group in initialParameters.indices) {
                val updatedParameterBytes = validCandidateParameterBytes[group]
                val unchangedActiveParameterBytes = ByteArray(parameterBytes[group].size)
                if (!downloadGpuTensor(
                        candidateParameterAllocations[group],
                        updatedParameterBytes
                    ) ||
                    !downloadGpuTensor(
                        parameterAllocations[group],
                        unchangedActiveParameterBytes
                    ) ||
                    !unchangedActiveParameterBytes.contentEquals(parameterBytes[group]) ||
                    !downloadGpuTensor(
                        candidateFirstMomentAllocations[group],
                        validCandidateFirstBytes[group]
                    ) ||
                    !downloadGpuTensor(
                        candidateSecondMomentAllocations[group],
                        validCandidateSecondBytes[group]
                    )) {
                    return false
                }

                val actualParameters = Float16ParameterArray(initialParameters[group].size)
                actualParameters.copyPrefixFromLittleEndianByteArray(
                    updatedParameterBytes,
                    initialParameters[group].size
                )
                if (!approximatelyEqual(
                        actualParameters.toFloatArray(),
                        expectedParameters[group],
                        2.0e-3f
                    ) ||
                    !approximatelyEqual(
                        decodeFloatArray(validCandidateFirstBytes[group]),
                        expectedFirstMoments[group],
                        1.0e-6f
                    ) ||
                    !approximatelyEqual(
                        decodeFloatArray(validCandidateSecondBytes[group]),
                        expectedSecondMoments[group],
                        1.0e-7f
                    )) {
                    return false
                }

                // The source FP32 moments remain untouched until the host-side registry swap.
                val activeFirstBytes = ByteArray(firstMomentBytes[group].size)
                val activeSecondBytes = ByteArray(secondMomentBytes[group].size)
                if (!downloadGpuTensor(firstMomentAllocations[group], activeFirstBytes) ||
                    !downloadGpuTensor(secondMomentAllocations[group], activeSecondBytes) ||
                    !activeFirstBytes.contentEquals(firstMomentBytes[group]) ||
                    !activeSecondBytes.contentEquals(secondMomentBytes[group])) {
                    return false
                }
            }

            // Invalid input must not mutate active parameters, source moments, or candidates.
            // Bypass the finite-value encoder only for this negative native-kernel test:
            // the GPU kernel, not the host staging helper, must reject the NaN payload.
            val invalidGradientBytes = gradientBytes.copyOf()
            ByteBuffer.wrap(invalidGradientBytes)
                .order(ByteOrder.LITTLE_ENDIAN)
                .putFloat(0, Float.NaN)
            if (!uploadGpuTensor(gradientTensor, invalidGradientBytes) ||
                !uploadGpuTensor(statusTensor, ByteArray(Int.SIZE_BYTES))) {
                return false
            }
            for (group in parameterAllocations.indices) {
                if (!uploadGpuTensor(parameterAllocations[group], parameterBytes[group])) {
                    return false
                }
            }
            if (!dispatch()) return false

            val rejectedStatus = ByteArray(Int.SIZE_BYTES)
            if (!downloadGpuTensor(statusTensor, rejectedStatus) ||
                ByteBuffer.wrap(rejectedStatus).order(ByteOrder.LITTLE_ENDIAN).int == 0) {
                return false
            }
            for (group in parameterAllocations.indices) {
                val rejectedParameters = ByteArray(parameterBytes[group].size)
                val rejectedCandidateParameters = ByteArray(parameterBytes[group].size)
                val rejectedFirstMoments = ByteArray(firstMomentBytes[group].size)
                val rejectedSecondMoments = ByteArray(secondMomentBytes[group].size)
                val rejectedCandidateFirst = ByteArray(firstMomentBytes[group].size)
                val rejectedCandidateSecond = ByteArray(secondMomentBytes[group].size)
                if (!downloadGpuTensor(parameterAllocations[group], rejectedParameters) ||
                    !downloadGpuTensor(
                        candidateParameterAllocations[group],
                        rejectedCandidateParameters
                    ) ||
                    !downloadGpuTensor(firstMomentAllocations[group], rejectedFirstMoments) ||
                    !downloadGpuTensor(secondMomentAllocations[group], rejectedSecondMoments) ||
                    !downloadGpuTensor(
                        candidateFirstMomentAllocations[group], rejectedCandidateFirst
                    ) ||
                    !downloadGpuTensor(
                        candidateSecondMomentAllocations[group], rejectedCandidateSecond
                    ) ||
                    !rejectedParameters.contentEquals(parameterBytes[group]) ||
                    !rejectedCandidateParameters.contentEquals(parameterBytes[group]) ||
                    !rejectedFirstMoments.contentEquals(firstMomentBytes[group]) ||
                    !rejectedSecondMoments.contentEquals(secondMomentBytes[group]) ||
                    !rejectedCandidateFirst.contentEquals(firstMomentBytes[group]) ||
                    !rejectedCandidateSecond.contentEquals(secondMomentBytes[group])) {
                    return false
                }
            }
            verified = true
        } catch (error: LinkageError) {
            ArgusDiagnosticBridge.error(TAG, "Resident AdamW JNI self-check entry point is unavailable.", error)
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident AdamW numerical self-check failed.", error)
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Resident AdamW self-check exceeded the available memory budget.", error)
        } finally {
            var cleanupSucceeded = true
            val temporary = ArrayList<VulkanTensorAllocation>(
                parameterAllocations.size +
                    candidateParameterAllocations.size +
                    firstMomentAllocations.size +
                    secondMomentAllocations.size +
                    candidateFirstMomentAllocations.size +
                    candidateSecondMomentAllocations.size + 2
            )
            statusAllocation?.let { temporary.add(it) }
            gradientAllocation?.let { temporary.add(it) }
            temporary.addAll(candidateSecondMomentAllocations.asReversed())
            temporary.addAll(candidateFirstMomentAllocations.asReversed())
            temporary.addAll(secondMomentAllocations.asReversed())
            temporary.addAll(firstMomentAllocations.asReversed())
            temporary.addAll(candidateParameterAllocations.asReversed())
            temporary.addAll(parameterAllocations.asReversed())
            for (allocation in temporary) {
                if (!releaseGpuTensor(allocation)) cleanupSucceeded = false
            }
            if (!cleanupSucceeded) {
                verified = false
                ArgusDiagnosticBridge.error(TAG, "Resident AdamW self-check could not release all temporary tensors.")
                val arenaCleared = runCatching { nativeClearTensorArena() }.getOrDefault(false)
                if (arenaCleared) {
                    tensorResidencyRegistry.invalidate(releaseAllocations = false)
                    activationTensorRegistry.invalidate(releaseAllocations = false)
                    optimizerStateResidencyRegistry.invalidate(releaseAllocations = false)
                    refreshTensorArenaMetrics()
                } else {
                    ArgusDiagnosticBridge.error(TAG, "Vulkan arena cleanup failed after the resident AdamW self-check.")
                    runCatching { destroyNativeRuntime() }
                        .onFailure { error ->
                            ArgusDiagnosticBridge.error(TAG, "Vulkan teardown after AdamW self-check cleanup failure failed.", error)
                        }
                }
            }
        }
        return verified
    }

    private fun verifyResidentBackwardKernels(): Boolean {
        val rows = 5
        val vocabulary = 19
        val logits = FloatArray(rows * vocabulary) { index ->
            (((index * 13) % 37) - 18) * 0.03125f
        }
        val targets = floatArrayOf(2f, 7f, 3f, 16f, 0f)
        val gradients = FloatArray(logits.size)
        val rowStatistics = FloatArray(rows * 2)
        if (!tryRunResidentBackward(
                operation = BACKWARD_CROSS_ENTROPY,
                rows = rows,
                columns = vocabulary,
                sequenceLength = 1,
                inputs = arrayOf(logits, targets, null),
                outputs = arrayOf(gradients, rowStatistics)
            )) {
            return false
        }

        val expectedGradients = FloatArray(logits.size)
        val expectedStatistics = FloatArray(rows * 2)
        for (row in 1 until rows - 1) {
            var maximum = -Float.MAX_VALUE
            for (column in 0 until vocabulary) {
                maximum = maxOf(maximum, logits[row * vocabulary + column])
            }
            var sum = 0.0
            for (column in 0 until vocabulary) {
                sum += kotlin.math.exp((logits[row * vocabulary + column] - maximum).toDouble())
            }
            val logSumExp = maximum + kotlin.math.ln(sum).toFloat()
            expectedStatistics[row * 2] =
                logSumExp - logits[row * vocabulary + targets[row].toInt()]
            expectedStatistics[row * 2 + 1] = 1.0f
            for (column in 0 until vocabulary) {
                val probability = kotlin.math.exp(
                    (logits[row * vocabulary + column] - maximum).toDouble()
                ).toFloat() / sum.toFloat()
                expectedGradients[row * vocabulary + column] =
                    probability - if (column == targets[row].toInt()) 1.0f else 0.0f
            }
        }
        if (!approximatelyEqual(gradients, expectedGradients, 2.0e-4f) ||
            !approximatelyEqual(rowStatistics, expectedStatistics, 2.0e-4f)) {
            return false
        }

        val activationRows = 3
        val activationColumns = 73
        val count = activationRows * activationColumns
        val gate = FloatArray(count) { index -> (((index * 7) % 31) - 15) * 0.125f }
        val up = FloatArray(count) { index -> (((index * 5) % 29) - 14) * 0.0625f }
        val activationGradient = FloatArray(count) { index -> (((index * 3) % 19) - 9) * 0.125f }
        val gateGradient = FloatArray(count)
        val upGradient = FloatArray(count)
        if (!tryRunResidentBackward(
                operation = BACKWARD_SWIGLU,
                rows = activationRows,
                columns = activationColumns,
                sequenceLength = 0,
                inputs = arrayOf(gate, up, activationGradient),
                outputs = arrayOf(gateGradient, upGradient)
            )) {
            return false
        }
        val expectedGateGradient = FloatArray(count)
        val expectedUpGradient = FloatArray(count)
        for (index in 0 until count) {
            val sigmoid = 1.0f / (1.0f + kotlin.math.exp(-gate[index]))
            val swish = gate[index] * sigmoid
            expectedGateGradient[index] = activationGradient[index] * up[index] *
                sigmoid * (1.0f + gate[index] * (1.0f - sigmoid))
            expectedUpGradient[index] = activationGradient[index] * swish
        }
        if (!approximatelyEqual(gateGradient, expectedGateGradient, 2.0e-4f) ||
            !approximatelyEqual(upGradient, expectedUpGradient, 2.0e-4f)) {
            return false
        }

        val normRows = 4
        val normColumns = 73
        val normCount = normRows * normColumns
        val normInputs = FloatArray(normCount) { index -> (((index * 11) % 47) - 23) * 0.0625f }
        val normIncoming = FloatArray(normCount) { index -> (((index * 5) % 31) - 15) * 0.125f }
        val normOutput = FloatArray(normCount)
        if (!tryRunResidentBackward(
                operation = BACKWARD_LAYER_NORM,
                rows = normRows,
                columns = normColumns,
                sequenceLength = 0,
                inputs = arrayOf(normIncoming, normInputs, null),
                outputs = arrayOf(normOutput)
            )) {
            return false
        }
        val expectedNormOutput = FloatArray(normCount)
        for (row in 0 until normRows) {
            var mean = 0.0
            for (column in 0 until normColumns) {
                mean += normInputs[row * normColumns + column]
            }
            mean /= normColumns
            var variance = 0.0
            for (column in 0 until normColumns) {
                val difference = normInputs[row * normColumns + column] - mean
                variance += difference * difference
            }
            variance /= normColumns
            val inverse = 1.0 / kotlin.math.sqrt(variance + 1.0e-5)
            var sumGradient = 0.0
            var sumNormalizedGradient = 0.0
            for (column in 0 until normColumns) {
                val normalized = (normInputs[row * normColumns + column] - mean) * inverse
                val incoming = normIncoming[row * normColumns + column]
                sumGradient += incoming
                sumNormalizedGradient += incoming * normalized
            }
            for (column in 0 until normColumns) {
                val index = row * normColumns + column
                val normalized = (normInputs[index] - mean) * inverse
                expectedNormOutput[index] = (inverse * (
                    normIncoming[index] - sumGradient / normColumns -
                        normalized * sumNormalizedGradient / normColumns
                    )).toFloat()
            }
        }
        if (!approximatelyEqual(normOutput, expectedNormOutput, 3.0e-4f)) return false

        val sequenceLength = 5
        val headCount = 2
        val softmaxRows = headCount * (sequenceLength - 1)
        val probabilities = FloatArray(softmaxRows * sequenceLength)
        val probabilityGradients = FloatArray(probabilities.size)
        for (row in 0 until softmaxRows) {
            val query = row % (sequenceLength - 1)
            var normalizer = 0.0f
            for (key in 0..query) {
                val value = ((row * 5 + key * 3) % 13 + 1).toFloat()
                probabilities[row * sequenceLength + key] = value
                normalizer += value
            }
            for (key in 0..query) {
                val index = row * sequenceLength + key
                probabilities[index] /= normalizer
            }
            for (key in 0 until sequenceLength) {
                probabilityGradients[row * sequenceLength + key] =
                    (((row * 7 + key * 11) % 17) - 8) * 0.0625f
            }
        }
        val scoreGradients = FloatArray(probabilities.size)
        val attentionScale = 0.5f
        if (!tryRunResidentBackward(
                operation = BACKWARD_CAUSAL_SOFTMAX,
                rows = softmaxRows,
                columns = sequenceLength,
                sequenceLength = sequenceLength,
                inputs = arrayOf(probabilities, probabilityGradients, null),
                outputs = arrayOf(scoreGradients),
                scale = attentionScale
            )) {
            return false
        }
        val expectedScoreGradients = FloatArray(probabilities.size)
        for (row in 0 until softmaxRows) {
            val query = row % (sequenceLength - 1)
            var weighted = 0.0f
            for (key in 0..query) {
                val index = row * sequenceLength + key
                weighted += probabilities[index] * probabilityGradients[index]
            }
            for (key in 0..query) {
                val index = row * sequenceLength + key
                expectedScoreGradients[index] =
                    probabilities[index] * (probabilityGradients[index] - weighted) * attentionScale
            }
        }
        if (!approximatelyEqual(scoreGradients, expectedScoreGradients, 3.0e-4f)) return false

        val vocabularySize = 17
        val modelWidth = 13
        val tokenIds = floatArrayOf(3f, 3f, 11f, 1f, 3f)
        val embeddingInput = FloatArray(tokenIds.size * modelWidth) { index ->
            (((index * 7) % 23) - 11) * 0.03125f
        }
        val embeddingGradient = FloatArray(vocabularySize * modelWidth)
        if (!tryRunResidentBackward(
                operation = BACKWARD_EMBEDDING_GRADIENT,
                rows = vocabularySize,
                columns = modelWidth,
                sequenceLength = tokenIds.size,
                inputs = arrayOf(tokenIds, embeddingInput, null),
                outputs = arrayOf(embeddingGradient),
                scale = 1.75f
            )) {
            return false
        }
        val expectedEmbeddingGradient = FloatArray(embeddingGradient.size)
        for (tokenIndex in tokenIds.indices) {
            val token = tokenIds[tokenIndex].toInt()
            for (dimension in 0 until modelWidth) {
                expectedEmbeddingGradient[token * modelWidth + dimension] +=
                    embeddingInput[tokenIndex * modelWidth + dimension] * 1.75f
            }
        }
        return approximatelyEqual(embeddingGradient, expectedEmbeddingGradient, 3.0e-4f)
    }

    private fun verifyResidentGradientAccumulator(): Boolean {
        val capacityA = 4_100
        val capacityB = 103
        val firstA = FloatArray(4_096) { index ->
            (((index * 17) % 29) - 14) * 0.0078125f
        }
        val firstB = FloatArray(101) { index ->
            (((index * 11) % 19) - 9) * 0.015625f
        }
        val secondA = FloatArray(capacityA) { index ->
            (((index * 7) % 23) - 11) * 0.00390625f
        }
        val secondB = FloatArray(capacityB) { index ->
            (((index * 5) % 17) - 8) * 0.0078125f
        }
        val shapes = arrayOf(FloatArray(capacityA), FloatArray(capacityB))
        val accumulator = createResidentGradientAccumulator(shapes) ?: return false

        var verified = false
        var cleanupSucceeded = false
        try {
            val output = Array(shapes.size) { index -> FloatArray(shapes[index].size) }
            val operationsSucceeded =
                accumulateResidentGradients(accumulator, arrayOf(firstA, firstB), 2) &&
                    accumulateResidentGradients(accumulator, arrayOf(secondA, secondB), 1) &&
                    finalizeResidentGradientAccumulator(
                        accumulator,
                        output,
                        targetCount = 3,
                        maximumGradientNorm = 0.5f
                    )

            if (operationsSucceeded) {
                var squaredNorm = 0.0
                val expected = arrayOf(
                    FloatArray(capacityA),
                    FloatArray(capacityB)
                )
                for (index in 0 until capacityA) {
                    val first = if (index < firstA.size) firstA[index] else 0.0f
                    val mean = (first * 2.0f + secondA[index]) / 3.0f
                    expected[0][index] = mean
                    squaredNorm += mean.toDouble() * mean.toDouble()
                }
                for (index in 0 until capacityB) {
                    val first = if (index < firstB.size) firstB[index] else 0.0f
                    val mean = (first * 2.0f + secondB[index]) / 3.0f
                    expected[1][index] = mean
                    squaredNorm += mean.toDouble() * mean.toDouble()
                }

                val norm = kotlin.math.sqrt(squaredNorm)
                val clipFactor = if (norm > 0.5) 0.5 / (norm + 1.0e-6) else 1.0
                for (tensorIndex in expected.indices) {
                    for (elementIndex in expected[tensorIndex].indices) {
                        expected[tensorIndex][elementIndex] =
                            (expected[tensorIndex][elementIndex] * clipFactor).toFloat()
                    }
                }
                verified = approximatelyEqual(output[0], expected[0], 4.0e-4f) &&
                    approximatelyEqual(output[1], expected[1], 4.0e-4f)
            }
        } catch (error: LinkageError) {
            ArgusDiagnosticBridge.error(TAG, "Resident gradient accumulator self-check JNI entry point is unavailable.", error)
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident gradient accumulator numerical self-check failed.", error)
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Resident gradient accumulator self-check exceeded available memory.", error)
        } finally {
            cleanupSucceeded = accumulator.released ||
                releaseResidentGradientAccumulator(accumulator)
            if (!cleanupSucceeded && isRuntimeAvailable) {
                ArgusDiagnosticBridge.error(TAG, "Resident gradient accumulator self-check could not release all Vulkan buffers.")
                runCatching { destroyNativeRuntime() }
                    .onFailure { error ->
                        ArgusDiagnosticBridge.error(TAG, "Vulkan teardown after gradient self-check cleanup failure failed.", error)
                    }
            }
        }
        return verified && cleanupSucceeded
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
                ArgusDiagnosticBridge.error(TAG, "Persistent batched FP16 Vulkan self-check entry point is unavailable.", error)
                return false
            } catch (error: RuntimeException) {
                ArgusDiagnosticBridge.error(TAG, "Persistent batched FP16 Vulkan self-check failed.", error)
                return false
            } catch (error: OutOfMemoryError) {
                ArgusDiagnosticBridge.warning(TAG, "Persistent batched FP16 Vulkan self-check exceeded available memory.", error)
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
                ArgusDiagnosticBridge.error(TAG, "Persistent single-matrix FP16 Vulkan self-check entry point is unavailable.", error)
                return false
            } catch (error: RuntimeException) {
                ArgusDiagnosticBridge.error(TAG, "Persistent single-matrix FP16 Vulkan self-check failed.", error)
                return false
            } catch (error: OutOfMemoryError) {
                ArgusDiagnosticBridge.warning(TAG, "Persistent single-matrix FP16 Vulkan self-check exceeded available memory.", error)
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
            ArgusDiagnosticBridge.error(TAG, "Resident Vulkan linear JNI entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident Vulkan linear dispatch failed.", error)
            false
        }
        if (applied) {
            refreshGpuPerformanceMetrics()
        } else {
            residentActivationLinearReady = false
            _currentBackend = "Kotlin CPU (Resident linear fallback)"
            _runtimeStatus.value =
                "Vulkan runtime verified — resident linear dispatch failed; this path uses Kotlin fallback"
            ArgusDiagnosticBridge.warning(TAG, "Resident activation-linear dispatch failed; this M3 path is disabled for this runtime.")
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
            ArgusDiagnosticBridge.error(TAG, "Resident Vulkan batched-linear JNI entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident Vulkan batched-linear dispatch failed.", error)
            false
        }
        if (applied) {
            refreshGpuPerformanceMetrics()
        } else {
            residentActivationLinearReady = false
            _currentBackend = "Kotlin CPU (Resident batched linear fallback)"
            _runtimeStatus.value =
                "Vulkan runtime verified — resident batched-linear dispatch failed; this path uses Kotlin fallback"
            ArgusDiagnosticBridge.warning(TAG, "Resident Vulkan batched-linear dispatch failed; this resident M3 path is disabled for this runtime.")
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
        if (!isFp16ComputeAvailable || !ensureModelTensorResidencyReady()) {
            return false
        }

        val gpuStartNanos = System.nanoTime()
        val flattenedInput: FloatArray
        val flattenedOutput: FloatArray
        try {
            flattenedInput = FloatArray(inputElements.toInt())
            flattenedOutput = FloatArray(outputElements.toInt())
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Not enough heap memory to stage a persistent-weight Vulkan linear operation.", error)
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
            ArgusDiagnosticBridge.error(TAG, "Persistent-weight Vulkan FP16 entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Persistent-weight Vulkan FP16 dispatch failed.", error)
            false
        }

        if (!updated) {
            fp16DisabledAfterRuntimeFailure = true
            fp16RuntimeReady = false
            invalidateTensorResidency(releaseAllocations = true)
            _currentBackend = "Kotlin CPU (FP16 linear fallback)"
            _runtimeStatus.value =
                "Vulkan runtime verified — FP16 linear dispatch failed; this path uses Kotlin fallback"
            ArgusDiagnosticBridge.warning(TAG, "Persistent-weight FP16 dispatch failed; using the Kotlin CPU path.")
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
        AdaptiveComputePerformanceMonitor.recordGpu(
            AdaptiveComputeOperation.LINEAR,
            operationCount,
            System.nanoTime() - gpuStartNanos
        )
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
        if (!isFp16ComputeAvailable || !ensureModelTensorResidencyReady()) {
            return false
        }

        val gpuStartNanos = System.nanoTime()
        val flattenedInput: FloatArray
        val flattenedOutput: FloatArray
        try {
            flattenedInput = FloatArray(inputElementsLong.toInt())
            flattenedOutput = FloatArray(outputElementsLong.toInt())
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Not enough heap memory to stage FP16 batched linear activations.", error)
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
            ArgusDiagnosticBridge.error(TAG, "Persistent batched FP16 Vulkan entry point is unavailable.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Persistent batched FP16 Vulkan dispatch failed.", error)
            false
        }

        if (!updated) {
            fp16DisabledAfterRuntimeFailure = true
            fp16RuntimeReady = false
            invalidateTensorResidency(releaseAllocations = true)
            _currentBackend = "Kotlin CPU (FP16 batched linear fallback)"
            _runtimeStatus.value =
                "Vulkan runtime verified — FP16 batched linear dispatch failed; this path uses Kotlin fallback"
            ArgusDiagnosticBridge.warning(TAG,
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

        AdaptiveComputePerformanceMonitor.recordGpu(
            AdaptiveComputeOperation.LINEAR,
            operationCount,
            System.nanoTime() - gpuStartNanos
        )
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
            disabledAfterRuntimeFailure) {
            return false
        }

        val gpuStartNanos = System.nanoTime()
        val flattenedInput: FloatArray
        val flattenedWeights: FloatArray
        val flattenedOutput: FloatArray
        try {
            flattenedInput = FloatArray(inputElementCountLong.toInt())
            flattenedWeights = FloatArray(weightElementCountLong.toInt())
            flattenedOutput = FloatArray(outputElementCountLong.toInt())
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Not enough heap memory to stage a Vulkan batched-linear operation.", error)
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
            ArgusDiagnosticBridge.error(TAG, "Vulkan batched-linear entry point is unavailable; disabling the backend.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Vulkan batched-linear dispatch failed; disabling the backend.", error)
            false
        }

        if (!updated) {
            disabledAfterRuntimeFailure = true
            _runtimeStatus.value = "Kotlin CPU fallback — Vulkan linear dispatch failed"
            _currentBackend = "Kotlin CPU"
            runCatching { destroyNativeRuntime() }
                .onFailure { error ->
                    ArgusDiagnosticBridge.warning(TAG, "Failed to release Vulkan after a batched-linear dispatch failure.", error)
                }
            ArgusDiagnosticBridge.warning(TAG, "Vulkan batched-linear dispatch was not completed; using the Kotlin CPU path.")
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
        AdaptiveComputePerformanceMonitor.recordGpu(
            AdaptiveComputeOperation.LINEAR,
            operationCount,
            System.nanoTime() - gpuStartNanos
        )
        refreshGpuPerformanceMetrics()
        _currentBackend = "Vulkan Compute (Batched Linear)"
        return true
    }

    /**
     * Updates resident FP16 weights and FP32 AdamW moments in one Vulkan transaction.
     * Host parameters are read back only as the recovery/inference mirror; moments remain
     * resident and are not copied back after each update.
     */
    @Synchronized
    internal fun tryApplyResidentFp16AdamWUpdate(
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
        gradientClipFactor: Float,
        residentGradient: VulkanTensorAllocation? = null
    ): Boolean {
        refreshModelTensorResidencyBudget()
        if (!isResidentAdamWAvailable ||
            parameters.isEmpty() ||
            parameters.size != gradients.size ||
            parameters.size != firstMoments.size ||
            parameters.size != secondMoments.size ||
            parameters.size != weightDecayByGroup.size ||
            !learningRate.isFinite() || learningRate < 0.0f ||
            !beta1.isFinite() || beta1 < 0.0f || beta1 >= 1.0f ||
            !beta2.isFinite() || beta2 < 0.0f || beta2 >= 1.0f ||
            !biasCorrection1.isFinite() || biasCorrection1 <= 0.0f ||
            !biasCorrection2.isFinite() || biasCorrection2 <= 0.0f ||
            !epsilon.isFinite() || epsilon <= 0.0f ||
            !gradientClipFactor.isFinite() ||
            gradientClipFactor < 0.0f || gradientClipFactor > 1.0f ||
            !ensureModelTensorResidencyReady()) {
            return false
        }

        var totalActiveElements = 0L
        var totalGradientBytes = 0L
        var totalParameterBytes = 0L
        var totalMomentBytes = 0L
        for (index in parameters.indices) {
            val parameterSize = parameters[index].size
            val activeSize = gradients[index].size
            if (parameterSize <= 0 || activeSize <= 0 || activeSize > parameterSize ||
                firstMoments[index].size != parameterSize ||
                secondMoments[index].size != parameterSize ||
                !weightDecayByGroup[index].isFinite() ||
                weightDecayByGroup[index] < 0.0f ||
                (residentGradient == null &&
                    gradients[index].any { value -> !value.isFinite() }) ||
                (optimizerStateMirrorReliable &&
                    (firstMoments[index].any { value -> !value.isFinite() } ||
                        secondMoments[index].any { value -> !value.isFinite() }))) {
                return false
            }

            totalActiveElements = runCatching {
                Math.addExact(totalActiveElements, activeSize.toLong())
            }.getOrNull() ?: return false
            totalGradientBytes = runCatching {
                Math.addExact(
                    totalGradientBytes,
                    Math.multiplyExact(parameterSize.toLong(), Float.SIZE_BYTES.toLong())
                )
            }.getOrNull() ?: return false
            totalParameterBytes = runCatching {
                Math.addExact(
                    totalParameterBytes,
                    Math.multiplyExact(parameterSize.toLong(), Short.SIZE_BYTES.toLong())
                )
            }.getOrNull() ?: return false
            totalMomentBytes = runCatching {
                val bytesPerMoment = Math.multiplyExact(
                    parameterSize.toLong(),
                    Float.SIZE_BYTES.toLong()
                )
                Math.addExact(totalMomentBytes, Math.multiplyExact(bytesPerMoment, 2L))
            }.getOrNull() ?: return false
        }

        if (totalActiveElements > Int.MAX_VALUE.toLong() ||
            totalGradientBytes <= 0L ||
            totalGradientBytes > Int.MAX_VALUE.toLong()) {
            return false
        }
        // Active FP32 moments and their candidate replacements must both fit in
        // the optimizer-state reservation during the atomic swap window.
        val optimizerBudget = _optimizerStateResidencyBudgetBytes.value.coerceAtLeast(0L)
        val maximumDoubleBufferedBytes = if (optimizerBudget > Long.MAX_VALUE / 2L) {
            Long.MAX_VALUE
        } else {
            optimizerBudget * 2L
        }

        val modelBudget = _modelTensorResidencyBudgetBytes.value.coerceAtLeast(0L)
        val maximumDoubleBufferedModelBytes = if (modelBudget > Long.MAX_VALUE / 2L) {
            Long.MAX_VALUE
        } else {
            modelBudget * 2L
        }
        val requiredModelArenaBytes = runCatching {
            Math.addExact(tensorResidencyRegistry.residentByteCount(), totalParameterBytes)
        }.getOrNull() ?: return false
        if (requiredModelArenaBytes > maximumDoubleBufferedModelBytes) {
            Log.d(TAG, "Resident AdamW candidate parameters exceed the reserved double-buffer budget.")
            return false
        }
        val requiredMomentArenaBytes = runCatching {
            Math.addExact(
                optimizerStateResidencyRegistry.residentByteCount(),
                totalMomentBytes
            )
        }.getOrNull() ?: return false
        if (requiredMomentArenaBytes > maximumDoubleBufferedBytes) {
            Log.d(TAG, "Resident AdamW candidate moments exceed the reserved double-buffer budget.")
            return false
        }

        val runtime = Runtime.getRuntime()
        val availableHeapBytes = runtime.freeMemory() +
            (runtime.maxMemory() - runtime.totalMemory()).coerceAtLeast(0L)
        val stagingBytes = if (residentGradient == null) {
            runCatching {
                Math.multiplyExact(totalGradientBytes, 2L)
            }.getOrNull() ?: return false
        } else {
            0L
        }
        if (stagingBytes > availableHeapBytes / 2L) {
            Log.d(TAG, "Resident AdamW gradient staging exceeds the safe host-heap budget.")
            return false
        }

        val momentInventory = ArrayList<FloatArray>(parameters.size * 2)
        momentInventory.addAll(firstMoments)
        momentInventory.addAll(secondMoments)
        if (!optimizerStateResidencyRegistry.synchronizeAll(momentInventory)) {
            Log.d(TAG, "FP32 AdamW moments do not fit the current persistent residency budget.")
            return false
        }

        val residentParameters = ArrayList<VulkanTensorAllocation>(parameters.size)
        val residentFirstMoments = ArrayList<VulkanTensorAllocation>(parameters.size)
        val residentSecondMoments = ArrayList<VulkanTensorAllocation>(parameters.size)
        for (index in parameters.indices) {
            val parameterAllocation = acquireResidentModelParameterTensor(parameters[index])
                ?: return false
            residentParameters.add(parameterAllocation)
            val firstMomentAllocation = optimizerStateResidencyRegistry.acquire(firstMoments[index])
                ?: return false
            residentFirstMoments.add(firstMomentAllocation)
            val secondMomentAllocation = optimizerStateResidencyRegistry.acquire(secondMoments[index])
                ?: return false
            residentSecondMoments.add(secondMomentAllocation)
        }

        var gradientAllocation: VulkanTensorAllocation? = null
        var statusAllocation: VulkanTensorAllocation? = null
        var ownsGradientAllocation = residentGradient == null
        val candidateParameters = ArrayList<VulkanTensorAllocation>(parameters.size)
        val candidateFirstMoments = ArrayList<VulkanTensorAllocation>(parameters.size)
        val candidateSecondMoments = ArrayList<VulkanTensorAllocation>(parameters.size)
        var candidateParametersCommitted = false
        var candidateMomentsCommitted = false
        var partialRegistryPublicationFailure = false
        var gradientReleased = false
        var statusReleased = false
        var dispatchAttempted = false
        var teardownRequired = false
        var deviceCommitCompleted = false

        try {
            for (group in parameters.indices) {
                val parameterBytes = Math.multiplyExact(
                    parameters[group].size.toLong(),
                    Short.SIZE_BYTES.toLong()
                )
                val candidateParameter = allocateGpuTensor(parameterBytes) ?: return false
                candidateParameters.add(candidateParameter)

                val momentBytes = Math.multiplyExact(
                    parameters[group].size.toLong(),
                    Float.SIZE_BYTES.toLong()
                )
                val candidateFirst = allocateGpuTensor(momentBytes) ?: return false
                candidateFirstMoments.add(candidateFirst)
                val candidateSecond = allocateGpuTensor(momentBytes) ?: return false
                candidateSecondMoments.add(candidateSecond)
            }

            val gradientTensor = if (residentGradient == null) {
                val flattenedGradientElements = (
                    totalGradientBytes / Float.SIZE_BYTES.toLong()
                ).toInt()
                val flattenedGradients = FloatArray(flattenedGradientElements)
                var gradientCursor = 0
                for (index in gradients.indices) {
                    val gradient = gradients[index]
                    System.arraycopy(
                        gradient, 0, flattenedGradients, gradientCursor, gradient.size
                    )
                    // The next group begins after full model-tensor capacity, matching
                    // VulkanResidentGradientAccumulator and native AdamW offset arithmetic.
                    gradientCursor = Math.addExact(gradientCursor, parameters[index].size)
                }
                check(gradientCursor == flattenedGradients.size) {
                    "Staged AdamW gradient layout does not match the full parameter inventory."
                }
                val allocation = allocateGpuTensor(totalGradientBytes) ?: return false
                gradientAllocation = allocation
                if (!uploadGpuTensor(allocation, encodeFloatArray(flattenedGradients))) {
                    ArgusDiagnosticBridge.warning(TAG, "Resident AdamW could not upload staged gradient values.")
                    return false
                }
                ownsGradientAllocation = true
                allocation
            } else {
                if (residentGradient.byteCount < totalGradientBytes) {
                    ArgusDiagnosticBridge.warning(TAG, "The resident M4 gradient tensor is smaller than the AdamW inventory.")
                    return false
                }
                ownsGradientAllocation = false
                residentGradient
            }
            val statusTensor = allocateGpuTensor(Int.SIZE_BYTES.toLong()) ?: return false
            statusAllocation = statusTensor
            if (!uploadGpuTensor(statusTensor, ByteArray(Int.SIZE_BYTES))) {
                ArgusDiagnosticBridge.warning(TAG, "Resident AdamW could not initialize its validation flag.")
                return false
            }

            dispatchAttempted = true
            tensorResidencyRegistry.markDeviceMutationPending(parameters.toList())
            modelParameterMirrorReliable = false
            optimizerStateMirrorReliable = false
            val dispatched = try {
                nativeApplyResidentAdamWUpdate(
                    parameterHandles = LongArray(residentParameters.size) { index ->
                        residentParameters[index].nativeHandle
                    },
                    candidateParameterHandles = LongArray(candidateParameters.size) { index ->
                        candidateParameters[index].nativeHandle
                    },
                    gradientHandle = gradientTensor.nativeHandle,
                    firstMomentHandles = LongArray(residentFirstMoments.size) { index ->
                        residentFirstMoments[index].nativeHandle
                    },
                    secondMomentHandles = LongArray(residentSecondMoments.size) { index ->
                        residentSecondMoments[index].nativeHandle
                    },
                    candidateFirstMomentHandles = LongArray(candidateFirstMoments.size) { index ->
                        candidateFirstMoments[index].nativeHandle
                    },
                    candidateSecondMomentHandles = LongArray(candidateSecondMoments.size) { index ->
                        candidateSecondMoments[index].nativeHandle
                    },
                    activeElementCounts = IntArray(gradients.size) { index -> gradients[index].size },
                    weightDecayByGroup = weightDecayByGroup,
                    learningRate = learningRate,
                    beta1 = beta1,
                    beta2 = beta2,
                    biasCorrection1 = biasCorrection1,
                    biasCorrection2 = biasCorrection2,
                    epsilon = epsilon,
                    gradientClipFactor = gradientClipFactor,
                    statusHandle = statusTensor.nativeHandle
                )
            } catch (error: LinkageError) {
                ArgusDiagnosticBridge.error(TAG, "Resident AdamW JNI entry point is unavailable.", error)
                false
            } catch (error: RuntimeException) {
                ArgusDiagnosticBridge.error(TAG, "Resident AdamW dispatch failed.", error)
                false
            }
            if (!dispatched) {
                teardownRequired = true
                return false
            }

            // The native call has waited for the compute fence. Read the status through
            // the common Tensor Arena transfer path, which inserts the compute-to-copy
            // dependency and invalidates non-coherent staging memory before CPU access.
            val statusBytes = ByteArray(Int.SIZE_BYTES)
            if (!downloadGpuTensor(statusTensor, statusBytes)) {
                teardownRequired = true
                ArgusDiagnosticBridge.error(TAG, "Resident AdamW transaction status could not be read back.")
                return false
            }
            val transactionStatus = ByteBuffer.wrap(statusBytes)
                .order(ByteOrder.LITTLE_ENDIAN)
                .int
            if (transactionStatus != 0) {
                // The validation phase rejected the step. Active weights and moments
                // were not written; previous GPU updates may still leave host mirrors stale.
                ArgusDiagnosticBridge.error(TAG, "Resident AdamW rejected a non-finite or out-of-range update.")
                return false
            }

            // Release transient staging before publishing candidate moment allocations.
            if (ownsGradientAllocation) {
                if (!releaseGpuTensor(gradientTensor)) {
                    teardownRequired = true
                    return false
                }
                gradientReleased = true
            }
            if (!releaseGpuTensor(statusTensor)) {
                teardownRequired = true
                return false
            }
            statusReleased = true

            val expectedMomentAllocations = ArrayList<VulkanTensorAllocation>(parameters.size * 2)
            expectedMomentAllocations.addAll(residentFirstMoments)
            expectedMomentAllocations.addAll(residentSecondMoments)
            val candidateMomentAllocations = ArrayList<VulkanTensorAllocation>(parameters.size * 2)
            candidateMomentAllocations.addAll(candidateFirstMoments)
            candidateMomentAllocations.addAll(candidateSecondMoments)
            val parameterInventory = parameters.toList()
            val expectedParameterAllocations = residentParameters.toList()
            val candidateParameterAllocations = candidateParameters.toList()

            // Preflight both registries before publishing either side of the optimizer
            // transaction. The backend lock prevents another owner from replacing them
            // between these checks and the two allocation swaps.
            if (!tensorResidencyRegistry.canReplaceAll(
                    parameterInventory,
                    expectedParameterAllocations,
                    candidateParameterAllocations
                ) ||
                !optimizerStateResidencyRegistry.canReplaceAll(
                    momentInventory,
                    expectedMomentAllocations,
                    candidateMomentAllocations
                )) {
                teardownRequired = true
                ArgusDiagnosticBridge.error(TAG, "Resident AdamW candidate parameter/moment registry preflight failed.")
                return false
            }

            if (!tensorResidencyRegistry.commitReplacementAfterPreflight(
                    parameterInventory,
                    expectedParameterAllocations,
                    candidateParameterAllocations
                )) {
                teardownRequired = true
                ArgusDiagnosticBridge.error(TAG, "Resident AdamW could not atomically publish candidate FP16 parameters.")
                return false
            }
            candidateParametersCommitted = true

            if (!optimizerStateResidencyRegistry.commitReplacementAfterPreflight(
                    momentInventory,
                    expectedMomentAllocations,
                    candidateMomentAllocations
                )) {
                // Both registries were preflighted and reserved. A remaining failure
                // indicates a broken ownership invariant; abort training rather than
                // falling back with a mixed model/moment revision.
                partialRegistryPublicationFailure = true
                teardownRequired = true
                throw IllegalStateException(
                    "Resident AdamW could not atomically publish candidate FP32 moments."
                )
            }
            candidateMomentsCommitted = true
            deviceCommitCompleted = true

            // Only now are both registries on the same optimizer step. Retired allocations
            // may be released; failed releases remain accounted and are retried later.
            val oldParametersReleased = runCatching {
                tensorResidencyRegistry.finishReplacementAfterPreflight()
            }.getOrDefault(false)
            val oldMomentsReleased = runCatching {
                optimizerStateResidencyRegistry.finishReplacementAfterPreflight()
            }.getOrDefault(false)
            if (!oldParametersReleased || !oldMomentsReleased) {
                ArgusDiagnosticBridge.warning(TAG,
                    "Resident AdamW committed successfully; one or more old buffers remain tracked for deferred release."
                )
            }

            // Candidate FP16 parameters and FP32 moments are now authoritative together.
            // Host mirrors remain stale until an explicit CPU/inference/checkpoint boundary.
            optimizerStateMirrorReliable = false

            _currentBackend = "Vulkan Compute (Resident FP16 AdamW)"
            runCatching { refreshGpuPerformanceMetrics() }
            return true
        } catch (error: OutOfMemoryError) {
            ArgusDiagnosticBridge.warning(TAG, "Resident AdamW exceeded the available host memory budget.", error)
            if (deviceCommitCompleted || candidateMomentsCommitted) {
                teardownRequired = true
                return true
            }
            if (dispatchAttempted) teardownRequired = true
            return false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Resident AdamW transaction failed.", error)
            if (partialRegistryPublicationFailure) throw error
            if (deviceCommitCompleted || candidateMomentsCommitted) {
                teardownRequired = true
                return true
            }
            if (dispatchAttempted) teardownRequired = true
            return false
        } finally {
            if (ownsGradientAllocation &&
                !gradientReleased &&
                gradientAllocation != null &&
                isRuntimeAvailable) {
                if (releaseGpuTensor(gradientAllocation)) {
                    gradientReleased = true
                } else {
                    teardownRequired = true
                }
            }
            if (!statusReleased && statusAllocation != null && isRuntimeAvailable) {
                if (releaseGpuTensor(statusAllocation)) {
                    statusReleased = true
                } else {
                    teardownRequired = true
                }
            }

            if (isRuntimeAvailable) {
                if (!candidateParametersCommitted) {
                    for (allocation in candidateParameters.asReversed()) {
                        if (!releaseGpuTensor(allocation)) teardownRequired = true
                    }
                }
                if (!candidateMomentsCommitted) {
                    for (allocation in candidateSecondMoments.asReversed() +
                        candidateFirstMoments.asReversed()) {
                        if (!releaseGpuTensor(allocation)) teardownRequired = true
                    }
                }
            }

            if (teardownRequired && nativeReady) {
                // When M4 owns the gradient allocation, keep the disabled native runtime
                // alive long enough for the caller to recover that gradient. The wrapper
                // then recovers model/moments, releases optimizer residency, and tears down.
                if (residentGradient != null &&
                    !partialRegistryPublicationFailure &&
                    !deviceCommitCompleted) {
                    residentGradientRecoveryPending = true
                    disabledAfterRuntimeFailure = true
                    residentAdamwReady = false
                    _currentBackend = "Kotlin CPU — resident gradient recovery pending"
                    _runtimeStatus.value =
                        "Vulkan dispatch disabled — recovering resident M4 gradients before teardown"
                    ArgusDiagnosticBridge.error(TAG,
                        "Resident AdamW dispatch failed; native teardown is postponed until the M4 gradient has been recovered."
                    )
                } else {
                    runCatching { destroyNativeRuntime() }
                        .onFailure { error ->
                            ArgusDiagnosticBridge.error(TAG, "Vulkan teardown after a failed resident AdamW transaction failed.", error)
                        }
                }
            }
            if (deviceCommitCompleted && !isRuntimeAvailable) {
                _currentBackend = "Kotlin CPU — GPU AdamW committed; host mirror recovery required"
            }
        }
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
        gradientClipFactor: Float,
        residentGradient: VulkanTensorAllocation? = null
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

        if (tryApplyResidentFp16AdamWUpdate(
                parameters = parameters,
                gradients = gradients,
                firstMoments = firstMoments,
                secondMoments = secondMoments,
                weightDecayByGroup = weightDecayByGroup,
                learningRate = learningRate,
                beta1 = beta1,
                beta2 = beta2,
                biasCorrection1 = biasCorrection1,
                biasCorrection2 = biasCorrection2,
                epsilon = epsilon,
                gradientClipFactor = gradientClipFactor,
                residentGradient = residentGradient
            )
        ) {
            return true
        }

        // The resident transaction can be rejected before commit. If M4 supplied a
        // GPU-resident gradient accumulator, recover its normalized/clipped values before
        // either staged Vulkan or CPU AdamW reads the host gradient arrays.
        if (residentGradient != null) {
            val expectedGradientElements = gradients.fold(0L) { total, gradient ->
                Math.addExact(total, gradient.size.toLong())
            }
            val expectedGradientBytes = Math.multiplyExact(
                expectedGradientElements,
                Float.SIZE_BYTES.toLong()
            )
            check(expectedGradientBytes == residentGradient.byteCount) {
                "Resident M4 gradient size does not match the active optimizer tensor inventory."
            }
            check(expectedGradientBytes <= Int.MAX_VALUE.toLong()) {
                "Resident M4 gradient exceeds the host recovery array limit."
            }
            val gradientBytes = ByteArray(expectedGradientBytes.toInt())
            check(downloadGpuTensor(residentGradient, gradientBytes)) {
                "Resident M4 gradient could not be recovered for the optimizer fallback."
            }
            val recoveredGradients = decodeFloatArray(gradientBytes)
            check(recoveredGradients.all { value -> value.isFinite() }) {
                "Resident M4 gradient contains a non-finite value."
            }
            copyGradientTensors(
                recoveredGradients,
                gradients,
                IntArray(gradients.size) { index -> gradients[index].size }
            )
        }

        // The resident transaction can be rejected before commit while previous GPU
        // updates still leave the model mirror stale. Recover both sides before fallback.
        check(synchronizeTrainingStateMirrors()) {
            "GPU model or optimizer state could not be recovered before the fallback update."
        }
        check(invalidateOptimizerStateResidency()) {
            "GPU optimizer moments could not be synchronized before the fallback update."
        }
        if (residentGradientRecoveryPending) {
            check(destroyNativeRuntime()) {
                "Vulkan runtime teardown could not complete after resident-gradient recovery."
            }
            residentGradientRecoveryPending = false
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

        if (!isRuntimeAvailable || disabledAfterRuntimeFailure) {
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
                gradientClipFactor = gradientClipFactor,
                recordAdaptiveTiming = false
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
            ArgusDiagnosticBridge.warning(TAG, "Not enough heap memory to stage FP16 parameters for Vulkan AdamW; retaining CPU fallback.", error)
            false
        } catch (error: IllegalArgumentException) {
            ArgusDiagnosticBridge.error(TAG, "Vulkan AdamW returned invalid staged tensors; retaining CPU fallback.", error)
            false
        }
    }

    /**
     * Runs native AdamW only against disposable FP32 arrays created by the FP16 adapter.
     * The adapter publishes them to model and optimizer state only after every GPU page
     * completes successfully.
     */
    @Synchronized
    private fun tryApplyAdamWUpdate(
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
        gradientClipFactor: Float,
        recordAdaptiveTiming: Boolean = true
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
            disabledAfterRuntimeFailure) {
            _currentBackend = "Kotlin CPU"
            return false
        }

        val gpuStartNanos = System.nanoTime()
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
            ArgusDiagnosticBridge.error(TAG, "Vulkan optimizer entry point is unavailable; disabling the backend.", error)
            false
        } catch (error: RuntimeException) {
            ArgusDiagnosticBridge.error(TAG, "Vulkan optimizer dispatch failed; disabling the backend.", error)
            false
        }

        if (updated) {
            if (recordAdaptiveTiming) {
                AdaptiveComputePerformanceMonitor.recordGpu(
                    AdaptiveComputeOperation.ADAMW,
                    totalElements,
                    System.nanoTime() - gpuStartNanos
                )
            }
            refreshGpuPerformanceMetrics()
            _currentBackend = "Vulkan Compute (AdamW)"
            return true
        }

        disabledAfterRuntimeFailure = true
        _runtimeStatus.value = "Kotlin CPU fallback — Vulkan optimizer dispatch failed"
        _currentBackend = "Kotlin CPU"
        runCatching { destroyNativeRuntime() }
            .onFailure { error ->
                ArgusDiagnosticBridge.warning(TAG, "Failed to release the Vulkan runtime after a dispatch failure.", error)
            }
        ArgusDiagnosticBridge.warning(TAG, "Vulkan AdamW dispatch was not completed; using the Kotlin CPU optimizer.")
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

    private external fun nativeRunResidentSwiGlu(
        gateHandle: Long,
        upHandle: Long,
        outputHandle: Long,
        rows: Int,
        columns: Int
    ): Boolean

    private external fun nativeRunResidentBackward(
        operation: Int,
        input0Handle: Long,
        input1Handle: Long,
        input2Handle: Long,
        output0Handle: Long,
        output1Handle: Long,
        rows: Int,
        columns: Int,
        sequenceLength: Int,
        scale: Float,
        epsilon: Float
    ): Boolean

    private external fun nativeRunResidentResidualLayerNorm(
        downHandle: Long,
        residualHandle: Long,
        projectedHandle: Long,
        normalizedHandle: Long,
        rows: Int,
        columns: Int
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

    private external fun nativeApplyResidentAdamWUpdate(
        parameterHandles: LongArray,
        candidateParameterHandles: LongArray,
        gradientHandle: Long,
        firstMomentHandles: LongArray,
        secondMomentHandles: LongArray,
        candidateFirstMomentHandles: LongArray,
        candidateSecondMomentHandles: LongArray,
        activeElementCounts: IntArray,
        weightDecayByGroup: FloatArray,
        learningRate: Float,
        beta1: Float,
        beta2: Float,
        biasCorrection1: Float,
        biasCorrection2: Float,
        epsilon: Float,
        gradientClipFactor: Float,
        statusHandle: Long
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
