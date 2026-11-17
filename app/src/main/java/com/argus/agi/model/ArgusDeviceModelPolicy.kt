package com.argus.agi.model

/**
 * Device-adaptive Argus architecture families.
 *
 * Family ranges are realistic on-device training targets for the current native
 * Transformer engine. They are not aliases for multi-billion-parameter inference
 * planning figures. The active parameter count is additionally clamped by measured
 * app-heap and available-system-memory budgets.
 */
enum class ArgusDeviceModelTier(
    val displayName: String,
    val minimumFamilyParameters: Long,
    val maximumFamilyParameters: Long,
    val defaultTrainingBatchSize: Int,
    val vocabularyCapacity: Int
) {
    COMPACT("Argus Compact", 1_000_000L, 1_400_000L, 1, 1_024),
    BALANCED("Argus Balanced", 1_400_000L, 1_900_000L, 2, 2_048),
    PERFORMANCE("Argus Performance", 1_900_000L, 4_800_000L, 4, 4_096),
    FLAGSHIP("Argus Flagship", 4_800_000L, 7_000_000L, 8, 8_192)
}

data class ArgusDeviceModelProfile(
    val tier: ArgusDeviceModelTier,
    val targetParameterCount: Long,
    val memorySafeParameterLimit: Long,
    val trainingBatchSize: Int,
    val availableRamGb: Double,
    val maxHeapMb: Long
) {
    init {
        require(targetParameterCount in MINIMUM_MODEL_PARAMETERS..MAXIMUM_MODEL_PARAMETERS)
        require(memorySafeParameterLimit >= MINIMUM_EXECUTABLE_PARAMETER_BUDGET)
        require(trainingBatchSize in MINIMUM_TRAINING_BATCH_SIZE..MAXIMUM_TRAINING_BATCH_SIZE)
    }

    companion object {
        const val MINIMUM_MODEL_PARAMETERS = 1_000_000L
        const val MINIMUM_EXECUTABLE_PARAMETER_BUDGET = 1_122_304L
        const val MAXIMUM_MODEL_PARAMETERS = 7_000_000L
        const val MINIMUM_TRAINING_BATCH_SIZE = 1
        const val MAXIMUM_TRAINING_BATCH_SIZE = 8
    }
}

/**
 * Selects a model family from installed RAM, available system memory, app heap,
 * and CPU concurrency. Parameter limits use the active FP16-weight / FP32-state
 * memory contract: 14 bytes per parameter without accumulation and 18 bytes with
 * one additional FP32 gradient-accumulation buffer.
 */
object ArgusDeviceModelPolicy {

    private const val MIB = 1024L * 1024L
    private const val GIB = 1024.0 * 1024.0 * 1024.0
    private const val TRAINING_STATE_HEADROOM = 0.55
    private const val HEAP_BUDGET_FRACTION = 0.45
    private const val SYSTEM_MEMORY_BUDGET_FRACTION = 0.25

    fun select(
        totalRamGb: Int,
        maxHeapMb: Long,
        availableRamGb: Float,
        cpuCores: Int,
        vulkanSupported: Boolean = true
    ): ArgusDeviceModelProfile {
        require(totalRamGb > 0) { "Total RAM must be positive." }
        require(maxHeapMb > 0L) { "App heap must be positive." }
        require(cpuCores > 0) { "CPU core count must be positive." }

        val safeAvailableRamGb = availableRamGb
            .takeIf { value -> value.isFinite() }
            ?.coerceAtLeast(0.0f)
            ?: 0.0f
        val availableBytes = (safeAvailableRamGb.toDouble() * GIB)
            .coerceAtMost(Long.MAX_VALUE.toDouble())
            .toLong()
        val heapBytes = Math.multiplyExact(maxHeapMb, MIB)

        val heapTrainingBudget = (heapBytes.toDouble() * HEAP_BUDGET_FRACTION).toLong()
        val systemTrainingBudget =
            (availableBytes.toDouble() * SYSTEM_MEMORY_BUDGET_FRACTION).toLong()
        val trainingStateBudget = minOf(heapTrainingBudget, systemTrainingBudget)
        val conservativeParameterBudget = (
            trainingStateBudget.toDouble() *
                TRAINING_STATE_HEADROOM /
                TrainingMemoryPlanner.ACCUMULATED_BYTES_PER_PARAMETER
            ).toLong()
        require(
            conservativeParameterBudget >= ArgusDeviceModelProfile.MINIMUM_EXECUTABLE_PARAMETER_BUDGET
        ) {
            "This device does not currently have enough safe memory to train Argus's minimum executable model."
        }

        val detectedTier = when {
            totalRamGb >= 12 && maxHeapMb >= 512L &&
                safeAvailableRamGb >= 2.5f && cpuCores >= 8 ->
                ArgusDeviceModelTier.FLAGSHIP

            totalRamGb >= 8 && maxHeapMb >= 384L &&
                safeAvailableRamGb >= 1.5f && cpuCores >= 6 ->
                ArgusDeviceModelTier.PERFORMANCE

            totalRamGb >= 6 && maxHeapMb >= 256L &&
                safeAvailableRamGb >= 0.75f ->
                ArgusDeviceModelTier.BALANCED

            else -> ArgusDeviceModelTier.COMPACT
        }
        // Vulkan is the primary training accelerator. Devices that do not advertise
        // Vulkan start with the compact architecture while retaining CPU fallback.
        val tier = if (vulkanSupported) detectedTier else ArgusDeviceModelTier.COMPACT

        val targetParameterCount = minOf(
            tier.maximumFamilyParameters,
            conservativeParameterBudget
        )
        require(targetParameterCount >= ArgusDeviceModelProfile.MINIMUM_EXECUTABLE_PARAMETER_BUDGET) {
            "The selected device budget cannot fit the minimum executable Argus architecture."
        }

        return ArgusDeviceModelProfile(
            tier = tier,
            targetParameterCount = targetParameterCount,
            memorySafeParameterLimit = conservativeParameterBudget,
            trainingBatchSize = tier.defaultTrainingBatchSize.coerceIn(
                ArgusDeviceModelProfile.MINIMUM_TRAINING_BATCH_SIZE,
                ArgusDeviceModelProfile.MAXIMUM_TRAINING_BATCH_SIZE
            ),
            availableRamGb = safeAvailableRamGb.toDouble(),
            maxHeapMb = maxHeapMb
        )
    }
}
