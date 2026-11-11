package com.argus.agi.model

/**
 * Four device-adaptive Argus architecture families.
 *
 * The parameter bands describe the intended family scale, not a promise that every
 * Android device can train at the band's upper bound. The active FP32 model is always
 * clamped by the measured app-heap and available-system-memory budget.
 */
enum class ArgusDeviceModelTier(
    val displayName: String,
    val minimumFamilyParameters: Long,
    val maximumFamilyParameters: Long,
    val defaultTrainingBatchSize: Int
) {
    COMPACT("Argus Compact", 1_000_000L, 10_000_000L, 1),
    BALANCED("Argus Balanced", 10_000_000L, 100_000_000L, 2),
    PERFORMANCE("Argus Performance", 100_000_000L, 1_000_000_000L, 4),
    FLAGSHIP("Argus Flagship", 1_000_000_000L, 7_000_000_000L, 8)
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
        require(memorySafeParameterLimit >= MINIMUM_MODEL_PARAMETERS)
        require(trainingBatchSize in MINIMUM_TRAINING_BATCH_SIZE..MAXIMUM_TRAINING_BATCH_SIZE)
    }

    companion object {
        const val MINIMUM_MODEL_PARAMETERS = 1_000_000L
        const val MAXIMUM_MODEL_PARAMETERS = 7_000_000_000L
        const val MINIMUM_TRAINING_BATCH_SIZE = 1
        const val MAXIMUM_TRAINING_BATCH_SIZE = 8
    }
}

/**
 * Selects a model family using stable installed RAM, app heap, CPU concurrency, and
 * the current free-memory snapshot. The model's actual parameter count is separately
 * capped using a conservative full-parameter FP32 training-state budget.
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
        cpuCores: Int
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
            ).toLong().coerceAtLeast(ArgusDeviceModelProfile.MINIMUM_MODEL_PARAMETERS)

        val tier = when {
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

        val targetParameterCount = minOf(
            tier.maximumFamilyParameters,
            conservativeParameterBudget
        ).coerceAtLeast(ArgusDeviceModelProfile.MINIMUM_MODEL_PARAMETERS)

        val batchSize = tier.defaultTrainingBatchSize.coerceIn(
            ArgusDeviceModelProfile.MINIMUM_TRAINING_BATCH_SIZE,
            ArgusDeviceModelProfile.MAXIMUM_TRAINING_BATCH_SIZE
        )

        return ArgusDeviceModelProfile(
            tier = tier,
            targetParameterCount = targetParameterCount,
            memorySafeParameterLimit = conservativeParameterBudget,
            trainingBatchSize = batchSize,
            availableRamGb = safeAvailableRamGb.toDouble(),
            maxHeapMb = maxHeapMb
        )
    }
}
