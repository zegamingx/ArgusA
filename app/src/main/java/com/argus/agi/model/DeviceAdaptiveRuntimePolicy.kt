package com.argus.agi.model

import android.os.PowerManager
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Runtime resource governor for the Vulkan training backend.
 *
 * Heap and Vulkan memory limits are recalculated from current headroom. The
 * device-local heap is preferred when Vulkan exposes heap flags; older drivers
 * retain a conservative all-heap fallback.
 */
internal object DeviceAdaptiveRuntimePolicy {

    private const val DEVICE_LOCAL_HEAP_FLAG = 0x1
    private const val MEBIBYTE = 1024L * 1024L

    internal data class TensorArenaBudget(
        val totalBytes: Long,
        val modelBytes: Long,
        val activationBytes: Long,
        val optimizerBytes: Long
    )

    fun deviceLocalHeapHeadroomBytes(
        budgets: List<Long>,
        usages: List<Long>,
        heapFlags: List<Int>
    ): List<Long> {
        if (budgets.size != usages.size) return emptyList()

        val selectedIndices = if (heapFlags.size == budgets.size) {
            heapFlags.indices.filter { index ->
                (heapFlags[index] and DEVICE_LOCAL_HEAP_FLAG) != 0
            }
        } else {
            budgets.indices.toList()
        }

        return selectedIndices.map { index ->
            (budgets[index] - usages[index]).coerceAtLeast(0L)
        }
    }

    fun calculateTensorArenaBudget(
        maximumAppHeapBytes: Long,
        availableAppHeapBytes: Long,
        deviceLocalHeadroomBytes: Long?,
        maximumStorageBufferRangeBytes: Long
    ): TensorArenaBudget {
        val safeMaximumHeap = maximumAppHeapBytes.coerceAtLeast(0L)
        val safeAvailableHeap = availableAppHeapBytes.coerceAtLeast(0L)
        val maximumAppShare = safeMaximumHeap / 4L
        val availableAppShare = safeAvailableHeap / 2L
        val appHeapBudget = minOf(maximumAppShare, availableAppShare)

        val gpuBudget = deviceLocalHeadroomBytes
            ?.coerceAtLeast(0L)
            ?.div(3L)
            ?: appHeapBudget
        val descriptorLimit = maximumStorageBufferRangeBytes
            .takeIf { value -> value > 0L }
            ?: Long.MAX_VALUE
        val totalBudget = minOf(appHeapBudget, gpuBudget, descriptorLimit)
            .coerceAtLeast(0L)

        return TensorArenaBudget(
            totalBytes = totalBudget,
            modelBytes = totalBudget / 12L,
            activationBytes = totalBudget / 6L,
            optimizerBytes = totalBudget / 3L
        )
    }

    /**
     * Returns a safe per-step sample limit from live memory, OS thermal state,
     * battery/power-saver state, and the previous OOM recovery signal.
     */
    fun trainingBatchLimit(
        configuredBatchSize: Int,
        thermalStatus: Int,
        availableHeapBytes: Long = Long.MAX_VALUE,
        maximumHeapBytes: Long = Long.MAX_VALUE,
        availableSystemMemoryBytes: Long = Long.MAX_VALUE,
        systemLowMemory: Boolean = false,
        isPowerSaveMode: Boolean = false,
        isCharging: Boolean = true,
        batteryPct: Int = 100,
        memoryRecovery: Boolean = false
    ): Int {
        require(configuredBatchSize in 1..8) {
            "Configured training batch size must be between 1 and 8."
        }

        if (thermalStatus >= PowerManager.THERMAL_STATUS_SEVERE) return 0

        var limit = configuredBatchSize
        if (thermalStatus >= PowerManager.THERMAL_STATUS_MODERATE) {
            limit = min(limit, ceil(configuredBatchSize / 2.0).toInt())
        }

        val heapRatio = availableHeapRatio(availableHeapBytes, maximumHeapBytes)
        val systemMemoryConstrained = systemLowMemory ||
            availableSystemMemoryBytes < 384L * MEBIBYTE

        if (memoryRecovery || systemMemoryConstrained || heapRatio < 0.15) {
            limit = 1
        } else if (heapRatio < 0.30 ||
            (!isCharging && (isPowerSaveMode || batteryPct < 25))) {
            limit = min(limit, ceil(configuredBatchSize / 2.0).toInt())
        }

        return limit.coerceIn(1, configuredBatchSize)
    }

    /**
     * Caps sequence windows without changing the model architecture or checkpoint.
     * The workspace estimate includes logits, layer activations, feed-forward
     * intermediates and quadratic attention-score storage.
     */
    fun trainingContextLimit(
        configuredContextLength: Int,
        availableHeapBytes: Long,
        maximumHeapBytes: Long,
        dModel: Int,
        feedForwardSize: Int,
        vocabularySize: Int,
        layerCount: Int,
        headCount: Int,
        thermalStatus: Int,
        availableSystemMemoryBytes: Long = Long.MAX_VALUE,
        systemLowMemory: Boolean = false,
        isPowerSaveMode: Boolean = false,
        isCharging: Boolean = true,
        memoryRecovery: Boolean = false
    ): Int {
        require(configuredContextLength >= 2)
        require(dModel > 0 && feedForwardSize > 0 && vocabularySize > 0)
        require(layerCount > 0 && headCount > 0)

        if (configuredContextLength <= 2) return configuredContextLength

        val minimumContext = min(configuredContextLength, 32)
        val heapRatio = availableHeapRatio(availableHeapBytes, maximumHeapBytes)
        val memoryConstrained = memoryRecovery ||
            systemLowMemory ||
            availableSystemMemoryBytes < 384L * MEBIBYTE ||
            heapRatio < 0.15
        val moderatelyConstrained = heapRatio < 0.30 ||
            thermalStatus >= PowerManager.THERMAL_STATUS_MODERATE ||
            (!isCharging && isPowerSaveMode)

        val targetContext = when {
            memoryConstrained -> max(minimumContext, configuredContextLength / 2)
            moderatelyConstrained -> max(
                minimumContext,
                floor(configuredContextLength * 0.75).toInt()
            )
            else -> configuredContextLength
        }

        val runtimeWorkspaceBudget = minOf(
            availableHeapBytes.coerceAtLeast(0L) / 3L,
            maximumHeapBytes.coerceAtLeast(0L) / 4L
        )
        var selectedContext = targetContext
        while (selectedContext > minimumContext &&
            estimateTrainingWindowWorkspaceBytes(
                contextLength = selectedContext,
                dModel = dModel,
                feedForwardSize = feedForwardSize,
                vocabularySize = vocabularySize,
                layerCount = layerCount,
                headCount = headCount
            ) > runtimeWorkspaceBudget) {
            selectedContext = max(minimumContext, selectedContext - max(1, selectedContext / 8))
        }

        return selectedContext.coerceIn(minimumContext, configuredContextLength)
    }

    fun estimateTrainingWindowWorkspaceBytes(
        contextLength: Int,
        dModel: Int,
        feedForwardSize: Int,
        vocabularySize: Int,
        layerCount: Int,
        headCount: Int
    ): Long {
        require(contextLength >= 2)
        require(dModel > 0 && feedForwardSize > 0 && vocabularySize > 0)
        require(layerCount > 0 && headCount > 0)

        val context = contextLength.toDouble()
        val linearElements = context * (
            dModel.toDouble() +
                vocabularySize.toDouble() +
                layerCount.toDouble() * (8.0 * dModel + 4.0 * feedForwardSize)
            )
        val attentionElements = layerCount.toDouble() *
            headCount.toDouble() * context * context
        val bytes = (linearElements + attentionElements) * Float.SIZE_BYTES.toDouble() * 2.0
        return when {
            !bytes.isFinite() || bytes >= Long.MAX_VALUE.toDouble() -> Long.MAX_VALUE
            bytes <= 0.0 -> 0L
            else -> ceil(bytes).toLong()
        }
    }

    fun trainingBatchDelayMillis(
        thermalStatus: Int,
        temperatureCelsius: Float,
        batteryPct: Int,
        isCharging: Boolean,
        isPowerSaveMode: Boolean
    ): Long = when {
        thermalStatus >= PowerManager.THERMAL_STATUS_CRITICAL -> 1_500L
        thermalStatus >= PowerManager.THERMAL_STATUS_MODERATE -> 500L
        temperatureCelsius >= 45.0f -> 600L
        temperatureCelsius >= 40.0f -> 300L
        !isCharging && (isPowerSaveMode || batteryPct < 25) -> 500L
        isPowerSaveMode -> 250L
        else -> 100L
    }

    private fun availableHeapRatio(
        availableHeapBytes: Long,
        maximumHeapBytes: Long
    ): Double {
        if (maximumHeapBytes <= 0L) return 0.0
        return (availableHeapBytes.coerceAtLeast(0L).toDouble() /
            maximumHeapBytes.toDouble()).coerceIn(0.0, 1.0)
    }
}

internal enum class AdaptiveComputeOperation {
    LINEAR,
    ADAMW
}

/**
 * Learns the end-to-end cost per operation from successful production workloads.
 * It keeps conservative bootstrap thresholds until enough CPU and GPU samples
 * exist, then selects Vulkan only when the measured GPU path is at least 5% faster.
 */
internal object AdaptiveComputeDispatchPolicy {

    private const val MIN_CALIBRATION_SAMPLES = 4
    private const val CPU_PROBE_INTERVAL = 16
    private const val REQUIRED_SPEEDUP = 0.95
    private const val EWMA_OLD_WEIGHT = 0.75
    private const val EWMA_NEW_WEIGHT = 0.25

    internal data class Snapshot(
        val cpuSamples: Int,
        val gpuSamples: Int,
        val cpuNanosecondsPerOperation: Double?,
        val gpuNanosecondsPerOperation: Double?
    )

    private data class Profile(
        var cpuSamples: Int = 0,
        var gpuSamples: Int = 0,
        var cpuProbePending: Boolean = false,
        var lastCpuProbeGpuSamples: Int = 0,
        var lastGpuProbeCpuSamples: Int = 0,
        var cpuNanosecondsPerOperation: Double = 0.0,
        var gpuNanosecondsPerOperation: Double = 0.0
    )

    private data class ProfileKey(
        val operation: AdaptiveComputeOperation,
        val operationSizeBucket: Int
    )

    private val profiles = HashMap<ProfileKey, Profile>()

    @Synchronized
    fun shouldUseGpu(
        operation: AdaptiveComputeOperation,
        operationCount: Long,
        bootstrapMinimumOperationCount: Long
    ): Boolean {
        if (operationCount < bootstrapMinimumOperationCount) return false
        val profile = profileFor(operation, operationCount)

        // A CPU probe stays selected through every fallback layer until a real
        // CPU timing sample is committed for this exact workload-size bucket.
        if (profile.cpuProbePending) return false

        // GPU needs initial samples first. Then periodically route one eligible
        // workload through CPU so CPU cost is measured from a real fallback path.
        if (profile.cpuSamples < MIN_CALIBRATION_SAMPLES) {
            val nextCpuProbe = CPU_PROBE_INTERVAL * (profile.cpuSamples + 1)
            if (profile.gpuSamples >= MIN_CALIBRATION_SAMPLES &&
                profile.gpuSamples >= nextCpuProbe &&
                profile.gpuSamples % CPU_PROBE_INTERVAL == 0 &&
                profile.lastCpuProbeGpuSamples != profile.gpuSamples) {
                profile.lastCpuProbeGpuSamples = profile.gpuSamples
                profile.cpuProbePending = true
                return false
            }
            return true
        }

        if (profile.gpuSamples < MIN_CALIBRATION_SAMPLES) return true
        val gpuIsCompetitive = profile.gpuNanosecondsPerOperation <
            profile.cpuNanosecondsPerOperation * REQUIRED_SPEEDUP
        if (gpuIsCompetitive) {
            if (profile.gpuSamples % CPU_PROBE_INTERVAL == 0 &&
                profile.lastCpuProbeGpuSamples != profile.gpuSamples) {
                profile.lastCpuProbeGpuSamples = profile.gpuSamples
                profile.cpuProbePending = true
                return false
            }
            return true
        }

        if (profile.cpuSamples % CPU_PROBE_INTERVAL == 0 &&
            profile.lastGpuProbeCpuSamples != profile.cpuSamples) {
            profile.lastGpuProbeCpuSamples = profile.cpuSamples
            return true
        }
        return false
    }

    @Synchronized
    fun recordCpu(
        operation: AdaptiveComputeOperation,
        operationCount: Long,
        elapsedNanoseconds: Long
    ) {
        record(operation, operationCount, elapsedNanoseconds, gpu = false)
    }

    @Synchronized
    fun recordGpu(
        operation: AdaptiveComputeOperation,
        operationCount: Long,
        elapsedNanoseconds: Long
    ) {
        record(operation, operationCount, elapsedNanoseconds, gpu = true)
    }

    @Synchronized
    fun snapshot(
        operation: AdaptiveComputeOperation,
        operationCount: Long = 1L
    ): Snapshot {
        val profile = profileFor(operation, operationCount)
        return Snapshot(
            cpuSamples = profile.cpuSamples,
            gpuSamples = profile.gpuSamples,
            cpuNanosecondsPerOperation = profile.cpuNanosecondsPerOperation
                .takeIf { profile.cpuSamples > 0 },
            gpuNanosecondsPerOperation = profile.gpuNanosecondsPerOperation
                .takeIf { profile.gpuSamples > 0 }
        )
    }

    @Synchronized
    internal fun resetForTests() {
        profiles.clear()
    }

    private fun profileFor(
        operation: AdaptiveComputeOperation,
        operationCount: Long
    ): Profile {
        val bucket = if (operationCount <= 0L) 0 else
            (java.lang.Long.SIZE - java.lang.Long.numberOfLeadingZeros(operationCount))
        return profiles.getOrPut(ProfileKey(operation, bucket)) { Profile() }
    }

    private fun record(
        operation: AdaptiveComputeOperation,
        operationCount: Long,
        elapsedNanoseconds: Long,
        gpu: Boolean
    ) {
        if (operationCount <= 0L || elapsedNanoseconds <= 0L) return
        val sample = elapsedNanoseconds.toDouble() / operationCount.toDouble()
        if (!sample.isFinite() || sample <= 0.0) return

        val profile = profileFor(operation, operationCount)
        if (gpu) {
            profile.gpuNanosecondsPerOperation =
                if (profile.gpuSamples == 0) sample
                else profile.gpuNanosecondsPerOperation * EWMA_OLD_WEIGHT +
                    sample * EWMA_NEW_WEIGHT
            profile.gpuSamples++
        } else {
            profile.cpuNanosecondsPerOperation =
                if (profile.cpuSamples == 0) sample
                else profile.cpuNanosecondsPerOperation * EWMA_OLD_WEIGHT +
                    sample * EWMA_NEW_WEIGHT
            profile.cpuSamples++
            profile.cpuProbePending = false
        }
    }
}
