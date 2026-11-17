package com.argus.agi.model

import android.os.PowerManager

/**
 * Runtime resource governor for the Vulkan training backend.
 *
 * Heap and Vulkan memory limits are recalculated from current headroom. The
 * device-local heap is preferred when Vulkan exposes heap flags; older drivers
 * retain the previous conservative all-heap fallback.
 */
internal object DeviceAdaptiveRuntimePolicy {

    private const val DEVICE_LOCAL_HEAP_FLAG = 0x1

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
     * Moderate thermal pressure halves the number of samples accumulated per
     * optimizer step. Severe and higher states return zero; DeviceGuard remains
     * the authority that prevents training until the device is safe again.
     */
    fun trainingBatchLimit(
        configuredBatchSize: Int,
        thermalStatus: Int
    ): Int {
        require(configuredBatchSize in 1..8) {
            "Configured training batch size must be between 1 and 8."
        }

        return when {
            thermalStatus >= PowerManager.THERMAL_STATUS_SEVERE -> 0
            thermalStatus >= PowerManager.THERMAL_STATUS_MODERATE ->
                (configuredBatchSize + 1) / 2
            else -> configuredBatchSize
        }
    }
}
