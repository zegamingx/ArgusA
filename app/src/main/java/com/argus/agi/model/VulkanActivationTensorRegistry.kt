package com.argus.agi.model

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.IdentityHashMap

internal class VulkanActivationTensor internal constructor(
    val rows: Int,
    val columns: Int,
    internal val allocation: VulkanTensorAllocation
)

internal class VulkanActivationTensorRegistry(
    private val allocate: (Long) -> VulkanTensorAllocation?,
    private val upload: (VulkanTensorAllocation, ByteArray) -> Boolean,
    private val download: (VulkanTensorAllocation, ByteArray) -> Boolean,
    private val release: (VulkanTensorAllocation) -> Boolean,
    private val dispatch: (
        VulkanTensorAllocation,
        VulkanTensorAllocation,
        VulkanTensorAllocation,
        Int,
        Int,
        Int,
        Boolean
    ) -> Boolean,
    private val maxResidentBytes: () -> Long,
    private val dispatchLinearSet: (
        VulkanTensorAllocation,
        List<VulkanTensorAllocation>,
        VulkanTensorAllocation,
        Int,
        Int,
        Int,
        Boolean
    ) -> Boolean = { _, _, _, _, _, _, _ -> false }
) {
    private val active = IdentityHashMap<VulkanActivationTensor, VulkanTensorAllocation>()
    private var residentBytes = 0L

    @Synchronized
    fun allocate(rows: Int, columns: Int): VulkanActivationTensor? {
        if (rows <= 0 || columns <= 0) return null
        val elementCount = runCatching {
            Math.multiplyExact(rows.toLong(), columns.toLong())
        }.getOrNull() ?: return null
        val byteCount = runCatching {
            Math.multiplyExact(elementCount, Float.SIZE_BYTES.toLong())
        }.getOrNull() ?: return null
        if (byteCount <= 0L || byteCount > Int.MAX_VALUE.toLong()) return null

        val nextResidentBytes = runCatching {
            Math.addExact(residentBytes, byteCount)
        }.getOrNull() ?: return null
        val budget = maxResidentBytes().coerceAtLeast(0L)
        if (nextResidentBytes > budget) return null

        val allocation = allocate(byteCount) ?: return null
        if (allocation.nativeHandle <= 0L ||
            allocation.byteCount != byteCount ||
            active.values.any { it.nativeHandle == allocation.nativeHandle }) {
            release(allocation)
            return null
        }

        val tensor = VulkanActivationTensor(rows, columns, allocation)
        active[tensor] = allocation
        residentBytes = nextResidentBytes
        return tensor
    }

    @Synchronized
    fun upload(tensor: VulkanActivationTensor, values: Array<FloatArray>): Boolean {
        val allocation = active[tensor] ?: return false
        if (values.size != tensor.rows ||
            values.any { row -> row.size != tensor.columns }) {
            return false
        }

        val bytes = try {
            val byteCount = Math.toIntExact(allocation.byteCount)
            val buffer = ByteBuffer.allocate(byteCount).order(ByteOrder.LITTLE_ENDIAN)
            for (row in values) {
                for (value in row) {
                    if (!value.isFinite()) return false
                    buffer.putFloat(value)
                }
            }
            buffer.array()
        } catch (error: OutOfMemoryError) {
            return false
        } catch (error: ArithmeticException) {
            return false
        }

        return upload(allocation, bytes)
    }

    /**
     * Dispatches directly between arena allocations. This operation deliberately does
     * not download the output; later GPU operations can consume it without a CPU round-trip.
     */
    @Synchronized
    fun applyLinear(
        input: VulkanActivationTensor,
        weights: VulkanTensorAllocation,
        output: VulkanActivationTensor,
        transposeWeights: Boolean = false
    ): Boolean {
        val inputAllocation = active[input] ?: return false
        val outputAllocation = active[output] ?: return false
        if (input === output ||
            inputAllocation.nativeHandle == outputAllocation.nativeHandle ||
            weights.nativeHandle == inputAllocation.nativeHandle ||
            weights.nativeHandle == outputAllocation.nativeHandle ||
            input.rows != output.rows ||
            input.columns <= 0 ||
            output.columns <= 0) {
            return false
        }

        val expectedWeightBytes = runCatching {
            Math.multiplyExact(
                Math.multiplyExact(input.columns.toLong(), output.columns.toLong()),
                Short.SIZE_BYTES.toLong()
            )
        }.getOrNull() ?: return false
        if (weights.byteCount < expectedWeightBytes) return false

        return dispatch(
            inputAllocation,
            weights,
            outputAllocation,
            input.rows,
            input.columns,
            output.columns,
            transposeWeights
        )
    }


    /**
     * Applies independent matrices to one shared resident input and packs their outputs
     * into consecutive row ranges of one resident output tensor. No activation readback
     * occurs between the matrix projections.
     */
    @Synchronized
    fun applyLinearSet(
        input: VulkanActivationTensor,
        weights: List<VulkanTensorAllocation>,
        output: VulkanActivationTensor,
        transposeWeights: Boolean = false
    ): Boolean {
        val inputAllocation = active[input] ?: return false
        val outputAllocation = active[output] ?: return false
        if (weights.isEmpty() ||
            input === output ||
            input.rows <= 0 ||
            input.columns <= 0 ||
            output.columns <= 0) {
            return false
        }

        val expectedOutputRows = runCatching {
            Math.multiplyExact(input.rows, weights.size)
        }.getOrNull() ?: return false
        if (output.rows != expectedOutputRows) return false

        val expectedWeightBytes = runCatching {
            Math.multiplyExact(
                Math.multiplyExact(input.columns.toLong(), output.columns.toLong()),
                Short.SIZE_BYTES.toLong()
            )
        }.getOrNull() ?: return false
        if (weights.any { weight ->
                weight.nativeHandle <= 0L ||
                    weight.byteCount < expectedWeightBytes ||
                    weight.nativeHandle == inputAllocation.nativeHandle ||
                    weight.nativeHandle == outputAllocation.nativeHandle
            }) {
            return false
        }
        if (weights.map { it.nativeHandle }.toSet().size != weights.size) return false

        return dispatchLinearSet(
            inputAllocation,
            weights,
            outputAllocation,
            input.rows,
            input.columns,
            output.columns,
            transposeWeights
        )
    }

    @Synchronized
    fun download(tensor: VulkanActivationTensor): Array<FloatArray>? {
        val allocation = active[tensor] ?: return null
        val bytes = try {
            ByteArray(Math.toIntExact(allocation.byteCount))
        } catch (error: OutOfMemoryError) {
            return null
        } catch (error: ArithmeticException) {
            return null
        }
        if (!download(allocation, bytes)) return null

        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val result = Array(tensor.rows) { FloatArray(tensor.columns) }
        for (row in result.indices) {
            for (column in result[row].indices) {
                val value = buffer.float
                if (!value.isFinite()) return null
                result[row][column] = value
            }
        }
        return result
    }

    @Synchronized
    fun release(tensor: VulkanActivationTensor): Boolean {
        val allocation = active[tensor] ?: return false
        if (!release(allocation)) return false
        active.remove(tensor)
        residentBytes -= allocation.byteCount
        return true
    }

    @Synchronized
    fun invalidate(releaseAllocations: Boolean): Boolean {
        if (!releaseAllocations) {
            active.clear()
            residentBytes = 0L
            return true
        }

        val iterator = active.entries.iterator()
        var allReleased = true
        while (iterator.hasNext()) {
            val entry = iterator.next()
            if (release(entry.value)) {
                residentBytes -= entry.value.byteCount
                iterator.remove()
            } else {
                allReleased = false
            }
        }
        return allReleased
    }

    @Synchronized
    fun residentTensorCount(): Int = active.size

    @Synchronized
    fun residentByteCount(): Long = residentBytes
}
