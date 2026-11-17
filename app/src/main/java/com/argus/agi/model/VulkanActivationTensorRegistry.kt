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
    ) -> Boolean = { _, _, _, _, _, _, _ -> false },
    private val dispatchAttention: (
        VulkanTensorAllocation,
        VulkanTensorAllocation,
        VulkanTensorAllocation,
        VulkanTensorAllocation,
        VulkanTensorAllocation,
        VulkanTensorAllocation,
        VulkanTensorAllocation,
        Int,
        Int,
        Int,
        Int
    ) -> Boolean = { _, _, _, _, _, _, _, _, _, _, _ -> false },
    private val dispatchSwiGlu: (
        VulkanTensorAllocation,
        VulkanTensorAllocation,
        VulkanTensorAllocation,
        Int,
        Int
    ) -> Boolean = { _, _, _, _, _ -> false },
    private val dispatchResidualLayerNorm: (
        VulkanTensorAllocation,
        VulkanTensorAllocation,
        VulkanTensorAllocation,
        VulkanTensorAllocation,
        Int,
        Int
    ) -> Boolean = { _, _, _, _, _, _ -> false }
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


    /**
     * Dispatches causal multi-head attention and W_O + residual between distinct
     * resident allocations. Shape validation is performed before entering native code.
     */
    @Synchronized
    fun applyAttention(
        qkv: VulkanActivationTensor,
        scores: VulkanActivationTensor,
        attentionOutput: VulkanActivationTensor,
        projectedOutput: VulkanActivationTensor,
        normalizedOutput: VulkanActivationTensor,
        residual: VulkanActivationTensor,
        outputWeights: VulkanTensorAllocation,
        sequenceLength: Int,
        modelWidth: Int,
        headCount: Int,
        headDimension: Int
    ): Boolean {
        val qkvAllocation = active[qkv] ?: return false
        val scoreAllocation = active[scores] ?: return false
        val attentionOutputAllocation = active[attentionOutput] ?: return false
        val projectedOutputAllocation = active[projectedOutput] ?: return false
        val normalizedOutputAllocation = active[normalizedOutput] ?: return false
        val residualAllocation = active[residual] ?: return false

        if (sequenceLength <= 0 ||
            modelWidth <= 0 ||
            headCount <= 0 ||
            headDimension <= 0 ||
            headCount.toLong() * headDimension.toLong() != modelWidth.toLong()) {
            return false
        }

        val qkvRows = runCatching { Math.multiplyExact(sequenceLength, 3) }.getOrNull()
            ?: return false
        val scoreRows = runCatching { Math.multiplyExact(sequenceLength, headCount) }.getOrNull()
            ?: return false
        val outputWeightBytes = runCatching {
            Math.multiplyExact(
                Math.multiplyExact(modelWidth.toLong(), modelWidth.toLong()),
                Short.SIZE_BYTES.toLong()
            )
        }.getOrNull() ?: return false

        if (qkv.rows != qkvRows ||
            qkv.columns != modelWidth ||
            scores.rows != scoreRows ||
            scores.columns != sequenceLength ||
            attentionOutput.rows != sequenceLength ||
            attentionOutput.columns != modelWidth ||
            projectedOutput.rows != sequenceLength ||
            projectedOutput.columns != modelWidth ||
            normalizedOutput.rows != sequenceLength ||
            normalizedOutput.columns != modelWidth ||
            residual.rows != sequenceLength ||
            residual.columns != modelWidth ||
            outputWeights.nativeHandle <= 0L ||
            outputWeights.byteCount < outputWeightBytes) {
            return false
        }

        val allocations = listOf(
            qkvAllocation,
            scoreAllocation,
            attentionOutputAllocation,
            projectedOutputAllocation,
            normalizedOutputAllocation,
            residualAllocation
        )
        val handles = allocations.map { it.nativeHandle } + outputWeights.nativeHandle
        if (handles.any { it <= 0L } || handles.toSet().size != handles.size) return false

        return dispatchAttention(
            qkvAllocation,
            scoreAllocation,
            attentionOutputAllocation,
            projectedOutputAllocation,
            normalizedOutputAllocation,
            residualAllocation,
            outputWeights,
            sequenceLength,
            modelWidth,
            headCount,
            headDimension
        )
    }

    /**
     * Executes element-wise SwiGLU on distinct resident Tensor Arena allocations.
     * Shape and allocation validation completes before native dispatch.
     */
    @Synchronized
    fun applySwiGlu(
        gate: VulkanActivationTensor,
        up: VulkanActivationTensor,
        output: VulkanActivationTensor
    ): Boolean {
        val gateAllocation = active[gate] ?: return false
        val upAllocation = active[up] ?: return false
        val outputAllocation = active[output] ?: return false
        if (gate === up || gate === output || up === output ||
            gate.rows <= 0 || gate.columns <= 0 ||
            gate.rows != up.rows || gate.columns != up.columns ||
            gate.rows != output.rows || gate.columns != output.columns) {
            return false
        }

        val handles = longArrayOf(
            gateAllocation.nativeHandle,
            upAllocation.nativeHandle,
            outputAllocation.nativeHandle
        )
        if (handles.any { it <= 0L } || handles.toSet().size != handles.size) return false

        val expectedBytes = runCatching {
            Math.multiplyExact(
                Math.multiplyExact(gate.rows.toLong(), gate.columns.toLong()),
                Float.SIZE_BYTES.toLong()
            )
        }.getOrNull() ?: return false
        if (gateAllocation.byteCount != expectedBytes ||
            upAllocation.byteCount != expectedBytes ||
            outputAllocation.byteCount != expectedBytes) return false

        return dispatchSwiGlu(
            gateAllocation, upAllocation, outputAllocation, gate.rows, gate.columns
        )
    }

    /**
     * Applies the feed-forward residual and post-block LayerNorm while every
     * intermediate remains resident in the Tensor Arena.
     */
    @Synchronized
    fun applyResidualLayerNorm(
        downProjection: VulkanActivationTensor,
        residual: VulkanActivationTensor,
        projectedOutput: VulkanActivationTensor,
        normalizedOutput: VulkanActivationTensor
    ): Boolean {
        val downAllocation = active[downProjection] ?: return false
        val residualAllocation = active[residual] ?: return false
        val projectedAllocation = active[projectedOutput] ?: return false
        val normalizedAllocation = active[normalizedOutput] ?: return false

        if (downProjection === residual ||
            downProjection === projectedOutput ||
            downProjection === normalizedOutput ||
            residual === projectedOutput ||
            residual === normalizedOutput ||
            projectedOutput === normalizedOutput ||
            downProjection.rows <= 0 ||
            downProjection.columns <= 0 ||
            downProjection.rows != residual.rows ||
            downProjection.columns != residual.columns ||
            downProjection.rows != projectedOutput.rows ||
            downProjection.columns != projectedOutput.columns ||
            downProjection.rows != normalizedOutput.rows ||
            downProjection.columns != normalizedOutput.columns) {
            return false
        }

        val handles = longArrayOf(
            downAllocation.nativeHandle,
            residualAllocation.nativeHandle,
            projectedAllocation.nativeHandle,
            normalizedAllocation.nativeHandle
        )
        if (handles.any { it <= 0L } || handles.toSet().size != handles.size) return false

        val expectedBytes = runCatching {
            Math.multiplyExact(
                Math.multiplyExact(downProjection.rows.toLong(), downProjection.columns.toLong()),
                Float.SIZE_BYTES.toLong()
            )
        }.getOrNull() ?: return false
        if (downAllocation.byteCount != expectedBytes ||
            residualAllocation.byteCount != expectedBytes ||
            projectedAllocation.byteCount != expectedBytes ||
            normalizedAllocation.byteCount != expectedBytes) {
            return false
        }

        return dispatchResidualLayerNorm(
            downAllocation,
            residualAllocation,
            projectedAllocation,
            normalizedAllocation,
            downProjection.rows,
            downProjection.columns
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
