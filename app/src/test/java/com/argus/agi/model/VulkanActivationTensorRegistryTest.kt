package com.argus.agi.model

import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class VulkanActivationTensorRegistryTest {

    @Test
    fun keepsActivationsResidentAcrossGpuDispatchUntilExplicitDownload() {
        var nextHandle = 1L
        var downloadCount = 0
        val allocations = mutableMapOf<Long, ByteArray>()
        val registry = VulkanActivationTensorRegistry(
            allocate = { byteCount ->
                val handle = nextHandle++
                allocations[handle] = ByteArray(byteCount.toInt())
                VulkanTensorAllocation(handle, byteCount)
            },
            upload = { allocation, bytes ->
                val target = allocations[allocation.nativeHandle]
                if (target == null || target.size != bytes.size) {
                    false
                } else {
                    bytes.copyInto(target)
                    true
                }
            },
            download = { allocation, destination ->
                downloadCount++
                val source = allocations[allocation.nativeHandle]
                if (source == null || source.size != destination.size) {
                    false
                } else {
                    source.copyInto(destination)
                    true
                }
            },
            release = { allocation -> allocations.remove(allocation.nativeHandle) != null },
            dispatch = { input, weights, output, rows, inputColumns, outputColumns, transpose ->
                val inputBytes = allocations[input.nativeHandle]
                val outputBytes = allocations[output.nativeHandle]
                if (rows != 2 || inputColumns != 3 || outputColumns != 2 ||
                    transpose || allocations[weights.nativeHandle] == null ||
                    inputBytes == null || outputBytes == null) {
                    false
                } else {
                    val source = ByteBuffer.wrap(inputBytes).order(ByteOrder.LITTLE_ENDIAN)
                    val target = ByteBuffer.wrap(outputBytes).order(ByteOrder.LITTLE_ENDIAN)
                    // A deterministic fake dispatch for the registry lifecycle test.
                    val matrix = floatArrayOf(1f, 0f, 0f, 1f, 0f, 0f)
                    for (row in 0 until rows) {
                        for (column in 0 until outputColumns) {
                            var sum = 0f
                            for (inner in 0 until inputColumns) {
                                sum += source.getFloat(
                                    (row * inputColumns + inner) * Float.SIZE_BYTES
                                ) * matrix[inner * outputColumns + column]
                            }
                            target.putFloat(
                                (row * outputColumns + column) * Float.SIZE_BYTES,
                                sum
                            )
                        }
                    }
                    true
                }
            },
            maxResidentBytes = { 128L }
        )
        val input = requireNotNull(registry.allocate(2, 3))
        val output = requireNotNull(registry.allocate(2, 2))
        allocations[100L] = ByteArray(3 * 2 * Short.SIZE_BYTES)
        val weights = VulkanTensorAllocation(100L, 3L * 2L * Short.SIZE_BYTES)

        assertTrue(registry.upload(input, arrayOf(
            floatArrayOf(1f, 2f, 3f),
            floatArrayOf(4f, 5f, 6f)
        )))
        assertTrue(registry.applyLinear(input, weights, output))
        assertEquals("Dispatch must not force an activation readback.", 0, downloadCount)
        assertEquals(2, registry.residentTensorCount())

        val actual = requireNotNull(registry.download(output))
        assertArrayEquals(floatArrayOf(1f, 2f), actual[0], 1e-6f)
        assertArrayEquals(floatArrayOf(4f, 5f), actual[1], 1e-6f)
        assertEquals(1, downloadCount)
    }


    @Test
    fun appliesProjectionSetWithOneResidentInputAndOneOutputReadback() {
        var nextHandle = 1L
        var uploadCount = 0
        var downloadCount = 0
        val allocations = mutableMapOf<Long, ByteArray>()
        val matrices = mutableMapOf<Long, FloatArray>()
        val registry = VulkanActivationTensorRegistry(
            allocate = { byteCount ->
                val handle = nextHandle++
                allocations[handle] = ByteArray(byteCount.toInt())
                VulkanTensorAllocation(handle, byteCount)
            },
            upload = { allocation, bytes ->
                uploadCount++
                val destination = allocations[allocation.nativeHandle]
                if (destination == null || destination.size != bytes.size) {
                    false
                } else {
                    bytes.copyInto(destination)
                    true
                }
            },
            download = { allocation, destination ->
                downloadCount++
                val source = allocations[allocation.nativeHandle]
                if (source == null || source.size != destination.size) {
                    false
                } else {
                    source.copyInto(destination)
                    true
                }
            },
            release = { allocation -> allocations.remove(allocation.nativeHandle) != null },
            dispatch = { _, _, _, _, _, _, _ -> false },
            maxResidentBytes = { 512L },
            dispatchLinearSet = { input, weights, output, rows, inputColumns, outputColumns, transpose ->
                val inputBytes = allocations[input.nativeHandle]
                val outputBytes = allocations[output.nativeHandle]
                if (transpose ||
                    weights.size != 3 ||
                    rows != 2 ||
                    inputColumns != 3 ||
                    outputColumns != 2 ||
                    downloadCount != 0 ||
                    inputBytes == null ||
                    outputBytes == null
                ) {
                    false
                } else {
                    val inputBuffer = ByteBuffer.wrap(inputBytes).order(ByteOrder.LITTLE_ENDIAN)
                    val outputBuffer = ByteBuffer.wrap(outputBytes).order(ByteOrder.LITTLE_ENDIAN)
                    var valid = true
                    for (batch in weights.indices) {
                        val matrix = matrices[weights[batch].nativeHandle]
                        if (matrix == null) {
                            valid = false
                        } else {
                            for (row in 0 until rows) {
                                for (column in 0 until outputColumns) {
                                    var sum = 0f
                                    for (inner in 0 until inputColumns) {
                                        sum += inputBuffer.getFloat(
                                            (row * inputColumns + inner) * Float.SIZE_BYTES
                                        ) * matrix[inner * outputColumns + column]
                                    }
                                    outputBuffer.putFloat(
                                        ((batch * rows + row) * outputColumns + column) * Float.SIZE_BYTES,
                                        sum
                                    )
                                }
                            }
                        }
                    }
                    valid
                }
            }
        )

        val input = requireNotNull(registry.allocate(2, 3))
        assertTrue(registry.upload(input, arrayOf(
            floatArrayOf(1f, 2f, 3f),
            floatArrayOf(4f, 5f, 6f)
        )))
        val weights = listOf(
            VulkanTensorAllocation(100L, 12L),
            VulkanTensorAllocation(101L, 12L),
            VulkanTensorAllocation(102L, 12L)
        )
        matrices[100L] = floatArrayOf(1f, 0f, 0f, 1f, 0f, 0f)
        matrices[101L] = floatArrayOf(0f, 1f, 1f, 0f, 0f, 0f)
        matrices[102L] = floatArrayOf(0f, 0f, 0f, 0f, 1f, 1f)
        weights.forEach { weight -> allocations[weight.nativeHandle] = ByteArray(12) }
        val output = requireNotNull(registry.allocate(6, 2))

        assertTrue(registry.applyLinearSet(input, weights, output))
        assertEquals(1, uploadCount)
        assertEquals("The complete projection set must finish before readback.", 0, downloadCount)

        val actual = requireNotNull(registry.download(output))
        assertArrayEquals(floatArrayOf(1f, 2f), actual[0], 1e-6f)
        assertArrayEquals(floatArrayOf(4f, 5f), actual[1], 1e-6f)
        assertArrayEquals(floatArrayOf(2f, 1f), actual[2], 1e-6f)
        assertArrayEquals(floatArrayOf(5f, 4f), actual[3], 1e-6f)
        assertArrayEquals(floatArrayOf(3f, 3f), actual[4], 1e-6f)
        assertArrayEquals(floatArrayOf(6f, 6f), actual[5], 1e-6f)
        assertEquals(1, downloadCount)
    }


    @Test
    fun dispatchesAttentionOnlyForCompleteDistinctResidentTensorSet() {
        var nextHandle = 1L
        var dispatched = false
        val registry = VulkanActivationTensorRegistry(
            allocate = { byteCount -> VulkanTensorAllocation(nextHandle++, byteCount) },
            upload = { _, _ -> true },
            download = { _, _ -> true },
            release = { true },
            dispatch = { _, _, _, _, _, _, _ -> false },
            maxResidentBytes = { 4096L },
            dispatchAttention = { qkv, scores, attentionOutput, projectedOutput, residual, weights,
                sequenceLength, modelWidth, headCount, headDimension ->
                dispatched = true
                qkv.nativeHandle > 0L && scores.nativeHandle > 0L &&
                    attentionOutput.nativeHandle > 0L && projectedOutput.nativeHandle > 0L &&
                    residual.nativeHandle > 0L && weights.nativeHandle == 100L &&
                    sequenceLength == 4 && modelWidth == 8 && headCount == 2 && headDimension == 4
            }
        )
        val qkv = requireNotNull(registry.allocate(12, 8))
        val scores = requireNotNull(registry.allocate(8, 4))
        val attentionOutput = requireNotNull(registry.allocate(4, 8))
        val projectedOutput = requireNotNull(registry.allocate(4, 8))
        val residual = requireNotNull(registry.allocate(4, 8))
        val weights = VulkanTensorAllocation(100L, 8L * 8L * Short.SIZE_BYTES)

        assertTrue(
            registry.applyAttention(
                qkv, scores, attentionOutput, projectedOutput, residual,
                weights, sequenceLength = 4, modelWidth = 8, headCount = 2, headDimension = 4
            )
        )
        assertTrue(dispatched)
    }

    @Test
    fun refusesAttentionShapeMismatchBeforeNativeDispatch() {
        var dispatchCount = 0
        var nextHandle = 1L
        val registry = VulkanActivationTensorRegistry(
            allocate = { byteCount -> VulkanTensorAllocation(nextHandle++, byteCount) },
            upload = { _, _ -> true },
            download = { _, _ -> true },
            release = { true },
            dispatch = { _, _, _, _, _, _, _ -> false },
            maxResidentBytes = { 4096L },
            dispatchAttention = { _, _, _, _, _, _, _, _, _, _ ->
                dispatchCount++
                true
            }
        )
        val qkv = requireNotNull(registry.allocate(11, 8))
        val scores = requireNotNull(registry.allocate(8, 4))
        val attentionOutput = requireNotNull(registry.allocate(4, 8))
        val projectedOutput = requireNotNull(registry.allocate(4, 8))
        val residual = requireNotNull(registry.allocate(4, 8))
        val weights = VulkanTensorAllocation(100L, 8L * 8L * Short.SIZE_BYTES)

        assertFalse(
            registry.applyAttention(
                qkv, scores, attentionOutput, projectedOutput, residual,
                weights, sequenceLength = 4, modelWidth = 8, headCount = 2, headDimension = 4
            )
        )
        assertEquals(0, dispatchCount)
    }

    @Test
    fun rejectsShapeMismatchAndNonFiniteActivationInput() {
        var allocationCount = 0
        val registry = VulkanActivationTensorRegistry(
            allocate = { byteCount ->
                allocationCount++
                VulkanTensorAllocation(allocationCount.toLong(), byteCount)
            },
            upload = { _, _ -> true },
            download = { _, _ -> true },
            release = { true },
            dispatch = { _, _, _, _, _, _, _ -> true },
            maxResidentBytes = { 256L }
        )
        val input = requireNotNull(registry.allocate(2, 3))
        assertFalse(registry.upload(input, arrayOf(
            floatArrayOf(1f, 2f),
            floatArrayOf(3f, 4f)
        )))
        assertFalse(registry.upload(input, arrayOf(
            floatArrayOf(1f, 2f, Float.NaN),
            floatArrayOf(3f, 4f, 5f)
        )))
        assertEquals(1, registry.residentTensorCount())
    }

    @Test
    fun refusesAllocationsBeyondItsReservedBudget() {
        var allocateCount = 0
        val registry = VulkanActivationTensorRegistry(
            allocate = { byteCount ->
                allocateCount++
                VulkanTensorAllocation(allocateCount.toLong(), byteCount)
            },
            upload = { _, _ -> true },
            download = { _, _ -> true },
            release = { true },
            dispatch = { _, _, _, _, _, _, _ -> true },
            maxResidentBytes = { 32L }
        )

        assertTrue(registry.allocate(2, 2) != null)
        assertTrue(registry.allocate(2, 2) != null)
        assertNull(registry.allocate(1, 1))
        assertEquals(2, allocateCount)
        assertEquals(32L, registry.residentByteCount())
    }

    @Test
    fun failedReleaseRemainsTrackedUntilArenaInvalidation() {
        val registry = VulkanActivationTensorRegistry(
            allocate = { byteCount -> VulkanTensorAllocation(1L, byteCount) },
            upload = { _, _ -> true },
            download = { _, _ -> true },
            release = { false },
            dispatch = { _, _, _, _, _, _, _ -> true },
            maxResidentBytes = { 64L }
        )
        val tensor = requireNotNull(registry.allocate(2, 2))
        assertFalse(registry.release(tensor))
        assertEquals(1, registry.residentTensorCount())
        assertEquals(16L, registry.residentByteCount())
        assertTrue(registry.invalidate(releaseAllocations = false))
        assertEquals(0, registry.residentTensorCount())
        assertEquals(0L, registry.residentByteCount())
    }

    @Test
    fun rejectsInvalidDimensionsAndOverflowBeforeCallingAllocator() {
        var allocateCount = 0
        val registry = VulkanActivationTensorRegistry(
            allocate = { byteCount ->
                allocateCount++
                VulkanTensorAllocation(allocateCount.toLong(), byteCount)
            },
            upload = { _, _ -> true },
            download = { _, _ -> true },
            release = { true },
            dispatch = { _, _, _, _, _, _, _ -> true },
            maxResidentBytes = { Long.MAX_VALUE }
        )

        assertNull(registry.allocate(0, 4))
        assertNull(registry.allocate(Int.MAX_VALUE, Int.MAX_VALUE))
        assertEquals(0, allocateCount)
    }
}
