package com.argus.agi.model

import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class VulkanActivationTensorRegistryTest {

    @Test
    fun keepsActivationsResidentAcrossGpuDispatchUntilExplicitDownload() {
        var nextHandle = 1L
        var downloadCount = 0
        val allocations = mutableMapOf<Long, ByteArray>()
        lateinit var registry: VulkanActivationTensorRegistry
        registry = VulkanActivationTensorRegistry(
            allocate = { byteCount ->
                val handle = nextHandle++
                allocations[handle] = ByteArray(byteCount.toInt())
                VulkanTensorAllocation(handle, byteCount)
            },
            upload = { allocation, bytes ->
                allocations[allocation.nativeHandle]?.also { target ->
                    bytes.copyInto(target)
                } != null
            },
            download = { allocation, destination ->
                downloadCount++
                val source = allocations[allocation.nativeHandle] ?: return@VulkanActivationTensorRegistry false
                source.copyInto(destination)
                true
            },
            release = { allocation -> allocations.remove(allocation.nativeHandle) != null },
            dispatch = { input, weights, output, rows, inputColumns, outputColumns, transpose ->
                assertEquals(2, rows)
                assertEquals(3, inputColumns)
                assertEquals(2, outputColumns)
                assertFalse(transpose)
                assertNotNull(allocations[input.nativeHandle])
                assertNotNull(allocations[weights.nativeHandle])
                val source = ByteBuffer.wrap(allocations[input.nativeHandle]!!)
                    .order(ByteOrder.LITTLE_ENDIAN)
                val targetBytes = allocations[output.nativeHandle] ?: return@VulkanActivationTensorRegistry false
                val target = ByteBuffer.wrap(targetBytes).order(ByteOrder.LITTLE_ENDIAN)
                val matrix = floatArrayOf(1f, 0f, 0f, 1f, 1f, 1f)
                for (row in 0 until rows) {
                    for (column in 0 until outputColumns) {
                        var sum = 0f
                        for (inner in 0 until inputColumns) {
                            sum += source.getFloat((row * inputColumns + inner) * Float.SIZE_BYTES) *
                                matrix[inner * outputColumns + column]
                        }
                        target.putFloat((row * outputColumns + column) * Float.SIZE_BYTES, sum)
                    }
                }
                true
            },
            maxResidentBytes = { 128L }
        )
        val input = registry.allocate(2, 3)
        val output = registry.allocate(2, 2)
        val weights = VulkanTensorAllocation(100L, 3L * 2L * Short.SIZE_BYTES)

        assertNotNull(input)
        assertNotNull(output)
        assertTrue(registry.upload(input!!, arrayOf(
            floatArrayOf(1f, 2f, 3f),
            floatArrayOf(4f, 5f, 6f)
        )))
        assertTrue(registry.applyLinear(input, weights, output!!))
        assertEquals("Dispatch must not force an activation readback.", 0, downloadCount)
        assertEquals(2, registry.residentTensorCount())

        val actual = registry.download(output)
        assertNotNull(actual)
        assertArrayEquals(floatArrayOf(1f, 5f), actual!![0], 1e-6f)
        assertArrayEquals(floatArrayOf(4f, 11f), actual[1], 1e-6f)
        assertEquals(1, downloadCount)
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
        val input = registry.allocate(2, 3)
        assertNotNull(input)
        assertFalse(registry.upload(input!!, arrayOf(
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

        assertNotNull(registry.allocate(2, 2))
        assertNotNull(registry.allocate(2, 2))
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
        val tensor = registry.allocate(2, 2)
        assertNotNull(tensor)
        assertFalse(registry.release(tensor!!))
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
