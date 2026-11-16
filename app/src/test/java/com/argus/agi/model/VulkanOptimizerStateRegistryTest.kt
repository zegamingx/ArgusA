package com.argus.agi.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class VulkanOptimizerStateRegistryTest {

    private class FakeArena {
        private var nextHandle = 1L
        val contents = LinkedHashMap<Long, ByteArray>()
        var uploadCount = 0
        var releaseFailure = false
        var downloadFailureHandle: Long? = null

        fun allocate(byteCount: Long): VulkanTensorAllocation? {
            if (byteCount <= 0L || byteCount > Int.MAX_VALUE.toLong()) return null
            val handle = nextHandle++
            contents[handle] = ByteArray(byteCount.toInt())
            return VulkanTensorAllocation(handle, byteCount)
        }

        fun upload(allocation: VulkanTensorAllocation, bytes: ByteArray): Boolean {
            val target = contents[allocation.nativeHandle] ?: return false
            if (target.size != bytes.size) return false
            bytes.copyInto(target)
            uploadCount++
            return true
        }

        fun download(allocation: VulkanTensorAllocation, destination: ByteArray): Boolean {
            if (allocation.nativeHandle == downloadFailureHandle) return false
            val source = contents[allocation.nativeHandle] ?: return false
            if (source.size != destination.size) return false
            source.copyInto(destination)
            return true
        }

        fun release(allocation: VulkanTensorAllocation): Boolean {
            if (releaseFailure) return false
            return contents.remove(allocation.nativeHandle) != null
        }
    }

    private fun registry(
        arena: FakeArena,
        maxBytes: () -> Long = { Long.MAX_VALUE }
    ): VulkanOptimizerStateRegistry = VulkanOptimizerStateRegistry(
        allocate = arena::allocate,
        upload = arena::upload,
        release = arena::release,
        download = arena::download,
        maxResidentBytes = maxBytes
    )

    @Test
    fun keepsMomentArraysResidentWithoutRepeatedUploads() {
        val arena = FakeArena()
        val state = registry(arena)
        val first = FloatArray(8) { it * 0.1f }
        val second = FloatArray(8) { it * 0.2f }

        assertTrue(state.synchronizeAll(listOf(first, second)))
        assertEquals(2, arena.uploadCount)
        assertEquals(2, state.residentTensorCount())

        assertTrue(state.synchronizeAll(listOf(first, second)))
        assertEquals(2, arena.uploadCount)
        assertEquals(64L, state.residentByteCount())
    }

    @Test
    fun releasesArraysRemovedFromTheOptimizerInventory() {
        val arena = FakeArena()
        val state = registry(arena)
        val first = FloatArray(4)
        val second = FloatArray(6)

        assertTrue(state.synchronizeAll(listOf(first, second)))
        assertTrue(state.synchronizeAll(listOf(second)))

        assertEquals(1, state.residentTensorCount())
        assertEquals(24L, state.residentByteCount())
    }

    @Test
    fun rejectsAnInventoryThatExceedsItsMemoryBudgetBeforeAllocation() {
        val arena = FakeArena()
        val state = registry(arena) { 31L }

        assertFalse(state.synchronizeAll(listOf(FloatArray(4))))
        assertEquals(0, arena.contents.size)
        assertEquals(0, arena.uploadCount)
    }

    @Test
    fun failedReleaseKeepsTheHandleTrackedForRetry() {
        val arena = FakeArena()
        val state = registry(arena)
        val first = FloatArray(4)

        assertTrue(state.synchronizeAll(listOf(first)))
        arena.releaseFailure = true

        assertFalse(state.invalidate(releaseAllocations = true))
        assertEquals(1, state.residentTensorCount())
        assertNotNull(state.acquire(first))

        arena.releaseFailure = false
        assertTrue(state.invalidate(releaseAllocations = true))
        assertEquals(0, state.residentTensorCount())
        assertNull(state.acquire(first))
    }

    @Test
    fun invalidationDropsGpuHandlesWithoutChangingHostMomentArrays() {
        val arena = FakeArena()
        val state = registry(arena)
        val first = floatArrayOf(0.25f, -0.5f, 0.75f)
        val before = first.copyOf()

        assertTrue(state.synchronizeAll(listOf(first)))
        assertTrue(state.invalidate(releaseAllocations = false))

        assertEquals(0, state.residentTensorCount())
        assertTrue(first.contentEquals(before))
        assertTrue(state.synchronizeAll(listOf(first)))
        assertEquals(2, arena.uploadCount)
    }
    @Test
    fun downloadsGpuMomentsOnlyWhenHostRecoveryIsRequested() {
        val arena = FakeArena()
        val state = registry(arena)
        val first = floatArrayOf(0.1f, 0.2f, 0.3f)
        assertTrue(state.synchronizeAll(listOf(first)))
        val allocation = state.acquire(first)
        assertNotNull(allocation)
        val gpuBytes = ByteBuffer.allocate(first.size * Float.SIZE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
            .putFloat(0.75f)
            .putFloat(-0.5f)
            .putFloat(1.25f)
            .array()
        val handle = requireNotNull(allocation)
        gpuBytes.copyInto(requireNotNull(arena.contents[handle.nativeHandle]))
        first.fill(9.0f)

        assertTrue(state.synchronizeHostMirrors())
        assertTrue(first.contentEquals(floatArrayOf(0.75f, -0.5f, 1.25f)))
        assertEquals(1, arena.uploadCount)
    }
    @Test
    fun commitsNoHostMomentUntilEveryGpuMomentReadbackSucceeds() {
        val arena = FakeArena()
        val state = registry(arena)
        val first = floatArrayOf(0.1f, 0.2f, 0.3f)
        val second = floatArrayOf(-0.1f, -0.2f, -0.3f)
        assertTrue(state.synchronizeAll(listOf(first, second)))

        val firstAllocation = state.acquire(first) ?: error("First optimizer-state allocation is missing.")
        val secondAllocation = state.acquire(second) ?: error("Second optimizer-state allocation is missing.")
        val firstGpuState = ByteBuffer.allocate(first.size * Float.SIZE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
            .putFloat(0.75f)
            .putFloat(-0.5f)
            .putFloat(1.25f)
            .array()
        val secondGpuState = ByteBuffer.allocate(second.size * Float.SIZE_BYTES)
            .order(ByteOrder.LITTLE_ENDIAN)
            .putFloat(-0.25f)
            .putFloat(0.5f)
            .putFloat(-0.75f)
            .array()
        firstGpuState.copyInto(requireNotNull(arena.contents[firstAllocation.nativeHandle]))
        secondGpuState.copyInto(requireNotNull(arena.contents[secondAllocation.nativeHandle]))
        first.fill(9.0f)
        second.fill(8.0f)
        val hostFirstBeforeFailure = first.copyOf()
        val hostSecondBeforeFailure = second.copyOf()

        arena.downloadFailureHandle = secondAllocation.nativeHandle
        assertFalse(state.synchronizeHostMirrors())
        assertTrue(first.contentEquals(hostFirstBeforeFailure))
        assertTrue(second.contentEquals(hostSecondBeforeFailure))

        arena.downloadFailureHandle = null
        assertTrue(state.synchronizeHostMirrors())
        assertTrue(first.contentEquals(floatArrayOf(0.75f, -0.5f, 1.25f)))
        assertTrue(second.contentEquals(floatArrayOf(-0.25f, 0.5f, -0.75f)))
    }


    @Test
    fun swapsAllMomentBuffersOnlyAfterEveryReplacementIsValid() {
        val arena = FakeArena()
        val state = registry(arena)
        val first = FloatArray(4) { 0.1f * (it + 1) }
        val second = FloatArray(4) { -0.2f * (it + 1) }
        assertTrue(state.synchronizeAll(listOf(first, second)))

        val oldFirst = state.acquire(first) ?: error("First active moment buffer is missing.")
        val oldSecond = state.acquire(second) ?: error("Second active moment buffer is missing.")
        val candidateFirst = arena.allocate(oldFirst.byteCount)
            ?: error("First candidate moment allocation failed.")
        val candidateSecond = arena.allocate(oldSecond.byteCount)
            ?: error("Second candidate moment allocation failed.")
        assertTrue(arena.upload(candidateFirst, ByteArray(candidateFirst.byteCount.toInt())))
        assertTrue(arena.upload(candidateSecond, ByteArray(candidateSecond.byteCount.toInt())))

        arena.releaseFailure = true
        assertTrue(
            state.replaceAll(
                tensors = listOf(first, second),
                expectedAllocations = listOf(oldFirst, oldSecond),
                replacementAllocations = listOf(candidateFirst, candidateSecond)
            )
        )
        assertEquals(candidateFirst.nativeHandle, state.acquire(first)?.nativeHandle)
        assertEquals(candidateSecond.nativeHandle, state.acquire(second)?.nativeHandle)
        assertEquals(64L, state.residentByteCount())

        arena.releaseFailure = false
        assertTrue(state.synchronizeAll(listOf(first, second)))
        assertEquals(32L, state.residentByteCount())
        assertFalse(arena.contents.containsKey(oldFirst.nativeHandle))
        assertFalse(arena.contents.containsKey(oldSecond.nativeHandle))
    }

    @Test
    fun rejectsIncompleteMomentBufferSwapWithoutChangingAnyActiveHandle() {
        val arena = FakeArena()
        val state = registry(arena)
        val first = FloatArray(3)
        val second = FloatArray(3)
        assertTrue(state.synchronizeAll(listOf(first, second)))

        val oldFirst = state.acquire(first) ?: error("First active moment buffer is missing.")
        val oldSecond = state.acquire(second) ?: error("Second active moment buffer is missing.")
        val candidateFirst = arena.allocate(oldFirst.byteCount)
            ?: error("Candidate moment allocation failed.")
        val duplicateCandidate = arena.allocate(oldSecond.byteCount)
            ?: error("Second candidate allocation failed.")

        assertFalse(
            state.replaceAll(
                tensors = listOf(first, second),
                expectedAllocations = listOf(oldFirst, oldFirst),
                replacementAllocations = listOf(candidateFirst, duplicateCandidate)
            )
        )
        assertEquals(oldFirst.nativeHandle, state.acquire(first)?.nativeHandle)
        assertEquals(oldSecond.nativeHandle, state.acquire(second)?.nativeHandle)
        assertEquals(2, state.residentTensorCount())
    }

}
