package com.argus.agi.model

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test

class VulkanTensorResidencyRegistryTest {

    @Test
    fun keepsStableTensorResidentUntilMutationRevisionChanges() {
        var nextHandle = 1L
        var uploadCount = 0
        val registry = VulkanTensorResidencyRegistry(
            allocate = { byteCount -> VulkanTensorAllocation(nextHandle++, byteCount) },
            upload = { _, _ -> uploadCount++; true },
            release = { true }
        )
        val weights = Float16ParameterArray.fromFloatArray(floatArrayOf(0.5f, -0.25f, 1.0f))

        val first = registry.acquire(weights)
        val second = registry.acquire(weights)
        assertNotNull(first)
        assertNotNull(second)
        assertEquals(first?.nativeHandle, second?.nativeHandle)
        assertEquals(1, uploadCount)
        assertEquals(1, registry.residentTensorCount())

        weights[1] = 0.75f
        val afterMutation = registry.acquire(weights)
        assertEquals(first?.nativeHandle, afterMutation?.nativeHandle)
        assertEquals(2, uploadCount)
    }

    @Test
    fun keepsIndependentBatchTensorsResidentAndRefreshesOnlyMutatedWeights() {
        var nextHandle = 1L
        val uploadedHandles = mutableListOf<Long>()
        val registry = VulkanTensorResidencyRegistry(
            allocate = { byteCount -> VulkanTensorAllocation(nextHandle++, byteCount) },
            upload = { allocation, _ -> uploadedHandles += allocation.nativeHandle; true },
            release = { true }
        )
        val queryWeights = Float16ParameterArray.fromFloatArray(floatArrayOf(0.1f, 0.2f))
        val keyWeights = Float16ParameterArray.fromFloatArray(floatArrayOf(0.3f, 0.4f))
        val valueWeights = Float16ParameterArray.fromFloatArray(floatArrayOf(0.5f, 0.6f))

        val firstBatch = listOf(queryWeights, keyWeights, valueWeights).map(registry::acquire)
        val secondBatch = listOf(queryWeights, keyWeights, valueWeights).map(registry::acquire)

        assertEquals(3, firstBatch.mapNotNull { it?.nativeHandle }.distinct().size)
        assertEquals(
            firstBatch.map { it?.nativeHandle },
            secondBatch.map { it?.nativeHandle }
        )
        assertEquals(3, uploadedHandles.size)
        assertEquals(3, registry.residentTensorCount())

        keyWeights[0] = 0.9f
        registry.acquire(queryWeights)
        registry.acquire(keyWeights)
        registry.acquire(valueWeights)

        assertEquals(4, uploadedHandles.size)
        assertEquals(3, registry.residentTensorCount())
    }

    @Test
    fun serializesHalfValuesInLittleEndianOrder() {
        val weights = Float16ParameterArray.fromFloatArray(floatArrayOf(1.0f, -2.0f))
        assertArrayEquals(
            byteArrayOf(0x00, 0x3C, 0x00, 0xC0.toByte()),
            weights.toLittleEndianByteArray()
        )
    }

    @Test
    fun releasesAllocationWhenUploadFails() {
        var releaseCount = 0
        val registry = VulkanTensorResidencyRegistry(
            allocate = { byteCount -> VulkanTensorAllocation(7L, byteCount) },
            upload = { _, _ -> false },
            release = { releaseCount++; true }
        )
        val allocation = registry.acquire(Float16ParameterArray.fromFloatArray(floatArrayOf(1f)))
        assertNull(allocation)
        assertEquals(1, releaseCount)
        assertEquals(0, registry.residentTensorCount())
    }

    @Test
    fun invalidationReleasesAllResidentTensors() {
        var nextHandle = 1L
        var releaseCount = 0
        val registry = VulkanTensorResidencyRegistry(
            allocate = { byteCount -> VulkanTensorAllocation(nextHandle++, byteCount) },
            upload = { _, _ -> true },
            release = { releaseCount++; true }
        )
        assertNotNull(registry.acquire(Float16ParameterArray.fromFloatArray(floatArrayOf(1f))))
        assertNotNull(registry.acquire(Float16ParameterArray.fromFloatArray(floatArrayOf(2f, 3f))))
        assertEquals(2, registry.residentTensorCount())

        registry.invalidate(releaseAllocations = true)
        assertEquals(2, releaseCount)
        assertEquals(0, registry.residentTensorCount())
    }
}
