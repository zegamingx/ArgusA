package com.argus.agi.model

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
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
    fun synchronizesCompleteModelInventoryAndReleasesReplacedTensors() {
        var nextHandle = 1L
        var uploadCount = 0
        val releasedHandles = mutableListOf<Long>()
        val registry = VulkanTensorResidencyRegistry(
            allocate = { byteCount -> VulkanTensorAllocation(nextHandle++, byteCount) },
            upload = { _, _ -> uploadCount++; true },
            release = { allocation -> releasedHandles += allocation.nativeHandle; true }
        )

        val embedding = Float16ParameterArray.fromFloatArray(floatArrayOf(0.1f, 0.2f))
        val outputHead = Float16ParameterArray.fromFloatArray(floatArrayOf(0.3f, 0.4f))
        val lora = Float16ParameterArray.fromFloatArray(floatArrayOf(0.5f, 0.6f))
        val initialInventory = listOf(
            VulkanModelParameterTensor("token.embedding.weight", embedding),
            VulkanModelParameterTensor("lm_head.weight", outputHead),
            VulkanModelParameterTensor("layers.0.adapter.q.a", lora)
        )

        assertTrue(registry.synchronizeAll(initialInventory))
        assertTrue(registry.isSynchronizedWith(initialInventory))
        assertEquals(3, registry.residentTensorCount())
        assertEquals(3, uploadCount)

        assertTrue(registry.synchronizeAll(initialInventory))
        assertEquals("Unchanged tensors must not be uploaded again.", 3, uploadCount)

        lora[0] = 0.9f
        assertFalse(registry.isSynchronizedWith(initialInventory))
        assertTrue(registry.synchronizeAll(initialInventory))
        assertEquals("Only the mutated tensor must be refreshed.", 4, uploadCount)
        assertTrue(registry.isSynchronizedWith(initialInventory))

        val replacementHead = Float16ParameterArray.fromFloatArray(floatArrayOf(0.7f, 0.8f))
        val replacementInventory = listOf(
            initialInventory[0],
            VulkanModelParameterTensor("lm_head.weight", replacementHead),
            initialInventory[2]
        )
        assertTrue(registry.synchronizeAll(replacementInventory))
        assertEquals(3, registry.residentTensorCount())
        assertEquals(1, releasedHandles.size)
        assertTrue(registry.isSynchronizedWith(replacementInventory))
    }

    @Test
    fun rollsBackPartialInventoryWhenARequiredAllocationFails() {
        var nextHandle = 1L
        var allocationCount = 0
        var releaseCount = 0
        val registry = VulkanTensorResidencyRegistry(
            allocate = { byteCount ->
                allocationCount++
                if (allocationCount == 2) null
                else VulkanTensorAllocation(nextHandle++, byteCount)
            },
            upload = { _, _ -> true },
            release = { releaseCount++; true }
        )
        val inventory = listOf(
            VulkanModelParameterTensor(
                "token.embedding.weight",
                Float16ParameterArray.fromFloatArray(floatArrayOf(1f, 2f))
            ),
            VulkanModelParameterTensor(
                "lm_head.weight",
                Float16ParameterArray.fromFloatArray(floatArrayOf(3f, 4f))
            )
        )

        assertFalse(registry.synchronizeAll(inventory))
        assertEquals(0, registry.residentTensorCount())
        assertEquals(1, releaseCount)
        assertFalse(registry.isSynchronizedWith(inventory))
    }

    @Test
    fun rejectsDuplicateTensorNamesAndDuplicateParameterIdentity() {
        val registry = VulkanTensorResidencyRegistry(
            allocate = { byteCount -> VulkanTensorAllocation(1L, byteCount) },
            upload = { _, _ -> true },
            release = { true }
        )
        val parameter = Float16ParameterArray.fromFloatArray(floatArrayOf(1f))
        val duplicateIdentity = listOf(
            VulkanModelParameterTensor("first", parameter),
            VulkanModelParameterTensor("second", parameter)
        )
        val duplicateName = listOf(
            VulkanModelParameterTensor("same", Float16ParameterArray(1)),
            VulkanModelParameterTensor("same", Float16ParameterArray(1))
        )

        org.junit.Assert.assertThrows(IllegalArgumentException::class.java) {
            registry.synchronizeAll(duplicateIdentity)
        }
        org.junit.Assert.assertThrows(IllegalArgumentException::class.java) {
            registry.synchronizeAll(duplicateName)
        }
        assertEquals(0, registry.residentTensorCount())
    }

    @Test
    fun rejectsWholeModelInventoryBeforeAllocatingWhenBudgetIsInsufficient() {
        var allocationCount = 0
        var releaseCount = 0
        val registry = VulkanTensorResidencyRegistry(
            allocate = { byteCount ->
                allocationCount++
                VulkanTensorAllocation(allocationCount.toLong(), byteCount)
            },
            upload = { _, _ -> true },
            release = { releaseCount++; true },
            maxResidentBytes = { 7L }
        )
        val inventory = listOf(
            VulkanModelParameterTensor(
                "embedding",
                Float16ParameterArray.fromFloatArray(floatArrayOf(1f, 2f))
            ),
            VulkanModelParameterTensor(
                "head",
                Float16ParameterArray.fromFloatArray(floatArrayOf(3f, 4f))
            )
        )

        assertFalse(registry.synchronizeAll(inventory))
        assertEquals(0, allocationCount)
        assertEquals(0, releaseCount)
        assertEquals(0, registry.residentTensorCount())
        assertEquals(0L, registry.residentByteCount())
    }

    @Test
    fun tracksResidentBytesAndRejectsAdditionalAllocationsBeyondBudget() {
        var nextHandle = 1L
        val registry = VulkanTensorResidencyRegistry(
            allocate = { byteCount -> VulkanTensorAllocation(nextHandle++, byteCount) },
            upload = { _, _ -> true },
            release = { true },
            maxResidentBytes = { 8L }
        )
        val first = Float16ParameterArray.fromFloatArray(floatArrayOf(1f, 2f))
        val second = Float16ParameterArray.fromFloatArray(floatArrayOf(3f, 4f))
        val third = Float16ParameterArray.fromFloatArray(floatArrayOf(5f))

        assertNotNull(registry.acquire(first))
        assertNotNull(registry.acquire(second))
        assertEquals(8L, registry.residentByteCount())
        assertNull(registry.acquire(third))

        assertTrue(registry.invalidate(releaseAllocations = true))
        assertEquals(0L, registry.residentByteCount())
        assertEquals(0, registry.residentTensorCount())
        assertNotNull(registry.acquire(third))
    }

    @Test
    fun keepsFailedReleasesTrackedUntilTheNativeArenaCanBeCleared() {
        val registry = VulkanTensorResidencyRegistry(
            allocate = { byteCount -> VulkanTensorAllocation(1L, byteCount) },
            upload = { _, _ -> true },
            release = { false },
            maxResidentBytes = { 8L }
        )
        val parameters = Float16ParameterArray.fromFloatArray(floatArrayOf(1f, 2f))

        assertNotNull(registry.acquire(parameters))
        assertFalse(registry.invalidate(releaseAllocations = true))
        assertEquals(1, registry.residentTensorCount())
        assertEquals(4L, registry.residentByteCount())

        assertTrue(registry.invalidate(releaseAllocations = false))
        assertEquals(0, registry.residentTensorCount())
        assertEquals(0L, registry.residentByteCount())
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
    @Test
    fun synchronizesHostParameterMirrorsOnlyAfterEveryGpuReadbackSucceeds() {
        var nextHandle = 1L
        val gpuBytes = LinkedHashMap<Long, ByteArray>()
        var failedHandle: Long? = null
        val registry = VulkanTensorResidencyRegistry(
            allocate = { byteCount ->
                VulkanTensorAllocation(nextHandle++, byteCount)
            },
            upload = { allocation, bytes ->
                gpuBytes[allocation.nativeHandle] = bytes.copyOf()
                true
            },
            download = { allocation, destination ->
                val source = gpuBytes[allocation.nativeHandle]
                when {
                    allocation.nativeHandle == failedHandle -> false
                    source == null || source.size != destination.size -> false
                    else -> {
                        source.copyInto(destination)
                        true
                    }
                }
            },
            release = { allocation ->
                gpuBytes.remove(allocation.nativeHandle) != null
            }
        )
        val first = Float16ParameterArray.fromFloatArray(floatArrayOf(0.25f, -0.5f))
        val second = Float16ParameterArray.fromFloatArray(floatArrayOf(0.75f, -1.0f))
        assertTrue(
            registry.synchronizeAll(
                listOf(
                    VulkanModelParameterTensor("first", first),
                    VulkanModelParameterTensor("second", second)
                )
            )
        )
        val firstAllocation = registry.acquire(first) ?: error("First allocation missing.")
        val secondAllocation = registry.acquire(second) ?: error("Second allocation missing.")
        val previousFirst = first.toShortArray()
        val previousSecond = second.toShortArray()
        registry.markDeviceMutationPending(listOf(first, second))
        gpuBytes[firstAllocation.nativeHandle] =
            Float16ParameterArray.fromFloatArray(floatArrayOf(1.25f, 1.5f)).toLittleEndianByteArray()
        gpuBytes[secondAllocation.nativeHandle] =
            Float16ParameterArray.fromFloatArray(floatArrayOf(1.75f, 2.0f)).toLittleEndianByteArray()
        failedHandle = secondAllocation.nativeHandle

        assertFalse(registry.synchronizeHostMirrors())
        assertArrayEquals(previousFirst, first.toShortArray())
        assertArrayEquals(previousSecond, second.toShortArray())

        failedHandle = null
        assertTrue(registry.synchronizeHostMirrors())
        assertArrayEquals(
            Float16ParameterArray.fromFloatArray(floatArrayOf(1.25f, 1.5f)).toShortArray(),
            first.toShortArray()
        )
        assertArrayEquals(
            Float16ParameterArray.fromFloatArray(floatArrayOf(1.75f, 2.0f)).toShortArray(),
            second.toShortArray()
        )
        assertTrue(registry.isSynchronizedWith(
            listOf(
                VulkanModelParameterTensor("first", first),
                VulkanModelParameterTensor("second", second)
            )
        ))
    }

    @Test
    fun synchronizesOnlyTheRequestedStaleParameterTensor() {
        var nextHandle = 1L
        var uploadCount = 0
        val gpuBytes = LinkedHashMap<Long, ByteArray>()
        val registry = VulkanTensorResidencyRegistry(
            allocate = { byteCount ->
                VulkanTensorAllocation(nextHandle++, byteCount).also { allocation ->
                    gpuBytes[allocation.nativeHandle] = ByteArray(byteCount.toInt())
                }
            },
            upload = { allocation, bytes ->
                bytes.copyInto(requireNotNull(gpuBytes[allocation.nativeHandle]))
                uploadCount++
                true
            },
            release = { allocation -> gpuBytes.remove(allocation.nativeHandle) != null },
            download = { allocation, destination ->
                val source = gpuBytes[allocation.nativeHandle]
                if (source == null || source.size != destination.size) {
                    false
                } else {
                    source.copyInto(destination)
                    true
                }
            }
        )
        val first = Float16ParameterArray.fromFloatArray(floatArrayOf(0.25f, -0.5f))
        val second = Float16ParameterArray.fromFloatArray(floatArrayOf(0.75f, -1.0f))
        val inventory = listOf(
            VulkanModelParameterTensor("first", first),
            VulkanModelParameterTensor("second", second)
        )
        assertTrue(registry.synchronizeAll(inventory))
        val firstAllocation = registry.acquire(first) ?: error("First resident parameter is missing.")
        val secondBefore = second.toShortArray()
        val firstGpuBytes = Float16ParameterArray.fromFloatArray(
            floatArrayOf(1.25f, 1.5f)
        ).toLittleEndianByteArray()

        registry.markDeviceMutationPending(listOf(first))
        firstGpuBytes.copyInto(requireNotNull(gpuBytes[firstAllocation.nativeHandle]))

        assertTrue(registry.synchronizeHostMirror(first))
        assertArrayEquals(
            Float16ParameterArray.fromFloatArray(floatArrayOf(1.25f, 1.5f)).toShortArray(),
            first.toShortArray()
        )
        assertArrayEquals(secondBefore, second.toShortArray())
        assertTrue(registry.areHostMirrorsCurrent())
        assertEquals(2, uploadCount)
        assertEquals(firstAllocation.nativeHandle, registry.acquire(first)?.nativeHandle)
    }

    @Test
    fun replacesAllResidentParameterHandlesWithoutUploadingHostSnapshots() {
        var nextHandle = 1L
        var uploadCount = 0
        val gpuBytes = LinkedHashMap<Long, ByteArray>()
        val registry = VulkanTensorResidencyRegistry(
            allocate = { byteCount ->
                VulkanTensorAllocation(nextHandle++, byteCount).also { allocation ->
                    gpuBytes[allocation.nativeHandle] = ByteArray(byteCount.toInt())
                }
            },
            upload = { allocation, bytes ->
                val destination = gpuBytes[allocation.nativeHandle]
                if (destination == null || destination.size != bytes.size) {
                    false
                } else {
                    bytes.copyInto(destination)
                    uploadCount++
                    true
                }
            },
            release = { allocation -> gpuBytes.remove(allocation.nativeHandle) != null },
            download = { allocation, destination ->
                val source = gpuBytes[allocation.nativeHandle]
                if (source == null || source.size != destination.size) {
                    false
                } else {
                    source.copyInto(destination)
                    true
                }
            }
        )
        val first = Float16ParameterArray.fromFloatArray(floatArrayOf(0.25f, -0.5f, 0.75f))
        val second = Float16ParameterArray.fromFloatArray(floatArrayOf(-0.125f, 0.375f))
        val inventory = listOf(
            VulkanModelParameterTensor("first", first),
            VulkanModelParameterTensor("second", second)
        )
        assertTrue(registry.synchronizeAll(inventory))
        val previousUploads = uploadCount
        val oldAllocations = listOf(
            registry.acquire(first) ?: error("First active parameter is missing."),
            registry.acquire(second) ?: error("Second active parameter is missing.")
        )
        val newValues = listOf(
            floatArrayOf(1.0f, 1.5f, -1.25f),
            floatArrayOf(0.5f, -0.75f)
        )
        val replacements = newValues.mapIndexed { index, values ->
            val allocation = VulkanTensorAllocation(nextHandle++, oldAllocations[index].byteCount)
            gpuBytes[allocation.nativeHandle] = Float16ParameterArray
                .fromFloatArray(values)
                .toLittleEndianByteArray()
            allocation
        }

        assertTrue(registry.canReplaceAll(
            listOf(first, second),
            oldAllocations,
            replacements
        ))
        assertTrue(registry.replaceAll(
            listOf(first, second),
            oldAllocations,
            replacements
        ))

        assertEquals(replacements[0].nativeHandle, registry.acquire(first)?.nativeHandle)
        assertEquals(replacements[1].nativeHandle, registry.acquire(second)?.nativeHandle)
        assertEquals(previousUploads, uploadCount)
        assertFalse(gpuBytes.containsKey(oldAllocations[0].nativeHandle))
        assertFalse(gpuBytes.containsKey(oldAllocations[1].nativeHandle))
        assertTrue(registry.synchronizeHostMirrors())
        assertArrayEquals(
            Float16ParameterArray.fromFloatArray(newValues[0]).toShortArray(),
            first.toShortArray()
        )
        assertArrayEquals(
            Float16ParameterArray.fromFloatArray(newValues[1]).toShortArray(),
            second.toShortArray()
        )
    }


}
