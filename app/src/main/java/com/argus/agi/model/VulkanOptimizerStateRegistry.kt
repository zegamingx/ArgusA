package com.argus.agi.model

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.IdentityHashMap

/**
 * Keeps FP32 AdamW moments resident in the Vulkan Tensor Arena.
 *
 * The engine-owned arrays remain the recovery/checkpoint mirror. They are updated
 * only after a complete optimizer transaction succeeds. A CPU fallback invalidates
 * this registry before changing the mirror, so a later GPU attempt uploads fresh state.
 */
internal class VulkanOptimizerStateRegistry(
    private val allocate: (Long) -> VulkanTensorAllocation?,
    private val upload: (VulkanTensorAllocation, ByteArray) -> Boolean,
    private val release: (VulkanTensorAllocation) -> Boolean,
    private val download: ((VulkanTensorAllocation, ByteArray) -> Boolean)? = null,
    private val maxResidentBytes: () -> Long = { Long.MAX_VALUE }
) {
    private data class Entry(
        var allocation: VulkanTensorAllocation,
        var uploaded: Boolean,
        var hostMirrorCurrent: Boolean
    )

    private data class PreparedReplacement(
        val tensors: List<FloatArray>,
        val expectedAllocations: List<VulkanTensorAllocation>,
        val replacementAllocations: List<VulkanTensorAllocation>,
        val retiredBytesAfterCommit: Long
    )

    private var preparedReplacement: PreparedReplacement? = null

    private val entries = IdentityHashMap<FloatArray, Entry>()
    private val retiredAllocations = ArrayList<VulkanTensorAllocation>()
    private var activeResidentBytes: Long = 0L
    private var retiredResidentBytes: Long = 0L

    @Synchronized
    fun synchronizeAll(tensors: List<FloatArray>): Boolean {
        preparedReplacement = null
        require(tensors.isNotEmpty()) {
            "The AdamW optimizer-state inventory cannot be empty."
        }

        val desired = IdentityHashMap<FloatArray, Boolean>()
        var requiredBytes = 0L
        for (tensor in tensors) {
            require(tensor.isNotEmpty()) {
                "AdamW optimizer-state tensors cannot be empty."
            }
            val existing = entries[tensor]
            if (existing == null || existing.hostMirrorCurrent) {
                require(tensor.all { value -> value.isFinite() }) {
                    "AdamW optimizer-state tensors must contain only finite values."
                }
            }
            require(desired.put(tensor, true) == null) {
                "An AdamW state array cannot be registered more than once."
            }
            val byteCount = byteCount(tensor) ?: return false
            requiredBytes = runCatching {
                Math.addExact(requiredBytes, byteCount)
            }.getOrNull() ?: return false
        }

        if (!releaseRetiredAllocations()) return false
        if (requiredBytes > maxResidentBytes().coerceAtLeast(0L)) return false

        val staleArrays = entries.keys.filter { tensor -> !desired.containsKey(tensor) }
        for (tensor in staleArrays) {
            val entry = entries[tensor] ?: continue
            if (!release(entry.allocation)) return false
            removeEntry(tensor, entry)
        }

        for (tensor in tensors) {
            if (acquireInternal(tensor) == null) {
                invalidate(releaseAllocations = true)
                return false
            }
        }

        return entries.size == tensors.size &&
            tensors.all { tensor -> entries[tensor]?.uploaded == true } &&
            activeResidentBytes == requiredBytes &&
            activeResidentBytes <= maxResidentBytes().coerceAtLeast(0L)
    }

    @Synchronized
    fun acquire(tensor: FloatArray): VulkanTensorAllocation? = acquireInternal(tensor)

    /**
     * Refreshes the engine-owned recovery mirror from persistent GPU moments.
     * The operation stages every tensor first, then commits all host arrays together.
     */
    @Synchronized
    fun synchronizeHostMirrors(): Boolean {
        if (entries.isEmpty()) return true
        val staleEntries = entries.entries.filter { (_, entry) -> !entry.hostMirrorCurrent }
        if (staleEntries.isEmpty()) return true
        val readback = download ?: return false
        val staged = ArrayList<Pair<FloatArray, FloatArray>>(staleEntries.size)
        for ((tensor, entry) in staleEntries) {
            if (!entry.uploaded) return false
            val requiredBytes = byteCount(tensor) ?: return false
            if (requiredBytes > Int.MAX_VALUE.toLong()) return false
            val bytes = try {
                ByteArray(requiredBytes.toInt())
            } catch (_: OutOfMemoryError) {
                return false
            }
            if (!readback(entry.allocation, bytes)) return false
            if (bytes.size % Float.SIZE_BYTES != 0) return false
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val values = FloatArray(bytes.size / Float.SIZE_BYTES) { buffer.float }
            if (values.any { value -> !value.isFinite() }) return false
            staged.add(tensor to values)
        }
        for ((tensor, values) in staged) {
            System.arraycopy(values, 0, tensor, 0, tensor.size)
            val entry = entries[tensor] ?: return false
            entry.hostMirrorCurrent = true
        }
        return true
    }

    @Synchronized
    fun areHostMirrorsCurrent(): Boolean =
        entries.values.all { entry -> entry.hostMirrorCurrent }
    /** Checks a complete candidate replacement without changing active optimizer state. */
    @Synchronized
    fun canReplaceAll(
        tensors: List<FloatArray>,
        expectedAllocations: List<VulkanTensorAllocation>,
        replacementAllocations: List<VulkanTensorAllocation>
    ): Boolean {
        preparedReplacement = null
        if (!validateReplacement(tensors, expectedAllocations, replacementAllocations)) return false

        val nextRetiredBytes = runCatching {
            expectedAllocations.fold(retiredResidentBytes) { total, allocation ->
                Math.addExact(total, allocation.byteCount)
            }
        }.getOrNull() ?: return false
        val preparedTensors = tensors.toList()
        val preparedExpected = expectedAllocations.toList()
        val preparedReplacements = replacementAllocations.toList()
        return runCatching {
            retiredAllocations.ensureCapacity(
                Math.addExact(retiredAllocations.size, preparedExpected.size)
            )
            preparedReplacement = PreparedReplacement(
                tensors = preparedTensors,
                expectedAllocations = preparedExpected,
                replacementAllocations = preparedReplacements,
                retiredBytesAfterCommit = nextRetiredBytes
            )
            true
        }.getOrDefault(false)
    }

    /** Publishes the prepared moment buffers without allocating during commit. */
    @Synchronized
    internal fun commitReplacementAfterPreflight(
        tensors: List<FloatArray>,
        expectedAllocations: List<VulkanTensorAllocation>,
        replacementAllocations: List<VulkanTensorAllocation>
    ): Boolean {
        val prepared = preparedReplacement ?: return false
        if (tensors.size != prepared.tensors.size ||
            expectedAllocations.size != prepared.expectedAllocations.size ||
            replacementAllocations.size != prepared.replacementAllocations.size) {
            return false
        }

        for (index in prepared.tensors.indices) {
            if (prepared.tensors[index] !== tensors[index] ||
                prepared.expectedAllocations[index].nativeHandle != expectedAllocations[index].nativeHandle ||
                prepared.replacementAllocations[index].nativeHandle != replacementAllocations[index].nativeHandle) {
                return false
            }
            val entry = entries[prepared.tensors[index]] ?: return false
            val expected = prepared.expectedAllocations[index]
            val replacement = prepared.replacementAllocations[index]
            if (entry.allocation.nativeHandle != expected.nativeHandle ||
                entry.allocation.byteCount != expected.byteCount ||
                replacement.byteCount != expected.byteCount) {
                return false
            }
        }

        for (index in prepared.tensors.indices) {
            val tensor = prepared.tensors[index]
            val entry = entries[tensor] ?: return false
            val previous = entry.allocation
            entry.allocation = prepared.replacementAllocations[index]
            entry.uploaded = true
            entry.hostMirrorCurrent = false
            retiredAllocations.add(previous)
        }
        retiredResidentBytes = prepared.retiredBytesAfterCommit
        preparedReplacement = null
        return true
    }

    /** Releases old moment buffers after the paired parameter registry has committed. */
    @Synchronized
    internal fun finishReplacementAfterPreflight(): Boolean =
        releaseRetiredAllocations()
    private fun validateReplacement(
        tensors: List<FloatArray>,
        expectedAllocations: List<VulkanTensorAllocation>,
        replacementAllocations: List<VulkanTensorAllocation>
    ): Boolean {
        if (tensors.isEmpty() ||
            tensors.size != expectedAllocations.size ||
            tensors.size != replacementAllocations.size ||
            entries.size != tensors.size) {
            return false
        }

        val seenTensors = IdentityHashMap<FloatArray, Boolean>()
        val seenReplacementHandles = HashSet<Long>()
        for (index in tensors.indices) {
            val tensor = tensors[index]
            if (seenTensors.put(tensor, true) != null) return false
            val entry = entries[tensor] ?: return false
            val expected = expectedAllocations[index]
            val replacement = replacementAllocations[index]
            if (entry.allocation.nativeHandle != expected.nativeHandle ||
                entry.allocation.byteCount != expected.byteCount ||
                replacement.byteCount != expected.byteCount ||
                replacement.nativeHandle <= 0L ||
                !seenReplacementHandles.add(replacement.nativeHandle)) {
                return false
            }
            if (entries.values.any { active ->
                    active.allocation.nativeHandle == replacement.nativeHandle
                } ||
                retiredAllocations.any { retired ->
                    retired.nativeHandle == replacement.nativeHandle
                }) {
                return false
            }
        }
        return true
    }

    @Synchronized
    fun invalidate(releaseAllocations: Boolean): Boolean {
        preparedReplacement = null
        if (!releaseAllocations) {
            entries.clear()
            retiredAllocations.clear()
            activeResidentBytes = 0L
            retiredResidentBytes = 0L
            return true
        }

        var allReleased = true
        val snapshot = entries.entries.map { it.key to it.value }
        for ((tensor, entry) in snapshot) {
            if (release(entry.allocation)) {
                removeEntry(tensor, entry)
            } else {
                allReleased = false
            }
        }
        if (!releaseRetiredAllocations()) allReleased = false
        return allReleased
    }

    @Synchronized
    fun residentTensorCount(): Int = entries.size

    /** Includes active tensors and allocations awaiting a successful deferred release. */
    @Synchronized
    fun residentByteCount(): Long = Math.addExact(activeResidentBytes, retiredResidentBytes)

    private fun releaseRetiredAllocations(): Boolean {
        var allReleased = true
        val iterator = retiredAllocations.iterator()
        while (iterator.hasNext()) {
            val allocation = iterator.next()
            if (release(allocation)) {
                retiredResidentBytes =
                    (retiredResidentBytes - allocation.byteCount).coerceAtLeast(0L)
                iterator.remove()
            } else {
                allReleased = false
            }
        }
        return allReleased
    }

    private fun acquireInternal(tensor: FloatArray): VulkanTensorAllocation? {
        val requiredBytes = byteCount(tensor) ?: return null
        var entry = entries[tensor]

        if (entry != null && entry.allocation.byteCount != requiredBytes) {
            if (!release(entry.allocation)) return null
            removeEntry(tensor, entry)
            entry = null
        }

        if (entry == null) {
            val nextActiveBytes = runCatching {
                Math.addExact(activeResidentBytes, requiredBytes)
            }.getOrNull() ?: return null
            val nextTotalBytes = runCatching {
                Math.addExact(nextActiveBytes, retiredResidentBytes)
            }.getOrNull() ?: return null
            if (nextTotalBytes > maxResidentBytes().coerceAtLeast(0L)) return null

            val allocation = allocate(requiredBytes) ?: return null
            entry = Entry(
                allocation = allocation,
                uploaded = false,
                hostMirrorCurrent = true
            )
            entries[tensor] = entry
            activeResidentBytes = nextActiveBytes
        }

        if (entry.uploaded) return entry.allocation

        val snapshot = try {
            encodeFloatArray(tensor)
        } catch (_: OutOfMemoryError) {
            discard(tensor, entry)
            return null
        }

        if (!upload(entry.allocation, snapshot)) {
            discard(tensor, entry)
            return null
        }

        entry.uploaded = true
        entry.hostMirrorCurrent = true
        return entry.allocation
    }

    private fun discard(tensor: FloatArray, entry: Entry) {
        if (release(entry.allocation)) {
            removeEntry(tensor, entry)
        } else {
            entry.uploaded = false
        }
    }

    private fun removeEntry(tensor: FloatArray, entry: Entry) {
        if (entries.remove(tensor) != null) {
            activeResidentBytes =
                (activeResidentBytes - entry.allocation.byteCount).coerceAtLeast(0L)
        }
    }

    private fun byteCount(tensor: FloatArray): Long? {
        if (tensor.isEmpty()) return null
        return runCatching {
            Math.multiplyExact(tensor.size.toLong(), Float.SIZE_BYTES.toLong())
        }.getOrNull()?.takeIf { it > 0L && it <= Int.MAX_VALUE.toLong() }
    }

    private fun encodeFloatArray(values: FloatArray): ByteArray {
        val byteCount = Math.multiplyExact(values.size, Float.SIZE_BYTES)
        val buffer = ByteBuffer.allocate(byteCount).order(ByteOrder.LITTLE_ENDIAN)
        for (value in values) {
            require(value.isFinite()) {
                "AdamW optimizer-state values must remain finite."
            }
            buffer.putFloat(value)
        }
        return buffer.array()
    }
}
