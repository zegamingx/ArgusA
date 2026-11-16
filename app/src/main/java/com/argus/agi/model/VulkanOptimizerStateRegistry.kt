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
    private val maxResidentBytes: () -> Long = { Long.MAX_VALUE }
) {
    private data class Entry(
        val allocation: VulkanTensorAllocation,
        var uploaded: Boolean
    )

    private val entries = IdentityHashMap<FloatArray, Entry>()
    private var residentBytes: Long = 0L

    @Synchronized
    fun synchronizeAll(tensors: List<FloatArray>): Boolean {
        require(tensors.isNotEmpty()) {
            "The AdamW optimizer-state inventory cannot be empty."
        }

        val desired = IdentityHashMap<FloatArray, Boolean>()
        var requiredBytes = 0L
        for (tensor in tensors) {
            require(tensor.isNotEmpty()) {
                "AdamW optimizer-state tensors cannot be empty."
            }
            require(tensor.all(Float::isFinite)) {
                "AdamW optimizer-state tensors must contain only finite values."
            }
            require(desired.put(tensor, true) == null) {
                "An AdamW state array cannot be registered more than once."
            }
            val byteCount = byteCount(tensor) ?: return false
            requiredBytes = runCatching {
                Math.addExact(requiredBytes, byteCount)
            }.getOrNull() ?: return false
        }

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
            residentBytes == requiredBytes &&
            residentBytes <= maxResidentBytes().coerceAtLeast(0L)
    }

    @Synchronized
    fun acquire(tensor: FloatArray): VulkanTensorAllocation? = acquireInternal(tensor)

    @Synchronized
    fun invalidate(releaseAllocations: Boolean): Boolean {
        if (!releaseAllocations) {
            entries.clear()
            residentBytes = 0L
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
        return allReleased
    }

    @Synchronized
    fun residentTensorCount(): Int = entries.size

    @Synchronized
    fun residentByteCount(): Long = residentBytes

    private fun acquireInternal(tensor: FloatArray): VulkanTensorAllocation? {
        val requiredBytes = byteCount(tensor) ?: return null
        var entry = entries[tensor]

        if (entry != null && entry.allocation.byteCount != requiredBytes) {
            if (!release(entry.allocation)) return null
            removeEntry(tensor, entry)
            entry = null
        }

        if (entry == null) {
            val nextBytes = runCatching {
                Math.addExact(residentBytes, requiredBytes)
            }.getOrNull() ?: return null
            if (nextBytes > maxResidentBytes().coerceAtLeast(0L)) return null

            val allocation = allocate(requiredBytes) ?: return null
            entry = Entry(allocation, uploaded = false)
            entries[tensor] = entry
            residentBytes = nextBytes
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
            residentBytes = (residentBytes - entry.allocation.byteCount).coerceAtLeast(0L)
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
