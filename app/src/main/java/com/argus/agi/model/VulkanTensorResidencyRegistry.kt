package com.argus.agi.model

import java.util.IdentityHashMap

/**
 * Keeps read-only GPU replicas of authoritative Kotlin FP16 model tensors.
 * Kotlin remains the source of truth; uploads occur only after mutation revisions change.
 */
internal class VulkanTensorResidencyRegistry(
    private val allocate: (Long) -> VulkanTensorAllocation?,
    private val upload: (VulkanTensorAllocation, ByteArray) -> Boolean,
    private val release: (VulkanTensorAllocation) -> Boolean
) {
    private data class Entry(
        val allocation: VulkanTensorAllocation,
        var uploadedRevision: Long
    )

    private val entries = IdentityHashMap<Float16ParameterArray, Entry>()

    @Synchronized
    fun acquire(parameters: Float16ParameterArray): VulkanTensorAllocation? {
        val byteCount = runCatching {
            Math.multiplyExact(parameters.size.toLong(), Short.SIZE_BYTES.toLong())
        }.getOrNull() ?: return null
        if (byteCount <= 0L || byteCount > Int.MAX_VALUE.toLong()) return null

        var candidate = entries[parameters]
        if (candidate != null && candidate.allocation.byteCount != byteCount) {
            release(candidate.allocation)
            entries.remove(parameters)
            candidate = null
        }
        if (candidate == null) {
            val allocation = allocate(byteCount) ?: return null
            candidate = Entry(allocation, uploadedRevision = Long.MIN_VALUE)
            entries[parameters] = candidate
        }
        val entry = candidate ?: return null

        repeat(MAX_SNAPSHOT_ATTEMPTS) {
            val revisionBefore = parameters.mutationVersion
            if (entry.uploadedRevision == revisionBefore) return entry.allocation

            val snapshot = try {
                parameters.toLittleEndianByteArray()
            } catch (_: OutOfMemoryError) {
                discard(parameters, entry)
                return null
            }
            val revisionAfterSnapshot = parameters.mutationVersion
            if (revisionBefore != revisionAfterSnapshot) return@repeat

            if (!upload(entry.allocation, snapshot)) {
                discard(parameters, entry)
                return null
            }
            val revisionAfterUpload = parameters.mutationVersion
            if (revisionAfterUpload == revisionAfterSnapshot) {
                entry.uploadedRevision = revisionAfterUpload
                return entry.allocation
            }
        }

        if (entry.uploadedRevision == Long.MIN_VALUE) discard(parameters, entry)
        return null
    }

    @Synchronized
    fun invalidate(releaseAllocations: Boolean) {
        if (releaseAllocations) entries.values.forEach { entry -> release(entry.allocation) }
        entries.clear()
    }

    @Synchronized
    fun residentTensorCount(): Int = entries.size

    private fun discard(parameters: Float16ParameterArray, entry: Entry) {
        entries.remove(parameters)
        release(entry.allocation)
    }

    private companion object {
        const val MAX_SNAPSHOT_ATTEMPTS = 3
    }
}
