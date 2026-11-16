package com.argus.agi.model

import java.util.IdentityHashMap

/** A named reference to one authoritative Kotlin model-parameter tensor. */
internal data class VulkanModelParameterTensor(
    val name: String,
    val parameters: Float16ParameterArray
)

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
        val entry: Entry = candidate

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

    /**
     * Reconciles the full model inventory and releases tensors no longer owned by it.
     * On failure, all entries are invalidated so partial residency is never reported ready.
     */
    @Synchronized
    fun synchronizeAll(tensors: List<VulkanModelParameterTensor>): Boolean {
        require(tensors.isNotEmpty()) {
            "The model parameter residency inventory cannot be empty."
        }

        val desired = IdentityHashMap<Float16ParameterArray, String>()
        val names = HashSet<String>(tensors.size)
        for (tensor in tensors) {
            require(tensor.name.isNotBlank()) {
                "Model parameter tensor names cannot be blank."
            }
            require(names.add(tensor.name)) {
                "Model parameter tensor names must be unique: ${tensor.name}"
            }
            require(desired.put(tensor.parameters, tensor.name) == null) {
                "A model parameter array cannot be registered under multiple names."
            }
        }

        val staleParameters = entries.keys.filter { parameters ->
            !desired.containsKey(parameters)
        }
        for (parameters in staleParameters) {
            val entry = entries[parameters] ?: continue
            if (!release(entry.allocation)) {
                invalidate(releaseAllocations = true)
                return false
            }
            entries.remove(parameters)
        }

        for (tensor in tensors) {
            if (acquire(tensor.parameters) == null) {
                invalidate(releaseAllocations = true)
                return false
            }
        }
        return isSynchronizedWith(tensors)
    }

    @Synchronized
    fun isSynchronizedWith(tensors: List<VulkanModelParameterTensor>): Boolean {
        if (tensors.isEmpty() || entries.size != tensors.size) return false

        val seen = IdentityHashMap<Float16ParameterArray, Boolean>()
        for (tensor in tensors) {
            if (seen.put(tensor.parameters, true) != null) return false
            val entry = entries[tensor.parameters] ?: return false
            if (entry.uploadedRevision != tensor.parameters.mutationVersion) return false
        }
        return true
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
