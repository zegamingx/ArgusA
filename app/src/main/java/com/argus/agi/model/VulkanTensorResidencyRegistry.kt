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
 *
 * The registry refuses partial model residency: its configured byte budget is checked
 * against the complete inventory before any allocation is created.
 */
internal class VulkanTensorResidencyRegistry(
    private val allocate: (Long) -> VulkanTensorAllocation?,
    private val upload: (VulkanTensorAllocation, ByteArray) -> Boolean,
    private val release: (VulkanTensorAllocation) -> Boolean,
    private val maxResidentBytes: () -> Long = { Long.MAX_VALUE }
) {
    private data class Entry(
        val allocation: VulkanTensorAllocation,
        var uploadedRevision: Long
    )

    private val entries = IdentityHashMap<Float16ParameterArray, Entry>()
    private var residentBytes: Long = 0L

    @Synchronized
    fun acquire(parameters: Float16ParameterArray): VulkanTensorAllocation? {
        val byteCount = parameterByteCount(parameters) ?: return null
        var candidate = entries[parameters]

        if (candidate != null && candidate.allocation.byteCount != byteCount) {
            if (!release(candidate.allocation)) return null
            removeEntry(parameters, candidate)
            candidate = null
        }

        if (candidate == null) {
            val nextResidentBytes = runCatching {
                Math.addExact(residentBytes, byteCount)
            }.getOrNull() ?: return null
            if (nextResidentBytes > maxResidentBytes().coerceAtLeast(0L)) return null

            val allocation = allocate(byteCount) ?: return null
            candidate = Entry(allocation, uploadedRevision = Long.MIN_VALUE)
            entries[parameters] = candidate
            residentBytes = nextResidentBytes
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
     * The complete byte budget is validated before old or new allocations are changed.
     */
    @Synchronized
    fun synchronizeAll(tensors: List<VulkanModelParameterTensor>): Boolean {
        require(tensors.isNotEmpty()) {
            "The model parameter residency inventory cannot be empty."
        }

        val desired = IdentityHashMap<Float16ParameterArray, String>()
        val names = HashSet<String>(tensors.size)
        var requiredBytes = 0L
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
            val byteCount = parameterByteCount(tensor.parameters) ?: return false
            requiredBytes = runCatching {
                Math.addExact(requiredBytes, byteCount)
            }.getOrNull() ?: return false
        }

        if (requiredBytes > maxResidentBytes().coerceAtLeast(0L)) return false

        var releaseSucceeded = true
        val staleParameters = entries.keys.filter { parameters ->
            !desired.containsKey(parameters)
        }
        for (parameters in staleParameters) {
            val entry = entries[parameters] ?: continue
            if (release(entry.allocation)) {
                removeEntry(parameters, entry)
            } else {
                releaseSucceeded = false
            }
        }
        if (!releaseSucceeded) {
            invalidate(releaseAllocations = true)
            return false
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
        return residentBytes <= maxResidentBytes().coerceAtLeast(0L)
    }

    /**
     * Releases every tracked allocation. Failed releases remain tracked so the caller
     * can retry or invalidate the native arena as a whole without losing their handles.
     */
    @Synchronized
    fun invalidate(releaseAllocations: Boolean): Boolean {
        if (!releaseAllocations) {
            entries.clear()
            residentBytes = 0L
            return true
        }

        var allReleased = true
        val snapshot = entries.entries.map { it.key to it.value }
        for ((parameters, entry) in snapshot) {
            if (release(entry.allocation)) {
                removeEntry(parameters, entry)
            } else {
                allReleased = false
            }
        }
        return allReleased
    }


    /**
     * Stages every resident parameter tensor from GPU before committing any host mirror.
     * This is required before CPU fallback, native runtime teardown, or checkpoint writing.
     */
    @Synchronized
    fun synchronizeHostMirrors(): Boolean {
        if (entries.isEmpty()) return true
        val readback = download ?: return false
        val staged = ArrayList<Pair<Float16ParameterArray, ByteArray>>(entries.size)
        for ((parameters, entry) in entries) {
            if (entry.uploadedRevision == Long.MIN_VALUE) return false
            val byteCount = parameterByteCount(parameters) ?: return false
            if (byteCount != entry.allocation.byteCount || byteCount > Int.MAX_VALUE.toLong()) {
                return false
            }
            val snapshot = try {
                ByteArray(byteCount.toInt())
            } catch (_: OutOfMemoryError) {
                return false
            }
            if (!readback(entry.allocation, snapshot)) return false
            try {
                parameters.validateLittleEndianPrefix(snapshot, parameters.size)
            } catch (_: IllegalArgumentException) {
                return false
            }
            staged.add(parameters to snapshot)
        }

        for ((parameters, snapshot) in staged) {
            parameters.copyPrefixFromLittleEndianByteArray(snapshot, parameters.size)
            val entry = entries[parameters] ?: return false
            entry.uploadedRevision = parameters.mutationVersion
        }
        return true
    }

    /**
     * Records a completed device-side parameter update without scheduling a redundant
     * host-to-device upload. The caller must first commit the matching GPU readback to
     * the Float16ParameterArray while holding the model-update lock.
     */
    @Synchronized
    fun markGpuReplicaSynchronized(
        parameters: Float16ParameterArray,
        allocation: VulkanTensorAllocation
    ): Boolean {
        val entry = entries[parameters] ?: return false
        if (entry.allocation.nativeHandle != allocation.nativeHandle ||
            entry.allocation.byteCount != allocation.byteCount) {
            return false
        }
        entry.uploadedRevision = parameters.mutationVersion
        return true
    }

    @Synchronized
    fun residentTensorCount(): Int = entries.size

    @Synchronized
    fun residentByteCount(): Long = residentBytes

    private fun discard(parameters: Float16ParameterArray, entry: Entry) {
        if (release(entry.allocation)) {
            removeEntry(parameters, entry)
        } else {
            entry.uploadedRevision = Long.MIN_VALUE
        }
    }

    private fun removeEntry(parameters: Float16ParameterArray, entry: Entry) {
        if (entries.remove(parameters) != null) {
            residentBytes = (residentBytes - entry.allocation.byteCount).coerceAtLeast(0L)
        }
    }

    private fun parameterByteCount(parameters: Float16ParameterArray): Long? {
        val bytes = runCatching {
            Math.multiplyExact(parameters.size.toLong(), Short.SIZE_BYTES.toLong())
        }.getOrNull() ?: return null
        return bytes.takeIf { it > 0L && it <= Int.MAX_VALUE.toLong() }
    }

    private companion object {
        const val MAX_SNAPSHOT_ATTEMPTS = 3
    }
}
