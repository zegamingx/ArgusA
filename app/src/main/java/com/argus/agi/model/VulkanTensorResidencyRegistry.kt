package com.argus.agi.model

import java.util.IdentityHashMap

/** A named reference to one authoritative model-parameter tensor. */
internal data class VulkanModelParameterTensor(
    val name: String,
    val parameters: Float16ParameterArray
)

/**
 * Owns revision-tracked FP16 model allocations in the Vulkan Tensor Arena.
 *
 * Host arrays are the checkpoint and CPU-recovery mirror. While a resident training
 * transaction is active, the GPU allocation is authoritative; host readback is deferred
 * until inference, CPU fallback, checkpointing, or runtime recovery needs it.
 *
 * Parameter replacement is all-or-nothing at the registry boundary. Replaced handles
 * remain tracked until native release succeeds, so an arena cannot silently undercount
 * memory after a failed release.
 */
internal class VulkanTensorResidencyRegistry(
    private val allocate: (Long) -> VulkanTensorAllocation?,
    private val upload: (VulkanTensorAllocation, ByteArray) -> Boolean,
    private val release: (VulkanTensorAllocation) -> Boolean,
    private val download: ((VulkanTensorAllocation, ByteArray) -> Boolean)? = null,
    private val maxResidentBytes: () -> Long = { Long.MAX_VALUE }
) {
    private data class Entry(
        var allocation: VulkanTensorAllocation,
        var uploadedRevision: Long,
        var hostMirrorCurrent: Boolean
    )

    private val entries = IdentityHashMap<Float16ParameterArray, Entry>()
    private val retiredAllocations = ArrayList<VulkanTensorAllocation>()
    private var residentBytes: Long = 0L
    private var retiredResidentBytes: Long = 0L

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
            val currentBytes = runCatching {
                Math.addExact(residentBytes, retiredResidentBytes)
            }.getOrNull() ?: return null
            val nextActiveBytes = runCatching {
                Math.addExact(residentBytes, byteCount)
            }.getOrNull() ?: return null
            val nextTotalBytes = runCatching {
                Math.addExact(nextActiveBytes, retiredResidentBytes)
            }.getOrNull() ?: return null
            if (nextTotalBytes > maxResidentBytes().coerceAtLeast(0L)) return null

            val allocation = allocate(byteCount) ?: return null
            candidate = Entry(
                allocation = allocation,
                uploadedRevision = Long.MIN_VALUE,
                hostMirrorCurrent = false
            )
            entries[parameters] = candidate
            residentBytes = nextActiveBytes
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
                entry.hostMirrorCurrent = true
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
        if (!releaseRetiredAllocations()) return false
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
            retiredAllocations.clear()
            residentBytes = 0L
            retiredResidentBytes = 0L
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
        if (!releaseRetiredAllocations()) allReleased = false
        return allReleased
    }


    /** Marks the host mirrors stale before a dispatch that may mutate resident weights. */
    @Synchronized
    fun markDeviceMutationPending(parameters: List<Float16ParameterArray>) {
        for (parameter in parameters) {
            val entry = entries[parameter] ?: continue
            entry.hostMirrorCurrent = false
        }
    }

    /** Synchronizes one host tensor only when its resident GPU copy has changed. */
    @Synchronized
    fun synchronizeHostMirror(parameters: Float16ParameterArray): Boolean {
        val entry = entries[parameters] ?: return true
        if (entry.hostMirrorCurrent) return true
        if (entry.uploadedRevision == Long.MIN_VALUE) return false
        val readback = download ?: return false
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
        parameters.copyPrefixFromLittleEndianByteArray(snapshot, parameters.size)
        entry.uploadedRevision = parameters.mutationVersion
        entry.hostMirrorCurrent = true
        return true
    }

    /**
     * Stages all stale resident parameter tensors from GPU before committing any host mirror.
     * This is used before CPU fallback, runtime teardown, or checkpoint serialization.
     */
    @Synchronized
    fun synchronizeHostMirrors(): Boolean {
        if (entries.isEmpty()) return true
        val staleEntries = entries.entries.filter { (_, entry) -> !entry.hostMirrorCurrent }
        if (staleEntries.isEmpty()) return true
        val readback = download ?: return false
        val staged = ArrayList<Pair<Float16ParameterArray, ByteArray>>(staleEntries.size)
        for ((parameters, entry) in staleEntries) {
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
            entry.hostMirrorCurrent = true
        }
        return true
    }

    @Synchronized
    fun areHostMirrorsCurrent(): Boolean =
        entries.values.all { entry -> entry.hostMirrorCurrent }

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
        entry.hostMirrorCurrent = false
        return true
    }

    /** Validates a complete replacement before either the caller or this registry commits it. */
    @Synchronized
    fun canReplaceAll(
        tensors: List<Float16ParameterArray>,
        expectedAllocations: List<VulkanTensorAllocation>,
        replacementAllocations: List<VulkanTensorAllocation>
    ): Boolean {
        if (!validateReplacement(tensors, expectedAllocations, replacementAllocations)) return false
        val nextRetiredBytes = runCatching {
            expectedAllocations.fold(retiredResidentBytes) { total, allocation ->
                Math.addExact(total, allocation.byteCount)
            }
        }.getOrNull() ?: return false
        if (nextRetiredBytes < retiredResidentBytes) return false
        return runCatching {
            retiredAllocations.ensureCapacity(
                Math.addExact(retiredAllocations.size, expectedAllocations.size)
            )
            true
        }.getOrDefault(false)
    }

    /**
     * Swaps every resident parameter handle as one registry transaction. GPU candidate
     * buffers are already fully initialized; host mirrors remain stale until requested.
     * Old handles are retained for deferred release if Vulkan refuses an individual free.
     */
    @Synchronized
    fun replaceAll(
        tensors: List<Float16ParameterArray>,
        expectedAllocations: List<VulkanTensorAllocation>,
        replacementAllocations: List<VulkanTensorAllocation>
    ): Boolean {
        if (!validateReplacement(tensors, expectedAllocations, replacementAllocations)) return false

        var nextRetiredBytes = retiredResidentBytes
        for (allocation in expectedAllocations) {
            nextRetiredBytes = runCatching {
                Math.addExact(nextRetiredBytes, allocation.byteCount)
            }.getOrNull() ?: return false
        }

        for (index in tensors.indices) {
            val tensor = tensors[index]
            val entry = entries[tensor] ?: return false
            val previous = entry.allocation
            entry.allocation = replacementAllocations[index]
            entry.uploadedRevision = tensor.mutationVersion
            entry.hostMirrorCurrent = false
            retiredAllocations.add(previous)
        }
        retiredResidentBytes = nextRetiredBytes
        releaseRetiredAllocations()
        return true
    }

    private fun validateReplacement(
        tensors: List<Float16ParameterArray>,
        expectedAllocations: List<VulkanTensorAllocation>,
        replacementAllocations: List<VulkanTensorAllocation>
    ): Boolean {
        if (tensors.isEmpty() ||
            tensors.size != expectedAllocations.size ||
            tensors.size != replacementAllocations.size ||
            entries.size != tensors.size) {
            return false
        }

        val seenTensors = IdentityHashMap<Float16ParameterArray, Boolean>()
        val seenReplacementHandles = HashSet<Long>()
        for (index in tensors.indices) {
            val tensor = tensors[index]
            if (seenTensors.put(tensor, true) != null) return false
            val entry = entries[tensor] ?: return false
            val expected = expectedAllocations[index]
            val replacement = replacementAllocations[index]
            if (entry.allocation.nativeHandle != expected.nativeHandle ||
                entry.allocation.byteCount != expected.byteCount ||
                replacement.nativeHandle <= 0L ||
                replacement.byteCount != expected.byteCount ||
                !seenReplacementHandles.add(replacement.nativeHandle)) {
                return false
            }
            if (entries.values.any { current ->
                    current.allocation.nativeHandle == replacement.nativeHandle
                } ||
                retiredAllocations.any { retired ->
                    retired.nativeHandle == replacement.nativeHandle
                }) {
                return false
            }
        }
        return true
    }

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

    @Synchronized
    fun residentTensorCount(): Int = entries.size

    @Synchronized
    fun residentByteCount(): Long = Math.addExact(residentBytes, retiredResidentBytes)

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
