package com.argus.agi.runtime

internal enum class AdaptiveComputeOperation {
    LINEAR,
    ADAMW
}

/**
 * Passive execution telemetry for successful Vulkan work and explicit CPU fallback.
 *
 * This monitor must never select a backend. Vulkan eligibility is determined by
 * runtime readiness, enabled device capabilities, kernel availability, and safe
 * resource/shape checks at the operation call site. CPU timings are recorded only
 * when a CPU fallback actually executes.
 */
internal object AdaptiveComputePerformanceMonitor {

    private const val EWMA_OLD_WEIGHT = 0.75
    private const val EWMA_NEW_WEIGHT = 0.25

    internal data class Snapshot(
        val cpuSamples: Int,
        val gpuSamples: Int,
        val cpuNanosecondsPerOperation: Double?,
        val gpuNanosecondsPerOperation: Double?
    )

    private data class Profile(
        var cpuSamples: Int = 0,
        var gpuSamples: Int = 0,
        var cpuNanosecondsPerOperation: Double = 0.0,
        var gpuNanosecondsPerOperation: Double = 0.0
    )

    private data class ProfileKey(
        val operation: AdaptiveComputeOperation,
        val operationSizeBucket: Int
    )

    private val profiles = HashMap<ProfileKey, Profile>()

    @Synchronized
    fun recordCpu(
        operation: AdaptiveComputeOperation,
        operationCount: Long,
        elapsedNanoseconds: Long
    ) {
        record(operation, operationCount, elapsedNanoseconds, gpu = false)
    }

    @Synchronized
    fun recordGpu(
        operation: AdaptiveComputeOperation,
        operationCount: Long,
        elapsedNanoseconds: Long
    ) {
        record(operation, operationCount, elapsedNanoseconds, gpu = true)
    }

    @Synchronized
    fun snapshot(
        operation: AdaptiveComputeOperation,
        operationCount: Long = 1L
    ): Snapshot {
        val profile = profileFor(operation, operationCount)
        return Snapshot(
            cpuSamples = profile.cpuSamples,
            gpuSamples = profile.gpuSamples,
            cpuNanosecondsPerOperation = profile.cpuNanosecondsPerOperation
                .takeIf { profile.cpuSamples > 0 },
            gpuNanosecondsPerOperation = profile.gpuNanosecondsPerOperation
                .takeIf { profile.gpuSamples > 0 }
        )
    }

    @Synchronized
    internal fun resetForTests() {
        profiles.clear()
    }

    private fun profileFor(
        operation: AdaptiveComputeOperation,
        operationCount: Long
    ): Profile {
        val bucket = if (operationCount <= 0L) 0 else
            (java.lang.Long.SIZE - java.lang.Long.numberOfLeadingZeros(operationCount))
        return profiles.getOrPut(ProfileKey(operation, bucket)) { Profile() }
    }

    private fun record(
        operation: AdaptiveComputeOperation,
        operationCount: Long,
        elapsedNanoseconds: Long,
        gpu: Boolean
    ) {
        if (operationCount <= 0L || elapsedNanoseconds <= 0L) return
        val sample = elapsedNanoseconds.toDouble() / operationCount.toDouble()
        if (!sample.isFinite() || sample <= 0.0) return

        val profile = profileFor(operation, operationCount)
        if (gpu) {
            profile.gpuNanosecondsPerOperation =
                if (profile.gpuSamples == 0) sample
                else profile.gpuNanosecondsPerOperation * EWMA_OLD_WEIGHT +
                    sample * EWMA_NEW_WEIGHT
            profile.gpuSamples++
        } else {
            profile.cpuNanosecondsPerOperation =
                if (profile.cpuSamples == 0) sample
                else profile.cpuNanosecondsPerOperation * EWMA_OLD_WEIGHT +
                    sample * EWMA_NEW_WEIGHT
            profile.cpuSamples++
        }
    }
}
