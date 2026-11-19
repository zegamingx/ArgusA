package com.argus.agi.backend

import com.argus.agi.device.DeviceProfile
import kotlinx.coroutines.flow.StateFlow

/**
 * Contract for an optional device accelerator and its recoverable resource lifecycle.
 *
 * The host-side neural engine remains authoritative for model and optimizer state.
 * Accelerator implementations must synchronize that state before releasing or
 * rebuilding device resources. Kotlin CPU execution remains the existing engine
 * fallback and is intentionally not represented by a placeholder implementation.
 */
internal interface ComputeBackend {

    val kind: ComputeBackendKind

    val isReady: Boolean

    /** True only when FP16 is enabled on the device and its matching kernel passed self-test. */
    val isFp16ComputeReady: Boolean

    val deviceProfile: StateFlow<DeviceProfile?>

    fun initializeAfterModelReady(): Boolean

    /** Reset failed native state before an explicit initialization retry. */
    fun resetForRetry()

    fun synchronizeAuthoritativeState(): Boolean

    fun releaseDeviceResourcesForMemoryPressure(): Boolean

    fun resumeDeviceResourcesAfterForeground()
}

internal enum class ComputeBackendKind {
    VULKAN
}
