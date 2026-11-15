package com.argus.agi.model

import android.app.Application
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = Application::class)
class VulkanTrainingBackendSafetyTest {

    @Test
    fun readingBackendStatusDoesNotLoadNativeRuntime() {
        assertFalse(VulkanTrainingBackend.hasAttemptedInitialization)
        assertFalse(VulkanTrainingBackend.isRuntimeAvailable)
        assertFalse(VulkanTrainingBackend.isFp16ComputeAvailable)
        assertEquals("Kotlin CPU", VulkanTrainingBackend.currentBackend)
        assertFalse(VulkanTrainingBackend.hasAttemptedInitialization)
    }

    @Test
    fun denseOperationUsesCpuFallbackUntilPostModelRuntimePreparation() {
        val input = Array(64) { FloatArray(64) }
        val weights = FloatArray(64 * 64)
        val output = Array(64) { FloatArray(64) }

        val updated = VulkanTrainingBackend.tryApplyLinear(
            inputRows = input,
            weights = weights,
            outputRows = output,
            inputColumns = 64,
            outputColumns = 64
        )

        assertFalse("The GPU path must remain inactive until post-model runtime preparation.", updated)
        assertFalse("CPU fallback must not attempt to load JNI.", VulkanTrainingBackend.hasAttemptedInitialization)
        assertFalse(VulkanTrainingBackend.isRuntimeAvailable)
    }
}
