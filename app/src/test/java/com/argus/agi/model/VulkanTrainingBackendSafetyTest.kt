package com.argus.agi.model

import android.app.Application
import org.junit.Assert.assertArrayEquals
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
    fun readingBackendStatusDoesNotInitializeOrMutateNativeRuntime() {
        val attemptedBefore = VulkanTrainingBackend.hasAttemptedInitialization
        val runtimeBefore = VulkanTrainingBackend.isRuntimeAvailable
        val fp16Before = VulkanTrainingBackend.isFp16ComputeAvailable
        val backendBefore = VulkanTrainingBackend.currentBackend

        repeat(2) {
            assertEquals(backendBefore, VulkanTrainingBackend.currentBackend)
            assertEquals(runtimeBefore, VulkanTrainingBackend.isRuntimeAvailable)
            assertEquals(fp16Before, VulkanTrainingBackend.isFp16ComputeAvailable)
            assertEquals(attemptedBefore, VulkanTrainingBackend.hasAttemptedInitialization)
        }
    }

    @Test
    fun denseOperationBelowGpuThresholdReturnsToCpuWithoutInitializingNativeRuntime() {
        val attemptedBefore = VulkanTrainingBackend.hasAttemptedInitialization
        val runtimeBefore = VulkanTrainingBackend.isRuntimeAvailable
        val fp16Before = VulkanTrainingBackend.isFp16ComputeAvailable
        val backendBefore = VulkanTrainingBackend.currentBackend

        // Keep this operation below the adaptive GPU-dispatch threshold, even when a
        // previous Robolectric test has already prepared the process-wide Vulkan singleton.
        val input = Array(8) { FloatArray(8) { 1.0f } }
        val weights = FloatArray(8 * 8) { 0.5f }
        val output = Array(8) { FloatArray(8) { 7.0f } }
        val outputBefore = Array(output.size) { index -> output[index].copyOf() }

        val updated = VulkanTrainingBackend.tryApplyLinear(
            inputRows = input,
            weights = weights,
            outputRows = output,
            inputColumns = 8,
            outputColumns = 8
        )

        assertFalse("A small operation must stay on the caller's CPU path.", updated)
        assertEquals(attemptedBefore, VulkanTrainingBackend.hasAttemptedInitialization)
        assertEquals(runtimeBefore, VulkanTrainingBackend.isRuntimeAvailable)
        assertEquals(fp16Before, VulkanTrainingBackend.isFp16ComputeAvailable)
        assertEquals(backendBefore, VulkanTrainingBackend.currentBackend)
        for (index in output.indices) {
            assertArrayEquals(outputBefore[index], output[index], 0.0f)
        }
    }
}
