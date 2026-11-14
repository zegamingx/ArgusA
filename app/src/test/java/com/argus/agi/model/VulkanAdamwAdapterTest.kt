package com.argus.agi.model

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Test

class VulkanAdamwAdapterTest {

    @Test
    fun disabledVulkanLeavesFp16WeightsAndFp32MomentsUntouchedForCpuFallback() {
        VulkanTrainingBackend.disableForCurrentSession()

        val parameters = arrayOf(
            Float16ParameterArray.fromFloatArray(floatArrayOf(0.25f, -0.5f, 0.75f)),
            Float16ParameterArray.fromFloatArray(floatArrayOf(1.0f, -1.0f, 0.125f))
        )
        val gradients = arrayOf(
            floatArrayOf(0.1f, -0.2f),
            floatArrayOf(-0.3f, 0.4f)
        )
        val firstMoments = arrayOf(
            floatArrayOf(0.01f, 0.02f, 0.03f),
            floatArrayOf(0.04f, 0.05f, 0.06f)
        )
        val secondMoments = arrayOf(
            floatArrayOf(0.001f, 0.002f, 0.003f),
            floatArrayOf(0.004f, 0.005f, 0.006f)
        )
        val parameterBefore = Array(parameters.size) { parameters[it].toFloatArray() }
        val firstMomentBefore = Array(firstMoments.size) { firstMoments[it].copyOf() }
        val secondMomentBefore = Array(secondMoments.size) { secondMoments[it].copyOf() }

        val updated = VulkanTrainingBackend.tryApplyFp16AdamWUpdate(
            parameters = parameters,
            gradients = gradients,
            firstMoments = firstMoments,
            secondMoments = secondMoments,
            weightDecayByGroup = floatArrayOf(0f, 0.01f),
            learningRate = 1e-3f,
            beta1 = 0.9f,
            beta2 = 0.999f,
            biasCorrection1 = 0.1f,
            biasCorrection2 = 0.001f,
            epsilon = 1e-8f,
            gradientClipFactor = 0.75f
        )

        assertFalse(updated)
        for (index in parameters.indices) {
            assertArrayEquals(parameterBefore[index], parameters[index].toFloatArray(), 0f)
            assertArrayEquals(firstMomentBefore[index], firstMoments[index], 0f)
            assertArrayEquals(secondMomentBefore[index], secondMoments[index], 0f)
        }
    }

    @Test
    fun rejectsMomentCapacityThatDoesNotMatchFp16ParameterCapacity() {
        VulkanTrainingBackend.disableForCurrentSession()

        val parameters = arrayOf(Float16ParameterArray.fromFloatArray(floatArrayOf(0.5f, -0.25f)))
        val gradients = arrayOf(floatArrayOf(0.1f, -0.1f))
        val firstMoments = arrayOf(floatArrayOf(0f))
        val secondMoments = arrayOf(floatArrayOf(0f, 0f))

        assertThrows(IllegalArgumentException::class.java) {
            VulkanTrainingBackend.tryApplyFp16AdamWUpdate(
                parameters = parameters,
                gradients = gradients,
                firstMoments = firstMoments,
                secondMoments = secondMoments,
                weightDecayByGroup = floatArrayOf(0f),
                learningRate = 1e-3f,
                beta1 = 0.9f,
                beta2 = 0.999f,
                biasCorrection1 = 0.1f,
                biasCorrection2 = 0.001f,
                epsilon = 1e-8f,
                gradientClipFactor = 1f
            )
        }
    }
}
