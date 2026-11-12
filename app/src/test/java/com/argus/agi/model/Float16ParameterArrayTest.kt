package com.argus.agi.model

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class Float16ParameterArrayTest {

    @Test
    fun binary16ValuesRoundTripAndShortExportIsDefensive() {
        val source = floatArrayOf(0f, -0f, 0.5f, -2f, 1.5f, 65504f)
        val parameters = Float16ParameterArray.fromFloatArray(source)
        val exported = parameters.toShortArray()

        assertEquals(source.size, exported.size)
        assertArrayEquals(source, parameters.toFloatArray(), 0f)

        exported[2] = Float16Codec.fromFloat(9f)
        assertEquals(0.5f, parameters[2], 0f)
    }

    @Test
    fun linearBackendAcceptsWeightCapacityAboveTheActiveMatrixSize() {
        val dispatched = VulkanTrainingBackend.tryApplyFp16Linear(
            inputRows = arrayOf(floatArrayOf(1f)),
            weights = Float16ParameterArray.fromFloatArray(floatArrayOf(2f, 99f)),
            outputRows = arrayOf(FloatArray(1)),
            inputColumns = 1,
            outputColumns = 1
        )

        // A one-element operation stays on the CPU/FP32 fallback, but validating
        // the active 1x1 matrix must not reject the larger backing capacity.
        assertFalse(dispatched)
    }

    @Test
    fun binary16CodecUsesFiniteRoundedValuesForModelParameters() {
        val values = floatArrayOf(0.1f, 1.0004f, -3.14159f)
        val parameters = Float16ParameterArray.fromFloatArray(values)

        for (value in parameters.toFloatArray()) {
            assertTrue(value.isFinite())
        }
        assertEquals(0.1f, parameters[0], 1e-3f)
        assertEquals(1.0f, parameters[1], 1e-3f)
        assertEquals(-3.140625f, parameters[2], 1e-3f)
    }
}
