package com.argus.agi.model

import org.junit.Assert.assertFalse
import org.junit.Test

class VulkanTrainingBackendBoundsTest {

    @Test
    fun fp16BatchedLinearRejectsWeightMatrixBeyondArrayLimit() {
        val accepted = VulkanTrainingBackend.tryApplyFp16BatchedLinear(
            inputBatches = arrayOf(arrayOf(floatArrayOf(0f))),
            weightBatches = arrayOf(shortArrayOf(0)),
            outputBatches = arrayOf(arrayOf(floatArrayOf(0f))),
            inputColumns = Int.MAX_VALUE,
            outputColumns = Int.MAX_VALUE
        )

        assertFalse(accepted)
    }
}
