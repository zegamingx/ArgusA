package com.argus.agi.model

import org.junit.Assert.assertEquals
import org.junit.Test

class ArgusNeuralEngineGenerationBudgetTest {

    @Test
    fun reservesRequestedOutputPositionsWithinContextWindow() {
        val window = ArgusNeuralEngine.resolveGenerationTokenWindow(
            maxSequenceLength = 128,
            requestedNewTokens = 42
        )

        assertEquals(86, window.promptTokenLimit)
        assertEquals(42, window.outputTokenLimit)
    }

    @Test
    fun clampsOutputBudgetWhenItExceedsContextCapacity() {
        val window = ArgusNeuralEngine.resolveGenerationTokenWindow(
            maxSequenceLength = 64,
            requestedNewTokens = 96
        )

        assertEquals(1, window.promptTokenLimit)
        assertEquals(63, window.outputTokenLimit)
    }

    @Test
    fun zeroOutputBudgetKeepsAllButOnePromptPosition() {
        val window = ArgusNeuralEngine.resolveGenerationTokenWindow(
            maxSequenceLength = 64,
            requestedNewTokens = 0
        )

        assertEquals(63, window.promptTokenLimit)
        assertEquals(0, window.outputTokenLimit)
    }
}
