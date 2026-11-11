package com.argus.agi.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ArgusArchitectureConfigTest {

    @Test
    fun oneLayerEstimateMatchesTheLegacyEngineTensorLayout() {
        val config = ArgusArchitectureConfig(
            layerCount = 1,
            hiddenSize = 16,
            attentionHeadCount = 4,
            feedForwardSize = 32,
            vocabularySize = 256,
            contextLength = 16,
            adapterRank = 2
        )

        val estimate = ArgusArchitectureParameterEstimator.estimate(config)

        assertEquals(4_096L, estimate.tokenEmbeddingParameters)
        assertEquals(4_096L, estimate.outputProjectionParameters)
        assertEquals(1_024L, estimate.attentionParameters)
        assertEquals(1_536L, estimate.feedForwardParameters)
        assertEquals(128L, estimate.adapterParameters)
        assertEquals(256L, estimate.fixedPositionBufferElements)
        assertEquals(10_880L, estimate.totalTrainableParameters)
        assertEquals(11_136L, estimate.totalResidentModelElements)
    }

    @Test
    fun blockParametersScaleWithLayerCountButEmbeddingsDoNot() {
        val singleLayer = ArgusArchitectureParameterEstimator.estimate(
            architecture(layerCount = 1)
        )
        val eightLayers = ArgusArchitectureParameterEstimator.estimate(
            architecture(layerCount = 8)
        )

        assertEquals(singleLayer.tokenEmbeddingParameters, eightLayers.tokenEmbeddingParameters)
        assertEquals(singleLayer.outputProjectionParameters, eightLayers.outputProjectionParameters)
        assertEquals(singleLayer.attentionParameters * 8L, eightLayers.attentionParameters)
        assertEquals(singleLayer.feedForwardParameters * 8L, eightLayers.feedForwardParameters)
        assertEquals(singleLayer.adapterParameters * 8L, eightLayers.adapterParameters)
    }

    @Test
    fun tiedTokenEmbeddingsDoNotCountASecondOutputMatrix() {
        val estimate = ArgusArchitectureParameterEstimator.estimate(
            architecture(tokenEmbeddingsAreTied = true)
        )

        assertEquals(64_000L, estimate.tokenEmbeddingParameters)
        assertEquals(0L, estimate.outputProjectionParameters)
    }

    @Test
    fun rotaryPositionEncodingDoesNotAllocateAFixedPositionTable() {
        val estimate = ArgusArchitectureParameterEstimator.estimate(
            architecture(
                positionEncodingKind = ArgusPositionEncodingKind.ROTARY
            )
        )

        assertEquals(0L, estimate.fixedPositionBufferElements)
    }

    @Test
    fun rejectsHeadDimensionsThatDoNotDivideTheHiddenSize() {
        val failure = runCatching {
            architecture(hiddenSize = 63, attentionHeadCount = 4)
        }.exceptionOrNull()

        assertTrue(failure is IllegalArgumentException)
    }

    @Test
    fun rejectsParameterCountOverflowInsteadOfReturningAnInvalidEstimate() {
        val config = ArgusArchitectureConfig(
            layerCount = Int.MAX_VALUE,
            hiddenSize = Int.MAX_VALUE,
            attentionHeadCount = 1,
            feedForwardSize = Int.MAX_VALUE,
            vocabularySize = Int.MAX_VALUE,
            contextLength = Int.MAX_VALUE,
            adapterRank = 0
        )

        val failure = runCatching {
            ArgusArchitectureParameterEstimator.estimate(config)
        }.exceptionOrNull()

        assertTrue(failure is ArithmeticException)
    }

    private fun architecture(
        layerCount: Int = 1,
        hiddenSize: Int = 64,
        attentionHeadCount: Int = 4,
        feedForwardSize: Int = 256,
        vocabularySize: Int = 1_000,
        contextLength: Int = 128,
        adapterRank: Int = 0,
        positionEncodingKind: ArgusPositionEncodingKind =
            ArgusPositionEncodingKind.FIXED_SINUSOIDAL_TABLE,
        tokenEmbeddingsAreTied: Boolean = false
    ): ArgusArchitectureConfig = ArgusArchitectureConfig(
        layerCount = layerCount,
        hiddenSize = hiddenSize,
        attentionHeadCount = attentionHeadCount,
        feedForwardSize = feedForwardSize,
        vocabularySize = vocabularySize,
        contextLength = contextLength,
        adapterRank = adapterRank,
        positionEncodingKind = positionEncodingKind,
        tokenEmbeddingsAreTied = tokenEmbeddingsAreTied
    )
}
