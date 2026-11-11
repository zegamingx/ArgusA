package com.argus.agi.model

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class ArgusBatchTrainingTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun accumulatesIndependentSamplesAndPerformsOneOptimizerUpdate() {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            checkpoint.delete()
            val engine = ArgusNeuralEngine.createBlocking(
                context,
                EngineConfig(
                    dModel = 16,
                    nHeads = 4,
                    dFf = 32,
                    maxSeqLen = 32,
                    loraRank = 2,
                    layerCount = 1,
                    tokenizerConfig = TokenizerConfig(
                        maxVocabularySize = 512,
                        maxTokenLength = 16
                    ),
                    trainingBatchSize = 2
                )
            )

            val result = engine.trainBatchDetailed(
                listOf(
                    "Argus learns local patterns from this first document.",
                    "This second document contributes to the same optimizer batch."
                )
            )

            assertEquals(2, result.samples.size)
            assertEquals(1, result.optimizerUpdateCount)
            assertEquals(1, result.samples.sumOf { it.optimizerUpdateCount })
            assertTrue(result.samples.all { it.loss.isFinite() })
            assertTrue(result.samples.all { it.predictionTargetCount > 0 })
            assertTrue(result.totalPredictionTargetCount > 0)
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    @Test
    fun legacyCompatibilityPathKeepsBatchSizeAtOne() {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            checkpoint.delete()
            val engine = ArgusNeuralEngine.createBlocking(
                context,
                EngineConfig(
                    dModel = 16,
                    nHeads = 4,
                    dFf = 32,
                    maxSeqLen = 32,
                    loraRank = 2,
                    layerCount = 1,
                    tokenizerConfig = TokenizerConfig(
                        maxVocabularySize = 512,
                        maxTokenLength = 16
                    ),
                    trainingBatchSize = 1,
                    gradientAccumulationEnabled = false
                )
            )

            val failure = runCatching {
                engine.trainBatchDetailed(
                    listOf("First legacy training sample.", "Second legacy training sample.")
                )
            }.exceptionOrNull()
            assertTrue(failure is IllegalArgumentException)

            // Legacy mode updates once per context window; keep this fixture to one window.
            val legacySample = "argus learns"
            val legacyTokenCount = engine.tokenizer.encode(legacySample).size
            assertTrue(legacyTokenCount in 2..engine.maxSeqLen)

            val result = engine.trainBatchDetailed(listOf(legacySample))
            assertEquals(1, result.samples.size)
            assertEquals(1, result.samples.single().contextWindowCount)
            assertEquals(1, result.optimizerUpdateCount)
            assertEquals(1, result.samples.single().optimizerUpdateCount)
            assertTrue(result.samples.single().loss.isFinite())
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    @Test
    fun rejectsBatchesLargerThanEightSamples() {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            checkpoint.delete()
            val engine = ArgusNeuralEngine.createBlocking(
                context,
                EngineConfig(
                    dModel = 16,
                    nHeads = 4,
                    dFf = 32,
                    maxSeqLen = 32,
                    loraRank = 2,
                    layerCount = 1,
                    tokenizerConfig = TokenizerConfig(
                        maxVocabularySize = 512,
                        maxTokenLength = 16
                    ),
                    trainingBatchSize = 1
                )
            )

            val failure = runCatching {
                engine.trainBatchDetailed(List(9) { "Training sample number $it." })
            }.exceptionOrNull()

            assertTrue(failure is IllegalArgumentException)
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    private fun restoreCheckpoint(checkpoint: File, previousCheckpoint: ByteArray?) {
        if (previousCheckpoint == null) {
            checkpoint.delete()
        } else {
            checkpoint.writeBytes(previousCheckpoint)
        }
        File(context.filesDir, "$CHECKPOINT_FILE_NAME.tmp").delete()
    }

    private companion object {
        const val CHECKPOINT_FILE_NAME = "argus_model_weights.bin"
    }
}
