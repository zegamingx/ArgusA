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
@Config(sdk = [34])
class GrpoOptimizerTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun sampledResponseGroupExecutesPolicyUpdateAndVerifiesCheckpoint() {
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
                        maxTokenLength = 32
                    ),
                    trainingBatchSize = 2,
                    gradientAccumulationEnabled = true
                )
            )
            val prompt = Tokenizer.USER_TOKEN + "2+2" + Tokenizer.ASSISTANT_TOKEN
            val promptTokens = engine.tokenizer.encodeToIds(prompt, autoRegister = false)
            val firstCompletion = engine.tokenizer.encodeToIds("4", autoRegister = false)
            val secondCompletion = engine.tokenizer.encodeToIds("5", autoRegister = false)
            assertTrue(promptTokens.isNotEmpty())
            assertTrue(firstCompletion.isNotEmpty())
            assertTrue(secondCompletion.isNotEmpty())

            val result = engine.trainGrpoGroup(
                samples = listOf(
                    GrpoPolicyTrainingSample(
                        tokenIds = promptTokens + firstCompletion,
                        firstTargetIndex = promptTokens.lastIndex,
                        behaviorLogProbabilities = FloatArray(firstCompletion.size) { -5f },
                        advantage = 1f
                    ),
                    GrpoPolicyTrainingSample(
                        tokenIds = promptTokens + secondCompletion,
                        firstTargetIndex = promptTokens.lastIndex,
                        behaviorLogProbabilities = FloatArray(secondCompletion.size) { -5f },
                        advantage = -1f
                    )
                ),
                clipEpsilon = 0.2f,
                klCoefficient = 0.02f,
                temperature = 0.8f,
                ppoEpochs = 1
            )

            assertEquals(2, result.responseCount)
            assertTrue(result.targetTokenCount > 0)
            assertEquals(1, result.optimizerUpdates)
            assertTrue(result.meanPolicyLoss.isFinite())
            assertTrue(engine.isCheckpointStateVerified)
            assertTrue(checkpoint.isFile && checkpoint.length() > 0L)
        } finally {
            if (previousCheckpoint == null) {
                checkpoint.delete()
            } else {
                checkpoint.writeBytes(previousCheckpoint)
            }
            File(context.filesDir, "$CHECKPOINT_FILE_NAME.tmp").delete()
        }
    }

    private companion object {
        const val CHECKPOINT_FILE_NAME = "argus_model_weights.bin"
    }
}
