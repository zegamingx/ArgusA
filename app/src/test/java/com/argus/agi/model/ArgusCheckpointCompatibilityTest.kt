package com.argus.agi.model

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class ArgusCheckpointCompatibilityTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun arg4RoundTripRestoresArchitectureTokenizerAndNamedTensorState() {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            checkpoint.delete()
            val tokenizerConfig = TokenizerConfig(
                algorithm = ArgusTokenizerAlgorithm.GREEDY_LONGEST_MATCH_V1,
                maxVocabularySize = 512,
                maxTokenLength = 32
            )
            val engineConfig = smallEngineConfig(tokenizerConfig)
            val original = ArgusNeuralEngine.createBlocking(context, engineConfig)
            val trainingText = "argus neural checkpoint tokenizer"
            val learnedMergeCount = original.tokenizer.trainBpe(
                corpus = List(64) { trainingText },
                targetVocabSize = original.tokenizer.vocabSize + 10,
                maxNewMerges = 10
            )
            assertTrue(learnedMergeCount > 0)
            val encodedTrainingText = original.tokenizer.encode(trainingText)
            assertTrue(original.saveCheckpoint())

            val checkpointBytes = checkpoint.readBytes()
            assertEquals('A'.code.toByte(), checkpointBytes[0])
            assertEquals('R'.code.toByte(), checkpointBytes[1])
            assertEquals('G'.code.toByte(), checkpointBytes[2])
            assertEquals('4'.code.toByte(), checkpointBytes[3])

            val restoredConfig = ArgusNeuralEngine.resolveConfig(
                context = context,
                totalRamGb = 2,
                maxHeapMb = 128L,
                availableRamGb = 1.0f
            )
            assertEquals(engineConfig.dModel, restoredConfig.dModel)
            assertEquals(engineConfig.dFf, restoredConfig.dFf)
            assertEquals(512, restoredConfig.tokenizerConfig.maxVocabularySize)
            assertEquals(32, restoredConfig.tokenizerConfig.maxTokenLength)
            assertEquals(
                tokenizerConfig.algorithm,
                restoredConfig.tokenizerConfig.algorithm
            )

            val restored = ArgusNeuralEngine.createBlocking(context, restoredConfig)
            assertTrue(restored.loadCheckpointIfExists())
            assertEquals(original.tokenizer.exportVocab(), restored.tokenizer.exportVocab())
            assertEquals(original.tokenizer.exportMerges(), restored.tokenizer.exportMerges())
            assertEquals(encodedTrainingText, restored.tokenizer.encode(trainingText))
            assertEquals(original.countTrainableParameters(), restored.countTrainableParameters())
            assertEquals(original.countTotalParameters(), restored.countTotalParameters())
            assertTrue(restored.saveCheckpoint())
            assertTrue(checkpoint.readBytes().contentEquals(checkpointBytes))
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    @Test
    fun arg4LoaderRejectsPayloadChecksumMismatch() {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            checkpoint.delete()
            val config = smallEngineConfig(TokenizerConfig())
            val original = ArgusNeuralEngine.createBlocking(context, config)
            assertTrue(original.saveCheckpoint())

            val corruptedBytes = checkpoint.readBytes()
            val payloadStart = 96
            assertTrue(corruptedBytes.size > payloadStart)
            corruptedBytes[corruptedBytes.lastIndex] =
                (corruptedBytes[corruptedBytes.lastIndex].toInt() xor 0x01).toByte()
            checkpoint.writeBytes(corruptedBytes)

            val restoredConfig = ArgusNeuralEngine.resolveConfig(
                context = context,
                totalRamGb = 2,
                maxHeapMb = 128L,
                availableRamGb = 1.0f
            )
            val restored = ArgusNeuralEngine.createBlocking(context, restoredConfig)
            assertTrue(!restored.loadCheckpointIfExists())
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    @Test
    fun arg3HeaderStillResolvesItsOriginalEngineDimensions() {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            val header = ByteBuffer.allocate(LEGACY_HEADER_SIZE)
                .order(ByteOrder.LITTLE_ENDIAN)
            header.put(byteArrayOf(0x41, 0x52, 0x47, 0x33))
            header.putInt(3)
            header.putInt(16)
            header.putInt(4)
            header.putInt(32)
            header.putInt(16)
            header.putInt(2)
            header.putInt(144)
            header.putInt(176)
            header.putInt(7)
            header.putInt(0)
            while (header.hasRemaining()) {
                header.put(0)
            }
            checkpoint.writeBytes(header.array())

            val restoredConfig = ArgusNeuralEngine.resolveConfig(
                context = context,
                totalRamGb = 12,
                maxHeapMb = 512L,
                availableRamGb = 8.0f
            )

            assertEquals(16, restoredConfig.dModel)
            assertEquals(4, restoredConfig.nHeads)
            assertEquals(32, restoredConfig.dFf)
            assertEquals(16, restoredConfig.maxSeqLen)
            assertEquals(2, restoredConfig.loraRank)
            assertEquals(
                TokenizerConfig.DEFAULT_MAX_VOCAB_SIZE,
                restoredConfig.tokenizerConfig.maxVocabularySize
            )
            assertEquals(
                ArgusTokenizerAlgorithm.GREEDY_LONGEST_MATCH_V1,
                restoredConfig.tokenizerConfig.algorithm
            )
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    @Test
    fun arg4MultiLayerRoundTripRestoresAllTransformerBlocks() {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            checkpoint.delete()
            val engineConfig = EngineConfig(
                layerCount = 2,
                dModel = 16,
                nHeads = 4,
                dFf = 32,
                maxSeqLen = 16,
                loraRank = 2,
                tokenizerConfig = TokenizerConfig(maxVocabularySize = 512, maxTokenLength = 32)
            )
            val original = ArgusNeuralEngine.createBlocking(context, engineConfig)
            assertEquals(2, original.layerCount)
            assertEquals(2, original.architectureConfig.layerCount)

            original.blocks[1].wDown[0] = 3.14159f
            assertTrue(original.saveCheckpoint())

            val restoredConfig = ArgusNeuralEngine.resolveConfig(
                context = context,
                totalRamGb = 2,
                maxHeapMb = 128L,
                availableRamGb = 1.0f
            )
            assertEquals(2, restoredConfig.layerCount)

            val restored = ArgusNeuralEngine.createBlocking(context, restoredConfig)
            assertTrue(restored.loadCheckpointIfExists())
            assertEquals(2, restored.layerCount)
            assertEquals(3.140625f, restored.blocks[1].wDown[0], 1e-6f)
            assertEquals(original.countTotalParameters(), restored.countTotalParameters())
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    @Test
    fun learnedBpeTokenizerTrainsMergesAndPersistsInArg4Checkpoint() {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            checkpoint.delete()
            val tokenizerConfig = TokenizerConfig(
                algorithm = ArgusTokenizerAlgorithm.LEARNED_BPE_V1,
                maxVocabularySize = 512,
                maxTokenLength = 32
            )
            val engineConfig = EngineConfig(
                layerCount = 2,
                dModel = 16,
                nHeads = 4,
                dFf = 32,
                maxSeqLen = 16,
                loraRank = 2,
                tokenizerConfig = tokenizerConfig
            )
            val original = ArgusNeuralEngine.createBlocking(context, engineConfig)
            val corpus = listOf(
                "argus neural engine learned bpe tokenizer merge test",
                "learned bpe tokenizer creates merge pairs from frequency",
                "argus neural engine learned bpe training"
            )
            val trainedMerges = original.tokenizer.trainBpe(corpus, targetVocabSize = original.tokenizer.vocabSize + 10)
            assertTrue(trainedMerges > 0)
            val originalMerges = original.tokenizer.exportMerges()
            assertTrue(originalMerges.isNotEmpty())

            assertTrue(original.saveCheckpoint())

            val restoredConfig = ArgusNeuralEngine.resolveConfig(
                context = context,
                totalRamGb = 2,
                maxHeapMb = 128L,
                availableRamGb = 1.0f
            )
            assertEquals(ArgusTokenizerAlgorithm.LEARNED_BPE_V1, restoredConfig.tokenizerConfig.algorithm)

            val restored = ArgusNeuralEngine.createBlocking(context, restoredConfig)
            assertTrue(restored.loadCheckpointIfExists())
            val restoredMerges = restored.tokenizer.exportMerges()
            assertEquals(originalMerges, restoredMerges)

            val sampleText = "argus neural engine"
            assertEquals(original.tokenizer.encode(sampleText), restored.tokenizer.encode(sampleText))
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    @Test
    fun multiLayerEngineGeneratesAndTrainsWithoutError() {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            checkpoint.delete()
            val engineConfig = EngineConfig(
                layerCount = 3,
                dModel = 16,
                nHeads = 4,
                dFf = 32,
                maxSeqLen = 16,
                loraRank = 2,
                isFullTrainingEnabled = true,
                tokenizerConfig = TokenizerConfig(maxVocabularySize = 512, maxTokenLength = 32)
            )
            val engine = ArgusNeuralEngine.createBlocking(context, engineConfig)
            assertEquals(3, engine.layerCount)

            val generated = kotlinx.coroutines.runBlocking {
                engine.generate("test prompt", maxNewTokens = 4, temperature = 0.5f)
            }
            assertTrue(generated.isNotEmpty())

            val initialLoss = engine.trainStep(
                text = "argus multi layer transformer training validation"
            )
            assertTrue(initialLoss > 0f)
            assertTrue(initialLoss.isFinite())
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    private fun smallEngineConfig(tokenizerConfig: TokenizerConfig): EngineConfig =
        EngineConfig(
            dModel = 16,
            nHeads = 4,
            dFf = 32,
            maxSeqLen = 16,
            loraRank = 2,
            tokenizerConfig = tokenizerConfig
        )

    private fun restoreCheckpoint(file: File, previous: ByteArray?) {
        if (previous == null) {
            file.delete()
        } else {
            file.writeBytes(previous)
        }
    }

    private companion object {
        const val CHECKPOINT_FILE_NAME = "argus_model_weights.bin"
        const val LEGACY_HEADER_SIZE = 48
    }
}

class ArgusTokenizerConfigTest {

    @Test
    fun configurableVocabularyCapacityStopsAtItsDeclaredBoundary() {
        val tokenizer = Tokenizer(
            TokenizerConfig(
                maxVocabularySize = TokenizerConfig.MINIMUM_VOCAB_SIZE,
                maxTokenLength = TokenizerConfig.MINIMUM_TOKEN_LENGTH
            )
        )
        var tokenIndex = 0
        while (tokenizer.vocabSize < TokenizerConfig.MINIMUM_VOCAB_SIZE) {
            tokenizer.registerToken("v${tokenIndex++}")
        }

        assertEquals(TokenizerConfig.MINIMUM_VOCAB_SIZE, tokenizer.vocabSize)
        assertEquals(Tokenizer.UNK_ID, tokenizer.registerToken("overflow"))
    }

    @Test
    fun configurableLongTokensSurviveVocabularyExportAndImport() {
        val config = TokenizerConfig(
            algorithm = ArgusTokenizerAlgorithm.GREEDY_LONGEST_MATCH_V1,
            maxVocabularySize = 512,
            maxTokenLength = 32
        )
        val tokenizer = Tokenizer(config)
        val token = "argus_neural_checkpoint"
        val tokenId = tokenizer.registerToken(token)

        assertEquals(listOf(tokenId), tokenizer.encode(token))
        assertEquals(token, tokenizer.decode(listOf(tokenId)))

        val restored = Tokenizer(config)
        restored.importVocab(tokenizer.exportVocab())

        assertEquals(tokenizer.exportVocab(), restored.exportVocab())
        assertEquals(listOf(tokenId), restored.encode(token))
        assertEquals(token, restored.decode(listOf(tokenId)))
    }

    @Test
    fun rejectsVocabularyEntriesOutsideConfiguredCapacity() {
        val config = TokenizerConfig(
            maxVocabularySize = TokenizerConfig.MINIMUM_VOCAB_SIZE,
            maxTokenLength = TokenizerConfig.MINIMUM_TOKEN_LENGTH
        )
        val tokenizer = Tokenizer(config)
        val failure = runCatching {
            tokenizer.registerToken(
                token = "overflow",
                id = config.maxVocabularySize
            )
        }.exceptionOrNull()

        assertTrue(failure is IllegalArgumentException)
    }

    @Test
    fun rejectsSpecialTokenLengthConfigurationThatCannotRepresentCanonicalTokens() {
        val failure = runCatching {
            TokenizerConfig(maxTokenLength = TokenizerConfig.MINIMUM_TOKEN_LENGTH - 1)
        }.exceptionOrNull()

        assertTrue(failure is IllegalArgumentException)
    }
}
