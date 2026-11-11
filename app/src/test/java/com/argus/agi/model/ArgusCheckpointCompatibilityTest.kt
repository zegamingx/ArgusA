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
                maxVocabularySize = 512,
                maxTokenLength = 32
            )
            val engineConfig = smallEngineConfig(tokenizerConfig)
            val original = ArgusNeuralEngine.createBlocking(context, engineConfig)
            val learnedToken = "argus_neural_checkpoint"
            val learnedTokenId = original.tokenizer.registerToken(learnedToken)

            assertTrue(learnedTokenId >= Tokenizer.THINK_END_ID)
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
            assertEquals(original.tokenizer.exportVocab(), restored.tokenizer.exportVocab())
            assertEquals(learnedTokenId, restored.tokenizer.encode(learnedToken).single())
            assertEquals(original.countTrainableParameters(), restored.countTrainableParameters())
            assertEquals(original.countTotalParameters(), restored.countTotalParameters())
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
