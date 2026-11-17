package com.argus.agi.model

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.zip.CRC32
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ArgusCheckpointCompatibilityTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun arg5RoundTripRestoresArchitectureTokenizerAndFingerprintBoundState() {
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
            val trainingText = "argus neural checkpoint tokenizer"
            val resultingVocabulary = original.tokenizer.trainBpe(
                corpus = List(64) { trainingText },
                targetVocabSize = original.tokenizer.vocabSize + 10,
                maxNewMerges = 10
            )
            assertTrue(resultingVocabulary > TokenizerConfig.MINIMUM_VOCAB_SIZE)

            val encodedBefore = original.tokenizer.encodeToIds(trainingText)
            val tokenizerFingerprint = original.tokenizer.snapshotFingerprint()
            assertTrue(original.saveCheckpoint())
            assertTrue(original.isCheckpointStateVerified)

            val checkpointBytes = checkpoint.readBytes()
            assertEquals('A'.code.toByte(), checkpointBytes[0])
            assertEquals('R'.code.toByte(), checkpointBytes[1])
            assertEquals('G'.code.toByte(), checkpointBytes[2])
            assertEquals('5'.code.toByte(), checkpointBytes[3])
            assertEquals(5, ByteBuffer.wrap(checkpointBytes).order(ByteOrder.LITTLE_ENDIAN).getInt(4))
            assertEquals(96, ByteBuffer.wrap(checkpointBytes).order(ByteOrder.LITTLE_ENDIAN).getInt(8))

            val header = ByteBuffer.wrap(checkpointBytes).order(ByteOrder.LITTLE_ENDIAN)
            val architectureIdLength = header.getInt(16)
            val fingerprintOffset = 96 + architectureIdLength
            val persistedFingerprint = checkpointBytes.copyOfRange(
                fingerprintOffset,
                fingerprintOffset + 32
            )
            assertArrayEquals(hexToBytes(tokenizerFingerprint), persistedFingerprint)

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
            assertEquals(ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V1, restoredConfig.tokenizerConfig.algorithm)

            val restored = ArgusNeuralEngine.createBlocking(context, restoredConfig)
            assertTrue(restored.isCheckpointStateVerified)
            assertTrue(restored.loadCheckpointIfExists())
            assertEquals(original.tokenizer.exportVocab(), restored.tokenizer.exportVocab())
            assertEquals(original.tokenizer.exportMerges(), restored.tokenizer.exportMerges())
            assertEquals(tokenizerFingerprint, restored.tokenizer.snapshotFingerprint())
            assertArrayEquals(encodedBefore, restored.tokenizer.encodeToIds(trainingText))
            assertEquals(original.countTrainableParameters(), restored.countTrainableParameters())
            assertEquals(original.countTotalParameters(), restored.countTotalParameters())
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    @Test
    fun arg5LoaderRejectsTokenizerFingerprintMismatchEvenWithRecomputedPayloadChecksum() {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            checkpoint.delete()
            val original = ArgusNeuralEngine.createBlocking(
                context,
                smallEngineConfig(TokenizerConfig(maxVocabularySize = 512, maxTokenLength = 32))
            )
            assertTrue(original.saveCheckpoint())

            val corrupted = checkpoint.readBytes()
            val header = ByteBuffer.wrap(corrupted).order(ByteOrder.LITTLE_ENDIAN)
            val architectureIdLength = header.getInt(16)
            val fingerprintOffset = 96 + architectureIdLength
            corrupted[fingerprintOffset] = (corrupted[fingerprintOffset].toInt() xor 0x01).toByte()

            val checksum = CRC32()
            checksum.update(corrupted, 96, corrupted.size - 96)
            ByteBuffer.wrap(corrupted).order(ByteOrder.LITTLE_ENDIAN)
                .putInt(88, checksum.value.toInt())
            checkpoint.writeBytes(corrupted)

            val restored = ArgusNeuralEngine(
                context.applicationContext,
                smallEngineConfig(TokenizerConfig(maxVocabularySize = 512, maxTokenLength = 32))
            )
            assertFalse(restored.loadCheckpointIfExists())
            assertFalse(restored.isCheckpointStateVerified)
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    @Test
    fun preReleaseArg3AndArg4CheckpointsAreRejectedWithoutMigration() {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            for (magic in listOf("ARG3", "ARG4")) {
                val legacyHeader = ByteArray(if (magic == "ARG3") 48 else 96)
                magic.toByteArray(Charsets.US_ASCII).copyInto(legacyHeader)
                checkpoint.writeBytes(legacyHeader)

                val engine = ArgusNeuralEngine(
                    context.applicationContext,
                    smallEngineConfig(TokenizerConfig(maxVocabularySize = 512, maxTokenLength = 32))
                )
                assertFalse("$magic is obsolete under the pre-release clean-break policy.", engine.loadCheckpointIfExists())
                assertFalse(engine.isCheckpointStateVerified)
                assertTrue("Unsupported checkpoint is not deleted automatically.", checkpoint.isFile)
            }
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    @Test
    fun multiLayerArg5RoundTripRestoresEveryTransformerBlock() {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            checkpoint.delete()
            val config = EngineConfig(
                layerCount = 2,
                dModel = 16,
                nHeads = 4,
                dFf = 32,
                maxSeqLen = 16,
                loraRank = 2,
                tokenizerConfig = TokenizerConfig(
                    maxVocabularySize = 512,
                    maxTokenLength = 32
                )
            )
            val original = ArgusNeuralEngine.createBlocking(context, config)
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
            assertTrue(restored.isCheckpointStateVerified)
            assertEquals(2, restored.layerCount)
            assertEquals(3.140625f, restored.blocks[1].wDown[0], 1e-6f)
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    @Test
    fun configurableVocabularyCapacityKeepsTheReservedBaseAndRejectsOverflow() {
        val tokenizer = Tokenizer(
            TokenizerConfig(
                maxVocabularySize = TokenizerConfig.MINIMUM_VOCAB_SIZE,
                maxTokenLength = TokenizerConfig.MINIMUM_TOKEN_LENGTH
            )
        )
        assertEquals(TokenizerConfig.MINIMUM_VOCAB_SIZE, tokenizer.vocabSize)
        val failure = runCatching { tokenizer.registerToken("overflow") }.exceptionOrNull()
        assertTrue(failure is IllegalStateException)
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

    private fun hexToBytes(hex: String): ByteArray =
        ByteArray(hex.length / 2) { index ->
            val start = index * 2
            hex.substring(start, start + 2).toInt(16).toByte()
        }

    private fun restoreCheckpoint(file: File, previous: ByteArray?) {
        if (previous == null) {
            file.delete()
        } else {
            file.writeBytes(previous)
        }
    }

    private companion object {
        const val CHECKPOINT_FILE_NAME = "argus_model_weights.bin"
    }
}
