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
import org.junit.Assert.assertNull
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
    fun arg6RoundTripRestoresArchitectureTokenizerAndOptimizerBoundState() {
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
            val modelStateBeforeSave = original.snapshotModelState()
            assertEquals(original.architectureConfig, modelStateBeforeSave.identity.architecture)
            assertEquals(original.tokenizer.vocabSize, modelStateBeforeSave.identity.tokenizer.vocabularySize)
            assertEquals(tokenizerFingerprint, modelStateBeforeSave.identity.tokenizer.snapshotFingerprint)
            assertEquals(0, modelStateBeforeSave.optimizerStep)
            assertNull(modelStateBeforeSave.checkpointOptimizerConfigurationFingerprint)
            assertEquals(ModelCheckpointFormat.ARG6, modelStateBeforeSave.checkpointFormat)
            assertFalse(modelStateBeforeSave.checkpointVerified)
            assertEquals(64, modelStateBeforeSave.identity.fingerprint.length)

            assertTrue(original.saveCheckpoint())
            val modelStateAfterSave = original.snapshotModelState()
            assertTrue(modelStateAfterSave.checkpointVerified)
            assertEquals(
                modelStateBeforeSave.optimizerConfiguration.fingerprint,
                modelStateAfterSave.checkpointOptimizerConfigurationFingerprint
            )
            assertEquals(modelStateBeforeSave.identity, modelStateAfterSave.identity)
            assertEquals(modelStateBeforeSave.identity.fingerprint, modelStateAfterSave.identity.fingerprint)
            assertTrue(original.isCheckpointStateVerified)

            val checkpointBytes = checkpoint.readBytes()
            assertEquals('A'.code.toByte(), checkpointBytes[0])
            assertEquals('R'.code.toByte(), checkpointBytes[1])
            assertEquals('G'.code.toByte(), checkpointBytes[2])
            assertEquals('6'.code.toByte(), checkpointBytes[3])
            assertEquals(6, ByteBuffer.wrap(checkpointBytes).order(ByteOrder.LITTLE_ENDIAN).getInt(4))
            assertEquals(96, ByteBuffer.wrap(checkpointBytes).order(ByteOrder.LITTLE_ENDIAN).getInt(8))

            val header = ByteBuffer.wrap(checkpointBytes).order(ByteOrder.LITTLE_ENDIAN)
            val architectureIdLength = header.getInt(16)
            val persistedArchitectureId = String(
                checkpointBytes,
                96,
                architectureIdLength,
                Charsets.UTF_8
            )
            assertEquals("argus.transformer.swiglu.rope.v2", persistedArchitectureId)
            val fingerprintOffset = 96 + architectureIdLength
            val persistedFingerprint = checkpointBytes.copyOfRange(
                fingerprintOffset,
                fingerprintOffset + 32
            )
            assertArrayEquals(hexToBytes(tokenizerFingerprint), persistedFingerprint)
            val persistedOptimizerFingerprint = checkpointBytes.copyOfRange(
                fingerprintOffset + 32,
                fingerprintOffset + 64
            )
            assertArrayEquals(
                hexToBytes(modelStateBeforeSave.optimizerConfiguration.fingerprint),
                persistedOptimizerFingerprint
            )

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
            val restoredModelState = restored.snapshotModelState()
            assertEquals(modelStateAfterSave.identity, restoredModelState.identity)
            assertEquals(modelStateAfterSave.identity.fingerprint, restoredModelState.identity.fingerprint)
            assertEquals(ModelCheckpointFormat.ARG6, restoredModelState.checkpointFormat)
            assertEquals(
                modelStateAfterSave.optimizerConfiguration.fingerprint,
                restoredModelState.checkpointOptimizerConfigurationFingerprint
            )
            assertEquals(modelStateAfterSave.optimizerStep, restoredModelState.optimizerStep)
            assertTrue(restoredModelState.checkpointVerified)
            assertArrayEquals(encodedBefore, restored.tokenizer.encodeToIds(trainingText))
            assertEquals(original.countTrainableParameters(), restored.countTrainableParameters())
            assertEquals(original.countTotalParameters(), restored.countTotalParameters())
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    @Test
    fun arg6LoaderRejectsTokenizerFingerprintMismatchEvenWithRecomputedPayloadChecksum() {
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

            val restored = ArgusNeuralEngine.createBlocking(
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
    fun arg5CheckpointRemainsReadableAndReportsMissingOptimizerIdentity() {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            checkpoint.delete()
            val config = smallEngineConfig(
                TokenizerConfig(maxVocabularySize = 512, maxTokenLength = 32)
            )
            val original = ArgusNeuralEngine.createBlocking(context, config)
            assertTrue(original.saveCheckpoint())
            convertArg6CheckpointToArg5(checkpoint)

            val restored = ArgusNeuralEngine.createBlocking(context, config)
            assertTrue(restored.isCheckpointStateVerified)
            assertFalse(restored.isCheckpointRestoreBlocked)
            assertEquals(ModelCheckpointFormat.ARG5, restored.snapshotModelState().checkpointFormat)
            assertNull(restored.snapshotModelState().checkpointOptimizerConfigurationFingerprint)
            assertEquals(original.tokenizer.exportVocab(), restored.tokenizer.exportVocab())
            assertEquals(original.tokenizer.snapshotFingerprint(), restored.tokenizer.snapshotFingerprint())
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    @Test
    fun arg6LoaderRejectsOptimizerConfigurationMismatchEvenWithRecomputedPayloadChecksum() {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            checkpoint.delete()
            val config = smallEngineConfig(
                TokenizerConfig(maxVocabularySize = 512, maxTokenLength = 32)
            )
            val original = ArgusNeuralEngine.createBlocking(context, config)
            assertTrue(original.saveCheckpoint())

            val corrupted = checkpoint.readBytes()
            val header = ByteBuffer.wrap(corrupted).order(ByteOrder.LITTLE_ENDIAN)
            val architectureIdLength = header.getInt(16)
            val optimizerFingerprintOffset = 96 + architectureIdLength + 32
            corrupted[optimizerFingerprintOffset] =
                (corrupted[optimizerFingerprintOffset].toInt() xor 0x01).toByte()

            val checksum = CRC32()
            checksum.update(corrupted, 96, corrupted.size - 96)
            ByteBuffer.wrap(corrupted).order(ByteOrder.LITTLE_ENDIAN)
                .putInt(88, checksum.value.toInt())
            checkpoint.writeBytes(corrupted)

            val restored = ArgusNeuralEngine.createBlocking(context, config)
            val rejectedBytes = checkpoint.readBytes()
            assertFalse(restored.loadCheckpointIfExists())
            assertFalse(restored.isCheckpointStateVerified)
            assertTrue(restored.isCheckpointRestoreBlocked)
            assertFalse(restored.saveCheckpoint())
            assertArrayEquals(rejectedBytes, checkpoint.readBytes())
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    @Test
    fun arg6LoaderRejectsPreRopeArchitectureIdUnderCleanBreak() {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            checkpoint.delete()
            val config = smallEngineConfig(
                TokenizerConfig(maxVocabularySize = 512, maxTokenLength = 32)
            )
            val original = ArgusNeuralEngine.createBlocking(context, config)
            assertTrue(original.saveCheckpoint())

            // Rebind the otherwise valid, checksum-consistent payload to the
            // retired sinusoidal architecture ID. Under the Section 8.0 clean
            // break the loader must refuse it at the architecture-ID gate
            // instead of interpreting its tensors with the rotary layout.
            rewritePersistedArchitectureId(checkpoint, "argus.transformer.swiglu.v1")

            val restored = ArgusNeuralEngine.createBlocking(context, config)
            val rejectedBytes = checkpoint.readBytes()
            assertFalse(restored.loadCheckpointIfExists())
            assertFalse(restored.isCheckpointStateVerified)
            assertTrue(restored.isCheckpointRestoreBlocked)
            assertArrayEquals(rejectedBytes, checkpoint.readBytes())
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

                val engine = ArgusNeuralEngine.createBlocking(
                    context.applicationContext,
                    smallEngineConfig(TokenizerConfig(maxVocabularySize = 512, maxTokenLength = 32))
                )
                val rejectedBytes = checkpoint.readBytes()
                assertFalse("$magic is obsolete under the pre-release clean-break policy.", engine.loadCheckpointIfExists())
                assertFalse(engine.isCheckpointStateVerified)
                assertTrue(engine.isCheckpointRestoreBlocked)
                assertFalse(engine.saveCheckpoint())
                assertArrayEquals(rejectedBytes, checkpoint.readBytes())
                assertTrue("Unsupported checkpoint is not deleted automatically.", checkpoint.isFile)
            }
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    @Test
    fun multiLayerArg6RoundTripRestoresEveryTransformerBlock() {
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
        val failure = runCatching { tokenizer.registerToken("overflow-token-abc") }.exceptionOrNull()
        assertTrue(failure is IllegalStateException)
        assertEquals(TokenizerConfig.MINIMUM_VOCAB_SIZE, tokenizer.vocabSize)
    }

    @Test
    fun heldOutEvaluationExcludesContextTokenFromBitsPerByteDenominator() {
        val engine = ArgusNeuralEngine.createBlocking(
            context,
            smallEngineConfig(TokenizerConfig(maxVocabularySize = 512, maxTokenLength = 32))
        )
        val text = "Aé"

        val evaluation = engine.evaluateLossWithTargetCount(text)

        assertTrue(evaluation.meanLoss.isFinite())
        assertEquals(2, evaluation.targetCount)
        assertEquals(2L, evaluation.scoredUtf8ByteCount)
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

    private fun convertArg6CheckpointToArg5(file: File) {
        val arg6Bytes = file.readBytes()
        val header = ByteBuffer.wrap(arg6Bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(arg6Bytes[3] == '6'.code.toByte())
        val architectureIdLength = header.getInt(16)
        val optimizerFingerprintOffset = 96 + architectureIdLength + 32
        val arg5Bytes = ByteArray(arg6Bytes.size - 32)
        arg6Bytes.copyInto(arg5Bytes, 0, 0, optimizerFingerprintOffset)
        arg6Bytes.copyInto(
            arg5Bytes,
            optimizerFingerprintOffset,
            optimizerFingerprintOffset + 32,
            arg6Bytes.size
        )

        val legacyHeader = ByteBuffer.wrap(arg5Bytes).order(ByteOrder.LITTLE_ENDIAN)
        arg5Bytes[3] = '5'.code.toByte()
        legacyHeader.putInt(4, 5)
        legacyHeader.putLong(80, header.getLong(80) - 32L)
        legacyHeader.putInt(92, 0)
        val checksum = CRC32().apply {
            update(arg5Bytes, 96, arg5Bytes.size - 96)
        }
        legacyHeader.putInt(88, checksum.value.toInt())
        file.writeBytes(arg5Bytes)
    }

    private fun rewritePersistedArchitectureId(file: File, replacementId: String) {
        val originalBytes = file.readBytes()
        val header = ByteBuffer.wrap(originalBytes).order(ByteOrder.LITTLE_ENDIAN)
        val currentIdLength = header.getInt(16)
        val replacementBytes = replacementId.toByteArray(Charsets.UTF_8)

        val payload = ByteArray(
            originalBytes.size - 96 - currentIdLength + replacementBytes.size
        )
        replacementBytes.copyInto(payload, 0)
        originalBytes.copyInto(
            payload,
            replacementBytes.size,
            96 + currentIdLength,
            originalBytes.size
        )

        val rewritten = ByteArray(96 + payload.size)
        originalBytes.copyInto(rewritten, 0, 0, 96)
        payload.copyInto(rewritten, 96)
        val rewrittenHeader = ByteBuffer.wrap(rewritten).order(ByteOrder.LITTLE_ENDIAN)
        rewrittenHeader.putInt(16, replacementBytes.size)
        rewrittenHeader.putLong(80, payload.size.toLong())
        val checksum = CRC32().apply { update(payload) }
        rewrittenHeader.putInt(88, checksum.value.toInt())
        file.writeBytes(rewritten)
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
