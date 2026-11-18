package com.argus.agi.model

import java.io.File
import java.nio.file.Files
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TrainingCommitJournalTest {

    @Test
    fun preparedJournalRecordsBaselineAndItemIdentity() {
        val directory = Files.createTempDirectory("argus-training-journal").toFile()
        val file = File(directory, TrainingCommitJournal.FILE_NAME)
        try {
            TrainingCommitJournal.writePrepared(
                file = file,
                baselineOptimizerStep = 41,
                itemIds = listOf("item-a", "item-b")
            )

            val restored = requireNotNull(TrainingCommitJournal.read(file))
            assertFalse(restored.isCommitted)
            assertEquals(41, restored.baselineOptimizerStep)
            assertNull(restored.committedOptimizerStep)
            assertEquals(listOf("item-a", "item-b"), restored.entries.map { it.itemId })
            assertNull(restored.modelIdentityFingerprint)
            assertTrue(
                "Version-1 journals preserve their legacy step-only recovery path.",
                TrainingCommitJournal.matchesModelIdentity(restored, "f".repeat(64))
            )
            assertTrue(restored.entries.all { it.result == null })
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun committedJournalRoundTripsEveryPerItemTrainingResult() {
        val directory = Files.createTempDirectory("argus-training-journal").toFile()
        val file = File(directory, TrainingCommitJournal.FILE_NAME)
        try {
            val result = NeuralTrainingStepResult(
                loss = 0.625f,
                inputTokenCount = 32,
                predictionTargetCount = 30,
                contextWindowCount = 2,
                optimizerUpdateCount = 1,
                vocabularySize = 512
            )
            TrainingCommitJournal.writeCommitted(
                file = file,
                baselineOptimizerStep = 41,
                committedOptimizerStep = 42,
                entries = listOf(
                    TrainingCommitJournalEntry(
                        itemId = "item-a",
                        result = result,
                        stepDurationMs = 85L
                    )
                )
            )

            val restored = requireNotNull(TrainingCommitJournal.read(file))
            val restoredEntry = restored.entries.single()
            val restoredResult = requireNotNull(restoredEntry.result)
            assertTrue(restored.isCommitted)
            assertEquals(41, restored.baselineOptimizerStep)
            assertEquals(42, restored.committedOptimizerStep)
            assertEquals("item-a", restoredEntry.itemId)
            assertEquals(result.loss, restoredResult.loss, 0f)
            assertEquals(result.inputTokenCount, restoredResult.inputTokenCount)
            assertEquals(result.predictionTargetCount, restoredResult.predictionTargetCount)
            assertEquals(result.contextWindowCount, restoredResult.contextWindowCount)
            assertEquals(result.optimizerUpdateCount, restoredResult.optimizerUpdateCount)
            assertEquals(result.vocabularySize, restoredResult.vocabularySize)
            assertEquals(85L, restoredEntry.stepDurationMs)
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun versionTwoJournalPersistsAndVerifiesModelIdentityFingerprint() {
        val directory = Files.createTempDirectory("argus-training-journal-v2").toFile()
        val file = File(directory, TrainingCommitJournal.FILE_NAME)
        val fingerprint = "0123456789abcdef".repeat(4)
        try {
            TrainingCommitJournal.writePrepared(
                file = file,
                baselineOptimizerStep = 12,
                itemIds = listOf("item-a"),
                modelIdentityFingerprint = fingerprint
            )

            val bytes = file.readBytes()
            assertEquals(
                2,
                java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.BIG_ENDIAN).getInt(8)
            )

            val restored = requireNotNull(TrainingCommitJournal.read(file))
            assertEquals(fingerprint, restored.modelIdentityFingerprint)
            assertNull(restored.optimizerConfigFingerprint)
            assertTrue(TrainingCommitJournal.matchesModelIdentity(restored, fingerprint))
            assertFalse(TrainingCommitJournal.matchesModelIdentity(restored, "f".repeat(64)))
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun versionThreeJournalBindsModelAndOptimizerConfigurationIdentities() {
        val directory = Files.createTempDirectory("argus-training-journal-v3").toFile()
        val file = File(directory, TrainingCommitJournal.FILE_NAME)
        val modelFingerprint = "0123456789abcdef".repeat(4)
        val optimizerFingerprint = OptimizerConfig.DEFAULT.fingerprint
        try {
            TrainingCommitJournal.writePrepared(
                file = file,
                baselineOptimizerStep = 13,
                itemIds = listOf("item-a"),
                modelIdentityFingerprint = modelFingerprint,
                optimizerConfigFingerprint = optimizerFingerprint
            )
            val bytes = file.readBytes()
            assertEquals(3, java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.BIG_ENDIAN).getInt(8))
            val restored = requireNotNull(TrainingCommitJournal.read(file))
            assertEquals(modelFingerprint, restored.modelIdentityFingerprint)
            assertEquals(optimizerFingerprint, restored.optimizerConfigFingerprint)
            assertTrue(TrainingCommitJournal.matchesModelIdentity(restored, modelFingerprint))
            assertFalse(TrainingCommitJournal.matchesModelIdentity(restored, "f".repeat(64)))
            assertTrue(TrainingCommitJournal.matchesOptimizerConfiguration(restored, optimizerFingerprint))
            assertFalse(TrainingCommitJournal.matchesOptimizerConfiguration(restored, "a".repeat(64)))
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun versionThreeAcceptsVerifiedLegacyArg5CheckpointAsUnverifiedCompatibility() {
        val directory = Files.createTempDirectory("argus-training-journal-v3-arg5").toFile()
        val file = File(directory, TrainingCommitJournal.FILE_NAME)
        val modelFingerprint = "1234567890abcdef".repeat(4)
        val optimizerFingerprint = OptimizerConfig.DEFAULT.fingerprint
        try {
            TrainingCommitJournal.writePrepared(
                file = file,
                baselineOptimizerStep = 8,
                itemIds = listOf("legacy-arg5"),
                modelIdentityFingerprint = modelFingerprint,
                optimizerConfigFingerprint = optimizerFingerprint
            )
            val snapshot = requireNotNull(TrainingCommitJournal.read(file))
            assertTrue(
                TrainingCommitJournal.allowsCheckpointForRecovery(
                    snapshot = snapshot,
                    checkpointFormat = ModelCheckpointFormat.ARG5,
                    checkpointVerified = true,
                    checkpointOptimizerConfigFingerprint = null
                )
            )
            assertFalse(
                TrainingCommitJournal.allowsCheckpointForRecovery(
                    snapshot = snapshot,
                    checkpointFormat = ModelCheckpointFormat.ARG5,
                    checkpointVerified = false,
                    checkpointOptimizerConfigFingerprint = null
                )
            )
            assertFalse(
                TrainingCommitJournal.allowsCheckpointForRecovery(
                    snapshot = snapshot,
                    checkpointFormat = ModelCheckpointFormat.ARG6,
                    checkpointVerified = true,
                    checkpointOptimizerConfigFingerprint = "f".repeat(64)
                )
            )
            assertTrue(
                TrainingCommitJournal.allowsCheckpointForRecovery(
                    snapshot = snapshot,
                    checkpointFormat = ModelCheckpointFormat.ARG6,
                    checkpointVerified = true,
                    checkpointOptimizerConfigFingerprint = optimizerFingerprint
                )
            )
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun versionThreeRejectsMalformedOptimizerConfigurationFingerprint() {
        val directory = Files.createTempDirectory("argus-training-journal-v3-invalid").toFile()
        val file = File(directory, TrainingCommitJournal.FILE_NAME)
        try {
            val failure = runCatching {
                TrainingCommitJournal.writePrepared(
                    file = file,
                    baselineOptimizerStep = 3,
                    itemIds = listOf("item-a"),
                    modelIdentityFingerprint = "f".repeat(64),
                    optimizerConfigFingerprint = "not-a-fingerprint"
                )
            }.exceptionOrNull()
            assertNotNull(failure)
            assertFalse(file.exists())
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun versionThreeCommittedJournalPersistsBothFingerprintsAndTrainingResults() {
        val directory = Files.createTempDirectory("argus-training-journal-v3-committed").toFile()
        val file = File(directory, TrainingCommitJournal.FILE_NAME)
        val modelFingerprint = "1234567890abcdef".repeat(4)
        val optimizerFingerprint = OptimizerConfig.DEFAULT.fingerprint
        val result = NeuralTrainingStepResult(
            loss = 0.375f,
            inputTokenCount = 18,
            predictionTargetCount = 17,
            contextWindowCount = 1,
            optimizerUpdateCount = 1,
            vocabularySize = 512
        )
        try {
            TrainingCommitJournal.writeCommitted(
                file = file,
                baselineOptimizerStep = 40,
                committedOptimizerStep = 41,
                entries = listOf(
                    TrainingCommitJournalEntry(
                        itemId = "committed-v3",
                        result = result,
                        stepDurationMs = 21L
                    )
                ),
                modelIdentityFingerprint = modelFingerprint,
                optimizerConfigFingerprint = optimizerFingerprint
            )

            val bytes = file.readBytes()
            assertEquals(
                3,
                java.nio.ByteBuffer.wrap(bytes).order(java.nio.ByteOrder.BIG_ENDIAN).getInt(8)
            )
            val restored = requireNotNull(TrainingCommitJournal.read(file))
            assertEquals(modelFingerprint, restored.modelIdentityFingerprint)
            assertEquals(optimizerFingerprint, restored.optimizerConfigFingerprint)
            assertEquals(41, restored.committedOptimizerStep)
            assertEquals("committed-v3", restored.entries.single().itemId)
            assertEquals(result.loss, requireNotNull(restored.entries.single().result).loss, 0f)
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun versionTwoCommittedJournalKeepsModelIdentityAlongsideTrainingResults() {
        val directory = Files.createTempDirectory("argus-training-journal-v2-committed").toFile()
        val file = File(directory, TrainingCommitJournal.FILE_NAME)
        val fingerprint = "abcdef0123456789".repeat(4)
        try {
            TrainingCommitJournal.writeCommitted(
                file = file,
                baselineOptimizerStep = 20,
                committedOptimizerStep = 21,
                entries = listOf(
                    TrainingCommitJournalEntry(
                        itemId = "committed-a",
                        result = NeuralTrainingStepResult(
                            loss = 0.75f,
                            inputTokenCount = 16,
                            predictionTargetCount = 15,
                            contextWindowCount = 1,
                            optimizerUpdateCount = 1,
                            vocabularySize = 512
                        ),
                        stepDurationMs = 9L
                    )
                ),
                modelIdentityFingerprint = fingerprint
            )

            val restored = requireNotNull(TrainingCommitJournal.read(file))
            assertEquals(fingerprint, restored.modelIdentityFingerprint)
            assertEquals(21, restored.committedOptimizerStep)
            assertEquals("committed-a", restored.entries.single().itemId)
            assertEquals(0.75f, requireNotNull(restored.entries.single().result).loss, 0f)
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun versionTwoRejectsMalformedModelIdentityFingerprints() {
        val directory = Files.createTempDirectory("argus-training-journal-v2-invalid").toFile()
        val file = File(directory, TrainingCommitJournal.FILE_NAME)
        try {
            val failure = runCatching {
                TrainingCommitJournal.writePrepared(
                    file = file,
                    baselineOptimizerStep = 3,
                    itemIds = listOf("item-a"),
                    modelIdentityFingerprint = "not-a-sha256-fingerprint"
                )
            }.exceptionOrNull()

            assertNotNull(failure)
            assertFalse(file.exists())
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun journalAcceptsOnlyExactBaselineOrCommittedOptimizerSteps() {
        val snapshot = TrainingCommitJournalSnapshot(
            baselineOptimizerStep = 41,
            committedOptimizerStep = 42,
            entries = listOf(
                TrainingCommitJournalEntry(
                    itemId = "item-a",
                    result = NeuralTrainingStepResult(
                        loss = 0.5f,
                        inputTokenCount = 12,
                        predictionTargetCount = 11,
                        contextWindowCount = 1,
                        optimizerUpdateCount = 1,
                        vocabularySize = 128
                    ),
                    stepDurationMs = 10L
                )
            )
        )

        assertTrue(TrainingCommitJournal.isExactBaselineCheckpoint(snapshot, 41))
        assertFalse(TrainingCommitJournal.isExactBaselineCheckpoint(snapshot, 40))
        assertFalse(TrainingCommitJournal.isExactBaselineCheckpoint(snapshot, 42))

        assertTrue(TrainingCommitJournal.isExactCommittedCheckpoint(snapshot, 42))
        assertFalse(TrainingCommitJournal.isExactCommittedCheckpoint(snapshot, 41))
        assertFalse(
            "A checkpoint ahead of the recorded commit must not be accepted.",
            TrainingCommitJournal.isExactCommittedCheckpoint(snapshot, 43)
        )
    }

    @Test
    fun preparedJournalRejectsDuplicateOrOversizedBatches() {
        val directory = Files.createTempDirectory("argus-training-journal").toFile()
        val file = File(directory, TrainingCommitJournal.FILE_NAME)
        try {
            val duplicateFailure = runCatching {
                TrainingCommitJournal.writePrepared(file, 4, listOf("same", "same"))
            }.exceptionOrNull()
            assertNotNull(duplicateFailure)

            val oversizedFailure = runCatching {
                TrainingCommitJournal.writePrepared(
                    file,
                    4,
                    (0..8).map { index -> "item-$index" }
                )
            }.exceptionOrNull()
            assertNotNull(oversizedFailure)
            assertFalse(file.exists())
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun corruptedJournalIsRejectedInsteadOfBeingReplayedBlindly() {
        val directory = Files.createTempDirectory("argus-training-journal").toFile()
        val file = File(directory, TrainingCommitJournal.FILE_NAME)
        try {
            TrainingCommitJournal.writePrepared(file, 7, listOf("item-a"))
            val bytes = file.readBytes()
            val checksumByte = bytes.lastIndex
            bytes[checksumByte] = (bytes[checksumByte].toInt() xor 0x01).toByte()
            file.writeBytes(bytes)

            val failure = runCatching { TrainingCommitJournal.read(file) }.exceptionOrNull()
            assertNotNull(failure)
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun deletingJournalRemovesCurrentAndTemporaryFiles() {
        val directory = Files.createTempDirectory("argus-training-journal").toFile()
        val file = File(directory, TrainingCommitJournal.FILE_NAME)
        try {
            TrainingCommitJournal.writePrepared(file, 2, listOf("item-a"))
            File(directory, "${file.name}.tmp").writeText("partial")
            TrainingCommitJournal.delete(file)

            assertFalse(file.exists())
            assertFalse(File(directory, "${file.name}.tmp").exists())
            assertNull(TrainingCommitJournal.read(file))
        } finally {
            directory.deleteRecursively()
        }
    }
}
