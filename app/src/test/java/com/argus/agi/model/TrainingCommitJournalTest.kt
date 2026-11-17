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
