package com.argus.agi.model

import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.zip.CRC32

internal data class TrainingCommitJournalEntry(
    val itemId: String,
    val result: NeuralTrainingStepResult?,
    val stepDurationMs: Long
)

internal data class TrainingCommitJournalSnapshot(
    val baselineOptimizerStep: Int,
    val committedOptimizerStep: Int?,
    val entries: List<TrainingCommitJournalEntry>
) {
    val isCommitted: Boolean
        get() = committedOptimizerStep != null
}

/**
 * A single-batch write-ahead journal that binds a Room queue reservation to the
 * checkpoint optimizer step. The model checkpoint is written after COMMITTED and
 * before Room changes the item status, making crash recovery deterministic.
 */
internal object TrainingCommitJournal {

    const val FILE_NAME = "argus_training_commit.journal"

    private const val MAGIC = 0x4154474A // ATGJ
    private const val VERSION = 1
    private const val STATE_PREPARED = 0
    private const val STATE_COMMITTED = 1
    private const val MAX_JOURNAL_BYTES = 64 * 1024
    private const val MAX_ITEMS = 8
    private const val MAX_ITEM_ID_BYTES = 128

    fun isExactBaselineCheckpoint(
        snapshot: TrainingCommitJournalSnapshot,
        optimizerStep: Int
    ): Boolean = optimizerStep == snapshot.baselineOptimizerStep

    fun isExactCommittedCheckpoint(
        snapshot: TrainingCommitJournalSnapshot,
        optimizerStep: Int
    ): Boolean = snapshot.isCommitted && optimizerStep == snapshot.committedOptimizerStep

    fun writePrepared(
        file: File,
        baselineOptimizerStep: Int,
        itemIds: List<String>
    ) {
        require(baselineOptimizerStep >= 0) {
            "Baseline optimizer step cannot be negative."
        }
        require(itemIds.size in 1..MAX_ITEMS) {
            "A journal batch must contain between one and eight items."
        }
        require(itemIds.distinct().size == itemIds.size) {
            "A journal batch cannot contain duplicate item IDs."
        }
        val entries = itemIds.map { itemId ->
            TrainingCommitJournalEntry(
                itemId = itemId,
                result = null,
                stepDurationMs = 0L
            )
        }
        writeSnapshot(
            file = file,
            state = STATE_PREPARED,
            baselineOptimizerStep = baselineOptimizerStep,
            committedOptimizerStep = null,
            entries = entries
        )
    }

    fun writeCommitted(
        file: File,
        baselineOptimizerStep: Int,
        committedOptimizerStep: Int,
        entries: List<TrainingCommitJournalEntry>
    ) {
        require(baselineOptimizerStep >= 0)
        require(committedOptimizerStep >= baselineOptimizerStep) {
            "Committed optimizer step cannot precede the batch baseline."
        }
        require(entries.size in 1..MAX_ITEMS)
        require(entries.map(TrainingCommitJournalEntry::itemId).distinct().size == entries.size)
        require(entries.all { entry ->
            val result = entry.result
            entry.itemId.isNotBlank() &&
                entry.stepDurationMs >= 0L &&
                result != null &&
                result.inputTokenCount >= 0 &&
                result.predictionTargetCount >= 0 &&
                result.contextWindowCount >= 0 &&
                result.optimizerUpdateCount >= 0 &&
                result.vocabularySize > 0
        }) {
            "Committed journal entries contain invalid training results."
        }
        writeSnapshot(
            file = file,
            state = STATE_COMMITTED,
            baselineOptimizerStep = baselineOptimizerStep,
            committedOptimizerStep = committedOptimizerStep,
            entries = entries
        )
    }

    fun read(file: File): TrainingCommitJournalSnapshot? {
        if (!file.isFile) return null
        require(file.length() in 16L..MAX_JOURNAL_BYTES.toLong()) {
            "Training commit journal size is invalid."
        }

        DataInputStream(BufferedInputStream(FileInputStream(file))).use { input ->
            val payloadSize = input.readInt()
            require(payloadSize in 1..MAX_JOURNAL_BYTES - Int.SIZE_BYTES - Long.SIZE_BYTES) {
                "Training commit journal payload size is invalid."
            }
            val payload = ByteArray(payloadSize)
            input.readFully(payload)
            val savedChecksum = input.readLong()
            require(input.read() == -1) {
                "Training commit journal contains trailing bytes."
            }
            val checksum = CRC32().apply { update(payload) }
            require(checksum.value == savedChecksum) {
                "Training commit journal checksum mismatch."
            }

            DataInputStream(ByteArrayInputStream(payload)).use { body ->
                require(body.readInt() == MAGIC) {
                    "Training commit journal magic is invalid."
                }
                require(body.readInt() == VERSION) {
                    "Training commit journal version is unsupported."
                }
                val state = body.readInt()
                require(state == STATE_PREPARED || state == STATE_COMMITTED) {
                    "Training commit journal state is invalid."
                }
                val baselineOptimizerStep = body.readInt()
                val committedOptimizerStep = body.readInt()
                val count = body.readInt()
                require(baselineOptimizerStep >= 0 && count in 1..MAX_ITEMS)
                require(
                    if (state == STATE_COMMITTED) committedOptimizerStep >= baselineOptimizerStep
                    else committedOptimizerStep == -1
                ) { "Training commit journal optimizer-step metadata is inconsistent." }

                val entries = ArrayList<TrainingCommitJournalEntry>(count)
                repeat(count) {
                    val idLength = body.readUnsignedShort()
                    require(idLength in 1..MAX_ITEM_ID_BYTES) {
                        "Training commit journal item ID length is invalid."
                    }
                    val idBytes = ByteArray(idLength)
                    body.readFully(idBytes)
                    val itemId = String(idBytes, Charsets.UTF_8)
                    require(itemId.isNotBlank()) {
                        "Training commit journal contains an empty item ID."
                    }

                    val entry = if (state == STATE_COMMITTED) {
                        val loss = Float.fromBits(body.readInt())
                        val inputTokens = body.readInt()
                        val predictionTargets = body.readInt()
                        val contextWindows = body.readInt()
                        val optimizerUpdates = body.readInt()
                        val vocabularySize = body.readInt()
                        val durationMs = body.readLong()
                        require(
                            inputTokens >= 0 &&
                                predictionTargets >= 0 &&
                                contextWindows >= 0 &&
                                optimizerUpdates >= 0 &&
                                vocabularySize > 0 &&
                                durationMs >= 0L
                        ) { "Training commit journal result metadata is invalid." }
                        TrainingCommitJournalEntry(
                            itemId = itemId,
                            result = NeuralTrainingStepResult(
                                loss = loss,
                                inputTokenCount = inputTokens,
                                predictionTargetCount = predictionTargets,
                                contextWindowCount = contextWindows,
                                optimizerUpdateCount = optimizerUpdates,
                                vocabularySize = vocabularySize
                            ),
                            stepDurationMs = durationMs
                        )
                    } else {
                        TrainingCommitJournalEntry(itemId, null, 0L)
                    }
                    entries.add(entry)
                }
                require(body.read() == -1) {
                    "Training commit journal payload contains trailing data."
                }
                require(entries.map(TrainingCommitJournalEntry::itemId).distinct().size == count) {
                    "Training commit journal contains duplicate item IDs."
                }

                return TrainingCommitJournalSnapshot(
                    baselineOptimizerStep = baselineOptimizerStep,
                    committedOptimizerStep = if (state == STATE_COMMITTED) {
                        committedOptimizerStep
                    } else {
                        null
                    },
                    entries = entries
                )
            }
        }
    }

    fun delete(file: File) {
        val parent = file.parentFile
        val temporary = parent?.let { File(it, "${file.name}.tmp") }
        if (temporary != null && temporary.exists() && !temporary.delete()) {
            throw IOException("Could not remove the temporary training commit journal.")
        }
        if (file.exists() && !file.delete()) {
            throw IOException("Could not remove the training commit journal.")
        }
    }

    private fun writeSnapshot(
        file: File,
        state: Int,
        baselineOptimizerStep: Int,
        committedOptimizerStep: Int?,
        entries: List<TrainingCommitJournalEntry>
    ) {
        val payloadBuffer = ByteArrayOutputStream()
        DataOutputStream(payloadBuffer).use { output ->
            output.writeInt(MAGIC)
            output.writeInt(VERSION)
            output.writeInt(state)
            output.writeInt(baselineOptimizerStep)
            output.writeInt(committedOptimizerStep ?: -1)
            output.writeInt(entries.size)
            for (entry in entries) {
                val idBytes = entry.itemId.toByteArray(Charsets.UTF_8)
                require(idBytes.isNotEmpty() && idBytes.size <= MAX_ITEM_ID_BYTES)
                output.writeShort(idBytes.size)
                output.write(idBytes)
                if (state == STATE_COMMITTED) {
                    val result = requireNotNull(entry.result)
                    output.writeInt(result.loss.toRawBits())
                    output.writeInt(result.inputTokenCount)
                    output.writeInt(result.predictionTargetCount)
                    output.writeInt(result.contextWindowCount)
                    output.writeInt(result.optimizerUpdateCount)
                    output.writeInt(result.vocabularySize)
                    output.writeLong(entry.stepDurationMs)
                }
            }
        }

        val payload = payloadBuffer.toByteArray()
        require(payload.size <= MAX_JOURNAL_BYTES - Int.SIZE_BYTES - Long.SIZE_BYTES) {
            "Training commit journal exceeds its maximum size."
        }
        val checksum = CRC32().apply { update(payload) }
        val parent = file.parentFile
        require(parent != null && (parent.isDirectory || parent.mkdirs())) {
            "Training commit journal directory is unavailable."
        }
        val temporary = File(parent, "${file.name}.tmp")
        try {
            FileOutputStream(temporary).use { stream ->
                val output = DataOutputStream(BufferedOutputStream(stream))
                output.writeInt(payload.size)
                output.write(payload)
                output.writeLong(checksum.value)
                output.flush()
                stream.fd.sync()
            }
            try {
                Files.move(
                    temporary.toPath(),
                    file.toPath(),
                    StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING
                )
            } catch (unsupported: AtomicMoveNotSupportedException) {
                try {
                    Files.move(
                        temporary.toPath(),
                        file.toPath(),
                        StandardCopyOption.REPLACE_EXISTING
                    )
                } catch (fallbackError: IOException) {
                    fallbackError.addSuppressed(unsupported)
                    throw fallbackError
                }
            }
        } finally {
            if (temporary.exists()) temporary.delete()
        }
    }
}
