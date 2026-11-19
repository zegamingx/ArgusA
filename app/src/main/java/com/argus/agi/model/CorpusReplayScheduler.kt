package com.argus.agi.model

import com.argus.agi.data.local.entity.CorpusChunkEntity
import com.argus.agi.data.local.entity.CorpusDocumentEntity
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.Random

/**
 * Controls only the order in which whole-document chunk streams are scheduled.
 * Chunk order inside a source document is always preserved.
 */
enum class CorpusReplayPolicy {
    SEQUENTIAL_DOCUMENTS,
    SHUFFLED_DOCUMENTS,
    INTERLEAVED_DOCUMENTS
}

data class CorpusReplayResult(
    val scheduledDocumentCount: Int,
    val scheduledChunkCount: Int,
    val epochs: Int,
    val seed: Long,
    val policy: CorpusReplayPolicy
)

internal data class CorpusReplaySource(
    val document: CorpusDocumentEntity,
    val chunks: List<CorpusChunkEntity>
)

internal data class ScheduledCorpusChunk(
    val document: CorpusDocumentEntity,
    val chunk: CorpusChunkEntity,
    val epochNumber: Int,
    val documentChunkIndex: Int,
    val documentTotalChunks: Int
)

internal object CorpusReplayScheduler {

    const val SCHEDULER_FORMAT_VERSION = 1
    private const val SOURCE_SNAPSHOT_FORMAT_VERSION = 1

    private data class Cursor(
        val source: CorpusReplaySource,
        var nextChunkIndex: Int = 0
    )

    /**
     * Inputs are sorted by (createdAtEpochMs, documentId) before seeded scheduling.
     * Identical source snapshots, policy, epochs, and seed yield identical plans.
     */
    fun createSchedule(
        sources: List<CorpusReplaySource>,
        epochs: Int,
        seed: Long,
        policy: CorpusReplayPolicy
    ): List<ScheduledCorpusChunk> {
        require(epochs in 1..TrainingManager.MAX_DOCUMENT_EPOCHS) {
            "epochs must be in 1..${TrainingManager.MAX_DOCUMENT_EPOCHS}."
        }
        require(sources.isNotEmpty()) { "At least one training corpus document is required." }

        val stableSources = canonicalSources(sources)

        val perEpochChunkCount = stableSources.sumOf { it.chunks.size.toLong() }
        val totalChunkCount = Math.multiplyExact(perEpochChunkCount, epochs.toLong())
        require(totalChunkCount <= Int.MAX_VALUE.toLong()) {
            "The requested replay plan exceeds the supported item count."
        }

        val random = Random(seed)
        val result = ArrayList<ScheduledCorpusChunk>(totalChunkCount.toInt())

        repeat(epochs) { epochIndex ->
            when (policy) {
                CorpusReplayPolicy.SEQUENTIAL_DOCUMENTS -> {
                    stableSources.forEach { source ->
                        appendDocumentEpoch(result, source, epochIndex, epochs)
                    }
                }

                CorpusReplayPolicy.SHUFFLED_DOCUMENTS -> {
                    val shuffled = stableSources.toMutableList()
                    for (index in shuffled.lastIndex downTo 1) {
                        val swapIndex = random.nextInt(index + 1)
                        val current = shuffled[index]
                        shuffled[index] = shuffled[swapIndex]
                        shuffled[swapIndex] = current
                    }
                    shuffled.forEach { source ->
                        appendDocumentEpoch(result, source, epochIndex, epochs)
                    }
                }

                CorpusReplayPolicy.INTERLEAVED_DOCUMENTS -> {
                    val active = stableSources.map { Cursor(it) }.toMutableList()
                    while (active.isNotEmpty()) {
                        val selectedIndex = random.nextInt(active.size)
                        val cursor = active[selectedIndex]
                        appendChunk(
                            result = result,
                            source = cursor.source,
                            chunk = cursor.source.chunks[cursor.nextChunkIndex],
                            chunkIndexWithinEpoch = cursor.nextChunkIndex,
                            epochIndex = epochIndex,
                            epochs = epochs
                        )

                        cursor.nextChunkIndex += 1
                        if (cursor.nextChunkIndex >= cursor.source.chunks.size) {
                            active.removeAt(selectedIndex)
                        }
                    }
                }
            }
        }

        return result
    }

    /**
     * Hashes the exact retained source snapshot independently of the replay policy and seed.
     * Variable-length fields are length-prefixed; document and chunk ordering is canonical.
     */
    fun sourceSnapshotFingerprint(sources: List<CorpusReplaySource>): String {
        val stableSources = canonicalSources(sources)
        val digest = MessageDigest.getInstance("SHA-256")
        val integerBuffer = ByteBuffer.allocate(Int.SIZE_BYTES).order(ByteOrder.BIG_ENDIAN)
        val longBuffer = ByteBuffer.allocate(Long.SIZE_BYTES).order(ByteOrder.BIG_ENDIAN)

        fun updateInt(value: Int) {
            integerBuffer.clear()
            integerBuffer.putInt(value)
            digest.update(integerBuffer.array())
        }

        fun updateLong(value: Long) {
            longBuffer.clear()
            longBuffer.putLong(value)
            digest.update(longBuffer.array())
        }

        fun updateString(value: String) {
            val bytes = value.toByteArray(StandardCharsets.UTF_8)
            updateInt(bytes.size)
            digest.update(bytes)
        }

        fun updateNullableString(value: String?) {
            digest.update(if (value == null) 0.toByte() else 1.toByte())
            if (value != null) updateString(value)
        }

        updateString("ARGUS-CORPUS-REPLAY-SOURCE")
        updateInt(SOURCE_SNAPSHOT_FORMAT_VERSION)
        updateInt(stableSources.size)
        stableSources.forEach { source ->
            val document = source.document
            updateString(document.id)
            updateString(document.contentFingerprint)
            updateString(document.fingerprintAlgorithm)
            updateString(document.source)
            updateString(document.title)
            updateNullableString(document.language)
            updateNullableString(document.canonicalUrl)
            updateString(document.datasetSplit)
            updateString(document.rightsStatus)
            updateInt(document.plannedEpochs)
            updateInt(document.characterCount)
            updateLong(document.createdAtEpochMs)
            updateInt(source.chunks.size)
            source.chunks.forEach { chunk ->
                updateString(chunk.id)
                updateInt(chunk.chunkIndex)
                updateInt(chunk.characterCount)
                updateString(chunk.content)
            }
        }

        return digest.digest().joinToString(separator = "") { byte ->
            "%02x".format(java.util.Locale.ROOT, byte.toInt() and 0xFF)
        }
    }

    private fun canonicalSources(
        sources: List<CorpusReplaySource>
    ): List<CorpusReplaySource> {
        require(sources.isNotEmpty()) {
            "At least one training corpus document is required."
        }
        val stableSources = sources
            .sortedWith(
                compareBy<CorpusReplaySource>(
                    { it.document.createdAtEpochMs },
                    { it.document.id }
                )
            )
            .map { source ->
                source.copy(chunks = source.chunks.sortedBy { it.chunkIndex })
            }
        require(stableSources.map { it.document.id }.toSet().size == stableSources.size) {
            "A corpus document may appear only once in a replay plan."
        }
        stableSources.forEach { source ->
            require(source.document.datasetSplit == "TRAIN") {
                "Validation-only documents cannot enter a corpus replay plan."
            }
            require(source.document.validationText == null) {
                "Training documents cannot contain a validation text payload."
            }
            require(source.chunks.size == source.document.chunkCount && source.chunks.isNotEmpty()) {
                "Corpus document ${source.document.id} has missing or inconsistent chunks."
            }
            require(source.chunks.all { it.documentId == source.document.id }) {
                "Corpus document ${source.document.id} contains a foreign chunk reference."
            }
            require(source.chunks.map { it.chunkIndex } == (1..source.chunks.size).toList()) {
                "Corpus document ${source.document.id} has non-contiguous chunk indexes."
            }
        }
        return stableSources
    }

    private fun appendDocumentEpoch(
        result: MutableList<ScheduledCorpusChunk>,
        source: CorpusReplaySource,
        epochIndex: Int,
        epochs: Int
    ) {
        source.chunks.forEachIndexed { index, chunk ->
            appendChunk(
                result = result,
                source = source,
                chunk = chunk,
                chunkIndexWithinEpoch = index,
                epochIndex = epochIndex,
                epochs = epochs
            )
        }
    }

    private fun appendChunk(
        result: MutableList<ScheduledCorpusChunk>,
        source: CorpusReplaySource,
        chunk: CorpusChunkEntity,
        chunkIndexWithinEpoch: Int,
        epochIndex: Int,
        epochs: Int
    ) {
        val chunksPerDocument = source.document.chunkCount
        result += ScheduledCorpusChunk(
            document = source.document,
            chunk = chunk,
            epochNumber = epochIndex + 1,
            documentChunkIndex = epochIndex * chunksPerDocument + chunkIndexWithinEpoch + 1,
            documentTotalChunks = chunksPerDocument * epochs
        )
    }
}
