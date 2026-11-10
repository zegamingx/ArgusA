package com.argus.agi.model

import com.argus.agi.data.local.entity.CorpusChunkEntity
import com.argus.agi.data.local.entity.CorpusDocumentEntity
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

        val stableSources = sources.sortedWith(
            compareBy<CorpusReplaySource>(
                { it.document.createdAtEpochMs },
                { it.document.id }
            )
        )
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

            val orderedChunks = source.chunks.sortedBy { it.chunkIndex }
            require(orderedChunks.all { it.documentId == source.document.id }) {
                "Corpus document ${source.document.id} contains a foreign chunk reference."
            }
            require(orderedChunks.map { it.chunkIndex } == (1..orderedChunks.size).toList()) {
                "Corpus document ${source.document.id} has non-contiguous chunk indexes."
            }
        }

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
                        val orderedChunks = cursor.source.chunks.sortedBy { it.chunkIndex }
                        appendChunk(
                            result = result,
                            source = cursor.source,
                            chunk = orderedChunks[cursor.nextChunkIndex],
                            chunkIndexWithinEpoch = cursor.nextChunkIndex,
                            epochIndex = epochIndex,
                            epochs = epochs
                        )

                        cursor.nextChunkIndex += 1
                        if (cursor.nextChunkIndex >= orderedChunks.size) {
                            active.removeAt(selectedIndex)
                        }
                    }
                }
            }
        }

        return result
    }

    private fun appendDocumentEpoch(
        result: MutableList<ScheduledCorpusChunk>,
        source: CorpusReplaySource,
        epochIndex: Int,
        epochs: Int
    ) {
        source.chunks.sortedBy { it.chunkIndex }.forEachIndexed { index, chunk ->
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
