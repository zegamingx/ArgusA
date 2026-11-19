package com.argus.agi.model

import com.argus.agi.data.local.entity.CorpusChunkEntity
import com.argus.agi.data.local.entity.CorpusDocumentEntity
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Test

class CorpusReplaySchedulerTest {

    @Test
    fun sourceSnapshotFingerprintIsStableAcrossInputOrder() {
        val sources = listOf(
            source(
                documentId = "document-b",
                createdAt = 200L,
                chunks = listOf(
                    chunk("b-2", "document-b", 2, "second chunk"),
                    chunk("b-1", "document-b", 1, "first chunk")
                )
            ),
            source(
                documentId = "document-a",
                createdAt = 100L,
                chunks = listOf(chunk("a-1", "document-a", 1, "article A"))
            )
        )
        val reordered = sources.reversed().map { source ->
            source.copy(chunks = source.chunks.reversed())
        }

        assertEquals(
            CorpusReplayScheduler.sourceSnapshotFingerprint(sources),
            CorpusReplayScheduler.sourceSnapshotFingerprint(reordered)
        )
    }

    @Test
    fun sourceSnapshotFingerprintChangesWhenChunkContentChanges() {
        val source = source(
            documentId = "document-a",
            createdAt = 100L,
            chunks = listOf(chunk("a-1", "document-a", 1, "original content"))
        )
        val changed = source.copy(
            chunks = listOf(source.chunks.single().copy(content = "modified content"))
        )

        assertNotEquals(
            CorpusReplayScheduler.sourceSnapshotFingerprint(listOf(source)),
            CorpusReplayScheduler.sourceSnapshotFingerprint(listOf(changed))
        )
    }

    @Test
    fun sourceSnapshotFingerprintChangesWhenRightsProvenanceChanges() {
        val original = source(
            documentId = "document-a",
            createdAt = 100L,
            chunks = listOf(chunk("a-1", "document-a", 1, "same content"))
        )
        val changed = original.copy(
            document = original.document.copy(rightsStatus = "LICENSE_VERIFIED")
        )

        assertNotEquals(
            CorpusReplayScheduler.sourceSnapshotFingerprint(listOf(original)),
            CorpusReplayScheduler.sourceSnapshotFingerprint(listOf(changed))
        )
    }

    private fun source(
        documentId: String,
        createdAt: Long,
        chunks: List<CorpusChunkEntity>
    ): CorpusReplaySource = CorpusReplaySource(
        document = CorpusDocumentEntity(
            id = documentId,
            contentFingerprint = if (documentId.endsWith("a")) "a".repeat(64) else "b".repeat(64),
            fingerprintAlgorithm = "SHA-256",
            source = "local-import",
            title = documentId,
            language = "id",
            canonicalUrl = null,
            characterCount = chunks.sumOf(CorpusChunkEntity::characterCount),
            chunkCount = chunks.size,
            datasetSplit = "TRAIN",
            validationText = null,
            plannedEpochs = 1,
            rightsStatus = "USER_DECLARED",
            status = "IMPORTED",
            createdAtEpochMs = createdAt,
            lastTrainedAtEpochMs = null,
            completedPassCount = 0
        ),
        chunks = chunks
    )

    private fun chunk(
        id: String,
        documentId: String,
        index: Int,
        content: String
    ): CorpusChunkEntity = CorpusChunkEntity(
        id = id,
        documentId = documentId,
        chunkIndex = index,
        content = content,
        characterCount = content.length
    )
}
