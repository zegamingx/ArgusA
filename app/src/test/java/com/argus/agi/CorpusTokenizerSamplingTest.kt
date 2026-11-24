package com.argus.agi

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.argus.agi.data.local.ArgusDatabase
import com.argus.agi.data.local.entity.CorpusChunkEntity
import com.argus.agi.data.local.entity.CorpusDocumentEntity
import com.argus.agi.data.repository.ArgusRepository
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class CorpusTokenizerSamplingTest {

    @Test
    fun tokenizerSamplingPagesLargeRowsWithoutOverflowingCursorWindow() = runTest {
        val context: Context = ApplicationProvider.getApplicationContext()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            ArgusDatabase::class.java
        ).allowMainThreadQueries().build()

        try {
            val documentId = "cursor-window-regression"
            val document = CorpusDocumentEntity(
                id = documentId,
                contentFingerprint = "cursor-window-regression-fingerprint",
                fingerprintAlgorithm = "SHA-256",
                source = "Tokenizer regression test",
                title = "Large corpus rows",
                language = "en",
                canonicalUrl = null,
                characterCount = (20 * 128 * 1024) + (3 * 1024 * 1024),
                chunkCount = 21,
                datasetSplit = "TRAIN",
                validationText = null,
                plannedEpochs = 1,
                rightsStatus = "UNVERIFIED",
                status = "READY",
                createdAtEpochMs = 1L,
                lastTrainedAtEpochMs = null,
                completedPassCount = 0
            )
            assertTrue(database.corpusDao().insertDocument(document) > 0L)

            val chunks = (1..21).map { index ->
                val characterCount = if (index == 21) {
                    3 * 1024 * 1024
                } else {
                    128 * 1024
                }
                val text = if (index == 21) {
                    "😀".repeat(characterCount / 2)
                } else {
                    "x".repeat(characterCount)
                }
                CorpusChunkEntity(
                    id = "$documentId:$index",
                    documentId = documentId,
                    chunkIndex = index,
                    content = text,
                    characterCount = text.length
                )
            }
            database.corpusDao().insertChunks(chunks)

            val repository = ArgusRepository(context, database)
            var cursor = 0
            var sampledCharacters = 0

            repeat(chunks.size) { expectedIndex ->
                val sample = repository.getNextTokenizerCorpusSample(
                    documentId = documentId,
                    afterChunkIndex = cursor,
                    maxCharacters = 8_192
                )

                assertNotNull(
                    "The next bounded corpus projection must be available.",
                    sample
                )
                val current = requireNotNull(sample)
                assertEquals(expectedIndex + 1, current.chunkIndex)
                assertEquals(8_192, current.content.length)
                if (expectedIndex + 1 == chunks.size) {
                    // The final row holds supplementary characters: 8,192
                    // UTF-16 code units are 4,096 complete emoji, never a
                    // split surrogate pair and never 8,192 code points
                    // (which would be 16,384 UTF-16 code units).
                    assertEquals("😀".repeat(4_096), current.content)
                    assertEquals(
                        4_096,
                        current.content.codePointCount(0, current.content.length)
                    )
                    assertTrue(current.content.last().isLowSurrogate())
                } else {
                    assertTrue(current.content.all { character -> character == 'x' })
                }

                cursor = current.chunkIndex
                sampledCharacters += current.content.length
            }

            assertEquals(21 * 8_192, sampledCharacters)

            val reconstructedLargeChunk = repository.getCorpusChunk(documentId, 21)
            assertNotNull(
                "A legacy multi-megabyte row must be reconstructed through bounded SQLite reads.",
                reconstructedLargeChunk
            )
            val reconstructedText = requireNotNull(reconstructedLargeChunk).content
            val expectedUnicodeText = "😀".repeat(1_536 * 1_024)
            assertEquals(3 * 1024 * 1024, reconstructedText.length)
            assertEquals(expectedUnicodeText, reconstructedText)
            assertEquals(1_536 * 1_024, reconstructedText.codePointCount(0, reconstructedText.length))
            assertTrue(reconstructedText.first().isHighSurrogate())
            assertTrue(reconstructedText.last().isLowSurrogate())

            val replayChunks = repository.getCorpusChunks(documentId)
            assertEquals(21, replayChunks.size)
            assertEquals(expectedUnicodeText, replayChunks.last().content)
            assertEquals(
                1_536 * 1_024,
                replayChunks.last().content.codePointCount(0, replayChunks.last().content.length)
            )

            assertNull(
                "The cursor must stop after the final corpus row.",
                repository.getNextTokenizerCorpusSample(
                    documentId = documentId,
                    afterChunkIndex = cursor,
                    maxCharacters = 8_192
                )
            )
        } finally {
            database.close()
        }
    }

    @Test
    fun tokenizerSamplesAreDistributedFromFirstToLastChunk() = runTest {
        val context: Context = ApplicationProvider.getApplicationContext()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            ArgusDatabase::class.java
        ).allowMainThreadQueries().build()

        try {
            val documentId = "stratified-tokenizer-sampling"
            val chunkCount = 21
            val document = CorpusDocumentEntity(
                id = documentId,
                contentFingerprint = "stratified-tokenizer-sampling-fingerprint",
                fingerprintAlgorithm = "SHA-256",
                source = "Tokenizer sampling regression test",
                title = "Distributed chunks",
                language = "en",
                canonicalUrl = null,
                characterCount = chunkCount * 128,
                chunkCount = chunkCount,
                datasetSplit = "TRAIN",
                validationText = null,
                plannedEpochs = 1,
                rightsStatus = "UNVERIFIED",
                status = "READY",
                createdAtEpochMs = 2L,
                lastTrainedAtEpochMs = null,
                completedPassCount = 0
            )
            assertTrue(database.corpusDao().insertDocument(document) > 0L)
            database.corpusDao().insertChunks(
                (1..chunkCount).map { index ->
                    val content = "chunk-$index-" + "x".repeat(100)
                    CorpusChunkEntity(
                        id = "$documentId:$index",
                        documentId = documentId,
                        chunkIndex = index,
                        content = content,
                        characterCount = content.length
                    )
                }
            )

            val result = ArgusRepository(context, database).getTokenizerCorpusSamples(
                documentId = documentId,
                maxSampleChunks = 5,
                maxCharactersPerChunk = 64
            )

            assertEquals(chunkCount, result.totalChunkCount)
            assertEquals(listOf(1, 6, 11, 16, 21), result.samples.map { it.chunkIndex })
            assertTrue(result.samples.all { sample -> sample.content.length <= 64 })
            assertTrue(result.samples.first().content.startsWith("chunk-1-"))
            assertTrue(result.samples.last().content.startsWith("chunk-21-"))
        } finally {
            database.close()
        }
    }

}
