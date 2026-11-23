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
                val text = "x".repeat(characterCount)
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
                    maxCharacters = 4_096
                )

                assertNotNull(
                    "The next bounded corpus projection must be available.",
                    sample
                )
                val current = requireNotNull(sample)
                assertEquals(expectedIndex + 1, current.chunkIndex)
                assertEquals(4_096, current.content.length)
                assertTrue(current.content.all { character -> character == 'x' })

                cursor = current.chunkIndex
                sampledCharacters += current.content.length
            }

            assertEquals(21 * 4_096, sampledCharacters)
            assertNull(
                "The cursor must stop after the final corpus row.",
                repository.getNextTokenizerCorpusSample(
                    documentId = documentId,
                    afterChunkIndex = cursor,
                    maxCharacters = 4_096
                )
            )
        } finally {
            database.close()
        }
    }
}
