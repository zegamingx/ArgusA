package com.argus.agi.model

import java.io.Reader
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TokenizerStreamingBoundaryTest {

    private class FragmentingReader(
        private val value: String,
        private val maximumReadSize: Int
    ) : Reader() {
        private var offset = 0

        override fun read(target: CharArray, targetOffset: Int, length: Int): Int {
            require(targetOffset >= 0 && length >= 0 && targetOffset + length <= target.size)
            if (length == 0) return 0
            if (offset >= value.length) return -1

            val count = minOf(length, maximumReadSize, value.length - offset)
            value.toCharArray(target, targetOffset, offset, offset + count)
            offset += count
            return count
        }

        override fun close() = Unit
    }

    private fun tokenizer(): Tokenizer = Tokenizer(
        TokenizerConfig(maxVocabularySize = 1_024, maxTokenLength = 32)
    )

    private fun flatten(chunks: List<TokenizerTextChunk>): IntArray =
        chunks.flatMap { chunk -> chunk.tokenIds.asIterable() }.toIntArray()

    @Test
    fun streamingEncodingMatchesWholeDocumentEncodingForArbitraryReaderFragments() {
        val tokenizer = tokenizer()
        val marker = Tokenizer.LITERAL_CONTROL_ESCAPE_MARKER
        val samples = listOf(
            "  leading spaces\tand trailing spaces  ",
            "Indonesia 2026: angka 12345, punctuation?! newline\nnext line",
            "Multilingual: 日本語 한국어 العربية हिन्दी emoji 😀🧑🏽‍🚀",
            "<SYSTEM>policy<USER>question<ASSISTANT>answer<EOS>",
            Tokenizer.escapeLiteralControlTokens(
                "Literal \${Tokenizer.USER_TOKEN}, \${Tokenizer.EOS_TOKEN}, and \${Tokenizer.IMAGE_TOKEN}"
            ),
            (1..5).joinToString("|") { count ->
                "prefix\${marker.toString().repeat(count)}\${Tokenizer.USER_TOKEN}suffix"
            },
            "combining e\u0301 and malformed \uD800 surrogate \uDC00"
        )

        for (sample in samples) {
            for (fragmentSize in listOf(1, 2, 3, 7, 31, 127)) {
                val chunks = ArrayList<TokenizerTextChunk>()
                tokenizer.forEachSafeTextChunk(
                    reader = FragmentingReader(sample, fragmentSize),
                    maxTokensPerChunk = 11,
                    overlapTokens = 0,
                    action = chunks::add
                )

                assertEquals(sample, chunks.joinToString(separator = "") { it.text })
                assertArrayEquals(tokenizer.encodeToIds(sample), flatten(chunks))
                for (chunk in chunks) {
                    assertArrayEquals(tokenizer.encodeToIds(chunk.text), chunk.tokenIds)
                }
            }
        }
    }

    @Test
    fun safeChunksNeverSplitAnIndivisiblePreToken() {
        val tokenizer = tokenizer()
        val sample = "thisisoneverylongpretokenwithoutanysafeinnerboundary"
        val chunks = ArrayList<TokenizerTextChunk>()

        tokenizer.forEachSafeTextChunk(
            reader = FragmentingReader(sample, 3),
            maxTokensPerChunk = 2,
            action = chunks::add
        )

        assertEquals(1, chunks.size)
        assertEquals(sample, chunks.single().text)
        assertArrayEquals(tokenizer.encodeToIds(sample), chunks.single().tokenIds)
        assertTrue(chunks.single().tokenIds.size > 2)
    }

    @Test
    fun overlapIsAppliedOnlyAtSafePreTokenBoundaries() {
        val tokenizer = tokenizer()
        val sample = "one two three four five six seven eight nine ten"
        val chunks = ArrayList<TokenizerTextChunk>()

        tokenizer.forEachSafeTextChunk(
            reader = FragmentingReader(sample, 5),
            maxTokensPerChunk = 8,
            overlapTokens = 2,
            action = chunks::add
        )

        assertTrue(chunks.size > 1)
        for (chunk in chunks) {
            assertArrayEquals(tokenizer.encodeToIds(chunk.text), chunk.tokenIds)
        }
    }

    @Test
    fun tokenizerSnapshotIsStableForTheDurationOfAStreamingPass() {
        val tokenizer = tokenizer()
        val sample = "repeatable byte-level BPE text across streaming fragments"
        val expected = tokenizer.encodeToIds(sample)
        val chunks = ArrayList<TokenizerTextChunk>()

        tokenizer.forEachSafeTextChunk(
            reader = FragmentingReader(sample, 4),
            maxTokensPerChunk = 6,
            action = chunks::add
        )

        tokenizer.trainBpe(
            corpus = List(32) { "repeatable byte-level BPE text across streaming fragments" },
            targetVocabSize = 384,
            maxNewMerges = 32
        )

        assertArrayEquals(expected, flatten(chunks))
    }
}
