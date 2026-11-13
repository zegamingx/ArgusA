package com.argus.agi.model

import java.nio.ByteBuffer
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TokenizerByteLevelBpeTest {

    private fun newTokenizer(): Tokenizer = Tokenizer(
        TokenizerConfig(
            maxVocabularySize = 1_024,
            maxTokenLength = 16
        )
    )

    @Test
    fun byteLevelBpeV2IsTheDefaultAlgorithm() {
        val tokenizer = Tokenizer()
        assertEquals(
            ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V2,
            tokenizer.config.algorithm
        )
        assertEquals(Tokenizer.CURRENT_SNAPSHOT_FORMAT_VERSION, 1)
        assertEquals(265, tokenizer.vocabSize)
    }

    @Test
    fun baseVocabularyContainsEveryByteWithStableCanonicalIds() {
        val tokenizer = newTokenizer()
        val tokensById = tokenizer.exportVocab().associate { entry -> entry.second to entry.first }

        for (byteValue in 0..255) {
            val expectedId = 9 + byteValue
            assertEquals(expectedId, tokenizer.baseByteTokenId(byteValue))
            val symbol = tokensById[expectedId]
            val resolvedSymbol = requireNotNull(symbol) {
                "Missing base byte token $byteValue at ID $expectedId."
            }
            assertEquals(1, resolvedSymbol.length)
            assertEquals(byteValue, resolvedSymbol[0].code)
        }
    }

    @Test
    fun baseByteEncodingMatchesStableKnownTestVectors() {
        val tokenizer = newTokenizer()
        val vectors = listOf(
            "A" to listOf(74),
            "hi" to listOf(113, 114),
            " " to listOf(41),
            "😀" to listOf(249, 168, 161, 137)
        )

        for ((input, expectedIds) in vectors) {
            assertEquals(expectedIds, tokenizer.encode(input))
        }
    }

    @Test
    fun encodeDecodePreservesExactUnicodeAndWhitespaceWithoutNormalization() {
        val tokenizer = newTokenizer()
        val samples = listOf(
            "  leading and trailing spaces  ",
            "\tTabs\r\nCRLF\rstandalone CR\n",
            "“curly quotes” — en dash – and … ellipsis",
            "Bahasa Indonesia: Argus belajar dari teks apa adanya.",
            "日本語の文章、한국어 문장، والنص العربي",
            "emoji 😀🧑🏽‍🚀 and combining e\u0301"
        )

        for (sample in samples) {
            val encoded = tokenizer.encode(sample)
            assertEquals(sample, tokenizer.decode(encoded))
        }
    }

    @Test
    fun explicitControlMarkersRoundTripWhenRequested() {
        val tokenizer = newTokenizer()
        val sample =
            "prefix<USER>payload<THINK>reason</THINK><ASSISTANT>answer<EOS>suffix"

        val encoded = tokenizer.encode(sample)
        assertEquals(
            sample,
            tokenizer.decode(encoded, keepControlTokens = true)
        )
    }

    @Test
    fun byteLevelTrainingIsDeterministicAndPreservesExistingIds() {
        val corpus = List(80) {
            "byte level tokenizer repeated deterministic pattern byte level tokenizer"
        }
        val sample = "byte level tokenizer repeated deterministic pattern"
        val baseline = newTokenizer()
        val originalVocabulary = baseline.exportVocab()
        val untrainedTokenCount = baseline.encode(sample).size

        val first = newTokenizer()
        val second = newTokenizer()
        first.trainBpe(corpus, targetVocabSize = 512, maxNewMerges = 64)
        second.trainBpe(corpus, targetVocabSize = 512, maxNewMerges = 64)

        assertTrue("Training should learn at least one byte merge.", first.mergeCount > 0)
        assertEquals(first.exportVocab(), second.exportVocab())
        assertEquals(first.exportMerges(), second.exportMerges())
        assertEquals(
            "New vocabulary entries must be appended without changing existing IDs.",
            originalVocabulary,
            first.exportVocab().take(originalVocabulary.size)
        )
        assertTrue(
            "Learned merges should reduce token count for the repeated training text.",
            first.encode(sample).size < untrainedTokenCount
        )
        assertEquals(sample, first.decode(first.encode(sample)))
        assertEquals(first.encode(sample), second.encode(sample))
    }

    @Test
    fun snapshotRoundTripPreservesVocabularyMergesFingerprintAndEncoding() {
        val tokenizer = newTokenizer()
        val corpus = List(64) {
            "snapshot fingerprint deterministic tokenizer byte sequence snapshot fingerprint"
        }
        tokenizer.trainBpe(corpus, targetVocabSize = 512, maxNewMerges = 48)

        val sample = "snapshot fingerprint 😀 preserves exact text"
        val encodedBefore = tokenizer.encode(sample)
        val vocabularyBefore = tokenizer.exportVocab()
        val mergesBefore = tokenizer.exportMerges()
        val fingerprintBefore = tokenizer.snapshotFingerprint()
        val snapshot = tokenizer.saveSnapshot()

        assertEquals(
            Tokenizer.CURRENT_SNAPSHOT_FORMAT_VERSION,
            ByteBuffer.wrap(snapshot).getInt(4)
        )

        val restored = newTokenizer()
        restored.loadSnapshot(snapshot)

        assertEquals(vocabularyBefore, restored.exportVocab())
        assertEquals(mergesBefore, restored.exportMerges())
        assertEquals(fingerprintBefore, restored.snapshotFingerprint())
        assertEquals(encodedBefore, restored.encode(sample))
        assertArrayEquals(snapshot, restored.saveSnapshot())
    }

    @Test
    fun corruptedSnapshotIsRejectedWithoutChangingActiveTokenizerState() {
        val tokenizer = newTokenizer()
        tokenizer.trainBpe(
            List(48) { "atomic snapshot validation must preserve current state" },
            targetVocabSize = 512,
            maxNewMerges = 32
        )
        val fingerprintBefore = tokenizer.snapshotFingerprint()
        val corrupted = tokenizer.saveSnapshot()
        corrupted[corrupted.lastIndex - 4] =
            (corrupted[corrupted.lastIndex - 4].toInt() xor 0x01).toByte()

        val failure = runCatching { tokenizer.loadSnapshot(corrupted) }.exceptionOrNull()
        assertTrue("Corrupted snapshot must fail SHA-256 verification.", failure is IllegalArgumentException)
        assertEquals(fingerprintBefore, tokenizer.snapshotFingerprint())
    }

    @Test
    fun invalidMergeImportIsAtomic() {
        val tokenizer = newTokenizer()
        tokenizer.trainBpe(
            List(48) { "atomic merge import validation must preserve existing merge rules" },
            targetVocabSize = 512,
            maxNewMerges = 32
        )
        val existingMerges = tokenizer.exportMerges()
        assertTrue(existingMerges.isNotEmpty())

        val invalidMerges = existingMerges + ("not-a-token" to "not-a-token-either")
        val failure = runCatching { tokenizer.importMerges(invalidMerges) }.exceptionOrNull()

        assertTrue("Invalid merge table must be rejected.", failure is IllegalArgumentException)
        assertEquals(existingMerges, tokenizer.exportMerges())
    }

    @Test
    fun byteLevelEncodingDoesNotGrowVocabularyEvenWhenAutoRegisterIsRequested() {
        val tokenizer = newTokenizer()
        val vocabularySizeBefore = tokenizer.vocabSize
        val sample = "unseen multilingual text 🧪"
        val encoded = tokenizer.encode(sample, autoRegister = true)

        assertTrue(encoded.isNotEmpty())
        assertEquals(vocabularySizeBefore, tokenizer.vocabSize)
        assertEquals(sample, tokenizer.decode(encoded))
    }

    @Test
    fun malformedUtf16IsRejectedInsteadOfSilentlyReplacingInput() {
        val tokenizer = newTokenizer()
        val malformed = "valid prefix followed by an unpaired high surrogate: " + 0xD800.toChar()

        val failure = runCatching { tokenizer.encode(malformed) }.exceptionOrNull()
        assertTrue("Malformed UTF-16 must be rejected.", failure is IllegalArgumentException)
    }
}
