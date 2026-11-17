package com.argus.agi.model

import java.nio.ByteBuffer
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class TokenizerByteLevelBpeTest {

    private fun replaceMalformedSurrogates(input: String): String {
        val result = StringBuilder(input.length)
        var index = 0
        while (index < input.length) {
            val current = input[index]
            if (
                current.isHighSurrogate() &&
                index + 1 < input.length &&
                input[index + 1].isLowSurrogate()
            ) {
                result.append(current).append(input[index + 1])
                index += 2
            } else if (current.code in 0xD800..0xDFFF) {
                result.append(0xFFFD.toChar())
                index++
            } else {
                result.append(current)
                index++
            }
        }
        return result.toString()
    }

    private fun newTokenizer(): Tokenizer = Tokenizer(
        TokenizerConfig(
            maxVocabularySize = 1_024,
            maxTokenLength = 32
        )
    )

    @Test
    fun byteLevelBpeV1IsTheOnlyCanonicalAlgorithm() {
        val tokenizer = Tokenizer()
        assertEquals(ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V1, tokenizer.config.algorithm)
        assertEquals(288, tokenizer.vocabSize)
        assertEquals(0, Tokenizer.PAD_ID)
        assertEquals(31, Tokenizer.MASK_ID)
        assertEquals(32, tokenizer.baseByteTokenId(0))
        assertEquals(287, tokenizer.baseByteTokenId(255))
    }

    @Test
    fun allByteIdsAreStableAndContiguous() {
        val tokenizer = newTokenizer()
        val byId = tokenizer.exportVocab().associate { (token, id) -> id to token }

        for (byteValue in 0..255) {
            val id = 32 + byteValue
            val symbol = requireNotNull(byId[id]) {
                "Missing byte token $byteValue at ID $id."
            }
            assertEquals(1, symbol.length)
            assertEquals(byteValue, symbol.first().code)
        }
    }

    @Test
    fun baseByteEncodingUsesCanonicalIds() {
        val tokenizer = newTokenizer()
        val vectors = listOf(
            "A" to intArrayOf(97),
            "hi" to intArrayOf(136, 137),
            " " to intArrayOf(64),
            "😀" to intArrayOf(272, 191, 184, 160)
        )

        for ((input, expected) in vectors) {
            assertArrayEquals(expected, tokenizer.encodeToIds(input))
        }
    }

    @Test
    fun encodeDecodePreservesUnicodeWhitespaceAndPunctuationExactly() {
        val tokenizer = newTokenizer()
        val samples = listOf(
            "  leading and trailing spaces  ",
            "\tTabs\r\nCRLF\rstandalone CR\n",
            "“curly quotes” — en dash – and … ellipsis",
            "Bahasa Indonesia: Argus belajar dari teks apa adanya.",
            "日本語の文章、한국어 문장، والنص العربي",
            "emoji 😀🧑🏽‍🚀 and combining e\u0301",
            "\u00A0non-breaking\u2003em-space\u3000ideographic"
        )

        for (sample in samples) {
            assertEquals(sample, tokenizer.decode(tokenizer.encodeToIds(sample)))
        }
    }

    @Test
    fun preTokenizerPrefixesWhitespaceAndSplitsPunctuationAndDigits() {
        val tokenizer = newTokenizer()
        assertEquals(
            listOf("  Hello", ",", "  world", "!", " 2", "0", "2", "6"),
            tokenizer.preTokenizeForTesting("  Hello,  world! 2026")
        )
        assertEquals(
            listOf("word", " \t"),
            tokenizer.preTokenizeForTesting("word \t")
        )
    }

    @Test
    fun specialControlTokensRoundTripAndUseFixedIds() {
        val tokenizer = newTokenizer()
        val sample = buildString {
            append(Tokenizer.SYSTEM_TOKEN)
            append(Tokenizer.USER_TOKEN)
            append("payload")
            append(Tokenizer.ASSISTANT_TOKEN)
            append(Tokenizer.THINK_START)
            append("reason")
            append(Tokenizer.THINK_END)
            append(Tokenizer.TOOL_CALL_TOKEN)
            append(Tokenizer.TOOL_RESULT_TOKEN)
            append(Tokenizer.WEB_RESULT_TOKEN)
            append(Tokenizer.IMAGE_TOKEN)
            append(Tokenizer.AUDIO_PERCEPTION_TOKEN)
            append(Tokenizer.EOS_TOKEN)
        }

        val encoded = tokenizer.encodeToIds(sample)
        assertTrue(encoded.contains(Tokenizer.SYSTEM_ID))
        assertTrue(encoded.contains(Tokenizer.USER_ID))
        assertTrue(encoded.contains(Tokenizer.ASSISTANT_ID))
        assertTrue(encoded.contains(Tokenizer.TOOL_CALL_ID))
        assertTrue(encoded.contains(Tokenizer.AUDIO_PERCEPTION_ID))
        assertEquals(sample, tokenizer.decode(encoded, keepControlTokens = true))
    }

    @Test
    fun escapedLiteralControlTextDoesNotBecomeControlIds() {
        val tokenizer = newTokenizer()
        val payload = "literal ${Tokenizer.USER_TOKEN} and ${Tokenizer.EOS_TOKEN}"
        val escaped = Tokenizer.escapeLiteralControlTokens(payload)
        val encoded = tokenizer.encodeToIds(escaped)

        assertFalse(encoded.contains(Tokenizer.USER_ID))
        assertFalse(encoded.contains(Tokenizer.EOS_ID))
        assertEquals(payload, tokenizer.decode(encoded))
    }

    @Test
    fun literalEscapeMarkerImmediatelyBeforeRealControlTokenIsUnambiguous() {
        val tokenizer = newTokenizer()
        val sample = "marker ${Tokenizer.LITERAL_CONTROL_ESCAPE_MARKER}${Tokenizer.EOS_TOKEN}"
        val encoded = tokenizer.encodeToIds(sample)

        assertTrue(encoded.contains(Tokenizer.EOS_ID))
        assertEquals(sample, tokenizer.decode(encoded, keepControlTokens = true))
    }

    @Test
    fun escapedMarkerAndControlTokenSequenceRoundTrips() {
        val tokenizer = newTokenizer()
        val payload = "${Tokenizer.LITERAL_CONTROL_ESCAPE_MARKER}${Tokenizer.USER_TOKEN} ${Tokenizer.LITERAL_CONTROL_ESCAPE_MARKER}"
        val escaped = Tokenizer.escapeLiteralControlTokens(payload)
        val encoded = tokenizer.encodeToIds(escaped)

        assertFalse(encoded.contains(Tokenizer.USER_ID))
        assertEquals(payload, tokenizer.decode(encoded))
    }

    @Test
    fun repeatedLiteralMarkersBeforeControlSpellingsRemainLiteralPayload() {
        val tokenizer = newTokenizer()
        val payload =
            "${Tokenizer.LITERAL_CONTROL_ESCAPE_MARKER}${Tokenizer.LITERAL_CONTROL_ESCAPE_MARKER}" +
                "${Tokenizer.USER_TOKEN}" +
                "${Tokenizer.LITERAL_CONTROL_ESCAPE_MARKER}${Tokenizer.LITERAL_CONTROL_ESCAPE_MARKER}" +
                "${Tokenizer.EOS_TOKEN}"
        val encoded = tokenizer.encodeToIds(
            Tokenizer.escapeLiteralControlTokens(payload)
        )

        assertFalse(encoded.contains(Tokenizer.USER_ID))
        assertFalse(encoded.contains(Tokenizer.EOS_ID))
        assertEquals(payload, tokenizer.decode(encoded))
    }

    @Test
    fun bpeTrainingIsDeterministicAndPreservesExistingIds() {
        val corpus = List(80) {
            "byte level tokenizer repeated deterministic pattern byte level tokenizer"
        }
        val sample = "byte level tokenizer repeated deterministic pattern"
        val baseline = newTokenizer()
        val originalVocabulary = baseline.exportVocab()
        val untrainedTokenCount = baseline.encodeToIds(sample).size

        val first = newTokenizer()
        val second = newTokenizer()
        first.trainBpe(corpus, targetVocabSize = 512, maxNewMerges = 64)
        second.trainBpe(corpus.reversed(), targetVocabSize = 512, maxNewMerges = 64)

        assertTrue(first.mergeCount > 0)
        assertTrue(
            first.exportMerges().none { (left, right) ->
                left + right in setOf(
                    Tokenizer.PAD_TOKEN,
                    Tokenizer.UNK_TOKEN,
                    Tokenizer.BOS_TOKEN,
                    Tokenizer.EOS_TOKEN,
                    Tokenizer.SYSTEM_TOKEN,
                    Tokenizer.USER_TOKEN,
                    Tokenizer.ASSISTANT_TOKEN,
                    Tokenizer.THINK_START,
                    Tokenizer.THINK_END,
                    Tokenizer.TOOL_CALL_TOKEN,
                    Tokenizer.TOOL_RESULT_TOKEN
                )
            }
        )
        assertEquals(first.exportVocab(), second.exportVocab())
        assertEquals(first.exportMerges(), second.exportMerges())
        assertEquals(originalVocabulary, first.exportVocab().take(originalVocabulary.size))
        assertTrue(first.encodeToIds(sample).size < untrainedTokenCount)
        assertEquals(sample, first.decode(first.encodeToIds(sample)))
        assertArrayEquals(first.encodeToIds(sample), second.encodeToIds(sample))
    }

    @Test
    fun trainingPublishesANewCacheGenerationWithoutStaleTokenIds() {
        val tokenizer = newTokenizer()
        val sample = "cached inference text repeats across local training"
        val before = tokenizer.encodeToIds(sample)
        assertTrue(before.size > 1)

        tokenizer.trainBpe(
            List(80) { sample },
            targetVocabSize = 512,
            maxNewMerges = 64
        )

        val after = tokenizer.encodeToIds(sample)
        assertTrue("New merge state must not reuse the previous encoded result.", after.size < before.size)
        assertEquals(sample, tokenizer.decode(after))
    }

    @Test
    fun importMergesRejectsPairsThatResolveToReservedControlTokens() {
        val tokenizer = newTokenizer()
        tokenizer.registerToken("USER>")
        val original = tokenizer.exportMerges()
        val failure = runCatching {
            tokenizer.importMerges(listOf("<" to "USER>"))
        }.exceptionOrNull()

        assertTrue(failure is IllegalArgumentException)
        assertEquals(original, tokenizer.exportMerges())
    }

    @Test
    fun trainingAndInferenceUseTheSameSpecialAndTextBoundaries() {
        val tokenizer = newTokenizer()
        val sample = "  leading\tspaces\r\nand${Tokenizer.USER_TOKEN}adjacent marker  "
        tokenizer.trainBpe(List(64) { sample }, targetVocabSize = 512, maxNewMerges = 48)

        assertTrue(tokenizer.mergeCount > 0)
        assertEquals(
            sample,
            tokenizer.decode(
                tokenizer.encodeToIds(sample),
                keepControlTokens = true
            )
        )
    }

    @Test
    fun snapshotRoundTripPreservesIdsMergesFingerprintAndEncoding() {
        val tokenizer = newTokenizer()
        val corpus = List(64) {
            "snapshot fingerprint deterministic tokenizer byte sequence snapshot fingerprint"
        }
        tokenizer.trainBpe(corpus, targetVocabSize = 512, maxNewMerges = 48)

        val sample = "snapshot fingerprint 😀 preserves exact text"
        val encodedBefore = tokenizer.encodeToIds(sample)
        val fingerprintBefore = tokenizer.snapshotFingerprint()
        val snapshot = tokenizer.saveSnapshot()

        assertEquals(
            Tokenizer.CURRENT_SNAPSHOT_FORMAT_VERSION,
            ByteBuffer.wrap(snapshot).getInt(4)
        )

        val restored = newTokenizer()
        restored.loadSnapshot(snapshot)

        assertEquals(tokenizer.exportVocab(), restored.exportVocab())
        assertEquals(tokenizer.exportMerges(), restored.exportMerges())
        assertEquals(fingerprintBefore, restored.snapshotFingerprint())
        assertArrayEquals(encodedBefore, restored.encodeToIds(sample))
        assertArrayEquals(snapshot, restored.saveSnapshot())
    }

    @Test
    fun corruptSnapshotIsRejectedWithoutChangingActiveState() {
        val tokenizer = newTokenizer()
        tokenizer.trainBpe(
            List(48) { "atomic snapshot validation preserves current state" },
            targetVocabSize = 512,
            maxNewMerges = 32
        )
        val fingerprintBefore = tokenizer.snapshotFingerprint()
        val corrupted = tokenizer.saveSnapshot()
        corrupted[corrupted.lastIndex - 4] =
            (corrupted[corrupted.lastIndex - 4].toInt() xor 0x01).toByte()

        val failure = runCatching { tokenizer.loadSnapshot(corrupted) }.exceptionOrNull()
        assertTrue(failure is IllegalArgumentException)
        assertEquals(fingerprintBefore, tokenizer.snapshotFingerprint())
    }

    @Test
    fun malformedUtf16IsReplacedInsteadOfCrashingInference() {
        val tokenizer = newTokenizer()
        val malformed = "prefix" + 0xD800.toChar() + "middle" + 0xDC00.toChar() + "😀"

        assertEquals(replaceMalformedSurrogates(malformed), tokenizer.decode(tokenizer.encodeToIds(malformed)))
    }

    @Test
    fun malformedGeneratedUtf8DecodesWithReplacementCharacter() {
        val tokenizer = newTokenizer()
        assertEquals("\uFFFD", tokenizer.decode(intArrayOf(tokenizer.baseByteTokenId(0xFF))))
    }

    @Test
    fun deterministicUnicodeFuzzRoundTripsAcrossSpecialAndEscapeBoundaries() {
        val tokenizer = newTokenizer()
        val samples = listOf(
            "",
            "\u0000",
            "\uE000",
            "\uE000${Tokenizer.EOS_TOKEN}",
            "${Tokenizer.USER_TOKEN}",
            "x${Tokenizer.USER_TOKEN}y",
            "x\uE000\uE000y",
            "a\uD800b",
            "a\uDC00b",
            "👩🏽‍💻🇮🇩",
            "e\u0301\u200Dक्‍ष",
            "  \t\r\n  ",
            "punctuation !?,.;:()[]{}",
            "digits 0123456789"
        )

        for (sample in samples) {
            assertEquals(
                "Failed round trip for input length ${sample.length}.",
                replaceMalformedSurrogates(sample),
                tokenizer.decode(
                    tokenizer.encodeToIds(sample),
                    keepControlTokens = true
                )
            )
        }
    }
}
