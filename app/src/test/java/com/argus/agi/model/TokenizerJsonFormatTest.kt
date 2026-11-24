package com.argus.agi.model

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TokenizerJsonFormatTest {

    @Test
    fun tokenizerJsonRoundTripPreservesVocabularyMergeRanksAndFingerprint() {
        val configuration = TokenizerConfig(
            maxVocabularySize = 1_024,
            maxTokenLength = 32
        )
        val original = Tokenizer(configuration)
        val corpus = List(96) {
            "Argus byte level tokenizer learns repeatable Bahasa Indonesia patterns."
        }
        original.trainBpe(
            corpus = corpus,
            targetVocabSize = 512,
            maxNewMerges = 224
        )

        val serialized = original.exportTokenizerJson()
        val root = JSONObject(serialized)
        assertEquals("argus-tokenizer-json", root.getString("format"))
        assertEquals("BPE", root.getJSONObject("model").getString("type"))

        val restored = Tokenizer(configuration)
        restored.loadTokenizerJson(serialized)

        assertEquals(original.exportVocab(), restored.exportVocab())
        assertEquals(original.exportMerges(), restored.exportMerges())
        assertEquals(original.snapshotFingerprint(), restored.snapshotFingerprint())
        val sample = "Bahasa Indonesia, byte-level BPE, whitespace  😀"
        assertEquals(
            original.encodeToIds(sample).toList(),
            restored.encodeToIds(sample).toList()
        )
        assertEquals(sample, restored.decode(restored.encodeToIds(sample)))
    }

    @Test
    fun tokenizerJsonImportRejectsTamperedFingerprintWithoutMutatingState() {
        val configuration = TokenizerConfig(
            maxVocabularySize = 512,
            maxTokenLength = 32
        )
        val source = Tokenizer(configuration)
        source.trainBpe(
            corpus = List(64) { "safe deterministic tokenizer artifact content" },
            targetVocabSize = 384,
            maxNewMerges = 96
        )
        val root = JSONObject(source.exportTokenizerJson())
        root.getJSONObject("argus").put("snapshot_fingerprint", "0".repeat(64))

        val target = Tokenizer(configuration)
        val before = target.snapshotFingerprint()
        val failure = runCatching {
            target.loadTokenizerJson(root.toString())
        }.exceptionOrNull()

        assertTrue(failure is IllegalArgumentException)
        assertEquals(before, target.snapshotFingerprint())
    }

    @Test
    fun bpeCanLearnMoreThanNinetySixMergesWhenCapacityAndCorpusAllowIt() {
        val configuration = TokenizerConfig(
            maxVocabularySize = 512,
            maxTokenLength = 32
        )
        val corpus = buildList {
            for (first in 0 until 16) {
                for (second in 0 until 16) {
                    for (third in 0 until 16) {
                        val word = buildString(3) {
                            append(('a'.code + first).toChar())
                            append(('a'.code + second).toChar())
                            append(('a'.code + third).toChar())
                        }
                        repeat(3) { add(word) }
                    }
                }
            }
        }

        val tokenizer = Tokenizer(configuration)
        val resultingVocabularySize = tokenizer.trainBpe(
            corpus = corpus,
            targetVocabSize = configuration.maxVocabularySize
        )

        assertTrue(
            "A full-capacity BPE pass must not retain the former 96-merge ceiling.",
            tokenizer.mergeCount > 96
        )
        assertTrue(resultingVocabularySize > 384)
        assertTrue(
            "Merge rules must remain within the maximum serializable tokenizer capacity.",
            tokenizer.mergeCount <= configuration.maxVocabularySize - TokenizerConfig.MINIMUM_VOCAB_SIZE
        )
    }
}
