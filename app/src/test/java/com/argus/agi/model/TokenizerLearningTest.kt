package com.argus.agi.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TokenizerLearningTest {

    @Test
    fun canonicalTokenizerLearnsOnlyAppendOnlyByteLevelMerges() {
        val tokenizer = Tokenizer(
            TokenizerConfig(
                maxVocabularySize = 512,
                maxTokenLength = 32
            )
        )
        val originalVocabulary = tokenizer.exportVocab()
        val corpus = List(64) {
            "Argus learns repeated local language patterns from its own training corpus."
        }

        val resultingSize = tokenizer.trainBpe(
            corpus = corpus,
            targetVocabSize = 384,
            maxNewMerges = 64
        )

        assertTrue(resultingSize > originalVocabulary.size)
        assertTrue(tokenizer.mergeCount > 0)
        assertEquals(
            originalVocabulary,
            tokenizer.exportVocab().take(originalVocabulary.size)
        )
        assertTrue(
            tokenizer.exportMerges().all { (left, right) ->
                left !in setOf(
                    Tokenizer.PAD_TOKEN,
                    Tokenizer.UNK_TOKEN,
                    Tokenizer.BOS_TOKEN,
                    Tokenizer.EOS_TOKEN,
                    Tokenizer.USER_TOKEN,
                    Tokenizer.ASSISTANT_TOKEN
                ) && right !in setOf(
                    Tokenizer.PAD_TOKEN,
                    Tokenizer.UNK_TOKEN,
                    Tokenizer.BOS_TOKEN,
                    Tokenizer.EOS_TOKEN,
                    Tokenizer.USER_TOKEN,
                    Tokenizer.ASSISTANT_TOKEN
                ) && left + right !in setOf(
                    Tokenizer.PAD_TOKEN,
                    Tokenizer.UNK_TOKEN,
                    Tokenizer.BOS_TOKEN,
                    Tokenizer.EOS_TOKEN,
                    Tokenizer.USER_TOKEN,
                    Tokenizer.ASSISTANT_TOKEN
                )
            }
        )

        val sample = corpus.first()
        assertEquals(sample, tokenizer.decode(tokenizer.encodeToIds(sample)))
        assertTrue(tokenizer.encodeToIds(sample).all { id -> id in 0 until tokenizer.vocabSize })
    }

    @Test
    fun incrementalTrainingKeepsPreviouslyLearnedMergeRanks() {
        val tokenizer = Tokenizer(
            TokenizerConfig(maxVocabularySize = 512, maxTokenLength = 32)
        )
        val corpus = List(48) {
            "incremental tokenizer learning retains old deterministic merge order"
        }
        tokenizer.trainBpe(corpus, targetVocabSize = 360, maxNewMerges = 32)
        val originalMerges = tokenizer.exportMerges()

        tokenizer.trainBpe(corpus, targetVocabSize = 400, maxNewMerges = 32)

        assertTrue(tokenizer.exportMerges().size >= originalMerges.size)
        assertEquals(
            originalMerges,
            tokenizer.exportMerges().take(originalMerges.size)
        )
    }

    @Test
    fun defaultConfigurationIsTheSingleSupportedAlgorithmAndSchema() {
        val config = TokenizerConfig()
        assertEquals(ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V1, config.algorithm)
        assertEquals(2, config.schemaVersion)
        assertEquals(2_048, config.maxVocabularySize)
        assertEquals(32, config.maxTokenLength)
        assertEquals(288, Tokenizer(config).vocabSize)
    }
}
