package com.argus.agi.model

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class TokenizerLearningTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun freshTokenizerLearnsBpeMergesFromCorpusAndPreservesExistingIds() {
        val tokenizer = Tokenizer(
            TokenizerConfig(
                maxVocabularySize = 512,
                maxTokenLength = 16
            )
        )
        assertEquals(ArgusTokenizerAlgorithm.LEARNED_BPE_V1, tokenizer.config.algorithm)
        assertEquals(2_048, TokenizerConfig.DEFAULT_MAX_VOCAB_SIZE)

        val initialVocabulary = tokenizer.exportVocab()
        assertTrue(
            "The bundled seed vocabulary must leave room for BPE growth.",
            initialVocabulary.size < 256
        )
        assertTrue(
            "Initial seed merges must fit the configured checkpoint merge limit.",
            tokenizer.mergeCount <= tokenizer.config.maxVocabularySize
        )
        assertTrue(
            "Control-token markup must not be registered as seed merges.",
            tokenizer.exportMerges().none { merge ->
                merge.first.contains('<') || merge.first.contains('>') ||
                    merge.second.contains('<') || merge.second.contains('>')
            }
        )
        val knownTokenId = initialVocabulary.single { it.first == "yang" }.second
        assertEquals(
            "Bundled seed words must use their registered vocabulary IDs under BPE.",
            listOf(knownTokenId),
            tokenizer.encode("yang")
        )
        val corpus = List(40) {
            "Argus learns artificial language patterns from repeated local training documents."
        }

        val resultingSize = tokenizer.trainBpe(
            corpus = corpus,
            targetVocabSize = 256,
            maxNewMerges = 64
        )

        assertTrue("BPE must add corpus-derived tokens.", resultingSize > initialVocabulary.size)
        assertTrue("BPE must retain merge rules for deterministic encoding.", tokenizer.mergeCount > 0)
        assertEquals(
            knownTokenId,
            tokenizer.exportVocab().single { it.first == "yang" }.second
        )

        val encoded = tokenizer.encode(corpus.first(), autoRegister = false)
        assertTrue(encoded.isNotEmpty())
        assertTrue(encoded.all { tokenId -> tokenId in 0 until tokenizer.vocabSize })

        val priorMerges = tokenizer.exportMerges()
        tokenizer.trainBpe(
            corpus = corpus,
            targetVocabSize = 512,
            maxNewMerges = 64
        )
        assertEquals(
            "Incremental BPE learning must preserve the priority of existing merges.",
            priorMerges,
            tokenizer.exportMerges().take(priorMerges.size)
        )
    }

    @Test
    fun bpeTrainingDoesNotConsumeMergesOnControlTokenMarkup() {
        val tokenizer = Tokenizer(
            TokenizerConfig(
                maxVocabularySize = 512,
                maxTokenLength = 16
            )
        )
        val corpus = List(40) {
            "<USER><ASSISTANT><THINK>reasoningpayload</THINK><ASSISTANT>answerpayload<EOS>"
        }

        tokenizer.trainBpe(corpus, targetVocabSize = 256, maxNewMerges = 32)

        val markupMerges = tokenizer.exportMerges().filter { merge ->
            merge.first.contains('<') || merge.first.contains('>') ||
                merge.second.contains('<') || merge.second.contains('>')
        }
        assertTrue("Control-token markup must not be learned as BPE fragments.", markupMerges.isEmpty())

        val encoded = tokenizer.encode(corpus.first(), autoRegister = false)
        assertTrue(encoded.contains(Tokenizer.USER_ID))
        assertTrue(encoded.contains(Tokenizer.ASSISTANT_ID))
        assertTrue(encoded.contains(Tokenizer.THINK_START_ID))
        assertTrue(encoded.contains(Tokenizer.THINK_END_ID))
        assertTrue(encoded.contains(Tokenizer.EOS_ID))
    }

    @Test
    fun vocabularyExpansionAndBpeMergesSurviveCheckpointRoundTrip() {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            checkpoint.delete()
            val config = EngineConfig(
                dModel = 16,
                nHeads = 4,
                dFf = 32,
                maxSeqLen = 32,
                loraRank = 2,
                layerCount = 1,
                tokenizerConfig = TokenizerConfig(
                    maxVocabularySize = 512,
                    maxTokenLength = 16
                )
            )
            val engine = ArgusNeuralEngine.createBlocking(context, config)
            val originalVocabulary = engine.tokenizer.exportVocab()
            val corpus = List(64) {
                "Argus builds a local language model by learning recurring token patterns from useful documents."
            }

            val update = engine.trainTokenizerFromCorpus(corpus)

            assertTrue(update.tokensAdded > 0)
            assertTrue(update.vocabularySize > update.previousVocabularySize)
            assertTrue(checkpoint.isFile)

            val restoredConfig = ArgusNeuralEngine.resolveConfig(
                context = context,
                totalRamGb = 2,
                maxHeapMb = 128L,
                availableRamGb = 1.0f
            )
            val restored = ArgusNeuralEngine.createBlocking(context, restoredConfig)

            assertTrue(restored.loadCheckpointIfExists())
            assertEquals(
                originalVocabulary,
                restored.tokenizer.exportVocab().take(originalVocabulary.size)
            )
            assertEquals(engine.tokenizer.exportVocab(), restored.tokenizer.exportVocab())
            assertEquals(engine.tokenizer.exportMerges(), restored.tokenizer.exportMerges())

            val trainingText = "Argus learns a short language sequence."
            val step = restored.trainStepDetailed(trainingText)
            assertTrue(step.loss.isFinite())
            assertEquals(
                restored.tokenizer.encode(trainingText, autoRegister = false).size,
                step.inputTokenCount
            )
            assertTrue(step.predictionTargetCount > 0)
            assertTrue(step.contextWindowCount > 0)
            assertTrue(step.optimizerUpdateCount > 0)
        } finally {
            if (previousCheckpoint == null) {
                checkpoint.delete()
            } else {
                checkpoint.writeBytes(previousCheckpoint)
            }
            File(context.filesDir, "$CHECKPOINT_FILE_NAME.tmp").delete()
        }
    }

    @Test
    fun legacyGreedyCheckpointKeepsItsAlgorithmAndUsesLearnedVocabulary() {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            checkpoint.delete()
            val config = EngineConfig(
                dModel = 16,
                nHeads = 4,
                dFf = 32,
                maxSeqLen = 32,
                loraRank = 2,
                layerCount = 1,
                tokenizerConfig = TokenizerConfig(
                    algorithm = ArgusTokenizerAlgorithm.GREEDY_LONGEST_MATCH_V1,
                    maxVocabularySize = 512,
                    maxTokenLength = 16
                )
            )
            val original = ArgusNeuralEngine.createBlocking(context, config)
            val corpus = List(48) {
                "Greedy legacy tokenizer learns repeated local language patterns from corpus documents."
            }
            val update = original.trainTokenizerFromCorpus(corpus)

            assertTrue(update.tokensAdded > 0)
            assertEquals(0, update.mergeCount)
            assertTrue(checkpoint.isFile)

            val restoredConfig = ArgusNeuralEngine.resolveConfig(
                context = context,
                totalRamGb = 2,
                maxHeapMb = 128L,
                availableRamGb = 1.0f
            )
            assertEquals(
                ArgusTokenizerAlgorithm.GREEDY_LONGEST_MATCH_V1,
                restoredConfig.tokenizerConfig.algorithm
            )
            val restored = ArgusNeuralEngine.createBlocking(context, restoredConfig)

            assertTrue(restored.loadCheckpointIfExists())
            assertEquals(original.tokenizer.exportVocab(), restored.tokenizer.exportVocab())

            val trainingText = "Greedy legacy tokenizer keeps stable token IDs."
            assertEquals(
                original.tokenizer.encode(trainingText, autoRegister = false),
                restored.tokenizer.encode(trainingText, autoRegister = false)
            )
            assertTrue(restored.trainStepDetailed(trainingText).loss.isFinite())
        } finally {
            if (previousCheckpoint == null) {
                checkpoint.delete()
            } else {
                checkpoint.writeBytes(previousCheckpoint)
            }
            File(context.filesDir, "$CHECKPOINT_FILE_NAME.tmp").delete()
        }
    }

    private companion object {
        const val CHECKPOINT_FILE_NAME = "argus_model_weights.bin"
    }
}
