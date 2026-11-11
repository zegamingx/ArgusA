package com.argus.agi.model

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class ReasoningTrainingPipelineTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun serializedReasoningSequenceProducesRealNeuralOptimizerUpdates() {
        val config = EngineConfig(
            dModel = 16,
            nHeads = 4,
            dFf = 32,
            maxSeqLen = 64,
            loraRank = 2,
            layerCount = 1
        )
        val engine = ArgusNeuralEngine.createBlocking(context, config)
        val sequence = TextModalityProcessor.process(
            rawText = """{"instruction":"What is 5 plus 7?","reasoning":["Start with five units.","Add seven more units.","Five plus seven equals twelve."],"response":"12."}"""
        ).single().content

        val result = engine.trainStepDetailed(sequence)

        assertTrue("Reasoning sequence must produce finite cross-entropy loss.", result.loss.isFinite())
        assertTrue("Reasoning tokens must be scored as next-token targets.", result.predictionTargetCount > 0)
        assertTrue("Reasoning sequence must create at least one context window.", result.contextWindowCount > 0)
        assertTrue("Reasoning sequence must update the neural optimizer.", result.optimizerUpdateCount > 0)
    }

    @Test
    fun reasoningRecordIsSerializedIntoAssistantThinkAndFinalAnswerTargets() {
        val record = """
            {
              "instruction": "What is 18 multiplied by 7?",
              "reasoning": [
                "18 multiplied by 7 is 18 multiplied by 5 plus 18 multiplied by 2.",
                "That equals 90 plus 36."
              ],
              "response": "126."
            }
        """.trimIndent()

        assertEquals(TextModality.REASONING, TextModalityProcessor.detectModality(record))

        val chunks = TextModalityProcessor.process(record, documentTitle = "Arithmetic reasoning")
        assertEquals(1, chunks.size)
        assertEquals(TextModality.REASONING, chunks.single().modality)

        val trainedSequence = chunks.single().content
        assertTrue(trainedSequence.contains("<USER>"))
        assertTrue(trainedSequence.contains("<ASSISTANT><THINK>"))
        assertTrue(trainedSequence.contains("Step 1: 18 multiplied by 7"))
        assertTrue(trainedSequence.contains("Step 2: That equals 90 plus 36."))
        assertTrue(trainedSequence.contains("</THINK>"))
        assertTrue(trainedSequence.contains("<ASSISTANT> 126."))
        assertTrue(trainedSequence.contains("<EOS>"))
    }

    @Test
    fun jsonArrayAndJsonlReasoningDatasetsAreNormalizedWithoutDroppingExamples() {
        val dataset = """
            [
              {
                "prompt": "Why does ice float?",
                "steps": ["Water expands when it freezes.", "The solid is less dense than liquid water."],
                "completion": "Ice floats because it is less dense than liquid water."
              },
              {
                "instruction": "What follows 2, 4, 8?",
                "reasoning": [
                  {"step": "Each term is twice the previous term."},
                  {"explanation": "The next term is 16."}
                ],
                "response": "16."
              }
            ]
        """.trimIndent()

        val chunks = TextModalityProcessor.process(dataset, documentTitle = "Reasoning dataset")
        assertTrue(chunks.isNotEmpty())
        assertTrue(chunks.all { it.modality == TextModality.REASONING })
        val combined = chunks.joinToString("\n") { it.content }
        assertEquals(2, Regex("<THINK>").findAll(combined).count())
        assertEquals(2, Regex("</THINK>").findAll(combined).count())
        assertTrue(combined.contains("Water expands when it freezes."))
        assertTrue(combined.contains("Each term is twice the previous term."))
        assertTrue(combined.contains("Ice floats because it is less dense than liquid water."))
        assertTrue(combined.contains("16."))
    }

    @Test
    fun ordinaryInstructionDataDoesNotInventReasoningTargets() {
        val record = """{"instruction":"Say hello.","response":"Hello."}"""

        val formatted = TextModalityProcessor.formatStructuredLineIfJson(record)
        assertEquals(TextModality.CONVERSATION, TextModalityProcessor.detectModality(formatted))
        assertFalse(formatted.contains(Tokenizer.THINK_START))
        assertFalse(formatted.contains(Tokenizer.THINK_END))
    }

    @Test
    fun reasoningGenerationStartsAtThinkTokenAndPreservesExistingAssistantPrefix() {
        assertEquals(
            "<USER>Explain the result.<ASSISTANT><THINK>",
            ArgusNeuralEngine.formatGenerationPrompt("Explain the result.", reasoningMode = true)
        )
        assertEquals(
            "<USER>Explain the result.<ASSISTANT>",
            ArgusNeuralEngine.formatGenerationPrompt("Explain the result.", reasoningMode = false)
        )
        assertEquals(
            "<USER>Explain the result.<ASSISTANT><THINK>",
            ArgusNeuralEngine.formatGenerationPrompt(
                "<USER>Explain the result.<ASSISTANT>",
                reasoningMode = true
            )
        )
        assertEquals(
            "<USER>Explain the result.<ASSISTANT><THINK>",
            ArgusNeuralEngine.formatGenerationPrompt(
                "<USER>Explain the result.<ASSISTANT><THINK>",
                reasoningMode = true
            )
        )
    }
}
