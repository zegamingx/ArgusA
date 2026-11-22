package com.argus.agi.model

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.json.JSONObject
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ReasoningTrainingTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun reasoningJsonlBecomesOrderedUserThinkAndFinalAnswerSequence() {
        val raw = """{"instruction":"Solve the proportion.","input":"3 machines make 30 units in 5 hours. Find the output of 6 machines in 5 hours.","reasoning":["Each machine produces 10 units in 5 hours.","Six machines produce six times that amount."],"response":"60 units."}"""

        assertEquals(TextModality.REASONING, TextModalityProcessor.detectModality(raw))
        val chunks = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "Proportion reasoning",
            chunkSize = 512,
            overlap = 0
        )
        assertEquals(1, chunks.size)
        assertEquals(TextModality.REASONING, chunks.single().modality)

        val formatted = chunks.single().content
        val thinkStart = formatted.indexOf(Tokenizer.THINK_START)
        val thinkEnd = formatted.indexOf(Tokenizer.THINK_END)
        val finalAssistant = formatted.indexOf(
            Tokenizer.ASSISTANT_TOKEN,
            thinkEnd + Tokenizer.THINK_END.length
        )
        val finalAnswer = formatted.indexOf("60 units.")

        assertTrue(formatted.startsWith(Tokenizer.USER_TOKEN))
        assertTrue(thinkStart > formatted.indexOf(Tokenizer.ASSISTANT_TOKEN))
        assertTrue(thinkEnd > thinkStart)
        assertTrue(finalAssistant > thinkEnd)
        assertTrue(finalAnswer > finalAssistant)
        assertTrue(formatted.endsWith(Tokenizer.EOS_TOKEN))

        val tokenizer = Tokenizer(
            TokenizerConfig(maxVocabularySize = 512, maxTokenLength = 32)
        )
        val ids = tokenizer.encode(formatted, autoRegister = false)
        assertTrue(ids.contains(Tokenizer.THINK_START_ID))
        assertTrue(ids.contains(Tokenizer.THINK_END_ID))
        assertFalse(tokenizer.decode(ids).contains(Tokenizer.THINK_START))
        val decodedWithReasoning = tokenizer.decode(ids, keepReasoningTokens = true)
        assertTrue(decodedWithReasoning.contains(Tokenizer.THINK_START))
        assertTrue(decodedWithReasoning.contains(Tokenizer.THINK_END))
        assertFalse(decodedWithReasoning.contains(Tokenizer.USER_TOKEN))
    }

    @Test
    fun textWrappedReasoningAndPromptAnswerJsonlAreNormalizedForSft() {
        val firstText = """
            Soal: Dalam konteks kalimat 'Pemandangan di desa itu sangat indah.', apa padanan dekat kata 'indah'?
            Tuliskan penalaran di dalam tag <think> dan jawaban akhir di dalam tag <answer>.
            Jawab: <think>Padanan dekat 'indah' pada konteks ini adalah 'elok'.</think><answer>elok</answer>
        """.trimIndent()
        val secondText = """
            Soal: Susun kalimat-kalimat berikut menjadi paragraf yang runtut berdasarkan hubungan waktu dan makna:
            - Dina memilih buku yang ingin dipinjam.
            - Ia membawa buku tersebut ke meja layanan.
            - Petugas mencatat peminjaman pada sistem.
            - Dina menerima buku dan membawanya pulang.
            Tuliskan penalaran di dalam tag <think> dan jawaban akhir di dalam tag <answer>.
            Jawab: <think>Urutkan tindakan dari memilih buku hingga menerima buku yang telah dicatat.</think><answer>Dina memilih buku, membawa buku ke meja layanan, petugas mencatat peminjaman, lalu Dina menerima buku.</answer>
        """.trimIndent()
        val taggedRecords = listOf(
            JSONObject().put("text", firstText).toString(),
            JSONObject().put("text", secondText).toString(),
            JSONObject()
                .put("prompt", "What is 15 multiplied by 5?")
                .put("answer", "75")
                .toString()
        )
        val jsonl = taggedRecords.joinToString("\\n")

        val formattedRecords = requireNotNull(
            TextModalityProcessor.formatTrainingDatasetRecords(jsonl)
        )
        assertEquals(3, formattedRecords.size)

        val firstFormatted = formattedRecords[0]
        assertTrue(firstFormatted.startsWith(Tokenizer.USER_TOKEN))
        assertTrue(firstFormatted.contains("apa padanan dekat kata 'indah'?"))
        assertTrue(firstFormatted.contains("Padanan dekat 'indah'"))
        assertTrue(firstFormatted.contains("elok"))
        assertFalse(firstFormatted.contains("Tuliskan penalaran"))
        assertTrue(firstFormatted.contains(Tokenizer.THINK_START))
        assertTrue(firstFormatted.contains(Tokenizer.THINK_END))
        assertTrue(firstFormatted.endsWith(Tokenizer.EOS_TOKEN))

        val secondFormatted = formattedRecords[1]
        assertTrue(secondFormatted.contains("Dina memilih buku yang ingin dipinjam."))
        assertTrue(secondFormatted.contains("Petugas mencatat peminjaman pada sistem."))
        assertFalse(secondFormatted.contains("Tuliskan penalaran"))
        assertTrue(secondFormatted.contains(Tokenizer.THINK_START))
        assertTrue(secondFormatted.contains(Tokenizer.THINK_END))

        val grpoFormatted = formattedRecords[2]
        assertTrue(grpoFormatted.startsWith(Tokenizer.USER_TOKEN))
        assertTrue(grpoFormatted.contains("What is 15 multiplied by 5?"))
        assertTrue(grpoFormatted.contains("75"))
        assertFalse(grpoFormatted.contains(Tokenizer.THINK_START))

        val chunks = TextModalityProcessor.process(
            rawText = jsonl,
            documentTitle = "Mixed SFT JSONL",
            chunkSize = 512,
            overlap = 0
        )
        assertEquals(
            listOf(
                TextModality.REASONING,
                TextModality.REASONING,
                TextModality.CONVERSATION
            ),
            chunks.map { it.modality }
        )
        assertEquals(3, chunks.size)
    }

    @Test
    fun malformedTextWrappedReasoningDoesNotPassStructuredTrainingValidation() {
        val malformed = JSONObject()
            .put(
                "text",
                "Soal: Add two numbers.\\nJawab: <think>Incomplete reasoning</think><answer></answer>"
            )
            .toString()

        assertEquals(
            null,
            TextModalityProcessor.formatTrainingDatasetRecords(malformed)
        )
    }

    @Test
    fun prettyPrintedJsonArrayIsExpandedIntoIndependentReasoningRecords() {
        val raw = """
            [
              {
                "instruction": "Add two numbers.",
                "reasoning": ["Identify both operands.", "Add them."],
                "response": "5."
              },
              {
                "instruction": "Multiply two numbers.",
                "steps": ["Identify both factors.", "Multiply them."],
                "response": "12."
              }
            ]
        """.trimIndent()

        val chunks = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "Reasoning set",
            chunkSize = 512,
            overlap = 0
        )
        assertEquals(TextModality.REASONING, chunks.first().modality)
        val formatted = chunks.joinToString("\n") { it.content }
        assertEquals(2, Regex("<THINK>").findAll(formatted).count())
        assertEquals(2, Regex("</THINK>").findAll(formatted).count())
        assertTrue(formatted.contains("5."))
        assertTrue(formatted.contains("12."))
    }

    @Test
    fun prettyPrintedReasoningArrayIsDetectedBeforeChunking() {
        val raw = """
            [
              {
                "instruction": "Find the sum.",
                "reasoning": ["Identify the operands.", "Add the values."],
                "response": "9."
              }
            ]
        """.trimIndent()

        assertEquals(TextModality.REASONING, TextModalityProcessor.detectModality(raw))
    }

    @Test
    fun emptyReasoningFieldFallsBackToStepsAlias() {
        val raw = """{"instruction":"Find the difference.","reasoning":"","steps":[{"step":"Subtract the smaller value from the larger value."}],"response":"4."}"""
        val formatted = TextModalityProcessor.formatStructuredLineIfJson(raw)

        assertTrue(formatted.contains(Tokenizer.THINK_START))
        assertTrue(formatted.contains("Subtract the smaller value"))
        assertTrue(formatted.contains(Tokenizer.THINK_END))
        assertTrue(formatted.contains("4."))
    }

    @Test
    fun streamedJsonBoundaryScannerIgnoresBracesInsideQuotedStrings() {
        val json = """{"instruction":"Symbols { } and an escaped quote: \"","reasoning":["Nested [brackets] are text."],"response":"Done."} trailing text"""
        val start = json.indexOf('{')
        val end = TextModalityProcessor.findCompleteJsonValueEnd(json, start)

        assertEquals(json.indexOf(" trailing text"), end)

        val incompleteJson = "{\"instruction\":\"unfinished"
        assertEquals(
            null,
            TextModalityProcessor.findCompleteJsonValueEnd(incompleteJson, 0)
        )
    }

    @Test
    fun streamFormatterAcceptsJsonlAndPrettyPrintedReasoningRecords() {
        val jsonl = listOf(
            """{"instruction":"Add 1 and 2.","reasoning":["Identify the operands.","Add them."],"response":"3."}""",
            """{"prompt":"Subtract 7 and 3.","steps":["Subtract the second value from the first."],"completion":"4."}"""
        ).joinToString("\n")
        val records = TextModalityProcessor.formatTrainingDatasetRecords(jsonl)

        assertEquals(2, records?.size)
        assertTrue(records.orEmpty().all { it.contains(Tokenizer.THINK_START) })
        assertTrue(records.orEmpty().all { it.contains(Tokenizer.THINK_END) })

        val prettyObject = """
            {
              "instruction": "Double 4.",
              "reasoning": ["Multiply 4 by 2."],
              "response": "8."
            }
        """.trimIndent()
        val prettyRecords = TextModalityProcessor.formatTrainingDatasetRecords(prettyObject)
        assertEquals(1, prettyRecords?.size)
        assertTrue(prettyRecords.orEmpty().single().contains("Multiply 4 by 2."))
    }

    @Test
    fun streamFormatterRejectsValidJsonlPrefixFollowedByUnrelatedText() {
        val validRecord = """{"instruction":"Add two numbers.","reasoning":["Add the operands."],"response":"5."}"""
        val malformedDataset = "$validRecord\nnot-a-json-record"

        assertEquals(
            null,
            TextModalityProcessor.formatTrainingDatasetRecords(malformedDataset)
        )
    }

    @Test
    fun streamFormatterAcceptsPrettyPrintedArrayAsIndependentRecords() {
        val prettyArray = """
            [
              {
                "instruction": "Add two numbers.",
                "reasoning": ["Add both operands."],
                "response": "5."
              },
              {
                "instruction": "Multiply two numbers.",
                "steps": ["Multiply both factors."],
                "response": "12."
              }
            ]
        """.trimIndent()

        val records = TextModalityProcessor.formatTrainingDatasetRecords(prettyArray)

        assertEquals(2, records?.size)
        assertTrue(records.orEmpty().all { it.contains(Tokenizer.THINK_START) })
        assertTrue(records.orEmpty().all { it.contains(Tokenizer.THINK_END) })

        val validationDocument = TextModalityProcessor.formatTrainingDocument(prettyArray)
        assertEquals(
            2,
            Regex(Regex.escape(Tokenizer.THINK_START)).findAll(validationDocument).count()
        )
    }

    @Test
    fun legacyInstructionResponseFormatRemainsUnchangedWithoutReasoning() {
        val raw = """{"instruction":"Explain gravity.","input":"For a falling object.","response":"Gravity attracts masses."}"""
        val formatted = TextModalityProcessor.formatStructuredLineIfJson(raw)

        assertEquals(
            "${Tokenizer.USER_TOKEN} Explain gravity.\nInput: For a falling object.\n" +
                "${Tokenizer.ASSISTANT_TOKEN} Gravity attracts masses. ${Tokenizer.EOS_TOKEN}",
            formatted
        )
        assertFalse(formatted.contains(Tokenizer.THINK_START))
    }

    @Test
    fun reasoningInferenceUsesTheSameThinkPrefixAsTrainingSerialization() {
        assertEquals(
            "${Tokenizer.USER_TOKEN}Explain the result.${Tokenizer.ASSISTANT_TOKEN}${Tokenizer.THINK_START}",
            ArgusNeuralEngine.formatGenerationPrompt("Explain the result.", reasoningMode = true)
        )
        assertEquals(
            "${Tokenizer.USER_TOKEN}Explain the result.${Tokenizer.ASSISTANT_TOKEN}",
            ArgusNeuralEngine.formatGenerationPrompt("Explain the result.", reasoningMode = false)
        )
        assertEquals(
            "${Tokenizer.USER_TOKEN}Explain the result.${Tokenizer.ASSISTANT_TOKEN}${Tokenizer.THINK_START}",
            ArgusNeuralEngine.formatGenerationPrompt(
                "${Tokenizer.USER_TOKEN}Explain the result.${Tokenizer.ASSISTANT_TOKEN}",
                reasoningMode = true
            )
        )
        assertEquals(
            "${Tokenizer.USER_TOKEN}Explain the result.${Tokenizer.ASSISTANT_TOKEN}${Tokenizer.THINK_START}",
            ArgusNeuralEngine.formatGenerationPrompt(
                "${Tokenizer.USER_TOKEN}Explain the result.${Tokenizer.ASSISTANT_TOKEN}${Tokenizer.THINK_START}",
                reasoningMode = true
            )
        )
    }

    @Test
    fun generatedReasoningIsSeparatedFromFinalAnswerOnlyWhenComplete() {
        val generated = "${Tokenizer.THINK_START}Check the known values.\nCompute the ratio." +
            "${Tokenizer.THINK_END}${Tokenizer.ASSISTANT_TOKEN}The answer is 60." +
            Tokenizer.EOS_TOKEN
        val parsed = ReasoningOutputParser.parse(generated)

        assertEquals("The answer is 60.", parsed.answer)
        assertEquals("Check the known values.\nCompute the ratio.", parsed.reasoning)

        val incompleteText = "${Tokenizer.THINK_START}A partial explanation without a final answer."
        val incomplete = ReasoningOutputParser.parse(incompleteText)
        assertEquals(incompleteText, incomplete.answer)
        assertEquals(null, incomplete.reasoning)
    }

    @Test
    fun reasoningFormattedSampleRunsThroughNeuralNextTokenTraining() {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            checkpoint.delete()
            val engine = ArgusNeuralEngine.createBlocking(
                context,
                EngineConfig(
                    dModel = 16,
                    nHeads = 4,
                    dFf = 32,
                    maxSeqLen = 32,
                    loraRank = 2,
                    layerCount = 1,
                    tokenizerConfig = TokenizerConfig(
                        maxVocabularySize = 512,
                        maxTokenLength = 32
                    )
                )
            )
            val sample = TextModalityProcessor.process(
                rawText = JSONObject().put(
                    "text",
                    "Soal: Add 2 and 3.\\n" +
                        "Tuliskan penalaran di dalam tag <think> dan jawaban akhir di dalam tag <answer>.\\n" +
                        "Jawab: <think>Identify the two values. Add 2 + 3.</think><answer>5.</answer>"
                ).toString(),
                documentTitle = "Reasoning smoke test",
                chunkSize = 512,
                overlap = 0
            ).single().content

            val step = engine.trainStepDetailed(sample)
            assertTrue(step.loss.isFinite())
            assertTrue(step.inputTokenCount > 2)
            assertTrue(step.predictionTargetCount > 0)
            assertTrue(step.contextWindowCount > 0)
            assertTrue(step.optimizerUpdateCount > 0)
            assertTrue(engine.saveCheckpoint())
            assertTrue(checkpoint.isFile)
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
