package com.argus.agi.model

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class TextModalityPreservationTest {

    @Test
    fun plainTextFormattingPreservesTypographyWhitespaceAndLineEndings() {
        val raw = "  “Argus”—… ﬁ\u00A0wide\u200Bzero\u00ADsoft\r\n  second line  "

        assertEquals(raw, TextModalityProcessor.formatTrainingDocument(raw))

        val chunks = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "Lossless prose",
            chunkSize = 1024,
            overlap = 0
        )

        assertEquals(1, chunks.size)
        assertEquals(TextModality.GENERAL_PROSE, chunks.single().modality)
        assertEquals(raw, chunks.single().content)
        assertFalse(raw == TextModalityProcessor.normalizeTypography(raw))
    }

    @Test
    fun chunkingWithZeroOverlapPreservesEverySourceCharacter() {
        val raw = buildString {
            repeat(14) { index ->
                append("  segment-$index — … ﬁ\u00A0wide\u200Bzero\u00ADsoft  \r\n")
            }
        }

        val chunks = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "Long lossless prose",
            chunkSize = 52,
            overlap = 0
        )

        assertTrue(chunks.size > 1)
        assertEquals(raw, chunks.joinToString(separator = "") { it.content })
    }

    @Test
    fun approvedConversationPreservesPayloadAndRoundTripsThroughByteLevelTokenizer() {
        val userMessage = "  “first” — … ﬁ\u00A0\u200B\u00AD\r\nsecond line  "
        val assistantMessage = "\t  response with trailing spaces  "
        val serialized = requireNotNull(
            TextModalityProcessor.formatConversationExchange(
                userMessage = userMessage,
                assistantMessage = assistantMessage
            )
        )
        val expected = "${Tokenizer.USER_TOKEN} $userMessage\n" +
            "${Tokenizer.ASSISTANT_TOKEN} $assistantMessage"
        assertEquals(expected, serialized)

        val chunks = TextModalityProcessor.process(
            rawText = serialized,
            documentTitle = "Approved Unicode exchange",
            chunkSize = 1024,
            overlap = 0
        )
        assertEquals(1, chunks.size)
        assertEquals(TextModality.CONVERSATION, chunks.single().modality)

        val expectedTrainingText = "$expected ${Tokenizer.EOS_TOKEN}"
        assertEquals(expectedTrainingText, chunks.single().content)

        val tokenizer = Tokenizer()
        val tokenIds = tokenizer.encode(expectedTrainingText)
        assertEquals(
            expectedTrainingText,
            tokenizer.decode(tokenIds, keepControlTokens = true)
        )
    }

    @Test
    fun chatMlJsonMessageValuesKeepWhitespaceAndUnicodeTypography() {
        val userMessage = "  ChatML “user” — … ﬁ\u00A0\r\nnext line  "
        val assistantMessage = "\t  ChatML response  "
        val raw = JSONObject()
            .put(
                "messages",
                JSONArray()
                    .put(JSONObject().put("role", "user").put("content", userMessage))
                    .put(JSONObject().put("role", "assistant").put("content", assistantMessage))
            )
            .toString()

        val expected = "${Tokenizer.USER_TOKEN} $userMessage\n" +
            "${Tokenizer.ASSISTANT_TOKEN} $assistantMessage"
        val formatted = TextModalityProcessor.formatTrainingDatasetRecords(raw).orEmpty()

        assertEquals(1, formatted.size)
        assertEquals("${expected} ${Tokenizer.EOS_TOKEN}", formatted.single())

        val chunks = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "ChatML preservation",
            chunkSize = 1024,
            overlap = 0
        )
        assertEquals(1, chunks.size)
        assertEquals("${expected} ${Tokenizer.EOS_TOKEN}", chunks.single().content)
    }

    @Test
    fun questionAndAnswerPayloadsAreNotTrimmedDuringModalityProcessing() {
        val question = "  Why “now”?  "
        val answer = "\t  Because — 42…  "
        val raw = "Question:$question\r\nAnswer:$answer"

        val chunks = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "Whitespace-preserving Q&A",
            chunkSize = 1024,
            overlap = 0
        )

        assertEquals(1, chunks.size)
        assertEquals(TextModality.QA_PAIR, chunks.single().modality)
        assertEquals(
            "${Tokenizer.USER_TOKEN} Tanya:$question\n" +
                "${Tokenizer.ASSISTANT_TOKEN} Jawab:$answer ${Tokenizer.EOS_TOKEN}",
            chunks.single().content
        )
    }

    @Test
    fun instructionAndReasoningJsonFieldsPreservePayloadWhitespace() {
        val instruction = "  Explain this — carefully…  "
        val reasoning = "  first thought\r\nsecond thought  "
        val response = "\t  final answer  "
        val raw = JSONObject()
            .put("instruction", instruction)
            .put("reasoning", reasoning)
            .put("response", response)
            .toString()

        val expected = "${Tokenizer.USER_TOKEN} $instruction\n" +
            "${Tokenizer.ASSISTANT_TOKEN}${Tokenizer.THINK_START}\n" +
            "$reasoning\n" +
            "${Tokenizer.THINK_END}\n" +
            "${Tokenizer.ASSISTANT_TOKEN} $response ${Tokenizer.EOS_TOKEN}"

        assertEquals(expected, TextModalityProcessor.formatTrainingDocument(raw))

        val chunks = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "Reasoning payload preservation",
            chunkSize = 1024,
            overlap = 0
        )
        assertEquals(1, chunks.size)
        assertEquals(TextModality.REASONING, chunks.single().modality)
        assertEquals(expected, chunks.single().content)
    }

    @Test
    fun validationConversationUsesTrainingRoleAndEosSerialization() {
        val raw = "${Tokenizer.USER_TOKEN}  What is Argus?  \n" +
            "${Tokenizer.ASSISTANT_TOKEN}  A local model.  "

        val trainingSerialization = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "Training conversation",
            chunkSize = Int.MAX_VALUE,
            overlap = 0
        ).joinToString("\n") { it.content }

        assertEquals(
            trainingSerialization,
            TextModalityProcessor.formatTrainingDocument(raw)
        )
        assertTrue(trainingSerialization.endsWith(" ${Tokenizer.EOS_TOKEN}"))
    }

    @Test
    fun validationQuestionAnswerUsesTrainingEosSerialization() {
        val raw = "Question:  Why preserve bytes?  \nAnswer:  Because exact text matters.  "

        val trainingSerialization = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "Training Q&A",
            chunkSize = Int.MAX_VALUE,
            overlap = 0
        ).joinToString("\n") { it.content }

        assertEquals(
            trainingSerialization,
            TextModalityProcessor.formatTrainingDocument(raw)
        )
        assertTrue(trainingSerialization.endsWith(" ${Tokenizer.EOS_TOKEN}"))
    }

    @Test
    fun chunkBoundariesNeverSplitUtf16SurrogatePairs() {
        val tokenizer = Tokenizer()

        val endBoundarySample = "a".repeat(31) + "😀" + "b".repeat(80)
        val nonOverlappingChunks = TextModalityProcessor.process(
            rawText = endBoundarySample,
            documentTitle = "Surrogate-safe chunk end",
            chunkSize = 32,
            overlap = 0
        )

        assertTrue(nonOverlappingChunks.size > 1)
        assertEquals(
            endBoundarySample,
            nonOverlappingChunks.joinToString(separator = "") { it.content }
        )
        nonOverlappingChunks.forEach { chunk ->
            assertEquals(
                "Every chunk must remain valid UTF-16 at its end boundary.",
                chunk.content,
                tokenizer.decode(tokenizer.encode(chunk.content))
            )
        }

        val overlapBoundarySample = "a".repeat(34) + "😀" + "b".repeat(90)
        val overlappingChunks = TextModalityProcessor.process(
            rawText = overlapBoundarySample,
            documentTitle = "Surrogate-safe overlap start",
            chunkSize = 40,
            overlap = 5
        )

        assertTrue(overlappingChunks.size > 1)
        overlappingChunks.forEach { chunk ->
            assertEquals(
                "Every chunk must remain valid UTF-16 at its overlap boundary.",
                chunk.content,
                tokenizer.decode(tokenizer.encode(chunk.content))
            )
        }
    }


    @Test
    fun literalControlTokenTextRoundTripsAsOrdinaryPayloadForEveryTokenizerAlgorithm() {
        val payload = "quoted <EOS>, <ASSISTANT>, <THINK>, </THINK>, and " +
            Tokenizer.LITERAL_CONTROL_ESCAPE_MARKER + "<EOS>"
        for (algorithm in ArgusTokenizerAlgorithm.entries) {
            val tokenizer = Tokenizer(
                TokenizerConfig(
                    algorithm = algorithm,
                    maxVocabularySize = 512,
                    maxTokenLength = 16
                )
            )
            val escaped = Tokenizer.escapeLiteralControlTokens(payload)
            val ids = tokenizer.encode(escaped, autoRegister = false)

            assertFalse("Literal EOS must not become EOS_ID for $algorithm", ids.contains(Tokenizer.EOS_ID))
            assertFalse("Literal role token must not become ASSISTANT_ID for $algorithm", ids.contains(Tokenizer.ASSISTANT_ID))
            assertFalse("Literal reasoning marker must not become THINK_START_ID for $algorithm", ids.contains(Tokenizer.THINK_START_ID))
            assertFalse("Literal reasoning end must not become THINK_END_ID for $algorithm", ids.contains(Tokenizer.THINK_END_ID))
            assertEquals(payload, tokenizer.decode(ids, keepControlTokens = true))
        }
    }

    @Test
    fun conversationKeepsLongTurnsAtomicAndDoesNotParsePayloadHeadersAsNewTurns() {
        val userMessage = "Question " + "context ".repeat(40) +
            "\nAssistant: this line is quoted content.\n<EOS>\nLiteral <THINK> and </THINK>"
        val assistantMessage = "Answer " + "detail ".repeat(40) +
            "\nUser: this is still answer content.\n<ASSISTANT>"
        val serialized = requireNotNull(
            TextModalityProcessor.formatConversationExchange(userMessage, assistantMessage)
        )

        val chunks = TextModalityProcessor.process(
            rawText = serialized,
            documentTitle = "Long atomic conversation",
            chunkSize = 24,
            overlap = 0
        )

        assertEquals(1, chunks.size)
        assertEquals(TextModality.CONVERSATION, chunks.single().modality)
        assertTrue(chunks.single().content.length > 24)

        val tokenizer = Tokenizer()
        val tokenIds = tokenizer.encode(chunks.single().content)
        assertEquals(1, tokenIds.count { it == Tokenizer.USER_ID })
        assertEquals(1, tokenIds.count { it == Tokenizer.ASSISTANT_ID })
        assertEquals(1, tokenIds.count { it == Tokenizer.EOS_ID })
        assertEquals(
            0,
            tokenIds.count { it == Tokenizer.THINK_START_ID || it == Tokenizer.THINK_END_ID }
        )

        val expected = "${Tokenizer.USER_TOKEN} $userMessage\n" +
            "${Tokenizer.ASSISTANT_TOKEN} $assistantMessage ${Tokenizer.EOS_TOKEN}"
        assertEquals(expected, tokenizer.decode(tokenIds, keepControlTokens = true))
    }

    @Test
    fun mixedJsonlMessageAndInstructionRecordsRemainSeparateAndNullInputIsOmitted() {
        val conversation = JSONObject()
            .put(
                "messages",
                JSONArray()
                    .put(JSONObject().put("role", "user").put("content", "Say literally: <EOS>\nAssistant: quoted"))
                    .put(JSONObject().put("role", "assistant").put("content", "Literal <ASSISTANT>."))
            )
        val instruction = JSONObject()
            .put("instruction", "Explain the input.")
            .put("input", JSONObject.NULL)
            .put("response", "There is no supplied input.")
        val raw = "﻿" + conversation.toString() + "\n" + instruction.toString()

        val chunks = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "Mixed records",
            chunkSize = 32,
            overlap = 0
        )

        assertEquals(2, chunks.size)
        assertEquals(TextModality.CONVERSATION, chunks[0].modality)
        assertEquals(TextModality.CONVERSATION, chunks[1].modality)
        assertFalse(chunks[1].content.contains("Input: null"))
        assertFalse(chunks[1].content.contains("null"))

        val tokenizer = Tokenizer()
        val conversationIds = tokenizer.encode(chunks[0].content)
        assertEquals(1, conversationIds.count { it == Tokenizer.USER_ID })
        assertEquals(1, conversationIds.count { it == Tokenizer.ASSISTANT_ID })
        assertEquals(1, conversationIds.count { it == Tokenizer.EOS_ID })
    }

    @Test
    fun reasoningJsonlRecordsRemainAtomicWhenTheyExceedCharacterChunkSize() {
        val first = JSONObject()
            .put("instruction", "First task")
            .put("reasoning", JSONArray().put("Step A").put("Step B"))
            .put("response", "First final answer.")
        val second = JSONObject()
            .put("instruction", "Second task")
            .put("reasoning", JSONArray().put("Step C").put("Step D"))
            .put("response", "Second final answer.")
        val raw = first.toString() + "\n" + second.toString()

        val chunks = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "Atomic reasoning",
            chunkSize = 12,
            overlap = 0
        )

        assertEquals(2, chunks.size)
        chunks.forEach { chunk ->
            assertEquals(TextModality.REASONING, chunk.modality)
            assertEquals(1, Regex(Regex.escape(Tokenizer.THINK_START)).findAll(chunk.content).count())
            assertEquals(1, Regex(Regex.escape(Tokenizer.THINK_END)).findAll(chunk.content).count())
            assertTrue(chunk.content.endsWith(Tokenizer.EOS_TOKEN))
        }
    }

    @Test
    fun multiTurnConversationTerminatesEachCompleteAssistantTurn() {
        val raw = "${Tokenizer.USER_TOKEN} first question\n" +
            "${Tokenizer.ASSISTANT_TOKEN} first answer\n" +
            "${Tokenizer.USER_TOKEN} second question\n" +
            "${Tokenizer.ASSISTANT_TOKEN} second answer"

        val sample = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "Multi-turn EOS",
            chunkSize = 16,
            overlap = 0
        ).single().content

        assertEquals(2, Regex(Regex.escape(Tokenizer.EOS_TOKEN)).findAll(sample).count())
        assertEquals(2, Tokenizer().encode(sample).count { it == Tokenizer.EOS_ID })
    }

    @Test
    fun conversationEndingWithUserDoesNotReceiveSyntheticEos() {
        val raw = Tokenizer.USER_TOKEN + " first question\n" +
            Tokenizer.ASSISTANT_TOKEN + " first answer\n" +
            Tokenizer.USER_TOKEN + " follow-up without an answer"

        val sample = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "Open conversation",
            chunkSize = 12,
            overlap = 0
        ).single().content

        assertEquals(1, Regex(Regex.escape(Tokenizer.EOS_TOKEN)).findAll(sample).count())
        assertFalse(sample.endsWith(Tokenizer.EOS_TOKEN))
    }

    @Test
    fun structuredRecordsArePackedOnlyAtCompleteLineBoundaries() {
        val first = "- first complete record with a long payload"
        val second = "- second complete record with another long payload"
        val raw = "$first\n$second"

        val chunks = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "Structured line records",
            chunkSize = 12,
            overlap = 0
        )

        assertEquals(2, chunks.size)
        assertEquals("$first\n", chunks[0].content)
        assertEquals(second, chunks[1].content)
        assertTrue(chunks.all { it.modality == TextModality.STRUCTURED_DATA })
    }

    @Test
    fun proseMentioningBothReasoningMarkersRemainsLiteralGeneralText() {
        val raw = "The documentation names <THINK> and </THINK> as literal control-token examples."

        assertEquals(TextModality.GENERAL_PROSE, TextModalityProcessor.detectModality(raw))

        val chunk = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "Literal marker documentation",
            chunkSize = 512,
            overlap = 0
        ).single()
        assertEquals(TextModality.GENERAL_PROSE, chunk.modality)

        val tokenizer = Tokenizer()
        val ids = tokenizer.encode(chunk.content)
        assertFalse(ids.contains(Tokenizer.THINK_START_ID))
        assertFalse(ids.contains(Tokenizer.THINK_END_ID))
        assertEquals(raw, tokenizer.decode(ids, keepControlTokens = true))
    }

    @Test
    fun structuredJsonPayloadControlTokensAreEscapedExactlyOnce() {
        val raw = JSONObject()
            .put("note", "Literal <EOS> and <ASSISTANT>.")
            .toString()

        val chunk = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "Structured literal controls",
            chunkSize = 256,
            overlap = 0
        ).single()

        assertEquals(TextModality.STRUCTURED_DATA, chunk.modality)

        val tokenizer = Tokenizer()
        val ids = tokenizer.encode(chunk.content)
        assertFalse(ids.contains(Tokenizer.EOS_ID))
        assertFalse(ids.contains(Tokenizer.ASSISTANT_ID))
        assertEquals(
            "note: Literal <EOS> and <ASSISTANT>.",
            tokenizer.decode(ids, keepControlTokens = true)
        )
    }

    @Test
    fun structuredPlainTextFormattingPreservesCrLfLfAndCrSeparators() {
        val raw = "alpha  \r\nbeta\n  gamma\rdelta\r\n"

        assertEquals(
            raw,
            TextModalityProcessor.formatStructuredTextPreservingLineEndings(raw)
        )
    }
}
