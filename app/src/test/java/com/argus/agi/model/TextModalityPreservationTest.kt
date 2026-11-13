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
@Config(sdk = [33])
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
        assertEquals(expected, formatted.single())

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
    fun structuredPlainTextFormattingPreservesCrLfLfAndCrSeparators() {
        val raw = "alpha  \r\nbeta\n  gamma\rdelta\r\n"

        assertEquals(
            raw,
            TextModalityProcessor.formatStructuredTextPreservingLineEndings(raw)
        )
    }
}
