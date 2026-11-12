package com.argus.agi.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class ConversationalDatasetTest {

    @Test
    fun chatMlMessagesPreserveSystemUserAndAssistantTurns() {
        val raw = """
            {
              "messages": [
                {"role": "system", "content": "You are Argus."},
                {"role": "user", "content": "Hello Argus."},
                {"role": "assistant", "content": "Hello. How can I help?"}
              ]
            }
        """.trimIndent()

        assertEquals(
            TextModality.CONVERSATION,
            TextModalityProcessor.detectModality(raw)
        )

        val formatted = TextModalityProcessor.formatTrainingDatasetRecords(raw)
        assertEquals(1, formatted?.size)
        val record = formatted.orEmpty().single()
        assertTrue(record.startsWith("${Tokenizer.SYSTEM_TOKEN} You are Argus."))
        assertTrue(record.contains("${Tokenizer.USER_TOKEN} Hello Argus."))
        assertTrue(record.contains("${Tokenizer.ASSISTANT_TOKEN} Hello. How can I help?"))

        val chunks = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "ChatML conversation",
            chunkSize = 512,
            overlap = 0
        )
        assertEquals(1, chunks.size)
        assertEquals(TextModality.CONVERSATION, chunks.single().modality)
        assertTrue(chunks.single().content.endsWith(Tokenizer.EOS_TOKEN))
    }

    @Test
    fun shareGptConversationsMapHumanAndGptRolesToArgusTokens() {
        val raw = """
            {
              "conversations": [
                {"from": "human", "value": "What is 2 + 2?"},
                {"from": "gpt", "value": "4."},
                {"from": "human", "value": "And 3 + 3?"},
                {"from": "gpt", "value": "6."}
              ]
            }
        """.trimIndent()

        val chunks = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "ShareGPT sample",
            chunkSize = 512,
            overlap = 0
        )
        val content = chunks.single().content

        assertEquals(TextModality.CONVERSATION, chunks.single().modality)
        assertEquals(2, Regex(Regex.escape(Tokenizer.USER_TOKEN)).findAll(content).count())
        assertEquals(2, Regex(Regex.escape(Tokenizer.ASSISTANT_TOKEN)).findAll(content).count())
        assertTrue(content.contains("What is 2 + 2?"))
        assertTrue(content.contains("4."))
        assertTrue(content.contains("And 3 + 3?"))
        assertTrue(content.contains("6."))
    }

    @Test
    fun openAiTextContentPartsAreJoinedAndNonTextPartsAreIgnored() {
        val raw = """
            {
              "messages": [
                {
                  "role": "user",
                  "content": [
                    {"type": "text", "text": "Describe this subject."},
                    {"type": "image_url", "image_url": {"url": "not-training-text"}}
                  ]
                },
                {
                  "role": "assistant",
                  "content": [
                    {"type": "output_text", "text": "It is a small blue object."}
                  ]
                }
              ]
            }
        """.trimIndent()

        val formatted = TextModalityProcessor.formatTrainingDatasetRecords(raw).orEmpty().single()

        assertTrue(formatted.contains("Describe this subject."))
        assertTrue(formatted.contains("It is a small blue object."))
        assertFalse(formatted.contains("not-training-text"))
    }

    @Test
    fun prettyPrintedArrayOfConversationRecordsIsExpandedInSourceOrder() {
        val raw = """
            [
              {
                "messages": [
                  {"role": "user", "content": "First question."},
                  {"role": "assistant", "content": "First answer."}
                ]
              },
              {
                "messages": [
                  {"role": "user", "content": "Second question."},
                  {"role": "assistant", "content": "Second answer."}
                ]
              }
            ]
        """.trimIndent()

        val formatted = TextModalityProcessor.formatTrainingDatasetRecords(raw).orEmpty()
        assertEquals(2, formatted.size)
        assertTrue(formatted[0].contains("First question."))
        assertTrue(formatted[0].contains("First answer."))
        assertTrue(formatted[1].contains("Second question."))
        assertTrue(formatted[1].contains("Second answer."))

        val chunks = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "Conversation array",
            chunkSize = 512,
            overlap = 0
        )
        assertEquals(TextModality.CONVERSATION, chunks.first().modality)
        val combined = chunks.joinToString("\n") { it.content }
        assertTrue(combined.contains("First question."))
        assertTrue(combined.contains("Second question."))
    }

    @Test
    fun jsonlChatMlAndShareGptRecordsAreAcceptedInSourceOrder() {
        val raw = listOf(
            """{"messages":[{"role":"user","content":"First prompt."},{"role":"assistant","content":"First reply."}]}""",
            """{"conversations":[{"from":"human","value":"Second prompt."},{"from":"gpt","value":"Second reply."}]}"""
        ).joinToString("\n")

        val records = TextModalityProcessor.formatTrainingDatasetRecords(raw).orEmpty()
        assertEquals(2, records.size)
        assertTrue(records[0].contains("First prompt."))
        assertTrue(records[0].contains("First reply."))
        assertTrue(records[1].contains("Second prompt."))
        assertTrue(records[1].contains("Second reply."))

        val chunks = TextModalityProcessor.process(
            rawText = raw,
            documentTitle = "Conversation JSONL",
            chunkSize = 512,
            overlap = 0
        )
        assertTrue(chunks.isNotEmpty())
        assertTrue(chunks.all { it.modality == TextModality.CONVERSATION })
        val combined = chunks.joinToString("\n") { it.content }
        assertTrue(combined.contains("First prompt."))
        assertTrue(combined.contains("Second prompt."))
    }

    @Test
    fun explicitlyApprovedExchangeUsesTheSameConversationTrainingFormat() {
        val sample = TextModalityProcessor.formatConversationExchange(
            userMessage = "Can you help me?",
            assistantMessage = "Yes. What do you need?"
        )

        assertTrue(sample != null)
        val chunks = TextModalityProcessor.process(
            rawText = sample.orEmpty(),
            documentTitle = "Approved exchange",
            chunkSize = 512,
            overlap = 0
        )

        assertEquals(1, chunks.size)
        assertEquals(TextModality.CONVERSATION, chunks.single().modality)
        assertTrue(chunks.single().content.contains("${Tokenizer.USER_TOKEN} Can you help me?"))
        assertTrue(chunks.single().content.contains("${Tokenizer.ASSISTANT_TOKEN} Yes. What do you need?"))
        assertTrue(chunks.single().content.endsWith(Tokenizer.EOS_TOKEN))
        assertNull(
            TextModalityProcessor.formatConversationExchange(
                userMessage = " ",
                assistantMessage = "An answer without a prompt."
            )
        )
    }

    @Test
    fun conversationRecordWithoutAssistantTurnIsRejectedAsTrainingRecord() {
        val raw = """
            {
              "messages": [
                {"role": "system", "content": "System context."},
                {"role": "user", "content": "A question without an answer."}
              ]
            }
        """.trimIndent()

        assertNull(TextModalityProcessor.formatTrainingDatasetRecords(raw))
    }
}
