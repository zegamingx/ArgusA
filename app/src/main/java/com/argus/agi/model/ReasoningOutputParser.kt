package com.argus.agi.model

data class ParsedReasoningOutput(
    val answer: String,
    val reasoning: String?
)

/**
 * Separates a generated reasoning span from the user-facing final answer.
 *
 * A response is split only when both boundary markers exist and a non-blank final
 * answer follows. Incomplete generations are returned unchanged instead of silently
 * discarding partially generated content.
 */
object ReasoningOutputParser {

    fun parse(generatedText: String): ParsedReasoningOutput {
        val source = generatedText.trim()
        if (source.isEmpty()) return ParsedReasoningOutput(answer = "", reasoning = null)

        val reasoningStart = source.indexOf(Tokenizer.THINK_START)
        if (reasoningStart < 0) {
            // In autoregressive reasoning mode, THINK_START is already part of the
            // prompt and therefore is not repeated in the generated token stream.
            // A generated THINK_END still closes that implicit reasoning prefix.
            val implicitReasoningEnd = source.indexOf(Tokenizer.THINK_END)
            if (implicitReasoningEnd < 0) {
                return ParsedReasoningOutput(answer = source, reasoning = null)
            }

            val reasoningBody = source.substring(0, implicitReasoningEnd).trim()
            val trailingText = source.substring(
                implicitReasoningEnd + Tokenizer.THINK_END.length
            ).trim()
            val answer = trailingText
                .removePrefix(Tokenizer.ASSISTANT_TOKEN)
                .trim()
                .removeSuffix(Tokenizer.EOS_TOKEN)
                .trim()

            if (answer.isBlank()) {
                return ParsedReasoningOutput(answer = source, reasoning = null)
            }
            return ParsedReasoningOutput(
                answer = answer,
                reasoning = reasoningBody.takeIf(String::isNotBlank)
            )
        }

        val reasoningContentStart = reasoningStart + Tokenizer.THINK_START.length
        val reasoningEnd = source.indexOf(Tokenizer.THINK_END, reasoningContentStart)
        if (reasoningEnd < reasoningContentStart) {
            return ParsedReasoningOutput(answer = source, reasoning = null)
        }

        val reasoningBody = source.substring(reasoningContentStart, reasoningEnd).trim()
        val precedingText = source.substring(0, reasoningStart).trim()
        val trailingText = source.substring(reasoningEnd + Tokenizer.THINK_END.length).trim()
        val answer = trailingText
            .removePrefix(Tokenizer.ASSISTANT_TOKEN)
            .trim()
            .removeSuffix(Tokenizer.EOS_TOKEN)
            .trim()

        if (reasoningBody.isBlank() || answer.isBlank()) {
            return ParsedReasoningOutput(answer = source, reasoning = null)
        }

        val reasoning = listOf(precedingText, reasoningBody)
            .filter { it.isNotBlank() }
            .joinToString("\n")
        return ParsedReasoningOutput(answer = answer, reasoning = reasoning)
    }
}
