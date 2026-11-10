package com.argus.agi.skill

import java.util.Locale

/**
 * Deterministic routing for explicit user requests to search the public web.
 *
 * This is deliberately narrow: ordinary questions continue through the local model
 * and its confidence-based retrieval policy. Search intent must be explicit.
 */
internal object WebSearchIntentRouter {

    private const val MAX_QUERY_LENGTH = 300

    private val commandPatterns = listOf(
        Regex("""^/web(?:\\s+|:\\s*)(.+)$""", RegexOption.IGNORE_CASE),
        Regex(
            """^(?:(?:tolong|please)\\s+)?(?:cari|carikan|telusuri|temukan)\\s+(?:(?:di|lewat|via)\\s+)?(?:web|internet|online)(?:\\s+(?:tentang|mengenai|soal|untuk))?\\s+(.+)$""",
            RegexOption.IGNORE_CASE
        ),
        Regex(
            """^(?:(?:tolong|please)\\s+)?(?:cari|carikan|telusuri|temukan)\\s+(.+?)\\s+(?:di|lewat|via)\\s+(?:web|internet|online)$""",
            RegexOption.IGNORE_CASE
        ),
        Regex(
            """^(?:please\\s+)?(?:search|look\\s+up|find)\\s+(?:the\\s+)?(?:web|internet|online)(?:\\s+(?:for|about))?\\s+(.+)$""",
            RegexOption.IGNORE_CASE
        )
    )

    fun extractQuery(input: String): String? {
        val normalizedInput = input.trim().replace(Regex("\\s+"), " ")
        if (normalizedInput.isBlank()) return null

        val extractedQuery = commandPatterns.firstNotNullOfOrNull { pattern ->
            pattern.matchEntire(normalizedInput)
                ?.groupValues
                ?.getOrNull(1)
                ?.trim()
        } ?: return null

        return extractedQuery
            .takeIf(String::isNotBlank)
            ?.take(MAX_QUERY_LENGTH)
    }
}
