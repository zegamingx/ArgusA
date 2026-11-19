package com.argus.agi.skill

/**
 * Parses the model's explicit tool decision.
 *
 * A tool call is accepted only when it is the complete final answer. Mentioning the
 * syntax inside ordinary prose, quoting it, or expressing uncertainty never executes
 * a tool. The model must deliberately emit one well-formed web-search directive.
 */
internal object ModelToolCallParser {

    private const val MAX_QUERY_LENGTH = 300

    private val webSearchPattern = Regex(
        pattern = """^\\s*<WEB_SEARCH>\\s*([^<>]*?)\\s*</WEB_SEARCH>\\s*$""",
        options = setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL)
    )

    fun extractWebSearchQuery(modelAnswer: String): String? {
        val match = webSearchPattern.matchEntire(modelAnswer.trim()) ?: return null
        val query = match.groupValues[1].replace(Regex("\\s+"), " ").trim()

        if (query.isBlank() || query.length > MAX_QUERY_LENGTH) return null
        if (query.any { Character.isISOControl(it) }) return null

        return query
    }
}
