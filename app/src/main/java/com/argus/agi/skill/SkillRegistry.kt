package com.argus.agi.skill

import com.argus.agi.sensor.ArticleExtractor
import com.argus.agi.sensor.ExtractedWebArticle
import com.argus.agi.sensor.SearchResultItem
import com.argus.agi.sensor.WebSearchEngine
import java.util.Locale

interface ArgusSkill {
    val id: String
    val description: String

    suspend fun execute(input: String): SkillOutput
}

interface SkillOutput

data class WebSearchSkillOutput(
    val query: String,
    val results: List<SearchResultItem>
) : SkillOutput

/**
 * Minimal deterministic skill registry.
 * Routing is explicit by skill ID; natural-language intent routing is a separate future layer.
 */
class SkillRegistry(skills: Collection<ArgusSkill>) {
    private val registeredSkills: Map<String, ArgusSkill> = buildMap {
        skills.forEach { skill ->
            val normalizedId = skill.id.trim().lowercase(Locale.ROOT)
            require(normalizedId.isNotBlank()) { "Skill ID must not be blank." }
            require(!containsKey(normalizedId)) { "Duplicate skill ID: $normalizedId" }
            put(normalizedId, skill)
        }
    }

    suspend fun execute(skillId: String, input: String): SkillOutput {
        val normalizedId = skillId.trim().lowercase(Locale.ROOT)
        require(normalizedId.isNotBlank()) { "Skill ID must not be blank." }

        val skill = registeredSkills[normalizedId]
            ?: throw IllegalArgumentException("Skill is not registered: $normalizedId")
        return skill.execute(input)
    }

    fun registeredSkillIds(): Set<String> = registeredSkills.keys
}

class WebSearchSkill(
    private val searchEngine: WebSearchEngine
) : ArgusSkill {
    companion object {
        const val ID = "web_search"
    }

    override val id: String = ID
    override val description: String =
        "Searches the public web and returns source titles, snippets, and destination URLs."

    override suspend fun execute(input: String): SkillOutput {
        val query = input.trim()
        require(query.isNotBlank()) { "Search query must not be blank." }
        return WebSearchSkillOutput(
            query = query,
            results = searchEngine.search(query)
        )
    }
}

data class WebArticleExtractionSkillOutput(
    val sourceUrl: String,
    val article: ExtractedWebArticle
) : SkillOutput

class WebArticleExtractionSkill(
    private val extractArticle: suspend (String) -> ExtractedWebArticle
) : ArgusSkill {
    constructor(articleExtractor: ArticleExtractor) : this(articleExtractor::extract)

    companion object {
        const val ID = "web_article_extract"
    }

    override val id: String = ID
    override val description: String =
        "Opens a selected HTTP(S) search result and extracts readable article text with source metadata."

    override suspend fun execute(input: String): SkillOutput {
        val sourceUrl = input.trim()
        require(sourceUrl.isNotBlank()) { "Article URL must not be blank." }

        return WebArticleExtractionSkillOutput(
            sourceUrl = sourceUrl,
            article = extractArticle(sourceUrl)
        )
    }
}
