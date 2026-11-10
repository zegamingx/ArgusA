package com.argus.agi.skill

import com.argus.agi.sensor.ExtractedWebArticle
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

class SkillRegistryTest {

    @Test
    fun rejectsDuplicateSkillIds() {
        val first = TestSkill("web_search")
        val second = TestSkill("WEB_SEARCH")

        assertThrows(IllegalArgumentException::class.java) {
            SkillRegistry(listOf(first, second))
        }
    }

    @Test
    fun rejectsUnknownSkillId() = runTest {
        val registry = SkillRegistry(emptyList())
        val error = runCatching { registry.execute("missing", "query") }.exceptionOrNull()

        assertTrue(error is IllegalArgumentException)
    }

    @Test
    fun articleExtractionSkillReturnsExtractedSourceMetadata() = runTest {
        val expected = ExtractedWebArticle(
            title = "Argus Article",
            description = "Technical article",
            sourceUrl = "https://example.org/source",
            canonicalUrl = "https://example.org/canonical",
            domain = "example.org",
            siteName = "Example Research",
            publishedAt = "2026-11-01",
            language = "en",
            text = "A sufficiently descriptive article body."
        )
        val skill = WebArticleExtractionSkill { url ->
            assertEquals("https://example.org/source", url)
            expected
        }
        val registry = SkillRegistry(listOf(skill))

        val output = registry.execute(
            WebArticleExtractionSkill.ID,
            " https://example.org/source "
        ) as WebArticleExtractionSkillOutput

        assertEquals(expected, output.article)
        assertEquals("https://example.org/source", output.sourceUrl)
    }

    @Test
    fun normalizesSkillIdBeforeExecution() = runTest {
        val registry = SkillRegistry(listOf(TestSkill("web_search")))
        val result = registry.execute(" WEB_SEARCH ", "  local AI  ")

        assertEquals("local AI", (result as TestOutput).input)
    }

    private class TestSkill(
        override val id: String
    ) : ArgusSkill {
        override val description: String = "Test skill"

        override suspend fun execute(input: String): SkillOutput =
            TestOutput(input.trim())
    }

    private data class TestOutput(val input: String) : SkillOutput
}
