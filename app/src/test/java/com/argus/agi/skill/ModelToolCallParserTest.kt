package com.argus.agi.skill

import com.argus.agi.model.ReasoningOutputParser
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class ModelToolCallParserTest {

    @Test
    fun parsesModelIssuedWebSearchCallFromFinalAnswer() {
        assertEquals(
            "latest Android 37 platform changes",
            ModelToolCallParser.extractWebSearchQuery(
                "<WEB_SEARCH>  latest Android 37 platform changes  </WEB_SEARCH>"
            )
        )
    }

    @Test
    fun parsesToolCallAfterReasoningHasBeenSeparated() {
        val parsed = ReasoningOutputParser.parse(
            "<THINK>External verification is materially needed.</THINK>" +
                "<WEB_SEARCH>Android 37 release notes</WEB_SEARCH>"
        )

        assertEquals(
            "Android 37 release notes",
            ModelToolCallParser.extractWebSearchQuery(parsed.answer)
        )
    }

    @Test
    fun parsesToolCallWhenThinkStartWasAlreadyInTheGenerationPrompt() {
        val parsed = ReasoningOutputParser.parse(
            "External verification is needed.</THINK>" +
                "<WEB_SEARCH>Android 37 release notes</WEB_SEARCH>"
        )

        assertEquals("External verification is needed.", parsed.reasoning)
        assertEquals(
            "Android 37 release notes",
            ModelToolCallParser.extractWebSearchQuery(parsed.answer)
        )
    }

    @Test
    fun doesNotSearchWhenModelOnlyExpressesUncertainty() {
        assertNull(
            ModelToolCallParser.extractWebSearchQuery(
                "I do not know this yet because it is not part of my learned knowledge."
            )
        )
    }

    @Test
    fun doesNotExecuteToolSyntaxEmbeddedInNormalProse() {
        assertNull(
            ModelToolCallParser.extractWebSearchQuery(
                "The syntax is <WEB_SEARCH>Android version</WEB_SEARCH>, but this is only an explanation."
            )
        )
    }

    @Test
    fun rejectsEmptyMalformedNestedAndOversizedQueries() {
        assertNull(ModelToolCallParser.extractWebSearchQuery("<WEB_SEARCH> </WEB_SEARCH>"))
        assertNull(ModelToolCallParser.extractWebSearchQuery("<WEB_SEARCH>Android"))
        assertNull(
            ModelToolCallParser.extractWebSearchQuery(
                "<WEB_SEARCH>nested <WEB_SEARCH>query</WEB_SEARCH></WEB_SEARCH>"
            )
        )
        assertNull(
            ModelToolCallParser.extractWebSearchQuery(
                "<WEB_SEARCH>${"x".repeat(301)}</WEB_SEARCH>"
            )
        )
    }

    @Test
    fun normalizesWhitespaceAndAcceptsCaseInsensitiveDirective() {
        assertEquals(
            "Android 37 release notes",
            ModelToolCallParser.extractWebSearchQuery(
                "<web_search> Android 37\nrelease   notes </web_search>"
            )
        )
    }
}
