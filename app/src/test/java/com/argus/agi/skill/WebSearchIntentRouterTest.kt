package com.argus.agi.skill

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class WebSearchIntentRouterTest {

    @Test
    fun extractsIndonesianPrefixSearchCommands() {
        assertEquals(
            "artikel terbaru tentang transformer on-device",
            WebSearchIntentRouter.extractQuery(
                "Tolong cari di web tentang artikel terbaru tentang transformer on-device"
            )
        )
        assertEquals(
            "kebijakan Android 16",
            WebSearchIntentRouter.extractQuery("cari web kebijakan Android 16")
        )
    }

    @Test
    fun extractsIndonesianSuffixSearchCommands() {
        assertEquals(
            "riset AI lokal",
            WebSearchIntentRouter.extractQuery("carikan riset AI lokal di internet")
        )
    }

    @Test
    fun extractsSlashAndEnglishSearchCommands() {
        assertEquals(
            "jadwal rilis Android",
            WebSearchIntentRouter.extractQuery("/web jadwal rilis Android")
        )
        assertEquals(
            "latest Android security bulletin",
            WebSearchIntentRouter.extractQuery("Search the web for latest Android security bulletin")
        )
    }

    @Test
    fun doesNotRouteOrdinaryQuestionsOrLocalSearchPhrases() {
        assertNull(WebSearchIntentRouter.extractQuery("Bagaimana cara kerja attention?"))
        assertNull(WebSearchIntentRouter.extractQuery("Cari file konfigurasi di aplikasi"))
        assertNull(WebSearchIntentRouter.extractQuery("cari di web"))
        assertNull(WebSearchIntentRouter.extractQuery("   "))
    }

    @Test
    fun capsExtractedQueryToSearchEngineLimit() {
        val input = "/web " + "a".repeat(400)

        assertEquals(
            300,
            WebSearchIntentRouter.extractQuery(input)?.length
        )
    }
}
