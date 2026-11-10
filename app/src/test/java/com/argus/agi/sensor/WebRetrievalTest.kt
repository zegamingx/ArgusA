package com.argus.agi.sensor

import java.io.ByteArrayInputStream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class WebRetrievalTest {

    @Test
    fun boundedResponseReaderRejectsOversizedContentLengthBeforeReading() {
        val input = ByteArrayInputStream("oversized".toByteArray())

        val result = readBoundedResponseBody(
            input = input,
            contentLength = 100L,
            maxBytes = 8
        )

        assertNull(result)
    }

    @Test
    fun boundedResponseReaderRejectsChunkedBodyThatExceedsLimit() {
        val input = ByteArrayInputStream("123456789".toByteArray())

        val result = readBoundedResponseBody(
            input = input,
            contentLength = -1L,
            maxBytes = 8
        )

        assertNull(result)
    }

    @Test
    fun boundedResponseReaderAcceptsBodyWithinLimit() {
        val input = ByteArrayInputStream("Argus search".toByteArray())

        val result = readBoundedResponseBody(
            input = input,
            contentLength = 12L,
            maxBytes = 12
        )

        assertEquals("Argus search", result)
    }

    @Test
    fun duckDuckGoParserUsesActualResultHrefAndTitle() {
        val html = """
            <div class="result results_links">
                <h2 class="result__title">
                    <a class="result__a" href="https://example.org/research/on-device-ai">
                        On-device AI research
                    </a>
                </h2>
                <a class="result__snippet">A detailed overview of local neural inference and training.</a>
                <a class="result__url">example.org › display-only-path</a>
            </div>
        """.trimIndent()

        val results = WebSearchEngine().parseDdgHtml(html)

        assertEquals(1, results.size)
        assertEquals("On-device AI research", results.single().title)
        assertEquals("https://example.org/research/on-device-ai", results.single().sourceUrl)
        assertTrue(results.single().snippet.contains("local neural inference"))
    }

    @Test
    fun duckDuckGoParserUnwrapsRedirectAndRejectsNonHttpLinks() {
        val html = """
            <div class="result">
                <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Farticle">Article title</a>
                <a class="result__snippet">Article snippet content with enough descriptive words.</a>
            </div>
            <div class="result">
                <a class="result__a" href="javascript:alert(1)">Unsafe link</a>
                <a class="result__snippet">This result must be discarded.</a>
            </div>
        """.trimIndent()

        val results = WebSearchEngine().parseDdgHtml(html)

        assertEquals(1, results.size)
        assertEquals("https://example.org/article", results.single().sourceUrl)
    }

    @Test
    fun articleParserRejectsPrivateCanonicalMetadataWithoutDnsLookup() {
        val paragraphs = (1..8).joinToString("\n") {
            "<p>Paragraph $it contains enough readable article text to pass extraction validation without network access.</p>"
        }
        val html = """
            <html>
              <head>
                <title>Local Canonical Test</title>
                <link rel="canonical" href="http://127.0.0.1/private-path">
              </head>
              <body><article>$paragraphs</article></body>
            </html>
        """.trimIndent()

        val article = parseArticleHtml(
            html = html,
            sourceUrl = "https://example.org/source",
            pageUrl = "https://example.org/final"
        )

        assertEquals("https://example.org/final", article.canonicalUrl)
    }

    @Test
    fun articleParserExtractsReadableBodyAndCanonicalMetadata() {
        val paragraphs = (1..8).joinToString("\n") {
            "<p>Paragraph $it explains how a local neural system processes language data on a personal device. " +
                "The training pipeline should preserve document provenance and separate source review from optimization.</p>"
        }
        val html = """
            <!doctype html>
            <html lang="id">
              <head>
                <title>Fallback title</title>
                <meta property="og:title" content="Artikel AI Lokal">
                <meta property="og:site_name" content="Example Research">
                <meta name="description" content="A technical article about local AI.">
                <meta property="article:published_time" content="2026-10-20">
                <link rel="canonical" href="https://example.org/canonical/article">
              </head>
              <body>
                <nav>Navigation should not be included</nav>
                <main><article><h1>Artikel AI Lokal</h1>$paragraphs</article></main>
                <footer>Footer should not be included</footer>
              </body>
            </html>
        """.trimIndent()

        val article = parseArticleHtml(
            html = html,
            sourceUrl = "https://example.org/redirected",
            pageUrl = "https://example.org/final"
        )

        assertEquals("Artikel AI Lokal", article.title)
        assertEquals("https://example.org/canonical/article", article.canonicalUrl)
        assertEquals("Example Research", article.siteName)
        assertEquals("2026-10-20", article.publishedAt)
        assertEquals("id", article.language)
        assertTrue(article.text.length >= 180)
        assertTrue(!article.text.contains("Navigation should not be included"))
        assertTrue(!article.text.contains("Footer should not be included"))
    }
}
