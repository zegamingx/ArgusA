package com.argus.agi.sensor

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONArray
import org.jsoup.Jsoup
import java.io.ByteArrayOutputStream
import java.io.InputStream
import java.net.URI
import java.net.URLDecoder
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.util.Locale
import java.util.concurrent.TimeUnit

data class SearchResultItem(
    val title: String,
    val snippet: String,
    val sourceUrl: String,
    val provider: String = "web"
)

/** Retrieves search metadata only; article bodies are handled by ArticleExtractor. */
class WebSearchEngine(
    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(12, TimeUnit.SECONDS)
        .readTimeout(15, TimeUnit.SECONDS)
        .callTimeout(20, TimeUnit.SECONDS)
        .followRedirects(true)
        .build()
) {
    companion object {
        private const val MAX_QUERY_LENGTH = 300
        private const val MAX_RESULTS = 8
        private const val MAX_WIKI_RESPONSE_BYTES = 512_000
        private const val MAX_DDG_RESPONSE_BYTES = 1_500_000
        private const val USER_AGENT =
            "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Mobile Safari/537.36"
    }

    suspend fun search(query: String): List<SearchResultItem> = withContext(Dispatchers.IO) {
        val normalizedQuery = query.trim().replace(Regex("\\s+"), " ").take(MAX_QUERY_LENGTH)
        if (normalizedQuery.isBlank()) return@withContext emptyList()

        // Search the open web first. Wikipedia is an enrichment/fallback source,
        // never the primary ranking signal for a general web-search request.
        val generalResults = runCatching {
            fetchDdgResults(normalizedQuery)
        }.onFailure { error ->
            android.util.Log.w("WebSearchEngine", "General web provider failed.", error)
        }.getOrDefault(emptyList())

        val supplementaryResults = if (generalResults.size < MAX_RESULTS) {
            runCatching {
                val encoded = URLEncoder.encode(normalizedQuery, StandardCharsets.UTF_8.name())
                val indonesianUrl =
                    "https://id.wikipedia.org/w/api.php?action=opensearch&search=$encoded&limit=5&namespace=0&format=json"
                val englishUrl =
                    "https://en.wikipedia.org/w/api.php?action=opensearch&search=$encoded&limit=5&namespace=0&format=json"

                val indonesian = fetchWikiResults(indonesianUrl)
                if (indonesian.isEmpty()) fetchWikiResults(englishUrl) else indonesian
            }.onFailure { error ->
                android.util.Log.w("WebSearchEngine", "Wikipedia enrichment provider failed.", error)
            }.getOrDefault(emptyList())
        } else {
            emptyList()
        }

        mergeSearchResults(
            primaryResults = generalResults,
            supplementaryResults = supplementaryResults,
            maxResults = MAX_RESULTS
        )
    }

    private fun fetchDdgResults(query: String): List<SearchResultItem> {
        val encoded = URLEncoder.encode(query, StandardCharsets.UTF_8.name())
        val request = Request.Builder()
            .url("https://html.duckduckgo.com/html/?q=$encoded")
            .header("User-Agent", USER_AGENT)
            .header("Accept", "text/html,application/xhtml+xml")
            .get()
            .build()

        return client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) return@use emptyList()

            val html = readBoundedResponseBody(
                input = response.body.byteStream(),
                contentLength = response.body.contentLength(),
                maxBytes = MAX_DDG_RESPONSE_BYTES
            )

            parseDdgHtml(html)
        }
    }

    private fun fetchWikiResults(url: String): List<SearchResultItem> {
        val request = Request.Builder()
            .url(url)
            .header("User-Agent", "ArgusAndroid/1.0 (on-device web retrieval)")
            .header("Accept", "application/json")
            .get()
            .build()

        return client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) return@use emptyList()

            val body = readBoundedResponseBody(
                input = response.body.byteStream(),
                contentLength = response.body.contentLength(),
                maxBytes = MAX_WIKI_RESPONSE_BYTES
            )
            if (body.isBlank()) return@use emptyList()

            val json = JSONArray(body)
            if (json.length() < 4) return@use emptyList()

            val titles = json.optJSONArray(1) ?: return@use emptyList()
            val descriptions = json.optJSONArray(2) ?: return@use emptyList()
            val urls = json.optJSONArray(3) ?: return@use emptyList()
            val output = ArrayList<SearchResultItem>(titles.length())

            for (index in 0 until titles.length()) {
                val title = titles.optString(index).trim()
                val description = descriptions.optString(index).trim()
                val sourceUrl = normalizeHttpUrl(urls.optString(index)) ?: continue
                if (title.isBlank()) continue

                output += SearchResultItem(
                    title = title,
                    snippet = description.ifBlank { "Wikipedia article. Open the source for its full content." },
                    sourceUrl = sourceUrl,
                    provider = "Wikipedia"
                )
            }
            output
        }
    }

    internal fun parseDdgHtml(html: String): List<SearchResultItem> {
        if (html.isBlank()) return emptyList()

        val document = Jsoup.parse(html, "https://html.duckduckgo.com/")
        val output = ArrayList<SearchResultItem>()

        document.select("a.result__a").forEach { anchor ->
            if (output.size >= MAX_RESULTS) return@forEach

            val title = anchor.text().trim()
            val sourceUrl = resolveDdgResultUrl(anchor.attr("href")) ?: return@forEach
            if (title.isBlank()) return@forEach

            val resultContainer = anchor.parents().firstOrNull { it.hasClass("result") }
            val snippet = resultContainer
                ?.selectFirst(".result__snippet")
                ?.text()
                ?.trim()
                .orEmpty()

            output += SearchResultItem(
                title = title,
                snippet = snippet.ifBlank { "Open the source page to read the article." },
                sourceUrl = sourceUrl,
                provider = "DuckDuckGo"
            )
        }

        return output.distinctBy { it.sourceUrl }
    }

    private fun resolveDdgResultUrl(rawHref: String): String? {
        if (rawHref.isBlank()) return null

        val resolved = runCatching {
            URI("https://html.duckduckgo.com/").resolve(rawHref)
        }.getOrNull() ?: return null

        val host = resolved.host?.lowercase(Locale.ROOT).orEmpty()
        val target = if (
            host == "duckduckgo.com" ||
            host == "www.duckduckgo.com" ||
            host == "html.duckduckgo.com"
        ) {
            val encodedTarget = resolved.rawQuery
                ?.split('&')
                ?.firstOrNull { it.substringBefore('=') == "uddg" }
                ?.substringAfter('=', missingDelimiterValue = "")
                .orEmpty()

            if (encodedTarget.isBlank()) return null
            runCatching {
                URLDecoder.decode(encodedTarget, StandardCharsets.UTF_8)
            }.getOrNull() ?: return null
        } else {
            resolved.toString()
        }

        return normalizeHttpUrl(target)
    }

    private fun normalizeHttpUrl(rawUrl: String): String? {
        val uri = runCatching { URI(rawUrl.trim()).normalize() }.getOrNull() ?: return null
        val scheme = uri.scheme?.lowercase(Locale.ROOT) ?: return null
        if (scheme != "http" && scheme != "https") return null

        val host = uri.host?.lowercase(Locale.ROOT) ?: return null
        if (
            host == "localhost" ||
            host.endsWith(".localhost") ||
            host.endsWith(".local") ||
            host.endsWith(".invalid")
        ) return null

        if (host == "duckduckgo.com" || host.endsWith(".duckduckgo.com")) return null
        if (uri.userInfo != null) return null

        return URI(
            scheme,
            null,
            host,
            uri.port,
            uri.path.ifBlank { "/" },
            uri.query,
            null
        ).toASCIIString()
    }
}

/**
 * Reads an HTTP response with a hard byte ceiling. Content-Length is only an early
 * rejection hint; the streaming counter remains authoritative for chunked responses.
 */
internal fun readBoundedResponseBody(
    input: InputStream,
    contentLength: Long,
    maxBytes: Int
): String? {
    require(maxBytes > 0) { "maxBytes must be positive." }
    if (contentLength > maxBytes.toLong()) return null

    val output = ByteArrayOutputStream()
    val buffer = ByteArray(8_192)
    var totalBytes = 0

    input.use { stream ->
        while (true) {
            val count = stream.read(buffer)
            if (count < 0) break
            totalBytes += count
            if (totalBytes > maxBytes) return null
            output.write(buffer, 0, count)
        }
    }

    return output.toString(StandardCharsets.UTF_8.name())
}


/**
 * Preserves the primary provider's ranking and only appends unique supplemental URLs.
 */
internal fun mergeSearchResults(
    primaryResults: List<SearchResultItem>,
    supplementaryResults: List<SearchResultItem>,
    maxResults: Int
): List<SearchResultItem> {
    require(maxResults > 0) { "maxResults must be positive." }
    val collected = LinkedHashMap<String, SearchResultItem>()
    primaryResults.forEach { item ->
        collected.putIfAbsent(item.sourceUrl, item)
    }
    supplementaryResults.forEach { item ->
        collected.putIfAbsent(item.sourceUrl, item)
    }
    return collected.values.take(maxResults)
}
