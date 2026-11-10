package com.argus.agi.sensor

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.jsoup.Jsoup
import org.jsoup.nodes.Document
import org.jsoup.nodes.Element
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.net.InetAddress
import java.net.URI
import java.nio.charset.StandardCharsets
import java.util.Locale
import java.util.concurrent.TimeUnit

data class ExtractedWebArticle(
    val title: String,
    val description: String,
    val sourceUrl: String,
    val canonicalUrl: String,
    val domain: String,
    val siteName: String,
    val publishedAt: String?,
    val language: String?,
    val text: String,
    val extractedAtEpochMs: Long = System.currentTimeMillis()
) {
    val characterCount: Int get() = text.length
}

class ArticleExtractionException(message: String, cause: Throwable? = null) :
    IOException(message, cause)

/** Fetches an explicitly selected page. It never enqueues content into model training. */
class ArticleExtractor(
    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(12, TimeUnit.SECONDS)
        .readTimeout(18, TimeUnit.SECONDS)
        .callTimeout(25, TimeUnit.SECONDS)
        .followRedirects(false)
        .followSslRedirects(false)
        .build()
) {
    companion object {
        private const val MAX_REDIRECTS = 4
        private const val MAX_BODY_BYTES = 2_500_000
        private const val USER_AGENT =
            "Mozilla/5.0 (Linux; Android 13) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Mobile Safari/537.36"
    }

    suspend fun extract(sourceUrl: String): ExtractedWebArticle = withContext(Dispatchers.IO) {
        var currentUri = validatePublicHttpUrl(sourceUrl)
            ?: throw ArticleExtractionException("URL sumber tidak valid atau tidak diizinkan.")

        repeat(MAX_REDIRECTS + 1) { redirectCount ->
            val request = Request.Builder()
                .url(currentUri.toString())
                .header("User-Agent", USER_AGENT)
                .header("Accept", "text/html,application/xhtml+xml;q=0.9")
                .header("Accept-Language", "id-ID,id;q=0.9,en;q=0.8")
                .get()
                .build()

            var redirectTarget: URI? = null
            var extracted: ExtractedWebArticle? = null

            try {
                client.newCall(request).execute().use { response ->
                    if (response.code in 300..399) {
                        if (redirectCount >= MAX_REDIRECTS) {
                            throw ArticleExtractionException("Terlalu banyak pengalihan halaman.")
                        }

                        val location = response.header("Location")
                            ?: throw ArticleExtractionException("Server mengirim redirect tanpa Location.")
                        val target = currentUri.resolve(location)
                        redirectTarget = validatePublicHttpUrl(target.toString())
                            ?: throw ArticleExtractionException("Redirect menuju URL yang tidak diizinkan.")
                    } else {
                        if (!response.isSuccessful) {
                            throw ArticleExtractionException("Server sumber mengembalikan HTTP ${response.code}.")
                        }

                        val body = response.body
                            ?: throw ArticleExtractionException("Halaman tidak memiliki body.")
                        val mediaType = body.contentType()?.let {
                            "${it.type.lowercase(Locale.ROOT)}/${it.subtype.lowercase(Locale.ROOT)}"
                        }.orEmpty()

                        if (mediaType != "text/html" && mediaType != "application/xhtml+xml") {
                            throw ArticleExtractionException("Sumber bukan halaman HTML ($mediaType).")
                        }

                        val bytes = body.byteStream().use { input ->
                            val output = ByteArrayOutputStream()
                            val buffer = ByteArray(8_192)
                            var total = 0

                            while (true) {
                                val count = input.read(buffer)
                                if (count < 0) break
                                total += count
                                if (total > MAX_BODY_BYTES) {
                                    throw ArticleExtractionException("Halaman melebihi batas 2,5 MB.")
                                }
                                output.write(buffer, 0, count)
                            }
                            output.toByteArray()
                        }

                        val charset = runCatching {
                            body.contentType()?.charset(StandardCharsets.UTF_8) ?: StandardCharsets.UTF_8
                        }.getOrDefault(StandardCharsets.UTF_8)
                        val html = String(bytes, charset)
                        extracted = parseArticleHtml(
                            html = html,
                            sourceUrl = sourceUrl,
                            pageUrl = response.request.url.toString()
                        )
                    }
                }
            } catch (error: ArticleExtractionException) {
                throw error
            } catch (error: Exception) {
                throw ArticleExtractionException("Gagal mengambil artikel: ${error.message ?: error.javaClass.simpleName}", error)
            }

            val next = redirectTarget
            if (next != null) {
                currentUri = next
            } else {
                return@withContext extracted
                    ?: throw ArticleExtractionException("Konten artikel tidak berhasil diekstrak.")
            }
        }

        throw ArticleExtractionException("Batas pengalihan halaman terlampaui.")
    }
}

internal fun parseArticleHtml(
    html: String,
    sourceUrl: String,
    pageUrl: String
): ExtractedWebArticle {
    val document = Jsoup.parse(html, pageUrl)
    document.select(
        "script,style,noscript,svg,canvas,iframe,nav,footer,header,aside,form,button," +
            "[aria-hidden=true],.advertisement,.advert,.ads,.cookie,.cookie-banner," +
            ".navigation,.navbar,.related,.social-share,.share-buttons"
    ).remove()

    val title = firstNonBlank(
        metaContent(document, "meta[property=og:title]"),
        metaContent(document, "meta[name=twitter:title]"),
        document.selectFirst("h1")?.text(),
        document.title()
    ) ?: "Untitled article"

    val description = firstNonBlank(
        metaContent(document, "meta[name=description]"),
        metaContent(document, "meta[property=og:description]"),
        metaContent(document, "meta[name=twitter:description]")
    ).orEmpty()

    val candidates = listOf(
        "[itemprop=articleBody]",
        "article",
        "main",
        ".article-body",
        ".post-content",
        ".entry-content",
        ".article-content",
        "#content"
    ).flatMap { selector ->
        document.select(selector).toList()
    }.distinct()

    val contentRoot = candidates
        .filter { element -> element.select("p,h1,h2,h3,li,blockquote,pre").size >= 2 }
        .maxByOrNull { element -> element.text().length }
        ?: document.body()

    val extractedText = contentRoot
        .select("h1,h2,h3,p,li,blockquote,pre")
        .map(Element::text)
        .map(String::trim)
        .filter(String::isNotBlank)
        .distinct()
        .joinToString("\n\n")
        .ifBlank { contentRoot.text() }
        .replace(Regex("[\\t\\u00A0 ]+"), " ")
        .replace(Regex("\\n{3,}"), "\n\n")
        .trim()
        .take(MAX_ARTICLE_CHARS)

    if (extractedText.length < 180) {
        throw ArticleExtractionException(
            "Teks artikel terlalu pendek (${extractedText.length} karakter). Halaman mungkin berupa landing page, login, atau konten yang diblokir."
        )
    }

    val canonicalFromPage = document.selectFirst("link[rel=canonical]")
        ?.absUrl("href")
        ?.takeIf(String::isNotBlank)
    val canonicalUrl = canonicalFromPage
        ?.let(::safeCanonicalUrl)
        ?: pageUrl

    val host = runCatching { URI(canonicalUrl).host }.getOrNull()
        ?: runCatching { URI(pageUrl).host }.getOrNull()
        ?: "unknown"
    val siteName = firstNonBlank(
        metaContent(document, "meta[property=og:site_name]"),
        metaContent(document, "meta[name=application-name]"),
        host
    ) ?: host

    val publishedAt = firstNonBlank(
        metaContent(document, "meta[property=article:published_time]"),
        metaContent(document, "meta[name=datePublished]"),
        metaContent(document, "meta[name=pubdate]"),
        metaContent(document, "meta[name=date]")
    )

    val language = document.selectFirst("html")
        ?.attr("lang")
        ?.trim()
        ?.takeIf(String::isNotBlank)

    return ExtractedWebArticle(
        title = title,
        description = description,
        sourceUrl = sourceUrl,
        canonicalUrl = canonicalUrl,
        domain = host,
        siteName = siteName,
        publishedAt = publishedAt,
        language = language,
        text = extractedText
    )
}

private const val MAX_ARTICLE_CHARS = 80_000

private fun metaContent(document: Document, selector: String): String? =
    document.selectFirst(selector)?.attr("content")?.trim()?.takeIf(String::isNotBlank)

private fun firstNonBlank(vararg values: String?): String? =
    values.firstOrNull { !it.isNullOrBlank() }?.trim()

private fun safeCanonicalUrl(rawUrl: String): String? {
    val uri = runCatching { URI(rawUrl.trim()).normalize() }.getOrNull() ?: return null
    val scheme = uri.scheme?.lowercase(Locale.ROOT) ?: return null
    if (scheme != "http" && scheme != "https") return null

    val host = uri.host?.lowercase(Locale.ROOT) ?: return null
    if (uri.userInfo != null) return null
    if (
        host == "localhost" ||
        host.endsWith(".localhost") ||
        host.endsWith(".local") ||
        host.endsWith(".invalid")
    ) return null

    val isIpLiteral = host.matches(Regex("^[0-9.]+$")) || host.contains(':')
    if (isIpLiteral) {
        val address = runCatching {
            InetAddress.getByName(host.removePrefix("[").removeSuffix("]"))
        }.getOrNull() ?: return null
        if (
            address.isAnyLocalAddress ||
            address.isLoopbackAddress ||
            address.isLinkLocalAddress ||
            address.isSiteLocalAddress ||
            address.isMulticastAddress
        ) return null
    }

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

private fun validatePublicHttpUrl(rawUrl: String): URI? {
    val uri = runCatching { URI(rawUrl.trim()).normalize() }.getOrNull() ?: return null
    val scheme = uri.scheme?.lowercase(Locale.ROOT) ?: return null
    if (scheme != "http" && scheme != "https") return null

    val host = uri.host?.lowercase(Locale.ROOT) ?: return null
    if (uri.userInfo != null) return null
    if (
        host == "localhost" ||
        host.endsWith(".localhost") ||
        host.endsWith(".local") ||
        host.endsWith(".invalid")
    ) return null

    val lookupHost = host.removePrefix("[").removeSuffix("]")
    val resolvedAddresses = runCatching { InetAddress.getAllByName(lookupHost) }.getOrNull()
        ?: return null
    if (resolvedAddresses.isEmpty()) return null

    if (resolvedAddresses.any { address ->
            address.isAnyLocalAddress ||
                address.isLoopbackAddress ||
                address.isLinkLocalAddress ||
                address.isSiteLocalAddress ||
                address.isMulticastAddress
        }
    ) return null

    return uri
}
