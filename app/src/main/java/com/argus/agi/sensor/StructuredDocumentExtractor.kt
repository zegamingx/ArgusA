package com.argus.agi.sensor

import android.content.Context
import android.util.Xml
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.io.MemoryUsageSetting
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import org.jsoup.Jsoup
import org.jsoup.parser.Parser
import org.xmlpull.v1.XmlPullParser
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.FilterInputStream
import java.io.IOException
import java.io.InputStream
import java.io.Writer
import java.net.URI
import java.nio.charset.StandardCharsets
import java.util.Locale
import java.util.zip.ZipInputStream

data class ExtractedDocument(
    val title: String,
    val format: String,
    val text: String,
    val extractedAtEpochMs: Long = System.currentTimeMillis()
) {
    val characterCount: Int get() = text.length
}

class DocumentTextExtractionException(
    message: String,
    cause: Throwable? = null
) : IOException(message, cause)

/**
 * Local-only extractors for common knowledge-file formats.
 * Scanned PDFs are reported as unsupported; OCR is never silently substituted.
 */
object StructuredDocumentExtractor {
    private const val MAX_TEXT_CHARACTERS = 2_000_000
    private const val MAX_PDF_BYTES = 50L * 1024L * 1024L
    private const val MAX_PDF_PAGES = 500
    private const val MAX_ARCHIVE_ENTRIES = 1_000
    private const val MAX_ARCHIVE_EXPANDED_BYTES = 40L * 1024L * 1024L
    private const val MAX_ARCHIVE_ENTRY_BYTES = 12L * 1024L * 1024L
    private const val MAX_EPUB_STORED_BYTES = 24L * 1024L * 1024L
    private const val MIN_EXTRACTED_CHARACTERS = 40

    private val pdfInitLock = Any()

    @Volatile
    private var pdfBoxInitialized = false

    fun extractPdf(
        context: Context,
        input: InputStream,
        title: String,
        maxInputBytes: Long = MAX_PDF_BYTES,
        maxTextCharacters: Int = MAX_TEXT_CHARACTERS
    ): ExtractedDocument {
        require(maxInputBytes > 0L)
        require(maxTextCharacters >= MIN_EXTRACTED_CHARACTERS)
        initializePdfBox(context)
        val tempFile = File.createTempFile("argus-pdf-", ".pdf", context.cacheDir)

        try {
            copyPdfToTempFile(input, tempFile, maxInputBytes)

            PDDocument.load(tempFile, MemoryUsageSetting.setupTempFileOnly()).use { document ->
                if (document.isEncrypted) {
                    throw DocumentTextExtractionException(
                        "PDF dilindungi password. Buka proteksinya terlebih dahulu."
                    )
                }
                if (document.numberOfPages <= 0) {
                    throw DocumentTextExtractionException("PDF tidak memiliki halaman.")
                }
                if (document.numberOfPages > MAX_PDF_PAGES) {
                    throw DocumentTextExtractionException(
                        "PDF memiliki " + document.numberOfPages +
                            " halaman; batas ekstraksi adalah " + MAX_PDF_PAGES + "."
                    )
                }

                val writer = BoundedTextWriter(maxTextCharacters)
                PDFTextStripper().apply {
                    sortByPosition = true
                }.writeText(document, writer)

                val text = normalizeExtractedText(writer.toString(), maxTextCharacters)
                validateText(text, "PDF")
                return ExtractedDocument(title = title, format = "PDF", text = text)
            }
        } catch (error: DocumentTextExtractionException) {
            throw error
        } catch (error: Exception) {
            throw DocumentTextExtractionException(
                "PDF gagal diekstrak: " + (error.message ?: error.javaClass.simpleName),
                error
            )
        } finally {
            tempFile.delete()
        }
    }

    fun extractDocx(
        input: InputStream,
        title: String,
        maxEntryBytes: Long = MAX_ARCHIVE_ENTRY_BYTES,
        maxExpandedBytes: Long = MAX_ARCHIVE_EXPANDED_BYTES,
        maxTextCharacters: Int = MAX_TEXT_CHARACTERS
    ): ExtractedDocument {
        require(maxEntryBytes > 0L && maxExpandedBytes > 0L)
        require(maxTextCharacters >= MIN_EXTRACTED_CHARACTERS)
        val zip = ZipInputStream(BufferedInputStream(input))
        var entryCount = 0
        var expandedBytes = 0L
        var documentText = ""

        try {
            var entry = zip.nextEntry
            while (entry != null) {
                entryCount++
                if (entryCount > MAX_ARCHIVE_ENTRIES) {
                    throw DocumentTextExtractionException("Arsip DOCX berisi terlalu banyak entri.")
                }

                val entryName = normalizeArchivePath(entry.name)
                val entryStream = EntryBoundedInputStream(zip, maxEntryBytes)
                if (entryName == "word/document.xml") {
                    documentText = parseWordDocumentXml(entryStream, maxTextCharacters)
                }
                drainEntry(entryStream)

                expandedBytes += entryStream.bytesRead
                if (expandedBytes > maxExpandedBytes) {
                    throw DocumentTextExtractionException(
                        "Ukuran hasil ekstraksi DOCX melewati batas aman."
                    )
                }

                zip.closeEntry()
                entry = zip.nextEntry
            }
        } catch (error: DocumentTextExtractionException) {
            throw error
        } catch (error: Exception) {
            throw DocumentTextExtractionException(
                "DOCX gagal dibaca: " + (error.message ?: error.javaClass.simpleName),
                error
            )
        } finally {
            runCatching { zip.close() }
        }

        val normalized = normalizeExtractedText(documentText, maxTextCharacters)
        validateText(normalized, "DOCX")
        return ExtractedDocument(title = title, format = "DOCX", text = normalized)
    }

    fun extractEpub(
        input: InputStream,
        fallbackTitle: String,
        maxEntryBytes: Long = MAX_ARCHIVE_ENTRY_BYTES,
        maxExpandedBytes: Long = MAX_ARCHIVE_EXPANDED_BYTES,
        maxStoredTextBytes: Long = MAX_EPUB_STORED_BYTES,
        maxTextCharacters: Int = MAX_TEXT_CHARACTERS
    ): ExtractedDocument {
        require(maxEntryBytes > 0L && maxExpandedBytes > 0L && maxStoredTextBytes > 0L)
        require(maxTextCharacters >= MIN_EXTRACTED_CHARACTERS)
        val zip = ZipInputStream(BufferedInputStream(input))
        val storedEntries = LinkedHashMap<String, ByteArray>()
        var entryCount = 0
        var expandedBytes = 0L
        var storedBytes = 0L

        try {
            var entry = zip.nextEntry
            while (entry != null) {
                entryCount++
                if (entryCount > MAX_ARCHIVE_ENTRIES) {
                    throw DocumentTextExtractionException("Arsip EPUB berisi terlalu banyak entri.")
                }

                val name = normalizeArchivePath(entry.name)
                val entryStream = EntryBoundedInputStream(zip, maxEntryBytes)
                val isReadableText = name == "META-INF/container.xml" ||
                    name.endsWith(".opf", ignoreCase = true) ||
                    name.endsWith(".xhtml", ignoreCase = true) ||
                    name.endsWith(".html", ignoreCase = true) ||
                    name.endsWith(".htm", ignoreCase = true)

                if (isReadableText && !entry.isDirectory) {
                    val bytes = readEntryBytes(entryStream)
                    storedBytes += bytes.size
                    if (storedBytes > maxStoredTextBytes) {
                        throw DocumentTextExtractionException(
                            "Konten teks EPUB melewati batas penyimpanan ekstraksi."
                        )
                    }
                    storedEntries[name] = bytes
                } else {
                    drainEntry(entryStream)
                }

                expandedBytes += entryStream.bytesRead
                if (expandedBytes > maxExpandedBytes) {
                    throw DocumentTextExtractionException(
                        "Ukuran hasil ekstraksi EPUB melewati batas aman."
                    )
                }

                zip.closeEntry()
                entry = zip.nextEntry
            }
        } catch (error: DocumentTextExtractionException) {
            throw error
        } catch (error: Exception) {
            throw DocumentTextExtractionException(
                "EPUB gagal dibaca: " + (error.message ?: error.javaClass.simpleName),
                error
            )
        } finally {
            runCatching { zip.close() }
        }

        val containerBytes = storedEntries["META-INF/container.xml"]
            ?: throw DocumentTextExtractionException(
                "EPUB tidak memiliki META-INF/container.xml."
            )
        val containerDocument = Jsoup.parse(
            String(containerBytes, StandardCharsets.UTF_8),
            "",
            Parser.xmlParser()
        )
        val packagePath = containerDocument.selectFirst("rootfile[full-path]")
            ?.attr("full-path")
            ?.let(::normalizeArchivePath)
            ?: throw DocumentTextExtractionException(
                "Lokasi package document EPUB tidak ditemukan."
            )

        val packageBytes = storedEntries[packagePath]
            ?: throw DocumentTextExtractionException(
                "Package document EPUB tidak ditemukan: " + packagePath
            )
        val packageDocument = Jsoup.parse(
            String(packageBytes, StandardCharsets.UTF_8),
            "",
            Parser.xmlParser()
        )

        val manifest = packageDocument.select("manifest item")
            .mapNotNull { item ->
                val id = item.attr("id").trim()
                val href = item.attr("href").trim()
                val mediaType = item.attr("media-type").trim().lowercase(Locale.ROOT)
                val properties = item.attr("properties").split(Regex("\\s+")).toSet()

                if (
                    id.isBlank() ||
                    href.isBlank() ||
                    mediaType != "application/xhtml+xml" ||
                    "nav" in properties
                ) {
                    null
                } else {
                    id to href
                }
            }
            .toMap()

        val spineHrefs = packageDocument.select("spine itemref")
            .mapNotNull { itemRef ->
                manifest[itemRef.attr("idref").trim()]
            }

        if (spineHrefs.isEmpty()) {
            throw DocumentTextExtractionException(
                "EPUB tidak memiliki urutan bab XHTML yang dapat dibaca."
            )
        }

        val packageDirectory = packagePath.substringBeforeLast('/', "")
        val chapterTexts = ArrayList<String>()
        var totalChapterCharacters = 0L

        for (href in spineHrefs) {
            val chapterPath = resolveArchivePath(packageDirectory, href) ?: continue
            val chapterBytes = storedEntries[chapterPath] ?: continue
            val chapterText = extractHtmlText(
                String(chapterBytes, StandardCharsets.UTF_8),
                maxTextCharacters
            )
            if (chapterText.isBlank()) continue

            totalChapterCharacters += chapterText.length.toLong()
            if (totalChapterCharacters > maxTextCharacters.toLong()) {
                throw DocumentTextExtractionException(
                    "Teks EPUB melewati batas " + maxTextCharacters + " karakter."
                )
            }
            chapterTexts += chapterText
        }

        val combinedText = normalizeExtractedText(
            chapterTexts.joinToString("\n\n"),
            maxTextCharacters
        )
        validateText(combinedText, "EPUB")

        val epubTitle = packageDocument.getElementsByTag("dc:title")
            .firstOrNull()
            ?.text()
            ?.trim()
            ?.takeIf(String::isNotBlank)
            ?: fallbackTitle

        return ExtractedDocument(
            title = epubTitle,
            format = "EPUB",
            text = combinedText
        )
    }

    fun extractHtml(
        input: InputStream,
        title: String,
        maxInputBytes: Long = MAX_ARCHIVE_ENTRY_BYTES,
        maxTextCharacters: Int = MAX_TEXT_CHARACTERS
    ): ExtractedDocument {
        require(maxInputBytes > 0L)
        require(maxTextCharacters >= MIN_EXTRACTED_CHARACTERS)
        val bytes = readBounded(input, maxInputBytes)
        val text = normalizeExtractedText(
            extractHtmlText(String(bytes, StandardCharsets.UTF_8), maxTextCharacters),
            maxTextCharacters
        )
        validateText(text, "HTML")
        return ExtractedDocument(title = title, format = "HTML", text = text)
    }

    private fun initializePdfBox(context: Context) {
        if (pdfBoxInitialized) return

        synchronized(pdfInitLock) {
            if (!pdfBoxInitialized) {
                PDFBoxResourceLoader.init(context.applicationContext)
                pdfBoxInitialized = true
            }
        }
    }

    private fun copyPdfToTempFile(input: InputStream, destination: File, maximumBytes: Long) {
        FileOutputStream(destination).use { output ->
            val buffer = ByteArray(16 * 1024)
            var totalBytes = 0L

            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                totalBytes += count
                if (totalBytes > maximumBytes) {
                    throw DocumentTextExtractionException(
                        "PDF melewati batas ukuran " + (maximumBytes / 1024L / 1024L) + " MB."
                    )
                }
                output.write(buffer, 0, count)
            }
            output.flush()
        }
    }

    private fun parseWordDocumentXml(input: InputStream, maxTextCharacters: Int): String {
        val parser = Xml.newPullParser()
        parser.setFeature(XmlPullParser.FEATURE_PROCESS_NAMESPACES, true)
        parser.setFeature(XmlPullParser.FEATURE_PROCESS_DOCDECL, false)
        parser.setInput(input, "UTF-8")

        val elementStack = ArrayDeque<String>()
        val text = StringBuilder()

        while (parser.eventType != XmlPullParser.END_DOCUMENT) {
            when (parser.eventType) {
                XmlPullParser.START_TAG -> {
                    val name = parser.name.orEmpty()
                    if (name == "p" && text.isNotEmpty() && text.last() != '\n') {
                        text.append('\n')
                    }
                    if (name == "tab") text.append('\t')
                    if (name == "br" && text.isNotEmpty() && text.last() != '\n') {
                        text.append('\n')
                    }
                    elementStack.addLast(name)
                }

                XmlPullParser.TEXT -> {
                    if (elementStack.contains("t")) {
                        val nextText = parser.text.orEmpty()
                        if (text.length.toLong() + nextText.length.toLong() > maxTextCharacters) {
                            throw DocumentTextExtractionException(
                                "Teks DOCX melewati batas " +
                                    maxTextCharacters + " karakter."
                            )
                        }
                        text.append(nextText)
                    }
                }

                XmlPullParser.END_TAG -> {
                    if (parser.name == "p" && text.isNotEmpty() && text.last() != '\n') {
                        text.append('\n')
                    }
                    if (elementStack.isNotEmpty()) elementStack.removeLast()
                }
            }
            parser.next()
        }

        return text.toString()
    }

    private fun extractHtmlText(html: String, maxTextCharacters: Int): String {
        val document = Jsoup.parse(html)
        document.select(
            "script,style,noscript,svg,canvas,iframe,nav,footer,header,aside,form,button," +
                "[aria-hidden=true],.advertisement,.advert,.ads,.cookie,.cookie-banner"
        ).remove()

        val text = document.select("h1,h2,h3,h4,p,li,blockquote,pre")
            .map { it.text().trim() }
            .filter(String::isNotBlank)
            .distinct()
            .joinToString("\n\n")
            .ifBlank { document.body().text() }

        if (text.length > maxTextCharacters) {
            throw DocumentTextExtractionException(
                "Konten HTML melewati batas " + maxTextCharacters + " karakter."
            )
        }
        return text
    }

    private fun resolveArchivePath(packageDirectory: String, href: String): String? {
        if (href.isBlank()) return null

        val base = runCatching {
            URI("https://epub.invalid/" + packageDirectory + "/")
        }.getOrNull() ?: return null
        val resolved = runCatching { base.resolve(href) }.getOrNull() ?: return null

        if (resolved.host != "epub.invalid" || resolved.scheme != "https") return null
        return resolved.path
            .removePrefix("/")
            .let(::normalizeArchivePath)
            .takeIf(String::isNotBlank)
    }

    private fun normalizeArchivePath(path: String): String {
        val normalized = path.replace('\\', '/')
        val segments = normalized.split('/')
        if (segments.any { it == ".." }) {
            throw DocumentTextExtractionException(
                "Path traversal pada arsip dokumen ditolak."
            )
        }
        return normalized.removePrefix("./").removePrefix("/")
    }

    private fun readEntryBytes(input: InputStream): ByteArray {
        val output = ByteArrayOutputStream()
        val buffer = ByteArray(8_192)

        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            output.write(buffer, 0, count)
        }
        return output.toByteArray()
    }

    private fun readBounded(input: InputStream, maximumBytes: Long): ByteArray {
        val output = ByteArrayOutputStream()
        val buffer = ByteArray(8_192)
        var totalBytes = 0L

        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
            totalBytes += count
            if (totalBytes > maximumBytes) {
                throw DocumentTextExtractionException(
                    "File melewati batas ukuran ekstraksi."
                )
            }
            output.write(buffer, 0, count)
        }
        return output.toByteArray()
    }

    private fun drainEntry(input: InputStream) {
        val buffer = ByteArray(8_192)
        while (input.read(buffer) >= 0) {
            // Read through the current entry so the ZIP stream can advance safely.
        }
    }

    private fun normalizeExtractedText(
        rawText: String,
        maxTextCharacters: Int = MAX_TEXT_CHARACTERS
    ): String =
        rawText
            .replace('\u0000', ' ')
            .replace(Regex("[\\t\\u00A0 ]+"), " ")
            .replace(Regex(" *\\n *"), "\n")
            .replace(Regex("\\n{3,}"), "\n\n")
            .trim()
            .take(maxTextCharacters)

    private fun validateText(text: String, format: String) {
        if (text.length < MIN_EXTRACTED_CHARACTERS) {
            val detail = if (format == "PDF") {
                "PDF hasil scan memerlukan OCR, yang belum disediakan."
            } else {
                "Pastikan dokumen berisi teks, bukan hanya gambar atau halaman kosong."
            }
            throw DocumentTextExtractionException(
                "Tidak ditemukan teks yang cukup pada " + format + ". " + detail
            )
        }
    }

    private class EntryBoundedInputStream(
        input: InputStream,
        private val maxBytes: Long
    ) : FilterInputStream(input) {
        var bytesRead: Long = 0L
            private set

        override fun read(): Int {
            val value = super.read()
            if (value >= 0) countBytes(1L)
            return value
        }

        override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
            val count = super.read(buffer, offset, length)
            if (count > 0) countBytes(count.toLong())
            return count
        }

        private fun countBytes(count: Long) {
            bytesRead += count
            if (bytesRead > maxBytes) {
                throw DocumentTextExtractionException(
                    "Satu entri arsip melewati batas ukuran aman."
                )
            }
        }

        override fun close() {
            // Keep the enclosing ZipInputStream open between entries.
        }
    }

    private class BoundedTextWriter(
        private val maxCharacters: Int
    ) : Writer() {
        private val content = StringBuilder()

        override fun write(buffer: CharArray, offset: Int, length: Int) {
            require(offset >= 0 && length >= 0 && offset + length <= buffer.size)
            if (content.length + length > maxCharacters) {
                throw DocumentTextExtractionException(
                    "PDF menghasilkan teks melebihi batas " + maxCharacters + " karakter."
                )
            }
            content.append(buffer, offset, length)
        }

        override fun flush() = Unit

        override fun close() = Unit

        override fun toString(): String = content.toString()
    }
}
