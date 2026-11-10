package com.argus.agi.sensor

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.pdmodel.PDPage
import com.tom_roush.pdfbox.pdmodel.PDPageContentStream
import com.tom_roush.pdfbox.pdmodel.font.PDType1Font
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class StructuredDocumentExtractorTest {

    @Test
    fun pdfExtractorReadsSelectableTextFromPdfPages() {
        val context: Context = ApplicationProvider.getApplicationContext()
        PDFBoxResourceLoader.init(context)

        val output = ByteArrayOutputStream()
        PDDocument().use { document ->
            val page = PDPage()
            document.addPage(page)

            PDPageContentStream(document, page).use { stream ->
                stream.beginText()
                stream.setFont(PDType1Font.HELVETICA, 12f)
                stream.newLineAtOffset(48f, 720f)
                stream.showText(
                    "Argus local document extraction preserves readable PDF text and provenance."
                )
                stream.endText()
            }

            document.save(output)
        }

        val extracted = StructuredDocumentExtractor.extractPdf(
            context = context,
            input = ByteArrayInputStream(output.toByteArray()),
            title = "research.pdf"
        )

        assertEquals("research.pdf", extracted.title)
        assertEquals("PDF", extracted.format)
        assertTrue(extracted.text.contains("Argus local document extraction"))
        assertTrue(extracted.characterCount > 40)
    }

    @Test
    fun docxExtractorReadsParagraphsAndTabsFromWordDocumentXml() {
        val documentXml = """
            <?xml version="1.0" encoding="UTF-8"?>
            <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
              <w:body>
                <w:p><w:r><w:t>DOCX heading</w:t></w:r></w:p>
                <w:p>
                  <w:r><w:t>Paragraph one contains meaningful training text.</w:t></w:r>
                  <w:r><w:tab/><w:t>Second column content.</w:t></w:r>
                </w:p>
                <w:p><w:r><w:t>Paragraph three completes the document with enough words for extraction.</w:t></w:r></w:p>
              </w:body>
            </w:document>
        """.trimIndent()

        val archive = zipOf(
            "word/document.xml" to documentXml,
            "docProps/core.xml" to "<coreProperties/>"
        )

        val extracted = StructuredDocumentExtractor.extractDocx(
            ByteArrayInputStream(archive),
            "manual.docx"
        )

        assertEquals("manual.docx", extracted.title)
        assertEquals("DOCX", extracted.format)
        assertTrue(extracted.text.contains("DOCX heading"))
        assertTrue(extracted.text.contains("Paragraph one"))
        assertTrue(extracted.text.contains("Second column content."))
        assertTrue(extracted.text.contains("Paragraph three"))
    }

    @Test
    fun epubExtractorFollowsSpineOrderAndExcludesNavigationDocument() {
        val container = """
            <?xml version="1.0" encoding="UTF-8"?>
            <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
              <rootfiles>
                <rootfile full-path="OPS/package.opf" media-type="application/oebps-package+xml"/>
              </rootfiles>
            </container>
        """.trimIndent()

        val packageDocument = """
            <?xml version="1.0" encoding="UTF-8"?>
            <package xmlns="http://www.idpf.org/2007/opf"
                     xmlns:dc="http://purl.org/dc/elements/1.1/"
                     version="3.0">
              <metadata><dc:title>Argus Field Guide</dc:title></metadata>
              <manifest>
                <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
                <item id="chapter-one" href="Text/chapter-one.xhtml" media-type="application/xhtml+xml"/>
                <item id="chapter-two" href="Text/chapter-two.xhtml" media-type="application/xhtml+xml"/>
              </manifest>
              <spine>
                <itemref idref="chapter-two"/>
                <itemref idref="chapter-one"/>
              </spine>
            </package>
        """.trimIndent()

        val chapterOne = """
            <html xmlns="http://www.w3.org/1999/xhtml">
              <body>
                <h1>Chapter One</h1>
                <p>First chapter content explains document ingestion and source provenance in detail.</p>
              </body>
            </html>
        """.trimIndent()

        val chapterTwo = """
            <html xmlns="http://www.w3.org/1999/xhtml">
              <body>
                <h1>Chapter Two</h1>
                <p>Second chapter content describes local next token learning and stable token identifiers.</p>
              </body>
            </html>
        """.trimIndent()

        val navigation = """
            <html xmlns="http://www.w3.org/1999/xhtml">
              <body><nav><p>NAVIGATION MUST NOT BE INCLUDED</p></nav></body>
            </html>
        """.trimIndent()

        val archive = zipOf(
            "META-INF/container.xml" to container,
            "OPS/package.opf" to packageDocument,
            "OPS/nav.xhtml" to navigation,
            "OPS/Text/chapter-one.xhtml" to chapterOne,
            "OPS/Text/chapter-two.xhtml" to chapterTwo
        )

        val extracted = StructuredDocumentExtractor.extractEpub(
            ByteArrayInputStream(archive),
            "fallback.epub"
        )

        assertEquals("Argus Field Guide", extracted.title)
        assertEquals("EPUB", extracted.format)
        assertTrue(extracted.text.indexOf("Chapter Two") < extracted.text.indexOf("Chapter One"))
        assertFalse(extracted.text.contains("NAVIGATION MUST NOT BE INCLUDED"))
    }

    @Test
    fun htmlExtractorRemovesNavigationScriptsAndFooter() {
        val html = """
            <!doctype html>
            <html lang="en">
              <body>
                <nav>Navigation must not be learned.</nav>
                <main>
                  <h1>Local AI</h1>
                  <p>This technical article contains enough body text to verify reliable HTML extraction. It describes source provenance, content structure, and local learning.</p>
                </main>
                <script>window.secret = "script text must not be learned";</script>
                <footer>Footer must not be learned.</footer>
              </body>
            </html>
        """.trimIndent()

        val extracted = StructuredDocumentExtractor.extractHtml(
            ByteArrayInputStream(html.toByteArray(Charsets.UTF_8)),
            "article.html"
        )

        assertEquals("HTML", extracted.format)
        assertTrue(extracted.text.contains("Local AI"))
        assertTrue(extracted.text.contains("technical article"))
        assertFalse(extracted.text.contains("Navigation must not be learned"))
        assertFalse(extracted.text.contains("script text must not be learned"))
        assertFalse(extracted.text.contains("Footer must not be learned"))
    }

    @Test
    fun docxExtractorRejectsArchivePathTraversal() {
        val archive = zipOf(
            "../outside.txt" to "This entry must not be accepted.",
            "word/document.xml" to """
                <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
                  <w:body><w:p><w:r><w:t>Valid enough document text content for the extractor.</w:t></w:r></w:p></w:body>
                </w:document>
            """.trimIndent()
        )

        val error = runCatching {
            StructuredDocumentExtractor.extractDocx(
                ByteArrayInputStream(archive),
                "unsafe.docx"
            )
        }.exceptionOrNull()

        assertTrue(error is DocumentTextExtractionException)
    }

    private fun zipOf(vararg entries: Pair<String, String>): ByteArray {
        val output = ByteArrayOutputStream()
        ZipOutputStream(output).use { zip ->
            entries.forEach { (name, content) ->
                zip.putNextEntry(ZipEntry(name))
                zip.write(content.toByteArray(Charsets.UTF_8))
                zip.closeEntry()
            }
        }
        return output.toByteArray()
    }
}
