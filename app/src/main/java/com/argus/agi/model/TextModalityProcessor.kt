package com.argus.agi.model

import java.util.regex.Pattern
import org.json.JSONArray
import org.json.JSONObject

enum class TextModality(val displayName: String, val description: String) {
    CONVERSATION("Percakapan", "Dialog multi-turn dengan token <USER> dan <ASSISTANT>"),
    QA_PAIR("Tanya Jawab", "Pasangan Pertanyaan & Jawaban terstruktur"),
    REASONING("Reasoning Training", "Instruksi, jejak penalaran terstruktur, dan jawaban akhir"),
    ARTICLE_NEWS("Artikel & Berita", "Teks ekspositori, tajuk berita, dan paragraf analitis"),
    STORY_NARRATIVE("Cerita & Narasi", "Prosa naratif, kutipan dialog sastra, dan alur cerita"),
    STRUCTURED_DATA("Data Terstruktur", "Daftar butir, key-value, dan ringkasan teknis"),
    GENERAL_PROSE("Prosa Umum", "Teks bahasa alami umum")
}

data class ProcessedChunk(
    val content: String,
    val modality: TextModality,
    val segmentTitle: String,
    val tokenHint: String? = null
)

/**
 * Pemroses dan auditor format teks cerdas untuk sistem Tokenizer & Training Argus.
 * Menjamin teks percakapan, tanya jawab, artikel/berita, cerita, dan format lainnya
 * dipersiapkan dengan struktur token optimal sebelum masuk ke Neural Core.
 */
object TextModalityProcessor {

    private enum class TurnRole { USER, ASSISTANT }

    // Regex pengenal giliran percakapan.
    // Kolon diwajibkan untuk kata biasa (User:, Pengguna:, Bot:, ...) agar tidak
    // salah menangkap prosa seperti "User experience is critical".
    // Token <USER>/<ASSISTANT> boleh tanpa kolon karena merupakan marker kanonikal.
    private val USER_TURN_PATTERN = Pattern.compile(
        "^\\s*(?:<USER>\\s*:?|User\\s*:|Pengguna\\s*:|Human\\s*:|P\\s*:)\\s*",
        Pattern.CASE_INSENSITIVE
    )
    private val ASSISTANT_TURN_PATTERN = Pattern.compile(
        "^\\s*(?:<ASSISTANT>\\s*:?|Assistant\\s*:|Asisten\\s*:|Argus\\s*:|A\\.R\\.G\\.U\\.S\\s*:|Bot\\s*:|J\\s*:)\\s*",
        Pattern.CASE_INSENSITIVE
    )

    // Regex format Tanya Jawab terpisah (termasuk Pertanyaan/Jawaban yang kini
    // TIDAK lagi berada di turn-pattern agar dapat terdeteksi sebagai QA_PAIR).
    private val QA_QUESTION_PATTERN = Pattern.compile(
        "^\\s*(?:Tanya|Pertanyaan|Q|Question)\\s*:\\s*(.+)$",
        Pattern.CASE_INSENSITIVE
    )
    private val QA_ANSWER_PATTERN = Pattern.compile(
        "^\\s*(?:Jawab|Jawaban|A|Answer)\\s*:\\s*(.+)$",
        Pattern.CASE_INSENSITIVE
    )

    private val BULLET_LINE_REGEX = Regex("^[\\-\\*\\+•●▪◦‣⁃]\\s+.*")
    private val NUMBERED_LINE_REGEX = Regex("^\\d+[.)]\\s+.*")

    /**
     * Normalisasi tipografi global:
     * - Curly quotes (“ ” ‘ ’ « »)  → ASCII (" ')
     * - Em/en dash (— –)            → hyphen (-)
     * - Ellipsis (…)                → tiga titik (...)
     * - Line/paragraph separator    → newline
     * - Spasi tipografis & zero-width → spasi tunggal / hapus
     * - Soft hyphen (U+00AD)        → hapus
     * - Ligatur (ﬁ ﬂ ﬀ ﬃ ﬄ)         → ekuivalen ASCII
     * - Backtick (`) TIDAK diubah, karena sintaksis markdown/code
     */
    fun normalizeTypography(input: String): String {
        if (input.isEmpty()) return ""
        val sb = StringBuilder(input.length)
        var i = 0
        while (i < input.length) {
            when (val c = input[i]) {
                '“', '”', '«', '»' -> sb.append('"')
                '‘', '’' -> sb.append('\'')
                '—', '–' -> sb.append('-')
                '…' -> sb.append("...")
                '\u2028', '\u2029' -> sb.append('\n')
                '\u00A0', '\u2002', '\u2003', '\u2007', '\u202F' -> sb.append(' ')
                '\u200B', '\uFEFF', '\u00AD' -> { /* skip zero-width & soft hyphen */ }
                'ﬁ' -> sb.append("fi")
                'ﬂ' -> sb.append("fl")
                'ﬀ' -> sb.append("ff")
                'ﬃ' -> sb.append("ffi")
                'ﬄ' -> sb.append("ffl")
                '\r' -> {
                    if (i + 1 < input.length && input[i + 1] == '\n') i++
                    sb.append('\n')
                }
                else -> sb.append(c)
            }
            i++
        }
        return sb.toString().trim()
    }

    /**
     * Canonicalizes a pretty-printed training object or an array of JSONL-style
     * records. Other JSON documents are left untouched.
     */
    private fun normalizeReasoningDatasetInput(text: String): String {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) return text

        if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
            val array = runCatching { JSONArray(trimmed) }.getOrNull()
            if (array != null && array.length() > 0) {
                val records = (0 until array.length()).mapNotNull { array.optJSONObject(it) }
                if (
                    records.size == array.length() &&
                    records.isNotEmpty() &&
                    records.all(::isTrainingRecord)
                ) {
                    return records.joinToString("\n") { it.toString() }
                }
            }
        }

        if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
            val record = runCatching { JSONObject(trimmed) }.getOrNull()
            if (record != null && isTrainingRecord(record)) return record.toString()
        }
        return text
    }

        /**
     * Returns the exclusive end offset of one complete JSON object or array.
     * Braces inside quoted strings and escaped quotes do not affect nesting.
     */
    internal fun findCompleteJsonValueEnd(text: String, startIndex: Int): Int? {
        if (startIndex !in text.indices) return null
        val firstCharacter = text[startIndex]
        if (firstCharacter != '{' && firstCharacter != '[') return null

        val nesting = ArrayList<Char>()
        var insideString = false
        var escaped = false

        for (index in startIndex until text.length) {
            val character = text[index]
            if (insideString) {
                when {
                    escaped -> escaped = false
                    character == '\\' -> escaped = true
                    character == '"' -> insideString = false
                }
                continue
            }

            when (character) {
                '"' -> insideString = true
                '{', '[' -> nesting.add(character)
                '}', ']' -> {
                    val expectedOpening = if (character == '}') '{' else '['
                    if (nesting.isEmpty() || nesting.removeAt(nesting.lastIndex) != expectedOpening) {
                        return null
                    }
                    if (nesting.isEmpty()) return index + 1
                }
            }
        }
        return null
    }

    /**
     * Parses JSONL records, a pretty-printed object, or a JSON array of training
     * records. Returns null unless every non-empty record is an instruction/answer
     * example, so general JSON documents continue through the regular text path.
     */
    internal fun formatTrainingDatasetRecords(rawText: String): List<String>? {
        val normalized = normalizeReasoningDatasetInput(normalizeTypography(rawText))
        val serializedRecords = normalized.lineSequence()
            .map { it.trim() }
            .filter { it.isNotBlank() }
            .toList()
        if (serializedRecords.isEmpty()) return null

        val parsedRecords = serializedRecords.map { serialized ->
            runCatching { JSONObject(serialized) }.getOrNull()
        }
        if (parsedRecords.any { it == null }) return null

        val objects = parsedRecords.filterNotNull()
        if (objects.isEmpty() || !objects.all(::isTrainingRecord)) return null

        return objects.map { record ->
            formatStructuredLineIfJson(record.toString())
        }
    }

private fun firstNonBlankText(obj: JSONObject, keys: List<String>): String {
        for (key in keys) {
            val value = obj.opt(key)
            if (value is String && value.isNotBlank()) return value.trim()
        }
        return ""
    }

    private fun isTrainingRecord(obj: JSONObject): Boolean {
        val hasPrompt = firstNonBlankText(obj, listOf("instruction", "prompt", "question")).isNotBlank()
        val hasAnswer = firstNonBlankText(obj, listOf("response", "completion", "answer")).isNotBlank()
        return hasPrompt && hasAnswer
    }

    private fun readReasoningStep(value: Any?): String {
        return when (value) {
            is String -> value.trim()
            is JSONObject -> firstNonBlankText(
                value,
                listOf("step", "text", "content", "reasoning", "explanation")
            )
            else -> ""
        }
    }

    private fun normalizeReasoningValue(raw: Any?): String {
        return when (raw) {
            is String -> raw.trim()
            is JSONArray -> {
                val steps = mutableListOf<String>()
                for (index in 0 until raw.length()) {
                    val step = readReasoningStep(raw.opt(index))
                    if (step.isNotBlank()) steps += "Step ${steps.size + 1}: $step"
                }
                steps.joinToString("\n")
            }
            is JSONObject -> readReasoningStep(raw)
            else -> ""
        }
    }

    /**
     * Accepts a plain reasoning string, an array of strings, or an array of
     * objects containing step/text/content/reasoning/explanation.
     * An empty reasoning field falls back to the alternative steps field.
     */
    private fun extractReasoningText(obj: JSONObject): String? {
        val reasoning = normalizeReasoningValue(obj.opt("reasoning"))
        val resolved = reasoning.ifBlank {
            normalizeReasoningValue(obj.opt("steps"))
        }
        return resolved.takeIf { it.isNotBlank() }
    }

    private fun formatInstructionResponse(
        instruction: String,
        input: String,
        reasoning: String?,
        response: String
    ): String {
        val prompt = listOf(
            instruction.trim(),
            input.trim().takeIf { it.isNotBlank() }?.let { "Input: $it" }
        ).filterNotNull().filter { it.isNotBlank() }.joinToString("\n")

        val reasoningBlock = reasoning?.trim()?.takeIf { it.isNotBlank() }
        return if (reasoningBlock == null) {
            "${Tokenizer.USER_TOKEN} $prompt\n${Tokenizer.ASSISTANT_TOKEN} $response ${Tokenizer.EOS_TOKEN}"
        } else {
            "${Tokenizer.USER_TOKEN} $prompt\n" +
                "${Tokenizer.ASSISTANT_TOKEN}${Tokenizer.THINK_START}\n" +
                "$reasoningBlock\n" +
                "${Tokenizer.THINK_END}\n" +
                "${Tokenizer.ASSISTANT_TOKEN} $response ${Tokenizer.EOS_TOKEN}"
        }
    }

    /**
     * Mendeteksi modalitas konten teks.
     */
    fun detectModality(text: String): TextModality {
        val normalizedInput = normalizeReasoningDatasetInput(text)
        val trimmed = normalizedInput.trim()
        if (trimmed.isEmpty()) return TextModality.GENERAL_PROSE

        val lines = trimmed.lines().map { it.trim() }.filter { it.isNotEmpty() }
        if (lines.isEmpty()) return TextModality.GENERAL_PROSE

        val containsReasoningMarkers =
            trimmed.contains(Tokenizer.THINK_START) &&
                trimmed.contains(Tokenizer.THINK_END)
        val containsReasoningRecords = lines.any { line ->
            runCatching { JSONObject(line) }
                .getOrNull()
                ?.let { record ->
                    isTrainingRecord(record) && !extractReasoningText(record).isNullOrBlank()
                } == true
        }
        if (containsReasoningMarkers || containsReasoningRecords) {
            return TextModality.REASONING
        }

        // 1. Percakapan
        var userTurnCount = 0
        var assistantTurnCount = 0
        for (line in lines) {
            if (USER_TURN_PATTERN.matcher(line).find()) userTurnCount++
            if (ASSISTANT_TURN_PATTERN.matcher(line).find()) assistantTurnCount++
        }
        // Deteksi literal token harus menyertakan KEDUANYA untuk menghindari
        // false-positive pada dokumentasi yang hanya menyebut salah satunya.
        val hasBothTokens = trimmed.contains(Tokenizer.USER_TOKEN) &&
                trimmed.contains(Tokenizer.ASSISTANT_TOKEN)
        if ((userTurnCount >= 1 && assistantTurnCount >= 1) || hasBothTokens) {
            return TextModality.CONVERSATION
        }

        // 2. Tanya Jawab
        var qCount = 0
        var aCount = 0
        for (line in lines) {
            if (QA_QUESTION_PATTERN.matcher(line).find()) qCount++
            if (QA_ANSWER_PATTERN.matcher(line).find()) aCount++
        }
        if (qCount >= 1 && aCount >= 1) return TextModality.QA_PAIR

        // 3. Cerita & Narasi
        val quoteCount = trimmed.count { it == '"' }
        val narrativeKeywords = listOf(
            "berkata", "ujarnya", "sahut", "jawabnya", "bisiknya", "gumam",
            "langkah", "malam itu", "tiba-tiba", "pada suatu", "kerajaan", "hutan",
            "he said", "she said", "replied", "whispered", "suddenly", "once upon"
        )
        val hasNarrativeTags = narrativeKeywords.any { trimmed.contains(it, ignoreCase = true) }
        if (quoteCount >= 4 && hasNarrativeTags) return TextModality.STORY_NARRATIVE

        // 4. Data Terstruktur (bullet, numbered, JSONL)
        val bulletCount = lines.count { BULLET_LINE_REGEX.matches(it) || NUMBERED_LINE_REGEX.matches(it) }
        val jsonCount = lines.count { it.startsWith("{") && it.endsWith("}") }
        if ((bulletCount + jsonCount) >= (lines.size + 1) / 2) return TextModality.STRUCTURED_DATA

        // 5. Artikel & Berita
        val newsKeywords = listOf(
            "menurut", "dilansir", "berdasarkan", "pemerintah", "presiden", "menteri",
            "melaporkan", "resmi", "perusahaan", "penelitian", "studi", "mengatakan bahwa",
            "reported", "according to", "spokesperson", "statement", "officials"
        )
        val hasNewsKeywords = newsKeywords.any { trimmed.contains(it, ignoreCase = true) }
        if (hasNewsKeywords && lines.size >= 3) return TextModality.ARTICLE_NEWS

        return TextModality.GENERAL_PROSE
    }

    /**
     * Menyiapkan dan membagi teks menjadi potongan chunk training sesuai modalitas.
     * Parameter divalidasi agar tidak terjadi infinite loop saat chunkSize/overlap tidak masuk akal.
     */
    fun process(
        rawText: String,
        documentTitle: String = "Dokumen",
        chunkSize: Int = 380,
        overlap: Int = 50
    ): List<ProcessedChunk> {
        require(chunkSize > 0) { "chunkSize harus > 0, diterima: $chunkSize" }
        require(overlap >= 0) { "overlap tidak boleh negatif, diterima: $overlap" }
        require(overlap < chunkSize) { "overlap ($overlap) harus < chunkSize ($chunkSize)" }

        val normalized = normalizeReasoningDatasetInput(normalizeTypography(rawText))
        if (normalized.isBlank()) return emptyList()

        val modality = detectModality(normalized)

        return when (modality) {
            TextModality.CONVERSATION -> processConversation(normalized, documentTitle, chunkSize)
            TextModality.QA_PAIR -> processQAPairs(normalized, documentTitle, chunkSize)
            TextModality.REASONING -> processReasoningDataset(normalized, documentTitle, chunkSize)
            TextModality.ARTICLE_NEWS -> processArticleNews(normalized, documentTitle, chunkSize, overlap)
            TextModality.STORY_NARRATIVE -> processStoryNarrative(normalized, documentTitle, chunkSize, overlap)
            TextModality.STRUCTURED_DATA -> processStructuredData(normalized, documentTitle, chunkSize)
            TextModality.GENERAL_PROSE -> processGeneralProse(normalized, documentTitle, chunkSize, overlap)
        }
    }

    /**
     * Memproses teks percakapan:
     * Menyusun giliran bicara dengan token kanonikal <USER> dan <ASSISTANT>.
     * Turn tunggal yang melebihi chunkSize dipecah di batas kalimat dengan
     * mempertahankan prefix role pada setiap potongan.
     */
    private fun processConversation(
        text: String,
        documentTitle: String,
        chunkSize: Int
    ): List<ProcessedChunk> {
        val lines = text.lines()
        val formattedTurns = mutableListOf<String>()
        var currentRole: TurnRole? = null
        val currentTurnText = StringBuilder()

        fun flushTurn() {
            if (currentRole != null && currentTurnText.isNotBlank()) {
                val prefix = if (currentRole == TurnRole.USER) Tokenizer.USER_TOKEN else Tokenizer.ASSISTANT_TOKEN
                val clean = currentTurnText.toString().trim()
                val fullTurn = "$prefix $clean"
                val available = chunkSize - prefix.length - 1
                if (fullTurn.length > chunkSize && available >= 50) {
                    val pieces = splitBySentenceBoundaries(clean, available, 0)
                    for (piece in pieces) formattedTurns.add("$prefix $piece")
                } else {
                    formattedTurns.add(fullTurn)
                }
                currentTurnText.setLength(0)
            }
        }

        for (rawLine in lines) {
            val line = rawLine.trim()
            if (line.isBlank()) continue

            val userMatcher = USER_TURN_PATTERN.matcher(line)
            val assistantMatcher = ASSISTANT_TURN_PATTERN.matcher(line)

            if (userMatcher.find()) {
                flushTurn()
                currentRole = TurnRole.USER
                currentTurnText.append(line.substring(userMatcher.end()).trim())
            } else if (assistantMatcher.find()) {
                flushTurn()
                currentRole = TurnRole.ASSISTANT
                currentTurnText.append(line.substring(assistantMatcher.end()).trim())
            } else {
                if (currentRole != null) {
                    if (currentTurnText.isNotEmpty()) currentTurnText.append(" ")
                    currentTurnText.append(line)
                } else {
                    // Prolog sistem di awal dokumen: perlakukan sebagai user turn pembuka.
                    currentRole = TurnRole.USER
                    currentTurnText.append(line)
                }
            }
        }
        flushTurn()

        if (formattedTurns.isEmpty()) return processGeneralProse(text, documentTitle, chunkSize, 40)

        val chunks = mutableListOf<ProcessedChunk>()
        val window = StringBuilder()
        var turnCount = 0

        fun flushChunk() {
            if (window.isEmpty()) return
            val content = window.toString().trim() + " " + Tokenizer.EOS_TOKEN
            chunks.add(
                ProcessedChunk(
                    content = content,
                    modality = TextModality.CONVERSATION,
                    segmentTitle = "$documentTitle [Dialog ${chunks.size + 1}]",
                    tokenHint = "Turn Dialog: $turnCount giliran"
                )
            )
            window.setLength(0)
            turnCount = 0
        }

        for (turn in formattedTurns) {
            if (window.isNotEmpty() && window.length + turn.length + 1 > chunkSize) flushChunk()
            if (window.isNotEmpty()) window.append("\n")
            window.append(turn)
            turnCount++
        }
        flushChunk()

        return chunks
    }

    /**
     * Memproses data Tanya Jawab:
     * Menjaga pasangan Pertanyaan & Jawaban tetap menyatu dalam satu window training.
     * Jawaban yatim (tanpa pertanyaan) dibuang. Pasangan yang melebihi chunkSize
     * dipecah di batas kalimat dengan mempertahankan header pertanyaan.
     */
    private fun processQAPairs(
        text: String,
        documentTitle: String,
        chunkSize: Int
    ): List<ProcessedChunk> {
        val lines = text.lines()
        val pairs = mutableListOf<Pair<String, String>>()
        var currentQ: String? = null
        val currentA = StringBuilder()

        fun flushPair() {
            if (currentQ != null && currentA.isNotBlank()) {
                pairs.add(currentQ!!.trim() to currentA.toString().trim())
            }
            // Selalu reset agar jawaban yatim tidak bocor ke pertanyaan berikutnya.
            currentQ = null
            currentA.setLength(0)
        }

        for (rawLine in lines) {
            val line = rawLine.trim()
            if (line.isBlank()) continue

            val qMatch = QA_QUESTION_PATTERN.matcher(line)
            val aMatch = QA_ANSWER_PATTERN.matcher(line)

            if (qMatch.find()) {
                flushPair()
                currentQ = qMatch.group(1)?.trim() ?: line
            } else if (aMatch.find()) {
                if (currentQ == null) continue // orphan answer — discard
                if (currentA.isNotEmpty()) currentA.append(" ")
                currentA.append(aMatch.group(1)?.trim() ?: line)
            } else {
                if (currentQ != null && currentA.isEmpty()) {
                    currentQ = "$currentQ $line"
                } else if (currentA.isNotEmpty()) {
                    currentA.append(" ").append(line)
                }
            }
        }
        flushPair()

        if (pairs.isEmpty()) return processGeneralProse(text, documentTitle, chunkSize, 40)

        val chunks = mutableListOf<ProcessedChunk>()
        var pairIdx = 0
        val totalPairs = pairs.size

        for ((q, a) in pairs) {
            val header = "${Tokenizer.USER_TOKEN} Tanya: $q\n${Tokenizer.ASSISTANT_TOKEN} Jawab: "
            val suffix = " ${Tokenizer.EOS_TOKEN}"
            val availableForAnswer = chunkSize - header.length - suffix.length

            if (availableForAnswer < 60 || a.length <= availableForAnswer) {
                pairIdx++
                chunks.add(
                    ProcessedChunk(
                        content = header + a + suffix,
                        modality = TextModality.QA_PAIR,
                        segmentTitle = "$documentTitle [T&J $pairIdx/$totalPairs]",
                        tokenHint = "Q&A Pasangan #$pairIdx"
                    )
                )
            } else {
                val answerPieces = splitBySentenceBoundaries(a, availableForAnswer, 30)
                for ((pi, piece) in answerPieces.withIndex()) {
                    pairIdx++
                    val isLast = pi == answerPieces.lastIndex
                    chunks.add(
                        ProcessedChunk(
                            content = header + piece + if (isLast) suffix else "",
                            modality = TextModality.QA_PAIR,
                            segmentTitle = "$documentTitle [T&J $pairIdx/$totalPairs]",
                            tokenHint = "Q&A Pasangan #$pairIdx (bagian ${pi + 1}/${answerPieces.size})"
                        )
                    )
                }
            }
        }

        return chunks
    }

    /**
     * Memproses Artikel dan Berita:
     * Memotong sepanjang batas semantik paragraf (\n\n) dan kalimat (. ? !)
     * tanpa memotong di tengah kalimat atau frasa subjek-predikat.
     */
    private fun processArticleNews(
        text: String,
        documentTitle: String,
        chunkSize: Int,
        overlap: Int
    ): List<ProcessedChunk> {
        val paragraphs = text.split(Regex("\n\\s*\n+")).map { it.trim() }.filter { it.isNotEmpty() }
        val rawChunks = mutableListOf<String>()
        val currentBuffer = StringBuilder()

        for (para in paragraphs) {
            if (currentBuffer.isNotEmpty() && currentBuffer.length + para.length + 2 > chunkSize) {
                rawChunks.add(currentBuffer.toString().trim())
                currentBuffer.setLength(0)
            }
            if (para.length > chunkSize) {
                if (currentBuffer.isNotEmpty()) {
                    rawChunks.add(currentBuffer.toString().trim())
                    currentBuffer.setLength(0)
                }
                rawChunks.addAll(splitBySentenceBoundaries(para, chunkSize, overlap))
            } else {
                if (currentBuffer.isNotEmpty()) currentBuffer.append("\n\n")
                currentBuffer.append(para)
            }
        }
        if (currentBuffer.isNotEmpty()) rawChunks.add(currentBuffer.toString().trim())

        val total = rawChunks.size
        return rawChunks.mapIndexed { idx, chunk ->
            ProcessedChunk(
                content = chunk,
                modality = TextModality.ARTICLE_NEWS,
                segmentTitle = "$documentTitle [Artikel ${idx + 1}/$total]",
                tokenHint = "Paragraf Ekspositori ${idx + 1}"
            )
        }
    }

    /**
     * Memproses Cerita & Narasi:
     * Menggunakan pemisah kalimat yang sadar-tanda-kutip agar dialog "..." tidak
     * terpotong di tengah kutipan.
     */
    private fun processStoryNarrative(
        text: String,
        documentTitle: String,
        chunkSize: Int,
        overlap: Int
    ): List<ProcessedChunk> {
        val segments = splitBySentenceBoundaries(text, chunkSize, overlap, quoteAware = true)
        val total = segments.size
        return segments.mapIndexed { idx, segment ->
            ProcessedChunk(
                content = segment,
                modality = TextModality.STORY_NARRATIVE,
                segmentTitle = "$documentTitle [Babak ${idx + 1}/$total]",
                tokenHint = "Alur Narasi & Dialog"
            )
        }
    }

    /**
     * Formats JSONL instruction-tuning examples. Reasoning examples are serialized as
     * USER -> ASSISTANT/THINK -> THINK_END -> ASSISTANT/final answer -> EOS.
     * Existing no-reasoning records retain their previous serialization.
     */
    fun formatStructuredLineIfJson(line: String): String {
        val trimmed = line.trim()
        if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
            val parsed = runCatching {
                val obj = JSONObject(trimmed)
                val reasoning = extractReasoningText(obj)
                when {
                    obj.has("kata") && obj.has("kata_dasar") -> {
                        val kata = obj.optString("kata")
                        val dasar = obj.optString("kata_dasar")
                        val sumber = obj.optString("sumber")
                        val extra = if (sumber.isNotBlank()) " [sumber: $sumber]" else ""
                        "Kata: $kata -> Kata dasar: $dasar$extra"
                    }
                    isTrainingRecord(obj) -> {
                        val explicitInstruction = firstNonBlankText(
                            obj,
                            listOf("instruction", "prompt")
                        )
                        val question = firstNonBlankText(obj, listOf("question"))
                        val isQuestionAnswer = explicitInstruction.isBlank() && question.isNotBlank()
                        val instruction = if (isQuestionAnswer) "Tanya: $question" else explicitInstruction
                        val responseText = firstNonBlankText(
                            obj,
                            listOf("response", "completion", "answer")
                        )
                        val response = if (isQuestionAnswer) "Jawab: $responseText" else responseText
                        formatInstructionResponse(
                            instruction = instruction,
                            input = obj.optString("input"),
                            reasoning = reasoning,
                            response = response
                        )
                    }
                    else -> {
                        val pairs = mutableListOf<String>()
                        val keys = obj.keys()
                        while (keys.hasNext()) {
                            val k = keys.next()
                            if (obj.isNull(k)) continue
                            pairs.add("$k: ${obj.opt(k)}")
                        }
                        pairs.joinToString(" | ")
                    }
                }
            }.getOrNull()
            if (parsed != null) return parsed
        }
        return line
    }

    /**
     * Keeps each reasoning record intact while batching records into bounded chunks.
     * A single long record may exceed chunkSize because the neural engine trains it
     * through overlapping context windows.
     */
    private fun processReasoningDataset(
        text: String,
        documentTitle: String,
        chunkSize: Int
    ): List<ProcessedChunk> {
        val records = text.lineSequence()
            .map { it.trim() }
            .filter { it.isNotBlank() }
            .map(::formatStructuredLineIfJson)
            .toList()
        if (records.isEmpty()) return emptyList()

        val chunks = mutableListOf<String>()
        val buffer = StringBuilder()
        for (record in records) {
            if (buffer.isNotEmpty() && buffer.length + record.length + 1 > chunkSize) {
                chunks += buffer.toString().trim()
                buffer.setLength(0)
            }
            if (buffer.isNotEmpty()) buffer.append('\n')
            buffer.append(record)
        }
        if (buffer.isNotEmpty()) chunks += buffer.toString().trim()

        val total = chunks.size
        return chunks.mapIndexed { index, chunk ->
            ProcessedChunk(
                content = chunk,
                modality = TextModality.REASONING,
                segmentTitle = "$documentTitle [Reasoning ${index + 1}/$total]",
                tokenHint = "Instruction, reasoning steps, and final answer"
            )
        }
    }

    /**
     * Memproses Data Terstruktur (Daftar, Spesifikasi, Kode, JSONL Leksikon).
     */
    private fun processStructuredData(
        text: String,
        documentTitle: String,
        chunkSize: Int
    ): List<ProcessedChunk> {
        val lines = text.lines().map { formatStructuredLineIfJson(it.trim()) }.filter { it.isNotBlank() }
        val chunks = mutableListOf<String>()
        val buffer = StringBuilder()

        for (line in lines) {
            if (buffer.isNotEmpty() && buffer.length + line.length + 1 > chunkSize) {
                chunks.add(buffer.toString().trim())
                buffer.setLength(0)
            }
            if (buffer.isNotEmpty()) buffer.append("\n")
            buffer.append(line)
        }
        if (buffer.isNotEmpty()) chunks.add(buffer.toString().trim())

        val total = chunks.size
        return chunks.mapIndexed { idx, chunk ->
            ProcessedChunk(
                content = chunk,
                modality = TextModality.STRUCTURED_DATA,
                segmentTitle = "$documentTitle [Struktur ${idx + 1}/$total]",
                tokenHint = "Entri Data Terstruktur"
            )
        }
    }

    /**
     * Memproses Prosa Umum (fallback standar yang aman).
     */
    private fun processGeneralProse(
        text: String,
        documentTitle: String,
        chunkSize: Int,
        overlap: Int
    ): List<ProcessedChunk> {
        val segments = splitBySentenceBoundaries(text, chunkSize, overlap)
        val total = segments.size
        return segments.mapIndexed { idx, seg ->
            ProcessedChunk(
                content = seg,
                modality = TextModality.GENERAL_PROSE,
                segmentTitle = if (total > 1) "$documentTitle [Blok ${idx + 1}/$total]" else documentTitle,
                tokenHint = null
            )
        }
    }

    /**
     * Pembagi kalimat adaptif:
     * - Memprioritaskan pemisahan pada \n\n, \n, ". ", "? ", "! "
     * - Jika quoteAware=true, menghindari pemisahan di dalam tanda kutip "..."
     * - Fallback ke spasi terdekat jika tidak ada pemisah kalimat yang layak
     */
    private fun splitBySentenceBoundaries(
        text: String,
        chunkSize: Int,
        overlap: Int,
        quoteAware: Boolean = false
    ): List<String> {
        if (text.length <= chunkSize) return listOf(text)

        val result = mutableListOf<String>()
        var start = 0
        while (start < text.length) {
            var end = minOf(start + chunkSize, text.length)
            if (end < text.length) {
                val window = text.substring(start, end)
                val lastDoubleNl = window.lastIndexOf("\n\n")
                val lastNl = window.lastIndexOf('\n')
                val lastDot = window.lastIndexOf(". ")
                val lastQm = window.lastIndexOf("? ")
                val lastEm = window.lastIndexOf("! ")
                val minBreak = chunkSize / 3

                val bestPunct = maxOf(lastDoubleNl, lastNl, lastDot, lastQm, lastEm)
                if (bestPunct > minBreak) {
                    var candidateEnd = start + bestPunct + 1
                    if (quoteAware) {
                        val quotesBefore = text.substring(start, candidateEnd).count { it == '"' }
                        if (quotesBefore % 2 == 1) {
                            // Kita berada di dalam tanda kutip — geser ke setelah kutip penutup.
                            val closingQuote = text.indexOf('"', candidateEnd)
                            val overshootLimit = start + chunkSize + 40
                            if (closingQuote in candidateEnd until overshootLimit) {
                                candidateEnd = closingQuote + 1
                            }
                        }
                    }
                    end = candidateEnd
                } else {
                    val lastSpace = window.lastIndexOf(' ')
                    if (lastSpace > minBreak) end = start + lastSpace + 1
                }
            }

            val slice = text.substring(start, end).trim()
            if (slice.isNotEmpty()) result.add(slice)

            if (end >= text.length) break
            val nextStart = end - overlap
            start = if (nextStart <= start) end else nextStart
        }
        return result.ifEmpty { listOf(text) }
    }
}