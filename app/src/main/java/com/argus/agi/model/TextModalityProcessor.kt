package com.argus.agi.model

import java.util.Locale
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

    private val defaultChunkTokenizer: Tokenizer by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        Tokenizer()
    }

    private const val LITERAL_LINE_ESCAPE = '\uE001'

    private enum class TurnRole(val token: String) {
        SYSTEM(Tokenizer.SYSTEM_TOKEN),
        USER(Tokenizer.USER_TOKEN),
        ASSISTANT(Tokenizer.ASSISTANT_TOKEN)
    }

    private data class ConversationTurn(
        val role: TurnRole,
        val content: String
    )

    private data class SourceLine(
        val content: String,
        val separator: String
    )

    private fun rolePayloadAfterHeader(line: String, headerEnd: Int): String {
        val payloadStart = if (headerEnd < line.length && line[headerEnd] == ' ') {
            headerEnd + 1
        } else {
            headerEnd
        }
        return line.substring(payloadStart)
    }

    private fun splitTextLinesPreservingSeparators(text: String): List<SourceLine> {
        if (text.isEmpty()) return emptyList()

        val lines = ArrayList<SourceLine>()
        var lineStart = 0
        var cursor = 0

        while (cursor < text.length) {
            when (text[cursor]) {
                '\r' -> {
                    val separator = if (cursor + 1 < text.length && text[cursor + 1] == '\n') {
                        "\r\n"
                    } else {
                        "\r"
                    }
                    lines += SourceLine(text.substring(lineStart, cursor), separator)
                    cursor += separator.length
                    lineStart = cursor
                }
                '\n' -> {
                    lines += SourceLine(text.substring(lineStart, cursor), "\n")
                    cursor++
                    lineStart = cursor
                }
                else -> cursor++
            }
        }

        lines += SourceLine(text.substring(lineStart), "")
        return lines
    }

    // Regex pengenal giliran percakapan.
    // Kolon diwajibkan untuk kata biasa (User:, Pengguna:, Bot:, ...) agar tidak
    // salah menangkap prosa seperti "User experience is critical".
    // Token <USER>/<ASSISTANT> boleh tanpa kolon karena merupakan marker kanonikal.
    private val USER_TURN_PATTERN = Pattern.compile(
        "^\\s*(?:<USER>:?|User\\s*:|Pengguna\\s*:|Human\\s*:|P\\s*:)",
        Pattern.CASE_INSENSITIVE
    )
    private val ASSISTANT_TURN_PATTERN = Pattern.compile(
        "^\\s*(?:<ASSISTANT>:?|Assistant\\s*:|Asisten\\s*:|Argus\\s*:|A\\.R\\.G\\.U\\.S\\s*:|Bot\\s*:|J\\s*:)",
        Pattern.CASE_INSENSITIVE
    )

    private val SYSTEM_TURN_PATTERN = Pattern.compile(
        "^\\s*(?:<SYSTEM>:?|System\\s*:|Developer\\s*:)",
        Pattern.CASE_INSENSITIVE
    )

    // Regex format Tanya Jawab terpisah (termasuk Pertanyaan/Jawaban yang kini
    // TIDAK lagi berada di turn-pattern agar dapat terdeteksi sebagai QA_PAIR).
    private val QA_QUESTION_PATTERN = Pattern.compile(
        "^\\s*(?:Tanya|Pertanyaan|Q|Question)\\s*:(.*)$",
        Pattern.CASE_INSENSITIVE
    )
    private val QA_ANSWER_PATTERN = Pattern.compile(
        "^\\s*(?:Jawab|Jawaban|A|Answer)\\s*:(.*)$",
        Pattern.CASE_INSENSITIVE
    )

    private val BULLET_LINE_REGEX = Regex("^[\\-\\*\\+•●▪◦‣⁃]\\s+.*")
    private val NUMBERED_LINE_REGEX = Regex("^\\d+[.)]\\s+.*")

    /**
     * Explicit, lossy typography normalization for callers that deliberately opt in.
     * Training ingestion and BYTE_LEVEL_BPE_V1 preserve source text instead.
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
     * Canonicalizes JSONL, ChatML/OpenAI messages, ShareGPT conversations, and
     * instruction-tuning records before modality detection and chunking.
     */
    private fun normalizeTrainingDatasetInput(text: String): String {
        val source = text.removePrefix("\uFEFF")
        val records = parseTrainingDatasetObjects(source) ?: return source
        return records.joinToString("\n\n") { record ->
            formatTrainingRecord(record)?.second ?: record.toString()
        }
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
     * Parses one JSON object, a top-level array of training objects, or a JSONL
     * stream. Every non-whitespace character must belong to a complete JSON value;
     * a valid prefix followed by unrelated text is rejected.
     */
    private fun parseTrainingDatasetObjects(rawText: String): List<JSONObject>? {
        val source = rawText.removePrefix("\uFEFF")
        if (source.isBlank()) return null

        fun skipWhitespace(startIndex: Int): Int {
            var index = startIndex
            while (index < source.length && source[index].isWhitespace()) {
                index++
            }
            return index
        }

        var cursor = skipWhitespace(0)
        if (cursor >= source.length) return null

        if (source[cursor] == '[') {
            val endExclusive = findCompleteJsonValueEnd(source, cursor) ?: return null
            if (skipWhitespace(endExclusive) != source.length) return null

            val array = runCatching {
                JSONArray(source.substring(cursor, endExclusive))
            }.getOrNull() ?: return null
            if (array.length() == 0) return null

            val records = ArrayList<JSONObject>(array.length())
            for (index in 0 until array.length()) {
                val record = array.optJSONObject(index) ?: return null
                records.add(record)
            }
            return records.takeIf { it.all(::isTrainingRecord) }
        }

        if (source[cursor] != '{') return null

        val records = ArrayList<JSONObject>()
        while (cursor < source.length) {
            cursor = skipWhitespace(cursor)
            if (cursor >= source.length) break
            if (source[cursor] != '{') return null

            val endExclusive = findCompleteJsonValueEnd(source, cursor) ?: return null
            val record = runCatching {
                JSONObject(source.substring(cursor, endExclusive))
            }.getOrNull() ?: return null
            records.add(record)
            cursor = endExclusive
        }

        if (records.isEmpty() || !records.all(::isTrainingRecord)) return null
        return records
    }

    internal fun formatTrainingDatasetRecords(rawText: String): List<String>? {
        val records = parseTrainingDatasetObjects(rawText) ?: return null
        return records.map { record ->
            formatTrainingRecord(record)?.second ?: formatStructuredLineIfJson(record.toString())
        }
    }

    /**
     * Canonicalizes one complete corpus document without splitting it into model
     * context windows. Structured training records are formatted independently and
     * kept in source order; other structured lines retain the existing formatter.
     */
    internal fun formatTrainingDocument(rawText: String): String {
        if (rawText.isBlank()) return ""

        // Validation keeps each source document whole so the engine can own context
        // windowing and target weighting. Reuse the exact modality serializer used
        // by training, including its control-token framing and EOS policy.
        return process(
            rawText = rawText,
            documentTitle = "Validation",
            chunkSize = Int.MAX_VALUE,
            overlap = 0
        ).joinToString("\n") { chunk -> chunk.content }
    }

    private fun firstNonBlankText(obj: JSONObject, keys: List<String>): String {
        for (key in keys) {
            val value = obj.opt(key)
            if (value is String && value.isNotBlank()) return value.trim()
        }
        return ""
    }

    private fun firstNonBlankPayload(obj: JSONObject, keys: List<String>): String {
        for (key in keys) {
            val value = obj.opt(key)
            if (value is String && value.isNotBlank()) return value
        }
        return ""
    }

    private fun isTrainingRecord(obj: JSONObject): Boolean {
        val hasPrompt = firstNonBlankText(
            obj,
            listOf("instruction", "prompt", "question")
        ).isNotBlank()
        val hasAnswer = firstNonBlankText(
            obj,
            listOf("response", "completion", "answer")
        ).isNotBlank()
        return (hasPrompt && hasAnswer) || isConversationRecord(obj)
    }

    private fun isConversationRecord(obj: JSONObject): Boolean =
        extractConversationTurns(obj) != null

    /**
     * Converts each validated source object directly to its own atomic training record.
     */
    private fun formatTrainingRecord(obj: JSONObject): Pair<TextModality, String>? {
        val turns = extractConversationTurns(obj)
        if (turns != null) return TextModality.CONVERSATION to serializeConversationTurns(turns)
        if (!isTrainingRecord(obj)) return null

        val reasoning = extractReasoningText(obj)
        val explicitInstruction = firstNonBlankPayload(obj, listOf("instruction", "prompt"))
        val question = firstNonBlankPayload(obj, listOf("question"))
        val isQuestionAnswer = explicitInstruction.isBlank() && question.isNotBlank()
        val instruction = if (isQuestionAnswer) "Tanya: $question" else explicitInstruction
        val responseText = firstNonBlankPayload(obj, listOf("response", "completion", "answer"))
        val response = if (isQuestionAnswer) "Jawab: $responseText" else responseText
        val input = firstNonBlankPayload(obj, listOf("input"))
        val modality = if (reasoning.isNullOrBlank()) TextModality.CONVERSATION else TextModality.REASONING

        return modality to formatInstructionResponse(
            instruction = instruction,
            input = input,
            reasoning = reasoning,
            response = response
        )
    }

    /**
     * Serializes parsed turns without interpreting payload lines as role headers.
     * Every complete assistant turn ends with EOS; user/system turns never do.
     */
    private fun serializeConversationTurns(turns: List<ConversationTurn>): String {
        return turns.joinToString("\n") { turn ->
            val payload = "${turn.role.token} ${Tokenizer.escapeLiteralControlTokens(turn.content)}"
            if (turn.role == TurnRole.ASSISTANT) {
                "$payload ${Tokenizer.EOS_TOKEN}"
            } else {
                payload
            }
        }
    }

    /**
     * Protects role-looking continuation lines in the legacy string exchange format.
     * processConversation removes the marker before payload serialization.
     */
    private fun escapeConversationHeaderLines(text: String): String {
        val lines = splitTextLinesPreservingSeparators(text)
        return buildString(text.length + 8) {
            lines.forEachIndexed { index, sourceLine ->
                val line = sourceLine.content
                val startsWithEscapeMarker =
                    line.isNotEmpty() && line[0] == LITERAL_LINE_ESCAPE
                val looksLikeRoleHeader =
                    USER_TURN_PATTERN.matcher(line).find() ||
                    ASSISTANT_TURN_PATTERN.matcher(line).find() ||
                    SYSTEM_TURN_PATTERN.matcher(line).find()
                if (startsWithEscapeMarker || (index > 0 && looksLikeRoleHeader)) {
                    append(LITERAL_LINE_ESCAPE)
                }
                append(line)
                append(sourceLine.separator)
            }
        }
    }

    /**
     * Reads text-only ChatML/OpenAI message content. Unsupported multimodal payloads
     * are ignored rather than converted into arbitrary JSON strings.
     */
    private fun readConversationMessageContent(value: Any?): String {
        return when (value) {
            is String -> value
            is JSONArray -> {
                val textParts = mutableListOf<String>()
                for (index in 0 until value.length()) {
                    val part = value.opt(index)
                    val text = when (part) {
                        is String -> part
                        is JSONObject -> {
                            val type = part.optString("type", "text")
                                .trim()
                                .lowercase(Locale.ROOT)
                            if (type in setOf("text", "input_text", "output_text")) {
                                firstNonBlankPayload(part, listOf("text", "content", "value"))
                            } else {
                                ""
                            }
                        }
                        else -> ""
                    }
                    if (text.isNotBlank()) textParts += text
                }
                textParts.joinToString("\n")
            }
            is JSONObject -> firstNonBlankPayload(value, listOf("text", "content", "value"))
            else -> ""
        }
    }

    /**
     * Accepts:
     * - ChatML/OpenAI: {"messages":[{"role":"user","content":"..."}]}
     * - ShareGPT: {"conversations":[{"from":"human","value":"..."}]}
     *
     * System and developer messages are preserved as SYSTEM turns. Tool/function
     * messages and unsupported modalities are deliberately excluded from SFT text.
     */
    private fun extractConversationTurns(obj: JSONObject): List<ConversationTurn>? {
        val messages = obj.optJSONArray("messages")
            ?: obj.optJSONArray("conversations")
            ?: return null
        if (messages.length() == 0) return null

        val turns = ArrayList<ConversationTurn>(messages.length())
        for (index in 0 until messages.length()) {
            val message = messages.optJSONObject(index) ?: continue
            val roleName = firstNonBlankText(
                message,
                listOf("role", "from", "speaker")
            ).lowercase(Locale.ROOT)
            val role = when (roleName) {
                "system", "developer" -> TurnRole.SYSTEM
                "user", "human", "prompter" -> TurnRole.USER
                "assistant", "gpt", "bot", "model" -> TurnRole.ASSISTANT
                else -> null
            } ?: continue

            val rawContent = when {
                message.has("content") && !message.isNull("content") -> message.opt("content")
                message.has("value") && !message.isNull("value") -> message.opt("value")
                message.has("text") && !message.isNull("text") -> message.opt("text")
                else -> null
            }
            val content = readConversationMessageContent(rawContent)
            if (content.isNotBlank()) {
                turns += ConversationTurn(role = role, content = content)
            }
        }

        val hasUser = turns.any { it.role == TurnRole.USER }
        val hasAssistant = turns.any { it.role == TurnRole.ASSISTANT }
        return turns.takeIf { hasUser && hasAssistant }
    }

    private fun formatConversationRecord(obj: JSONObject): String? {
        val turns = extractConversationTurns(obj) ?: return null
        return serializeConversationTurns(turns)
    }

    /**
     * Serializes one user-approved exchange using the same role markers consumed
     * by imported conversation datasets and the on-device next-token trainer.
     */
    fun formatConversationExchange(
        userMessage: String,
        assistantMessage: String
    ): String? {
        if (userMessage.isBlank() || assistantMessage.isBlank()) return null

        val safeUserMessage = escapeConversationHeaderLines(userMessage)
        val safeAssistantMessage = escapeConversationHeaderLines(assistantMessage)
        return "${Tokenizer.USER_TOKEN} $safeUserMessage\n" +
            "${Tokenizer.ASSISTANT_TOKEN} $safeAssistantMessage"
    }

    private fun readReasoningStep(value: Any?): String {
        return when (value) {
            is String -> value
            is JSONObject -> firstNonBlankPayload(
                value,
                listOf("step", "text", "content", "reasoning", "explanation")
            )
            else -> ""
        }
    }

    private fun normalizeReasoningValue(raw: Any?): String {
        return when (raw) {
            is String -> raw
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
        val prompt = buildString {
            if (instruction.isNotBlank()) append(instruction)
            if (input.isNotBlank()) {
                if (isNotEmpty()) append('\n')
                append("Input: ").append(input)
            }
        }
        val safePrompt = Tokenizer.escapeLiteralControlTokens(prompt)
        val safeResponse = Tokenizer.escapeLiteralControlTokens(response)
        val safeReasoning = reasoning
            ?.takeIf { it.isNotBlank() }
            ?.let { value -> Tokenizer.escapeLiteralControlTokens(value) }

        return if (safeReasoning == null) {
            "${Tokenizer.USER_TOKEN} $safePrompt\n" +
                "${Tokenizer.ASSISTANT_TOKEN} $safeResponse ${Tokenizer.EOS_TOKEN}"
        } else {
            "${Tokenizer.USER_TOKEN} $safePrompt\n" +
                "${Tokenizer.ASSISTANT_TOKEN}${Tokenizer.THINK_START}\n" +
                "$safeReasoning\n" +
                "${Tokenizer.THINK_END}\n" +
                "${Tokenizer.ASSISTANT_TOKEN} $safeResponse ${Tokenizer.EOS_TOKEN}"
        }
    }

    /**
     * Mendeteksi modalitas konten teks.
     */
    fun detectModality(text: String): TextModality {
        val normalizedInput = normalizeTrainingDatasetInput(text)
        val trimmed = normalizedInput.trim()
        if (trimmed.isEmpty()) return TextModality.GENERAL_PROSE

        val lines = trimmed.lines().map { it.trim() }.filter { it.isNotEmpty() }
        if (lines.isEmpty()) return TextModality.GENERAL_PROSE

        val containsReasoningMarkers =
            trimmed.contains(Tokenizer.THINK_START) &&
                trimmed.contains(Tokenizer.THINK_END)
        val containsReasoningRecords = lines.any { line ->
            val candidate = line.trimStart()
            candidate.startsWith("{") &&
                runCatching { JSONObject(candidate) }
                    .getOrNull()
                    ?.let { record ->
                        isTrainingRecord(record) && !extractReasoningText(record).isNullOrBlank()
                    } == true
        }
        if (containsReasoningRecords) return TextModality.REASONING

        // Determine conversation framing before treating literal THINK text as a
        // reasoning instruction. Genuine serialized reasoning records use the
        // assistant/THINK prefix and return to an assistant final-answer turn.
        var userTurnCount = 0
        var assistantTurnCount = 0
        for (line in lines) {
            if (USER_TURN_PATTERN.matcher(line).find()) userTurnCount++
            if (ASSISTANT_TURN_PATTERN.matcher(line).find()) assistantTurnCount++
        }
        val hasBothTokens = trimmed.contains(Tokenizer.USER_TOKEN) &&
            trimmed.contains(Tokenizer.ASSISTANT_TOKEN)
        val hasConversationStructure =
            (userTurnCount >= 1 && assistantTurnCount >= 1) || hasBothTokens

        val thinkStartIndex = trimmed.indexOf(Tokenizer.THINK_START)
        val thinkEndIndex = trimmed.indexOf(
            Tokenizer.THINK_END,
            thinkStartIndex.coerceAtLeast(0) + Tokenizer.THINK_START.length
        )
        val hasAssistantThinkFraming =
            trimmed.contains(Tokenizer.ASSISTANT_TOKEN + Tokenizer.THINK_START) &&
                thinkEndIndex >= 0 &&
                trimmed.indexOf(
                    Tokenizer.ASSISTANT_TOKEN,
                    thinkEndIndex + Tokenizer.THINK_END.length
                ) >= 0

        // Reasoning markers are structural only inside the canonical assistant
        // reasoning frame. Their occurrence in prose/documentation is literal text.
        if (containsReasoningMarkers && hasAssistantThinkFraming) {
            return TextModality.REASONING
        }
        if (hasConversationStructure) return TextModality.CONVERSATION

        // 2. Tanya Jawab
        var qCount = 0
        var aCount = 0
        for (line in lines) {
            if (QA_QUESTION_PATTERN.matcher(line).find()) qCount++
            if (QA_ANSWER_PATTERN.matcher(line).find()) aCount++
        }
        if (qCount >= 1 && aCount >= 1) return TextModality.QA_PAIR

        // 3. Cerita & Narasi
        val quoteCount = trimmed.count { character -> character in "\"“”‘’«»" }
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
    /**
     * Processes a complete source document. chunkSize and overlap are measured in
     * tokens emitted by the supplied tokenizer, not UTF-16 characters. Atomic
     * conversations, reasoning records, and JSONL records may exceed this soft
     * ingestion target; the neural engine owns model context-windowing.
     */
    fun process(
        rawText: String,
        documentTitle: String = "Dokumen",
        chunkSize: Int = 380,
        overlap: Int = 50,
        tokenizer: Tokenizer = defaultChunkTokenizer
    ): List<ProcessedChunk> {
        require(chunkSize > 0) { "chunkSize harus > 0, diterima: $chunkSize" }
        require(overlap >= 0) { "overlap tidak boleh negatif, diterima: $overlap" }
        require(overlap < chunkSize) { "overlap ($overlap) harus < chunkSize ($chunkSize)" }

        val source = rawText.removePrefix("\uFEFF")
        if (source.isBlank()) return emptyList()

        // Keep JSON/JSONL records typed until serialization; never reparse payload text.
        val records = parseTrainingDatasetObjects(source)
        if (records != null) return processTrainingRecords(records, documentTitle)

        return when (detectModality(source)) {
            TextModality.CONVERSATION -> processConversation(source, documentTitle, chunkSize, tokenizer)
            TextModality.QA_PAIR -> processQAPairs(source, documentTitle, chunkSize, tokenizer)
            TextModality.REASONING -> processReasoningDataset(source, documentTitle, chunkSize)
            TextModality.ARTICLE_NEWS -> processArticleNews(source, documentTitle, chunkSize, overlap, tokenizer)
            TextModality.STORY_NARRATIVE -> processStoryNarrative(source, documentTitle, chunkSize, overlap, tokenizer)
            TextModality.STRUCTURED_DATA -> processStructuredData(source, documentTitle, chunkSize, tokenizer)
            TextModality.GENERAL_PROSE -> processGeneralProse(source, documentTitle, chunkSize, overlap, tokenizer)
        }
    }

    /**
     * One complete JSON source record becomes one training sample. Token-level
     * context windowing remains the responsibility of the neural training engine.
     */
    private fun processTrainingRecords(
        records: List<JSONObject>,
        documentTitle: String
    ): List<ProcessedChunk> = records.mapIndexedNotNull { index, record ->
        val formatted = formatTrainingRecord(record) ?: return@mapIndexedNotNull null
        ProcessedChunk(
            content = formatted.second,
            modality = formatted.first,
            segmentTitle = "$documentTitle [Record ${index + 1}/${records.size}]",
            tokenHint = if (formatted.first == TextModality.REASONING) {
                "Atomic instruction, reasoning, and final answer"
            } else {
                "Atomic conversation or instruction record"
            }
        )
    }

    /**
     * Memproses teks percakapan:
     * Menyusun giliran bicara dengan token kanonikal <USER> dan <ASSISTANT>.
     * The complete conversation remains one sample; context windowing is delegated
     * to the training engine so neither user nor assistant turns are split here.
     */
    private fun processConversation(
        text: String,
        documentTitle: String,
        chunkSize: Int,
        tokenizer: Tokenizer
    ): List<ProcessedChunk> {
        val turns = mutableListOf<ConversationTurn>()
        var currentRole: TurnRole? = null
        val currentTurnText = StringBuilder()
        var pendingSeparator = ""

        fun flushTurn() {
            val role = currentRole
            val payload = currentTurnText.toString()
            if (role != null && payload.isNotEmpty()) {
                turns += ConversationTurn(role = role, content = payload)
            }
            currentTurnText.setLength(0)
            currentRole = null
            pendingSeparator = ""
        }

        for (sourceLine in splitTextLinesPreservingSeparators(text)) {
            val rawLine = sourceLine.content
            val escapedLiteralLine =
                rawLine.isNotEmpty() && rawLine[0] == LITERAL_LINE_ESCAPE
            if (escapedLiteralLine) {
                // Remove one marker. A doubled marker preserves the original leading marker.
                if (currentRole == null) currentRole = TurnRole.USER
                currentTurnText.append(pendingSeparator).append(rawLine.substring(1))
                pendingSeparator = sourceLine.separator
                continue
            }
            val line = rawLine

            

            val systemMatcher = SYSTEM_TURN_PATTERN.matcher(line)
            val userMatcher = USER_TURN_PATTERN.matcher(line)
            val assistantMatcher = ASSISTANT_TURN_PATTERN.matcher(line)
            when {
                systemMatcher.find() -> {
                    flushTurn()
                    currentRole = TurnRole.SYSTEM
                    currentTurnText.append(rolePayloadAfterHeader(line, systemMatcher.end()))
                    pendingSeparator = sourceLine.separator
                }
                userMatcher.find() -> {
                    flushTurn()
                    currentRole = TurnRole.USER
                    currentTurnText.append(rolePayloadAfterHeader(line, userMatcher.end()))
                    pendingSeparator = sourceLine.separator
                }
                assistantMatcher.find() -> {
                    flushTurn()
                    currentRole = TurnRole.ASSISTANT
                    currentTurnText.append(rolePayloadAfterHeader(line, assistantMatcher.end()))
                    pendingSeparator = sourceLine.separator
                }
                currentRole != null -> {
                    currentTurnText.append(pendingSeparator).append(line)
                    pendingSeparator = sourceLine.separator
                }
                line.isNotEmpty() -> {
                    currentRole = TurnRole.USER
                    currentTurnText.append(line)
                    pendingSeparator = sourceLine.separator
                }
            }
        }
        flushTurn()

        if (turns.isEmpty()) return processGeneralProse(text, documentTitle, chunkSize, 0, tokenizer)
        return processConversationTurns(turns, documentTitle)
    }

    private fun processConversationTurns(
        turns: List<ConversationTurn>,
        documentTitle: String
    ): List<ProcessedChunk> {
        if (turns.isEmpty()) return emptyList()
        return listOf(
            ProcessedChunk(
                content = serializeConversationTurns(turns),
                modality = TextModality.CONVERSATION,
                segmentTitle = "$documentTitle [Dialog 1]",
                tokenHint = "Turn Dialog: ${turns.size} giliran"
            )
        )
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
        chunkSize: Int,
        tokenizer: Tokenizer
    ): List<ProcessedChunk> {
        val pairs = mutableListOf<Pair<String, String>>()
        var currentQuestion: StringBuilder? = null
        var answerStarted = false
        val currentAnswer = StringBuilder()
        var pendingSeparator = ""

        fun flushPair() {
            val question = currentQuestion
            if (question != null && answerStarted && currentAnswer.isNotBlank()) {
                pairs.add(question.toString() to currentAnswer.toString())
            }
            currentQuestion = null
            answerStarted = false
            currentAnswer.setLength(0)
            pendingSeparator = ""
        }

        for (sourceLine in splitTextLinesPreservingSeparators(text)) {
            val line = sourceLine.content
            val questionMatcher = QA_QUESTION_PATTERN.matcher(line)
            val answerMatcher = QA_ANSWER_PATTERN.matcher(line)

            when {
                questionMatcher.find() -> {
                    flushPair()
                    currentQuestion = StringBuilder(questionMatcher.group(1).orEmpty())
                    pendingSeparator = sourceLine.separator
                }
                answerMatcher.find() -> {
                    if (currentQuestion != null) {
                        answerStarted = true
                        currentAnswer.append(answerMatcher.group(1).orEmpty())
                    }
                    pendingSeparator = sourceLine.separator
                }
                else -> {
                    val activeQuestion = currentQuestion
                    if (activeQuestion != null) {
                        if (answerStarted) {
                            currentAnswer.append(pendingSeparator).append(line)
                        } else {
                            activeQuestion.append(pendingSeparator).append(line)
                        }
                    }
                    pendingSeparator = sourceLine.separator
                }
            }
        }
        flushPair()

        if (pairs.isEmpty()) {
            return processGeneralProse(text, documentTitle, chunkSize, 0, tokenizer)
        }

        val chunks = mutableListOf<ProcessedChunk>()
        val totalPairs = pairs.size
        var pairIndex = 0

        for ((question, answer) in pairs) {
            val pairNumber = ++pairIndex
            val safeQuestion = Tokenizer.escapeLiteralControlTokens(question)
            val safeAnswer = Tokenizer.escapeLiteralControlTokens(answer)
            val header = "${Tokenizer.USER_TOKEN} Tanya:$safeQuestion\n" +
                "${Tokenizer.ASSISTANT_TOKEN} Jawab:"
            val framedAnswer = header + safeAnswer + " " + Tokenizer.EOS_TOKEN
            val framedTokenCount = tokenizer.encodeToIds(framedAnswer, autoRegister = false).size
            val headerTokenCount = tokenizer.encodeToIds(
                header + " " + Tokenizer.EOS_TOKEN,
                autoRegister = false
            ).size
            val availableForAnswer = (chunkSize - headerTokenCount).coerceAtLeast(0)

            if (framedTokenCount <= chunkSize || availableForAnswer == 0) {
                chunks.add(
                    ProcessedChunk(
                        content = framedAnswer,
                        modality = TextModality.QA_PAIR,
                        segmentTitle = "$documentTitle [T&J $pairNumber/$totalPairs]",
                        tokenHint = "Q&A Pasangan #$pairNumber"
                    )
                )
            } else {
                val answerPieces = splitBySentenceBoundaries(
                    safeAnswer,
                    availableForAnswer,
                    0,
                    tokenizer
                )
                for ((pieceIndex, piece) in answerPieces.withIndex()) {
                    val isLast = pieceIndex == answerPieces.lastIndex
                    chunks.add(
                        ProcessedChunk(
                            content = header + piece + if (isLast) " " + Tokenizer.EOS_TOKEN else "",
                            modality = TextModality.QA_PAIR,
                            segmentTitle = "$documentTitle [T&J $pairNumber/$totalPairs]",
                            tokenHint = "Q&A Pasangan #$pairNumber (part ${pieceIndex + 1}/${answerPieces.size})"
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
        overlap: Int,
        tokenizer: Tokenizer
    ): List<ProcessedChunk> {
        val safeText = Tokenizer.escapeLiteralControlTokens(text)
        val segments = splitBySentenceBoundaries(safeText, chunkSize, overlap, tokenizer)
        val total = segments.size
        return segments.mapIndexed { index, segment ->
            ProcessedChunk(
                content = segment,
                modality = TextModality.ARTICLE_NEWS,
                segmentTitle = "$documentTitle [Artikel ${index + 1}/$total]",
                tokenHint = "Paragraf Ekspositori ${index + 1}"
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
        overlap: Int,
        tokenizer: Tokenizer
    ): List<ProcessedChunk> {
        val safeText = Tokenizer.escapeLiteralControlTokens(text)
        val segments = splitBySentenceBoundaries(
            safeText,
            chunkSize,
            overlap,
            tokenizer,
            quoteAware = true
        )
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
        if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) return line

        val formatted = runCatching {
            val obj = JSONObject(trimmed)
            formatTrainingRecord(obj)?.second ?: when {
                obj.has("kata") && obj.has("kata_dasar") -> {
                    val kata = firstNonBlankPayload(obj, listOf("kata"))
                    val dasar = firstNonBlankPayload(obj, listOf("kata_dasar"))
                    val sumber = firstNonBlankPayload(obj, listOf("sumber"))
                    val extra = if (sumber.isNotBlank()) " [sumber: $sumber]" else ""
                    Tokenizer.escapeLiteralControlTokens("Kata: $kata -> Kata dasar: $dasar$extra")
                }
                else -> {
                    val pairs = mutableListOf<String>()
                    val keys = obj.keys()
                    while (keys.hasNext()) {
                        val key = keys.next()
                        if (obj.isNull(key)) continue
                        pairs += "$key: ${obj.opt(key)}"
                    }
                    Tokenizer.escapeLiteralControlTokens(pairs.joinToString(" | "))
                }
            }
        }.getOrNull()
        return formatted ?: line
    }

    /**
     * Formats recognized structured records line-by-line while retaining every original
     * CR, LF, and CRLF separator. Unrecognized plain-text lines remain unchanged.
     */
    internal fun formatStructuredTextPreservingLineEndings(
        text: String,
        escapePlainText: Boolean = false
    ): String {
        if (text.isEmpty()) return text

        val result = StringBuilder(text.length)
        var lineStart = 0
        var cursor = 0

        while (cursor < text.length) {
            val character = text[cursor]
            if (character == '\r' || character == '\n') {
                val line = text.substring(lineStart, cursor)
                val formatted = formatStructuredLineIfJson(line)
                result.append(
                    if (escapePlainText && formatted == line) {
                        Tokenizer.escapeLiteralControlTokens(line)
                    } else {
                        formatted
                    }
                )
                if (character == '\r' && cursor + 1 < text.length && text[cursor + 1] == '\n') {
                    result.append("\r\n")
                    cursor += 2
                } else {
                    result.append(character)
                    cursor++
                }
                lineStart = cursor
            } else {
                cursor++
            }
        }

        if (lineStart < text.length) {
            val line = text.substring(lineStart)
            val formatted = formatStructuredLineIfJson(line)
            result.append(
                if (escapePlainText && formatted == line) {
                    Tokenizer.escapeLiteralControlTokens(line)
                } else {
                    formatted
                }
            )
        }
        return result.toString()
    }

    /**
     * Keeps one complete reasoning record intact. The neural engine owns context
     * windowing; this formatter must not split THINK from its final answer.
     */
    private fun processReasoningDataset(
        text: String,
        documentTitle: String,
        @Suppress("UNUSED_PARAMETER") chunkSize: Int
    ): List<ProcessedChunk> {
        val formattedText = formatStructuredTextPreservingLineEndings(text)
        if (formattedText.isBlank()) return emptyList()

        // Keep reasoning and its final answer in one record; the engine owns context windows.
        return listOf(
            ProcessedChunk(
                content = formattedText,
                modality = TextModality.REASONING,
                segmentTitle = "$documentTitle [Reasoning 1]",
                tokenHint = "Atomic instruction, reasoning steps, and final answer"
            )
        )
    }

    /**
     * Memproses Data Terstruktur (Daftar, Spesifikasi, Kode, JSONL Leksikon).
     */
    private fun processStructuredData(
        text: String,
        documentTitle: String,
        chunkSize: Int,
        tokenizer: Tokenizer
    ): List<ProcessedChunk> {
        val formattedText = formatStructuredTextPreservingLineEndings(
            text,
            escapePlainText = true
        )
        if (formattedText.isBlank()) return emptyList()

        val records = splitByCompleteLines(formattedText, chunkSize, tokenizer)
        val total = records.size
        return records.mapIndexed { index, record ->
            ProcessedChunk(
                content = record,
                modality = TextModality.STRUCTURED_DATA,
                segmentTitle = "$documentTitle [Struktur ${index + 1}/$total]",
                tokenHint = "Complete structured record(s)"
            )
        }
    }

    /**
     * Packs complete source lines into bounded chunks. A single line may exceed the
     * soft limit but is never split into partial structured records.
     */
    private fun splitByCompleteLines(
        text: String,
        chunkSize: Int,
        tokenizer: Tokenizer
    ): List<String> {
        if (tokenizer.encodeToIds(text, autoRegister = false).size <= chunkSize) {
            return listOf(text)
        }
        val chunks = mutableListOf<String>()
        val current = StringBuilder()

        fun flush() {
            if (current.isNotEmpty()) {
                chunks += current.toString()
                current.setLength(0)
            }
        }

        for (line in splitTextLinesPreservingSeparators(text)) {
            val record = line.content + line.separator
            val candidate = current.toString() + record
            if (
                current.isNotEmpty() &&
                tokenizer.encodeToIds(candidate, autoRegister = false).size > chunkSize
            ) {
                flush()
            }
            // A single structured record remains atomic even when it exceeds this soft target.
            // The neural engine, not this processor, owns context-window construction.
            current.append(record)
        }
        flush()
        return chunks.ifEmpty { listOf(text) }
    }

    /**
     * Memproses Prosa Umum (fallback standar yang aman).
     */
    private fun processGeneralProse(
        text: String,
        documentTitle: String,
        chunkSize: Int,
        overlap: Int,
        tokenizer: Tokenizer
    ): List<ProcessedChunk> {
        val safeText = Tokenizer.escapeLiteralControlTokens(text)
        val segments = splitBySentenceBoundaries(safeText, chunkSize, overlap, tokenizer)
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
     * Moves a UTF-16 boundary away from the middle of a surrogate pair.
     * Chunk starts prefer the preceding boundary to retain the complete code point
     * in the next window; chunk ends prefer the following boundary to retain it
     * in the current window.
     */
    internal fun alignUtf16Boundary(
        text: CharSequence,
        boundary: Int,
        preferAfter: Boolean
    ): Int {
        val index = boundary.coerceIn(0, text.length)
        val splitsSurrogatePair = index > 0 &&
            index < text.length &&
            Character.isHighSurrogate(text[index - 1]) &&
            Character.isLowSurrogate(text[index])

        if (!splitsSurrogatePair) return index
        return if (preferAfter) index + 1 else index - 1
    }

    private fun unclosedQuoteCloser(text: String): Char? {
        val closers = ArrayList<Char>()
        for (character in text) {
            when (character) {
                '"' -> {
                    if (closers.lastOrNull() == '"') closers.removeAt(closers.lastIndex)
                    else closers.add('"')
                }
                '“' -> closers.add('”')
                '”' -> if (closers.lastOrNull() == '”') closers.removeAt(closers.lastIndex)
                '‘' -> closers.add('’')
                '’' -> if (closers.lastOrNull() == '’') closers.removeAt(closers.lastIndex)
                '«' -> closers.add('»')
                '»' -> if (closers.lastOrNull() == '»') closers.removeAt(closers.lastIndex)
            }
        }
        return closers.lastOrNull()
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
        tokenizer: Tokenizer,
        quoteAware: Boolean = false
    ): List<String> {
        if (tokenizer.encodeToIds(text, autoRegister = false).size <= chunkSize) {
            return listOf(text)
        }

        val result = mutableListOf<String>()
        var start = 0
        while (start < text.length) {
            val tokenBudgetEnd = findTokenBudgetEnd(text, start, chunkSize, tokenizer)
            var end = tokenBudgetEnd
            if (tokenBudgetEnd < text.length) {
                val window = text.substring(start, tokenBudgetEnd)
                val lastDoubleNl = window.lastIndexOf("\n\n")
                val lastNl = window.lastIndexOf('\n')
                val lastDot = window.lastIndexOf(". ")
                val lastQm = window.lastIndexOf("? ")
                val lastEm = window.lastIndexOf("! ")
                val minimumBreakTokens = chunkSize / 3

                val bestPunct = maxOf(lastDoubleNl, lastNl, lastDot, lastQm, lastEm)
                var candidateEnd = if (bestPunct >= 0) start + bestPunct + 1 else start
                var candidateTokenCount = if (candidateEnd > start) {
                    tokenizer.encodeToIds(
                        text.substring(start, candidateEnd),
                        autoRegister = false
                    ).size
                } else {
                    0
                }

                if (candidateTokenCount < minimumBreakTokens) {
                    val lastSpace = window.lastIndexOf(' ')
                    candidateEnd = if (lastSpace >= 0) start + lastSpace + 1 else start
                    candidateTokenCount = if (candidateEnd > start) {
                        tokenizer.encodeToIds(
                            text.substring(start, candidateEnd),
                            autoRegister = false
                        ).size
                    } else {
                        0
                    }
                }

                if (candidateEnd > start && candidateTokenCount >= minimumBreakTokens) {
                    if (quoteAware) {
                        val expectedCloser = unclosedQuoteCloser(text.substring(start, candidateEnd))
                        if (expectedCloser != null) {
                            val closingQuote = text.indexOf(expectedCloser, candidateEnd)
                            if (closingQuote >= candidateEnd && closingQuote < tokenBudgetEnd) {
                                val quotedEnd = alignUtf16Boundary(
                                    text,
                                    closingQuote + 1,
                                    preferAfter = true
                                )
                                if (
                                    tokenizer.encodeToIds(
                                        text.substring(start, quotedEnd),
                                        autoRegister = false
                                    ).size <= chunkSize
                                ) {
                                    candidateEnd = quotedEnd
                                }
                            }
                        }
                    }
                    end = candidateEnd
                }
            }

            end = alignUtf16Boundary(text, end, preferAfter = true)
            val slice = text.substring(start, end)
            if (slice.isNotEmpty()) result.add(slice)
            if (end >= text.length) break

            var nextStart = findTokenOverlapStart(text, start, end, overlap, tokenizer)
            if (overlap > 0) {
                while (nextStart < text.length && text[nextStart].isWhitespace()) nextStart++
            }
            start = if (nextStart <= start) end else nextStart
        }
        return result.ifEmpty { listOf(text) }
    }

    private fun findTokenBudgetEnd(
        text: String,
        start: Int,
        tokenBudget: Int,
        tokenizer: Tokenizer
    ): Int {
        if (start >= text.length) return text.length
        require(tokenBudget > 0) { "Token budget must be positive." }

        var accepted = start
        var step = tokenBudget.coerceAtLeast(1).toLong()
        var candidate = alignUtf16Boundary(
            text,
            (start.toLong() + step).coerceAtMost(text.length.toLong()).toInt(),
            preferAfter = true
        )

        while (candidate > accepted) {
            val count = tokenizer.encodeToIds(
                text.substring(start, candidate),
                autoRegister = false
            ).size
            if (count > tokenBudget) break
            accepted = candidate
            if (accepted >= text.length) return text.length
            step = (step * 2L).coerceAtMost((text.length - accepted).toLong())
            candidate = alignUtf16Boundary(
                text,
                (accepted.toLong() + step).coerceAtMost(text.length.toLong()).toInt(),
                preferAfter = true
            )
            if (candidate <= accepted) candidate = nextCodePointBoundary(text, accepted)
        }

        var low = accepted
        var high = candidate
        if (high <= low) high = nextCodePointBoundary(text, low)

        while (high - low > 1) {
            val midpoint = low + (high - low) / 2
            val alignedMidpoint = alignUtf16Boundary(text, midpoint, preferAfter = true)
            if (alignedMidpoint <= low || alignedMidpoint >= high) break
            val count = tokenizer.encodeToIds(
                text.substring(start, alignedMidpoint),
                autoRegister = false
            ).size
            if (count <= tokenBudget) low = alignedMidpoint else high = alignedMidpoint
        }

        var end = low
        while (
            end > start &&
            tokenizer.encodeToIds(text.substring(start, end), autoRegister = false).size > tokenBudget
        ) {
            end = previousCodePointBoundary(text, end)
        }
        while (end < text.length) {
            val next = nextCodePointBoundary(text, end)
            if (
                tokenizer.encodeToIds(text.substring(start, next), autoRegister = false).size > tokenBudget
            ) break
            end = next
        }
        return if (end == start) nextCodePointBoundary(text, start) else end
    }

    private fun findTokenOverlapStart(
        text: String,
        chunkStart: Int,
        chunkEnd: Int,
        overlapTokens: Int,
        tokenizer: Tokenizer
    ): Int {
        if (overlapTokens <= 0) return chunkEnd

        var low = chunkStart
        var high = chunkEnd
        while (high - low > 1) {
            val midpoint = low + (high - low) / 2
            val alignedMidpoint = alignUtf16Boundary(text, midpoint, preferAfter = true)
            if (alignedMidpoint <= low || alignedMidpoint >= high) break
            val count = tokenizer.encodeToIds(
                text.substring(alignedMidpoint, chunkEnd),
                autoRegister = false
            ).size
            if (count <= overlapTokens) high = alignedMidpoint else low = alignedMidpoint
        }

        while (
            high < chunkEnd &&
            tokenizer.encodeToIds(text.substring(high, chunkEnd), autoRegister = false).size > overlapTokens
        ) {
            high = nextCodePointBoundary(text, high)
        }
        return alignUtf16Boundary(text, high, preferAfter = false)
    }

    private fun nextCodePointBoundary(text: String, index: Int): Int {
        if (index >= text.length) return text.length
        val codePoint = Character.codePointAt(text, index)
        return (index + Character.charCount(codePoint)).coerceAtMost(text.length)
    }

    private fun previousCodePointBoundary(text: String, index: Int): Int {
        if (index <= 0) return 0
        val codePoint = Character.codePointBefore(text, index)
        return (index - Character.charCount(codePoint)).coerceAtLeast(0)
    }
}