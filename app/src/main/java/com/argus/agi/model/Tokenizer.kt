package com.argus.agi.model

import java.util.concurrent.ConcurrentHashMap

class Tokenizer {

    companion object {
        const val PAD_TOKEN = "<PAD>"
        const val UNK_TOKEN = "<UNK>"
        const val BOS_TOKEN = "<BOS>"
        const val EOS_TOKEN = "<EOS>"
        const val SYSTEM_TOKEN = "<SYSTEM>"
        const val USER_TOKEN = "<USER>"
        const val ASSISTANT_TOKEN = "<ASSISTANT>"
        const val THINK_START = "<THINK>"
        const val THINK_END = "</THINK>"

        const val PAD_ID = 0
        const val UNK_ID = 1
        const val BOS_ID = 2
        const val EOS_ID = 3
        const val SYSTEM_ID = 4
        const val USER_ID = 5
        const val ASSISTANT_ID = 6
        const val THINK_START_ID = 7
        const val THINK_END_ID = 8

        const val MAX_TOKEN_LEN = 16
        const val MAX_VOCAB_SIZE = 1024

        private val SPECIAL_TOKENS = listOf(
            THINK_START, THINK_END, SYSTEM_TOKEN, USER_TOKEN,
            ASSISTANT_TOKEN, BOS_TOKEN, EOS_TOKEN
        )

        private val SPECIAL_IDS: Map<String, Int> = linkedMapOf(
            PAD_TOKEN to PAD_ID,
            UNK_TOKEN to UNK_ID,
            BOS_TOKEN to BOS_ID,
            EOS_TOKEN to EOS_ID,
            SYSTEM_TOKEN to SYSTEM_ID,
            USER_TOKEN to USER_ID,
            ASSISTANT_TOKEN to ASSISTANT_ID,
            THINK_START to THINK_START_ID,
            THINK_END to THINK_END_ID
        )

        private val CONTROL_TOKENS: Set<String> = SPECIAL_IDS.keys +
                setOf(PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
    }

    // Atomic mutation via synchronized; reads are safe via CHM
    private val tokenToId = ConcurrentHashMap<String, Int>()
    private val idToToken = ConcurrentHashMap<Int, String>()

    @Volatile private var nextId = 0
    @Volatile private var maxRegisteredLen = 1

    init {
        initBaseVocabulary()
    }

    private fun initBaseVocabulary() {
        ensureSpecialTokens()

        val baseChars = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?:;-_+*=/\\@#$%^&()[]{}|<>\"'~\n\r\t“”‘’—–…«»"
        for (c in baseChars) {
            runCatching { registerToken(c.toString()) }
        }

        val seedWords = listOf(
            "yang", "dan", "di", "ini", "itu", "ke", "dari", "untuk", "pada", "adalah", "dengan",
            "saya", "kamu", "bisa", "ada", "tidak", "apa", "bagaimana", "argus", "agi", "model",
            "training", "data", "layar", "kamera", "the", "and", "is", "in", "to", "of", "for",
            "nama", "kita", "kami", "aneh", "dimana", "kenapa", "siapa", "serius", "bener"
        )
        for (w in seedWords) {
            runCatching { registerToken(w) }
        }
    }

    private fun ensureSpecialTokens() {
        for ((tok, id) in SPECIAL_IDS) {
            registerToken(tok, id)
        }
    }

    val vocabSize: Int
        get() = maxOf(nextId, tokenToId.size)

    @Synchronized
    fun registerToken(token: String, id: Int? = null): Int {
        require(token.isNotEmpty()) { "Token tidak boleh kosong" }
        require(token.length <= MAX_TOKEN_LEN) {
            "Token '$token' melebihi MAX_TOKEN_LEN=$MAX_TOKEN_LEN (len=${token.length})"
        }

        // Token sudah terdaftar? Kembalikan ID lama; tolak override eksplisit.
        tokenToId[token]?.let { existingId ->
            if (id != null && id != existingId) {
                throw IllegalArgumentException(
                    "Token '$token' sudah di id=$existingId, tidak bisa override ke id=$id"
                )
            }
            return existingId
        }

        // Guard kapasitas — berlaku untuk alokasi otomatis, TIDAK untuk special token / import
        if (id == null && tokenToId.size >= MAX_VOCAB_SIZE) {
            return UNK_ID
        }

        val resolved = id ?: nextId
        val occupant = idToToken[resolved]
        require(occupant == null) {
            "ID $resolved sudah dipakai token '$occupant', tidak bisa untuk '$token'"
        }

        // Validasi lulus — tulis ke kedua map
        idToToken[resolved] = token
        tokenToId[token] = resolved

        if (resolved >= nextId) nextId = resolved + 1
        if (token.length > maxRegisteredLen) maxRegisteredLen = token.length
        return resolved
    }

    @Synchronized
    fun exportVocab(): List<Pair<String, Int>> =
        tokenToId.entries.map { it.key to it.value }.sortedBy { it.second }

    /**
     * Restore vocab dari checkpoint secara atomik.
     *
     * Validasi dilakukan SEBELUM mutasi. Special token yang di-import wajib
     * memiliki ID canonical; override akan melempar exception.
     *
     * Catatan: metode ini melakukan clear + putAll di bawah lock. Jangan
     * memanggilnya secara bersamaan dengan encode() dari thread lain — panggil
     * saat cold start (genesis) sebelum tokenizer dipakai.
     */
    @Synchronized
    fun importVocab(entries: List<Pair<String, Int>>) {
        require(entries.size <= MAX_VOCAB_SIZE) {
            "Vocab import melebihi MAX_VOCAB_SIZE: ${entries.size} > $MAX_VOCAB_SIZE"
        }
        require(entries.map { it.second }.toSet().size == entries.size) {
            "Duplicate IDs pada vocab import"
        }
        require(entries.map { it.first }.toSet().size == entries.size) {
            "Duplicate tokens pada vocab import"
        }

        // Validasi canonical IDs untuk special tokens
        for ((tok, id) in entries) {
            val canonical = SPECIAL_IDS[tok]
            if (canonical != null && canonical != id) {
                throw IllegalArgumentException(
                    "Special token '$tok' wajib di id=$canonical, bukan id=$id"
                )
            }
        }

        // Bangun state baru secara lokal (validasi penuh sebelum commit)
        val newTokenToId = HashMap<String, Int>(entries.size * 2)
        val newIdToToken = HashMap<Int, String>(entries.size * 2)

        for ((tok, id) in SPECIAL_IDS) {
            newTokenToId[tok] = id
            newIdToToken[id] = tok
        }

        var newMaxLen = 1
        for ((tok, id) in entries) {
            require(tok.isNotEmpty()) { "Token kosong pada vocab import" }
            require(tok.length <= MAX_TOKEN_LEN) {
                "Token '$tok' melebihi MAX_TOKEN_LEN"
            }
            val occupant = newIdToToken[id]
            require(occupant == null || occupant == tok) {
                "ID $id bentrok: '$occupant' vs '$tok'"
            }
            newTokenToId[tok] = id
            newIdToToken[id] = tok
            if (tok.length > newMaxLen) newMaxLen = tok.length
        }

        // Commit
        tokenToId.clear(); tokenToId.putAll(newTokenToId)
        idToToken.clear(); idToToken.putAll(newIdToToken)
        nextId = (newIdToToken.keys.maxOrNull() ?: -1) + 1
        maxRegisteredLen = newMaxLen
    }

    fun encode(text: String, autoRegister: Boolean = true): List<Int> {
        val cleanText = TextModalityProcessor.normalizeTypography(text)
        val n = cleanText.length
        if (n == 0) return emptyList()

        val result = ArrayList<Int>(minOf(n, 256))
        val localMaxLen = maxRegisteredLen.coerceIn(1, MAX_TOKEN_LEN)
        var i = 0

        while (i < n) {
            val c = cleanText[i]

            // 1. Special tokens — semua diawali '<', jadi bisa dipangkas cepat
            if (c == '<') {
                var matched = false
                for (sp in SPECIAL_TOKENS) {
                    if (cleanText.startsWith(sp, i)) {
                        result.add(tokenToId[sp] ?: UNK_ID)
                        i += sp.length
                        matched = true
                        break
                    }
                }
                if (matched) continue
            }

            // 2. Greedy subword — batasi panjang ke maxRegisteredLen aktual
            var matchedSubword = false
            var len = minOf(localMaxLen, n - i)
            while (len >= 2) {
                val id = tokenToId[cleanText.substring(i, i + len)]
                if (id != null) {
                    result.add(id)
                    i += len
                    matchedSubword = true
                    break
                }
                len--
            }
            if (matchedSubword) continue

            // 3. Code point tunggal (surrogate pair / emoji safe)
            val codePoint = Character.codePointAt(cleanText, i)
            val charLen = Character.charCount(codePoint)
            val charStr = cleanText.substring(i, i + charLen)

            val id = tokenToId[charStr] ?: run {
                if (!autoRegister) return@run UNK_ID
                runCatching { registerToken(charStr) }.getOrDefault(UNK_ID)
            }
            result.add(id)
            i += charLen
        }

        return result
    }

    /**
     * Decode token ID → teks.
     *
     * @param keepControlTokens jika false (default), semua special token
     *        (<PAD>, <BOS>, <EOS>, <UNK>, <SYSTEM>, <USER>, <ASSISTANT>,
     *        <THINK>, </THINK>) dihapus dari output user-facing.
     *        Set true jika ingin merekonstruksi prompt terstruktur.
     */
    fun decode(tokenIds: List<Int>, keepControlTokens: Boolean = false): String {
        val sb = StringBuilder(tokenIds.size * 2)
        for (id in tokenIds) {
            val token = idToToken[id] ?: continue
            if (token.isEmpty()) continue
            if (!keepControlTokens && token in CONTROL_TOKENS) continue
            sb.append(token)
        }
        return sb.toString()
    }
}