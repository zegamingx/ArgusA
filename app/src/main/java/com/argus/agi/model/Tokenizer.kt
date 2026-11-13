package com.argus.agi.model

import java.util.concurrent.ConcurrentHashMap

enum class ArgusTokenizerAlgorithm(val checkpointId: Int) {
    GREEDY_LONGEST_MATCH_V1(1),
    LEARNED_BPE_V1(2);

    companion object {
        fun fromCheckpointId(value: Int): ArgusTokenizerAlgorithm =
            entries.firstOrNull { it.checkpointId == value }
                ?: throw IllegalArgumentException("Unsupported tokenizer algorithm ID: $value.")
    }
}

data class TokenizerConfig(
    val schemaVersion: Int = CURRENT_SCHEMA_VERSION,
    val algorithm: ArgusTokenizerAlgorithm = ArgusTokenizerAlgorithm.LEARNED_BPE_V1,
    val maxVocabularySize: Int = DEFAULT_MAX_VOCAB_SIZE,
    val maxTokenLength: Int = DEFAULT_MAX_TOKEN_LENGTH
) {
    init {
        require(schemaVersion == CURRENT_SCHEMA_VERSION) {
            "Unsupported tokenizer schema version: $schemaVersion."
        }
        require(maxVocabularySize >= MINIMUM_VOCAB_SIZE) {
            "Tokenizer capacity must be at least $MINIMUM_VOCAB_SIZE."
        }
        require(maxVocabularySize <= MAX_SUPPORTED_VOCAB_SIZE) {
            "Tokenizer capacity cannot exceed $MAX_SUPPORTED_VOCAB_SIZE."
        }
        require(maxTokenLength in MINIMUM_TOKEN_LENGTH..MAX_SUPPORTED_TOKEN_LENGTH) {
            "Tokenizer token length must be between $MINIMUM_TOKEN_LENGTH and $MAX_SUPPORTED_TOKEN_LENGTH."
        }
    }

    companion object {
        const val CURRENT_SCHEMA_VERSION = 1
        const val DEFAULT_MAX_VOCAB_SIZE = 2048
        const val DEFAULT_MAX_TOKEN_LENGTH = 16
        const val MINIMUM_VOCAB_SIZE = 256
        const val MINIMUM_TOKEN_LENGTH = 11
        const val MAX_SUPPORTED_VOCAB_SIZE = 1_000_000
        const val MAX_SUPPORTED_TOKEN_LENGTH = 256
    }
}

class Tokenizer(val config: TokenizerConfig = TokenizerConfig()) {

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
        private const val DEFAULT_MAX_MERGES_PER_TRAINING_PASS = 64
        private const val SEED_CAPACITY_DIVISOR = 4
        private const val MAX_INITIAL_SEED_WORDS = 256
        private const val MAX_INITIAL_SEED_MERGES = 512

        const val MAX_TOKEN_LEN = TokenizerConfig.DEFAULT_MAX_TOKEN_LENGTH
        const val MAX_VOCAB_SIZE = TokenizerConfig.DEFAULT_MAX_VOCAB_SIZE

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

// Derived from SPECIAL_IDS — single source of truth.
// Sorted longest-first for correct longest-match behavior
// (important if you ever add overlapping prefixes like <USER> vs <USER_X>).
        private val SPECIAL_TOKENS: List<String> =
        SPECIAL_IDS.keys.sortedByDescending { it.length }

        private val CONTROL_TOKENS: Set<String> = SPECIAL_IDS.keys +
                setOf(PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)
        private val REASONING_TOKENS: Set<String> = setOf(THINK_START, THINK_END)
    }

    // Atomic mutation via synchronized; reads are safe via CHM
    private val tokenToId = ConcurrentHashMap<String, Int>()
    private val idToToken = ConcurrentHashMap<Int, String>()
    private val merges = mutableListOf<Pair<String, String>>()
    private val mergeRanks = ConcurrentHashMap<Pair<String, String>, Int>()

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
    "<pad>", "<unk>", "<bos>", "<eos>", "<mask>", "yang", "dan", "di", "ini", "itu",
    "ke", "dari", "untuk", "pada", "adalah", "dengan", "saya", "kamu", "bisa", "ada",
    "tidak", "apa", "bagaimana", "argus", "agi", "model", "training", "data", "layar", "kamera",
    "the", "and", "is", "in", "to", "of", "for", "nama", "kita", "kami",
    "aneh", "dimana", "kenapa", "siapa", "serius", "bener", "aku", "gue", "gua", "gw",
    "lo", "lu", "elu", "dia", "mereka", "beliau", "kalian", "anda", "engkau", "kau",
    "ia", "nya", "ku", "mu", "makan", "minum", "tidur", "bangun", "jalan", "lari",
    "duduk", "berdiri", "pergi", "pulang", "datang", "melihat", "mendengar", "bicara", "ngomong", "bilang",
    "tanya", "jawab", "pikir", "tahu", "paham", "mengerti", "belajar", "mengajar", "baca", "tulis",
    "dengar", "lihat", "cari", "temukan", "buat", "bikin", "coba", "mulai", "selesai", "berhenti",
    "lanjut", "bantu", "tolong", "minta", "kasih", "beri", "ambil", "taruh", "simpan", "buka",
    "tutup", "nyala", "mati", "hidup", "jalankan", "pasang", "copot", "ganti", "ubah", "tambah",
    "kurang", "hapus", "edit", "kirim", "terima", "bayar", "beli", "jual", "pakai", "pake",
    "main", "kerja", "istirahat", "nonton", "dengerin", "nyanyi", "mainkan", "rumah", "sekolah", "kampus",
    "kantor", "mobil", "motor", "hp", "handphone", "komputer", "laptop", "internet", "wifi", "baterai",
    "charger", "kabel", "foto", "video", "lagu", "musik", "film", "game", "bola", "buku",
    "pena", "kertas", "meja", "kursi", "pintu", "jendela", "dapur", "kamar", "mandi", "air",
    "nasi", "roti", "kopi", "teh", "susu", "gula", "garam", "sayur", "buah", "daging",
    "ikan", "ayam", "telur", "mie", "bakso", "sate", "baju", "celana", "sepatu", "tas",
    "dompet", "kunci", "uang", "toko", "pasar", "mall", "raya", "desa", "kota", "negara",
    "dunia", "langit", "bumi", "matahari", "bulan", "bintang", "hujan", "angin", "api", "tanah",
    "pohon", "bunga", "kucing", "anjing", "burung", "manusia", "anak", "bayi", "ibu", "bapak",
    "kakak", "adik", "teman", "sahabat", "pacar", "keluarga", "guru", "dosen", "dokter", "polisi",
    "tentara", "petani", "besar", "kecil", "tinggi", "rendah", "panjang", "pendek", "lebar", "sempit",
    "cepat", "lambat", "panas", "dingin", "terang", "gelap", "baru", "lama", "bagus", "jelek",
    "cantik", "ganteng", "pintar", "bodoh", "baik", "jahat", "senang", "sedih", "marah", "takut",
    "capek", "lelah", "sehat", "sakit", "kaya", "miskin", "mahal", "murah", "ramai", "sepi",
    "bersih", "kotor", "basah", "kering", "keras", "lembut", "kuat", "lemah", "berat", "ringan",
    "jauh", "dekat", "penuh", "kosong", "benar", "salah", "jelas", "rahasia", "penting", "berbahaya",
    "aman", "sulit", "mudah", "susah", "asyik", "seru", "lucu", "keren", "mantap", "unik",
    "biasa", "hari", "minggu", "tahun", "pagi", "siang", "sore", "malam", "sekarang", "tadi",
    "nanti", "besok", "kemarin", "lusa", "dulu", "sebentar", "jam", "menit", "detik", "waktu",
    "saat", "ketika", "sebelum", "sesudah", "setelah", "selama", "sampai", "sejak", "setiap", "selalu",
    "sering", "jarang", "pernah", "belum", "sudah", "masih", "lagi", "akan", "sedang", "telah",
    "kapan", "berapa", "apakah", "bukan", "jangan", "boleh", "harus", "perlu", "mau", "ingin",
    "suka", "benci", "deh", "dong", "sih", "kok", "lah", "nih", "tuh", "gitu",
    "gini", "gak", "nggak", "ga", "tau", "udah", "aja", "yuk", "ayo", "gas",
    "bro", "sis", "cuy", "woy", "anjir", "buset", "gila", "santai", "oke", "ok",
    "siap", "halo", "hai", "hehe", "haha", "wkwk", "ya", "iya", "atau", "tapi",
    "tetapi", "karena", "kalau", "jika", "bila", "agar", "supaya", "meskipun", "walaupun", "sedangkan",
    "lalu", "kemudian", "terus", "juga", "bahkan", "hanya", "cuma", "saja", "pun", "malah",
    "justru", "apalagi", "makanya", "soalnya", "padahal", "ternyata", "akhirnya", "sebenarnya", "memang", "satu",
    "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan", "sepuluh", "sebelas",
    "belas", "seratus", "seribu", "nol", "pertama", "kedua", "ketiga", "banyak", "sedikit", "semua",
    "beberapa", "lain", "sendiri", "masing", "neural", "ai", "robot", "sistem", "program", "kode",
    "aplikasi", "android", "telepon", "sensor", "mata", "telinga", "otak", "pikiran", "memori", "latih",
    "dataset", "token", "vocab", "parameter", "neuron", "layer", "embedding", "loss", "epoch", "batch",
    "learning", "rate", "optimizer", "checkpoint", "file", "folder", "json", "jsonl", "prompt", "reply",
    "chat", "bot", "asisten", "user", "admin", "server", "cloud", "colab", "drive", "google",
    "github", "studio", "jstudio", "kotlin", "java", "python", "error", "bug", "fix", "update",
    "install", "download", "upload", "login", "logout", "password", "akun", "email", "nomor", "pesan",
    "notifikasi", "setting"
)
        // Keep the bundled vocabulary bounded so small model profiles retain
        // capacity for corpus-derived tokens and tokenizer training.
        val seedWordLimit = minOf(
            seedWords.size,
            MAX_INITIAL_SEED_WORDS,
            config.maxVocabularySize / SEED_CAPACITY_DIVISOR
        )
        val bundledSeedWords = seedWords
            .asSequence()
            .filterNot(::isMarkupSeedToken)
            .take(seedWordLimit)
            .toList()

        for (word in bundledSeedWords) {
            runCatching { registerToken(word) }
        }
        if (config.algorithm == ArgusTokenizerAlgorithm.LEARNED_BPE_V1) {
            initializeSeedMerges(bundledSeedWords)
        }
    }

    /**
     * Seed merges make the bundled base-word tokens usable by BPE without
     * allocating additional vocabulary entries or changing their stable IDs.
     */
    private fun isMarkupSeedToken(token: String): Boolean =
        token.length >= 3 && token.first() == '<' && token.last() == '>'

    private fun initializeSeedMerges(seedWords: List<String>) {
        val mergeLimit = minOf(
            MAX_INITIAL_SEED_MERGES,
            config.maxVocabularySize / SEED_CAPACITY_DIVISOR
        )
        for (word in seedWords) {
            if (merges.size >= mergeLimit) break

            val pieces = word.map { character -> character.toString() }.toMutableList()
            while (pieces.size > 1 && merges.size < mergeLimit) {
                val pair = pieces[0] to pieces[1]
                if (!mergeRanks.containsKey(pair)) {
                    merges.add(pair)
                    mergeRanks[pair] = merges.lastIndex
                }
                pieces[0] = pair.first + pair.second
                pieces.removeAt(1)
            }
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
        require(token.length <= config.maxTokenLength) {
            "Token length ${token.length} exceeds configured maxTokenLength=${config.maxTokenLength}."
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
        if (id == null && nextId >= config.maxVocabularySize) {
            return UNK_ID
        }

        val resolved = id ?: nextId
        require(resolved in 0 until config.maxVocabularySize) {
            "Token ID $resolved exceeds configured vocabulary capacity ${config.maxVocabularySize}."
        }
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
        require(entries.size <= config.maxVocabularySize) {
            "Vocabulary import exceeds configured capacity ${config.maxVocabularySize}."
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
            require(tok.length <= config.maxTokenLength) {
                "Token length ${tok.length} exceeds configured maxTokenLength=${config.maxTokenLength}."
            }
            require(id in 0 until config.maxVocabularySize) {
                "Token ID $id exceeds configured vocabulary capacity ${config.maxVocabularySize}."
            }
            val occupant = newIdToToken[id]
            require(occupant == null || occupant == tok) {
                "ID $id bentrok: '$occupant' vs '$tok'"
            }
            newTokenToId[tok] = id
            newIdToToken[id] = tok
            if (tok.length > newMaxLen) newMaxLen = tok.length
        }

        require(newTokenToId.size <= config.maxVocabularySize) {
            "Imported vocabulary plus canonical tokens exceeds configured capacity."
        }

        // Commit
        tokenToId.clear(); tokenToId.putAll(newTokenToId)
        idToToken.clear(); idToToken.putAll(newIdToToken)
        nextId = (newIdToToken.keys.maxOrNull() ?: -1) + 1
        maxRegisteredLen = newMaxLen
    }

    @Synchronized
    fun exportMerges(): List<Pair<String, String>> = ArrayList(merges)

    val mergeCount: Int
        get() = synchronized(this) { merges.size }

    @Synchronized
    fun importMerges(importedMerges: List<Pair<String, String>>) {
        merges.clear()
        mergeRanks.clear()
        for ((idx, pair) in importedMerges.withIndex()) {
            require(pair.first.isNotEmpty() && pair.second.isNotEmpty()) {
                "Merge pair components must not be empty."
            }
            merges.add(pair)
            mergeRanks[pair] = idx
        }
    }

    /**
     * Learns deterministic BPE merge rules from a bounded corpus sample.
     *
     * Existing merge rules are replayed before new pairs are counted. This makes
     * incremental corpus learning safe: previously learned token IDs and merge
     * priorities remain stable, and the same corpus does not append duplicate rules.
     */
    @Synchronized
    fun trainBpe(
        corpus: List<String>,
        targetVocabSize: Int = config.maxVocabularySize,
        maxNewMerges: Int = DEFAULT_MAX_MERGES_PER_TRAINING_PASS
    ): Int {
        require(maxNewMerges > 0) { "maxNewMerges must be positive." }
        val target = minOf(targetVocabSize, config.maxVocabularySize)
        if (vocabSize >= target || corpus.isEmpty()) return vocabSize

        val wordFrequencies = LinkedHashMap<String, Int>()
        for (document in corpus) {
            var normalized = TextModalityProcessor.normalizeTypography(document)
            // Control tokens already have fixed IDs and are split atomically during
            // encoding. Do not spend BPE merges learning fragments of their markup,
            // including when a marker touches the surrounding text.
            for (specialToken in SPECIAL_TOKENS) {
                normalized = normalized.replace(specialToken, " ")
            }
            for (word in normalized.split(Regex("\\s+"))) {
                if (word.isEmpty()) continue
                val frequency = wordFrequencies[word] ?: 0
                wordFrequencies[word] = if (frequency == Int.MAX_VALUE) frequency else frequency + 1
            }
        }
        if (wordFrequencies.isEmpty()) return vocabSize

        val wordSymbols = LinkedHashMap<String, MutableList<String>>(wordFrequencies.size)
        for (word in wordFrequencies.keys) {
            val symbols = mutableListOf<String>()
            var index = 0
            while (index < word.length) {
                val codePoint = Character.codePointAt(word, index)
                val codePointLength = Character.charCount(codePoint)
                symbols.add(word.substring(index, index + codePointLength))
                index += codePointLength
            }
            wordSymbols[word] = symbols
        }

        // Continue from the existing BPE state instead of relearning old merges.
        for (merge in merges) {
            val combined = merge.first + merge.second
            for (symbols in wordSymbols.values) {
                applyMerge(symbols, merge.first, merge.second, combined)
            }
        }

        var newMergeCount = 0
        while (vocabSize < target && newMergeCount < maxNewMerges) {
            val pairCounts = LinkedHashMap<Pair<String, String>, Int>()
            for ((word, frequency) in wordFrequencies) {
                val symbols = wordSymbols[word] ?: continue
                for (index in 0 until symbols.size - 1) {
                    val pair = symbols[index] to symbols[index + 1]
                    if (mergeRanks.containsKey(pair)) continue
                    if (pair.first.length + pair.second.length > config.maxTokenLength) continue
                    val previous = pairCounts[pair] ?: 0
                    pairCounts[pair] = if (previous > Int.MAX_VALUE - frequency) {
                        Int.MAX_VALUE
                    } else {
                        previous + frequency
                    }
                }
            }

            val bestEntry = pairCounts.entries.maxByOrNull { it.value } ?: break
            if (bestEntry.value < 2) break
            val bestPair = bestEntry.key
            val combined = bestPair.first + bestPair.second
            if (combined.length > config.maxTokenLength) break

            val tokenId = tokenToId[combined] ?: registerToken(combined)
            if (tokenId == UNK_ID && combined != UNK_TOKEN) break

            merges.add(bestPair)
            mergeRanks[bestPair] = merges.lastIndex
            newMergeCount++
            for (symbols in wordSymbols.values) {
                applyMerge(symbols, bestPair.first, bestPair.second, combined)
            }
        }

        return vocabSize
    }

    private fun applyMerge(
        symbols: MutableList<String>,
        left: String,
        right: String,
        combined: String
    ) {
        var index = 0
        while (index < symbols.size - 1) {
            if (symbols[index] == left && symbols[index + 1] == right) {
                symbols[index] = combined
                symbols.removeAt(index + 1)
            } else {
                index++
            }
        }
    }

    private fun encodeLearnedBpe(cleanText: String, autoRegister: Boolean): List<Int> {
        val n = cleanText.length
        val result = ArrayList<Int>(minOf(n, 256))
        var i = 0

        while (i < n) {
            val c = cleanText[i]

            // 1. Special tokens
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

            // Find next special token or end of text
            var segmentEnd = n
            for (sp in SPECIAL_TOKENS) {
                val spIdx = cleanText.indexOf(sp, i)
                if (spIdx in i until segmentEnd) {
                    segmentEnd = spIdx
                }
            }
            val segment = cleanText.substring(i, segmentEnd)
            i = segmentEnd

            // Split segment into initial code point pieces
            val pieces = mutableListOf<String>()
            var segIdx = 0
            while (segIdx < segment.length) {
                val cp = Character.codePointAt(segment, segIdx)
                val len = Character.charCount(cp)
                pieces.add(segment.substring(segIdx, segIdx + len))
                segIdx += len
            }

            // Apply BPE merges greedily by priority rank
            if (merges.isNotEmpty()) {
                while (pieces.size >= 2) {
                    var minRank = Int.MAX_VALUE
                    var bestIdx = -1

                    for (pIdx in 0 until pieces.size - 1) {
                        val pair = pieces[pIdx] to pieces[pIdx + 1]
                        val rank = mergeRanks[pair]
                        if (rank != null && rank < minRank) {
                            minRank = rank
                            bestIdx = pIdx
                        }
                    }

                    if (bestIdx == -1) break

                    val mergedPiece = pieces[bestIdx] + pieces[bestIdx + 1]
                    pieces[bestIdx] = mergedPiece
                    pieces.removeAt(bestIdx + 1)
                }
            }

            // Convert pieces to token IDs
            for (piece in pieces) {
                if (piece == "~") {
                    val tildeId = tokenToId["~"] ?: UNK_ID
                    result.add(tildeId)
                    result.add(tildeId)
                    continue
                }

                val id = tokenToId[piece]
                if (id != null) {
                    result.add(id)
                } else if (autoRegister) {
                    result.add(runCatching { registerToken(piece) }.getOrDefault(UNK_ID))
                } else {
                    var pieceIdx = 0
                    while (pieceIdx < piece.length) {
                        val cp = Character.codePointAt(piece, pieceIdx)
                        val len = Character.charCount(cp)
                        val charStr = piece.substring(pieceIdx, pieceIdx + len)
                        val charId = tokenToId[charStr]
                        if (charId != null) {
                            result.add(charId)
                        } else {
                            val codePointMarker = "~u{${cp.toString(16).uppercase(java.util.Locale.ROOT)}}~"
                            for (markerChar in codePointMarker) {
                                result.add(tokenToId[markerChar.toString()] ?: UNK_ID)
                            }
                        }
                        pieceIdx += len
                    }
                }
            }
        }

        return result
    }

    fun encode(text: String, autoRegister: Boolean = false): List<Int> {
        val cleanText = TextModalityProcessor.normalizeTypography(text)
        val n = cleanText.length
        if (n == 0) return emptyList()

        if (config.algorithm == ArgusTokenizerAlgorithm.LEARNED_BPE_V1) {
            return encodeLearnedBpe(cleanText, autoRegister)
        }

        val result = ArrayList<Int>(minOf(n, 256))
        val localMaxLen = maxRegisteredLen.coerceIn(1, config.maxTokenLength)
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

            if (charStr == "~") {
                val tildeId = tokenToId["~"] ?: UNK_ID
                result.add(tildeId)
                result.add(tildeId)
                i += charLen
                continue
            }

            val existingId = tokenToId[charStr]
            if (existingId != null) {
                result.add(existingId)
            } else if (autoRegister) {
                result.add(runCatching { registerToken(charStr) }.getOrDefault(UNK_ID))
            } else {
                // Stable Unicode escape fallback encoded with existing ASCII token IDs.
                // No vocabulary or embedding resize is needed for unseen code points.
                val codePointMarker = "~u{${codePoint.toString(16).uppercase(java.util.Locale.ROOT)}}~"
                for (markerChar in codePointMarker) {
                    result.add(tokenToId[markerChar.toString()] ?: UNK_ID)
                }
            }
            i += charLen
        }

        return result
    }

    /**
     * Decode token ID → teks.
     *
     * @param keepControlTokens jika false (default), semua special token
     *        selain penanda reasoning yang diminta secara eksplisit dihapus.
     * @param keepReasoningTokens jika true, hanya <THINK> dan </THINK> yang
     *        dipertahankan sementara token kontrol lain tetap disembunyikan.
     */
    fun decode(
        tokenIds: List<Int>,
        keepControlTokens: Boolean = false,
        keepReasoningTokens: Boolean = false
    ): String {
        val sb = StringBuilder(tokenIds.size * 2)
        for (id in tokenIds) {
            val token = idToToken[id] ?: continue
            if (token.isEmpty()) continue
            val preserveToken = keepControlTokens ||
                (keepReasoningTokens && token in REASONING_TOKENS)
            if (!preserveToken && token in CONTROL_TOKENS) continue
            sb.append(token)
        }
        return decodeUnicodeFallbacks(sb.toString())
    }

    private fun decodeUnicodeFallbacks(encoded: String): String {
        val decoded = StringBuilder(encoded.length)
        var index = 0

        while (index < encoded.length) {
            if (encoded.startsWith("~~", index)) {
                decoded.append('~')
                index += 2
                continue
            }

            if (encoded.startsWith("~u{", index)) {
                val terminator = encoded.indexOf("}~", startIndex = index + 3)
                if (terminator >= 0) {
                    val codePoint = encoded.substring(index + 3, terminator).toIntOrNull(16)
                    if (
                        codePoint != null &&
                        Character.isValidCodePoint(codePoint) &&
                        codePoint !in 0xD800..0xDFFF
                    ) {
                        decoded.appendCodePoint(codePoint)
                        index = terminator + 2
                        continue
                    }
                }
            }

            decoded.append(encoded[index])
            index++
        }

        return decoded.toString()
    }
}
