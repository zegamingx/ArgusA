package com.argus.agi.model

import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.nio.ByteBuffer
import java.nio.CharBuffer
import java.nio.charset.CharacterCodingException
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.PriorityQueue
import java.util.concurrent.ConcurrentHashMap

enum class ArgusTokenizerAlgorithm(val checkpointId: Int) {
    GREEDY_LONGEST_MATCH_V1(1),
    LEARNED_BPE_V1(2),
    BYTE_LEVEL_BPE_V2(3);

    val usesBpeMerges: Boolean
        get() = this == LEARNED_BPE_V1 || this == BYTE_LEVEL_BPE_V2

    companion object {
        fun fromCheckpointId(value: Int): ArgusTokenizerAlgorithm =
            entries.firstOrNull { it.checkpointId == value }
                ?: throw IllegalArgumentException("Unsupported tokenizer algorithm ID: $value.")
    }
}

data class TokenizerConfig(
    val schemaVersion: Int = CURRENT_SCHEMA_VERSION,
    val algorithm: ArgusTokenizerAlgorithm = ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V2,
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
        const val MINIMUM_VOCAB_SIZE = 265
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
        private const val FIRST_BYTE_TOKEN_ID = 9
        private const val BYTE_ALPHABET_SIZE = 256
        private const val SNAPSHOT_MAGIC = 0x41544B53 // "ATKS"
        const val CURRENT_SNAPSHOT_FORMAT_VERSION = 1
        private const val SNAPSHOT_DIGEST_BYTES = 32
        private const val MAX_SNAPSHOT_BYTES = 128 * 1024 * 1024
        private val BYTE_LEVEL_PRETOKENIZER = Regex("(?U)\\s+|[^\\s]+")

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

    // All public reads and writes share this tokenizer's monitor.
    private val tokenToId = ConcurrentHashMap<String, Int>()
    private val idToToken = ConcurrentHashMap<Int, String>()
    private val merges = mutableListOf<Pair<String, String>>()
    // Nested lookup avoids allocating a Pair for every adjacent-token probe at encode time.
    private val mergeRanks = HashMap<String, HashMap<String, Int>>()

    @Volatile private var nextId = 0
    @Volatile private var maxRegisteredLen = 1

    private fun hasMerge(left: String, right: String): Boolean =
        mergeRanks[left]?.containsKey(right) == true

    private fun mergeRank(left: String, right: String): Int? =
        mergeRanks[left]?.get(right)

    private fun putMergeRank(left: String, right: String, rank: Int) {
        mergeRanks.getOrPut(left) { HashMap() }[right] = rank
    }

    init {
        initBaseVocabulary()
    }

    private fun initBaseVocabulary() {
        ensureSpecialTokens()

        if (config.algorithm == ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V2) {
            for (byteValue in 0 until BYTE_ALPHABET_SIZE) {
                registerToken(
                    byteSymbol(byteValue),
                    FIRST_BYTE_TOKEN_ID + byteValue
                )
            }
            return
        }

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
                if (!hasMerge(pair.first, pair.second)) {
                    merges.add(pair)
                    putMergeRank(pair.first, pair.second, merges.lastIndex)
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
        @Synchronized get() = maxOf(nextId, tokenToId.size)

    internal fun baseByteTokenId(byteValue: Int): Int {
        require(byteValue in 0 until BYTE_ALPHABET_SIZE) {
            "Byte value must be in 0..255."
        }
        require(config.algorithm == ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V2) {
            "The byte alphabet is available only for BYTE_LEVEL_BPE_V2."
        }
        return FIRST_BYTE_TOKEN_ID + byteValue
    }

    @Synchronized
    fun registerToken(token: String, id: Int? = null): Int {
        require(token.isNotEmpty()) { "Token tidak boleh kosong" }
        require(token.length <= config.maxTokenLength) {
            "Token length ${token.length} exceeds configured maxTokenLength=${config.maxTokenLength}."
        }
        if (config.algorithm == ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V2) {
            require(token.all { character -> character.code <= 0xFF || token in SPECIAL_IDS }) {
                "Byte-Level BPE vocabulary entries must use the reversible 0..255 byte alphabet."
            }
        }
        if (
            config.algorithm == ArgusTokenizerAlgorithm.LEARNED_BPE_V1 &&
            token != "~" &&
            token.contains('~')
        ) {
            throw IllegalArgumentException(
                "LEARNED_BPE_V1 reserves '~' for its legacy Unicode escape format."
            )
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

        // Capacity exhaustion is explicit. Callers must choose a fallback deliberately;
        // silently returning UNK would corrupt learned text and hide vocabulary overflow.
        if (id == null && nextId >= config.maxVocabularySize) {
            throw IllegalStateException(
                "Tokenizer vocabulary capacity ${config.maxVocabularySize} is exhausted."
            )
        }

        val resolved = id ?: nextId
        if (config.algorithm == ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V2) {
            require(resolved == nextId) {
                "Byte-Level BPE vocabulary IDs are append-only; expected ID $nextId, received $resolved."
            }
        }
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
     * The complete replacement is validated before commit. Encoding, decoding,
     * vocabulary reads, merge reads, and imports use the same monitor, so readers
     * cannot observe a partially replaced vocabulary.
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

        val requiredEntries = requiredBaseEntries()
        val importedIdsByToken = entries.toMap()
        for ((token, canonicalId) in requiredEntries) {
            val importedId = importedIdsByToken[token]
            require(importedId == null || importedId == canonicalId) {
                "Canonical token '$token' must keep ID $canonicalId, not $importedId."
            }
        }

        // Build both maps completely before publishing either one. All public
        // readers (encode/decode/export/vocabSize) use this same monitor.
        val newTokenToId = HashMap<String, Int>(entries.size * 2 + requiredEntries.size)
        val newIdToToken = HashMap<Int, String>(entries.size * 2 + requiredEntries.size)
        var newMaxLen = 1

        for ((token, id) in requiredEntries) {
            require(token.length <= config.maxTokenLength) {
                "Canonical token '$token' exceeds the configured token length."
            }
            newTokenToId[token] = id
            newIdToToken[id] = token
            newMaxLen = maxOf(newMaxLen, token.length)
        }

        for ((token, id) in entries) {
            require(token.isNotEmpty()) { "Token kosong pada vocab import" }
            require(token.length <= config.maxTokenLength) {
                "Token length ${token.length} exceeds configured maxTokenLength=${config.maxTokenLength}."
            }
            require(id in 0 until config.maxVocabularySize) {
                "Token ID $id exceeds configured vocabulary capacity ${config.maxVocabularySize}."
            }
            if (config.algorithm == ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V2 &&
                token !in SPECIAL_IDS
            ) {
                require(token.all { character -> character.code <= 0xFF }) {
                    "Imported Byte-Level BPE tokens must contain only byte symbols."
                }
            }
            if (
                config.algorithm == ArgusTokenizerAlgorithm.LEARNED_BPE_V1 &&
                token != "~" &&
                token.contains('~')
            ) {
                throw IllegalArgumentException(
                    "Imported LEARNED_BPE_V1 tokens cannot contain the reserved '~' escape marker."
                )
            }

            val existingTokenId = newTokenToId[token]
            require(existingTokenId == null || existingTokenId == id) {
                "Token '$token' is assigned to both ID $existingTokenId and ID $id."
            }
            val occupant = newIdToToken[id]
            require(occupant == null || occupant == token) {
                "ID $id bentrok: '$occupant' vs '$token'"
            }

            newTokenToId[token] = id
            newIdToToken[id] = token
            newMaxLen = maxOf(newMaxLen, token.length)
        }

        require(newTokenToId.size <= config.maxVocabularySize) {
            "Imported vocabulary plus canonical tokens exceeds configured capacity."
        }

        if (config.algorithm == ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V2) {
            val maximumId = newIdToToken.keys.maxOrNull() ?: -1
            require(newIdToToken.size == maximumId + 1) {
                "Byte-Level BPE vocabulary IDs must be contiguous and append-only."
            }
        }

        // Atomic publication relative to every synchronized tokenizer operation.
        tokenToId.clear()
        tokenToId.putAll(newTokenToId)
        idToToken.clear()
        idToToken.putAll(newIdToToken)
        nextId = (newIdToToken.keys.maxOrNull() ?: -1) + 1
        maxRegisteredLen = newMaxLen
    }

    private fun requiredBaseEntries(): List<Pair<String, Int>> {
        val required = ArrayList<Pair<String, Int>>(SPECIAL_IDS.size + BYTE_ALPHABET_SIZE)
        for ((token, id) in SPECIAL_IDS) {
            required += token to id
        }
        if (config.algorithm == ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V2) {
            for (byteValue in 0 until BYTE_ALPHABET_SIZE) {
                required += byteSymbol(byteValue) to (FIRST_BYTE_TOKEN_ID + byteValue)
            }
        }
        return required
    }

    @Synchronized
    fun exportMerges(): List<Pair<String, String>> = ArrayList(merges)

    val mergeCount: Int
        get() = synchronized(this) { merges.size }

    @Synchronized
    fun importMerges(importedMerges: List<Pair<String, String>>) {
        require(config.algorithm.usesBpeMerges) {
            "Merge rules are not supported by ${config.algorithm}."
        }
        require(importedMerges.size <= config.maxVocabularySize) {
            "Merge count exceeds configured vocabulary capacity."
        }

        // Validate into a temporary map first. A malformed item must never leave
        // the active merge table partially cleared or partially populated.
        val candidateRanks = HashMap<Pair<String, String>, Int>(importedMerges.size * 2)
        for ((index, pair) in importedMerges.withIndex()) {
            val (left, right) = pair
            require(left.isNotEmpty() && right.isNotEmpty()) {
                "Merge pair components must not be empty."
            }
            require(left.length + right.length <= config.maxTokenLength) {
                "Merge result exceeds configured maximum token length."
            }
            require(left !in CONTROL_TOKENS && right !in CONTROL_TOKENS) {
                "Special tokens cannot be merge components."
            }
            val combined = left + right
            require(combined !in CONTROL_TOKENS) {
                "A merge cannot resolve to a reserved control token."
            }
            if (config.algorithm == ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V2) {
                require(tokenToId.containsKey(left) && tokenToId.containsKey(right)) {
                    "Byte-Level BPE merge components must already exist in the vocabulary."
                }
                require(tokenToId.containsKey(combined)) {
                    "Merged byte-level token '$combined' is missing from the vocabulary."
                }
                require(combined.all { character -> character.code <= 0xFF }) {
                    "Byte-Level BPE merges must remain inside the byte-symbol alphabet."
                }
            }
            if (config.algorithm == ArgusTokenizerAlgorithm.LEARNED_BPE_V1) {
                // Legacy seed merges intentionally contain intermediate symbols
                // (for example "ya") that are not standalone vocabulary entries.
                require(!left.contains('~') && !right.contains('~')) {
                    "LEARNED_BPE_V1 merge rules cannot consume the reserved '~' escape marker."
                }
            }
            require(candidateRanks.put(pair, index) == null) {
                "Duplicate BPE merge pair: '$left' + '$right'."
            }
        }

        merges.clear()
        merges.addAll(importedMerges)
        mergeRanks.clear()
        for ((pair, rank) in candidateRanks) {
            putMergeRank(pair.first, pair.second, rank)
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
        // Legacy GREEDY_LONGEST_MATCH_V1 may still register corpus-derived
        // vocabulary entries; its encoder intentionally ignores the merge table.
        require(maxNewMerges > 0) { "maxNewMerges must be positive." }
        val target = minOf(targetVocabSize, config.maxVocabularySize)
        if (vocabSize >= target || corpus.isEmpty()) return vocabSize

        val wordFrequencies = LinkedHashMap<String, Int>()

        fun addWordFrequency(word: String) {
            if (word.isEmpty()) return
            val previous = wordFrequencies[word] ?: 0
            wordFrequencies[word] = if (previous == Int.MAX_VALUE) {
                Int.MAX_VALUE
            } else {
                previous + 1
            }
        }

        for (document in corpus) {
            if (config.algorithm == ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V2) {
                // Byte-Level BPE is intentionally lossless: no typography
                // normalization, trimming, or Unicode code-point splitting.
                forEachNormalTextSegment(document) { segment ->
                    for (match in BYTE_LEVEL_PRETOKENIZER.findAll(segment)) {
                        addWordFrequency(match.value)
                    }
                }
            } else {
                val normalized = TextModalityProcessor.normalizeTypography(document)
                forEachNormalTextSegment(normalized) { segment ->
                    for (word in segment.split(Regex("\\s+"))) {
                        addWordFrequency(word)
                    }
                }
            }
        }
        if (wordFrequencies.isEmpty()) return vocabSize

        val wordSymbols = LinkedHashMap<String, MutableList<String>>(wordFrequencies.size)
        for (word in wordFrequencies.keys) {
            val symbols = mutableListOf<String>()
            if (config.algorithm == ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V2) {
                for (byte in encodeUtf8Strict(word)) {
                    symbols.add(byteSymbol(byte.toInt() and 0xFF))
                }
            } else {
                var index = 0
                while (index < word.length) {
                    val codePoint = Character.codePointAt(word, index)
                    val codePointLength = Character.charCount(codePoint)
                    symbols.add(word.substring(index, index + codePointLength))
                    index += codePointLength
                }
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
                    if (hasMerge(pair.first, pair.second)) continue
                    if (
                        config.algorithm == ArgusTokenizerAlgorithm.LEARNED_BPE_V1 &&
                        (pair.first.contains('~') || pair.second.contains('~'))
                    ) continue
                    if (pair.first.length + pair.second.length > config.maxTokenLength) continue
                    val previous = pairCounts[pair] ?: 0
                    pairCounts[pair] = if (previous > Int.MAX_VALUE - frequency) {
                        Int.MAX_VALUE
                    } else {
                        previous + frequency
                    }
                }
            }

            // Count wins first; lexical byte-symbol order breaks ties so the
            // learned merge table is independent of corpus document ordering.
            val bestEntry = pairCounts.entries.maxWithOrNull(
                compareBy<Map.Entry<Pair<String, String>, Int>> { entry -> entry.value }
                    .thenBy { entry -> entry.key.first }
                    .thenBy { entry -> entry.key.second }
            ) ?: break
            if (bestEntry.value < 2) break
            val bestPair = bestEntry.key
            val combined = bestPair.first + bestPair.second
            if (combined.length > config.maxTokenLength) break

            val tokenId = tokenToId[combined] ?: runCatching {
                registerToken(combined)
            }.getOrNull() ?: break
            if (tokenId == UNK_ID && combined != UNK_TOKEN) break

            merges.add(bestPair)
            putMergeRank(bestPair.first, bestPair.second, merges.lastIndex)
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
        // Single forward compaction pass. Repeated removeAt(index) shifted the
        // remaining tail for every match and made this operation quadratic.
        var readIndex = 0
        var writeIndex = 0
        while (readIndex < symbols.size) {
            if (
                readIndex + 1 < symbols.size &&
                symbols[readIndex] == left &&
                symbols[readIndex + 1] == right
            ) {
                symbols[writeIndex] = combined
                writeIndex++
                readIndex += 2
            } else {
                symbols[writeIndex] = symbols[readIndex]
                writeIndex++
                readIndex++
            }
        }
        while (symbols.size > writeIndex) {
            symbols.removeAt(symbols.lastIndex)
        }
    }

    private class BpeNode(
        val position: Int,
        var value: String
    ) {
        var previous: BpeNode? = null
        var next: BpeNode? = null
        var generation: Int = 0
        var active: Boolean = true
    }

    private data class MergeCandidate(
        val rank: Int,
        val position: Int,
        val left: BpeNode,
        val right: BpeNode,
        val leftGeneration: Int,
        val rightGeneration: Int
    )

    private fun byteSymbol(byteValue: Int): String =
        (byteValue and 0xFF).toChar().toString()

    private fun encodeUtf8Strict(text: String): ByteArray {
        return try {
            val encoded = StandardCharsets.UTF_8.newEncoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .encode(CharBuffer.wrap(text))
            val bytes = ByteArray(encoded.remaining())
            encoded.get(bytes)
            bytes
        } catch (exception: CharacterCodingException) {
            throw IllegalArgumentException(
                "Tokenizer input contains malformed UTF-16 and cannot be encoded as UTF-8.",
                exception
            )
        }
    }

    private fun decodeUtf8Strict(bytes: ByteArray): String {
        return try {
            StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(bytes))
                .toString()
        } catch (exception: CharacterCodingException) {
            throw IllegalStateException(
                "Tokenizer byte sequence is not valid UTF-8.",
                exception
            )
        }
    }

    private fun forEachNormalTextSegment(
        text: String,
        action: (String) -> Unit
    ) {
        var cursor = 0
        var segmentStart = 0
        while (cursor < text.length) {
            val specialToken = if (text[cursor] == '<') {
                SPECIAL_TOKENS.firstOrNull { token -> text.startsWith(token, cursor) }
            } else {
                null
            }
            if (specialToken != null) {
                if (segmentStart < cursor) {
                    action(text.substring(segmentStart, cursor))
                }
                cursor += specialToken.length
                segmentStart = cursor
            } else {
                cursor++
            }
        }
        if (segmentStart < text.length) {
            action(text.substring(segmentStart))
        }
    }

    /**
     * Applies the learned merge ranks with a min-heap and a linked token sequence.
     * Stale candidates are rejected by node generations, so each local neighborhood
     * is reconsidered only when one of its nodes changes.
     */
    private fun applyRankedMerges(initialSymbols: List<String>): List<String> {
        if (initialSymbols.size < 2 || mergeRanks.isEmpty()) return initialSymbols

        val nodes = ArrayList<BpeNode>(initialSymbols.size)
        for ((index, symbol) in initialSymbols.withIndex()) {
            nodes += BpeNode(index, symbol)
        }
        for (index in 0 until nodes.lastIndex) {
            nodes[index].next = nodes[index + 1]
            nodes[index + 1].previous = nodes[index]
        }

        val queue = PriorityQueue<MergeCandidate>(
            compareBy<MergeCandidate> { candidate -> candidate.rank }
                .thenBy { candidate -> candidate.position }
        )

        fun offer(left: BpeNode?, right: BpeNode?) {
            if (left == null || right == null || !left.active || !right.active) return
            val rank = mergeRank(left.value, right.value) ?: return
            queue.add(
                MergeCandidate(
                    rank = rank,
                    position = left.position,
                    left = left,
                    right = right,
                    leftGeneration = left.generation,
                    rightGeneration = right.generation
                )
            )
        }

        for (index in 0 until nodes.lastIndex) {
            offer(nodes[index], nodes[index + 1])
        }

        while (queue.isNotEmpty()) {
            val candidate = queue.remove()
            val left = candidate.left
            val right = candidate.right
            if (
                !left.active ||
                !right.active ||
                left.next !== right ||
                left.generation != candidate.leftGeneration ||
                right.generation != candidate.rightGeneration
            ) {
                continue
            }

            val next = right.next
            left.value += right.value
            left.generation++
            left.next = next
            right.active = false
            right.generation++
            right.previous = null
            right.next = null
            next?.previous = left

            offer(left.previous, left)
            offer(left, next)
        }

        val merged = ArrayList<String>(nodes.size)
        var node: BpeNode? = nodes.first()
        while (node != null) {
            if (node.active) merged += node.value
            node = node.next
        }
        return merged
    }

    private fun encodeByteLevelBpe(text: String): List<Int> {
        if (text.isEmpty()) return emptyList()
        val result = ArrayList<Int>(text.length)
        var cursor = 0
        var segmentStart = 0

        while (cursor < text.length) {
            val specialToken = if (text[cursor] == '<') {
                SPECIAL_TOKENS.firstOrNull { token -> text.startsWith(token, cursor) }
            } else {
                null
            }
            if (specialToken != null) {
                if (segmentStart < cursor) {
                    encodeByteLevelSegment(text.substring(segmentStart, cursor), result)
                }
                result += tokenToId[specialToken]
                    ?: throw IllegalStateException("Canonical special token '$specialToken' is missing.")
                cursor += specialToken.length
                segmentStart = cursor
            } else {
                cursor++
            }
        }

        if (segmentStart < text.length) {
            encodeByteLevelSegment(text.substring(segmentStart), result)
        }
        return result
    }

    private fun encodeByteLevelSegment(
        segment: String,
        destination: MutableList<Int>
    ) {
        for (match in BYTE_LEVEL_PRETOKENIZER.findAll(segment)) {
            val bytes = encodeUtf8Strict(match.value)
            val initialSymbols = ArrayList<String>(bytes.size)
            for (byte in bytes) {
                initialSymbols += byteSymbol(byte.toInt() and 0xFF)
            }
            for (piece in applyRankedMerges(initialSymbols)) {
                destination += tokenToId[piece]
                    ?: throw IllegalStateException(
                        "Byte-Level BPE produced a token that is absent from the vocabulary."
                    )
            }
        }
    }

    private fun decodeByteLevel(
        tokenIds: List<Int>,
        keepControlTokens: Boolean,
        keepReasoningTokens: Boolean
    ): String {
        val output = StringBuilder(tokenIds.size * 2)
        val byteBuffer = ByteArrayOutputStream()

        fun flushBytes() {
            if (byteBuffer.size() == 0) return
            output.append(decodeUtf8Strict(byteBuffer.toByteArray()))
            byteBuffer.reset()
        }

        for (id in tokenIds) {
            val token = idToToken[id] ?: continue
            if (token in CONTROL_TOKENS) {
                val preserveToken = keepControlTokens ||
                    (keepReasoningTokens && token in REASONING_TOKENS)
                if (preserveToken) {
                    flushBytes()
                    output.append(token)
                }
                continue
            }

            for (symbol in token) {
                require(symbol.code <= 0xFF) {
                    "Byte-Level BPE token contains a non-byte symbol."
                }
                byteBuffer.write(symbol.code)
            }
        }

        flushBytes()
        return output.toString()
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
                        val rank = mergeRank(pieces[pIdx], pieces[pIdx + 1])
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
                    result.add(registerToken(piece))
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

    @Synchronized
    fun encode(text: String, autoRegister: Boolean = false): List<Int> {
        if (config.algorithm == ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V2) {
            return encodeByteLevelBpe(text)
        }

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
                result.add(registerToken(charStr))
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
    @Synchronized
    fun decode(
        tokenIds: List<Int>,
        keepControlTokens: Boolean = false,
        keepReasoningTokens: Boolean = false
    ): String {
        if (config.algorithm == ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V2) {
            return decodeByteLevel(tokenIds, keepControlTokens, keepReasoningTokens)
        }

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

    /**
     * Serializes the complete tokenizer state into a deterministic, versioned snapshot.
     * The final 32 bytes are SHA-256 over every preceding byte.
     */
    @Synchronized
    fun saveSnapshot(): ByteArray {
        val payload = buildSnapshotPayload()
        val digest = MessageDigest.getInstance("SHA-256").digest(payload)
        return payload + digest
    }

    /** Returns the lowercase SHA-256 fingerprint of the canonical snapshot payload. */
    @Synchronized
    fun snapshotFingerprint(): String =
        MessageDigest.getInstance("SHA-256")
            .digest(buildSnapshotPayload())
            .joinToString(separator = "") { byte ->
                "%02x".format(java.util.Locale.ROOT, byte.toInt() and 0xFF)
            }

    @Synchronized
    fun loadSnapshot(snapshot: ByteArray) {
        require(snapshot.size >= SNAPSHOT_DIGEST_BYTES + 28) {
            "Tokenizer snapshot is truncated."
        }
        require(snapshot.size <= MAX_SNAPSHOT_BYTES) {
            "Tokenizer snapshot exceeds the supported maximum size."
        }

        val payloadLength = snapshot.size - SNAPSHOT_DIGEST_BYTES
        val payload = snapshot.copyOfRange(0, payloadLength)
        val storedDigest = snapshot.copyOfRange(payloadLength, snapshot.size)
        val actualDigest = MessageDigest.getInstance("SHA-256").digest(payload)
        require(MessageDigest.isEqual(storedDigest, actualDigest)) {
            "Tokenizer snapshot SHA-256 verification failed."
        }

        val input = DataInputStream(ByteArrayInputStream(payload))
        require(input.readInt() == SNAPSHOT_MAGIC) {
            "Tokenizer snapshot magic is invalid."
        }
        val formatVersion = input.readInt()
        require(formatVersion == CURRENT_SNAPSHOT_FORMAT_VERSION) {
            "Unsupported tokenizer snapshot format version: $formatVersion."
        }
        val schemaVersion = input.readInt()
        require(schemaVersion == config.schemaVersion) {
            "Tokenizer snapshot schema does not match the active configuration."
        }
        val algorithmId = input.readInt()
        require(algorithmId == config.algorithm.checkpointId) {
            "Tokenizer snapshot algorithm does not match the active configuration."
        }
        val vocabularyCapacity = input.readInt()
        require(vocabularyCapacity == config.maxVocabularySize) {
            "Tokenizer snapshot vocabulary capacity does not match the active configuration."
        }
        val maximumTokenLength = input.readInt()
        require(maximumTokenLength == config.maxTokenLength) {
            "Tokenizer snapshot maximum token length does not match the active configuration."
        }

        val vocabularyCount = input.readInt()
        require(vocabularyCount in 1..config.maxVocabularySize) {
            "Tokenizer snapshot vocabulary count is invalid."
        }
        val vocabulary = ArrayList<Pair<String, Int>>(vocabularyCount)
        repeat(vocabularyCount) {
            val id = input.readInt()
            val token = readSnapshotString(input)
            vocabulary += token to id
        }

        val mergeCount = input.readInt()
        require(mergeCount in 0..config.maxVocabularySize) {
            "Tokenizer snapshot merge count is invalid."
        }
        val importedMerges = ArrayList<Pair<String, String>>(mergeCount)
        repeat(mergeCount) {
            importedMerges += readSnapshotString(input) to readSnapshotString(input)
        }
        require(input.available() == 0) {
            "Tokenizer snapshot contains unexpected trailing payload data."
        }
        require(config.algorithm.usesBpeMerges || importedMerges.isEmpty()) {
            "A non-BPE tokenizer snapshot cannot contain merge rules."
        }

        // Restore into a fully validated candidate first. The active instance is
        // not changed if any vocabulary or merge invariant fails.
        val candidate = Tokenizer(config)
        candidate.importVocab(vocabulary)
        require(candidate.vocabSize == vocabularyCount) {
            "Tokenizer snapshot vocabulary count does not match its canonical entries."
        }
        if (config.algorithm.usesBpeMerges) {
            candidate.importMerges(importedMerges)
        }

        tokenToId.clear()
        tokenToId.putAll(candidate.tokenToId)
        idToToken.clear()
        idToToken.putAll(candidate.idToToken)
        merges.clear()
        merges.addAll(candidate.merges)
        mergeRanks.clear()
        for ((left, rightRanks) in candidate.mergeRanks) {
            mergeRanks[left] = HashMap(rightRanks)
        }
        nextId = candidate.nextId
        maxRegisteredLen = candidate.maxRegisteredLen
    }

    private fun buildSnapshotPayload(): ByteArray {
        val output = ByteArrayOutputStream()
        DataOutputStream(output).use { data ->
            data.writeInt(SNAPSHOT_MAGIC)
            data.writeInt(CURRENT_SNAPSHOT_FORMAT_VERSION)
            data.writeInt(config.schemaVersion)
            data.writeInt(config.algorithm.checkpointId)
            data.writeInt(config.maxVocabularySize)
            data.writeInt(config.maxTokenLength)

            val vocabulary = exportVocab()
            data.writeInt(vocabulary.size)
            for ((token, id) in vocabulary) {
                data.writeInt(id)
                writeSnapshotString(data, token)
            }

            val mergeRules = if (config.algorithm.usesBpeMerges) {
                exportMerges()
            } else {
                emptyList()
            }
            data.writeInt(mergeRules.size)
            for ((left, right) in mergeRules) {
                writeSnapshotString(data, left)
                writeSnapshotString(data, right)
            }
        }
        return output.toByteArray()
    }

    private fun writeSnapshotString(output: DataOutputStream, value: String) {
        val bytes = encodeUtf8Strict(value)
        require(bytes.isNotEmpty() && bytes.size <= config.maxTokenLength * 4) {
            "Tokenizer snapshot string length is invalid."
        }
        output.writeInt(bytes.size)
        output.write(bytes)
    }

    private fun readSnapshotString(input: DataInputStream): String {
        val length = input.readInt()
        require(length in 1..(config.maxTokenLength * 4)) {
            "Tokenizer snapshot string length is invalid."
        }
        val bytes = ByteArray(length)
        input.readFully(bytes)
        return decodeUtf8Strict(bytes)
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
