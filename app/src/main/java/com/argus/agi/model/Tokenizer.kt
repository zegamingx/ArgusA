package com.argus.agi.model

import java.io.BufferedReader
import java.io.ByteArrayInputStream
import java.io.PushbackReader
import java.io.Reader
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream
import java.nio.ByteBuffer
import java.nio.charset.CharacterCodingException
import java.nio.charset.CodingErrorAction
import java.nio.charset.StandardCharsets
import java.security.MessageDigest
import java.util.ArrayDeque
import java.util.LinkedHashMap
import java.util.concurrent.atomic.AtomicReferenceArray
import java.util.PriorityQueue

enum class ArgusTokenizerAlgorithm(val checkpointId: Int) {
    BYTE_LEVEL_BPE_V1(1);

    val usesBpeMerges: Boolean
        get() = true

    companion object {
        fun fromCheckpointId(value: Int): ArgusTokenizerAlgorithm =
            entries.firstOrNull { it.checkpointId == value }
                ?: throw IllegalArgumentException("Unsupported tokenizer algorithm ID: $value.")
    }
}

data class TokenizerConfig(
    val schemaVersion: Int = CURRENT_SCHEMA_VERSION,
    val algorithm: ArgusTokenizerAlgorithm = ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V1,
    val maxVocabularySize: Int = DEFAULT_MAX_VOCAB_SIZE,
    val maxTokenLength: Int = DEFAULT_MAX_TOKEN_LENGTH
) {
    init {
        require(schemaVersion == CURRENT_SCHEMA_VERSION) {
            "Unsupported tokenizer schema version: $schemaVersion."
        }
        require(maxVocabularySize in MINIMUM_VOCAB_SIZE..MAX_SUPPORTED_VOCAB_SIZE) {
            "Tokenizer capacity must be between $MINIMUM_VOCAB_SIZE and $MAX_SUPPORTED_VOCAB_SIZE."
        }
        require(maxTokenLength in MINIMUM_TOKEN_LENGTH..MAX_SUPPORTED_TOKEN_LENGTH) {
            "Tokenizer token length must be between $MINIMUM_TOKEN_LENGTH and $MAX_SUPPORTED_TOKEN_LENGTH."
        }
    }

    companion object {
        const val CURRENT_SCHEMA_VERSION = 2
        const val DEFAULT_MAX_VOCAB_SIZE = 2_048
        const val DEFAULT_MAX_TOKEN_LENGTH = 32
        const val MINIMUM_VOCAB_SIZE = 288
        const val MINIMUM_TOKEN_LENGTH = 18
        const val MAX_SUPPORTED_VOCAB_SIZE = 1_000_000
        const val MAX_SUPPORTED_TOKEN_LENGTH = 256
    }
}

enum class TokenizerTrainingStage(val displayName: String) {
    PREPARING_CORPUS("Preparing tokenizer corpus"),
    COUNTING_FREQUENCIES("Counting byte-level frequencies"),
    BUILDING_SYMBOLS("Building byte symbols"),
    REPLAYING_EXISTING_MERGES("Applying existing merge rules"),
    LEARNING_MERGES("Learning BPE merge rules"),
    COMPLETED("BPE training complete"),
    CANCELLED("BPE training cancelled"),
    FAILED("BPE training failed")
}

data class TokenizerTrainingProgress(
    val stage: TokenizerTrainingStage,
    val fraction: Float,
    val processedItems: Long,
    val totalItems: Long,
    val learnedMerges: Int,
    val targetMerges: Int,
    val vocabularySize: Int
)

/**
 * Canonical byte-level BPE tokenizer.
 *
 * IDs 0..31 are a fixed control-token manifest, IDs 32..287 are the complete
 * reversible byte alphabet, and learned merges are appended from ID 288.
 * Inference reads one immutable state snapshot without acquiring the training lock.
 */
/**
 * One source-preserving tokenizer chunk. The token IDs use the same tokenizer
 * generation as the complete streaming pass.
 */
data class TokenizerTextChunk(
    val text: String,
    val tokenIds: IntArray
)

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
        const val DEVELOPER_TOKEN = "<DEVELOPER>"
        const val TOOL_TOKEN = "<TOOL>"
        const val TOOL_CALL_TOKEN = "<TOOL_CALL>"
        const val TOOL_RESULT_TOKEN = "<TOOL_RESULT>"
        const val WEB_TOKEN = "<WEB>"
        const val WEB_RESULT_TOKEN = "<WEB_RESULT>"
        const val FIM_PREFIX_TOKEN = "<FIM_PREFIX>"
        const val FIM_MIDDLE_TOKEN = "<FIM_MIDDLE>"
        const val FIM_SUFFIX_TOKEN = "<FIM_SUFFIX>"
        const val IMAGE_TOKEN = "<IMAGE>"
        const val IMAGE_PATCH_TOKEN = "<IMAGE_PATCH>"
        const val AUDIO_TOKEN = "<AUDIO>"
        const val AUDIO_TEXT_TOKEN = "<AUDIO_TEXT>"
        const val AUDIO_PERCEPTION_TOKEN = "<AUDIO_PERCEPTION>"
        const val AUDIO_UNCOND_TOKEN = "<AUDIO_UNCOND>"
        const val AUDIO_SENTINEL_TOKEN = "<AUDIO_SENTINEL>"
        const val END_OF_TASK_TOKEN = "<END_OF_TASK>"
        const val SPEAKER_ID_0_TOKEN = "<SPEAKER_ID_0>"
        const val VIDEO_FRAME_TOKEN = "<VIDEO_FRAME>"
        const val TIMESTAMP_TOKEN = "<TIMESTAMP>"
        const val REFUSAL_TOKEN = "<REFUSAL>"
        const val DISC_SCORE_TOKEN = "<DISC_SCORE>"
        const val MASK_TOKEN = "<MASK>"

        const val PAD_ID = 0
        const val UNK_ID = 1
        const val BOS_ID = 2
        const val EOS_ID = 3
        const val SYSTEM_ID = 4
        const val USER_ID = 5
        const val ASSISTANT_ID = 6
        const val THINK_START_ID = 7
        const val THINK_END_ID = 8
        const val DEVELOPER_ID = 9
        const val TOOL_ID = 10
        const val TOOL_CALL_ID = 11
        const val TOOL_RESULT_ID = 12
        const val WEB_ID = 13
        const val WEB_RESULT_ID = 14
        const val FIM_PREFIX_ID = 15
        const val FIM_MIDDLE_ID = 16
        const val FIM_SUFFIX_ID = 17
        const val IMAGE_ID = 18
        const val IMAGE_PATCH_ID = 19
        const val AUDIO_ID = 20
        const val AUDIO_TEXT_ID = 21
        const val AUDIO_PERCEPTION_ID = 22
        const val AUDIO_UNCOND_ID = 23
        const val AUDIO_SENTINEL_ID = 24
        const val END_OF_TASK_ID = 25
        const val SPEAKER_ID_0_ID = 26
        const val VIDEO_FRAME_ID = 27
        const val TIMESTAMP_ID = 28
        const val REFUSAL_ID = 29
        const val DISC_SCORE_ID = 30
        const val MASK_ID = 31

        private const val FIRST_BYTE_TOKEN_ID = 32
        private const val BYTE_ALPHABET_SIZE = 256
        private const val BASE_VOCABULARY_SIZE = FIRST_BYTE_TOKEN_ID + BYTE_ALPHABET_SIZE
        private const val SNAPSHOT_MAGIC = 0x41544231 // "ATB1"
        const val CURRENT_SNAPSHOT_FORMAT_VERSION = 1
        private const val SNAPSHOT_DIGEST_BYTES = 32
        private const val MAX_SNAPSHOT_BYTES = 128 * 1024 * 1024
        private const val DEFAULT_MAX_MERGES_PER_TRAINING_PASS = 64
        private const val MAX_ENCODE_CACHE_ENTRIES = 128
        private const val MAX_CACHEABLE_TEXT_LENGTH = 2_048
        private const val MAX_CACHEABLE_TOKEN_IDS = 4_096
        private const val MAX_TRAINING_PRETOKEN_CACHE_ENTRIES = 128
        private const val MAX_TRAINING_PRETOKEN_CACHE_SEGMENT_LENGTH = 4_096

        const val MAX_TOKEN_LEN = TokenizerConfig.DEFAULT_MAX_TOKEN_LENGTH
        const val MAX_VOCAB_SIZE = TokenizerConfig.DEFAULT_MAX_VOCAB_SIZE

        internal const val LITERAL_CONTROL_ESCAPE_MARKER: Char = '\uE000'

        private val SPECIAL_IDS: Map<String, Int> = linkedMapOf(
            PAD_TOKEN to PAD_ID,
            UNK_TOKEN to UNK_ID,
            BOS_TOKEN to BOS_ID,
            EOS_TOKEN to EOS_ID,
            SYSTEM_TOKEN to SYSTEM_ID,
            USER_TOKEN to USER_ID,
            ASSISTANT_TOKEN to ASSISTANT_ID,
            THINK_START to THINK_START_ID,
            THINK_END to THINK_END_ID,
            DEVELOPER_TOKEN to DEVELOPER_ID,
            TOOL_TOKEN to TOOL_ID,
            TOOL_CALL_TOKEN to TOOL_CALL_ID,
            TOOL_RESULT_TOKEN to TOOL_RESULT_ID,
            WEB_TOKEN to WEB_ID,
            WEB_RESULT_TOKEN to WEB_RESULT_ID,
            FIM_PREFIX_TOKEN to FIM_PREFIX_ID,
            FIM_MIDDLE_TOKEN to FIM_MIDDLE_ID,
            FIM_SUFFIX_TOKEN to FIM_SUFFIX_ID,
            IMAGE_TOKEN to IMAGE_ID,
            IMAGE_PATCH_TOKEN to IMAGE_PATCH_ID,
            AUDIO_TOKEN to AUDIO_ID,
            AUDIO_TEXT_TOKEN to AUDIO_TEXT_ID,
            AUDIO_PERCEPTION_TOKEN to AUDIO_PERCEPTION_ID,
            AUDIO_UNCOND_TOKEN to AUDIO_UNCOND_ID,
            AUDIO_SENTINEL_TOKEN to AUDIO_SENTINEL_ID,
            END_OF_TASK_TOKEN to END_OF_TASK_ID,
            SPEAKER_ID_0_TOKEN to SPEAKER_ID_0_ID,
            VIDEO_FRAME_TOKEN to VIDEO_FRAME_ID,
            TIMESTAMP_TOKEN to TIMESTAMP_ID,
            REFUSAL_TOKEN to REFUSAL_ID,
            DISC_SCORE_TOKEN to DISC_SCORE_ID,
            MASK_TOKEN to MASK_ID
        )

        private val SPECIAL_TOKENS: List<String> =
            SPECIAL_IDS.keys.sortedWith(
                compareByDescending<String> { token -> token.length }.thenBy { token -> token }
            )

        private val CONTROL_TOKENS: Set<String> = SPECIAL_IDS.keys
        private val REASONING_TOKENS: Set<String> = setOf(THINK_START, THINK_END)

        private fun specialTokenAt(text: String, index: Int): String? {
            if (index !in text.indices || text[index] != '<') return null
            return SPECIAL_TOKENS.firstOrNull { token -> text.startsWith(token, index) }
        }

        /**
         * Escapes reserved control spellings before payload text is combined with structural
         * tokens. Literal U+E000 characters remain unchanged; only a doubled marker directly
         * before a reserved spelling has escape meaning. A single marker before a real control
         * token therefore remains payload and does not suppress that control token.
         */
        internal fun escapeLiteralControlTokens(text: String): String {
            if (text.isEmpty()) return text
            val escaped = StringBuilder(text.length + 8)
            var index = 0
            while (index < text.length) {
                if (text[index] == LITERAL_CONTROL_ESCAPE_MARKER) {
                    escaped.append(LITERAL_CONTROL_ESCAPE_MARKER)
                    index++
                    continue
                }
                val special = specialTokenAt(text, index)
                if (special != null) {
                    escaped.append(LITERAL_CONTROL_ESCAPE_MARKER)
                        .append(LITERAL_CONTROL_ESCAPE_MARKER)
                        .append(special)
                    index += special.length
                } else {
                    escaped.append(text[index])
                    index++
                }
            }
            return escaped.toString()
        }
    }

    private data class MergePair(val left: String, val right: String)

    private data class CachedEncoding(
        val text: String,
        val tokenIds: IntArray
    )

    /**
     * Small direct-mapped cache. Atomic slots keep concurrent encode calls lock-free;
     * each immutable tokenizer generation owns a fresh cache, so training cannot
     * leave token IDs from an older merge table in the active inference path.
     */
    private class EncodedTextCache {
        private val entries =
            AtomicReferenceArray<CachedEncoding?>(MAX_ENCODE_CACHE_ENTRIES)

        fun get(text: String): IntArray? {
            if (text.length > MAX_CACHEABLE_TEXT_LENGTH) return null
            val slot = text.hashCode() and (MAX_ENCODE_CACHE_ENTRIES - 1)
            val cached = entries.get(slot) ?: return null
            return if (cached.text == text) cached.tokenIds.copyOf() else null
        }

        fun put(text: String, tokenIds: IntArray) {
            if (
                text.length > MAX_CACHEABLE_TEXT_LENGTH ||
                tokenIds.size > MAX_CACHEABLE_TOKEN_IDS
            ) {
                return
            }
            val slot = text.hashCode() and (MAX_ENCODE_CACHE_ENTRIES - 1)
            entries.set(slot, CachedEncoding(text, tokenIds.copyOf()))
        }
    }

    private data class TokenizerState(
        val tokenToId: Map<String, Int>,
        val idToToken: Map<Int, String>,
        val merges: List<MergePair>,
        val mergeRanks: Map<String, Map<String, Int>>,
        val nextId: Int,
        val encodedTextCache: EncodedTextCache
    )

    private data class MutableTokenizerState(
        val tokenToId: HashMap<String, Int>,
        val idToToken: HashMap<Int, String>,
        val merges: ArrayList<MergePair>,
        val mergeRanks: HashMap<String, HashMap<String, Int>>,
        var nextId: Int
    )

    private class IntAccumulator(initialCapacity: Int) {
        private var values = IntArray(initialCapacity.coerceAtLeast(8))
        var size: Int = 0
            private set

        fun add(value: Int) {
            if (size == values.size) {
                values = values.copyOf(values.size + (values.size shr 1) + 8)
            }
            values[size] = value
            size++
        }

        fun toIntArray(): IntArray = values.copyOf(size)
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

    @Volatile
    private var state: TokenizerState = createBaseState()

    val vocabSize: Int
        get() = state.nextId

    val mergeCount: Int
        get() = state.merges.size

    private fun createBaseState(): TokenizerState {
        val builder = MutableTokenizerState(
            tokenToId = HashMap(config.maxVocabularySize),
            idToToken = HashMap(config.maxVocabularySize),
            merges = ArrayList(),
            mergeRanks = HashMap(),
            nextId = BASE_VOCABULARY_SIZE
        )
        for ((token, id) in SPECIAL_IDS) {
            builder.tokenToId[token] = id
            builder.idToToken[id] = token
        }
        for (byteValue in 0 until BYTE_ALPHABET_SIZE) {
            val token = byteSymbol(byteValue)
            val id = FIRST_BYTE_TOKEN_ID + byteValue
            builder.tokenToId[token] = id
            builder.idToToken[id] = token
        }
        return freeze(builder)
    }

    private fun copyMutableState(source: TokenizerState): MutableTokenizerState {
        val mutableRanks = HashMap<String, HashMap<String, Int>>(source.mergeRanks.size)
        for ((left, rightRanks) in source.mergeRanks) {
            mutableRanks[left] = HashMap(rightRanks)
        }
        return MutableTokenizerState(
            tokenToId = HashMap(source.tokenToId),
            idToToken = HashMap(source.idToToken),
            merges = ArrayList(source.merges),
            mergeRanks = mutableRanks,
            nextId = source.nextId
        )
    }

    private fun freeze(source: MutableTokenizerState): TokenizerState {
        val immutableRanks = HashMap<String, Map<String, Int>>(source.mergeRanks.size)
        for ((left, rightRanks) in source.mergeRanks) {
            immutableRanks[left] = rightRanks.toMap()
        }
        return TokenizerState(
            tokenToId = source.tokenToId.toMap(),
            idToToken = source.idToToken.toMap(),
            merges = source.merges.toList(),
            mergeRanks = immutableRanks.toMap(),
            nextId = source.nextId,
            encodedTextCache = EncodedTextCache()
        )
    }

    private fun byteSymbol(byteValue: Int): String =
        (byteValue and 0xFF).toChar().toString()

    internal fun baseByteTokenId(byteValue: Int): Int {
        require(byteValue in 0 until BYTE_ALPHABET_SIZE) {
            "Byte value must be in 0..255."
        }
        return FIRST_BYTE_TOKEN_ID + byteValue
    }

    /**
     * Counts the exact UTF-8 bytes used by this tokenizer after malformed UTF-16
     * replacement. Used to normalize held-out likelihood independently of BPE merges.
     */
    internal fun utf8ByteCount(text: String): Long =
        encodeUtf8WithReplacement(text).size.toLong()

    /**
     * Counts the original UTF-8 bytes represented by an encoded token suffix.
     *
     * BYTE_LEVEL_BPE_V1 stores each byte as one UTF-16 code unit in byte symbols.
     * Learned merges concatenate those symbols, while reserved token spellings are
     * ASCII. Their string lengths therefore equal the represented UTF-8 byte count.
     * [startIndex] excludes context-only tokens from held-out bits-per-byte metrics.
     */
    internal fun utf8ByteCountForTokenIds(
        tokenIds: IntArray,
        startIndex: Int = 0
    ): Long {
        require(startIndex in 0..tokenIds.size) {
            "startIndex must be within the encoded token array."
        }
        val snapshot = state
        var byteCount = 0L
        for (index in startIndex until tokenIds.size) {
            val tokenId = tokenIds[index]
            val token = snapshot.idToToken[tokenId]
                ?: throw IllegalArgumentException("Tokenizer ID $tokenId is not registered.")
            byteCount = Math.addExact(byteCount, token.length.toLong())
        }
        return byteCount
    }

    @Synchronized
    fun registerToken(token: String, id: Int? = null): Int {
        val current = state
        val existing = current.tokenToId[token]
        if (existing != null) {
            require(id == null || id == existing) {
                "Token '$token' is already assigned to ID $existing."
            }
            return existing
        }

        val candidate = copyMutableState(current)
        val registered = registerToken(candidate, token, id)
        state = freeze(candidate)
        return registered
    }

    private fun registerToken(
        candidate: MutableTokenizerState,
        token: String,
        requestedId: Int? = null
    ): Int {
        require(token.isNotEmpty()) { "Tokenizer tokens must not be empty." }
        require(token.length <= config.maxTokenLength) {
            "Token length ${token.length} exceeds configured maxTokenLength=${config.maxTokenLength}."
        }
        require(token !in CONTROL_TOKENS) {
            "Reserved control tokens cannot be registered as learned byte merges."
        }
        require(token.all { character -> character.code <= 0xFF }) {
            "BYTE_LEVEL_BPE_V1 vocabulary entries must contain only reversible byte symbols."
        }
        val existingId = candidate.tokenToId[token]
        if (existingId != null) {
            require(requestedId == null || requestedId == existingId) {
                "Token '$token' is already assigned to ID $existingId."
            }
            return existingId
        }

        val resolvedId = requestedId ?: candidate.nextId
        require(resolvedId == candidate.nextId) {
            "BYTE_LEVEL_BPE_V1 vocabulary IDs are append-only; expected ID ${candidate.nextId}, received $resolvedId."
        }
        check(resolvedId < config.maxVocabularySize) {
            "Tokenizer vocabulary capacity ${config.maxVocabularySize} is exhausted."
        }
        require(candidate.idToToken[resolvedId] == null) {
            "Tokenizer ID $resolvedId is already occupied."
        }

        candidate.tokenToId[token] = resolvedId
        candidate.idToToken[resolvedId] = token
        candidate.nextId = Math.addExact(resolvedId, 1)
        return resolvedId
    }

    fun exportVocab(): List<Pair<String, Int>> =
        state.tokenToId.entries
            .map { entry -> entry.key to entry.value }
            .sortedBy { entry -> entry.second }

    @Synchronized
    fun importVocab(entries: List<Pair<String, Int>>) {
        require(entries.size in BASE_VOCABULARY_SIZE..config.maxVocabularySize) {
            "Imported vocabulary size is outside the configured capacity."
        }
        require(entries.map { entry -> entry.first }.toSet().size == entries.size) {
            "Imported vocabulary contains duplicate tokens."
        }
        require(entries.map { entry -> entry.second }.toSet().size == entries.size) {
            "Imported vocabulary contains duplicate IDs."
        }

        val byToken = entries.toMap()
        val byId = entries.associate { entry -> entry.second to entry.first }
        for ((token, canonicalId) in SPECIAL_IDS) {
            require(byToken[token] == canonicalId) {
                "Canonical control token '$token' must be assigned to ID $canonicalId."
            }
        }
        for (byteValue in 0 until BYTE_ALPHABET_SIZE) {
            val token = byteSymbol(byteValue)
            val expectedId = FIRST_BYTE_TOKEN_ID + byteValue
            require(byToken[token] == expectedId) {
                "Canonical byte token $byteValue must be assigned to ID $expectedId."
            }
        }

        for ((token, id) in entries) {
            require(id in 0 until config.maxVocabularySize) {
                "Imported token ID $id exceeds configured vocabulary capacity."
            }
            require(token.isNotEmpty() && token.length <= config.maxTokenLength) {
                "Imported token length is invalid."
            }
            if (id >= BASE_VOCABULARY_SIZE) {
                require(token !in CONTROL_TOKENS) {
                    "Learned vocabulary entries cannot reuse reserved control tokens."
                }
                require(token.all { character -> character.code <= 0xFF }) {
                    "Imported byte-level tokens must contain only byte symbols."
                }
            }
        }
        require(byId.size == entries.size && byId.keys.sorted() == (0 until entries.size).toList()) {
            "BYTE_LEVEL_BPE_V1 vocabulary IDs must be contiguous and append-only."
        }

        val candidate = MutableTokenizerState(
            tokenToId = HashMap(byToken),
            idToToken = HashMap(byId),
            merges = ArrayList(),
            mergeRanks = HashMap(),
            nextId = entries.size
        )
        state = freeze(candidate)
    }

    fun exportMerges(): List<Pair<String, String>> =
        state.merges.map { merge -> merge.left to merge.right }

    @Synchronized
    fun importMerges(importedMerges: List<Pair<String, String>>) {
        require(importedMerges.size <= config.maxVocabularySize - BASE_VOCABULARY_SIZE) {
            "Imported merge count exceeds the learnable vocabulary range."
        }

        val current = state
        val candidate = copyMutableState(current)
        val ranks = HashMap<Pair<String, String>, Int>(importedMerges.size * 2)
        val newMerges = ArrayList<MergePair>(importedMerges.size)

        for ((rank, pair) in importedMerges.withIndex()) {
            val (left, right) = pair
            require(left.isNotEmpty() && right.isNotEmpty()) {
                "Merge components must not be empty."
            }
            require(left.length + right.length <= config.maxTokenLength) {
                "Merge result exceeds configured maximum token length."
            }
            require(left !in CONTROL_TOKENS && right !in CONTROL_TOKENS) {
                "Control tokens cannot be merge components."
            }
            val combined = left + right
            require(combined !in CONTROL_TOKENS) {
                "A merge cannot resolve to a reserved control token."
            }
            require(left.all { character -> character.code <= 0xFF } &&
                right.all { character -> character.code <= 0xFF }
            ) {
                "Merge components must remain inside the byte-symbol alphabet."
            }
            require(candidate.tokenToId.containsKey(left) && candidate.tokenToId.containsKey(right)) {
                "Merge components must exist in the imported vocabulary."
            }
            require(candidate.tokenToId.containsKey(combined)) {
                "Merged token '$combined' is missing from the imported vocabulary."
            }
            require(ranks.put(pair, rank) == null) {
                "Duplicate BPE merge pair: '$left' + '$right'."
            }
            newMerges += MergePair(left, right)
        }

        candidate.merges.clear()
        candidate.merges.addAll(newMerges)
        candidate.mergeRanks.clear()
        for ((pair, rank) in ranks) {
            candidate.mergeRanks.getOrPut(pair.first) { HashMap() }[pair.second] = rank
        }
        state = freeze(candidate)
    }

    /**
     * Learns deterministic BPE rules. Existing merge ranks are replayed before
     * new pairs are counted, so incremental learning never renumbers old tokens.
     */
    @Synchronized
    fun trainBpe(
        corpus: List<String>,
        targetVocabSize: Int = config.maxVocabularySize,
        maxNewMerges: Int = DEFAULT_MAX_MERGES_PER_TRAINING_PASS,
        onProgress: (TokenizerTrainingProgress) -> Unit = {}
    ): Int {
        require(maxNewMerges > 0) { "maxNewMerges must be positive." }
        val current = state
        val target = minOf(targetVocabSize, config.maxVocabularySize)
        val requestedMerges = if (current.nextId < target) maxNewMerges else 0
        var lastFraction = 0f
        var learnedMerges = 0
        var currentVocabularySize = current.nextId

        fun report(
            stage: TokenizerTrainingStage,
            fraction: Float,
            processedItems: Long,
            totalItems: Long
        ) {
            val monotonicFraction = fraction.coerceIn(lastFraction, 1f)
            lastFraction = monotonicFraction
            onProgress(
                TokenizerTrainingProgress(
                    stage = stage,
                    fraction = monotonicFraction,
                    processedItems = processedItems.coerceAtLeast(0L),
                    totalItems = totalItems.coerceAtLeast(0L),
                    learnedMerges = learnedMerges,
                    targetMerges = requestedMerges,
                    vocabularySize = currentVocabularySize
                )
            )
        }

        if (current.nextId >= target || corpus.isEmpty()) {
            report(
                stage = TokenizerTrainingStage.COMPLETED,
                fraction = 1f,
                processedItems = 0L,
                totalItems = 0L
            )
            return current.nextId
        }

        val candidate = copyMutableState(current)
        val wordFrequencies = LinkedHashMap<String, Long>()
        val preTokenCache = LinkedHashMap<String, List<String>>(16, 0.75f, true)

        fun cachedPreTokens(segment: String): List<String> {
            if (segment.length > MAX_TRAINING_PRETOKEN_CACHE_SEGMENT_LENGTH) {
                return buildList {
                    forEachByteLevelPreToken(segment) { preToken -> add(preToken) }
                }
            }

            val cached = preTokenCache[segment]
            if (cached != null) return cached

            val computed = buildList {
                forEachByteLevelPreToken(segment) { preToken -> add(preToken) }
            }
            if (computed.isNotEmpty()) {
                if (preTokenCache.size >= MAX_TRAINING_PRETOKEN_CACHE_ENTRIES) {
                    val oldest = preTokenCache.entries.iterator()
                    if (oldest.hasNext()) {
                        oldest.next()
                        oldest.remove()
                    }
                }
                preTokenCache[segment] = computed
            }
            return computed
        }

        fun addWordFrequency(word: String) {
            if (word.isEmpty()) return
            val previous = wordFrequencies[word] ?: 0L
            wordFrequencies[word] = if (previous == Long.MAX_VALUE) {
                Long.MAX_VALUE
            } else {
                previous + 1L
            }
        }

        val totalCorpusCharacters = corpus.sumOf { document -> document.length.toLong() }
        var countedCharacters = 0L
        var nextCountingReport = 8_192L
        report(
            stage = TokenizerTrainingStage.COUNTING_FREQUENCIES,
            fraction = 0f,
            processedItems = 0L,
            totalItems = totalCorpusCharacters
        )
        for (document in corpus) {
            forEachNormalTextSegment(document) { segment ->
                for (preToken in cachedPreTokens(segment)) {
                    addWordFrequency(preToken)
                }
                countedCharacters = Math.addExact(countedCharacters, segment.length.toLong())
                if (countedCharacters >= nextCountingReport) {
                    val ratio = countedCharacters.toFloat() /
                        totalCorpusCharacters.coerceAtLeast(1L).toFloat()
                    report(
                        stage = TokenizerTrainingStage.COUNTING_FREQUENCIES,
                        fraction = 0.15f * ratio.coerceIn(0f, 1f),
                        processedItems = countedCharacters,
                        totalItems = totalCorpusCharacters
                    )
                    nextCountingReport = Math.addExact(countedCharacters, 8_192L)
                }
            }
        }
        report(
            stage = TokenizerTrainingStage.COUNTING_FREQUENCIES,
            fraction = 0.15f,
            processedItems = totalCorpusCharacters,
            totalItems = totalCorpusCharacters
        )

        if (wordFrequencies.isEmpty()) {
            report(
                stage = TokenizerTrainingStage.COMPLETED,
                fraction = 1f,
                processedItems = 0L,
                totalItems = 0L
            )
            return current.nextId
        }

        val wordSymbols = LinkedHashMap<String, MutableList<String>>(wordFrequencies.size)
        val totalWords = wordFrequencies.size.toLong()
        var builtWords = 0L
        val symbolReportInterval = maxOf(32L, totalWords / 32L)
        report(
            stage = TokenizerTrainingStage.BUILDING_SYMBOLS,
            fraction = 0.15f,
            processedItems = 0L,
            totalItems = totalWords
        )
        for (word in wordFrequencies.keys) {
            val bytes = encodeUtf8WithReplacement(word)
            val symbols = ArrayList<String>(bytes.size)
            for (byte in bytes) {
                symbols += byteSymbol(byte.toInt() and 0xFF)
            }
            wordSymbols[word] = symbols
            builtWords++
            if (builtWords % symbolReportInterval == 0L || builtWords == totalWords) {
                report(
                    stage = TokenizerTrainingStage.BUILDING_SYMBOLS,
                    fraction = 0.15f + 0.15f * builtWords.toFloat() / totalWords.toFloat(),
                    processedItems = builtWords,
                    totalItems = totalWords
                )
            }
        }

        val totalReplayOperations = candidate.merges.size.toLong() * totalWords
        val replayReportInterval = maxOf(128L, totalReplayOperations / 32L)
        var replayedOperations = 0L
        report(
            stage = TokenizerTrainingStage.REPLAYING_EXISTING_MERGES,
            fraction = 0.30f,
            processedItems = 0L,
            totalItems = totalReplayOperations
        )
        for (merge in candidate.merges) {
            val combined = merge.left + merge.right
            for (symbols in wordSymbols.values) {
                applyMerge(symbols, merge.left, merge.right, combined)
                replayedOperations++
                if (
                    replayedOperations % replayReportInterval == 0L ||
                    replayedOperations == totalReplayOperations
                ) {
                    val ratio = replayedOperations.toFloat() /
                        totalReplayOperations.coerceAtLeast(1L).toFloat()
                    report(
                        stage = TokenizerTrainingStage.REPLAYING_EXISTING_MERGES,
                        fraction = 0.30f + 0.10f * ratio,
                        processedItems = replayedOperations,
                        totalItems = totalReplayOperations
                    )
                }
            }
        }
        report(
            stage = TokenizerTrainingStage.REPLAYING_EXISTING_MERGES,
            fraction = 0.40f,
            processedItems = totalReplayOperations,
            totalItems = totalReplayOperations
        )

        val wordScanReportInterval = maxOf(16L, totalWords / 8L)
        while (candidate.nextId < target && learnedMerges < maxNewMerges) {
            val pairCounts = HashMap<MergePair, Long>()
            var scannedWords = 0L
            for ((word, frequency) in wordFrequencies) {
                val symbols = wordSymbols[word] ?: continue
                for (index in 0 until symbols.size - 1) {
                    val left = symbols[index]
                    val right = symbols[index + 1]
                    if (candidate.mergeRanks[left]?.containsKey(right) == true) continue
                    val combinedLength = left.length + right.length
                    if (combinedLength > config.maxTokenLength) continue
                    val combined = left + right
                    if (combined in CONTROL_TOKENS) continue

                    val pair = MergePair(left, right)
                    val previous = pairCounts[pair] ?: 0L
                    pairCounts[pair] = if (previous > Long.MAX_VALUE - frequency) {
                        Long.MAX_VALUE
                    } else {
                        previous + frequency
                    }
                }
                scannedWords++
                if (scannedWords % wordScanReportInterval == 0L || scannedWords == totalWords) {
                    val passProgress = (
                        learnedMerges.toFloat() +
                            scannedWords.toFloat() / totalWords.coerceAtLeast(1L).toFloat()
                        ) / requestedMerges.coerceAtLeast(1).toFloat()
                    report(
                        stage = TokenizerTrainingStage.LEARNING_MERGES,
                        fraction = 0.40f + 0.59f * passProgress.coerceIn(0f, 1f),
                        processedItems = scannedWords,
                        totalItems = totalWords
                    )
                }
            }

            val bestEntry = pairCounts.entries.maxWithOrNull(
                compareBy<Map.Entry<MergePair, Long>> { entry -> entry.value }
                    .thenBy { entry -> entry.key.left }
                    .thenBy { entry -> entry.key.right }
            ) ?: break
            if (bestEntry.value < 2L) break

            val bestPair = bestEntry.key
            val combined = bestPair.left + bestPair.right
            if (combined.length > config.maxTokenLength || combined in CONTROL_TOKENS) continue

            if (!candidate.tokenToId.containsKey(combined)) {
                registerToken(candidate, combined)
                currentVocabularySize = candidate.nextId
            }
            val rank = candidate.merges.size
            candidate.merges += bestPair
            candidate.mergeRanks.getOrPut(bestPair.left) { HashMap() }[bestPair.right] = rank
            learnedMerges++

            for (symbols in wordSymbols.values) {
                applyMerge(symbols, bestPair.left, bestPair.right, combined)
            }
            val mergeProgress = learnedMerges.toFloat() /
                requestedMerges.coerceAtLeast(1).toFloat()
            report(
                stage = TokenizerTrainingStage.LEARNING_MERGES,
                fraction = 0.40f + 0.59f * mergeProgress,
                processedItems = learnedMerges.toLong(),
                totalItems = requestedMerges.toLong()
            )
        }

        state = freeze(candidate)
        report(
            stage = TokenizerTrainingStage.COMPLETED,
            fraction = 1f,
            processedItems = learnedMerges.toLong(),
            totalItems = requestedMerges.toLong()
        )
        return state.nextId
    }

    private fun applyMerge(
        symbols: MutableList<String>,
        left: String,
        right: String,
        combined: String
    ) {
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

    private fun encodeUtf8WithReplacement(text: String): ByteArray =
        sanitizeMalformedUtf16(text).toByteArray(StandardCharsets.UTF_8)

    private fun sanitizeMalformedUtf16(text: String): String {
        var index = 0
        var changed = false
        val sanitized = StringBuilder(text.length)
        while (index < text.length) {
            val character = text[index]
            when {
                character.isHighSurrogate() && index + 1 < text.length &&
                    text[index + 1].isLowSurrogate() -> {
                    sanitized.append(character).append(text[index + 1])
                    index += 2
                }
                (character.code in 0xD800..0xDFFF) -> {
                    sanitized.append('\uFFFD')
                    changed = true
                    index++
                }
                else -> {
                    sanitized.append(character)
                    index++
                }
            }
        }
        return if (changed) sanitized.toString() else text
    }

    private fun decodeUtf8WithReplacement(bytes: ByteArray): String =
        StandardCharsets.UTF_8.newDecoder()
            .onMalformedInput(CodingErrorAction.REPLACE)
            .onUnmappableCharacter(CodingErrorAction.REPLACE)
            .replaceWith("\uFFFD")
            .decode(ByteBuffer.wrap(bytes))
            .toString()

    private fun forEachNormalTextSegment(
        text: String,
        action: (String) -> Unit
    ) {
        scanSpecialAwareText(text) { plainText, _ ->
            if (plainText != null) action(plainText)
        }
    }

    /**
     * The one scanner used by training, inference, and payload escaping.
     * U+E000 is ordinary payload unless a doubled marker is immediately followed
     * by a reserved control spelling. That exact three-part sequence escapes the
     * control spelling. Marker runs without a following control spelling are emitted
     * one character at a time, preserving arbitrary literal marker sequences.
     */
    private fun scanSpecialAwareText(
        text: String,
        action: (plainText: String?, specialId: Int?) -> Unit
    ) {
        val plain = StringBuilder()
        var index = 0

        fun flushPlain() {
            if (plain.isNotEmpty()) {
                action(plain.toString(), null)
                plain.setLength(0)
            }
        }

        while (index < text.length) {
            if (text[index] == LITERAL_CONTROL_ESCAPE_MARKER) {
                if (
                    index + 1 < text.length &&
                    text[index + 1] == LITERAL_CONTROL_ESCAPE_MARKER
                ) {
                    val escapedSpecial = specialTokenAt(text, index + 2)
                    if (escapedSpecial != null) {
                        plain.append(escapedSpecial)
                        index += 2 + escapedSpecial.length
                    } else {
                        plain.append(LITERAL_CONTROL_ESCAPE_MARKER)
                        index++
                    }
                } else {
                    plain.append(LITERAL_CONTROL_ESCAPE_MARKER)
                    index++
                }
                continue
            }

            val special = specialTokenAt(text, index)
            if (special != null) {
                flushPlain()
                val specialId = SPECIAL_IDS[special]
                    ?: throw IllegalStateException("Special token '$special' has no canonical ID.")
                action(null, specialId)
                index += special.length
            } else {
                plain.append(text[index])
                index++
            }
        }
        flushPlain()
    }

    /**
     * GPT-style deterministic pre-token boundaries without locale- or ICU-dependent
     * character classification: whitespace prefixes the next token, ASCII punctuation
     * is isolated, and every ASCII digit is emitted separately. All text remains lossless.
     */
    private fun forEachByteLevelPreToken(
        input: String,
        action: (String) -> Unit
    ) {
        if (input.isEmpty()) return
        val text = sanitizeMalformedUtf16(input)
        val pendingWhitespace = StringBuilder()
        val currentWord = StringBuilder()

        fun flushWord() {
            if (currentWord.isNotEmpty()) {
                action(currentWord.toString())
                currentWord.setLength(0)
            }
        }

        var cursor = 0
        while (cursor < text.length) {
            val codePoint = Character.codePointAt(text, cursor)
            val codePointLength = Character.charCount(codePoint)
            val value = String(Character.toChars(codePoint))

            when {
                isUnicodeWhiteSpace(codePoint) -> {
                    flushWord()
                    pendingWhitespace.append(value)
                }
                isAsciiDigit(codePoint) || isPunctuation(codePoint) -> {
                    flushWord()
                    val isolated = StringBuilder(pendingWhitespace.length + value.length)
                        .append(pendingWhitespace)
                        .append(value)
                    pendingWhitespace.setLength(0)
                    action(isolated.toString())
                }
                else -> {
                    if (currentWord.isEmpty() && pendingWhitespace.isNotEmpty()) {
                        currentWord.append(pendingWhitespace)
                        pendingWhitespace.setLength(0)
                    }
                    currentWord.append(value)
                }
            }
            cursor += codePointLength
        }

        flushWord()
        if (pendingWhitespace.isNotEmpty()) {
            action(pendingWhitespace.toString())
        }
    }

    internal fun preTokenizeForTesting(text: String): List<String> {
        val tokens = ArrayList<String>()
        forEachByteLevelPreToken(text) { token -> tokens += token }
        return tokens
    }

    private fun isAsciiDigit(codePoint: Int): Boolean =
        codePoint in '0'.code..'9'.code

    private fun isPunctuation(codePoint: Int): Boolean =
        codePoint in 0x21..0x2F ||
            codePoint in 0x3A..0x40 ||
            codePoint in 0x5B..0x60 ||
            codePoint in 0x7B..0x7E ||
            codePoint in 0x2000..0x206F ||
            codePoint in 0x3001..0x303F ||
            codePoint in 0xFE10..0xFE1F ||
            codePoint in 0xFE30..0xFE4F ||
            codePoint in 0xFF01..0xFF0F ||
            codePoint in 0xFF1A..0xFF20 ||
            codePoint in 0xFF3B..0xFF40 ||
            codePoint in 0xFF5B..0xFF65

    private fun isUnicodeWhiteSpace(codePoint: Int): Boolean =
        codePoint in 0x0009..0x000D ||
            codePoint == 0x0020 ||
            codePoint == 0x0085 ||
            codePoint == 0x00A0 ||
            codePoint == 0x1680 ||
            codePoint in 0x2000..0x200A ||
            codePoint in 0x2028..0x2029 ||
            codePoint == 0x202F ||
            codePoint == 0x205F ||
            codePoint == 0x3000

    private fun mergeRank(
        current: TokenizerState,
        left: String,
        right: String
    ): Int? = current.mergeRanks[left]?.get(right)

    private fun applyRankedMerges(
        initialSymbols: List<String>,
        current: TokenizerState
    ): List<String> {
        if (initialSymbols.size < 2 || current.mergeRanks.isEmpty()) return initialSymbols

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
            val rank = mergeRank(current, left.value, right.value) ?: return
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

    private fun encodeNormalSegment(
        segment: String,
        current: TokenizerState,
        output: IntAccumulator
    ) {
        forEachByteLevelPreToken(segment) { preToken ->
            val bytes = encodeUtf8WithReplacement(preToken)
            val initialSymbols = ArrayList<String>(bytes.size)
            for (byte in bytes) {
                initialSymbols += byteSymbol(byte.toInt() and 0xFF)
            }
            val mergedSymbols = applyRankedMerges(initialSymbols, current)
            for (symbol in mergedSymbols) {
                val id = current.tokenToId[symbol]
                    ?: throw IllegalStateException(
                        "BPE emitted a token that is absent from the published vocabulary."
                    )
                output.add(id)
            }
        }
    }

    /**
     * Lock-free inference entry point. It reads one immutable vocabulary/merge snapshot
     * and returns primitive IDs to keep the training/generation hot path allocation-light.
     */
    private fun encodeTextWithState(text: String, current: TokenizerState): IntArray {
        if (text.isEmpty()) return IntArray(0)
        val output = IntAccumulator(text.length + 8)
        scanSpecialAwareText(text) { plainText, specialId ->
            if (specialId != null) {
                output.add(specialId)
            } else if (plainText != null) {
                encodeNormalSegment(plainText, current, output)
            }
        }
        return output.toIntArray()
    }

    @Suppress("UNUSED_PARAMETER")
    fun encodeToIds(text: String, autoRegister: Boolean = false): IntArray {
        if (text.isEmpty()) return IntArray(0)
        val current = state
        val cached = current.encodedTextCache.get(text)
        if (cached != null) return cached

        val encoded = encodeTextWithState(text, current)
        current.encodedTextCache.put(text, encoded)
        return encoded
    }

    /**
     * Reads incrementally and emits chunks whose boundaries never split a tokenizer
     * pre-token or a reserved control token. One immutable tokenizer generation is
     * used for the entire pass, even if BPE training publishes a new state concurrently.
     *
     * An indivisible pre-token may exceed maxTokensPerChunk; it is kept intact rather
     * than split and assigned different BPE IDs. The caller owns the Reader.
     */
    fun forEachSafeTextChunk(
        reader: Reader,
        maxTokensPerChunk: Int,
        overlapTokens: Int = 0,
        literalControlTokens: Boolean = false,
        allowSpecialTokens: Boolean = true,
        action: (TokenizerTextChunk) -> Unit
    ) {
        require(maxTokensPerChunk > 0) { "maxTokensPerChunk must be positive." }
        require(overlapTokens in 0 until maxTokensPerChunk) {
            "overlapTokens must be between zero and maxTokensPerChunk - 1."
        }

        // Keep the tokenizer identity fixed for the complete streaming document.
        val current = state
        val maximumSpecialLength = SPECIAL_TOKENS.maxOfOrNull { token -> token.length } ?: 1
        val input = PushbackReader(BufferedReader(reader, 8_192), maximumSpecialLength + 2)

        data class SafeUnit(val sourceText: String, val tokenIds: IntArray)

        val currentUnits = ArrayList<SafeUnit>()
        var currentTokenCount = 0
        val pendingWhitespace = StringBuilder()
        val currentWord = StringBuilder()

        fun emitCurrentChunk() {
            if (currentUnits.isEmpty()) return
            val textLength = currentUnits.sumOf { unit -> unit.sourceText.length }
            val tokenCount = currentUnits.sumOf { unit -> unit.tokenIds.size }
            val chunkText = StringBuilder(textLength)
            val chunkTokenIds = IntArray(tokenCount)
            var tokenOffset = 0
            for (unit in currentUnits) {
                chunkText.append(unit.sourceText)
                unit.tokenIds.copyInto(chunkTokenIds, tokenOffset)
                tokenOffset += unit.tokenIds.size
            }
            action(TokenizerTextChunk(chunkText.toString(), chunkTokenIds))
            currentUnits.clear()
            currentTokenCount = 0
        }

        fun appendSafeUnit(sourceText: String, encodingText: String = sourceText) {
            if (sourceText.isEmpty()) return
            val encoded = encodeTextWithState(encodingText, current)
            if (encoded.isEmpty()) return

            if (
                currentUnits.isNotEmpty() &&
                currentTokenCount.toLong() + encoded.size.toLong() > maxTokensPerChunk.toLong()
            ) {
                val previousUnits = currentUnits.toList()
                emitCurrentChunk()

                if (overlapTokens > 0) {
                    val suffix = ArrayDeque<SafeUnit>()
                    var suffixTokenCount = 0
                    for (unit in previousUnits.asReversed()) {
                        val nextCount = suffixTokenCount.toLong() + unit.tokenIds.size.toLong()
                        if (nextCount > overlapTokens.toLong()) break
                        suffix.addFirst(unit)
                        suffixTokenCount += unit.tokenIds.size
                    }
                    currentUnits.addAll(suffix)
                    currentTokenCount = suffixTokenCount
                }
            }

            currentUnits += SafeUnit(sourceText, encoded)
            currentTokenCount = Math.addExact(currentTokenCount, encoded.size)
        }

        fun flushWord() {
            if (currentWord.isNotEmpty()) {
                appendSafeUnit(currentWord.toString())
                currentWord.setLength(0)
            }
        }

        fun flushPlainText() {
            flushWord()
            if (pendingWhitespace.isNotEmpty()) {
                appendSafeUnit(pendingWhitespace.toString())
                pendingWhitespace.setLength(0)
            }
        }

        fun acceptPlainCodePoint(codePoint: Int, sourceText: String) {
            when {
                isUnicodeWhiteSpace(codePoint) -> {
                    flushWord()
                    pendingWhitespace.append(sourceText)
                }
                isAsciiDigit(codePoint) || isPunctuation(codePoint) -> {
                    flushWord()
                    val isolated = StringBuilder(pendingWhitespace.length + sourceText.length)
                        .append(pendingWhitespace)
                        .append(sourceText)
                    pendingWhitespace.setLength(0)
                    appendSafeUnit(isolated.toString())
                }
                else -> {
                    if (currentWord.isEmpty() && pendingWhitespace.isNotEmpty()) {
                        currentWord.append(pendingWhitespace)
                        pendingWhitespace.setLength(0)
                    }
                    currentWord.append(sourceText)
                }
            }
        }

        fun readCodePoint(firstCharacter: Int): Pair<Int, String> {
            val first = firstCharacter.toChar()
            return when {
                first.isHighSurrogate() -> {
                    val next = input.read()
                    if (next >= 0 && next.toChar().isLowSurrogate()) {
                        val second = next.toChar()
                        Character.toCodePoint(first, second) to "$first$second"
                    } else {
                        if (next >= 0) input.unread(next)
                        0xFFFD to first.toString()
                    }
                }
                first.isLowSurrogate() -> 0xFFFD to first.toString()
                else -> first.code to first.toString()
            }
        }

        fun readSpecialAfterOpenAngle(): String? {
            val lookahead = StringBuilder(maximumSpecialLength - 1)
            while (lookahead.length < maximumSpecialLength - 1) {
                val next = input.read()
                if (next < 0) break
                lookahead.append(next.toChar())
            }

            val candidate = "<$lookahead"
            val special = SPECIAL_TOKENS.firstOrNull { token -> candidate.startsWith(token) }
            if (special == null) {
                for (index in lookahead.length - 1 downTo 0) {
                    input.unread(lookahead[index].code)
                }
                return null
            }

            val consumedAfterOpenAngle = special.length - 1
            for (index in lookahead.length - 1 downTo consumedAfterOpenAngle) {
                input.unread(lookahead[index].code)
            }
            return special
        }

        while (true) {
            val next = input.read()
            if (next < 0) break

            when (next.toChar()) {
                '<' -> {
                    val special = readSpecialAfterOpenAngle()
                    if (special != null) {
                        if (literalControlTokens || !allowSpecialTokens) {
                            // Keep the imported spelling byte-for-byte in the chunk text;
                            // escape it only in the tokenization view to block control injection.
                            flushWord()
                            val sourceLiteral = buildString {
                                append(pendingWhitespace)
                                append(special)
                            }
                            val escapedLiteral = buildString {
                                append(pendingWhitespace)
                                append(LITERAL_CONTROL_ESCAPE_MARKER)
                                append(LITERAL_CONTROL_ESCAPE_MARKER)
                                append(special)
                            }
                            pendingWhitespace.setLength(0)
                            appendSafeUnit(sourceLiteral, escapedLiteral)
                        } else {
                            flushPlainText()
                            appendSafeUnit(special)
                        }
                    } else {
                        acceptPlainCodePoint('<'.code, "<")
                    }
                }

                LITERAL_CONTROL_ESCAPE_MARKER -> {
                    var markerCount = 1
                    while (true) {
                        val marker = input.read()
                        if (marker == LITERAL_CONTROL_ESCAPE_MARKER.code) {
                            markerCount = Math.addExact(markerCount, 1)
                        } else {
                            if (marker >= 0) input.unread(marker)
                            break
                        }
                    }

                    val following = input.read()
                    if (markerCount >= 2 && following == '<'.code) {
                        val special = readSpecialAfterOpenAngle()
                        if (special != null) {
                            // Mirror scanSpecialAwareText exactly: the final two markers
                            // escape the control spelling; earlier markers in a longer run
                            // remain ordinary payload characters.
                            repeat(markerCount - 2) {
                                acceptPlainCodePoint(
                                    LITERAL_CONTROL_ESCAPE_MARKER.code,
                                    LITERAL_CONTROL_ESCAPE_MARKER.toString()
                                )
                            }
                            // The escape pair must stay adjacent to the complete control
                            // spelling until encodeTextWithState can recognize it. Keep any
                            // prefix whitespace in the same unit so pre-tokenization remains
                            // identical to the whole-text path.
                            flushWord()
                            val escapedLiteral = buildString {
                                append(pendingWhitespace)
                                append(LITERAL_CONTROL_ESCAPE_MARKER)
                                append(LITERAL_CONTROL_ESCAPE_MARKER)
                                append(special)
                            }
                            pendingWhitespace.setLength(0)
                            appendSafeUnit(escapedLiteral)
                        } else {
                            input.unread('<'.code)
                            repeat(markerCount) {
                                acceptPlainCodePoint(
                                    LITERAL_CONTROL_ESCAPE_MARKER.code,
                                    LITERAL_CONTROL_ESCAPE_MARKER.toString()
                                )
                            }
                        }
                    } else {
                        if (following >= 0) input.unread(following)
                        repeat(markerCount) {
                            acceptPlainCodePoint(
                                LITERAL_CONTROL_ESCAPE_MARKER.code,
                                LITERAL_CONTROL_ESCAPE_MARKER.toString()
                            )
                        }
                    }
                }

                else -> {
                    val (codePoint, sourceText) = readCodePoint(next)
                    acceptPlainCodePoint(codePoint, sourceText)
                }
            }
        }

        flushPlainText()
        emitCurrentChunk()
    }

    @Suppress("UNUSED_PARAMETER")
    fun encode(text: String, autoRegister: Boolean = false): List<Int> =
        encodeToIds(text, autoRegister).toList()

    fun decode(
        tokenIds: List<Int>,
        keepControlTokens: Boolean = false,
        keepReasoningTokens: Boolean = false
    ): String = decode(tokenIds.toIntArray(), keepControlTokens, keepReasoningTokens)

    fun decode(
        tokenIds: IntArray,
        keepControlTokens: Boolean = false,
        keepReasoningTokens: Boolean = false
    ): String {
        if (tokenIds.isEmpty()) return ""
        val current = state
        val output = StringBuilder(tokenIds.size * 2)
        val byteBuffer = ByteArrayOutputStream()

        fun flushBytes() {
            if (byteBuffer.size() == 0) return
            output.append(decodeUtf8WithReplacement(byteBuffer.toByteArray()))
            byteBuffer.reset()
        }

        for (id in tokenIds) {
            val token = current.idToToken[id] ?: continue
            if (token in CONTROL_TOKENS) {
                val preserveToken = keepControlTokens ||
                    (keepReasoningTokens && token in REASONING_TOKENS)
                if (preserveToken) {
                    flushBytes()
                    output.append(token)
                }
            } else {
                for (symbol in token) {
                    require(symbol.code <= 0xFF) {
                        "Byte-level vocabulary contains a non-byte symbol."
                    }
                    byteBuffer.write(symbol.code)
                }
            }
        }

        flushBytes()
        return output.toString()
    }

    fun saveSnapshot(): ByteArray {
        val payload = buildSnapshotPayload(state)
        val digest = MessageDigest.getInstance("SHA-256").digest(payload)
        return payload + digest
    }

    fun snapshotFingerprint(): String {
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(buildSnapshotPayload(state))
        return digest.joinToString(separator = "") { byte ->
            "%02x".format(java.util.Locale.ROOT, byte.toInt() and 0xFF)
        }
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
        require(input.readInt() == config.schemaVersion) {
            "Tokenizer snapshot schema does not match the active configuration."
        }
        require(input.readInt() == config.algorithm.checkpointId) {
            "Tokenizer snapshot algorithm does not match the active configuration."
        }
        require(input.readInt() == config.maxVocabularySize) {
            "Tokenizer snapshot vocabulary capacity does not match the active configuration."
        }
        require(input.readInt() == config.maxTokenLength) {
            "Tokenizer snapshot maximum token length does not match the active configuration."
        }

        val vocabularyCount = input.readInt()
        require(vocabularyCount in BASE_VOCABULARY_SIZE..config.maxVocabularySize) {
            "Tokenizer snapshot vocabulary count is invalid."
        }
        val vocabulary = ArrayList<Pair<String, Int>>(vocabularyCount)
        repeat(vocabularyCount) {
            val id = input.readInt()
            val token = readSnapshotString(input)
            vocabulary += token to id
        }

        val mergeCount = input.readInt()
        require(mergeCount in 0..(config.maxVocabularySize - BASE_VOCABULARY_SIZE)) {
            "Tokenizer snapshot merge count is invalid."
        }
        val importedMerges = ArrayList<Pair<String, String>>(mergeCount)
        repeat(mergeCount) {
            importedMerges += readSnapshotString(input) to readSnapshotString(input)
        }
        require(input.available() == 0) {
            "Tokenizer snapshot contains unexpected trailing payload data."
        }

        val candidate = Tokenizer(config)
        candidate.importVocab(vocabulary)
        candidate.importMerges(importedMerges)
        require(candidate.vocabSize == vocabularyCount) {
            "Tokenizer snapshot vocabulary count does not match its canonical entries."
        }
        state = candidate.state
    }

    private fun buildSnapshotPayload(current: TokenizerState): ByteArray {
        val output = ByteArrayOutputStream()
        DataOutputStream(output).use { data ->
            data.writeInt(SNAPSHOT_MAGIC)
            data.writeInt(CURRENT_SNAPSHOT_FORMAT_VERSION)
            data.writeInt(config.schemaVersion)
            data.writeInt(config.algorithm.checkpointId)
            data.writeInt(config.maxVocabularySize)
            data.writeInt(config.maxTokenLength)

            val vocabulary = current.tokenToId.entries
                .map { entry -> entry.key to entry.value }
                .sortedBy { entry -> entry.second }
            data.writeInt(vocabulary.size)
            for ((token, id) in vocabulary) {
                data.writeInt(id)
                writeSnapshotString(data, token)
            }

            data.writeInt(current.merges.size)
            for (merge in current.merges) {
                writeSnapshotString(data, merge.left)
                writeSnapshotString(data, merge.right)
            }
        }
        return output.toByteArray()
    }

    private fun writeSnapshotString(output: DataOutputStream, value: String) {
        val bytes = value.toByteArray(StandardCharsets.UTF_8)
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
        return try {
            StandardCharsets.UTF_8.newDecoder()
                .onMalformedInput(CodingErrorAction.REPORT)
                .onUnmappableCharacter(CodingErrorAction.REPORT)
                .decode(ByteBuffer.wrap(bytes))
                .toString()
        } catch (exception: CharacterCodingException) {
            throw IllegalArgumentException(
                "Tokenizer snapshot contains invalid UTF-8.",
                exception
            )
        }
    }
}
