package com.argus.agi.model

import android.content.Context
import com.argus.agi.data.repository.ArgusRepository
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap

data class WorldEntity(
    val key: String,
    val value: String,
    val source: String,
    val timestamp: Long = System.currentTimeMillis()
)

data class ReflectionEvaluation(
    val isConfident: Boolean,
    val confidenceScore: Float,
    val reasoningNote: String,
    val suggestedAction: String?
)

class WorldModel(
    private val context: Context,
    private val repository: ArgusRepository? = null
) {

    private val repo: ArgusRepository? = repository ?: runCatching { ArgusRepository(context) }.getOrNull()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    companion object {
        private const val MAX_ENTITIES = 200
        private const val ENTITY_TTL_MS = 24L * 60 * 60 * 1000 // 24 jam
        private const val SNIPPET_MAX_LEN = 120

        private val TIME_FORMATTER: DateTimeFormatter =
            DateTimeFormatter.ofPattern("HH:mm:ss")

        // Pola redaksi data sensitif sebelum masuk world-model
        private val OTP_PATTERN   = Regex("\\b\\d{4,8}\\b")
        private val EMAIL_PATTERN = Regex("\\b[\\w.+-]+@[\\w.-]+\\.[a-zA-Z]{2,}\\b")

        private val MATH_KEYWORDS = listOf(
            "berapa", "hitung", "calculate", "compute", "hasil",
            "sum", "total", "=", "×", "÷"
        )
    }

    private val entities = ConcurrentHashMap<String, WorldEntity>()
    private val _entitiesFlow = MutableStateFlow<List<WorldEntity>>(emptyList())
    val entitiesFlow: StateFlow<List<WorldEntity>> = _entitiesFlow.asStateFlow()

    private val _currentContextSummary =
        MutableStateFlow("System ready. Waiting for environment and screen input.")
    val currentContextSummary: StateFlow<String> = _currentContextSummary.asStateFlow()

    init {
        storeEntities(
            WorldEntity("identity_name", "A.R.G.U.S", "Core Architecture"),
            WorldEntity(
                "privacy_boundary",
                "Local model inference and training; web, article, and speech services may contact external providers",
                "System Policy"
            ),
            WorldEntity("architecture_type",
                "Autonomous Micro-Transformer with LoRA Adapters", "Neural Specification")
        )

        // Muat memory persisten dari Room database on-device
        scope.launch {
            repo?.let { r ->
                runCatching {
                    val saved = r.getMemoriesSnapshot()
                    if (saved.isNotEmpty()) {
                        saved.forEach { entities[it.key] = it }
                    }

                    // A persisted legacy policy must not restore the old "zero cloud" claim.
                    val currentPrivacyPolicy = WorldEntity(
                        key = "privacy_boundary",
                        value = "Local model inference and training; web, article, and speech services may contact external providers",
                        source = "System Policy"
                    )
                    entities[currentPrivacyPolicy.key] = currentPrivacyPolicy
                    pruneIfNeeded()
                    publishSnapshot()
                    r.saveMemories(listOf(currentPrivacyPolicy))
                }
            }
        }
    }

    fun storeEntity(key: String, value: String, source: String) {
        storeEntities(WorldEntity(key, value, source))
    }

    fun storeEntities(vararg newEntities: WorldEntity) {
        newEntities.forEach { entities[it.key] = it }
        pruneIfNeeded()
        publishSnapshot()
        scope.launch {
            repo?.let { r ->
                runCatching { r.saveMemories(newEntities.toList()) }
            }
        }
    }

    /**
     * Publishes short-lived sensor state without writing high-frequency telemetry to Room.
     * Sensor samples describe the current environment, not durable user knowledge.
     */
    fun updateTransientEntities(vararg newEntities: WorldEntity) {
        newEntities.forEach { entities[it.key] = it }
        pruneInMemoryEntities()
        publishSnapshot()
    }

    private fun pruneInMemoryEntities() {
        val now = System.currentTimeMillis()
        entities.entries.removeIf { (_, entity) ->
            now - entity.timestamp > ENTITY_TTL_MS
        }

        if (entities.size > MAX_ENTITIES) {
            entities.entries
                .sortedBy { it.value.timestamp }
                .take(entities.size - MAX_ENTITIES)
                .forEach { (key, entity) -> entities.remove(key, entity) }
        }
    }

    private fun pruneIfNeeded() {
        pruneInMemoryEntities()
        scope.launch {
            repo?.let { r ->
                runCatching { r.pruneExpiredMemories(ENTITY_TTL_MS) }
            }
        }
    }

    private fun publishSnapshot() {
        _entitiesFlow.update { _ ->
            entities.values.sortedByDescending { it.timestamp }
        }
    }

    fun updateScreenContext(packageName: String, visibleTextSnippet: String) {
        val cleanSnippet = sanitizeSnippet(visibleTextSnippet)
        storeEntities(
            WorldEntity("active_screen_app", packageName, "Accessibility Perception"),
            WorldEntity("last_screen_content", cleanSnippet, "Screen OCR/Text Parser")
        )
        val timeStr = TIME_FORMATTER.format(Instant.now().atZone(ZoneId.systemDefault()))
        _currentContextSummary.value =
            "App: $packageName | Detected: \"$cleanSnippet\" ($timeStr)"
    }

    private fun sanitizeSnippet(raw: String): String {
        return raw.replace(Regex("\\s+"), " ")
            .trim()
            .take(SNIPPET_MAX_LEN)
            .replace(EMAIL_PATTERN, "<email>")
            .replace(OTP_PATTERN) { m ->
                // Redaksi angka 4–8 digit yang kemungkinan OTP/PIN
                if (m.value.length in 4..8) "****" else m.value
            }
    }

    fun updateCameraContext(lumaLevel: Float, visualSummary: String) {
        val safeLuma = if (lumaLevel.isFinite()) lumaLevel.coerceIn(0f, 255f) else 0f
        updateTransientEntities(
            WorldEntity("visual_environment", visualSummary.take(200), "Camera Vision Sensor"),
            WorldEntity(
                "camera_luma",
                String.format(Locale.US, "%.1f/255 relative luminance", safeLuma),
                "Camera Y-plane Sensor",
            ),
        )
    }

    /**
     * Provides diagnostic confidence only. This heuristic must never authorize external tools.
     */
    fun reflectOnResponse(query: String, rawResponse: String): ReflectionEvaluation {
        val trimmed = rawResponse.trim()

        // (1) Respon kosong / terlalu pendek
        if (trimmed.length < 5) {
            return ReflectionEvaluation(
                isConfident = false,
                confidenceScore = 0.05f,
                reasoningNote = "Response is empty or too short to evaluate.",
                suggestedAction = "RETRY_RESPONSE"
            )
        }

        // (2) Indikator ketidaktahuan eksplisit
        val lowerResp = trimmed.lowercase()
        val uncertaintyPatterns = listOf(
            "tidak tahu", "belum ada", "tidak yakin", "tidak ditemukan",
            "tidak dapat menentukan", "unknown", "not sure",
            "i don't know", "cannot determine", "no information"
        )
        if (uncertaintyPatterns.any { lowerResp.contains(it) }) {
            return ReflectionEvaluation(
                isConfident = false,
                confidenceScore = 0.25f,
                reasoningNote = "The model expressed uncertainty; confidence diagnostics do not select external actions.",
                suggestedAction = null
            )
        }

        // (3) Deteksi query matematika dengan heuristik lebih ketat
        if (isMathQuery(query)) {
            val hasNumericAnswer = trimmed.any { it.isDigit() }
            return ReflectionEvaluation(
                isConfident = hasNumericAnswer,
                confidenceScore = if (hasNumericAnswer) 0.90f else 0.40f,
                reasoningNote = if (hasNumericAnswer)
                    "Arithmetic query; the response contains a numeric result."
                else
                    "Arithmetic query; the response does not contain a numeric result.",
                suggestedAction = if (hasNumericAnswer) null else "VERIFY_ARITHMETIC"
            )
        }

        // (4) Skor dasar berdasarkan informasi (panjang & rujukan ke world model)
        val lengthScore = when {
            trimmed.length > 200 -> 0.85f
            trimmed.length > 50  -> 0.75f
            else                 -> 0.65f
        }
        val mentionsKnownEntity = entities.values.any {
            lowerResp.contains(it.value.lowercase().take(20))
        }
        val finalScore = (lengthScore + if (mentionsKnownEntity) 0.05f else 0f)
            .coerceIn(0f, 1f)

        return ReflectionEvaluation(
            isConfident = finalScore >= 0.60f,
            confidenceScore = finalScore,
            reasoningNote = buildString {
                append("Response length score %.2f.".format(lengthScore))
                if (mentionsKnownEntity) append(" References a World Model entity.")
            },
            suggestedAction = null
        )
    }

    private fun isMathQuery(query: String): Boolean {
        val q = query.trim().lowercase()
        val hasDigit = q.any { it.isDigit() }
        val hasOperator = q.any { it in "+-*/^×÷" }
        if (!hasDigit || !hasOperator) return false

        val hasKeyword = MATH_KEYWORDS.any { q.contains(it) }
        val mathCharRatio = q.count {
            it.isDigit() || it in "+-*/^×÷=(). "
        }.toFloat() / q.length

        return hasKeyword || mathCharRatio > 0.6f
    }

    fun getAllEntities(): List<WorldEntity> = entitiesFlow.value

    fun getEntitiesBySource(source: String): List<WorldEntity> =
        entities.values.filter { it.source == source }

    /** Membersihkan konteks sementara (mis. saat app di-background). */
    fun clearTransientContext(
        prefixes: List<String> =
            listOf("active_screen_", "last_screen_", "visual_", "ambient_")
    ) {
        entities.keys.removeIf { key -> prefixes.any { key.startsWith(it) } }
        publishSnapshot()
        _currentContextSummary.value = "System ready."
    }
}