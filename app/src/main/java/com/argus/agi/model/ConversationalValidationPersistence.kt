package com.argus.agi.model

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.security.MessageDigest
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption

internal class ConversationalValidationPersistence(
    private val directory: File
) {
    companion object {
        private const val METADATA_FILE = "argus_conversation_quality.json"
        private const val FORMAT_VERSION = 1
        private const val ACCEPTED_PREFIX = "argus_accepted_"
    }

    private data class StoredSnapshot(
        val acceptedCheckpointFileName: String? = null,
        val acceptedCheckpointSha256: String? = null,
        val baselineReport: ConversationalValidationReport? = null,
        val latestReport: ConversationalValidationReport? = null,
        val lastDecision: ConversationPromotionDecision? = null,
        val status: ConversationValidationStatus = ConversationValidationStatus.IDLE,
        val message: String = "The conversational quality gate is waiting for a training run.",
        val updatedAtEpochMs: Long? = null
    )

    @Synchronized
    fun loadState(): ConversationValidationState {
        val stored = readSnapshot() ?: return ConversationValidationState()
        val interrupted = stored.status == ConversationValidationStatus.PREPARING_BASELINE ||
            stored.status == ConversationValidationStatus.TRAINING ||
            stored.status == ConversationValidationStatus.EVALUATING
        val restoredStatus = if (interrupted) {
            if (stored.baselineReport == null) {
                ConversationValidationStatus.IDLE
            } else {
                ConversationValidationStatus.BASELINE_READY
            }
        } else {
            stored.status
        }
        val restoredMessage = if (interrupted) {
            "A previous validation cycle was interrupted. The last accepted checkpoint will be restored before training resumes."
        } else {
            stored.message
        }
        return ConversationValidationState(
            status = restoredStatus,
            baselineReport = stored.baselineReport,
            latestReport = stored.latestReport,
            lastDecision = stored.lastDecision,
            message = restoredMessage,
            updatedAtEpochMs = stored.updatedAtEpochMs
        )
    }

    @Synchronized
    fun hasAcceptedCheckpoint(): Boolean {
        val stored = readSnapshot() ?: return false
        return isAcceptedCheckpointValid(stored)
    }

    @Synchronized
    fun ensureAcceptedCheckpoint(activeCheckpoint: File): Boolean {
        val existing = readSnapshot()
        if (existing != null && isAcceptedCheckpointValid(existing)) return true
        if (!activeCheckpoint.isFile || activeCheckpoint.length() < 4L) return false

        val captured = captureCheckpoint(activeCheckpoint) ?: return false
        val next = (existing ?: StoredSnapshot()).copy(
            acceptedCheckpointFileName = captured.first,
            acceptedCheckpointSha256 = captured.second,
            baselineReport = null,
            latestReport = null,
            lastDecision = null,
            status = ConversationValidationStatus.IDLE,
            message = "The current checkpoint was secured. Its conversational baseline must be measured before training.",
            updatedAtEpochMs = System.currentTimeMillis()
        )
        if (!writeSnapshot(next)) return false
        existing?.acceptedCheckpointFileName
            ?.takeIf { it != captured.first }
            ?.let { File(directory, it).delete() }
        return true
    }

    @Synchronized
    fun restoreAcceptedCheckpoint(activeCheckpoint: File): Boolean {
        val stored = readSnapshot() ?: return false
        if (!isAcceptedCheckpointValid(stored)) return false
        val source = File(directory, requireNotNull(stored.acceptedCheckpointFileName))
        return copyAtomically(source, activeCheckpoint)
    }

    @Synchronized
    fun saveBaseline(report: ConversationalValidationReport): Boolean {
        val stored = readSnapshot() ?: StoredSnapshot()
        return writeSnapshot(
            stored.copy(
                baselineReport = report,
                latestReport = report,
                lastDecision = null,
                status = ConversationValidationStatus.BASELINE_READY,
                message = "A fixed, human-authored conversational baseline is saved. Future training runs are evaluated against it.",
                updatedAtEpochMs = System.currentTimeMillis()
            )
        )
    }

    @Synchronized
    fun promoteCheckpoint(
        activeCheckpoint: File,
        report: ConversationalValidationReport,
        decision: ConversationPromotionDecision
    ): Boolean {
        if (!decision.accepted) return false
        val previous = readSnapshot()
        val captured = captureCheckpoint(activeCheckpoint) ?: return false
        val nextStatus = if (decision.qualityReady) {
            ConversationValidationStatus.PROMOTED
        } else {
            ConversationValidationStatus.PROMOTED_NOT_READY
        }
        val next = (previous ?: StoredSnapshot()).copy(
            acceptedCheckpointFileName = captured.first,
            acceptedCheckpointSha256 = captured.second,
            baselineReport = report,
            latestReport = report,
            lastDecision = decision,
            status = nextStatus,
            message = decision.reason,
            updatedAtEpochMs = System.currentTimeMillis()
        )
        if (!writeSnapshot(next)) return false
        previous?.acceptedCheckpointFileName
            ?.takeIf { it != captured.first }
            ?.let { File(directory, it).delete() }
        return true
    }

    @Synchronized
    fun recordCandidate(
        report: ConversationalValidationReport,
        decision: ConversationPromotionDecision?,
        status: ConversationValidationStatus,
        message: String
    ): Boolean {
        val stored = readSnapshot() ?: StoredSnapshot()
        return writeSnapshot(
            stored.copy(
                latestReport = report,
                lastDecision = decision,
                status = status,
                message = message,
                updatedAtEpochMs = System.currentTimeMillis()
            )
        )
    }

    @Synchronized
    fun recordStatus(
        status: ConversationValidationStatus,
        message: String,
        clearDecision: Boolean = false
    ): Boolean {
        val stored = readSnapshot() ?: StoredSnapshot()
        return writeSnapshot(
            stored.copy(
                lastDecision = if (clearDecision) null else stored.lastDecision,
                status = status,
                message = message,
                updatedAtEpochMs = System.currentTimeMillis()
            )
        )
    }

    private fun isAcceptedCheckpointValid(stored: StoredSnapshot): Boolean {
        val name = stored.acceptedCheckpointFileName ?: return false
        val expectedHash = stored.acceptedCheckpointSha256 ?: return false
        if (!name.startsWith(ACCEPTED_PREFIX) || !name.endsWith(".bin") || name.contains('/')) {
            return false
        }
        val checkpoint = File(directory, name)
        if (!checkpoint.isFile || checkpoint.length() < 4L) return false
        return runCatching { sha256(checkpoint) == expectedHash }.getOrDefault(false)
    }

    private fun captureCheckpoint(source: File): Pair<String, String>? {
        if (!directory.exists() && !directory.mkdirs()) return null
        return try {
            val hash = sha256(source)
            val name = "$ACCEPTED_PREFIX$hash.bin"
            val target = File(directory, name)
            if (!target.isFile || runCatching { sha256(target) != hash }.getOrDefault(true)) {
                if (!copyAtomically(source, target)) return null
            }
            name to hash
        } catch (_: Exception) {
            null
        }
    }

    private fun copyAtomically(source: File, destination: File): Boolean {
        val parent = destination.parentFile ?: directory
        if (!parent.exists() && !parent.mkdirs()) return false
        val temporary = File(parent, "${destination.name}.${System.nanoTime()}.tmp")
        return try {
            FileInputStream(source).use { input ->
                FileOutputStream(temporary).use { output ->
                    input.copyTo(output)
                    output.fd.sync()
                }
            }
            moveAtomically(temporary, destination)
            true
        } catch (_: Exception) {
            temporary.delete()
            false
        }
    }

    private fun writeSnapshot(stored: StoredSnapshot): Boolean {
        if (!directory.exists() && !directory.mkdirs()) return false
        val temporary = File(directory, "${METADATA_FILE}.${System.nanoTime()}.tmp")
        return try {
            val bytes = encodeSnapshot(stored).toString().toByteArray(Charsets.UTF_8)
            FileOutputStream(temporary).use { output ->
                output.write(bytes)
                output.fd.sync()
            }
            moveAtomically(temporary, File(directory, METADATA_FILE))
            true
        } catch (_: Exception) {
            temporary.delete()
            false
        }
    }

    private fun moveAtomically(source: File, destination: File) {
        try {
            Files.move(
                source.toPath(),
                destination.toPath(),
                StandardCopyOption.ATOMIC_MOVE,
                StandardCopyOption.REPLACE_EXISTING
            )
        } catch (_: AtomicMoveNotSupportedException) {
            Files.move(
                source.toPath(),
                destination.toPath(),
                StandardCopyOption.REPLACE_EXISTING
            )
        }
    }

    private fun readSnapshot(): StoredSnapshot? {
        val file = File(directory, METADATA_FILE)
        if (!file.isFile) return null
        return try {
            val json = JSONObject(file.readText(Charsets.UTF_8))
            require(json.optInt("version") == FORMAT_VERSION)
            StoredSnapshot(
                acceptedCheckpointFileName = json.optString("acceptedCheckpointFileName")
                    .takeIf(String::isNotBlank),
                acceptedCheckpointSha256 = json.optString("acceptedCheckpointSha256")
                    .takeIf(String::isNotBlank),
                baselineReport = if (json.isNull("baselineReport")) null
                    else decodeReport(json.getJSONObject("baselineReport")),
                latestReport = if (json.isNull("latestReport")) null
                    else decodeReport(json.getJSONObject("latestReport")),
                lastDecision = if (json.isNull("lastDecision")) null
                    else decodeDecision(json.getJSONObject("lastDecision")),
                status = runCatching {
                    ConversationValidationStatus.valueOf(json.optString("status"))
                }.getOrDefault(ConversationValidationStatus.IDLE),
                message = json.optString(
                    "message",
                    "The conversational quality gate is waiting for a training run."
                ),
                updatedAtEpochMs = if (json.isNull("updatedAtEpochMs")) null
                    else json.optLong("updatedAtEpochMs")
            )
        } catch (_: Exception) {
            null
        }
    }

    private fun encodeSnapshot(stored: StoredSnapshot): JSONObject = JSONObject()
        .put("version", FORMAT_VERSION)
        .put("acceptedCheckpointFileName", stored.acceptedCheckpointFileName ?: JSONObject.NULL)
        .put("acceptedCheckpointSha256", stored.acceptedCheckpointSha256 ?: JSONObject.NULL)
        .put("baselineReport", stored.baselineReport?.let(::encodeReport) ?: JSONObject.NULL)
        .put("latestReport", stored.latestReport?.let(::encodeReport) ?: JSONObject.NULL)
        .put("lastDecision", stored.lastDecision?.let(::encodeDecision) ?: JSONObject.NULL)
        .put("status", stored.status.name)
        .put("message", stored.message)
        .put("updatedAtEpochMs", stored.updatedAtEpochMs ?: JSONObject.NULL)

    private fun encodeReport(report: ConversationalValidationReport): JSONObject = JSONObject()
        .put("caseCount", report.caseCount)
        .put("turnCount", report.turnCount)
        .put("expectedFactGroupCount", report.expectedFactGroupCount)
        .put("matchedFactGroupCount", report.matchedFactGroupCount)
        .put("factCoverage", report.factCoverage.toDouble())
        .put("instructionCheckCount", report.instructionCheckCount)
        .put("passedInstructionCheckCount", report.passedInstructionCheckCount)
        .put("instructionCompliance", report.instructionCompliance.toDouble())
        .put("followUpFactGroupCount", report.followUpFactGroupCount)
        .put("matchedFollowUpFactGroupCount", report.matchedFollowUpFactGroupCount)
        .put("multiTurnConsistency", report.multiTurnConsistency.toDouble())
        .put("forbiddenClaimCount", report.forbiddenClaimCount)
        .put("matchedForbiddenClaimCount", report.matchedForbiddenClaimCount)
        .put("contradictionRate", report.contradictionRate.toDouble())
        .put("overallScore", report.overallScore.toDouble())
        .put("meetsThresholds", report.meetsThresholds)
        .put("failedThresholds", JSONArray().apply {
            report.failedThresholds.forEach(::put)
        })
        .put("turnResults", JSONArray().apply {
            report.turnResults.forEach { result ->
                put(
                    JSONObject()
                        .put("caseId", result.caseId)
                        .put("turnIndex", result.turnIndex)
                        .put("expectedFactGroupCount", result.expectedFactGroupCount)
                        .put("matchedFactGroupCount", result.matchedFactGroupCount)
                        .put("passedInstructionCheckCount", result.passedInstructionCheckCount)
                        .put("instructionCheckCount", result.instructionCheckCount)
                        .put("forbiddenClaimCount", result.forbiddenClaimCount)
                        .put("matchedForbiddenClaimCount", result.matchedForbiddenClaimCount)
                )
            }
        })

    private fun decodeReport(json: JSONObject): ConversationalValidationReport {
        val failed = json.optJSONArray("failedThresholds") ?: JSONArray()
        val turnResults = json.optJSONArray("turnResults") ?: JSONArray()
        return ConversationalValidationReport(
            caseCount = json.getInt("caseCount"),
            turnCount = json.getInt("turnCount"),
            expectedFactGroupCount = json.getInt("expectedFactGroupCount"),
            matchedFactGroupCount = json.getInt("matchedFactGroupCount"),
            factCoverage = json.getDouble("factCoverage").toFloat(),
            instructionCheckCount = json.getInt("instructionCheckCount"),
            passedInstructionCheckCount = json.getInt("passedInstructionCheckCount"),
            instructionCompliance = json.getDouble("instructionCompliance").toFloat(),
            followUpFactGroupCount = json.getInt("followUpFactGroupCount"),
            matchedFollowUpFactGroupCount = json.getInt("matchedFollowUpFactGroupCount"),
            multiTurnConsistency = json.getDouble("multiTurnConsistency").toFloat(),
            forbiddenClaimCount = json.getInt("forbiddenClaimCount"),
            matchedForbiddenClaimCount = json.getInt("matchedForbiddenClaimCount"),
            contradictionRate = json.getDouble("contradictionRate").toFloat(),
            overallScore = json.getDouble("overallScore").toFloat(),
            meetsThresholds = json.getBoolean("meetsThresholds"),
            failedThresholds = List(failed.length()) { index -> failed.getString(index) },
            turnResults = List(turnResults.length()) { index ->
                val item = turnResults.getJSONObject(index)
                ConversationValidationTurnResult(
                    caseId = item.getString("caseId"),
                    turnIndex = item.getInt("turnIndex"),
                    expectedFactGroupCount = item.getInt("expectedFactGroupCount"),
                    matchedFactGroupCount = item.getInt("matchedFactGroupCount"),
                    passedInstructionCheckCount = item.getInt("passedInstructionCheckCount"),
                    instructionCheckCount = item.getInt("instructionCheckCount"),
                    forbiddenClaimCount = item.getInt("forbiddenClaimCount"),
                    matchedForbiddenClaimCount = item.getInt("matchedForbiddenClaimCount")
                )
            }
        )
    }

    private fun encodeDecision(decision: ConversationPromotionDecision): JSONObject = JSONObject()
        .put("accepted", decision.accepted)
        .put("reason", decision.reason)
        .put("baselineScore", decision.baselineScore.toDouble())
        .put("candidateScore", decision.candidateScore.toDouble())
        .put("qualityReady", decision.qualityReady)

    private fun decodeDecision(json: JSONObject): ConversationPromotionDecision =
        ConversationPromotionDecision(
            accepted = json.getBoolean("accepted"),
            reason = json.getString("reason"),
            baselineScore = json.getDouble("baselineScore").toFloat(),
            candidateScore = json.getDouble("candidateScore").toFloat(),
            qualityReady = json.optBoolean("qualityReady", false)
        )

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        FileInputStream(file).use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        val hash = digest.digest()
        val alphabet = "0123456789abcdef"
        return buildString(hash.size * 2) {
            hash.forEach { byte ->
                val value = byte.toInt() and 0xFF
                append(alphabet[value ushr 4])
                append(alphabet[value and 0x0F])
            }
        }
    }
}
