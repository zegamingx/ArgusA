package com.argus.agi.model

import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [33])
class ConversationalValidationPersistenceTest {

    @Test
    fun acceptedCheckpointAndBaselineSurviveStoreRecreation() {
        val directory = createTemporaryDirectory()
        try {
            val activeCheckpoint = File(directory, "argus_model_weights.bin")
            val acceptedBytes = ByteArray(128) { index -> (index * 3).toByte() }
            activeCheckpoint.writeBytes(acceptedBytes)

            val firstStore = ConversationalValidationPersistence(directory)
            assertTrue(firstStore.ensureAcceptedCheckpoint(activeCheckpoint))
            assertTrue(firstStore.saveBaseline(report(0.30f, meetsThresholds = false)))

            activeCheckpoint.writeBytes(ByteArray(128) { index -> (255 - index).toByte() })

            val secondStore = ConversationalValidationPersistence(directory)
            assertTrue(secondStore.restoreAcceptedCheckpoint(activeCheckpoint))
            assertArrayEquals(acceptedBytes, activeCheckpoint.readBytes())

            val restored = secondStore.loadState()
            assertEquals(ConversationValidationStatus.BASELINE_READY, restored.status)
            assertEquals(0.30f, requireNotNull(restored.baselineReport).overallScore, 0f)
            assertEquals(ConversationValidationContract.SUITE_VERSION, requireNotNull(restored.baselineReport).suiteVersion)
            assertEquals("a".repeat(64), requireNotNull(restored.baselineReport).suiteFingerprint)
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun promotedCheckpointReplacesAcceptedBaselineOnlyAfterMetadataCommit() {
        val directory = createTemporaryDirectory()
        try {
            val activeCheckpoint = File(directory, "argus_model_weights.bin")
            activeCheckpoint.writeBytes(ByteArray(64) { 7 })

            val store = ConversationalValidationPersistence(directory)
            assertTrue(store.ensureAcceptedCheckpoint(activeCheckpoint))
            assertTrue(store.saveBaseline(report(0.20f, meetsThresholds = false)))

            val candidateBytes = ByteArray(64) { 9 }
            activeCheckpoint.writeBytes(candidateBytes)
            val candidate = report(0.42f, meetsThresholds = false)
            val decision = ConversationPromotionDecision(
                accepted = true,
                reason = "Improved without material regression.",
                baselineScore = 0.20f,
                candidateScore = 0.42f,
                qualityReady = false
            )

            assertTrue(store.promoteCheckpoint(activeCheckpoint, candidate, decision))
            val recreated = ConversationalValidationPersistence(directory)
            assertTrue(recreated.restoreAcceptedCheckpoint(activeCheckpoint))
            assertArrayEquals(candidateBytes, activeCheckpoint.readBytes())

            val state = recreated.loadState()
            assertEquals(ConversationValidationStatus.PROMOTED_NOT_READY, state.status)
            assertEquals(0.42f, requireNotNull(state.baselineReport).overallScore, 0f)
            assertFalse(requireNotNull(state.lastDecision).qualityReady)
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test
    fun legacyVersionOneSnapshotLoadsButItsBaselineIsMarkedForReevaluation() {
        val directory = createTemporaryDirectory()
        try {
            val activeCheckpoint = File(directory, "argus_model_weights.bin")
            val acceptedBytes = ByteArray(64) { 11 }
            activeCheckpoint.writeBytes(acceptedBytes)

            val store = ConversationalValidationPersistence(directory)
            assertTrue(store.ensureAcceptedCheckpoint(activeCheckpoint))
            assertTrue(store.saveBaseline(report(0.30f, meetsThresholds = false)))

            val metadata = File(directory, "argus_conversation_quality.json")
            val root = JSONObject(metadata.readText(Charsets.UTF_8))
            root.put("version", 1)
            for (key in listOf("baselineReport", "latestReport")) {
                if (!root.isNull(key)) {
                    val reportJson = root.getJSONObject(key)
                    reportJson.remove("suiteVersion")
                    reportJson.remove("evaluatorVersion")
                    reportJson.remove("suiteFingerprint")
                    reportJson.remove("generationProfileId")
                }
            }
            metadata.writeText(root.toString(), Charsets.UTF_8)

            val recreated = ConversationalValidationPersistence(directory)
            val restored = recreated.loadState()
            val legacyBaseline = requireNotNull(restored.baselineReport)

            assertTrue(recreated.hasAcceptedCheckpoint())
            assertEquals(ConversationValidationStatus.BASELINE_READY, restored.status)
            assertEquals(0, legacyBaseline.suiteVersion)
            assertEquals("legacy", legacyBaseline.suiteFingerprint)
            assertFalse(ConversationValidationContract.isCurrent(legacyBaseline))
            assertTrue(recreated.restoreAcceptedCheckpoint(activeCheckpoint))
            assertArrayEquals(acceptedBytes, activeCheckpoint.readBytes())
        } finally {
            directory.deleteRecursively()
        }
    }

    private fun createTemporaryDirectory(): File {
        val directory = File.createTempFile("argus-quality-gate", "test")
        assertTrue(directory.delete())
        assertTrue(directory.mkdirs())
        return directory
    }

    private fun report(
        score: Float,
        meetsThresholds: Boolean
    ): ConversationalValidationReport = ConversationalValidationReport(
        caseCount = 2,
        turnCount = 3,
        expectedFactGroupCount = 4,
        matchedFactGroupCount = 2,
        factCoverage = score,
        instructionCheckCount = 4,
        passedInstructionCheckCount = 2,
        instructionCompliance = score,
        followUpFactGroupCount = 2,
        matchedFollowUpFactGroupCount = 1,
        multiTurnConsistency = score,
        forbiddenClaimCount = 2,
        matchedForbiddenClaimCount = 0,
        contradictionRate = 0f,
        overallScore = score,
        meetsThresholds = meetsThresholds,
        failedThresholds = if (meetsThresholds) emptyList() else listOf("factCoverage"),
        turnResults = emptyList(),
        suiteVersion = ConversationValidationContract.SUITE_VERSION,
        evaluatorVersion = ConversationValidationContract.EVALUATOR_VERSION,
        suiteFingerprint = "a".repeat(64),
        generationProfileId = ConversationValidationContract.GENERATION_PROFILE_ID
    )
}
