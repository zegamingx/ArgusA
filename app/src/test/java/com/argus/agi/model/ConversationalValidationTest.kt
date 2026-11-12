package com.argus.agi.model

import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ConversationalValidationTest {

    @Test
    fun multiTurnEvaluationRetainsGeneratedHistoryAndScoresAgainstFixedFacts() = runBlocking {
        val prompts = mutableListOf<String>()
        val answers = ArrayDeque(
            listOf(
                "The target is Android 15 and the app uses Kotlin.",
                "You specified Android 15 with Kotlin."
            )
        )
        val suite = listOf(
            ConversationValidationCase(
                id = "multi-turn-fixture",
                turns = listOf(
                    ConversationValidationTurn(
                        userPrompt = "Remember Android 15 and Kotlin.",
                        expectedFactGroups = listOf(setOf("android 15"), setOf("kotlin")),
                        constraints = ConversationInstructionConstraints(maxWordCount = 12)
                    ),
                    ConversationValidationTurn(
                        userPrompt = "Which platform and language did I specify?",
                        expectedFactGroups = listOf(setOf("android 15"), setOf("kotlin")),
                        forbiddenClaims = setOf("android 14", "java"),
                        constraints = ConversationInstructionConstraints(maxWordCount = 12)
                    )
                )
            )
        )

        val report = ConversationalValidationEvaluator.evaluate(
            cases = suite,
            thresholds = ConversationalValidationThresholds(
                minimumFactCoverage = 1f,
                minimumInstructionCompliance = 1f,
                minimumMultiTurnConsistency = 1f,
                maximumContradictionRate = 0f,
                minimumOverallScore = 1f
            )
        ) { prompt ->
            prompts += prompt
            answers.removeFirst()
        }

        assertEquals(2, prompts.size)
        assertTrue(
            prompts[1].contains(
                Tokenizer.ASSISTANT_TOKEN + " The target is Android 15 and the app uses Kotlin."
            )
        )
        assertEquals(1f, report.factCoverage, 0f)
        assertEquals(1f, report.instructionCompliance, 0f)
        assertEquals(1f, report.multiTurnConsistency, 0f)
        assertEquals(0f, report.contradictionRate, 0f)
        assertTrue(report.meetsThresholds)
        assertEquals(ConversationValidationContract.SUITE_VERSION, report.suiteVersion)
        assertEquals(ConversationValidationContract.EVALUATOR_VERSION, report.evaluatorVersion)
        assertTrue(report.suiteFingerprint.matches(Regex("[0-9a-f]{64}")))
    }

    @Test
    fun evaluatorChecksPhraseOrderSingleLineAndMarkdownConstraints() = runBlocking {
        val constraints = ConversationInstructionConstraints(
            minWordCount = 1,
            maxWordCount = 20,
            requiredPhraseOrder = listOf("red", "green", "blue"),
            requireSingleLine = true,
            forbidMarkdown = true
        )
        val report = ConversationalValidationEvaluator.evaluate(
            cases = listOf(
                ConversationValidationCase(
                    id = "ordered-output",
                    turns = listOf(
                        ConversationValidationTurn(
                            userPrompt = "Return red, green, blue in this order on one plain line.",
                            expectedFactGroups = listOf(
                                setOf("red"),
                                setOf("green"),
                                setOf("blue")
                            ),
                            constraints = constraints
                        )
                    )
                )
            )
        ) { "red, green, blue" }

        assertEquals(1f, report.instructionCompliance, 0f)
        assertTrue(report.meetsThresholds)
    }

    @Test
    fun evaluatorRejectsWrongPhraseOrderAndMarkdownFormatting() = runBlocking {
        val report = ConversationalValidationEvaluator.evaluate(
            cases = listOf(
                ConversationValidationCase(
                    id = "invalid-order-and-format",
                    turns = listOf(
                        ConversationValidationTurn(
                            userPrompt = "Return red, green, blue in order on one plain line.",
                            constraints = ConversationInstructionConstraints(
                                requiredPhraseOrder = listOf("red", "green", "blue"),
                                requireSingleLine = true,
                                forbidMarkdown = true
                            )
                        )
                    )
                )
            )
        ) { "# blue green red\nsecond line" }

        assertEquals(0f, report.instructionCompliance, 0f)
        assertFalse(report.meetsThresholds)
    }

    @Test
    fun promotionGateRejectsReportsFromDifferentEvaluationContracts() {
        val baseline = report(
            factCoverage = 0.80f,
            instructionCompliance = 0.80f,
            multiTurnConsistency = 0.80f,
            contradictionRate = 0f,
            overallScore = 0.80f,
            meetsThresholds = true
        )
        val changedSuite = baseline.copy(suiteFingerprint = "b".repeat(64))
        val decision = ConversationalValidationGate.assessPromotion(baseline, changedSuite)

        assertFalse(decision.accepted)
        assertTrue(decision.reason.contains("contracts do not match"))
    }

    @Test
    fun evaluationFingerprintChangesWhenSuiteContentOrThresholdsChange() {
        val thresholds = ConversationalValidationThresholds()
        val original = ConversationValidationSuite.cases
        val originalFingerprint = ConversationValidationContract.fingerprint(original, thresholds)
        val changedCases = original.toMutableList()
        val firstCase = changedCases.first()
        val firstTurn = firstCase.turns.first()
        changedCases[0] = firstCase.copy(
            turns = listOf(firstTurn.copy(userPrompt = firstTurn.userPrompt + " Use the new wording."))
        )

        val changedContentFingerprint = ConversationValidationContract.fingerprint(changedCases, thresholds)
        val changedThresholdFingerprint = ConversationValidationContract.fingerprint(
            original,
            thresholds.copy(minimumOverallScore = 0.76f)
        )

        assertFalse(originalFingerprint == changedContentFingerprint)
        assertFalse(originalFingerprint == changedThresholdFingerprint)
        assertTrue(original.size >= 15)
        assertTrue(original.sumOf { it.turns.size } >= 20)
    }

    @Test
    fun failedFormatAndContradictionChecksReduceTheReport() = runBlocking {
        val report = ConversationalValidationEvaluator.evaluate(
            cases = listOf(
                ConversationValidationCase(
                    id = "instruction-failure",
                    turns = listOf(
                        ConversationValidationTurn(
                            userPrompt = "Answer in exactly two numbered steps.",
                            expectedFactGroups = listOf(setOf("archive")),
                            forbiddenClaims = setOf("delete"),
                            constraints = ConversationInstructionConstraints(
                                minWordCount = 1,
                                maxWordCount = 12,
                                exactNumberedItems = 2
                            )
                        )
                    )
                )
            )
        ) {
            "Save it. Also delete it."
        }

        assertEquals(0f, report.factCoverage, 0f)
        assertTrue(report.instructionCompliance < 1f)
        assertEquals(1f, report.contradictionRate, 0f)
        assertFalse(report.meetsThresholds)
    }

    @Test
    fun promotionGateRequiresImprovementAndNoMaterialMetricRegression() {
        val baseline = report(
            factCoverage = 0.80f,
            instructionCompliance = 0.80f,
            multiTurnConsistency = 0.80f,
            contradictionRate = 0f,
            overallScore = 0.80f,
            meetsThresholds = true
        )
        val improved = report(
            factCoverage = 0.90f,
            instructionCompliance = 0.90f,
            multiTurnConsistency = 0.90f,
            contradictionRate = 0f,
            overallScore = 0.90f,
            meetsThresholds = true
        )

        val accepted = ConversationalValidationGate.assessPromotion(baseline, improved)
        assertTrue(accepted.accepted)

        val regressed = improved.copy(
            instructionCompliance = 0.60f,
            overallScore = 0.92f
        )
        val rejected = ConversationalValidationGate.assessPromotion(baseline, regressed)
        assertFalse(rejected.accepted)
        assertTrue(rejected.reason.contains("instructionCompliance"))
    }

    @Test
    fun belowThresholdCandidateCanAdvanceAsBestKnownWithoutBecomingReleaseReady() {
        val baseline = report(
            factCoverage = 0.20f,
            instructionCompliance = 0.30f,
            multiTurnConsistency = 0.20f,
            contradictionRate = 0.10f,
            overallScore = 0.22f,
            meetsThresholds = false
        )
        val improved = report(
            factCoverage = 0.40f,
            instructionCompliance = 0.50f,
            multiTurnConsistency = 0.40f,
            contradictionRate = 0.05f,
            overallScore = 0.43f,
            meetsThresholds = false
        )

        val decision = ConversationalValidationGate.assessPromotion(baseline, improved)

        assertTrue(decision.accepted)
        assertFalse(decision.qualityReady)
        assertTrue(decision.reason.contains("not release-ready"))
    }

    @Test
    fun previouslyReadyCheckpointCannotBeReplacedByCandidateBelowAbsoluteThresholds() {
        val baseline = report(
            factCoverage = 0.90f,
            instructionCompliance = 0.90f,
            multiTurnConsistency = 0.90f,
            contradictionRate = 0f,
            overallScore = 0.90f,
            meetsThresholds = true
        )
        val regressed = report(
            factCoverage = 0.86f,
            instructionCompliance = 0.86f,
            multiTurnConsistency = 0.86f,
            contradictionRate = 0f,
            overallScore = 0.88f,
            meetsThresholds = false
        )

        val decision = ConversationalValidationGate.assessPromotion(baseline, regressed)

        assertFalse(decision.accepted)
        assertFalse(decision.qualityReady)
        assertTrue(decision.reason.contains("absolute conversational quality thresholds"))
    }

    private fun report(
        factCoverage: Float,
        instructionCompliance: Float,
        multiTurnConsistency: Float,
        contradictionRate: Float,
        overallScore: Float,
        meetsThresholds: Boolean
    ): ConversationalValidationReport = ConversationalValidationReport(
        caseCount = 1,
        turnCount = 1,
        expectedFactGroupCount = 1,
        matchedFactGroupCount = 1,
        factCoverage = factCoverage,
        instructionCheckCount = 1,
        passedInstructionCheckCount = 1,
        instructionCompliance = instructionCompliance,
        followUpFactGroupCount = 1,
        matchedFollowUpFactGroupCount = 1,
        multiTurnConsistency = multiTurnConsistency,
        forbiddenClaimCount = 1,
        matchedForbiddenClaimCount = 0,
        contradictionRate = contradictionRate,
        overallScore = overallScore,
        meetsThresholds = meetsThresholds,
        failedThresholds = emptyList(),
        turnResults = emptyList(),
        suiteVersion = ConversationValidationContract.SUITE_VERSION,
        evaluatorVersion = ConversationValidationContract.EVALUATOR_VERSION,
        suiteFingerprint = "a".repeat(64),
        generationProfileId = ConversationValidationContract.GENERATION_PROFILE_ID
    )
}
