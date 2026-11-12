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
        turnResults = emptyList()
    )
}
