package com.argus.agi.model

import java.io.StringReader
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class GrpoTrainerTest {
    @Test
    fun parsesPromptAnswerJsonlBeforeTrainingStarts() {
        val jsonl = listOf(
            JSONObject().put("prompt", "What is 15 multiplied by 5?").put("answer", "75").toString(),
            JSONObject().put("prompt", "Who directed Titanic?").put("answer", "James Cameron").toString()
        ).joinToString("\n")
        val records = parseGrpoJsonl(StringReader(jsonl))
        assertEquals(2, records.size)
        assertEquals("What is 15 multiplied by 5?", records[0].prompt)
        assertEquals("75", records[0].answer)
        assertEquals(2, records[1].lineNumber)
    }

    @Test
    fun rejectsMalformedOrIncompleteRowsBeforeReturningAnyDataset() {
        val jsonl = listOf(
            JSONObject().put("prompt", "valid").put("answer", "row").toString(),
            JSONObject().put("prompt", "missing answer").toString()
        ).joinToString("\n")
        val error = runCatching { parseGrpoJsonl(StringReader(jsonl)) }.exceptionOrNull()
        assertTrue(error is IllegalArgumentException)
        assertTrue(error?.message.orEmpty().contains("line 2"))
    }

    @Test
    fun referenceAnswerRewardScoresExactPartialAndIncorrectResponses() {
        val exact = ReferenceAnswerReward.evaluate("question", "James Cameron", "James Cameron")
        val partial = ReferenceAnswerReward.evaluate("question", "James Cameron", "James")
        val incorrect = ReferenceAnswerReward.evaluate("question", "James Cameron", "Christopher Nolan")
        assertEquals(1f, exact, 0f)
        assertTrue(partial > 0f && partial < exact)
        assertEquals(0f, incorrect, 0f)
    }

    @Test
    fun answerTagsAreUsedWhenScoringReasoningStyleResponses() {
        val reward = ReferenceAnswerReward.evaluate(
            "question", "elok",
            "<think>Reasoning text.</think><answer>elok</answer>"
        )
        assertEquals(1f, reward, 0f)
    }

    @Test
    fun k3KlPenaltyIsZeroAtBehaviorPolicyAndPositiveWhenPoliciesDiverge() {
        val identical = calculateGrpoTokenObjective(
            oldLogProbability = -0.7f,
            currentLogProbability = -0.7f,
            advantage = 1f,
            clipEpsilon = 0.2f,
            klCoefficient = 0.1f
        )
        val diverged = calculateGrpoTokenObjective(
            oldLogProbability = -0.7f,
            currentLogProbability = -0.2f,
            advantage = 1f,
            clipEpsilon = 0.2f,
            klCoefficient = 0.1f
        )
        assertEquals(1f, identical.importanceRatio, 1.0e-6f)
        assertEquals(0f, identical.klPenalty, 1.0e-6f)
        assertTrue(diverged.klPenalty > 0f)
        assertTrue(diverged.isPolicyClipped)
        assertTrue(diverged.derivativeWrtCurrentLogProbability > 0f)
    }

    @Test
    fun clippedNegativeAdvantageUsesLowerRatioBoundary() {
        val result = calculateGrpoTokenObjective(
            oldLogProbability = -0.2f,
            currentLogProbability = -0.9f,
            advantage = -1f,
            clipEpsilon = 0.2f,
            klCoefficient = 0f
        )
        assertTrue(result.importanceRatio < 0.8f)
        assertTrue(result.isPolicyClipped)
        assertEquals(0f, result.derivativeWrtCurrentLogProbability, 0f)
    }

    @Test
    fun groupRelativeAdvantagesAreStandardizedAndUniformGroupsAreSkipped() {
        val advantages = requireNotNull(computeGroupRelativeAdvantages(listOf(0f, 0.5f, 1f)))
        assertEquals(0f, advantages.average().toFloat(), 1.0e-6f)
        assertTrue(advantages[0] < 0f)
        assertEquals(null, computeGroupRelativeAdvantages(listOf(0.5f, 0.5f)))
    }

    @Test
    fun parserRejectsDatasetThatExceedsAggregateCharacterBudget() {
        val records = listOf(
            JSONObject().put("prompt", "first prompt").put("answer", "first answer").toString(),
            JSONObject().put("prompt", "second prompt").put("answer", "second answer").toString()
        ).joinToString("\n")
        val failure = runCatching {
            parseGrpoJsonl(StringReader(records), maxTotalCharacters = 32L)
        }.exceptionOrNull()
        assertTrue(failure is IllegalArgumentException)
        assertTrue(failure?.message.orEmpty().contains("character limit"))
    }

    @Test
    fun configurationRejectsSingleResponseGroupsAndUnsafeValues() {
        assertTrue(runCatching { GrpoTrainingConfig(groupSize = 1) }.isFailure)
        assertTrue(runCatching { GrpoTrainingConfig(temperature = Float.NaN) }.isFailure)
        assertTrue(runCatching { GrpoTrainingConfig(clipEpsilon = 0.9f) }.isFailure)
    }
}
