package com.argus.agi.model

import java.util.Locale

/**
 * Human-authored instructions and acceptance criteria for conversational SFT.
 *
 * This suite is evaluation-only. It is never passed to TrainingManager.enqueueDocument
 * and its reference facts are not produced from model outputs.
 */
data class ConversationInstructionConstraints(
    val minWordCount: Int = 1,
    val maxWordCount: Int = Int.MAX_VALUE,
    val exactNumberedItems: Int? = null,
    val maxSentenceCount: Int? = null
) {
    init {
        require(minWordCount >= 0) { "Minimum word count cannot be negative." }
        require(maxWordCount >= minWordCount) {
            "Maximum word count cannot be smaller than the minimum word count."
        }
        require(exactNumberedItems == null || exactNumberedItems >= 0) {
            "Exact numbered-item count cannot be negative."
        }
        require(maxSentenceCount == null || maxSentenceCount >= 0) {
            "Maximum sentence count cannot be negative."
        }
    }
}

data class ConversationValidationTurn(
    val userPrompt: String,
    /** Every group is one required fact; any normalized alternative satisfies that group. */
    val expectedFactGroups: List<Set<String>> = emptyList(),
    /** Any matched phrase is counted as a contradiction or explicitly forbidden claim. */
    val forbiddenClaims: Set<String> = emptySet(),
    val constraints: ConversationInstructionConstraints = ConversationInstructionConstraints()
) {
    init {
        require(userPrompt.isNotBlank()) { "A validation turn must contain a user prompt." }
        require(expectedFactGroups.all { group -> group.isNotEmpty() && group.all(String::isNotBlank) }) {
            "Expected fact groups must contain non-blank alternatives."
        }
        require(forbiddenClaims.all(String::isNotBlank)) {
            "Forbidden claims cannot contain blank phrases."
        }
    }
}

data class ConversationValidationCase(
    val id: String,
    val turns: List<ConversationValidationTurn>
) {
    init {
        require(id.isNotBlank()) { "Conversation validation case ID cannot be blank." }
        require(turns.isNotEmpty()) { "A validation case must contain at least one turn." }
    }
}

data class ConversationalValidationThresholds(
    val minimumFactCoverage: Float = 0.70f,
    val minimumInstructionCompliance: Float = 0.80f,
    val minimumMultiTurnConsistency: Float = 0.70f,
    val maximumContradictionRate: Float = 0.05f,
    val minimumOverallScore: Float = 0.75f
) {
    init {
        require(
            listOf(
                minimumFactCoverage,
                minimumInstructionCompliance,
                minimumMultiTurnConsistency,
                maximumContradictionRate,
                minimumOverallScore
            ).all { it.isFinite() && it in 0f..1f }
        ) { "Conversational validation thresholds must be finite values from 0 to 1." }
    }
}

data class ConversationValidationTurnResult(
    val caseId: String,
    val turnIndex: Int,
    val expectedFactGroupCount: Int,
    val matchedFactGroupCount: Int,
    val passedInstructionCheckCount: Int,
    val instructionCheckCount: Int,
    val forbiddenClaimCount: Int,
    val matchedForbiddenClaimCount: Int
)

data class ConversationalValidationReport(
    val caseCount: Int,
    val turnCount: Int,
    val expectedFactGroupCount: Int,
    val matchedFactGroupCount: Int,
    val factCoverage: Float,
    val instructionCheckCount: Int,
    val passedInstructionCheckCount: Int,
    val instructionCompliance: Float,
    val followUpFactGroupCount: Int,
    val matchedFollowUpFactGroupCount: Int,
    val multiTurnConsistency: Float,
    val forbiddenClaimCount: Int,
    val matchedForbiddenClaimCount: Int,
    val contradictionRate: Float,
    val overallScore: Float,
    val meetsThresholds: Boolean,
    val failedThresholds: List<String>,
    val turnResults: List<ConversationValidationTurnResult>
)

data class ConversationPromotionDecision(
    val accepted: Boolean,
    val reason: String,
    val baselineScore: Float,
    val candidateScore: Float,
    val qualityReady: Boolean = false
)

enum class ConversationValidationStatus {
    IDLE,
    PREPARING_BASELINE,
    BASELINE_READY,
    TRAINING,
    EVALUATING,
    PROMOTED,
    PROMOTED_NOT_READY,
    ROLLED_BACK,
    FAILED
}

data class ConversationValidationState(
    val status: ConversationValidationStatus = ConversationValidationStatus.IDLE,
    val baselineReport: ConversationalValidationReport? = null,
    val latestReport: ConversationalValidationReport? = null,
    val lastDecision: ConversationPromotionDecision? = null,
    val message: String = "The conversational quality gate is waiting for a training run.",
    val updatedAtEpochMs: Long? = null
)

/**
 * Fixed, hand-authored evaluation prompts. Keep separate from any train or replay corpus.
 * Follow-up prompts depend on the preceding model answer, so the model must preserve context.
 */
object ConversationValidationSuite {
    val cases: List<ConversationValidationCase> = listOf(
        ConversationValidationCase(
            id = "short-factual-answer",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "What is 2 + 2? Reply with only the result.",
                    expectedFactGroups = listOf(setOf("4", "four")),
                    constraints = ConversationInstructionConstraints(
                        minWordCount = 1,
                        maxWordCount = 2
                    )
                )
            )
        ),
        ConversationValidationCase(
            id = "format-and-instruction-following",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "Give exactly two numbered steps for saving a file. Do not add an introduction.",
                    expectedFactGroups = listOf(
                        setOf("save", "saving"),
                        setOf("file", "document")
                    ),
                    constraints = ConversationInstructionConstraints(
                        minWordCount = 6,
                        maxWordCount = 48,
                        exactNumberedItems = 2,
                        maxSentenceCount = 3
                    )
                )
            )
        ),
        ConversationValidationCase(
            id = "multi-turn-context-retention",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "For this conversation, remember that the release target is Android 15 and the implementation language is Kotlin. Confirm both briefly.",
                    expectedFactGroups = listOf(
                        setOf("android 15"),
                        setOf("kotlin")
                    ),
                    constraints = ConversationInstructionConstraints(
                        minWordCount = 2,
                        maxWordCount = 18
                    )
                ),
                ConversationValidationTurn(
                    userPrompt = "Which Android version and implementation language did I specify? Answer in one sentence.",
                    expectedFactGroups = listOf(
                        setOf("android 15"),
                        setOf("kotlin")
                    ),
                    forbiddenClaims = setOf("android 14", "android 16", "java"),
                    constraints = ConversationInstructionConstraints(
                        minWordCount = 4,
                        maxWordCount = 20,
                        maxSentenceCount = 1
                    )
                )
            )
        ),
        ConversationValidationCase(
            id = "independent-arithmetic-check",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "Calculate 3 + 3. Reply with only the result.",
                    expectedFactGroups = listOf(setOf("6", "six")),
                    constraints = ConversationInstructionConstraints(
                        minWordCount = 1,
                        maxWordCount = 2,
                        maxSentenceCount = 1
                    )
                )
            )
        ),
        ConversationValidationCase(
            id = "corrected-multi-turn-context",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "Remember these exact project choices: the release branch is main, and the test framework is JUnit. Confirm both briefly.",
                    expectedFactGroups = listOf(setOf("main"), setOf("junit")),
                    forbiddenClaims = setOf("master", "testng"),
                    constraints = ConversationInstructionConstraints(
                        minWordCount = 2,
                        maxWordCount = 18
                    )
                ),
                ConversationValidationTurn(
                    userPrompt = "Which release branch and test framework did I specify? Answer in one sentence.",
                    expectedFactGroups = listOf(setOf("main"), setOf("junit")),
                    forbiddenClaims = setOf("master", "testng"),
                    constraints = ConversationInstructionConstraints(
                        minWordCount = 4,
                        maxWordCount = 20,
                        maxSentenceCount = 1
                    )
                )
            )
        )
    )
}

object ConversationalValidationEvaluator {

    private val wordPattern = Regex("[\\p{L}\\p{N}]+")
    private val numberedItemPattern = Regex("^\\d+[.)]\\s+\\S.*")

    suspend fun evaluate(
        cases: List<ConversationValidationCase>,
        thresholds: ConversationalValidationThresholds = ConversationalValidationThresholds(),
        generate: suspend (conversationPrompt: String) -> String
    ): ConversationalValidationReport {
        require(cases.isNotEmpty()) { "At least one conversational validation case is required." }
        require(cases.map(ConversationValidationCase::id).distinct().size == cases.size) {
            "Conversational validation case IDs must be unique."
        }

        var expectedFactGroupCount = 0
        var matchedFactGroupCount = 0
        var instructionCheckCount = 0
        var passedInstructionCheckCount = 0
        var followUpFactGroupCount = 0
        var matchedFollowUpFactGroupCount = 0
        var forbiddenClaimCount = 0
        var matchedForbiddenClaimCount = 0
        var turnCount = 0
        val turnResults = mutableListOf<ConversationValidationTurnResult>()

        for (case in cases) {
            val conversation = StringBuilder()
            for ((turnIndex, turn) in case.turns.withIndex()) {
                if (conversation.isNotEmpty()) conversation.append('\n')
                conversation.append(Tokenizer.USER_TOKEN)
                    .append(' ')
                    .append(turn.userPrompt.trim())

                // The previous generated assistant turn is retained in context. Expected
                // facts remain fixed in the suite and are never inferred from generated text.
                val answer = generate(conversation.toString()).trim()
                conversation.append('\n')
                    .append(Tokenizer.ASSISTANT_TOKEN)
                    .append(' ')
                    .append(answer)

                val normalizedAnswer = normalize(answer)
                val matchedFacts = turn.expectedFactGroups.count { group ->
                    group.any { phrase -> containsPhrase(normalizedAnswer, phrase) }
                }
                val instructionResults = evaluateInstructions(answer, turn.constraints)
                val matchedForbidden = turn.forbiddenClaims.count { phrase ->
                    containsPhrase(normalizedAnswer, phrase)
                }

                expectedFactGroupCount += turn.expectedFactGroups.size
                matchedFactGroupCount += matchedFacts
                if (turnIndex > 0) {
                    followUpFactGroupCount += turn.expectedFactGroups.size
                    matchedFollowUpFactGroupCount += matchedFacts
                }
                instructionCheckCount += instructionResults.size
                passedInstructionCheckCount += instructionResults.count { it }
                forbiddenClaimCount += turn.forbiddenClaims.size
                matchedForbiddenClaimCount += matchedForbidden
                turnCount++

                turnResults += ConversationValidationTurnResult(
                    caseId = case.id,
                    turnIndex = turnIndex,
                    expectedFactGroupCount = turn.expectedFactGroups.size,
                    matchedFactGroupCount = matchedFacts,
                    passedInstructionCheckCount = instructionResults.count { it },
                    instructionCheckCount = instructionResults.size,
                    forbiddenClaimCount = turn.forbiddenClaims.size,
                    matchedForbiddenClaimCount = matchedForbidden
                )
            }
        }

        val factCoverage = ratio(matchedFactGroupCount, expectedFactGroupCount)
        val instructionCompliance = ratio(passedInstructionCheckCount, instructionCheckCount)
        val multiTurnConsistency = ratio(
            matchedFollowUpFactGroupCount,
            followUpFactGroupCount
        )
        val contradictionRate = if (forbiddenClaimCount == 0) {
            0f
        } else {
            ratio(matchedForbiddenClaimCount, forbiddenClaimCount)
        }
        val answerQuality = 0.75f * factCoverage + 0.25f * (1f - contradictionRate)
        val overallScore = (
            0.50f * answerQuality +
                0.25f * instructionCompliance +
                0.25f * multiTurnConsistency
            ).coerceIn(0f, 1f)

        val failedThresholds = buildList {
            if (factCoverage < thresholds.minimumFactCoverage) add("factCoverage")
            if (instructionCompliance < thresholds.minimumInstructionCompliance) {
                add("instructionCompliance")
            }
            if (multiTurnConsistency < thresholds.minimumMultiTurnConsistency) {
                add("multiTurnConsistency")
            }
            if (contradictionRate > thresholds.maximumContradictionRate) add("contradictionRate")
            if (overallScore < thresholds.minimumOverallScore) add("overallScore")
        }

        return ConversationalValidationReport(
            caseCount = cases.size,
            turnCount = turnCount,
            expectedFactGroupCount = expectedFactGroupCount,
            matchedFactGroupCount = matchedFactGroupCount,
            factCoverage = factCoverage,
            instructionCheckCount = instructionCheckCount,
            passedInstructionCheckCount = passedInstructionCheckCount,
            instructionCompliance = instructionCompliance,
            followUpFactGroupCount = followUpFactGroupCount,
            matchedFollowUpFactGroupCount = matchedFollowUpFactGroupCount,
            multiTurnConsistency = multiTurnConsistency,
            forbiddenClaimCount = forbiddenClaimCount,
            matchedForbiddenClaimCount = matchedForbiddenClaimCount,
            contradictionRate = contradictionRate,
            overallScore = overallScore,
            meetsThresholds = failedThresholds.isEmpty(),
            failedThresholds = failedThresholds,
            turnResults = turnResults
        )
    }

    private fun evaluateInstructions(
        answer: String,
        constraints: ConversationInstructionConstraints
    ): List<Boolean> {
        val words = wordPattern.findAll(answer).count()
        val checks = mutableListOf<Boolean>()
        checks += words >= constraints.minWordCount
        if (constraints.maxWordCount != Int.MAX_VALUE) {
            checks += words <= constraints.maxWordCount
        }
        constraints.exactNumberedItems?.let { expected ->
            val actual = answer.lineSequence()
                .count { line -> numberedItemPattern.matches(line.trim()) }
            checks += actual == expected
        }
        constraints.maxSentenceCount?.let { maximum ->
            val sentenceCount = answer.trim()
                .split(Regex("[.!?]+"))
                .count(String::isNotBlank)
            checks += sentenceCount <= maximum
        }
        return checks
    }

    private fun normalize(value: String): String =
        value.lowercase(Locale.ROOT)
            .replace(Regex("[^\\p{L}\\p{N}]+"), " ")
            .trim()
            .replace(Regex("\\s+"), " ")

    private fun containsPhrase(normalizedAnswer: String, phrase: String): Boolean {
        val normalizedPhrase = normalize(phrase)
        if (normalizedPhrase.isBlank()) return false
        return " $normalizedAnswer ".contains(" $normalizedPhrase ")
    }

    private fun ratio(numerator: Int, denominator: Int): Float =
        if (denominator == 0) 1f else numerator.toFloat() / denominator.toFloat()
}

/**
 * A candidate must clear absolute thresholds, improve aggregate score over the baseline,
 * and avoid a material regression in any key metric.
 */
object ConversationalValidationGate {

    /**
     * A candidate advances only when it improves the accepted baseline and avoids
     * material regression. A below-threshold model may advance as best-known, but it
     * remains explicitly not release-ready until every absolute threshold is met.
     */
    fun assessPromotion(
        baseline: ConversationalValidationReport,
        candidate: ConversationalValidationReport,
        minimumOverallGain: Float = 0.01f,
        maximumMetricRegression: Float = 0.02f
    ): ConversationPromotionDecision {
        require(minimumOverallGain.isFinite() && minimumOverallGain >= 0f) {
            "Minimum overall gain must be finite and non-negative."
        }
        require(maximumMetricRegression.isFinite() && maximumMetricRegression >= 0f) {
            "Maximum metric regression must be finite and non-negative."
        }

        if (baseline.meetsThresholds && !candidate.meetsThresholds) {
            return rejected(
                baseline,
                candidate,
                "Candidate no longer clears the absolute conversational quality thresholds: " +
                    candidate.failedThresholds.joinToString() + "."
            )
        }

        val regressions = listOf(
            "factCoverage" to (baseline.factCoverage - candidate.factCoverage),
            "instructionCompliance" to
                (baseline.instructionCompliance - candidate.instructionCompliance),
            "multiTurnConsistency" to
                (baseline.multiTurnConsistency - candidate.multiTurnConsistency),
            "contradictionRate" to (candidate.contradictionRate - baseline.contradictionRate)
        ).filter { (_, regression) -> regression > maximumMetricRegression }

        if (regressions.isNotEmpty()) {
            return rejected(
                baseline,
                candidate,
                "Candidate regressed beyond tolerance in: " +
                    regressions.joinToString { it.first } + "."
            )
        }

        if (candidate.overallScore < baseline.overallScore + minimumOverallGain) {
            return rejected(
                baseline,
                candidate,
                "Candidate did not improve overall conversational score by the required gain."
            )
        }

        val qualityReady = candidate.meetsThresholds
        val reason = if (qualityReady) {
            "Candidate improved the baseline, cleared all absolute conversational thresholds, " +
                "and stayed within per-metric regression tolerance."
        } else {
            "Candidate improved the best-known baseline without material metric regression. " +
                "It is promoted for continued local training but is not release-ready; " +
                "remaining thresholds: ${candidate.failedThresholds.joinToString()}."
        }
        return ConversationPromotionDecision(
            accepted = true,
            reason = reason,
            baselineScore = baseline.overallScore,
            candidateScore = candidate.overallScore,
            qualityReady = qualityReady
        )
    }

    private fun rejected(
        baseline: ConversationalValidationReport,
        candidate: ConversationalValidationReport,
        reason: String
    ): ConversationPromotionDecision = ConversationPromotionDecision(
        accepted = false,
        reason = reason,
        baselineScore = baseline.overallScore,
        candidateScore = candidate.overallScore,
        qualityReady = false
    )
}
