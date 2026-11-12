package com.argus.agi.model

import java.security.MessageDigest
import java.nio.charset.StandardCharsets
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
    val maxSentenceCount: Int? = null,
    val requiredPhraseOrder: List<String> = emptyList(),
    val requireSingleLine: Boolean = false,
    val forbidMarkdown: Boolean = false
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
        require(requiredPhraseOrder.all(String::isNotBlank)) {
            "Required phrase order cannot contain blank phrases."
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
    val turnResults: List<ConversationValidationTurnResult>,
    val suiteVersion: Int = 0,
    val evaluatorVersion: Int = 0,
    val suiteFingerprint: String = "legacy",
    val generationProfileId: String = "legacy"
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
 * Defines the exact evaluation contract used by baseline/candidate comparisons.
 * A changed suite, scorer, threshold set, or generation profile requires a new baseline.
 */
object ConversationValidationContract {
    const val SUITE_VERSION = 2
    const val EVALUATOR_VERSION = 2
    const val GENERATION_PROFILE_ID = "greedy-max80-no-reasoning-v1"

    fun fingerprint(
        cases: List<ConversationValidationCase>,
        thresholds: ConversationalValidationThresholds
    ): String {
        val canonical = buildString {
            appendField(SUITE_VERSION.toString())
            appendField(EVALUATOR_VERSION.toString())
            appendField(GENERATION_PROFILE_ID)
            appendField(thresholds.minimumFactCoverage.toBits().toString())
            appendField(thresholds.minimumInstructionCompliance.toBits().toString())
            appendField(thresholds.minimumMultiTurnConsistency.toBits().toString())
            appendField(thresholds.maximumContradictionRate.toBits().toString())
            appendField(thresholds.minimumOverallScore.toBits().toString())
            appendField(cases.size.toString())

            cases.forEachIndexed { caseIndex, case ->
                appendField(caseIndex.toString())
                appendField(case.id)
                appendField(case.turns.size.toString())
                case.turns.forEachIndexed { turnIndex, turn ->
                    appendField(turnIndex.toString())
                    appendField(turn.userPrompt.trim())
                    appendField(turn.expectedFactGroups.size.toString())
                    turn.expectedFactGroups.forEach { group ->
                        val alternatives = group.sorted()
                        appendField(alternatives.size.toString())
                        alternatives.forEach { alternative -> appendField(alternative) }
                    }
                    val forbidden = turn.forbiddenClaims.sorted()
                    appendField(forbidden.size.toString())
                    forbidden.forEach { claim -> appendField(claim) }
                    appendField(turn.constraints.minWordCount.toString())
                    appendField(turn.constraints.maxWordCount.toString())
                    appendField(turn.constraints.exactNumberedItems?.toString() ?: "none")
                    appendField(turn.constraints.maxSentenceCount?.toString() ?: "none")
                    appendField(turn.constraints.requiredPhraseOrder.size.toString())
                    turn.constraints.requiredPhraseOrder.forEach { phrase -> appendField(phrase) }
                    appendField(turn.constraints.requireSingleLine.toString())
                    appendField(turn.constraints.forbidMarkdown.toString())
                }
            }
        }
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(canonical.toByteArray(StandardCharsets.UTF_8))
        return digest.joinToString(separator = "") { byte ->
            "%02x".format(Locale.ROOT, byte.toInt() and 0xFF)
        }
    }

    fun isCurrent(
        report: ConversationalValidationReport,
        cases: List<ConversationValidationCase> = ConversationValidationSuite.cases,
        thresholds: ConversationalValidationThresholds = ConversationalValidationThresholds()
    ): Boolean {
        return report.suiteVersion == SUITE_VERSION &&
            report.evaluatorVersion == EVALUATOR_VERSION &&
            report.generationProfileId == GENERATION_PROFILE_ID &&
            report.suiteFingerprint == fingerprint(cases, thresholds)
    }

    fun hasCompatibleContract(
        baseline: ConversationalValidationReport,
        candidate: ConversationalValidationReport
    ): Boolean {
        return isValidMetadata(baseline) &&
            isValidMetadata(candidate) &&
            baseline.suiteVersion == candidate.suiteVersion &&
            baseline.evaluatorVersion == candidate.evaluatorVersion &&
            baseline.suiteFingerprint == candidate.suiteFingerprint &&
            baseline.generationProfileId == candidate.generationProfileId
    }

    private fun isValidMetadata(report: ConversationalValidationReport): Boolean {
        return report.suiteVersion == SUITE_VERSION &&
            report.evaluatorVersion == EVALUATOR_VERSION &&
            report.generationProfileId == GENERATION_PROFILE_ID &&
            report.suiteFingerprint.matches(Regex("[0-9a-f]{64}"))
    }

    private fun StringBuilder.appendField(value: String) {
        append(value.length)
            .append(':')
            .append(value)
            .append(';')
    }
}

/**
 * Fixed, hand-authored evaluation prompts. Keep separate from any train or replay corpus.
 * Follow-up prompts depend on the preceding model answer, so the model must preserve context.
 */
object ConversationValidationSuite {
    const val VERSION = ConversationValidationContract.SUITE_VERSION

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
    ) + listOf(
        ConversationValidationCase(
            id = "indonesian-arithmetic-check",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "Berapa 8 + 5? Jawab hanya angka.",
                    expectedFactGroups = listOf(setOf("13", "tiga belas")),
                    constraints = ConversationInstructionConstraints(
                        minWordCount = 1,
                        maxWordCount = 3,
                        requireSingleLine = true,
                        forbidMarkdown = true
                    )
                )
            )
        ),
        ConversationValidationCase(
            id = "indonesian-translation-good-night",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "Terjemahkan frasa 'Good night' ke bahasa Indonesia. Jawab hanya terjemahannya.",
                    expectedFactGroups = listOf(setOf("selamat malam")),
                    constraints = ConversationInstructionConstraints(
                        minWordCount = 2,
                        maxWordCount = 3,
                        maxSentenceCount = 1,
                        requireSingleLine = true
                    )
                )
            )
        ),
        ConversationValidationCase(
            id = "indonesian-project-context-retention",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "Ingat untuk percakapan ini: nama proyeknya Argus dan target platformnya Android. Konfirmasi singkat.",
                    expectedFactGroups = listOf(setOf("argus"), setOf("android")),
                    constraints = ConversationInstructionConstraints(minWordCount = 2, maxWordCount = 14)
                ),
                ConversationValidationTurn(
                    userPrompt = "Apa nama proyek dan platform target yang tadi saya sebutkan? Jawab satu kalimat.",
                    expectedFactGroups = listOf(setOf("argus"), setOf("android")),
                    forbiddenClaims = setOf("ios", "neural"),
                    constraints = ConversationInstructionConstraints(
                        minWordCount = 3,
                        maxWordCount = 16,
                        maxSentenceCount = 1
                    )
                )
            )
        ),
        ConversationValidationCase(
            id = "indonesian-uncertainty-withheld-information",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "Berapa nomor seri perangkat saya? Saya belum pernah memberikan nomor serinya. Jangan menebak; jawab bahwa informasinya belum tersedia.",
                    expectedFactGroups = listOf(
                        setOf("belum tersedia", "tidak tersedia", "belum diberikan", "tidak tahu")
                    ),
                    forbiddenClaims = setOf("12345678", "98765432", "abcdef"),
                    constraints = ConversationInstructionConstraints(
                        minWordCount = 2,
                        maxWordCount = 16,
                        maxSentenceCount = 2
                    )
                )
            )
        ),
        ConversationValidationCase(
            id = "corrected-fact-replaces-obsolete-value",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "Untuk contoh ini, versi target awal adalah Android 14. Konfirmasi singkat.",
                    expectedFactGroups = listOf(setOf("android 14"))
                ),
                ConversationValidationTurn(
                    userPrompt = "Koreksi: versi target sekarang Android 15, bukan Android 14. Gunakan informasi terbaru ini.",
                    expectedFactGroups = listOf(setOf("android 15")),
                    forbiddenClaims = setOf("android 14")
                ),
                ConversationValidationTurn(
                    userPrompt = "Apa versi target terbaru setelah koreksi saya?",
                    expectedFactGroups = listOf(setOf("android 15")),
                    forbiddenClaims = setOf("android 14"),
                    constraints = ConversationInstructionConstraints(maxWordCount = 12)
                )
            )
        ),
        ConversationValidationCase(
            id = "updated-numeric-preference",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "Anggaran awal untuk contoh ini adalah 100 ribu rupiah. Ingat angka tersebut.",
                    expectedFactGroups = listOf(setOf("100"))
                ),
                ConversationValidationTurn(
                    userPrompt = "Perbarui anggarannya menjadi 150 ribu rupiah. Angka 100 ribu sudah tidak berlaku.",
                    expectedFactGroups = listOf(setOf("150")),
                    forbiddenClaims = setOf("100 ribu")
                ),
                ConversationValidationTurn(
                    userPrompt = "Berapa anggaran terbaru yang harus digunakan?",
                    expectedFactGroups = listOf(setOf("150")),
                    forbiddenClaims = setOf("100 ribu"),
                    constraints = ConversationInstructionConstraints(maxWordCount = 10)
                )
            )
        ),
        ConversationValidationCase(
            id = "three-turn-attribute-consistency",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "Ingat detail item contoh: warna cyan dan ukuran medium.",
                    expectedFactGroups = listOf(setOf("cyan"), setOf("medium"))
                ),
                ConversationValidationTurn(
                    userPrompt = "Apa warna item tersebut?",
                    expectedFactGroups = listOf(setOf("cyan")),
                    forbiddenClaims = setOf("magenta", "red")
                ),
                ConversationValidationTurn(
                    userPrompt = "Apa ukurannya?",
                    expectedFactGroups = listOf(setOf("medium")),
                    forbiddenClaims = setOf("small", "large")
                )
            )
        ),
        ConversationValidationCase(
            id = "ordered-numbered-instructions",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "Tulis tiga warna ini dalam urutan yang tepat sebagai daftar bernomor: merah, hijau, biru. Jangan tambahkan teks lain.",
                    expectedFactGroups = listOf(setOf("merah"), setOf("hijau"), setOf("biru")),
                    constraints = ConversationInstructionConstraints(
                        minWordCount = 6,
                        maxWordCount = 18,
                        exactNumberedItems = 3,
                        maxSentenceCount = 3,
                        requiredPhraseOrder = listOf("merah", "hijau", "biru")
                    )
                )
            )
        ),
        ConversationValidationCase(
            id = "source-grounded-summary",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "Ringkas informasi ini dalam satu kalimat: Rapat berlangsung Jumat pukul 14:00 di Jakarta. Agendanya adalah validasi model.",
                    expectedFactGroups = listOf(
                        setOf("jumat", "friday"),
                        setOf("14:00", "14 00", "2 pm"),
                        setOf("jakarta"),
                        setOf("validasi model", "model validation")
                    ),
                    forbiddenClaims = setOf("kamis", "thursday", "15:00", "surabaya"),
                    constraints = ConversationInstructionConstraints(
                        minWordCount = 5,
                        maxWordCount = 24,
                        maxSentenceCount = 1
                    )
                )
            )
        ),
        ConversationValidationCase(
            id = "entity-disambiguation-java",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "Apakah Java dan JavaScript merupakan bahasa pemrograman yang sama? Jawab secara langsung.",
                    expectedFactGroups = listOf(
                        setOf("berbeda", "tidak sama", "different", "distinct", "bukan bahasa yang sama", "not the same")
                    ),
                    forbiddenClaims = setOf("sama persis", "identical"),
                    constraints = ConversationInstructionConstraints(minWordCount = 3, maxWordCount = 24)
                )
            )
        ),
        ConversationValidationCase(
            id = "english-uncertainty-withheld-information",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "What is the exact private access code for this test? I have not provided it. Do not guess; say that the information is unavailable.",
                    expectedFactGroups = listOf(
                        setOf("not provided", "not available", "do not know", "unknown")
                    ),
                    forbiddenClaims = setOf("12345678", "98765432", "letmein"),
                    constraints = ConversationInstructionConstraints(
                        minWordCount = 2,
                        maxWordCount = 16,
                        maxSentenceCount = 2
                    )
                )
            )
        ),
        ConversationValidationCase(
            id = "single-line-direct-answer",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "What is the capital of Japan? Reply with the city name only on one line.",
                    expectedFactGroups = listOf(setOf("tokyo")),
                    forbiddenClaims = setOf("kyoto", "osaka"),
                    constraints = ConversationInstructionConstraints(
                        minWordCount = 1,
                        maxWordCount = 2,
                        requireSingleLine = true,
                        forbidMarkdown = true
                    )
                )
            )
        ),
        ConversationValidationCase(
            id = "follow-up-topic-switch",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "The first sample project is called Atlas.",
                    expectedFactGroups = listOf(setOf("atlas"))
                ),
                ConversationValidationTurn(
                    userPrompt = "Switch topics. The new project is called Orion; use Orion from now on.",
                    expectedFactGroups = listOf(setOf("orion"))
                ),
                ConversationValidationTurn(
                    userPrompt = "What is the current project name?",
                    expectedFactGroups = listOf(setOf("orion")),
                    forbiddenClaims = setOf("atlas"),
                    constraints = ConversationInstructionConstraints(maxWordCount = 10)
                )
            )
        ),
        ConversationValidationCase(
            id = "grounded-extraction-with-no-invention",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "Extract the owner and status only from this record: owner = Nadia; status = pending. Do not infer any other fields.",
                    expectedFactGroups = listOf(setOf("nadia"), setOf("pending")),
                    forbiddenClaims = setOf("approved", "completed", "rejected"),
                    constraints = ConversationInstructionConstraints(minWordCount = 2, maxWordCount = 18)
                )
            )
        ),
        ConversationValidationCase(
            id = "indonesian-instruction-format",
            turns = listOf(
                ConversationValidationTurn(
                    userPrompt = "Sebutkan tepat dua langkah untuk menyimpan dokumen. Gunakan daftar bernomor dan tanpa pengantar.",
                    expectedFactGroups = listOf(
                        setOf("simpan", "menyimpan"),
                        setOf("dokumen", "file")
                    ),
                    constraints = ConversationInstructionConstraints(
                        minWordCount = 5,
                        maxWordCount = 36,
                        exactNumberedItems = 2,
                        maxSentenceCount = 3
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
        val suiteFingerprint = ConversationValidationContract.fingerprint(cases, thresholds)

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
            turnResults = turnResults,
            suiteVersion = ConversationValidationSuite.VERSION,
            evaluatorVersion = ConversationValidationContract.EVALUATOR_VERSION,
            suiteFingerprint = suiteFingerprint,
            generationProfileId = ConversationValidationContract.GENERATION_PROFILE_ID
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
        if (constraints.requiredPhraseOrder.isNotEmpty()) {
            val paddedAnswer = " ${normalize(answer)} "
            val phrasePositions = constraints.requiredPhraseOrder.map { phrase ->
                paddedAnswer.indexOf(" ${normalize(phrase)} ")
            }
            checks += phrasePositions.all { it >= 0 } &&
                phrasePositions.zipWithNext().all { (first, second) -> first < second }
        }
        if (constraints.requireSingleLine) {
            checks += answer.lineSequence().count() <= 1
        }
        if (constraints.forbidMarkdown) {
            val markdownPattern = Regex("(?m)^\\s*(?:#{1,6}\\s|[-*+]\\s|\\d+[.)]\\s)|[`*_]{1,3}")
            checks += !markdownPattern.containsMatchIn(answer)
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
        if (!ConversationValidationContract.hasCompatibleContract(baseline, candidate)) {
            return rejected(
                baseline,
                candidate,
                "Baseline and candidate evaluation contracts do not match. Rebuild the baseline before checkpoint promotion."
            )
        }
        if (listOf(
                baseline.factCoverage,
                baseline.instructionCompliance,
                baseline.multiTurnConsistency,
                baseline.contradictionRate,
                baseline.overallScore,
                candidate.factCoverage,
                candidate.instructionCompliance,
                candidate.multiTurnConsistency,
                candidate.contradictionRate,
                candidate.overallScore
            ).any { !it.isFinite() || it !in 0f..1f }
        ) {
            return rejected(
                baseline,
                candidate,
                "Baseline or candidate contains an invalid conversational metric."
            )
        }
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
