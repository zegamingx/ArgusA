package com.argus.agi.model

import org.json.JSONException
import org.json.JSONObject
import java.io.BufferedReader
import java.io.Reader
import java.text.Normalizer
import java.util.Locale
import kotlin.coroutines.coroutineContext
import kotlinx.coroutines.ensureActive

internal data class GeneratedPolicyRollout(
    val text: String,
    val promptTokenIds: IntArray,
    val completionTokenIds: IntArray,
    val behaviorLogProbabilities: FloatArray
)
internal data class GrpoPolicyTrainingSample(
    val tokenIds: IntArray,
    val firstTargetIndex: Int,
    val behaviorLogProbabilities: FloatArray,
    val advantage: Float
)
internal data class GrpoOptimizerResult(
    val meanPolicyLoss: Float,
    val responseCount: Int,
    val targetTokenCount: Int,
    val optimizerUpdates: Int
)

data class GrpoTrainingConfig(
    val groupSize: Int = 4,
    val ppoEpochs: Int = 2,
    val maxNewTokens: Int = 32,
    val temperature: Float = 0.8f,
    val clipEpsilon: Float = 0.2f,
    val klCoefficient: Float = 0.02f,
    val maxExamples: Int = 10_000
) {
    init {
        require(groupSize in 2..8)
        require(ppoEpochs in 1..4)
        require(maxNewTokens in 1..128)
        require(temperature.isFinite() && temperature in 0.1f..2f)
        require(clipEpsilon.isFinite() && clipEpsilon in 0.01f..0.5f)
        require(klCoefficient.isFinite() && klCoefficient >= 0f)
        require(maxExamples in 1..50_000)
    }
}

data class GrpoTrainingExample(val prompt: String, val answer: String, val lineNumber: Int)
data class GrpoTrainingProgress(
    val exampleIndex: Int,
    val totalExamples: Int,
    val trainedGroups: Int,
    val skippedUniformRewardGroups: Int,
    val generatedResponses: Int,
    val meanReward: Float,
    val meanPolicyLoss: Float
)
data class GrpoTrainingReport(
    val recordCount: Int,
    val trainedGroupCount: Int,
    val skippedUniformRewardGroups: Int,
    val generatedResponses: Int,
    val exactMatchCount: Int,
    val optimizerUpdates: Int,
    val meanReward: Float,
    val meanPolicyLoss: Float,
    val effectiveGroupSize: Int = 4
)

internal data class GrpoRuntimeBudget(
    val groupSize: Int,
    val contextLength: Int,
    val maxNewTokens: Int
)

internal data class GrpoTokenObjectiveResult(
    val loss: Float,
    val derivativeWrtCurrentLogProbability: Float,
    val importanceRatio: Float,
    val klPenalty: Float,
    val isPolicyClipped: Boolean
)

/**
 * Clipped PPO surrogate plus the non-negative k3 estimate of KL(old || current),
 * sampled from the behavior policy: exp(log(current / old)) - log(current / old) - 1.
 */
internal fun calculateGrpoTokenObjective(
    oldLogProbability: Float,
    currentLogProbability: Float,
    advantage: Float,
    clipEpsilon: Float,
    klCoefficient: Float
): GrpoTokenObjectiveResult {
    require(oldLogProbability.isFinite() && currentLogProbability.isFinite())
    require(advantage.isFinite())
    require(clipEpsilon.isFinite() && clipEpsilon in 0.01f..0.5f)
    require(klCoefficient.isFinite() && klCoefficient >= 0f)
    val logRatio = (currentLogProbability - oldLogProbability).coerceIn(-20f, 20f)
    val ratio = kotlin.math.exp(logRatio)
    val clippedRatio = ratio.coerceIn(1f - clipEpsilon, 1f + clipEpsilon)
    val surrogate = minOf(ratio * advantage, clippedRatio * advantage)
    val isClipped = (advantage >= 0f && ratio > 1f + clipEpsilon) ||
        (advantage < 0f && ratio < 1f - clipEpsilon)
    val klPenalty = (ratio - logRatio - 1f).coerceAtLeast(0f)
    val policyDerivative = if (isClipped) 0f else -advantage * ratio
    return GrpoTokenObjectiveResult(
        loss = -surrogate + klCoefficient * klPenalty,
        derivativeWrtCurrentLogProbability = policyDerivative + klCoefficient * (ratio - 1f),
        importanceRatio = ratio,
        klPenalty = klPenalty,
        isPolicyClipped = isClipped
    )
}

internal fun computeGroupRelativeAdvantages(rewards: List<Float>): FloatArray? {
    require(rewards.size in 2..8)
    require(rewards.all { it.isFinite() && it in 0f..1f })
    val mean = rewards.average().toFloat()
    var variance = 0.0
    rewards.forEach { reward ->
        val delta = reward.toDouble() - mean.toDouble()
        variance += delta * delta
    }
    val standardDeviation = kotlin.math.sqrt(variance / rewards.size.toDouble()).toFloat()
    if (!standardDeviation.isFinite() || standardDeviation < 1.0e-6f) return null
    return FloatArray(rewards.size) { index ->
        (rewards[index] - mean) / (standardDeviation + 1.0e-6f)
    }
}

fun interface GrpoRewardEvaluator {
    fun evaluate(prompt: String, expectedAnswer: String, candidateAnswer: String): Float
}

/** Deterministic answer-key verifier: normalized exact match or Unicode-aware token F1. */
object ReferenceAnswerReward : GrpoRewardEvaluator {
    private val answerBlock = Regex("<answer>(.*?)</answer>", setOf(
        RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL
    ))
    private val tokenPattern = Regex("[\\p{L}\\p{N}]+")

    override fun evaluate(prompt: String, expectedAnswer: String, candidateAnswer: String): Float {
        val expected = tokens(expectedAnswer)
        val payload = answerBlock.find(candidateAnswer)?.groupValues?.get(1) ?: candidateAnswer
        val actual = tokens(payload)
        if (expected.isEmpty() || actual.isEmpty()) return 0f
        if (expected == actual) return 1f
        val remaining = expected.groupingBy { it }.eachCount().toMutableMap()
        var overlap = 0
        actual.forEach { token ->
            val count = remaining[token] ?: 0
            if (count > 0) {
                overlap++
                remaining[token] = count - 1
            }
        }
        if (overlap == 0) return 0f
        val precision = overlap.toFloat() / actual.size.toFloat()
        val recall = overlap.toFloat() / expected.size.toFloat()
        return if (precision + recall > 0f) {
            (2f * precision * recall / (precision + recall)).coerceIn(0f, 1f)
        } else 0f
    }

    fun isExactMatch(expectedAnswer: String, candidateAnswer: String): Boolean {
        val payload = answerBlock.find(candidateAnswer)?.groupValues?.get(1) ?: candidateAnswer
        val expected = tokens(expectedAnswer)
        return expected.isNotEmpty() && expected == tokens(payload)
    }

    private fun tokens(value: String): List<String> {
        val normalized = Normalizer.normalize(value, Normalizer.Form.NFKC).lowercase(Locale.ROOT)
        return tokenPattern.findAll(normalized).map { it.value }.toList()
    }
}

/**
 * Full local GRPO loop. Each answer-key record creates a same-prompt rollout group,
 * reward normalization produces per-response advantages, and the engine applies the
 * clipped policy objective plus a sampled reverse-KL penalty without a critic.
 */
class GrpoTrainer(
    private val neuralEngine: ArgusNeuralEngine,
    private val rewardEvaluator: GrpoRewardEvaluator = ReferenceAnswerReward
) {
    suspend fun train(
        reader: Reader,
        config: GrpoTrainingConfig = GrpoTrainingConfig(),
        onProgress: (GrpoTrainingProgress) -> Unit = {},
        resolveRuntimeBudget: suspend (GrpoTrainingConfig) -> GrpoRuntimeBudget = {
            GrpoRuntimeBudget(it.groupSize, neuralEngine.config.maxSeqLen, it.maxNewTokens)
        },
        beforeRollout: suspend (GrpoRuntimeBudget) -> Unit = {},
        beforeGroupUpdate: suspend (GrpoRuntimeBudget) -> Unit = {},
        runtimeSafetyCheck: (GrpoRuntimeBudget) -> Unit = {}
    ): GrpoTrainingReport {
        val examples = parseGrpoJsonl(reader, config.maxExamples)
        require(examples.isNotEmpty()) { "The GRPO JSONL dataset contains no records." }
        val runtimeBudget = resolveRuntimeBudget(config)
        require(runtimeBudget.groupSize in 2..config.groupSize)
        require(runtimeBudget.contextLength in 2..neuralEngine.config.maxSeqLen)
        require(runtimeBudget.maxNewTokens in 1..config.maxNewTokens

        var trainedGroups = 0
        var skippedGroups = 0
        var generatedResponses = 0
        var exactMatches = 0
        var optimizerUpdates = 0
        var rewardSum = 0.0
        var policyLossSum = 0.0
        var policyLossGroupCount = 0

        examples.forEachIndexed { index, example ->
            coroutineContext.ensureActive()
            val rollouts = ArrayList<GeneratedPolicyRollout>(runtimeBudget.groupSize)
            val rewards = ArrayList<Float>(runtimeBudget.groupSize)
            repeat(runtimeBudget.groupSize) {
                coroutineContext.ensureActive()
                beforeRollout(runtimeBudget)
                val rollout = neuralEngine.generatePolicyRollout(
                    prompt = example.prompt,
                    maxNewTokens = runtimeBudget.maxNewTokens,
                    temperature = config.temperature,
                    maxContextTokens = runtimeBudget.contextLength
                )
                if (
                    rollout.promptTokenIds.isNotEmpty() &&
                    rollout.completionTokenIds.isNotEmpty() &&
                    rollout.completionTokenIds.size == rollout.behaviorLogProbabilities.size
                ) {
                    val reward = rewardEvaluator.evaluate(example.prompt, example.answer, rollout.text)
                    require(reward.isFinite() && reward in 0f..1f) {
                        "GRPO reward evaluator must return a finite score in [0, 1]."
                    }
                    rollouts += rollout
                    rewards += reward
                    rewardSum += reward.toDouble()
                    generatedResponses++
                    if (
                        rewardEvaluator === ReferenceAnswerReward &&
                        ReferenceAnswerReward.isExactMatch(example.answer, rollout.text)
                    ) exactMatches++
                }
            }

            if (rollouts.size < 2) {
                skippedGroups++
            } else {
                val advantages = computeGroupRelativeAdvantages(rewards)
                if (advantages == null) {
                    skippedGroups++
                } else {
                    val samples = rollouts.mapIndexed { responseIndex, rollout ->
                        val sequence = IntArray(rollout.promptTokenIds.size + rollout.completionTokenIds.size)
                        System.arraycopy(rollout.promptTokenIds, 0, sequence, 0, rollout.promptTokenIds.size)
                        System.arraycopy(
                            rollout.completionTokenIds, 0, sequence, rollout.promptTokenIds.size,
                            rollout.completionTokenIds.size
                        )
                        GrpoPolicyTrainingSample(
                            tokenIds = sequence,
                            firstTargetIndex = rollout.promptTokenIds.lastIndex,
                            behaviorLogProbabilities = rollout.behaviorLogProbabilities,
                            advantage = advantages[responseIndex]
                        )
                    }
                    beforeGroupUpdate(runtimeBudget)
                    val update = neuralEngine.trainGrpoGroup(
                        samples = samples,
                        clipEpsilon = config.clipEpsilon,
                        klCoefficient = config.klCoefficient,
                        temperature = config.temperature,
                        ppoEpochs = config.ppoEpochs,
                        runtimeSafetyCheck = { runtimeSafetyCheck(runtimeBudget) }
                    )
                    if (update.optimizerUpdates > 0) {
                        trainedGroups++
                        optimizerUpdates += update.optimizerUpdates
                        if (update.meanPolicyLoss.isFinite()) {
                            policyLossSum += update.meanPolicyLoss.toDouble()
                            policyLossGroupCount++
                        }
                    } else {
                        skippedGroups++
                    }
                }
            }

            onProgress(
                GrpoTrainingProgress(
                    exampleIndex = index + 1,
                    totalExamples = examples.size,
                    trainedGroups = trainedGroups,
                    skippedUniformRewardGroups = skippedGroups,
                    generatedResponses = generatedResponses,
                    meanReward = if (generatedResponses > 0) {
                        (rewardSum / generatedResponses).toFloat()
                    } else Float.NaN,
                    meanPolicyLoss = if (policyLossGroupCount > 0) {
                        (policyLossSum / policyLossGroupCount).toFloat()
                    } else Float.NaN
                )
            )
        }

        return GrpoTrainingReport(
            recordCount = examples.size,
            trainedGroupCount = trainedGroups,
            skippedUniformRewardGroups = skippedGroups,
            generatedResponses = generatedResponses,
            exactMatchCount = exactMatches,
            optimizerUpdates = optimizerUpdates,
            meanReward = if (generatedResponses > 0) {
                (rewardSum / generatedResponses).toFloat()
            } else Float.NaN,
            meanPolicyLoss = if (policyLossGroupCount > 0) {
                (policyLossSum / policyLossGroupCount).toFloat()
            } else Float.NaN,
            effectiveGroupSize = runtimeBudget.groupSize
        )
    }

    companion object {
        private const val ADVANTAGE_EPSILON = 1.0e-6f
    }
}

internal fun parseGrpoJsonl(
    reader: Reader,
    maxExamples: Int = 10_000,
    maxTotalCharacters: Long = 16L * 1024L * 1024L
): List<GrpoTrainingExample> {
    require(maxExamples in 1..50_000)
    require(maxTotalCharacters > 0L)
    val buffered = if (reader is BufferedReader) reader else BufferedReader(reader)
    val examples = ArrayList<GrpoTrainingExample>()
    var lineNumber = 0
    var totalCharacters = 0L
    while (true) {
        val line = readBoundedGrpoLine(buffered, MAX_GRPO_JSONL_LINE_CHARS) ?: break
        lineNumber++
        totalCharacters = Math.addExact(totalCharacters, line.length.toLong())
        require(totalCharacters <= maxTotalCharacters) {
            "GRPO dataset exceeds the configured $maxTotalCharacters UTF-16 character limit."
        }
        if (line.isBlank()) continue
        try {
            val record = JSONObject(line)
            val prompt = record.opt("prompt") as? String
                ?: throw IllegalArgumentException("Missing string field 'prompt'.")
            val answer = record.opt("answer") as? String
                ?: throw IllegalArgumentException("Missing string field 'answer'.")
            require(prompt.isNotBlank()) { "Field 'prompt' cannot be blank." }
            require(answer.isNotBlank()) { "Field 'answer' cannot be blank." }
            examples += GrpoTrainingExample(prompt.trim(), answer.trim(), lineNumber)
            require(examples.size <= maxExamples) {
                "GRPO dataset exceeds configured maximum of ${maxExamples} examples."
            }
        } catch (error: JSONException) {
            throw IllegalArgumentException(
                "Invalid GRPO JSONL object at line ${lineNumber}: ${error.message}", error
            )
        } catch (error: IllegalArgumentException) {
            throw IllegalArgumentException(
                "Invalid GRPO JSONL row at line ${lineNumber}: ${error.message}", error
            )
        }
    }
    return examples
}

private const val MAX_GRPO_JSONL_LINE_CHARS = 256 * 1024
private fun readBoundedGrpoLine(reader: BufferedReader, maximumCharacters: Int): String? {
    val line = StringBuilder()
    var readAny = false
    while (true) {
        val character = reader.read()
        if (character < 0) return if (readAny) line.toString() else null
        readAny = true
        if (character == '\n'.code) return line.toString()
        if (character == '\r'.code) continue
        if (line.length >= maximumCharacters) {
            throw IllegalArgumentException(
                "A GRPO JSONL row exceeds the ${maximumCharacters} UTF-16 character limit."
            )
        }
        line.append(character.toChar())
    }
}
