package com.argus.agi.model

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt
import kotlin.random.Random

data class BenchmarkResult(
    val tokensPerSecond: Float,
    val latencyMsPerToken: Float,
    val memoryUsageMb: Float,
    val testTokensEvaluated: Int,
    val totalTimeMs: Long
)

data class EngineConfig(
    val dModel: Int = 128,
    val nHeads: Int = 4,
    val dFf: Int = 512,
    val maxSeqLen: Int = 128,
    val loraRank: Int = 4,
    val isFullTrainingEnabled: Boolean = true,
    val tokenizerConfig: TokenizerConfig = TokenizerConfig()
) {
    init {
        require(dModel > 0) { "dModel must be positive." }
        require(nHeads > 0 && dModel % nHeads == 0) {
            "nHeads must be positive and divide dModel evenly."
        }
        require(dFf > 0) { "dFf must be positive." }
        require(maxSeqLen >= 2) { "maxSeqLen must be at least 2." }
        require(loraRank > 0) { "loraRank must be positive." }
    }

    val architectureLabel: String
        get() = when {
            dModel >= 128 && dFf >= 512 ->
                "Flagship Dynamic Core"
            dModel >= 96 && dFf >= 384 ->
                "High Memory Neural Core"
            dModel >= 64 && dFf >= 256 ->
                "Balanced Neural Core"
            else ->
                "Compact Neural Core"
        } + " (d_model $dModel, d_ff $dFf, context $maxSeqLen, LoRA rank $loraRank)"

    companion object {
        /**
         * Selects a fresh model shape from stable device capacity, not volatile free RAM.
         * Existing checkpoint configurations are resolved separately before engine creation.
         */
        fun forDevice(
            totalRamGb: Int,
            maxHeapMb: Long,
            availableRamGb: Float = totalRamGb.toFloat()
        ): EngineConfig = when {
            totalRamGb >= 12 && maxHeapMb >= 512L && availableRamGb >= 3.0f ->
                EngineConfig(dModel = 128, nHeads = 4, dFf = 512, maxSeqLen = 128, loraRank = 4)

            totalRamGb >= 8 && maxHeapMb >= 384L && availableRamGb >= 2.0f ->
                EngineConfig(dModel = 96, nHeads = 4, dFf = 384, maxSeqLen = 128, loraRank = 4)

            totalRamGb >= 6 && maxHeapMb >= 256L && availableRamGb >= 2.0f ->
                EngineConfig(dModel = 64, nHeads = 4, dFf = 256, maxSeqLen = 96, loraRank = 2)

            else ->
                EngineConfig(dModel = 48, nHeads = 4, dFf = 128, maxSeqLen = 64, loraRank = 2)
        }
    }
}

/** Aggregate next-token loss and the exact number of scored validation targets. */
data class LossEvaluation(
    val meanLoss: Float,
    val targetCount: Int,
    val evaluatedDocumentCount: Int
)

internal data class GradientCheckResult(
    val parameterName: String,
    val parameterIndex: Int,
    val meanLoss: Float,
    val analyticalGradient: Float,
    val numericalGradient: Float,
    val absoluteError: Float,
    val relativeError: Float
)

/**
 * On-Device Neural Engine A.R.G.U.S:
 * Micro-Transformer SwiGLU with full-parameter backpropagation and LoRA adapters.
 * Uses sequential gradient accumulation and Adam without premature in-place mutation.
 */
class ArgusNeuralEngine private constructor(
    private val context: Context,
    val config: EngineConfig = EngineConfig()
) {

    val tokenizer = Tokenizer(config.tokenizerConfig)

    // =====================================================================
    // Dynamic Architecture Dimensions (Sesuai Hardware Tier Terpilih)
    // =====================================================================
    val dModel = config.dModel
    val nHeads = config.nHeads
    val headDim = dModel / nHeads
    val dFf = config.dFf
    val maxSeqLen = config.maxSeqLen
    val loraRank = config.loraRank
    val isFullTrainingEnabled = config.isFullTrainingEnabled

    val optimizerBackend: String
        get() = VulkanTrainingBackend.currentBackend
    val isVulkanRuntimeReady: Boolean
        get() = VulkanTrainingBackend.isRuntimeAvailable
    val checkpointInterval: Int
        get() = CHECKPOINT_INTERVAL
    private val scale = 1.0f / sqrt(headDim.toFloat())

    private var vocabCap = max(tokenizer.vocabSize + 32, 160)

    // Embeddings
    private var wEmbed = FloatArray(vocabCap * dModel)
    private val wPos = FloatArray(maxSeqLen * dModel)

    // Attention Projection Weights
    private val wQ = FloatArray(dModel * dModel)
    private val wK = FloatArray(dModel * dModel)
    private val wV = FloatArray(dModel * dModel)
    private val wO = FloatArray(dModel * dModel)

    // LoRA Adapter Matrices
    private val loraQa = FloatArray(dModel * loraRank)
    private val loraQb = FloatArray(loraRank * dModel)
    private val loraVa = FloatArray(dModel * loraRank)
    private val loraVb = FloatArray(loraRank * dModel)

    // SwiGLU MLP Weights
    private val wGate = FloatArray(dModel * dFf)
    private val wUp = FloatArray(dModel * dFf)
    private val wDown = FloatArray(dFf * dModel)

    // LM Output Head
    private var wHead = FloatArray(vocabCap * dModel)

    // =====================================================================
    // Adam Optimizer State (1st and 2nd Moments for Full Backprop)
    // =====================================================================
    private var mHead = FloatArray(vocabCap * dModel)
    private var vHead = FloatArray(vocabCap * dModel)
    private var mEmbed = FloatArray(vocabCap * dModel)
    private var vEmbed = FloatArray(vocabCap * dModel)

    // LoRA Moments
    private val mLoraQa = FloatArray(dModel * loraRank)
    private val vLoraQa = FloatArray(dModel * loraRank)
    private val mLoraQb = FloatArray(loraRank * dModel)
    private val vLoraQb = FloatArray(loraRank * dModel)
    private val mLoraVa = FloatArray(dModel * loraRank)
    private val vLoraVa = FloatArray(dModel * loraRank)
    private val mLoraVb = FloatArray(loraRank * dModel)
    private val vLoraVb = FloatArray(loraRank * dModel)

    // Full Transformer Layer Moments
    private val mWQ = FloatArray(dModel * dModel)
    private val vWQ = FloatArray(dModel * dModel)
    private val mWK = FloatArray(dModel * dModel)
    private val vWK = FloatArray(dModel * dModel)
    private val mWV = FloatArray(dModel * dModel)
    private val vWV = FloatArray(dModel * dModel)
    private val mWO = FloatArray(dModel * dModel)
    private val vWO = FloatArray(dModel * dModel)

    private val mWGate = FloatArray(dModel * dFf)
    private val vWGate = FloatArray(dModel * dFf)
    private val mWUp = FloatArray(dModel * dFf)
    private val vWUp = FloatArray(dModel * dFf)
    private val mWDown = FloatArray(dFf * dModel)
    private val vWDown = FloatArray(dFf * dModel)

    private var adamStep = 0
    private val weightDecay = 0.01f
    private val baseLearningRate = 0.005f
    private val beta1 = 0.9f
    private val beta2 = 0.999f
    private val epsilon = 1e-8f
    private val maxGradNorm = 1.0f

    // Reusable scratch buffer untuk attention weights
    private val attnWeightsCache: Array<Array<FloatArray>> =
        Array(nHeads) { Array(maxSeqLen) { FloatArray(maxSeqLen) } }

    private val modelLock = Any()

    private data class GradientProbe(
        val parameterName: String,
        var parameterIndex: Int,
        var analyticalGradient: Float = Float.NaN
    )

    companion object {
        private const val TAG = "ArgusNeuralEngine"
        private const val MAGIC_0 = 0x41.toByte()  // 'A'
        private const val MAGIC_1 = 0x52.toByte()  // 'R'
        private const val MAGIC_2 = 0x47.toByte()  // 'G'
        private const val MAGIC_3 = 0x33.toByte()  // '3'
        private const val HEADER_SIZE = 48
        private const val CHECKPOINT_VERSION = 3
        private const val CHECKPOINT_INTERVAL = 8

        /**
         * Preserve an existing ARG3 model's tensor dimensions when reopening the app.
         * Fresh installations use a deterministic RAM/heap tier instead.
         */
        fun resolveConfig(
            context: Context,
            totalRamGb: Int,
            maxHeapMb: Long,
            availableRamGb: Float = totalRamGb.toFloat()
        ): EngineConfig {
            val checkpoint = File(context.filesDir, "argus_model_weights.bin")
            if (checkpoint.isFile && checkpoint.length() >= HEADER_SIZE) {
                val savedConfig = runCatching {
                    FileInputStream(checkpoint).use { input ->
                        val bytes = ByteArray(HEADER_SIZE)
                        var offset = 0
                        while (offset < bytes.size) {
                            val count = input.read(bytes, offset, bytes.size - offset)
                            if (count < 0) error("Incomplete ARG3 checkpoint header.")
                            offset += count
                        }

                        val header = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                        require(
                            header.get() == MAGIC_0 &&
                                header.get() == MAGIC_1 &&
                                header.get() == MAGIC_2 &&
                                header.get() == MAGIC_3
                        ) { "Unsupported checkpoint magic." }
                        require(header.int == CHECKPOINT_VERSION) {
                            "Unsupported checkpoint version."
                        }

                        val savedDModel = header.int
                        val savedNHeads = header.int
                        val savedDFf = header.int
                        val savedMaxSeq = header.int
                        val savedLoraRank = header.int
                        val savedVocabSize = header.int
                        val savedVocabCap = header.int
                        header.int // saved optimizer step
                        header.int // reserved field

                        require(savedVocabSize > 0 && savedVocabCap >= savedVocabSize) {
                            "Invalid checkpoint vocabulary dimensions."
                        }

                        EngineConfig(
                            dModel = savedDModel,
                            nHeads = savedNHeads,
                            dFf = savedDFf,
                            maxSeqLen = savedMaxSeq,
                            loraRank = savedLoraRank,
                            isFullTrainingEnabled = true
                        )
                    }
                }.getOrNull()

                if (savedConfig != null) return savedConfig
            }

            return EngineConfig.forDevice(totalRamGb, maxHeapMb, availableRamGb)
        }

        @JvmStatic
        fun createBlocking(context: Context, config: EngineConfig = EngineConfig()): ArgusNeuralEngine =
            ArgusNeuralEngine(context.applicationContext, config).also { engine ->
                engine.loadCheckpointIfExists()
            }

        suspend fun create(context: Context, config: EngineConfig = EngineConfig()): ArgusNeuralEngine =
            withContext(Dispatchers.IO) {
                createBlocking(context, config)
            }
    }

    init {
        initializeWeights()
    }

    // =====================================================================
    // Weight Initialization (Xavier / He Normal & Sinusoids)
    // =====================================================================
    private fun initializeWeights() {
        val stdEmbed = 1.0f / sqrt(dModel.toFloat())
        for (i in wEmbed.indices) wEmbed[i] = (Random.nextFloat() * 2f - 1f) * stdEmbed

        for (pos in 0 until maxSeqLen) {
            for (i in 0 until dModel) {
                val freq = 1.0 / Math.pow(10000.0, (2 * (i / 2)).toDouble() / dModel)
                val raw = if (i % 2 == 0) Math.sin(pos * freq).toFloat()
                else Math.cos(pos * freq).toFloat()
                wPos[pos * dModel + i] = raw * 0.2f
            }
        }

        val stdAttn = 1.0f / sqrt(dModel.toFloat())
        initFlatMatrix(wQ, dModel, dModel, stdAttn)
        initFlatMatrix(wK, dModel, dModel, stdAttn)
        initFlatMatrix(wV, dModel, dModel, stdAttn)
        initFlatMatrix(wO, dModel, dModel, stdAttn)

        val stdLora = 1.0f / sqrt(loraRank.toFloat())
        initFlatMatrix(loraQa, dModel, loraRank, stdLora)
        loraQb.fill(0f)
        initFlatMatrix(loraVa, dModel, loraRank, stdLora)
        loraVb.fill(0f)

        val stdMlp = 1.0f / sqrt(dModel.toFloat())
        initFlatMatrix(wGate, dModel, dFf, stdMlp)
        initFlatMatrix(wUp, dModel, dFf, stdMlp)
        initFlatMatrix(wDown, dFf, dModel, 1.0f / sqrt(dFf.toFloat()))

        val stdHead = 1.0f / sqrt(dModel.toFloat())
        for (i in wHead.indices) wHead[i] = (Random.nextFloat() * 2f - 1f) * stdHead
    }

    private fun initFlatMatrix(mat: FloatArray, inDim: Int, outDim: Int, std: Float) {
        val size = inDim * outDim
        for (i in 0 until size) {
            mat[i] = (Random.nextFloat() * 2f - 1f) * std
        }
    }

    /**
     * Architecture metadata exposed by the active legacy engine.
     * The current execution graph is a single Transformer block.
     */
    val architectureConfig: ArgusArchitectureConfig
        get() = config.toArgusArchitectureConfig(tokenizer.vocabSize)

    val parameterBreakdown: ArgusParameterBreakdown
        get() = ArgusArchitectureParameterEstimator.estimate(architectureConfig)

    /**
     * Preserves the historical resident-tensor element count, including the
     * fixed sinusoidal position table. Use countTrainableParameters() when a
     * strict trainable-weight count is required.
     */
    fun countTotalParameters(): Long =
        parameterBreakdown.totalResidentModelElements

    fun countTrainableParameters(): Long =
        parameterBreakdown.totalTrainableParameters

    fun calculateAdapterSizeBytes(): Long =
        ((dModel * loraRank + loraRank * dModel) * 2).toLong() * 4L

    // =====================================================================
    // Vocab Capacity Management
    // =====================================================================
    private fun ensureVocabCapacity(requiredSize: Int) {
        if (requiredSize <= vocabCap) return
        val newCap = max(requiredSize + 64, vocabCap * 2)
        resizeVocabCapacityTo(newCap)
    }

    private fun resizeVocabCapacityTo(targetCap: Int) {
        if (targetCap <= 0 || targetCap == vocabCap) return
        val newEmbed = FloatArray(targetCap * dModel)
        val newHead = FloatArray(targetCap * dModel)
        val newMEmbed = FloatArray(targetCap * dModel)
        val newVEmbed = FloatArray(targetCap * dModel)
        val newMHead = FloatArray(targetCap * dModel)
        val newVHead = FloatArray(targetCap * dModel)

        System.arraycopy(wEmbed, 0, newEmbed, 0, min(wEmbed.size, newEmbed.size))
        System.arraycopy(wHead, 0, newHead, 0, min(wHead.size, newHead.size))
        System.arraycopy(mEmbed, 0, newMEmbed, 0, min(mEmbed.size, newMEmbed.size))
        System.arraycopy(vEmbed, 0, newVEmbed, 0, min(vEmbed.size, newVEmbed.size))
        System.arraycopy(mHead, 0, newMHead, 0, min(mHead.size, newMHead.size))
        System.arraycopy(vHead, 0, newVHead, 0, min(vHead.size, newVHead.size))

        if (targetCap > vocabCap) {
            val stdEmbed = 1.0f / sqrt(dModel.toFloat())
            for (i in (vocabCap * dModel) until (targetCap * dModel)) {
                newEmbed[i] = (Random.nextFloat() * 2f - 1f) * stdEmbed
                newHead[i] = (Random.nextFloat() * 2f - 1f) * stdEmbed
            }
        }

        wEmbed = newEmbed
        wHead = newHead
        mEmbed = newMEmbed
        vEmbed = newVEmbed
        mHead = newMHead
        vHead = newVHead
        vocabCap = targetCap
    }

    // =====================================================================
    // Generation (Deterministic inference with frozen vocabulary)
    // =====================================================================
    suspend fun generate(
        prompt: String,
        maxNewTokens: Int = 60,
        temperature: Float = 0.7f
    ): String = withContext(Dispatchers.Default) {
        synchronized(modelLock) {
            val formattedPrompt = if (prompt.contains(Tokenizer.USER_TOKEN)) {
                if (prompt.endsWith(Tokenizer.ASSISTANT_TOKEN)) prompt
                else "$prompt ${Tokenizer.ASSISTANT_TOKEN}"
            } else {
                "${Tokenizer.USER_TOKEN}$prompt${Tokenizer.ASSISTANT_TOKEN}"
            }

            // PENTING: autoRegister = false saat inference agar kamus token stabil
            val rawTokens = tokenizer.encode(formattedPrompt, autoRegister = false)
            if (rawTokens.isEmpty()) return@withContext "Input kosong."

            val currentVocab = tokenizer.vocabSize
            val maxPromptLen = maxSeqLen - 1
            val effective = if (rawTokens.size > maxPromptLen) rawTokens.takeLast(maxPromptLen)
            else rawTokens
            val currentTokens = effective.toMutableList()
            val generatedTokens = mutableListOf<Int>()

            val kCache = FloatArray(maxSeqLen * dModel)
            val vCache = FloatArray(maxSeqLen * dModel)

            val initialLen = currentTokens.size
            val x = FloatArray(dModel)
            val q = FloatArray(dModel)
            val k = FloatArray(dModel)
            val v = FloatArray(dModel)
            val loraMid = FloatArray(loraRank)
            val attnOut = FloatArray(dModel)
            val h1 = FloatArray(dModel)
            val gate = FloatArray(dFf)
            val up = FloatArray(dFf)
            val act = FloatArray(dFf)
            val down = FloatArray(dModel)
            val h2 = FloatArray(dModel)
            val logits = FloatArray(currentVocab)

            val embedScale = sqrt(dModel.toFloat())

            // Prefill: K,V untuk semua posisi prompt
            for (pos in 0 until initialLen) {
                computeTokenEmbedAndPos(currentTokens[pos], pos, embedScale, x)
                computeLinear(x, wK, dModel, dModel, k)
                System.arraycopy(k, 0, kCache, pos * dModel, dModel)
                computeLinearWithLora(x, wV, loraVa, loraVb, dModel, dModel, loraRank, loraMid, v)
                System.arraycopy(v, 0, vCache, pos * dModel, dModel)
            }

            // Decode loop
            var currentPos = initialLen - 1
            for (step in 0 until maxNewTokens) {
                if (currentPos >= maxSeqLen - 1) break

                val lastTok = currentTokens[currentPos]
                computeTokenEmbedAndPos(lastTok, currentPos, embedScale, x)

                computeLinearWithLora(x, wQ, loraQa, loraQb, dModel, dModel, loraRank, loraMid, q)

                if (step > 0) {
                    computeLinear(x, wK, dModel, dModel, k)
                    System.arraycopy(k, 0, kCache, currentPos * dModel, dModel)
                    computeLinearWithLora(x, wV, loraVa, loraVb, dModel, dModel, loraRank, loraMid, v)
                    System.arraycopy(v, 0, vCache, currentPos * dModel, dModel)
                }

                attnOut.fill(0f)
                val seqSoFar = currentPos + 1
                val scores = FloatArray(seqSoFar)

                for (h in 0 until nHeads) {
                    val hOffset = h * headDim
                    var maxScore = -Float.MAX_VALUE
                    for (j in 0 until seqSoFar) {
                        var dot = 0f
                        val kOffset = j * dModel + hOffset
                        for (d in 0 until headDim) dot += q[hOffset + d] * kCache[kOffset + d]
                        val sc = dot * scale
                        scores[j] = sc
                        if (sc > maxScore) maxScore = sc
                    }
                    var sumExp = 0f
                    for (j in 0 until seqSoFar) {
                        val e = exp(scores[j] - maxScore)
                        scores[j] = e; sumExp += e
                    }
                    val invSum = 1.0f / max(sumExp, 1e-9f)
                    for (j in 0 until seqSoFar) scores[j] *= invSum

                    for (d in 0 until headDim) {
                        var weightedV = 0f
                        for (j in 0 until seqSoFar) {
                            val vOffset = j * dModel + hOffset
                            weightedV += scores[j] * vCache[vOffset + d]
                        }
                        attnOut[hOffset + d] = weightedV
                    }
                }

                val projBuf = FloatArray(dModel)
                computeLinear(attnOut, wO, dModel, dModel, projBuf)
                for (d in 0 until dModel) h1[d] = x[d] + projBuf[d]
                layerNormInPlace(h1, dModel)

                computeLinear(h1, wGate, dModel, dFf, gate)
                computeLinear(h1, wUp, dModel, dFf, up)
                for (d in 0 until dFf) {
                    val sig = 1.0f / (1.0f + exp(-gate[d]))
                    act[d] = (gate[d] * sig) * up[d]
                }
                computeLinear(act, wDown, dFf, dModel, down)
                for (d in 0 until dModel) h2[d] = h1[d] + down[d]
                layerNormInPlace(h2, dModel)

                for (vIdx in 0 until currentVocab) {
                    var sum = 0f
                    val wOff = vIdx * dModel
                    for (d in 0 until dModel) sum += h2[d] * wHead[wOff + d]
                    logits[vIdx] = sum
                }

                val nextToken = sampleLogits(logits, currentVocab, temperature)
                if (nextToken == Tokenizer.EOS_ID) break

                generatedTokens.add(nextToken)
                currentTokens.add(nextToken)
                currentPos++
            }

            val decoded = tokenizer.decode(generatedTokens)
            if (decoded.isBlank()) "..." else decoded
        }
    }

    // =====================================================================
    // Training: Full-Parameter Backprop with Per-Window Gradient Accumulation
    // =====================================================================
    /**
     * Computes next-token cross-entropy without changing weights, optimizer state,
     * tokenizer vocabulary, or checkpoint files. Use only on held-out validation text.
     */
    fun evaluateLoss(text: String): Float = evaluateLossWithTargetCount(text).meanLoss

    /**
     * Computes held-out next-token cross-entropy and the exact number of scored targets.
     * No parameters, optimizer state, tokenizer vocabulary, or checkpoint files are changed.
     */
    fun evaluateLossWithTargetCount(text: String): LossEvaluation = synchronized(modelLock) {
        val tokens = tokenizer.encode(text, autoRegister = false)
        if (tokens.size < 2 || tokenizer.vocabSize > vocabCap) {
            return@synchronized LossEvaluation(Float.NaN, 0, 0)
        }

        val currentVocab = tokenizer.vocabSize
        if (tokens.size <= maxSeqLen) {
            val loss = trainSingleWindow(
                tokens,
                currentVocab,
                updateParameters = false
            )
            return@synchronized LossEvaluation(
                meanLoss = loss,
                targetCount = if (loss.isFinite()) tokens.size - 1 else 0,
                evaluatedDocumentCount = if (loss.isFinite()) 1 else 0
            )
        }

        val overlapTokens = min(24, maxSeqLen - 1)
        val stride = (maxSeqLen - overlapTokens).coerceAtLeast(1)
        var weightedLoss = 0f
        var evaluatedTargets = 0
        var start = 0

        while (start < tokens.size - 1) {
            val end = min(start + maxSeqLen, tokens.size)
            val windowTokens = tokens.subList(start, end)
            if (windowTokens.size >= 2) {
                val firstTargetIndex = if (start == 0 || overlapTokens == 0) 0 else overlapTokens - 1
                val targetCount = (windowTokens.size - 1 - firstTargetIndex).coerceAtLeast(0)
                if (targetCount > 0) {
                    val windowLoss = trainSingleWindow(
                        seq = windowTokens,
                        currentVocab = currentVocab,
                        updateParameters = false,
                        firstTargetIndex = firstTargetIndex
                    )
                    if (windowLoss.isFinite()) {
                        weightedLoss += windowLoss * targetCount
                        evaluatedTargets += targetCount
                    }
                }
            }
            if (end >= tokens.size) break
            start += stride
        }

        val meanLoss = if (evaluatedTargets > 0) {
            weightedLoss / evaluatedTargets
        } else {
            Float.NaN
        }
        LossEvaluation(
            meanLoss = meanLoss,
            targetCount = evaluatedTargets,
            evaluatedDocumentCount = if (evaluatedTargets > 0) 1 else 0
        )
    }

    /**
     * Aggregates document losses by scored next-token targets, not by document count.
     * The model lock prevents training from changing weights between validation documents.
     */
    fun evaluateCorpusLoss(texts: List<String>): LossEvaluation = synchronized(modelLock) {
        var weightedLoss = 0.0
        var targetCount = 0
        var evaluatedDocumentCount = 0
        texts.forEach { text ->
            val result = evaluateLossWithTargetCount(text)
            if (result.targetCount > 0 && result.meanLoss.isFinite()) {
                weightedLoss += result.meanLoss.toDouble() * result.targetCount
                targetCount += result.targetCount
                evaluatedDocumentCount += result.evaluatedDocumentCount
            }
        }
        val meanLoss = if (targetCount > 0) {
            (weightedLoss / targetCount).toFloat()
        } else {
            Float.NaN
        }
        LossEvaluation(meanLoss, targetCount, evaluatedDocumentCount)
    }

    /**
     * Returns exp(crossEntropy). Use the same fixed held-out corpus between evaluations.
     */
    fun evaluatePerplexity(text: String): Float {
        val loss = evaluateLoss(text)
        return if (loss.isFinite()) exp(loss) else Float.NaN
    }

    /**
     * Compares one analytic backprop gradient with a central finite-difference estimate.
     * This diagnostic is internal to the module, only supports short single-window inputs,
     * and never applies Adam, changes the tokenizer, or writes a checkpoint. Pass -1 as the
     * parameter index to inspect the coordinate with the largest absolute analytic gradient.
     */
    internal fun checkGradient(
        text: String,
        parameterName: String,
        parameterIndex: Int,
        epsilon: Float = 1e-2f
    ): GradientCheckResult = synchronized(modelLock) {
        require(epsilon.isFinite() && epsilon > 0f) {
            "Finite-difference epsilon must be finite and positive."
        }
        val tokens = tokenizer.encode(text, autoRegister = false)
        require(tokens.size in 2..maxSeqLen) {
            "Gradient checking requires 2..${maxSeqLen} encoded tokens; got ${tokens.size}."
        }

        val parameter = gradientParameterArray(parameterName)
        require(parameterIndex == -1 || parameterIndex in parameter.indices) {
            "Parameter index must be -1 (largest-magnitude gradient) or a valid index in ${parameterName}."
        }

        val probe = GradientProbe(parameterName, parameterIndex)
        val currentVocab = tokenizer.vocabSize
        val baseLoss = trainSingleWindow(
            seq = tokens,
            currentVocab = currentVocab,
            updateParameters = false,
            gradientProbe = probe
        )
        check(baseLoss.isFinite() && probe.analyticalGradient.isFinite()) {
            "Analytic gradient computation returned a non-finite value."
        }

        val resolvedIndex = probe.parameterIndex
        val originalValue = parameter[resolvedIndex]
        var plusLoss = Float.NaN
        var minusLoss = Float.NaN
        try {
            parameter[resolvedIndex] = originalValue + epsilon
            plusLoss = trainSingleWindow(tokens, currentVocab, updateParameters = false)

            parameter[resolvedIndex] = originalValue - epsilon
            minusLoss = trainSingleWindow(tokens, currentVocab, updateParameters = false)
        } finally {
            parameter[resolvedIndex] = originalValue
        }

        check(plusLoss.isFinite() && minusLoss.isFinite()) {
            "Finite-difference loss evaluation returned a non-finite value."
        }
        val numericalGradient = (plusLoss - minusLoss) / (2f * epsilon)
        val absoluteError = kotlin.math.abs(probe.analyticalGradient - numericalGradient)
        val magnitude = max(
            kotlin.math.abs(probe.analyticalGradient),
            kotlin.math.abs(numericalGradient)
        )
        val relativeError = if (magnitude > 1e-8f) {
            absoluteError / magnitude
        } else {
            absoluteError
        }

        GradientCheckResult(
            parameterName = parameterName,
            parameterIndex = resolvedIndex,
            meanLoss = baseLoss,
            analyticalGradient = probe.analyticalGradient,
            numericalGradient = numericalGradient,
            absoluteError = absoluteError,
            relativeError = relativeError
        )
    }

    private fun gradientParameterArray(parameterName: String): FloatArray = when (parameterName) {
        "wHead" -> wHead
        "wEmbed" -> wEmbed
        "wDown" -> wDown
        "wUp" -> wUp
        "wGate" -> wGate
        "wO" -> wO
        "wQ" -> wQ
        "wK" -> wK
        "wV" -> wV
        "loraQa" -> loraQa
        "loraQb" -> loraQb
        "loraVa" -> loraVa
        "loraVb" -> loraVb
        else -> throw IllegalArgumentException("Unsupported gradient-check parameter: $parameterName")
    }

    fun trainStep(text: String): Float {
        var pendingSnapshot: ByteArray? = null
        val loss = synchronized(modelLock) {
            // Keep token IDs stable while training; unknown Unicode uses the tokenizer's deterministic Unicode escape.
            val tokens = tokenizer.encode(text, autoRegister = false)
            if (tokens.size < 2) return@synchronized Float.NaN

            ensureVocabCapacity(tokenizer.vocabSize)
            val currentVocab = tokenizer.vocabSize

            val result = if (tokens.size <= maxSeqLen) {
                trainSingleWindow(tokens, currentVocab)
            } else {
                val overlapTokens = min(24, maxSeqLen - 1)
                val stride = (maxSeqLen - overlapTokens).coerceAtLeast(1)
                var weightedLoss = 0f
                var trainedTargets = 0
                var start = 0

                while (start < tokens.size - 1) {
                    val end = min(start + maxSeqLen, tokens.size)
                    val windowSeq = tokens.subList(start, end)
                    if (windowSeq.size >= 2) {
                        val firstTargetIndex = if (start == 0 || overlapTokens == 0) 0 else overlapTokens - 1
                        val targetCount = (windowSeq.size - 1 - firstTargetIndex).coerceAtLeast(0)
                        if (targetCount > 0) {
                            val windowLoss = trainSingleWindow(
                                seq = windowSeq,
                                currentVocab = currentVocab,
                                firstTargetIndex = firstTargetIndex
                            )
                            if (windowLoss.isFinite()) {
                                weightedLoss += windowLoss * targetCount
                                trainedTargets += targetCount
                            }
                        }
                    }
                    if (end >= tokens.size) break
                    start += stride
                }

                if (trainedTargets > 0) weightedLoss / trainedTargets else Float.NaN
            }

            if (adamStep % CHECKPOINT_INTERVAL == 0) pendingSnapshot = buildCheckpointBytes()
            result
        }

        pendingSnapshot?.let { bytes ->
            runCatching { writeBytesToDisk(bytes) }
                .onFailure { Log.e(TAG, "Auto-checkpoint save failed", it) }
        }
        return loss
    }

    private fun computeCrossEntropyLoss(
        logits: Array<FloatArray>,
        sequence: List<Int>,
        currentVocab: Int,
        firstTargetIndex: Int
    ): Float {
        require(sequence.size >= 2) { "A validation window must contain at least two tokens." }
        require(firstTargetIndex in 0 until sequence.size - 1) {
            "firstTargetIndex must identify at least one next-token target."
        }

        var totalLoss = 0f
        var validTargets = 0

        for (timeIndex in firstTargetIndex until sequence.size - 1) {
            val target = sequence[timeIndex + 1]
            if (target !in 0 until currentVocab) continue

            val row = logits[timeIndex]
            var maxLogit = -Float.MAX_VALUE
            for (vocabularyIndex in 0 until currentVocab) {
                if (row[vocabularyIndex] > maxLogit) {
                    maxLogit = row[vocabularyIndex]
                }
            }

            var sumExp = 0f
            for (vocabularyIndex in 0 until currentVocab) {
                sumExp += exp(row[vocabularyIndex] - maxLogit)
            }

            val logProbability =
                (row[target] - maxLogit) - ln(max(sumExp, 1e-9f))
            val loss = -logProbability
            if (loss.isFinite()) {
                totalLoss += loss
                validTargets++
            }
        }

        return if (validTargets > 0) totalLoss / validTargets else Float.NaN
    }

    private fun trainSingleWindow(
        seq: List<Int>,
        currentVocab: Int,
        updateParameters: Boolean = true,
        firstTargetIndex: Int = 0,
        gradientProbe: GradientProbe? = null
    ): Float {
        val seqLen = seq.size
        if (seqLen < 2) return Float.NaN

        if (updateParameters) adamStep++
        val lr = if (updateParameters) {
            baseLearningRate / (1f + 0.0002f * adamStep)
        } else {
            0f
        }
        val embedScale = sqrt(dModel.toFloat())

        // ============================= FORWARD =============================
        val x = Array(seqLen) { FloatArray(dModel) }
        val q = Array(seqLen) { FloatArray(dModel) }
        val k = Array(seqLen) { FloatArray(dModel) }
        val v = Array(seqLen) { FloatArray(dModel) }
        val mQ = Array(seqLen) { FloatArray(loraRank) }
        val mV = Array(seqLen) { FloatArray(loraRank) }
        val attnOut = Array(seqLen) { FloatArray(dModel) }
        val h1PreLN = Array(seqLen) { FloatArray(dModel) }
        val h1 = Array(seqLen) { FloatArray(dModel) }
        val gate = Array(seqLen) { FloatArray(dFf) }
        val up = Array(seqLen) { FloatArray(dFf) }
        val act = Array(seqLen) { FloatArray(dFf) }
        val h2PreLN = Array(seqLen) { FloatArray(dModel) }
        val h2 = Array(seqLen) { FloatArray(dModel) }
        val logits = Array(seqLen) { FloatArray(currentVocab) }

        for (t in 0 until seqLen) {
            computeTokenEmbedAndPos(seq[t], t, embedScale, x[t])
        }

        computeBatchedLinear(
            inputs = arrayOf(x, x, x),
            weights = arrayOf(wQ, wK, wV),
            outputs = arrayOf(q, k, v),
            inputColumns = dModel,
            outputColumns = dModel
        )

        for (t in 0 until seqLen) {
            addLoraProjection(x[t], loraQa, loraQb, dModel, dModel, loraRank, mQ[t], q[t])
            addLoraProjection(x[t], loraVa, loraVb, dModel, dModel, loraRank, mV[t], v[t])
        }

        val attnWeights = attnWeightsCache
        for (h in 0 until nHeads) {
            val hOffset = h * headDim
            for (i in 0 until seqLen) {
                var maxScore = -Float.MAX_VALUE
                for (j in 0..i) {
                    var dot = 0f
                    for (d in 0 until headDim) dot += q[i][hOffset + d] * k[j][hOffset + d]
                    val sc = dot * scale
                    attnWeights[h][i][j] = sc
                    if (sc > maxScore) maxScore = sc
                }
                var sumExp = 0f
                for (j in 0..i) {
                    val e = exp(attnWeights[h][i][j] - maxScore)
                    attnWeights[h][i][j] = e; sumExp += e
                }
                val invSum = 1.0f / max(sumExp, 1e-9f)
                for (j in 0..i) attnWeights[h][i][j] *= invSum
                for (d in 0 until headDim) {
                    var weightedV = 0f
                    for (j in 0..i) weightedV += attnWeights[h][i][j] * v[j][hOffset + d]
                    attnOut[i][hOffset + d] = weightedV
                }
            }
        }

        computeLinearBatch(attnOut, wO, dModel, dModel, h1PreLN)
        for (t in 0 until seqLen) {
            for (d in 0 until dModel) h1PreLN[t][d] += x[t][d]
            System.arraycopy(h1PreLN[t], 0, h1[t], 0, dModel)
            layerNormInPlace(h1[t], dModel)
        }

        computeLinearBatch(h1, wGate, dModel, dFf, gate)
        computeLinearBatch(h1, wUp, dModel, dFf, up)
        for (t in 0 until seqLen) {
            for (d in 0 until dFf) {
                val sig = 1.0f / (1.0f + exp(-gate[t][d]))
                act[t][d] = (gate[t][d] * sig) * up[t][d]
            }
        }

        computeLinearBatch(act, wDown, dFf, dModel, h2PreLN)
        for (t in 0 until seqLen) {
            for (d in 0 until dModel) h2PreLN[t][d] += h1[t][d]
            System.arraycopy(h2PreLN[t], 0, h2[t], 0, dModel)
            layerNormInPlace(h2[t], dModel)
        }

        computeLinearBatch(
            inputs = h2,
            weights = wHead,
            inputColumns = dModel,
            outputColumns = currentVocab,
            outputs = logits,
            transposeWeights = true
        )

        // ============================ BACKWARD ============================
        // Loss-only evaluation exits before allocating gradient or backpropagation buffers.
        if (!updateParameters && gradientProbe == null) {
            return computeCrossEntropyLoss(
                logits = logits,
                sequence = seq,
                currentVocab = currentVocab,
                firstTargetIndex = firstTargetIndex
            )
        }

        var totalLoss = 0f
        var validSteps = 0
        require(firstTargetIndex in 0 until seqLen - 1) {
            "firstTargetIndex must identify at least one next-token target."
        }

        // Backpropagation buffers are allocated only for optimizer updates or explicit diagnostics.
        val gradWHead = FloatArray(currentVocab * dModel)
        val gradWEmbed = FloatArray(vocabCap * dModel)
        val gradWDown = FloatArray(dFf * dModel)
        val gradWUp = FloatArray(dModel * dFf)
        val gradWGate = FloatArray(dModel * dFf)
        val gradWO = FloatArray(dModel * dModel)
        val gradWQ = FloatArray(dModel * dModel)
        val gradWK = FloatArray(dModel * dModel)
        val gradWV = FloatArray(dModel * dModel)

        val gradLoraQa = FloatArray(dModel * loraRank)
        val gradLoraQb = FloatArray(loraRank * dModel)
        val gradLoraVa = FloatArray(dModel * loraRank)
        val gradLoraVb = FloatArray(loraRank * dModel)

        val dLogits = Array(seqLen) { FloatArray(currentVocab) }
        val dL_dh2PostLN = Array(seqLen) { FloatArray(dModel) }
        val dL_dh2PreLN = Array(seqLen) { FloatArray(dModel) }
        val dL_dh1PostLN = Array(seqLen) { FloatArray(dModel) }
        val dL_dh1PreLN = Array(seqLen) { FloatArray(dModel) }
        val dL_dattnOut = Array(seqLen) { FloatArray(dModel) }
        val dQ = Array(seqLen) { FloatArray(dModel) }
        val dK = Array(seqLen) { FloatArray(dModel) }
        val dV = Array(seqLen) { FloatArray(dModel) }

        // (1) Cross-Entropy & Gradient Accumulation for the Output Head.
        for (t in firstTargetIndex until seqLen - 1) {
            val target = seq[t + 1]
            if (target !in 0 until currentVocab) continue

            val logitRow = logits[t]
            var maxLogit = -Float.MAX_VALUE
            for (vocabIndex in 0 until currentVocab) {
                if (logitRow[vocabIndex] > maxLogit) maxLogit = logitRow[vocabIndex]
            }

            var sumExp = 0f
            for (vocabIndex in 0 until currentVocab) {
                sumExp += exp(logitRow[vocabIndex] - maxLogit)
            }

            val logProbability =
                (logitRow[target] - maxLogit) - ln(max(sumExp, 1e-9f))
            val loss = -logProbability
            if (loss.isFinite()) {
                totalLoss += loss
                validSteps++
            }

            val inverseSumExp = 1f / max(sumExp, 1e-9f)
            for (vocabIndex in 0 until currentVocab) {
                val probability = exp(logitRow[vocabIndex] - maxLogit) * inverseSumExp
                val gradient = if (vocabIndex == target) probability - 1f else probability
                dLogits[t][vocabIndex] = gradient
            }
        }

        // Output-head backward is two matrix products:
        // dHidden = dLogits × WHead and dWHead = transpose(dLogits) × hidden.
        computeLinearBatch(
            inputs = dLogits,
            weights = wHead,
            inputColumns = currentVocab,
            outputColumns = dModel,
            outputs = dL_dh2PostLN
        )

        val dLogitsByVocab = Array(currentVocab) { vocabIndex ->
            FloatArray(seqLen) { timeIndex -> dLogits[timeIndex][vocabIndex] }
        }
        val hiddenRows = flattenRows(h2)
        val gradWHeadRows = Array(currentVocab) { FloatArray(dModel) }
        computeLinearBatch(
            inputs = dLogitsByVocab,
            weights = hiddenRows,
            inputColumns = seqLen,
            outputColumns = dModel,
            outputs = gradWHeadRows
        )
        for (vocabIndex in 0 until currentVocab) {
            System.arraycopy(
                gradWHeadRows[vocabIndex],
                0,
                gradWHead,
                vocabIndex * dModel,
                dModel
            )
        }

        // (2) SwiGLU MLP & Residual Backward
        // The final sequence row has no next-token target, so its backward buffers
        // remain zero. Keeping it in the batch preserves the original t < seqLen - 1
        // gradient-accumulation boundary.
        for (t in 0 until seqLen - 1) {
            layerNormBackward(dL_dh2PostLN[t], h2PreLN[t], dL_dh2PreLN[t], dModel)
        }

        // dAct = dL_dh2PreLN × transpose(wDown).
        val dActRows = Array(seqLen) { FloatArray(dFf) }
        computeLinearBatch(
            inputs = dL_dh2PreLN,
            weights = wDown,
            inputColumns = dModel,
            outputColumns = dFf,
            outputs = dActRows,
            transposeWeights = true
        )

        val dUpRows = Array(seqLen) { FloatArray(dFf) }
        val dGateRows = Array(seqLen) { FloatArray(dFf) }
        for (t in 0 until seqLen - 1) {
            for (dimension in 0 until dFf) {
                val sigmoid = 1.0f / (1.0f + exp(-gate[t][dimension]))
                val swish = gate[t][dimension] * sigmoid
                val dActValue = dActRows[t][dimension]
                dUpRows[t][dimension] = dActValue * swish
                val dSwish = dActValue * up[t][dimension]
                dGateRows[t][dimension] =
                    dSwish * (sigmoid * (1f + gate[t][dimension] * (1f - sigmoid)))
            }
        }

        // Accumulate all three FFN weight gradients as batched matrix products.
        // gradWDown = transpose(act) × dL_dh2PreLN.
        val actTransposed = transposeRows(act)
        val gradWDownRows = Array(dFf) { FloatArray(dModel) }
        computeLinearBatch(
            inputs = actTransposed,
            weights = flattenRows(dL_dh2PreLN),
            inputColumns = seqLen,
            outputColumns = dModel,
            outputs = gradWDownRows
        )
        for (row in gradWDownRows.indices) {
            System.arraycopy(gradWDownRows[row], 0, gradWDown, row * dModel, dModel)
        }

        // gradWUp = transpose(h1) × dUp; gradWGate = transpose(h1) × dGate.
        val h1Transposed = transposeRows(h1)
        val gradWUpRows = Array(dModel) { FloatArray(dFf) }
        computeLinearBatch(
            inputs = h1Transposed,
            weights = flattenRows(dUpRows),
            inputColumns = seqLen,
            outputColumns = dFf,
            outputs = gradWUpRows
        )
        for (row in gradWUpRows.indices) {
            System.arraycopy(gradWUpRows[row], 0, gradWUp, row * dFf, dFf)
        }

        val gradWGateRows = Array(dModel) { FloatArray(dFf) }
        computeLinearBatch(
            inputs = h1Transposed,
            weights = flattenRows(dGateRows),
            inputColumns = seqLen,
            outputColumns = dFf,
            outputs = gradWGateRows
        )
        for (row in gradWGateRows.indices) {
            System.arraycopy(gradWGateRows[row], 0, gradWGate, row * dFf, dFf)
        }

        // Propagate dGate and dUp through their corresponding FFN projections.
        val dHiddenFromGate = Array(seqLen) { FloatArray(dModel) }
        computeLinearBatch(
            inputs = dGateRows,
            weights = wGate,
            inputColumns = dFf,
            outputColumns = dModel,
            outputs = dHiddenFromGate,
            transposeWeights = true
        )

        val dHiddenFromUp = Array(seqLen) { FloatArray(dModel) }
        computeLinearBatch(
            inputs = dUpRows,
            weights = wUp,
            inputColumns = dFf,
            outputColumns = dModel,
            outputs = dHiddenFromUp,
            transposeWeights = true
        )

        for (t in 0 until seqLen - 1) {
            for (dimension in 0 until dModel) {
                dL_dh1PostLN[t][dimension] =
                    dHiddenFromGate[t][dimension] +
                    dHiddenFromUp[t][dimension] +
                    dL_dh2PreLN[t][dimension]
            }
            layerNormBackward(dL_dh1PostLN[t], h1PreLN[t], dL_dh1PreLN[t], dModel)
        }

        val attentionBackwardRows = seqLen - 1
        val attentionOutputRows = Array(attentionBackwardRows) { row -> attnOut[row] }
        val outputProjectionGradientRows =
            Array(attentionBackwardRows) { row -> dL_dh1PreLN[row] }

        val gradWORows = Array(dModel) { FloatArray(dModel) }
        computeLinearBatch(
            inputs = transposeRows(attentionOutputRows),
            weights = flattenRows(outputProjectionGradientRows),
            inputColumns = attentionBackwardRows,
            outputColumns = dModel,
            outputs = gradWORows
        )
        System.arraycopy(flattenRows(gradWORows), 0, gradWO, 0, gradWO.size)

        val attentionInputGradientRows =
            Array(attentionBackwardRows) { FloatArray(dModel) }
        computeLinearBatch(
            inputs = outputProjectionGradientRows,
            weights = wO,
            inputColumns = dModel,
            outputColumns = dModel,
            outputs = attentionInputGradientRows,
            transposeWeights = true
        )
        for (time in 0 until attentionBackwardRows) {
            System.arraycopy(
                attentionInputGradientRows[time],
                0,
                dL_dattnOut[time],
                0,
                dModel
            )
        }

        // (3) Causal attention backward. Independent heads are processed as a batch,
        // avoiding one queue submission and synchronization for every individual head.
        val attentionGradientByHead = Array(nHeads) { head ->
            val headOffset = head * headDim
            Array(seqLen) { time ->
                FloatArray(headDim) { dimension ->
                    dL_dattnOut[time][headOffset + dimension]
                }
            }
        }
        val queryRowsByHead = Array(nHeads) { head ->
            val headOffset = head * headDim
            Array(seqLen) { time ->
                FloatArray(headDim) { dimension -> q[time][headOffset + dimension] }
            }
        }
        val keyRowsByHead = Array(nHeads) { head ->
            val headOffset = head * headDim
            Array(seqLen) { time ->
                FloatArray(headDim) { dimension -> k[time][headOffset + dimension] }
            }
        }
        val valueRowsByHead = Array(nHeads) { head ->
            val headOffset = head * headDim
            Array(seqLen) { time ->
                FloatArray(headDim) { dimension -> v[time][headOffset + dimension] }
            }
        }

        // dV = transpose(P) × dAttention. P is lower triangular.
        val transposedAttentionByHead = Array(nHeads) { head ->
            Array(seqLen) { keyIndex ->
                FloatArray(seqLen) { queryIndex ->
                    if (queryIndex >= keyIndex) {
                        attnWeights[head][queryIndex][keyIndex]
                    } else {
                        0f
                    }
                }
            }
        }
        val dValueByHead = Array(nHeads) { Array(seqLen) { FloatArray(headDim) } }
        computeBatchedLinear(
            inputs = transposedAttentionByHead,
            weights = Array(nHeads) { head -> flattenRows(attentionGradientByHead[head]) },
            outputs = dValueByHead,
            inputColumns = seqLen,
            outputColumns = headDim
        )

        // dW = dAttention × transpose(V); the causal mask is applied below.
        val dWeightsByHead = Array(nHeads) { Array(seqLen) { FloatArray(seqLen) } }
        computeBatchedLinear(
            inputs = attentionGradientByHead,
            weights = Array(nHeads) { head -> flattenRows(valueRowsByHead[head]) },
            outputs = dWeightsByHead,
            inputColumns = headDim,
            outputColumns = seqLen,
            transposeWeights = true
        )

        val dScoreByHead = Array(nHeads) { Array(seqLen) { FloatArray(seqLen) } }
        for (head in 0 until nHeads) {
            for (queryIndex in 0 until seqLen) {
                var weightedGradient = 0f
                for (keyIndex in 0..queryIndex) {
                    weightedGradient +=
                        attnWeights[head][queryIndex][keyIndex] *
                        dWeightsByHead[head][queryIndex][keyIndex]
                }
                for (keyIndex in 0..queryIndex) {
                    dScoreByHead[head][queryIndex][keyIndex] =
                        attnWeights[head][queryIndex][keyIndex] *
                        (dWeightsByHead[head][queryIndex][keyIndex] - weightedGradient) *
                        scale
                }
            }
        }

        // dQ = dScore × K; dK = transpose(dScore) × Q.
        val dQueryByHead = Array(nHeads) { Array(seqLen) { FloatArray(headDim) } }
        computeBatchedLinear(
            inputs = dScoreByHead,
            weights = Array(nHeads) { head -> flattenRows(keyRowsByHead[head]) },
            outputs = dQueryByHead,
            inputColumns = seqLen,
            outputColumns = headDim
        )

        val transposedDScoreByHead = Array(nHeads) { head ->
            transposeRows(dScoreByHead[head])
        }
        val dKeyByHead = Array(nHeads) { Array(seqLen) { FloatArray(headDim) } }
        computeBatchedLinear(
            inputs = transposedDScoreByHead,
            weights = Array(nHeads) { head -> flattenRows(queryRowsByHead[head]) },
            outputs = dKeyByHead,
            inputColumns = seqLen,
            outputColumns = headDim
        )

        for (head in 0 until nHeads) {
            val headOffset = head * headDim
            for (time in 0 until seqLen) {
                System.arraycopy(dQueryByHead[head][time], 0, dQ[time], headOffset, headDim)
                System.arraycopy(dKeyByHead[head][time], 0, dK[time], headOffset, headDim)
                System.arraycopy(dValueByHead[head][time], 0, dV[time], headOffset, headDim)
            }
        }

        // (4) Q/K/V parameter gradients. The small rank-by-model LoRA updates
        // remain on CPU to avoid extra synchronization for low-rank workloads.
        val inputRowsForGradient = Array(attentionBackwardRows) { row -> x[row] }
        val transposedInputRows = transposeRows(inputRowsForGradient)
        val queryGradientRows = Array(attentionBackwardRows) { row -> dQ[row] }
        val keyGradientRows = Array(attentionBackwardRows) { row -> dK[row] }
        val valueGradientRows = Array(attentionBackwardRows) { row -> dV[row] }

        val qkvGradientOutputs = Array(3) { Array(dModel) { FloatArray(dModel) } }
        computeBatchedLinear(
            inputs = arrayOf(transposedInputRows, transposedInputRows, transposedInputRows),
            weights = arrayOf(
                flattenRows(queryGradientRows),
                flattenRows(keyGradientRows),
                flattenRows(valueGradientRows)
            ),
            outputs = qkvGradientOutputs,
            inputColumns = attentionBackwardRows,
            outputColumns = dModel
        )
        System.arraycopy(flattenRows(qkvGradientOutputs[0]), 0, gradWQ, 0, gradWQ.size)
        System.arraycopy(flattenRows(qkvGradientOutputs[1]), 0, gradWK, 0, gradWK.size)
        System.arraycopy(flattenRows(qkvGradientOutputs[2]), 0, gradWV, 0, gradWV.size)

        // LoRA Q/V gradients remain local and use the original rank-sized equations.
        for (time in 0 until attentionBackwardRows) {
            for (rankIndex in 0 until loraRank) {
                val queryMidValue = mQ[time][rankIndex]
                val valueMidValue = mV[time][rankIndex]
                val queryGradientOffset = rankIndex * dModel
                for (dimension in 0 until dModel) {
                    gradLoraQb[queryGradientOffset + dimension] +=
                        queryMidValue * dQ[time][dimension]
                    gradLoraVb[queryGradientOffset + dimension] +=
                        valueMidValue * dV[time][dimension]
                }
            }
            for (rankIndex in 0 until loraRank) {
                var queryBackProjection = 0f
                var valueBackProjection = 0f
                val queryWeightOffset = rankIndex * dModel
                for (dimension in 0 until dModel) {
                    queryBackProjection +=
                        dQ[time][dimension] * loraQb[queryWeightOffset + dimension]
                    valueBackProjection +=
                        dV[time][dimension] * loraVb[queryWeightOffset + dimension]
                }
                for (dimension in 0 until dModel) {
                    gradLoraQa[dimension * loraRank + rankIndex] +=
                        x[time][dimension] * queryBackProjection
                    gradLoraVa[dimension * loraRank + rankIndex] +=
                        x[time][dimension] * valueBackProjection
                }
            }
        }

        // (5) Embedding Backward Accumulation
        val wQEff = FloatArray(dModel * dModel)
        val wVEff = FloatArray(dModel * dModel)
        for (i in 0 until dModel) {
            for (j in 0 until dModel) {
                var qSum = wQ[i * dModel + j]
                var vSum = wV[i * dModel + j]
                for (r in 0 until loraRank) {
                    qSum += loraQa[i * loraRank + r] * loraQb[r * dModel + j]
                    vSum += loraVa[i * loraRank + r] * loraVb[r * dModel + j]
                }
                wQEff[i * dModel + j] = qSum
                wVEff[i * dModel + j] = vSum
            }
        }

        val inputGradientByProjection =
            Array(3) { Array(attentionBackwardRows) { FloatArray(dModel) } }
        computeBatchedLinear(
            inputs = arrayOf(queryGradientRows, keyGradientRows, valueGradientRows),
            weights = arrayOf(wQEff, wK, wVEff),
            outputs = inputGradientByProjection,
            inputColumns = dModel,
            outputColumns = dModel,
            transposeWeights = true
        )

        for (time in 0 until attentionBackwardRows) {
            val token = seq[time]
            if (token !in 0 until vocabCap) continue

            val embeddingOffset = token * dModel
            val queryInputGradient = inputGradientByProjection[0][time]
            val keyInputGradient = inputGradientByProjection[1][time]
            val valueInputGradient = inputGradientByProjection[2][time]
            for (dimension in 0 until dModel) {
                val inputGradient =
                    dL_dh1PreLN[time][dimension] +
                    queryInputGradient[dimension] +
                    keyInputGradient[dimension] +
                    valueInputGradient[dimension]
                gradWEmbed[embeddingOffset + dimension] += inputGradient * embedScale
            }
        }

        // ==================== GLOBAL GRADIENT CLIPPING & ADAM STEP ====================
        val stepDiv = max(validSteps, 1).toFloat()
        normalizeGradient(gradWHead, stepDiv)
        normalizeGradient(gradWEmbed, stepDiv)
        normalizeGradient(gradWDown, stepDiv)
        normalizeGradient(gradWUp, stepDiv)
        normalizeGradient(gradWGate, stepDiv)
        normalizeGradient(gradWO, stepDiv)
        normalizeGradient(gradWQ, stepDiv)
        normalizeGradient(gradWK, stepDiv)
        normalizeGradient(gradWV, stepDiv)
        normalizeGradient(gradLoraQa, stepDiv)
        normalizeGradient(gradLoraQb, stepDiv)
        normalizeGradient(gradLoraVa, stepDiv)
        normalizeGradient(gradLoraVb, stepDiv)

        if (gradientProbe != null) {
            val selectedGradient = when (gradientProbe.parameterName) {
                "wHead" -> gradWHead
                "wEmbed" -> gradWEmbed
                "wDown" -> gradWDown
                "wUp" -> gradWUp
                "wGate" -> gradWGate
                "wO" -> gradWO
                "wQ" -> gradWQ
                "wK" -> gradWK
                "wV" -> gradWV
                "loraQa" -> gradLoraQa
                "loraQb" -> gradLoraQb
                "loraVa" -> gradLoraVa
                "loraVb" -> gradLoraVb
                else -> throw IllegalArgumentException(
                    "Unsupported gradient-check parameter: ${gradientProbe.parameterName}"
                )
            }
            val resolvedIndex = if (gradientProbe.parameterIndex == -1) {
                selectedGradient.indices.maxByOrNull {
                    kotlin.math.abs(selectedGradient[it])
                } ?: throw IllegalStateException("Cannot inspect an empty gradient tensor.")
            } else {
                gradientProbe.parameterIndex
            }
            require(resolvedIndex in selectedGradient.indices) {
                "Gradient index ${resolvedIndex} is outside ${gradientProbe.parameterName}."
            }
            gradientProbe.parameterIndex = resolvedIndex
            gradientProbe.analyticalGradient = selectedGradient[resolvedIndex]
        }

        if (!updateParameters) {
            return if (validSteps > 0) totalLoss / validSteps else Float.NaN
        }

        // Hitung total L2 Norm untuk Gradient Clipping
        var sumSquares = 0f
        sumSquares += computeSumSquares(gradWHead)
        sumSquares += computeSumSquares(gradWEmbed)
        sumSquares += computeSumSquares(gradWDown)
        sumSquares += computeSumSquares(gradWUp)
        sumSquares += computeSumSquares(gradWGate)
        sumSquares += computeSumSquares(gradWO)
        sumSquares += computeSumSquares(gradWQ)
        sumSquares += computeSumSquares(gradWK)
        sumSquares += computeSumSquares(gradWV)
        sumSquares += computeSumSquares(gradLoraQa)
        sumSquares += computeSumSquares(gradLoraQb)
        sumSquares += computeSumSquares(gradLoraVa)
        sumSquares += computeSumSquares(gradLoraVb)

        val totalNorm = sqrt(sumSquares)
        if (!totalNorm.isFinite()) { adamStep--; return Float.NaN }
        val clipFactor = if (totalNorm > maxGradNorm) maxGradNorm / (totalNorm + 1e-6f) else 1.0f

        // Keep the existing clipped-gradient AdamW math available as the exact CPU fallback.
        val beta1Pow = Math.pow(beta1.toDouble(), adamStep.toDouble()).toFloat()
        val beta2Pow = Math.pow(beta2.toDouble(), adamStep.toDouble()).toFloat()
        val biasCorr1 = 1f - beta1Pow
        val biasCorr2 = 1f - beta2Pow

        val parameterGroups = arrayOf(
            wHead, wEmbed, wDown, wUp, wGate, wO, wQ, wK, wV,
            loraQa, loraQb, loraVa, loraVb
        )
        val gradientGroups = arrayOf(
            gradWHead, gradWEmbed, gradWDown, gradWUp, gradWGate, gradWO,
            gradWQ, gradWK, gradWV, gradLoraQa, gradLoraQb, gradLoraVa, gradLoraVb
        )
        val firstMomentGroups = arrayOf(
            mHead, mEmbed, mWDown, mWUp, mWGate, mWO, mWQ, mWK, mWV,
            mLoraQa, mLoraQb, mLoraVa, mLoraVb
        )
        val secondMomentGroups = arrayOf(
            vHead, vEmbed, vWDown, vWUp, vWGate, vWO, vWQ, vWK, vWV,
            vLoraQa, vLoraQb, vLoraVa, vLoraVb
        )
        val weightDecayByGroup = floatArrayOf(
            0f, 0f,
            weightDecay, weightDecay, weightDecay, weightDecay, weightDecay, weightDecay, weightDecay,
            0f, 0f, 0f, 0f
        )

        val updatedByVulkan = VulkanTrainingBackend.tryApplyAdamWUpdate(
            parameters = parameterGroups,
            gradients = gradientGroups,
            firstMoments = firstMomentGroups,
            secondMoments = secondMomentGroups,
            weightDecayByGroup = weightDecayByGroup,
            learningRate = lr,
            beta1 = beta1,
            beta2 = beta2,
            biasCorrection1 = biasCorr1,
            biasCorrection2 = biasCorr2,
            epsilon = epsilon,
            gradientClipFactor = clipFactor
        )

        if (!updatedByVulkan) {
            applyClip(gradWHead, clipFactor)
            applyClip(gradWEmbed, clipFactor)
            applyClip(gradWDown, clipFactor)
            applyClip(gradWUp, clipFactor)
            applyClip(gradWGate, clipFactor)
            applyClip(gradWO, clipFactor)
            applyClip(gradWQ, clipFactor)
            applyClip(gradWK, clipFactor)
            applyClip(gradWV, clipFactor)
            applyClip(gradLoraQa, clipFactor)
            applyClip(gradLoraQb, clipFactor)
            applyClip(gradLoraVa, clipFactor)
            applyClip(gradLoraVb, clipFactor)

            adamUpdate(wHead, gradWHead, mHead, vHead, lr, biasCorr1, biasCorr2)
            adamUpdate(wEmbed, gradWEmbed, mEmbed, vEmbed, lr, biasCorr1, biasCorr2)
            adamUpdate(wDown, gradWDown, mWDown, vWDown, lr, biasCorr1, biasCorr2, weightDecay)
            adamUpdate(wUp, gradWUp, mWUp, vWUp, lr, biasCorr1, biasCorr2, weightDecay)
            adamUpdate(wGate, gradWGate, mWGate, vWGate, lr, biasCorr1, biasCorr2, weightDecay)
            adamUpdate(wO, gradWO, mWO, vWO, lr, biasCorr1, biasCorr2, weightDecay)
            adamUpdate(wQ, gradWQ, mWQ, vWQ, lr, biasCorr1, biasCorr2, weightDecay)
            adamUpdate(wK, gradWK, mWK, vWK, lr, biasCorr1, biasCorr2, weightDecay)
            adamUpdate(wV, gradWV, mWV, vWV, lr, biasCorr1, biasCorr2, weightDecay)
            adamUpdate(loraQa, gradLoraQa, mLoraQa, vLoraQa, lr, biasCorr1, biasCorr2)
            adamUpdate(loraQb, gradLoraQb, mLoraQb, vLoraQb, lr, biasCorr1, biasCorr2)
            adamUpdate(loraVa, gradLoraVa, mLoraVa, vLoraVa, lr, biasCorr1, biasCorr2)
            adamUpdate(loraVb, gradLoraVb, mLoraVb, vLoraVb, lr, biasCorr1, biasCorr2)
        }

        return if (validSteps > 0) totalLoss / validSteps else Float.NaN
    }

    private fun normalizeGradient(grad: FloatArray, div: Float) {
        val inv = 1.0f / div
        for (i in grad.indices) grad[i] *= inv
    }

    private fun computeSumSquares(arr: FloatArray): Float {
        var s = 0f
        for (i in arr.indices) s += arr[i] * arr[i]
        return s
    }

    private fun applyClip(arr: FloatArray, factor: Float) {
        if (factor == 1.0f) return
        for (i in arr.indices) arr[i] *= factor
    }

    private fun adamUpdate(
        param: FloatArray,
        grad: FloatArray,
        m: FloatArray,
        v: FloatArray,
        lr: Float,
        biasCorr1: Float,
        biasCorr2: Float,
        wd: Float = 0f
    ) {
        val limit = min(param.size, grad.size)
        for (i in 0 until limit) {
            val g = grad[i]
            m[i] = beta1 * m[i] + (1f - beta1) * g
            v[i] = beta2 * v[i] + (1f - beta2) * g * g
            val mHat = m[i] / biasCorr1
            val vHat = v[i] / biasCorr2
            param[i] -= lr * (mHat / (sqrt(vHat) + epsilon) + wd * param[i])
        }
    }

    private fun computeTokenEmbedAndPos(tokenId: Int, pos: Int, scale: Float, out: FloatArray) {
        val embOffset = if (tokenId in 0 until vocabCap) tokenId * dModel else 0
        val posOffset = (pos % maxSeqLen) * dModel
        for (d in 0 until dModel) out[d] = wEmbed[embOffset + d] * scale + wPos[posOffset + d]
    }

    private fun computeLinear(x: FloatArray, w: FloatArray, inDim: Int, outDim: Int, out: FloatArray) {
        for (c in 0 until outDim) {
            var sum = 0f
            for (r in 0 until inDim) sum += x[r] * w[r * outDim + c]
            out[c] = sum
        }
    }

    private fun flattenRows(rows: Array<FloatArray>): FloatArray {
        if (rows.isEmpty()) return FloatArray(0)
        val columnCount = rows[0].size
        require(rows.all { row -> row.size == columnCount }) {
            "Matrix rows must have a consistent width."
        }
        val elementCount = Math.multiplyExact(rows.size, columnCount)
        return FloatArray(elementCount).also { flattened ->
            for (rowIndex in rows.indices) {
                System.arraycopy(
                    rows[rowIndex],
                    0,
                    flattened,
                    rowIndex * columnCount,
                    columnCount
                )
            }
        }
    }

    private fun transposeRows(rows: Array<FloatArray>): Array<FloatArray> {
        if (rows.isEmpty()) return emptyArray()
        val columnCount = rows[0].size
        require(rows.all { row -> row.size == columnCount }) {
            "Matrix rows must have a consistent width."
        }
        return Array(columnCount) { column ->
            FloatArray(rows.size) { row -> rows[row][column] }
        }
    }

    private fun computeLinearBatch(
        inputs: Array<FloatArray>,
        weights: FloatArray,
        inputColumns: Int,
        outputColumns: Int,
        outputs: Array<FloatArray>,
        transposeWeights: Boolean = false
    ) {
        val dispatchedToVulkan = VulkanTrainingBackend.tryApplyLinear(
            inputRows = inputs,
            weights = weights,
            outputRows = outputs,
            inputColumns = inputColumns,
            outputColumns = outputColumns,
            transposeWeights = transposeWeights
        )
        if (dispatchedToVulkan) return

        if (transposeWeights) {
            for (row in inputs.indices) {
                computeLinearTransposed(
                    inputs[row],
                    weights,
                    inputColumns,
                    outputColumns,
                    outputs[row]
                )
            }
        } else {
            for (row in inputs.indices) {
                computeLinear(
                    inputs[row],
                    weights,
                    inputColumns,
                    outputColumns,
                    outputs[row]
                )
            }
        }
    }

    private fun computeBatchedLinear(
        inputs: Array<Array<FloatArray>>,
        weights: Array<FloatArray>,
        outputs: Array<Array<FloatArray>>,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean = false
    ) {
        require(inputs.isNotEmpty()) { "Batched linear inputs must not be empty." }
        require(inputs.size == weights.size && inputs.size == outputs.size) {
            "Batched linear inputs, weights, and outputs must have matching batch counts."
        }

        val dispatchedToVulkan = VulkanTrainingBackend.tryApplyBatchedLinear(
            inputBatches = inputs,
            weightBatches = weights,
            outputBatches = outputs,
            inputColumns = inputColumns,
            outputColumns = outputColumns,
            transposeWeights = transposeWeights
        )
        if (dispatchedToVulkan) return

        for (batch in inputs.indices) {
            val weight = weights[batch]
            require(inputs[batch].size == outputs[batch].size) {
                "Batched linear input and output row counts must match."
            }
            for (row in inputs[batch].indices) {
                if (transposeWeights) {
                    computeLinearTransposed(
                        inputs[batch][row],
                        weight,
                        inputColumns,
                        outputColumns,
                        outputs[batch][row]
                    )
                } else {
                    computeLinear(
                        inputs[batch][row],
                        weight,
                        inputColumns,
                        outputColumns,
                        outputs[batch][row]
                    )
                }
            }
        }
    }

    private fun computeLinearTransposed(
        input: FloatArray,
        weights: FloatArray,
        inputColumns: Int,
        outputColumns: Int,
        output: FloatArray
    ) {
        for (column in 0 until outputColumns) {
            var sum = 0f
            val weightOffset = column * inputColumns
            for (inner in 0 until inputColumns) {
                sum += input[inner] * weights[weightOffset + inner]
            }
            output[column] = sum
        }
    }

    private fun computeLinearWithLora(
        x: FloatArray, wBase: FloatArray, loraA: FloatArray, loraB: FloatArray,
        inDim: Int, outDim: Int, rank: Int,
        midBuf: FloatArray, finalOut: FloatArray
    ) {
        computeLinear(x, wBase, inDim, outDim, finalOut)
        addLoraProjection(x, loraA, loraB, inDim, outDim, rank, midBuf, finalOut)
    }

    private fun addLoraProjection(
        input: FloatArray,
        loraA: FloatArray,
        loraB: FloatArray,
        inputColumns: Int,
        outputColumns: Int,
        rank: Int,
        intermediate: FloatArray,
        output: FloatArray
    ) {
        for (rankIndex in 0 until rank) {
            var sum = 0f
            for (inputIndex in 0 until inputColumns) {
                sum += input[inputIndex] * loraA[inputIndex * rank + rankIndex]
            }
            intermediate[rankIndex] = sum
        }
        for (outputIndex in 0 until outputColumns) {
            var sum = 0f
            for (rankIndex in 0 until rank) {
                sum += intermediate[rankIndex] * loraB[rankIndex * outputColumns + outputIndex]
            }
            output[outputIndex] += sum
        }
    }

    private fun layerNormInPlace(vec: FloatArray, dim: Int) {
        var mean = 0f
        for (i in 0 until dim) mean += vec[i]
        mean /= dim
        var variance = 0f
        for (i in 0 until dim) {
            val diff = vec[i] - mean
            variance += diff * diff
        }
        variance /= dim
        val invStd = 1.0f / sqrt(variance + 1e-5f)
        for (i in 0 until dim) vec[i] = (vec[i] - mean) * invStd
    }

    private fun layerNormBackward(dY: FloatArray, xPreLN: FloatArray, dX: FloatArray, dim: Int) {
        var mean = 0f
        for (i in 0 until dim) mean += xPreLN[i]
        mean /= dim
        var variance = 0f
        for (i in 0 until dim) {
            val d = xPreLN[i] - mean
            variance += d * d
        }
        variance /= dim
        val invStd = 1.0f / sqrt(variance + 1e-5f)

        var sumDY = 0f
        var sumDYY = 0f
        for (i in 0 until dim) {
            val y = (xPreLN[i] - mean) * invStd
            sumDY += dY[i]
            sumDYY += dY[i] * y
        }
        val invDim = 1f / dim
        for (i in 0 until dim) {
            val y = (xPreLN[i] - mean) * invStd
            dX[i] = invStd * (dY[i] - sumDY * invDim - y * sumDYY * invDim)
        }
    }

    private fun sampleLogits(logits: FloatArray, vocabSize: Int, temperature: Float): Int {
        val temp = max(temperature, 0.1f)
        var maxVal = -Float.MAX_VALUE
        for (i in 0 until vocabSize) {
            val scaled = logits[i] / temp
            if (scaled > maxVal) maxVal = scaled
        }
        var sumExp = 0f
        val expScores = FloatArray(vocabSize)
        for (i in 0 until vocabSize) {
            val e = exp(min(max(logits[i] / temp - maxVal, -50f), 50f))
            expScores[i] = e; sumExp += e
        }
        if (sumExp <= 0f || sumExp.isNaN()) return Tokenizer.EOS_ID
        val rand = Random.nextFloat() * sumExp
        var cum = 0f
        for (i in 0 until vocabSize) {
            cum += expScores[i]
            if (cum >= rand) return i
        }
        return vocabSize - 1
    }

    // =====================================================================
    // Benchmark
    // =====================================================================
    suspend fun runBenchmark(iterations: Int = 1): BenchmarkResult = withContext(Dispatchers.Default) {
        val startNs = System.nanoTime()
        generate("A.R.G.U.S benchmark probe", maxNewTokens = 20, temperature = 0.5f)
        val elapsedNs = System.nanoTime() - startNs
        val elapsedMs = elapsedNs / 1_000_000L
        val seconds = elapsedNs / 1_000_000_000.0f
        val tokens = 20
        val tps = if (seconds > 0) tokens / seconds else 0f
        val msPerToken = if (tokens > 0) elapsedMs.toFloat() / tokens else 0f
        val memMb = (countTotalParameters() * 4L) / (1024f * 1024f)
        BenchmarkResult(tps, msPerToken, memMb, tokens, elapsedMs)
    }

    // =====================================================================
    // Checkpoint (Format ARG3 dengan Verifikasi Transaksional)
    // =====================================================================
    fun saveCheckpoint(): Boolean {
        val bytes = synchronized(modelLock) { buildCheckpointBytes() }
        return try {
            writeBytesToDisk(bytes)
            val file = File(context.filesDir, "argus_model_weights.bin")
            file.exists() && file.length() >= HEADER_SIZE
        } catch (t: Throwable) {
            Log.e(TAG, "Checkpoint save failed", t)
            false
        }
    }

    private fun buildCheckpointBytes(): ByteArray {
        val vocabList = tokenizer.exportVocab()

        val header = ByteBuffer.allocate(HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN)
        header.put(MAGIC_0).put(MAGIC_1).put(MAGIC_2).put(MAGIC_3)
        header.putInt(CHECKPOINT_VERSION)
        header.putInt(dModel)
        header.putInt(nHeads)
        header.putInt(dFf)
        header.putInt(maxSeqLen)
        header.putInt(loraRank)
        header.putInt(tokenizer.vocabSize)
        header.putInt(vocabCap)
        header.putInt(adamStep)
        header.putInt(0) // reserved

        var vocabBytesLen = 4
        vocabList.forEach { vocabBytesLen += 8 + it.first.toByteArray(Charsets.UTF_8).size }
        val vocabBuf = ByteBuffer.allocate(vocabBytesLen).order(ByteOrder.LITTLE_ENDIAN)
        vocabBuf.putInt(vocabList.size)
        vocabList.forEach { (tok, id) ->
            val b = tok.toByteArray(Charsets.UTF_8)
            vocabBuf.putInt(id); vocabBuf.putInt(b.size); vocabBuf.put(b)
        }

        val arrays = listOf(
            wEmbed, mEmbed, vEmbed,
            wHead, mHead, vHead,
            loraQa, loraQb, loraVa, loraVb,
            mLoraQa, mLoraQb, mLoraVa, mLoraVb,
            vLoraQa, vLoraQb, vLoraVa, vLoraVb,
            wQ, wK, wV, wO,
            mWQ, vWQ, mWK, vWK, mWV, vWV, mWO, vWO,
            wGate, wUp, wDown,
            mWGate, vWGate, mWUp, vWUp, mWDown, vWDown
        )
        val totalFloats = arrays.sumOf { it.size }
        val floatBuf = ByteBuffer.allocate(totalFloats * 4).order(ByteOrder.LITTLE_ENDIAN)
        arrays.forEach { arr -> arr.forEach { floatBuf.putFloat(it) } }

        return header.array() + vocabBuf.array() + floatBuf.array()
    }

    private fun writeBytesToDisk(bytes: ByteArray) {
        val target = File(context.filesDir, "argus_model_weights.bin")
        val temp = File(context.filesDir, "argus_model_weights.bin.tmp")
        FileOutputStream(temp).use { fos ->
            BufferedOutputStream(fos).use { bos ->
                bos.write(bytes)
                bos.flush()
                fos.fd.sync()
            }
        }
        Files.move(
            temp.toPath(), target.toPath(),
            StandardCopyOption.REPLACE_EXISTING,
            StandardCopyOption.ATOMIC_MOVE
        )
    }

    fun loadCheckpointIfExists(): Boolean {
        try {
            val file = File(context.filesDir, "argus_model_weights.bin")
            if (!file.exists() || file.length() < HEADER_SIZE) return false

            FileInputStream(file).use { fis ->
                BufferedInputStream(fis).use { bis ->
                    val headerBytes = bis.readFully(HEADER_SIZE)
                    val header = ByteBuffer.wrap(headerBytes).order(ByteOrder.LITTLE_ENDIAN)

                    if (header.get() != MAGIC_0 || header.get() != MAGIC_1 ||
                        header.get() != MAGIC_2 || header.get() != MAGIC_3) {
                        Log.w(TAG, "Checkpoint bukan format ARG3; skip")
                        return false
                    }
                    if (header.int != CHECKPOINT_VERSION) {
                        Log.w(TAG, "Checkpoint versi tidak cocok; skip")
                        return false
                    }

                    val savedDModel = header.int
                    val savedNHeads = header.int
                    val savedDFf = header.int
                    val savedMaxSeq = header.int
                    val savedLoraRank = header.int
                    val savedVocabSize = header.int
                    val savedVocabCap = header.int
                    val savedAdamStep = header.int
                    header.int // reserved

                    if (savedDModel != dModel || savedLoraRank != loraRank ||
                        savedNHeads != nHeads || savedDFf != dFf || savedMaxSeq != maxSeqLen) {
                        Log.w(TAG, "Arsitektur mismatch: saved($savedDModel,$savedDFf) vs current($dModel,$dFf); skip")
                        return false
                    }
                    if (savedVocabCap <= 0) {
                        Log.w(TAG, "vocabCap invalid: $savedVocabCap")
                        return false
                    }

                    val countBuf = bis.readFully(4)
                    val vocabCount = ByteBuffer.wrap(countBuf).order(ByteOrder.LITTLE_ENDIAN).int
                    if (vocabCount < 0 || vocabCount > 1_000_000) {
                        Log.w(TAG, "vocabCount suspicious: $vocabCount")
                        return false
                    }
                    val entries = ArrayList<Pair<String, Int>>(vocabCount)
                    repeat(vocabCount) {
                        val meta = ByteBuffer.wrap(bis.readFully(8)).order(ByteOrder.LITTLE_ENDIAN)
                        val id = meta.int
                        val len = meta.int
                        if (len < 0 || len > 256) throw IllegalStateException("Token len $len invalid")
                        val tokBytes = bis.readFully(len)
                        entries.add(String(tokBytes, Charsets.UTF_8) to id)
                    }

                    synchronized(modelLock) {
                        tokenizer.importVocab(entries)
                        resizeVocabCapacityTo(savedVocabCap)

                        val arrays = listOf(
                            wEmbed, mEmbed, vEmbed,
                            wHead, mHead, vHead,
                            loraQa, loraQb, loraVa, loraVb,
                            mLoraQa, mLoraQb, mLoraVa, mLoraVb,
                            vLoraQa, vLoraQb, vLoraVa, vLoraVb,
                            wQ, wK, wV, wO,
                            mWQ, vWQ, mWK, vWK, mWV, vWV, mWO, vWO,
                            wGate, wUp, wDown,
                            mWGate, vWGate, mWUp, vWUp, mWDown, vWDown
                        )
                        val totalFloats = arrays.sumOf { it.size }
                        val floatBytes = bis.readFully(totalFloats * 4)
                        val floatBuf = ByteBuffer.wrap(floatBytes).order(ByteOrder.LITTLE_ENDIAN)

                        arrays.forEach { arr ->
                            for (i in arr.indices) arr[i] = floatBuf.float
                        }
                        adamStep = savedAdamStep
                    }
                    return true
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Gagal memuat checkpoint", e)
            return false
        }
    }

    private fun InputStream.readFully(size: Int): ByteArray {
        val buf = ByteArray(size)
        var offset = 0
        while (offset < size) {
            val r = read(buf, offset, size - offset)
            if (r == -1) throw IllegalStateException("EOF sebelum selesai membaca $size bytes (terbaca $offset)")
            offset += r
        }
        return buf
    }
}