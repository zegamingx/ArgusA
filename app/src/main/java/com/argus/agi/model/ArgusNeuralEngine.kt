package com.argus.agi.model

import android.content.Context
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import java.util.zip.CRC32
import java.util.zip.CheckedOutputStream
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
    val layerCount: Int = 1,
    val isFullTrainingEnabled: Boolean = true,
    val tokenizerConfig: TokenizerConfig = TokenizerConfig(),
    val modelTier: ArgusDeviceModelTier = ArgusDeviceModelTier.COMPACT,
    val trainingBatchSize: Int = modelTier.defaultTrainingBatchSize,
    val gradientAccumulationEnabled: Boolean = true,
    val initialVocabularyCapacity: Int = tokenizerConfig.maxVocabularySize
) {
    init {
        require(dModel > 0) { "dModel must be positive." }
        require(nHeads > 0 && dModel % nHeads == 0) {
            "nHeads must be positive and divide dModel evenly."
        }
        require(dFf > 0) { "dFf must be positive." }
        require(maxSeqLen >= 2) { "maxSeqLen must be at least 2." }
        require(loraRank in 1..dModel) {
            "loraRank must be between one and dModel."
        }
        require(layerCount in 1..32) {
            "layerCount must be between 1 and 32."
        }
        require(trainingBatchSize in 1..8) {
            "Training batch size must be between 1 and 8."
        }
        require(initialVocabularyCapacity > 0) {
            "Initial vocabulary capacity must be positive."
        }
        require(initialVocabularyCapacity <= tokenizerConfig.maxVocabularySize) {
            "Initial vocabulary capacity cannot exceed the tokenizer's configured maximum."
        }

        // FP16 parameter storage with FP32 gradients and AdamW state.
        val hidden = dModel.toLong()
        val feedForward = dFf.toLong()
        val estimatedArchitecture = ArgusArchitectureConfig(
            layerCount = layerCount,
            hiddenSize = dModel,
            attentionHeadCount = nHeads,
            feedForwardSize = dFf,
            vocabularySize = initialVocabularyCapacity,
            contextLength = maxSeqLen,
            adapterRank = loraRank
        )
        val parameterCount = ArgusArchitectureParameterEstimator
            .estimate(estimatedArchitecture)
            .totalTrainableParameters
        val trainingFloorBytes = if (gradientAccumulationEnabled) {
            TrainingMemoryPlanner.minimumAccumulatedTrainingStateBytes(parameterCount)
        } else {
            TrainingMemoryPlanner.minimumTrainingStateBytes(parameterCount)
        }
        val positionElements = Math.multiplyExact(maxSeqLen.toLong(), hidden)
        require(positionElements <= Int.MAX_VALUE.toLong()) {
            "Position tensor dimensions exceed JVM array limits."
        }
        val positionBufferBytes = Math.multiplyExact(
            positionElements,
            Float.SIZE_BYTES.toLong()
        )
        val attentionScratchBytes = Math.multiplyExact(
            Math.multiplyExact(
                Math.multiplyExact(nHeads.toLong(), maxSeqLen.toLong()),
                maxSeqLen.toLong()
            ),
            Float.SIZE_BYTES.toLong()
        )
        val minimumRuntimeBytes = Math.addExact(
            Math.addExact(trainingFloorBytes, positionBufferBytes),
            attentionScratchBytes
        )
        val safeHeapBudgetBytes = Runtime.getRuntime().maxMemory().toDouble() * 0.7
        require(minimumRuntimeBytes.toDouble() <= safeHeapBudgetBytes) {
            "The FP16/FP32 engine configuration exceeds the safe app-heap budget."
        }
        require(Math.multiplyExact(hidden, feedForward) <= Int.MAX_VALUE.toLong()) {
            "Feed-forward tensor dimensions exceed JVM array limits."
        }
        require(Math.multiplyExact(hidden, hidden) <= Int.MAX_VALUE.toLong()) {
            "Attention tensor dimensions exceed JVM array limits."
        }
    }

    val architectureLabel: String
        get() = "${modelTier.displayName} (${layerCount}L, d_model $dModel, d_ff $dFf, " +
            "context $maxSeqLen, LoRA rank $loraRank, batch $trainingBatchSize)"

    companion object {
        /**
         * Selects a fresh model shape from stable device capacity, not volatile free RAM.
         * Existing checkpoint configurations are resolved separately before engine creation.
         */
        fun forDevice(
            totalRamGb: Int,
            maxHeapMb: Long,
            availableRamGb: Float = totalRamGb.toFloat(),
            cpuCores: Int = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
        ): EngineConfig {
            val profile = ArgusDeviceModelPolicy.select(
                totalRamGb = totalRamGb,
                maxHeapMb = maxHeapMb,
                availableRamGb = availableRamGb,
                cpuCores = cpuCores
            )
            val tier = profile.tier
            val shape = selectExecutableShape(
                targetParameters = profile.targetParameterCount,
                tier = tier
            )
            val tokenizerConfig = TokenizerConfig(maxVocabularySize = tier.vocabularyCapacity)
            return EngineConfig(
                dModel = shape.dModel,
                nHeads = shape.nHeads,
                dFf = shape.dFf,
                maxSeqLen = shape.contextLength,
                loraRank = shape.loraRank,
                layerCount = shape.layerCount,
                tokenizerConfig = tokenizerConfig,
                modelTier = tier,
                trainingBatchSize = profile.trainingBatchSize,
                initialVocabularyCapacity = tier.vocabularyCapacity
            )
        }

        private data class ExecutableShape(
            val dModel: Int,
            val nHeads: Int,
            val dFf: Int,
            val layerCount: Int,
            val loraRank: Int,
            val contextLength: Int,
            val parameterCount: Long
        )

        /**
         * Selects a real device-tier architecture using its configured vocabulary.
         * Balanced is fixed at d_model 128, d_ff 512, four heads, context 128,
         * and vocabulary capacity 2,048.
         */
        private fun selectExecutableShape(
            targetParameters: Long,
            tier: ArgusDeviceModelTier
        ): ExecutableShape {
            val candidates = ArrayList<ExecutableShape>()
            val dimensions = when (tier) {
                ArgusDeviceModelTier.COMPACT -> Triple(128, 384, 4)
                ArgusDeviceModelTier.BALANCED -> Triple(128, 512, 4)
                ArgusDeviceModelTier.PERFORMANCE -> Triple(192, 768, 8)
                ArgusDeviceModelTier.FLAGSHIP -> Triple(256, 1_024, 8)
            }
            val hidden = dimensions.first
            val feedForward = dimensions.second
            val heads = dimensions.third
            val rank = if (
                tier == ArgusDeviceModelTier.PERFORMANCE ||
                tier == ArgusDeviceModelTier.FLAGSHIP
            ) 8 else 4
            val contextLength = when (tier) {
                ArgusDeviceModelTier.COMPACT -> 64
                ArgusDeviceModelTier.BALANCED -> 128
                ArgusDeviceModelTier.PERFORMANCE -> 160
                ArgusDeviceModelTier.FLAGSHIP -> 192
            }

            for (layers in 1..32) {
                val architecture = ArgusArchitectureConfig(
                    layerCount = layers,
                    hiddenSize = hidden,
                    attentionHeadCount = heads,
                    feedForwardSize = feedForward,
                    vocabularySize = tier.vocabularyCapacity,
                    contextLength = contextLength,
                    adapterRank = rank
                )
                val parameterCount = ArgusArchitectureParameterEstimator
                    .estimate(architecture)
                    .totalTrainableParameters
                if (parameterCount <= targetParameters) {
                    candidates += ExecutableShape(
                        dModel = hidden,
                        nHeads = heads,
                        dFf = feedForward,
                        layerCount = layers,
                        loraRank = rank,
                        contextLength = contextLength,
                        parameterCount = parameterCount
                    )
                }
            }

            return candidates.maxByOrNull(ExecutableShape::parameterCount)
                ?: throw IllegalArgumentException(
                    "The available budget cannot fit the minimum ${tier.displayName} architecture."
                )
        }
    }
}

/** Aggregate next-token loss and the exact number of scored validation targets. */
data class LossEvaluation(
    val meanLoss: Float,
    val targetCount: Int,
    val evaluatedDocumentCount: Int
)

/** Result of a corpus-driven tokenizer vocabulary update. */
data class TokenizerTrainingResult(
    val previousVocabularySize: Int,
    val vocabularySize: Int,
    val tokensAdded: Int,
    val previousMergeCount: Int,
    val mergeCount: Int
)

/** Exact accounting for one text passed through the neural optimizer. */
data class NeuralTrainingStepResult(
    val loss: Float,
    val inputTokenCount: Int,
    val predictionTargetCount: Int,
    val contextWindowCount: Int,
    val optimizerUpdateCount: Int,
    val vocabularySize: Int
)

/** Metrics for one real optimizer update over a serial micro-batch of independent samples. */
data class NeuralTrainingBatchResult(
    val samples: List<NeuralTrainingStepResult>,
    val optimizerUpdateCount: Int,
    val totalPredictionTargetCount: Int
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
 * Argus-native on-device causal Transformer with configurable SwiGLU blocks and Q/V LoRA.
 * Supports full-parameter backpropagation, serial micro-batch gradient accumulation,
 * AdamW updates, and Argus-owned checkpoint serialization.
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
    val layerCount = config.layerCount
    val isFullTrainingEnabled = config.isFullTrainingEnabled

    val optimizerBackend: String
        get() = VulkanTrainingBackend.currentBackend
    val isVulkanRuntimeReady: Boolean
        get() = VulkanTrainingBackend.isRuntimeAvailable
    val computeBackendStatus: StateFlow<String>
        get() = VulkanTrainingBackend.runtimeStatus
    val vulkanInitializationProgress: StateFlow<VulkanInitializationProgress>
        get() = VulkanTrainingBackend.initializationProgress
    val vulkanDeviceCapabilities: StateFlow<VulkanDeviceCapabilities?>
        get() = VulkanTrainingBackend.deviceCapabilities
    val vulkanGpuPerformanceMetrics: StateFlow<VulkanGpuPerformanceMetrics>
        get() = VulkanTrainingBackend.gpuPerformanceMetrics
    val vulkanModelTensorResidencyReady: StateFlow<Boolean>
        get() = VulkanTrainingBackend.modelTensorResidencyReady

    fun initializeOptionalVulkanRuntime(): Boolean {
        val runtimeReady = VulkanTrainingBackend.initializeAfterModelReady()
        if (!runtimeReady) return false
        synchronizeVulkanModelTensorResidency("model initialization")
        return true
    }

    fun releaseVulkanTensorResidencyForMemoryPressure(): Boolean =
        VulkanTrainingBackend.releaseModelTensorResidencyForMemoryPressure()

    fun resumeVulkanTensorResidencyAfterForeground() {
        VulkanTrainingBackend.resumeModelTensorResidencyAfterForeground()
    }

    fun resetOptionalVulkanRuntimeForRetry() {
        VulkanTrainingBackend.resetRuntimeForRetry()
    }

    private fun synchronizeVulkanModelTensorResidency(reason: String) {
        if (!VulkanTrainingBackend.isFp16ComputeAvailable &&
            !VulkanTrainingBackend.isResidentActivationLinearAvailable) {
            return
        }
        val synchronized = VulkanTrainingBackend.synchronizeModelTensorResidency(
            currentNamedParameterTensors()
        )
        if (!synchronized) {
            Log.w(
                TAG,
                "Complete FP16 model residency was not established after $reason; " +
                    "the verified FP32 Vulkan path or Kotlin CPU fallback remains active."
            )
        }
    }

    val checkpointInterval: Int
        get() = CHECKPOINT_INTERVAL
    private val scale = 1.0f / sqrt(headDim.toFloat())

    private var vocabCap = max(tokenizer.vocabSize + 32, config.initialVocabularyCapacity)

    // Embeddings and trainable weights use FP16; activations and optimizer state remain FP32.
    private var wEmbed = Float16ParameterArray(vocabCap * dModel)
    private val wPos = FloatArray(maxSeqLen * dModel)

    // LM Output Head
    private var wHead = Float16ParameterArray(vocabCap * dModel)

    // LM and Embedding Optimizer Moments
    private var mHead = FloatArray(vocabCap * dModel)
    private var vHead = FloatArray(vocabCap * dModel)
    private var mEmbed = FloatArray(vocabCap * dModel)
    private var vEmbed = FloatArray(vocabCap * dModel)

    internal class TransformerBlock(
        val layerIndex: Int,
        dModel: Int,
        dFf: Int,
        loraRank: Int
    ) {
        val wQ = Float16ParameterArray(dModel * dModel)
        val wK = Float16ParameterArray(dModel * dModel)
        val wV = Float16ParameterArray(dModel * dModel)
        val wO = Float16ParameterArray(dModel * dModel)

        val loraQa = Float16ParameterArray(dModel * loraRank)
        val loraQb = Float16ParameterArray(loraRank * dModel)
        val loraVa = Float16ParameterArray(dModel * loraRank)
        val loraVb = Float16ParameterArray(loraRank * dModel)

        val wGate = Float16ParameterArray(dModel * dFf)
        val wUp = Float16ParameterArray(dModel * dFf)
        val wDown = Float16ParameterArray(dFf * dModel)

        val mWQ = FloatArray(dModel * dModel)
        val vWQ = FloatArray(dModel * dModel)
        val mWK = FloatArray(dModel * dModel)
        val vWK = FloatArray(dModel * dModel)
        val mWV = FloatArray(dModel * dModel)
        val vWV = FloatArray(dModel * dModel)
        val mWO = FloatArray(dModel * dModel)
        val vWO = FloatArray(dModel * dModel)

        val mLoraQa = FloatArray(dModel * loraRank)
        val vLoraQa = FloatArray(dModel * loraRank)
        val mLoraQb = FloatArray(loraRank * dModel)
        val vLoraQb = FloatArray(loraRank * dModel)
        val mLoraVa = FloatArray(dModel * loraRank)
        val vLoraVa = FloatArray(dModel * loraRank)
        val mLoraVb = FloatArray(loraRank * dModel)
        val vLoraVb = FloatArray(loraRank * dModel)

        val mWGate = FloatArray(dModel * dFf)
        val vWGate = FloatArray(dModel * dFf)
        val mWUp = FloatArray(dModel * dFf)
        val vWUp = FloatArray(dModel * dFf)
        val mWDown = FloatArray(dFf * dModel)
        val vWDown = FloatArray(dFf * dModel)
    }

    internal val blocks: Array<TransformerBlock> = Array(layerCount) { layerIndex ->
        TransformerBlock(layerIndex, dModel, dFf, loraRank)
    }

    // Layer 0 delegates for compatibility & single-layer callers
    val wQ get() = blocks[0].wQ
    val wK get() = blocks[0].wK
    val wV get() = blocks[0].wV
    val wO get() = blocks[0].wO
    val loraQa get() = blocks[0].loraQa
    val loraQb get() = blocks[0].loraQb
    val loraVa get() = blocks[0].loraVa
    val loraVb get() = blocks[0].loraVb
    val wGate get() = blocks[0].wGate
    val wUp get() = blocks[0].wUp
    val wDown get() = blocks[0].wDown

    val mWQ get() = blocks[0].mWQ
    val vWQ get() = blocks[0].vWQ
    val mWK get() = blocks[0].mWK
    val vWK get() = blocks[0].vWK
    val mWV get() = blocks[0].mWV
    val vWV get() = blocks[0].vWV
    val mWO get() = blocks[0].mWO
    val vWO get() = blocks[0].vWO
    val mLoraQa get() = blocks[0].mLoraQa
    val vLoraQa get() = blocks[0].vLoraQa
    val mLoraQb get() = blocks[0].mLoraQb
    val vLoraQb get() = blocks[0].vLoraQb
    val mLoraVa get() = blocks[0].mLoraVa
    val vLoraVa get() = blocks[0].vLoraVa
    val mLoraVb get() = blocks[0].mLoraVb
    val vLoraVb get() = blocks[0].vLoraVb
    val mWGate get() = blocks[0].mWGate
    val vWGate get() = blocks[0].vWGate
    val mWUp get() = blocks[0].mWUp
    val vWUp get() = blocks[0].vWUp
    val mWDown get() = blocks[0].mWDown
    val vWDown get() = blocks[0].vWDown

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

    private class BatchGradientAccumulator(
        val gradients: Array<FloatArray>,
        var resident: VulkanResidentGradientAccumulator? = null
    ) {
        var targetCount: Int = 0
    }

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
        private const val MAGIC_3 = 0x33.toByte()  // '3' (legacy)
        private const val MAGIC_4 = 0x34.toByte()  // '4'
        private const val LEGACY_HEADER_SIZE = 48
        private const val ARG4_HEADER_SIZE = 96
        private const val LEGACY_CHECKPOINT_VERSION = 3
        private const val ARG4_CHECKPOINT_VERSION = 4
        private const val ARG4_CHECKSUM_OFFSET = 88
        private const val FLOAT32_TENSOR_TYPE = 1
        private const val TENSOR_WRITE_BUFFER_BYTES = 64 * 1024
        private const val CHECKPOINT_INTERVAL = 8

        /**
         * Builds the assistant prefix used for ordinary completion or reasoning generation.
         * Reasoning mode starts at the dedicated THINK token so the model is prompted with
         * the same serialized sequence used by reasoning training examples.
         */
        internal fun formatGenerationPrompt(prompt: String, reasoningMode: Boolean): String {
            val assistantPrompt = when {
                prompt.endsWith(Tokenizer.THINK_START) -> prompt
                prompt.contains(Tokenizer.USER_TOKEN) &&
                    prompt.endsWith(Tokenizer.ASSISTANT_TOKEN) -> prompt
                prompt.contains(Tokenizer.USER_TOKEN) ->
                    "$prompt ${Tokenizer.ASSISTANT_TOKEN}"
                else ->
                    "${Tokenizer.USER_TOKEN}$prompt${Tokenizer.ASSISTANT_TOKEN}"
            }

            return if (reasoningMode && !assistantPrompt.endsWith(Tokenizer.THINK_START)) {
                "$assistantPrompt${Tokenizer.THINK_START}"
            } else {
                assistantPrompt
            }
        }

        /**
         * Preserve an existing ARG3 model's tensor dimensions when reopening the app.
         * Fresh installations use a deterministic RAM/heap tier instead.
         */
        fun resolveConfig(
            context: Context,
            totalRamGb: Int,
            maxHeapMb: Long,
            availableRamGb: Float = totalRamGb.toFloat(),
            cpuCores: Int = Runtime.getRuntime().availableProcessors().coerceAtLeast(1)
        ): EngineConfig {
            val checkpoint = File(context.filesDir, "argus_model_weights.bin")
            val savedConfig = runCatching {
                if (!checkpoint.isFile) return@runCatching null
                val prefix = readCheckpointPrefix(checkpoint)
                when {
                    prefix[0] == MAGIC_0 && prefix[1] == MAGIC_1 &&
                        prefix[2] == MAGIC_2 && prefix[3] == MAGIC_3 ->
                        readLegacyConfig(checkpoint)

                    prefix[0] == MAGIC_0 && prefix[1] == MAGIC_1 &&
                        prefix[2] == MAGIC_2 && prefix[3] == MAGIC_4 ->
                        readArg4Config(checkpoint)

                    else -> null
                }
            }.getOrNull()

            val deviceConfig = runCatching {
                EngineConfig.forDevice(
                    totalRamGb = totalRamGb,
                    maxHeapMb = maxHeapMb,
                    availableRamGb = availableRamGb,
                    cpuCores = cpuCores
                )
            }.getOrNull()

            // The tokenizer is moving to a new byte-level vocabulary contract.
            // Old ARG3/ARG4 checkpoints are intentionally not restored; a fresh
            // model must be initialized with the matching BYTE_LEVEL_BPE_V2 IDs.
            val compatibleSavedConfig = savedConfig?.takeIf { saved ->
                saved.tokenizerConfig.algorithm == ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V2
            }
            val resolvedConfig = compatibleSavedConfig ?: requireNotNull(deviceConfig) {
                "This device cannot currently allocate the minimum executable Argus model safely."
            }
            val savedParameterCount = compatibleSavedConfig?.let { saved ->
                ArgusArchitectureParameterEstimator.estimate(
                    ArgusArchitectureConfig(
                        layerCount = saved.layerCount,
                        hiddenSize = saved.dModel,
                        attentionHeadCount = saved.nHeads,
                        feedForwardSize = saved.dFf,
                        vocabularySize = saved.initialVocabularyCapacity,
                        contextLength = saved.maxSeqLen,
                        adapterRank = saved.loraRank
                    )
                ).totalTrainableParameters
            }
            val currentDeviceParameterCapacity = deviceConfig?.let { selected ->
                ArgusArchitectureParameterEstimator.estimate(
                    ArgusArchitectureConfig(
                        layerCount = selected.layerCount,
                        hiddenSize = selected.dModel,
                        attentionHeadCount = selected.nHeads,
                        feedForwardSize = selected.dFf,
                        vocabularySize = selected.initialVocabularyCapacity,
                        contextLength = selected.maxSeqLen,
                        adapterRank = selected.loraRank
                    )
                ).totalTrainableParameters
            }
            val accumulatedTrainingIsSafe = deviceConfig != null &&
                (compatibleSavedConfig == null ||
                    (
                        savedParameterCount != null &&
                            currentDeviceParameterCapacity != null &&
                            savedParameterCount <= currentDeviceParameterCapacity &&
                            runCatching {
                                compatibleSavedConfig.copy(gradientAccumulationEnabled = true)
                            }.isSuccess
                    ))
            val selectedTier = deviceConfig?.modelTier ?: resolvedConfig.modelTier
            val selectedBatchSize = if (accumulatedTrainingIsSafe) {
                deviceConfig.trainingBatchSize
            } else {
                1
            }
            return resolvedConfig.copy(
                modelTier = selectedTier,
                trainingBatchSize = selectedBatchSize,
                gradientAccumulationEnabled = accumulatedTrainingIsSafe
            )
        }

        private fun readCheckpointPrefix(file: File): ByteArray {
            require(file.length() >= 4L) { "Checkpoint header is incomplete." }
            FileInputStream(file).use { input ->
                val prefix = ByteArray(4)
                var offset = 0
                while (offset < prefix.size) {
                    val count = input.read(prefix, offset, prefix.size - offset)
                    if (count < 0) throw IllegalStateException("Checkpoint header is incomplete.")
                    offset += count
                }
                return prefix
            }
        }

        private fun readLegacyConfig(file: File): EngineConfig {
            require(file.length() >= LEGACY_HEADER_SIZE.toLong()) {
                "Legacy checkpoint header is incomplete."
            }
            val bytes = readFilePrefix(file, LEGACY_HEADER_SIZE)
            val header = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            require(
                header.get() == MAGIC_0 && header.get() == MAGIC_1 &&
                    header.get() == MAGIC_2 && header.get() == MAGIC_3
            ) { "Invalid ARG3 checkpoint magic." }
            require(header.int == LEGACY_CHECKPOINT_VERSION) {
                "Unsupported ARG3 checkpoint version."
            }

            val savedDModel = header.int
            val savedNHeads = header.int
            val savedDFf = header.int
            val savedMaxSeq = header.int
            val savedLoraRank = header.int
            val savedVocabSize = header.int
            val savedVocabCap = header.int
            header.int // optimizer step
            header.int // reserved
            require(savedVocabSize > 0) {
                "ARG3 vocabulary size is invalid."
            }
            require(savedVocabCap >= savedVocabSize) {
                "ARG3 vocabulary capacity is invalid."
            }
            return EngineConfig(
                dModel = savedDModel,
                nHeads = savedNHeads,
                dFf = savedDFf,
                maxSeqLen = savedMaxSeq,
                loraRank = savedLoraRank,
                isFullTrainingEnabled = true,
                tokenizerConfig = TokenizerConfig(
                    algorithm = ArgusTokenizerAlgorithm.GREEDY_LONGEST_MATCH_V1
                ),
                initialVocabularyCapacity = savedVocabCap,
                gradientAccumulationEnabled = false
            )
        }

        private fun readArg4Config(file: File): EngineConfig {
            require(file.length() >= ARG4_HEADER_SIZE.toLong()) {
                "ARG4 checkpoint header is incomplete."
            }
            val bytes = readFilePrefix(file, ARG4_HEADER_SIZE)
            val header = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            require(
                header.get() == MAGIC_0 && header.get() == MAGIC_1 &&
                    header.get() == MAGIC_2 && header.get() == MAGIC_4
            ) { "Invalid ARG4 checkpoint magic." }
            require(header.int == ARG4_CHECKPOINT_VERSION) {
                "Unsupported ARG4 checkpoint version."
            }
            require(header.int == ARG4_HEADER_SIZE) {
                "Invalid ARG4 checkpoint header size."
            }

            val architectureSchemaVersion = header.int
            val architectureIdLength = header.int
            val layerCount = header.int
            val savedDModel = header.int
            val savedNHeads = header.int
            val savedDFf = header.int
            val savedMaxSeq = header.int
            val savedLoraRank = header.int
            val savedVocabSize = header.int
            val savedVocabCap = header.int
            header.int // optimizer step
            val tokenizerSchemaVersion = header.int
            val tokenizerAlgorithmId = header.int
            val tokenizerMaxVocab = header.int
            val tokenizerMaxTokenLength = header.int
            val vocabularyEntryCount = header.int
            val tensorCount = header.int
            val payloadBytes = header.long
            header.int // payload checksum
            val flags = header.int

            require(architectureSchemaVersion == ArgusArchitectureConfig.CURRENT_SCHEMA_VERSION) {
                "Unsupported Argus architecture schema."
            }
            require(architectureIdLength in 1..512) {
                "ARG4 architecture ID length is invalid."
            }
            require(layerCount in 1..32) {
                "ARG4 layer count is invalid: $layerCount."
            }
            require(savedVocabSize > 0) {
                "ARG4 vocabulary size is invalid."
            }
            require(savedVocabCap >= savedVocabSize) {
                "ARG4 vocabulary capacity is invalid."
            }
            require(vocabularyEntryCount in 1..tokenizerMaxVocab) {
                "ARG4 vocabulary entry count is invalid."
            }
            require(tensorCount in 1..2048) {
                "ARG4 tensor count is invalid: $tensorCount."
            }
            require(payloadBytes >= architectureIdLength.toLong()) {
                "ARG4 payload length is invalid."
            }
            require(flags == 0) {
                "ARG4 checkpoint uses unsupported architecture flags."
            }

            val tokenizerConfig = TokenizerConfig(
                schemaVersion = tokenizerSchemaVersion,
                algorithm = ArgusTokenizerAlgorithm.fromCheckpointId(tokenizerAlgorithmId),
                maxVocabularySize = tokenizerMaxVocab,
                maxTokenLength = tokenizerMaxTokenLength
            )
            return EngineConfig(
                dModel = savedDModel,
                nHeads = savedNHeads,
                dFf = savedDFf,
                maxSeqLen = savedMaxSeq,
                loraRank = savedLoraRank,
                layerCount = layerCount,
                isFullTrainingEnabled = true,
                tokenizerConfig = tokenizerConfig,
                initialVocabularyCapacity = savedVocabCap,
                gradientAccumulationEnabled = false
            )
        }

        private fun readFilePrefix(file: File, size: Int): ByteArray {
            FileInputStream(file).use { input ->
                val bytes = ByteArray(size)
                var offset = 0
                while (offset < bytes.size) {
                    val count = input.read(bytes, offset, bytes.size - offset)
                    if (count < 0) throw IllegalStateException("Checkpoint header is incomplete.")
                    offset += count
                }
                return bytes
            }
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
        val stdLora = 1.0f / sqrt(loraRank.toFloat())
        val stdMlp = 1.0f / sqrt(dModel.toFloat())

        for (b in blocks) {
            initFlatMatrix(b.wQ, dModel, dModel, stdAttn)
            initFlatMatrix(b.wK, dModel, dModel, stdAttn)
            initFlatMatrix(b.wV, dModel, dModel, stdAttn)
            initFlatMatrix(b.wO, dModel, dModel, stdAttn)

            initFlatMatrix(b.loraQa, dModel, loraRank, stdLora)
            b.loraQb.fill(0f)
            initFlatMatrix(b.loraVa, dModel, loraRank, stdLora)
            b.loraVb.fill(0f)

            initFlatMatrix(b.wGate, dModel, dFf, stdMlp)
            initFlatMatrix(b.wUp, dModel, dFf, stdMlp)
            initFlatMatrix(b.wDown, dFf, dModel, 1.0f / sqrt(dFf.toFloat()))
        }

        val stdHead = 1.0f / sqrt(dModel.toFloat())
        for (i in wHead.indices) wHead[i] = (Random.nextFloat() * 2f - 1f) * stdHead
    }

    private fun initFlatMatrix(mat: Float16ParameterArray, inDim: Int, outDim: Int, std: Float) {
        val size = inDim * outDim
        for (i in 0 until size) {
            mat[i] = (Random.nextFloat() * 2f - 1f) * std
        }
    }

    /**
     * Architecture metadata exposed by the active engine.
     */
    val architectureConfig: ArgusArchitectureConfig
        get() = config.toArgusArchitectureConfig(
            vocabularySize = vocabCap,
            layerCount = layerCount
        )

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
        ((dModel.toLong() * loraRank + loraRank.toLong() * dModel) * 2L) *
            Short.SIZE_BYTES.toLong() * layerCount.toLong()

    // =====================================================================
    // Vocab Capacity Management
    // =====================================================================
    private fun ensureVocabCapacity(requiredSize: Int) {
        if (requiredSize <= vocabCap) return
        val newCap = max(requiredSize + 64, vocabCap * 2)
        resizeVocabCapacityTo(newCap)
    }

    /**
     * Learns tokenizer merges from a document sample before its first neural update.
     * Existing token IDs and all existing tensor rows are retained when the vocabulary grows.
     */
    fun trainTokenizerFromCorpus(corpus: List<String>): TokenizerTrainingResult {
        val result = synchronized(modelLock) {
            val previousVocabularySize = tokenizer.vocabSize
            val usesLearnedBpe = config.tokenizerConfig.algorithm.usesBpeMerges
            val previousMergeCount = if (usesLearnedBpe) tokenizer.mergeCount else 0
            val mergeBudget = when {
                dModel < 64 -> 32
                dModel < 96 -> 64
                else -> 96
            }
            val updatedVocabularySize = tokenizer.trainBpe(
                corpus = corpus,
                maxNewMerges = mergeBudget
            )
            ensureVocabCapacity(updatedVocabularySize)
            TokenizerTrainingResult(
                previousVocabularySize = previousVocabularySize,
                vocabularySize = tokenizer.vocabSize,
                tokensAdded = (tokenizer.vocabSize - previousVocabularySize).coerceAtLeast(0),
                previousMergeCount = previousMergeCount,
                mergeCount = if (usesLearnedBpe) tokenizer.mergeCount else 0
            )
        }

        if (result.tokensAdded > 0 || result.mergeCount > result.previousMergeCount) {
            if (!saveCheckpoint()) {
                Log.w(TAG, "Tokenizer vocabulary changed but the updated checkpoint could not be saved.")
            }
        }
        return result
    }

    private fun resizeVocabCapacityTo(targetCap: Int) {
        if (targetCap <= 0 || targetCap == vocabCap) return
        requireSafeVocabularyCapacity(targetCap)
        val targetElements = Math.multiplyExact(targetCap, dModel)
        val newEmbed = Float16ParameterArray(targetElements)
        val newHead = Float16ParameterArray(targetElements)
        val newMEmbed = FloatArray(targetElements)
        val newVEmbed = FloatArray(targetElements)
        val newMHead = FloatArray(targetElements)
        val newVHead = FloatArray(targetElements)

        newEmbed.copyFrom(wEmbed, 0, 0, min(wEmbed.size, newEmbed.size))
        newHead.copyFrom(wHead, 0, 0, min(wHead.size, newHead.size))
        System.arraycopy(mEmbed, 0, newMEmbed, 0, min(mEmbed.size, newMEmbed.size))
        System.arraycopy(vEmbed, 0, newVEmbed, 0, min(vEmbed.size, newVEmbed.size))
        System.arraycopy(mHead, 0, newMHead, 0, min(mHead.size, newMHead.size))
        System.arraycopy(vHead, 0, newVHead, 0, min(vHead.size, newVHead.size))

        if (targetCap > vocabCap) {
            val stdEmbed = 1.0f / sqrt(dModel.toFloat())
            for (i in (vocabCap * dModel) until targetElements) {
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
        synchronizeVulkanModelTensorResidency("vocabulary-capacity replacement")
    }

    // =====================================================================
    // Generation (Deterministic inference with frozen vocabulary)
    // =====================================================================
    suspend fun generate(
        prompt: String,
        maxNewTokens: Int = 60,
        temperature: Float = 0.7f,
        includeReasoningMarkers: Boolean = false,
        reasoningMode: Boolean = false
    ): String = withContext(Dispatchers.Default) {
        val generationContext = currentCoroutineContext()
        generationContext.ensureActive()
        synchronized(modelLock) {
            generationContext.ensureActive()
            val formattedPrompt = formatGenerationPrompt(prompt, reasoningMode)

            // PENTING: autoRegister = false saat inference agar kamus token stabil
            val rawTokens = tokenizer.encode(formattedPrompt, autoRegister = false)
            generationContext.ensureActive()
            if (rawTokens.isEmpty()) return@withContext "Input kosong."

            val currentVocab = tokenizer.vocabSize
            val maxPromptLen = maxSeqLen - 1
            val effective = if (rawTokens.size > maxPromptLen) rawTokens.takeLast(maxPromptLen)
            else rawTokens
            val currentTokens = effective.toMutableList()
            val generatedTokens = mutableListOf<Int>()

            val kCache = Array(layerCount) { FloatArray(maxSeqLen * dModel) }
            val vCache = Array(layerCount) { FloatArray(maxSeqLen * dModel) }

            val initialLen = currentTokens.size
            val hidden = FloatArray(dModel)
            val q = FloatArray(dModel)
            val k = FloatArray(dModel)
            val v = FloatArray(dModel)
            val loraMid = FloatArray(loraRank)
            val attnOut = FloatArray(dModel)
            val projBuf = FloatArray(dModel)
            val h1 = FloatArray(dModel)
            val gate = FloatArray(dFf)
            val up = FloatArray(dFf)
            val act = FloatArray(dFf)
            val down = FloatArray(dModel)
            val h2 = FloatArray(dModel)
            val logits = FloatArray(currentVocab)

            val embedScale = sqrt(dModel.toFloat())

            // Prefill: K,V untuk semua posisi prompt dan aktivasi multi-layer
            for (pos in 0 until initialLen) {
                generationContext.ensureActive()
                computeTokenEmbedAndPos(currentTokens[pos], pos, embedScale, hidden)
                for (layer in 0 until layerCount) {
                    generationContext.ensureActive()
                    val b = blocks[layer]
                    computeLinear(hidden, b.wK, dModel, dModel, k)
                    System.arraycopy(k, 0, kCache[layer], pos * dModel, dModel)
                    computeLinearWithLora(hidden, b.wV, b.loraVa, b.loraVb, dModel, dModel, loraRank, loraMid, v)
                    System.arraycopy(v, 0, vCache[layer], pos * dModel, dModel)

                    computeLinearWithLora(hidden, b.wQ, b.loraQa, b.loraQb, dModel, dModel, loraRank, loraMid, q)

                    attnOut.fill(0f)
                    val seqSoFar = pos + 1
                    val scores = FloatArray(seqSoFar)

                    for (h in 0 until nHeads) {
                        generationContext.ensureActive()
                        val hOffset = h * headDim
                        var maxScore = -Float.MAX_VALUE
                        for (j in 0 until seqSoFar) {
                            if ((j and 0x3F) == 0) generationContext.ensureActive()
                            var dot = 0f
                            val kOffset = j * dModel + hOffset
                            for (d in 0 until headDim) dot += q[hOffset + d] * kCache[layer][kOffset + d]
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
                                if ((j and 0x3F) == 0) generationContext.ensureActive()
                                val vOffset = j * dModel + hOffset
                                weightedV += scores[j] * vCache[layer][vOffset + d]
                            }
                            attnOut[hOffset + d] = weightedV
                        }
                    }

                    computeLinear(attnOut, b.wO, dModel, dModel, projBuf)
                    for (d in 0 until dModel) h1[d] = hidden[d] + projBuf[d]
                    layerNormInPlace(h1, dModel)

                    computeLinear(h1, b.wGate, dModel, dFf, gate)
                    computeLinear(h1, b.wUp, dModel, dFf, up)
                    for (d in 0 until dFf) {
                        val sig = 1.0f / (1.0f + exp(-gate[d]))
                        act[d] = (gate[d] * sig) * up[d]
                    }
                    computeLinear(act, b.wDown, dFf, dModel, down)
                    for (d in 0 until dModel) h2[d] = h1[d] + down[d]
                    layerNormInPlace(h2, dModel)
                    System.arraycopy(h2, 0, hidden, 0, dModel)
                }
            }

            // Decode loop
            var currentPos = initialLen - 1
            for (step in 0 until maxNewTokens) {
                generationContext.ensureActive()
                if (currentPos >= maxSeqLen - 1) break

                val lastTok = currentTokens[currentPos]
                computeTokenEmbedAndPos(lastTok, currentPos, embedScale, hidden)

                for (layer in 0 until layerCount) {
                    generationContext.ensureActive()
                    val b = blocks[layer]
                    if (step > 0) {
                        computeLinear(hidden, b.wK, dModel, dModel, k)
                        System.arraycopy(k, 0, kCache[layer], currentPos * dModel, dModel)
                        computeLinearWithLora(hidden, b.wV, b.loraVa, b.loraVb, dModel, dModel, loraRank, loraMid, v)
                        System.arraycopy(v, 0, vCache[layer], currentPos * dModel, dModel)
                    }

                    computeLinearWithLora(hidden, b.wQ, b.loraQa, b.loraQb, dModel, dModel, loraRank, loraMid, q)

                    attnOut.fill(0f)
                    val seqSoFar = currentPos + 1
                    val scores = FloatArray(seqSoFar)

                    for (h in 0 until nHeads) {
                        generationContext.ensureActive()
                        val hOffset = h * headDim
                        var maxScore = -Float.MAX_VALUE
                        for (j in 0 until seqSoFar) {
                            if ((j and 0x3F) == 0) generationContext.ensureActive()
                            var dot = 0f
                            val kOffset = j * dModel + hOffset
                            for (d in 0 until headDim) dot += q[hOffset + d] * kCache[layer][kOffset + d]
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
                                if ((j and 0x3F) == 0) generationContext.ensureActive()
                                val vOffset = j * dModel + hOffset
                                weightedV += scores[j] * vCache[layer][vOffset + d]
                            }
                            attnOut[hOffset + d] = weightedV
                        }
                    }

                    computeLinear(attnOut, b.wO, dModel, dModel, projBuf)
                    for (d in 0 until dModel) h1[d] = hidden[d] + projBuf[d]
                    layerNormInPlace(h1, dModel)

                    computeLinear(h1, b.wGate, dModel, dFf, gate)
                    computeLinear(h1, b.wUp, dModel, dFf, up)
                    for (d in 0 until dFf) {
                        val sig = 1.0f / (1.0f + exp(-gate[d]))
                        act[d] = (gate[d] * sig) * up[d]
                    }
                    computeLinear(act, b.wDown, dFf, dModel, down)
                    for (d in 0 until dModel) h2[d] = h1[d] + down[d]
                    layerNormInPlace(h2, dModel)
                    System.arraycopy(h2, 0, hidden, 0, dModel)
                }

                for (vIdx in 0 until currentVocab) {
                    if ((vIdx and 0x3F) == 0) generationContext.ensureActive()
                    var sum = 0f
                    val wOff = vIdx * dModel
                    for (d in 0 until dModel) sum += hidden[d] * wHead[wOff + d]
                    logits[vIdx] = sum
                }

                val nextToken = sampleLogits(logits, currentVocab, temperature)
                if (nextToken == Tokenizer.EOS_ID) break

                generatedTokens.add(nextToken)
                currentTokens.add(nextToken)
                currentPos++
            }

            generationContext.ensureActive()
            val decoded = tokenizer.decode(
                generatedTokens,
                keepReasoningTokens = includeReasoningMarkers || reasoningMode
            )
            if (decoded.isBlank()) "..." else decoded
        }
    }

    /**
     * Generates a response beginning at the trained THINK prefix. The returned text
     * retains <THINK> and </THINK> so callers can distinguish the reasoning trace
     * from the final assistant answer.
     */
    suspend fun generateReasoning(
        prompt: String,
        maxNewTokens: Int = 120,
        temperature: Float = 0.7f
    ): String = generate(
        prompt = prompt,
        maxNewTokens = maxNewTokens,
        temperature = temperature,
        includeReasoningMarkers = true,
        reasoningMode = true
    )

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
        var plusParameterValue = originalValue
        var minusParameterValue = originalValue
        val probeEpsilon = max(epsilon, 0.05f)
        try {
            parameter[resolvedIndex] = originalValue + probeEpsilon
            plusParameterValue = parameter[resolvedIndex]
            plusLoss = trainSingleWindow(tokens, currentVocab, updateParameters = false)

            parameter[resolvedIndex] = originalValue - probeEpsilon
            minusParameterValue = parameter[resolvedIndex]
            minusLoss = trainSingleWindow(tokens, currentVocab, updateParameters = false)
        } finally {
            parameter[resolvedIndex] = originalValue
        }

        check(plusLoss.isFinite() && minusLoss.isFinite()) {
            "Finite-difference loss evaluation returned a non-finite value."
        }
        val representedDelta = plusParameterValue - minusParameterValue
        check(representedDelta != 0f) {
            "FP16 parameter quantization is too coarse for this finite-difference probe."
        }
        val numericalGradient = (plusLoss - minusLoss) / representedDelta
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

    private fun gradientParameterArray(parameterName: String): Float16ParameterArray {
        if (parameterName == "wHead") return wHead
        if (parameterName == "wEmbed") return wEmbed

        val (targetLayer, simpleName) = if (parameterName.startsWith("layers.")) {
            val parts = parameterName.split(".", limit = 3)
            val layerIdx = parts[1].toIntOrNull()
                ?: throw IllegalArgumentException("Invalid layer index in parameter: $parameterName")
            require(layerIdx in 0 until layerCount) {
                "Layer index $layerIdx out of range for $layerCount layers."
            }
            layerIdx to parts[2]
        } else {
            0 to parameterName
        }

        val b = blocks[targetLayer]
        return when (simpleName) {
            "wDown", "feed_forward.down.weight" -> b.wDown
            "wUp", "feed_forward.up.weight" -> b.wUp
            "wGate", "feed_forward.gate.weight" -> b.wGate
            "wO", "attention.output.weight" -> b.wO
            "wQ", "attention.query.weight" -> b.wQ
            "wK", "attention.key.weight" -> b.wK
            "wV", "attention.value.weight" -> b.wV
            "loraQa", "adapter.q.a" -> b.loraQa
            "loraQb", "adapter.q.b" -> b.loraQb
            "loraVa", "adapter.v.a" -> b.loraVa
            "loraVb", "adapter.v.b" -> b.loraVb
            else -> throw IllegalArgumentException("Unsupported gradient-check parameter: $parameterName")
        }
    }

    fun trainStep(text: String): Float = trainStepDetailed(text).loss

    fun trainStepDetailed(text: String): NeuralTrainingStepResult =
        trainBatchDetailed(listOf(text)).samples.single()

    /**
     * Performs a serial micro-batch of independent sequences with one shared AdamW update.
     * Activations are released between samples; only the accumulated parameter gradients
     * remain resident. The accepted batch size is strictly bounded to 1..8.
     */
    fun trainBatchDetailed(texts: List<String>): NeuralTrainingBatchResult {
        require(texts.size in 1..8) {
            "Training batch size must be between 1 and 8."
        }
        if (!config.gradientAccumulationEnabled) {
            require(texts.size == 1) {
                "This legacy checkpoint requires single-sample compatibility training."
            }
            return trainSingleSampleSequential(texts.single())
        }

        var shouldSaveCheckpoint = false
        val result = synchronized(modelLock) {
            val initialAdamStep = adamStep
            ensureVocabCapacity(tokenizer.vocabSize)
            val accumulator = createBatchGradientAccumulator()
            val sampleResults = ArrayList<NeuralTrainingStepResult>(texts.size)

            try {
                for (text in texts) {
                    val tokens = tokenizer.encode(text, autoRegister = false)
                    if (tokens.size < 2) {
                        sampleResults += NeuralTrainingStepResult(
                            loss = Float.NaN,
                            inputTokenCount = tokens.size,
                            predictionTargetCount = 0,
                            contextWindowCount = 0,
                            optimizerUpdateCount = 0,
                            vocabularySize = tokenizer.vocabSize
                        )
                        continue
                    }

                    ensureVocabCapacity(tokenizer.vocabSize)
                    val currentVocab = tokenizer.vocabSize
                    var contextWindowCount = 0
                    var predictionTargetCount = 0
                    var weightedLoss = 0f
                    var trainedTargets = 0

                    if (tokens.size <= maxSeqLen) {
                        val windowLoss = trainSingleWindow(
                            seq = tokens,
                            currentVocab = currentVocab,
                            gradientAccumulator = accumulator
                        )
                        if (windowLoss.isFinite()) {
                            contextWindowCount = 1
                            predictionTargetCount = tokens.size - 1
                            trainedTargets = tokens.size - 1
                            weightedLoss = windowLoss * (tokens.size - 1)
                        }
                    } else {
                        val overlapTokens = min(24, maxSeqLen - 1)
                        val stride = (maxSeqLen - overlapTokens).coerceAtLeast(1)
                        var start = 0

                        while (start < tokens.size - 1) {
                            val end = min(start + maxSeqLen, tokens.size)
                            val windowSequence = tokens.subList(start, end)
                            if (windowSequence.size >= 2) {
                                val firstTargetIndex =
                                    if (start == 0 || overlapTokens == 0) 0 else overlapTokens - 1
                                val targetCount =
                                    (windowSequence.size - 1 - firstTargetIndex).coerceAtLeast(0)
                                if (targetCount > 0) {
                                    val windowLoss = trainSingleWindow(
                                        seq = windowSequence,
                                        currentVocab = currentVocab,
                                        firstTargetIndex = firstTargetIndex,
                                        gradientAccumulator = accumulator
                                    )
                                    if (windowLoss.isFinite()) {
                                        contextWindowCount++
                                        weightedLoss += windowLoss * targetCount
                                        trainedTargets += targetCount
                                        predictionTargetCount += targetCount
                                    }
                                }
                            }
                            if (end >= tokens.size) break
                            start += stride
                        }
                    }

                    sampleResults += NeuralTrainingStepResult(
                        loss = if (trainedTargets > 0) {
                            weightedLoss / trainedTargets
                        } else {
                            Float.NaN
                        },
                        inputTokenCount = tokens.size,
                        predictionTargetCount = predictionTargetCount,
                        contextWindowCount = contextWindowCount,
                        optimizerUpdateCount = 0,
                        vocabularySize = currentVocab
                    )
                }

                if (accumulator.targetCount > 0) {
                    applyAccumulatedGradients(accumulator)
                }
            } finally {
                val remainingResident = accumulator.resident
                if (remainingResident != null) {
                    VulkanTrainingBackend.discardResidentGradientAccumulator(remainingResident)
                    accumulator.resident = null
                }
            }

            val optimizerUpdates = (adamStep - initialAdamStep).coerceAtLeast(0)
            if (adamStep / CHECKPOINT_INTERVAL > initialAdamStep / CHECKPOINT_INTERVAL) {
                shouldSaveCheckpoint = true
            }

            val firstTrainableSample = sampleResults.indexOfFirst {
                it.loss.isFinite() && it.predictionTargetCount > 0
            }
            val reportedSamples = sampleResults.mapIndexed { index, sample ->
                sample.copy(
                    optimizerUpdateCount = if (index == firstTrainableSample) {
                        optimizerUpdates
                    } else {
                        0
                    }
                )
            }

            NeuralTrainingBatchResult(
                samples = reportedSamples,
                optimizerUpdateCount = optimizerUpdates,
                totalPredictionTargetCount = accumulator.targetCount
            )
        }

        if (shouldSaveCheckpoint) {
            runCatching { saveCheckpoint() }
                .onFailure { Log.e(TAG, "Auto-checkpoint save failed", it) }
        }
        return result
    }

    /**
     * Compatibility path for older checkpoints whose existing model state fits the
     * legacy 16-byte floor but cannot safely reserve an additional accumulation tensor.
     * It retains the original per-window Adam update behavior and is restricted to batch 1.
     */
    private fun trainSingleSampleSequential(text: String): NeuralTrainingBatchResult {
        var shouldSaveCheckpoint = false
        val result = synchronized(modelLock) {
            val initialAdamStep = adamStep
            val tokens = tokenizer.encode(text, autoRegister = false)
            if (tokens.size < 2) {
                return@synchronized NeuralTrainingBatchResult(
                    samples = listOf(
                        NeuralTrainingStepResult(
                            loss = Float.NaN,
                            inputTokenCount = tokens.size,
                            predictionTargetCount = 0,
                            contextWindowCount = 0,
                            optimizerUpdateCount = 0,
                            vocabularySize = tokenizer.vocabSize
                        )
                    ),
                    optimizerUpdateCount = 0,
                    totalPredictionTargetCount = 0
                )
            }

            ensureVocabCapacity(tokenizer.vocabSize)
            val currentVocab = tokenizer.vocabSize
            var contextWindowCount = 0
            var predictionTargetCount = 0
            var weightedLoss = 0f
            var trainedTargets = 0

            if (tokens.size <= maxSeqLen) {
                val loss = trainSingleWindow(tokens, currentVocab)
                if (loss.isFinite()) {
                    contextWindowCount = 1
                    predictionTargetCount = tokens.size - 1
                    trainedTargets = tokens.size - 1
                    weightedLoss = loss * (tokens.size - 1)
                }
            } else {
                val overlapTokens = min(24, maxSeqLen - 1)
                val stride = (maxSeqLen - overlapTokens).coerceAtLeast(1)
                var start = 0
                while (start < tokens.size - 1) {
                    val end = min(start + maxSeqLen, tokens.size)
                    val windowSequence = tokens.subList(start, end)
                    if (windowSequence.size >= 2) {
                        val firstTargetIndex =
                            if (start == 0 || overlapTokens == 0) 0 else overlapTokens - 1
                        val targetCount =
                            (windowSequence.size - 1 - firstTargetIndex).coerceAtLeast(0)
                        if (targetCount > 0) {
                            val loss = trainSingleWindow(
                                seq = windowSequence,
                                currentVocab = currentVocab,
                                firstTargetIndex = firstTargetIndex
                            )
                            if (loss.isFinite()) {
                                contextWindowCount++
                                predictionTargetCount += targetCount
                                trainedTargets += targetCount
                                weightedLoss += loss * targetCount
                            }
                        }
                    }
                    if (end >= tokens.size) break
                    start += stride
                }
            }

            if (adamStep / CHECKPOINT_INTERVAL > initialAdamStep / CHECKPOINT_INTERVAL) {
                shouldSaveCheckpoint = true
            }
            val optimizerUpdates = (adamStep - initialAdamStep).coerceAtLeast(0)
            val stepResult = NeuralTrainingStepResult(
                loss = if (trainedTargets > 0) weightedLoss / trainedTargets else Float.NaN,
                inputTokenCount = tokens.size,
                predictionTargetCount = predictionTargetCount,
                contextWindowCount = contextWindowCount,
                optimizerUpdateCount = optimizerUpdates,
                vocabularySize = currentVocab
            )
            NeuralTrainingBatchResult(
                samples = listOf(stepResult),
                optimizerUpdateCount = optimizerUpdates,
                totalPredictionTargetCount = predictionTargetCount
            )
        }

        if (shouldSaveCheckpoint) {
            runCatching { saveCheckpoint() }
                .onFailure { Log.e(TAG, "Auto-checkpoint save failed", it) }
        }
        return result
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
        gradientProbe: GradientProbe? = null,
        gradientAccumulator: BatchGradientAccumulator? = null
    ): Float {
        val seqLen = seq.size
        if (seqLen < 2) return Float.NaN

        if (updateParameters && gradientAccumulator == null) adamStep++
        val lr = if (updateParameters && gradientAccumulator == null) {
            baseLearningRate / (1f + 0.0002f * adamStep)
        } else {
            0f
        }
        val embedScale = sqrt(dModel.toFloat())

        // ============================= FORWARD =============================
        val x = Array(seqLen) { FloatArray(dModel) }
        val logits = Array(seqLen) { FloatArray(currentVocab) }

        for (t in 0 until seqLen) {
            computeTokenEmbedAndPos(seq[t], t, embedScale, x[t])
        }

        class LayerForward(seqLen: Int, dModel: Int, dFf: Int, loraRank: Int, nHeads: Int) {
            val q = Array(seqLen) { FloatArray(dModel) }
            val k = Array(seqLen) { FloatArray(dModel) }
            val v = Array(seqLen) { FloatArray(dModel) }
            val mQ = Array(seqLen) { FloatArray(loraRank) }
            val mV = Array(seqLen) { FloatArray(loraRank) }
            val attnWeights = Array(nHeads) { Array(seqLen) { FloatArray(seqLen) } }
            val attnOut = Array(seqLen) { FloatArray(dModel) }
            val h1PreLN = Array(seqLen) { FloatArray(dModel) }
            val h1 = Array(seqLen) { FloatArray(dModel) }
            val gate = Array(seqLen) { FloatArray(dFf) }
            val up = Array(seqLen) { FloatArray(dFf) }
            val act = Array(seqLen) { FloatArray(dFf) }
            val h2PreLN = Array(seqLen) { FloatArray(dModel) }
            val h2 = Array(seqLen) { FloatArray(dModel) }
        }

        val layerForwards = Array(layerCount) {
            LayerForward(seqLen, dModel, dFf, loraRank, nHeads)
        }

        for (layer in 0 until layerCount) {
            val b = blocks[layer]
            val lf = layerForwards[layer]
            val layerInput = if (layer == 0) x else layerForwards[layer - 1].h2

            computeBatchedLinear(
                inputs = arrayOf(layerInput, layerInput, layerInput),
                weights = arrayOf(b.wQ, b.wK, b.wV),
                outputs = arrayOf(lf.q, lf.k, lf.v),
                inputColumns = dModel,
                outputColumns = dModel
            )

            for (t in 0 until seqLen) {
                addLoraProjection(layerInput[t], b.loraQa, b.loraQb, dModel, dModel, loraRank, lf.mQ[t], lf.q[t])
                addLoraProjection(layerInput[t], b.loraVa, b.loraVb, dModel, dModel, loraRank, lf.mV[t], lf.v[t])
            }


            val residentAttentionCompleted = VulkanTrainingBackend.tryApplyResidentAttention(
                queryRows = lf.q,
                keyRows = lf.k,
                valueRows = lf.v,
                residualRows = layerInput,
                outputWeights = b.wO,
                attentionWeights = lf.attnWeights,
                attentionOutputRows = lf.attnOut,
                projectedOutputRows = lf.h1PreLN,
                normalizedOutputRows = lf.h1,
                modelWidth = dModel,
                headCount = nHeads
            )
            if (!residentAttentionCompleted) {
                for (h in 0 until nHeads) {
                    val hOffset = h * headDim
                    for (i in 0 until seqLen) {
                        var maxScore = -Float.MAX_VALUE
                        for (j in 0..i) {
                            var dot = 0f
                            for (d in 0 until headDim) dot += lf.q[i][hOffset + d] * lf.k[j][hOffset + d]
                            val sc = dot * scale
                            lf.attnWeights[h][i][j] = sc
                            if (sc > maxScore) maxScore = sc
                        }
                        var sumExp = 0f
                        for (j in 0..i) {
                            val e = exp(lf.attnWeights[h][i][j] - maxScore)
                            lf.attnWeights[h][i][j] = e; sumExp += e
                        }
                        val invSum = 1.0f / max(sumExp, 1e-9f)
                        for (j in 0..i) lf.attnWeights[h][i][j] *= invSum
                        for (d in 0 until headDim) {
                            var weightedV = 0f
                            for (j in 0..i) weightedV += lf.attnWeights[h][i][j] * lf.v[j][hOffset + d]
                            lf.attnOut[i][hOffset + d] = weightedV
                        }
                    }
                }
    
    
                computeLinearBatch(lf.attnOut, b.wO, dModel, dModel, lf.h1PreLN)
                for (t in 0 until seqLen) {
                    for (d in 0 until dModel) lf.h1PreLN[t][d] += layerInput[t][d]
                }
            }
            if (!residentAttentionCompleted) {
                for (t in 0 until seqLen) {
                    System.arraycopy(lf.h1PreLN[t], 0, lf.h1[t], 0, dModel)
                    layerNormInPlace(lf.h1[t], dModel)
                }
            }

            val residentFeedForwardCompleted =
                VulkanTrainingBackend.tryApplyResidentFeedForward(
                    inputRows = lf.h1,
                    gateWeights = b.wGate,
                    upWeights = b.wUp,
                    downWeights = b.wDown,
                    gateRows = lf.gate,
                    upRows = lf.up,
                    activationRows = lf.act,
                    projectedOutputRows = lf.h2PreLN,
                    normalizedOutputRows = lf.h2
                )
            if (!residentFeedForwardCompleted) {
                computeLinearBatch(lf.h1, b.wGate, dModel, dFf, lf.gate)
                computeLinearBatch(lf.h1, b.wUp, dModel, dFf, lf.up)
                val residentSwiGluCompleted = VulkanTrainingBackend.tryApplyResidentSwiGlu(
                    gateRows = lf.gate,
                    upRows = lf.up,
                    outputRows = lf.act
                )
                if (!residentSwiGluCompleted) {
                    for (t in 0 until seqLen) {
                        for (d in 0 until dFf) {
                            val sigmoid = 1.0f / (1.0f + exp(-lf.gate[t][d]))
                            lf.act[t][d] = (lf.gate[t][d] * sigmoid) * lf.up[t][d]
                        }
                    }
                }

                computeLinearBatch(lf.act, b.wDown, dFf, dModel, lf.h2PreLN)
                for (t in 0 until seqLen) {
                    for (d in 0 until dModel) lf.h2PreLN[t][d] += lf.h1[t][d]
                    System.arraycopy(lf.h2PreLN[t], 0, lf.h2[t], 0, dModel)
                    layerNormInPlace(lf.h2[t], dModel)
                }
            }
        }

        val lastHidden = layerForwards[layerCount - 1].h2
        computeLinearBatch(
            inputs = lastHidden,
            weights = wHead,
            inputColumns = dModel,
            outputColumns = currentVocab,
            outputs = logits,
            transposeWeights = true
        )

        // ============================ BACKWARD ============================
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

        val gradWHead = FloatArray(currentVocab * dModel)
        val gradWEmbed = FloatArray(vocabCap * dModel)

        class LayerGradients(dModel: Int, dFf: Int, loraRank: Int) {
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
        }
        val layerGrads = Array(layerCount) { LayerGradients(dModel, dFf, loraRank) }

        val dLogits = Array(seqLen) { FloatArray(currentVocab) }

        // (1) Fused causal cross-entropy and output-head gradient.
        val targetRows = FloatArray(seqLen) { row ->
            if (row + 1 < seqLen) seq[row + 1].toFloat() else 0.0f
        }
        val stagedLogitGradient = FloatArray(Math.multiplyExact(seqLen, currentVocab))
        val rowLossStatistics = FloatArray(Math.multiplyExact(seqLen, 2))
        val crossEntropyOnGpu = VulkanTrainingBackend.isResidentBackwardAvailable &&
            VulkanTrainingBackend.tryRunResidentBackward(
                operation = VulkanTrainingBackend.BACKWARD_CROSS_ENTROPY,
                rows = seqLen,
                columns = currentVocab,
                sequenceLength = firstTargetIndex,
                inputs = arrayOf(flattenRows(logits), targetRows, null),
                outputs = arrayOf(stagedLogitGradient, rowLossStatistics)
            )
        if (crossEntropyOnGpu) {
            for (time in 0 until seqLen) {
                System.arraycopy(
                    stagedLogitGradient,
                    time * currentVocab,
                    dLogits[time],
                    0,
                    currentVocab
                )
                val statisticOffset = time * 2
                totalLoss += rowLossStatistics[statisticOffset]
                validSteps += rowLossStatistics[statisticOffset + 1].toInt()
            }
        } else {
            // Numerically equivalent fallback for unsupported or failed Vulkan dispatch.
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
        }

        val dL_dh2PostLN = Array(seqLen) { FloatArray(dModel) }
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
        val hiddenRows = flattenRows(lastHidden)
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

        var incomingGrad = dL_dh2PostLN

        // (2) Reverse Backpropagation through Transformer Blocks
        for (layer in layerCount - 1 downTo 0) {
            val b = blocks[layer]
            val lf = layerForwards[layer]
            val lg = layerGrads[layer]
            val layerInput = if (layer == 0) x else layerForwards[layer - 1].h2

            val dL_dh2PreLN = Array(seqLen) { FloatArray(dModel) }
            val dL_dh1PostLN = Array(seqLen) { FloatArray(dModel) }
            val dL_dh1PreLN = Array(seqLen) { FloatArray(dModel) }
            val dL_dattnOut = Array(seqLen) { FloatArray(dModel) }
            val dQ = Array(seqLen) { FloatArray(dModel) }
            val dK = Array(seqLen) { FloatArray(dModel) }
            val dV = Array(seqLen) { FloatArray(dModel) }

            layerNormBackwardBatch(
                incomingGradients = incomingGrad,
                preLayerNormRows = lf.h2PreLN,
                outputGradients = dL_dh2PreLN,
                rowCount = seqLen - 1
            )

            val dActRows = Array(seqLen) { FloatArray(dFf) }
            computeLinearBatch(
                inputs = dL_dh2PreLN,
                weights = b.wDown,
                inputColumns = dModel,
                outputColumns = dFf,
                outputs = dActRows,
                transposeWeights = true
            )

            val dUpRows = Array(seqLen) { FloatArray(dFf) }
            val dGateRows = Array(seqLen) { FloatArray(dFf) }
            val backwardRows = seqLen - 1
            val swigluElementCount = Math.multiplyExact(backwardRows, dFf)
            val stagedGateGradient = FloatArray(swigluElementCount)
            val stagedUpGradient = FloatArray(swigluElementCount)
            val swigluBackwardOnGpu = VulkanTrainingBackend.isResidentBackwardAvailable &&
                VulkanTrainingBackend.tryRunResidentBackward(
                    operation = VulkanTrainingBackend.BACKWARD_SWIGLU,
                    rows = backwardRows,
                    columns = dFf,
                    sequenceLength = 0,
                    inputs = arrayOf(
                        flattenRows(Array(backwardRows) { row -> lf.gate[row] }),
                        flattenRows(Array(backwardRows) { row -> lf.up[row] }),
                        flattenRows(Array(backwardRows) { row -> dActRows[row] })
                    ),
                    outputs = arrayOf(stagedGateGradient, stagedUpGradient)
                )
            if (swigluBackwardOnGpu) {
                for (row in 0 until backwardRows) {
                    System.arraycopy(stagedGateGradient, row * dFf, dGateRows[row], 0, dFf)
                    System.arraycopy(stagedUpGradient, row * dFf, dUpRows[row], 0, dFf)
                }
            } else {
                for (t in 0 until backwardRows) {
                    for (dimension in 0 until dFf) {
                        val sigmoid = 1.0f / (1.0f + exp(-lf.gate[t][dimension]))
                        val swish = lf.gate[t][dimension] * sigmoid
                        val dActValue = dActRows[t][dimension]
                        dUpRows[t][dimension] = dActValue * swish
                        val dSwish = dActValue * lf.up[t][dimension]
                        dGateRows[t][dimension] =
                            dSwish * (sigmoid * (1f + lf.gate[t][dimension] * (1f - sigmoid)))
                    }
                }
            }

            val actTransposed = transposeRows(lf.act)
            val gradWDownRows = Array(dFf) { FloatArray(dModel) }
            computeLinearBatch(
                inputs = actTransposed,
                weights = flattenRows(dL_dh2PreLN),
                inputColumns = seqLen,
                outputColumns = dModel,
                outputs = gradWDownRows
            )
            for (row in gradWDownRows.indices) {
                System.arraycopy(gradWDownRows[row], 0, lg.gradWDown, row * dModel, dModel)
            }

            val h1Transposed = transposeRows(lf.h1)
            val gradWUpRows = Array(dModel) { FloatArray(dFf) }
            computeLinearBatch(
                inputs = h1Transposed,
                weights = flattenRows(dUpRows),
                inputColumns = seqLen,
                outputColumns = dFf,
                outputs = gradWUpRows
            )
            for (row in gradWUpRows.indices) {
                System.arraycopy(gradWUpRows[row], 0, lg.gradWUp, row * dFf, dFf)
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
                System.arraycopy(gradWGateRows[row], 0, lg.gradWGate, row * dFf, dFf)
            }

            val dHiddenFromGate = Array(seqLen) { FloatArray(dModel) }
            computeLinearBatch(
                inputs = dGateRows,
                weights = b.wGate,
                inputColumns = dFf,
                outputColumns = dModel,
                outputs = dHiddenFromGate,
                transposeWeights = true
            )

            val dHiddenFromUp = Array(seqLen) { FloatArray(dModel) }
            computeLinearBatch(
                inputs = dUpRows,
                weights = b.wUp,
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
            }
            layerNormBackwardBatch(
                incomingGradients = dL_dh1PostLN,
                preLayerNormRows = lf.h1PreLN,
                outputGradients = dL_dh1PreLN,
                rowCount = seqLen - 1
            )

            val attentionBackwardRows = seqLen - 1
            val attentionOutputRows = Array(attentionBackwardRows) { row -> lf.attnOut[row] }
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
            System.arraycopy(flattenRows(gradWORows), 0, lg.gradWO, 0, lg.gradWO.size)

            val attentionInputGradientRows =
                Array(attentionBackwardRows) { FloatArray(dModel) }
            computeLinearBatch(
                inputs = outputProjectionGradientRows,
                weights = b.wO,
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
                    FloatArray(headDim) { dimension -> lf.q[time][headOffset + dimension] }
                }
            }
            val keyRowsByHead = Array(nHeads) { head ->
                val headOffset = head * headDim
                Array(seqLen) { time ->
                    FloatArray(headDim) { dimension -> lf.k[time][headOffset + dimension] }
                }
            }
            val valueRowsByHead = Array(nHeads) { head ->
                val headOffset = head * headDim
                Array(seqLen) { time ->
                    FloatArray(headDim) { dimension -> lf.v[time][headOffset + dimension] }
                }
            }

            val transposedAttentionByHead = Array(nHeads) { head ->
                Array(seqLen) { keyIndex ->
                    FloatArray(seqLen) { queryIndex ->
                        if (queryIndex >= keyIndex) {
                            lf.attnWeights[head][queryIndex][keyIndex]
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
            val softmaxBackwardRows = Math.multiplyExact(nHeads, attentionBackwardRows)
            val flattenedProbabilities = FloatArray(
                Math.multiplyExact(softmaxBackwardRows, seqLen)
            )
            val flattenedProbabilityGradients = FloatArray(flattenedProbabilities.size)
            for (head in 0 until nHeads) {
                for (query in 0 until attentionBackwardRows) {
                    val flatRow = head * attentionBackwardRows + query
                    System.arraycopy(
                        lf.attnWeights[head][query],
                        0,
                        flattenedProbabilities,
                        flatRow * seqLen,
                        seqLen
                    )
                    System.arraycopy(
                        dWeightsByHead[head][query],
                        0,
                        flattenedProbabilityGradients,
                        flatRow * seqLen,
                        seqLen
                    )
                }
            }
            val flattenedScoreGradients = FloatArray(flattenedProbabilities.size)
            val softmaxBackwardOnGpu = VulkanTrainingBackend.isResidentBackwardAvailable &&
                VulkanTrainingBackend.tryRunResidentBackward(
                    operation = VulkanTrainingBackend.BACKWARD_CAUSAL_SOFTMAX,
                    rows = softmaxBackwardRows,
                    columns = seqLen,
                    sequenceLength = seqLen,
                    inputs = arrayOf(
                        flattenedProbabilities,
                        flattenedProbabilityGradients,
                        null
                    ),
                    outputs = arrayOf(flattenedScoreGradients),
                    scale = scale
                )
            if (softmaxBackwardOnGpu) {
                for (head in 0 until nHeads) {
                    for (query in 0 until attentionBackwardRows) {
                        val flatRow = (head * attentionBackwardRows + query) * seqLen
                        System.arraycopy(
                            flattenedScoreGradients,
                            flatRow,
                            dScoreByHead[head][query],
                            0,
                            seqLen
                        )
                    }
                }
            } else {
                for (head in 0 until nHeads) {
                    for (queryIndex in 0 until seqLen) {
                        var weightedGradient = 0f
                        for (keyIndex in 0..queryIndex) {
                            weightedGradient +=
                                lf.attnWeights[head][queryIndex][keyIndex] *
                                dWeightsByHead[head][queryIndex][keyIndex]
                        }
                        for (keyIndex in 0..queryIndex) {
                            dScoreByHead[head][queryIndex][keyIndex] =
                                lf.attnWeights[head][queryIndex][keyIndex] *
                                (dWeightsByHead[head][queryIndex][keyIndex] - weightedGradient) *
                                scale
                        }
                    }
                }
            }

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

            val inputRowsForGradient = Array(attentionBackwardRows) { row -> layerInput[row] }
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
            System.arraycopy(flattenRows(qkvGradientOutputs[0]), 0, lg.gradWQ, 0, lg.gradWQ.size)
            System.arraycopy(flattenRows(qkvGradientOutputs[1]), 0, lg.gradWK, 0, lg.gradWK.size)
            System.arraycopy(flattenRows(qkvGradientOutputs[2]), 0, lg.gradWV, 0, lg.gradWV.size)

            // LoRA B gradients: M^T × dProjection.
            val queryMidRows = Array(attentionBackwardRows) { row -> lf.mQ[row] }
            val valueMidRows = Array(attentionBackwardRows) { row -> lf.mV[row] }
            val queryMidGradientB = Array(loraRank) { FloatArray(dModel) }
            val valueMidGradientB = Array(loraRank) { FloatArray(dModel) }
            computeLinearBatch(
                inputs = transposeRows(queryMidRows),
                weights = flattenRows(queryGradientRows),
                inputColumns = attentionBackwardRows,
                outputColumns = dModel,
                outputs = queryMidGradientB
            )
            computeLinearBatch(
                inputs = transposeRows(valueMidRows),
                weights = flattenRows(valueGradientRows),
                inputColumns = attentionBackwardRows,
                outputColumns = dModel,
                outputs = valueMidGradientB
            )
            System.arraycopy(flattenRows(queryMidGradientB), 0, lg.gradLoraQb, 0, lg.gradLoraQb.size)
            System.arraycopy(flattenRows(valueMidGradientB), 0, lg.gradLoraVb, 0, lg.gradLoraVb.size)

            // LoRA A gradients and the low-rank contribution to dLayerInput.
            val queryRankGradients = Array(attentionBackwardRows) { FloatArray(loraRank) }
            val valueRankGradients = Array(attentionBackwardRows) { FloatArray(loraRank) }
            computeLinearBatch(
                inputs = queryGradientRows,
                weights = b.loraQb,
                inputColumns = dModel,
                outputColumns = loraRank,
                outputs = queryRankGradients,
                transposeWeights = true
            )
            computeLinearBatch(
                inputs = valueGradientRows,
                weights = b.loraVb,
                inputColumns = dModel,
                outputColumns = loraRank,
                outputs = valueRankGradients,
                transposeWeights = true
            )

            val layerInputTranspose = transposeRows(inputRowsForGradient)
            val queryGradientA = Array(dModel) { FloatArray(loraRank) }
            val valueGradientA = Array(dModel) { FloatArray(loraRank) }
            computeLinearBatch(
                inputs = layerInputTranspose,
                weights = flattenRows(queryRankGradients),
                inputColumns = attentionBackwardRows,
                outputColumns = loraRank,
                outputs = queryGradientA
            )
            computeLinearBatch(
                inputs = layerInputTranspose,
                weights = flattenRows(valueRankGradients),
                inputColumns = attentionBackwardRows,
                outputColumns = loraRank,
                outputs = valueGradientA
            )
            System.arraycopy(flattenRows(queryGradientA), 0, lg.gradLoraQa, 0, lg.gradLoraQa.size)
            System.arraycopy(flattenRows(valueGradientA), 0, lg.gradLoraVa, 0, lg.gradLoraVa.size)

            val queryLoRaInputGradient = Array(attentionBackwardRows) { FloatArray(dModel) }
            val valueLoRaInputGradient = Array(attentionBackwardRows) { FloatArray(dModel) }
            computeLinearBatch(
                inputs = queryRankGradients,
                weights = b.loraQa,
                inputColumns = loraRank,
                outputColumns = dModel,
                outputs = queryLoRaInputGradient,
                transposeWeights = true
            )
            computeLinearBatch(
                inputs = valueRankGradients,
                weights = b.loraVa,
                inputColumns = loraRank,
                outputColumns = dModel,
                outputs = valueLoRaInputGradient,
                transposeWeights = true
            )

            // Base Q/K/V and LoRA input gradients are dispatched as separate Vulkan
            // products. This avoids materializing effective Q/V matrices on the host.
            val inputGradientByProjection =
                Array(3) { Array(attentionBackwardRows) { FloatArray(dModel) } }
            computeBatchedLinear(
                inputs = arrayOf(queryGradientRows, keyGradientRows, valueGradientRows),
                weights = arrayOf(b.wQ, b.wK, b.wV),
                outputs = inputGradientByProjection,
                inputColumns = dModel,
                outputColumns = dModel,
                transposeWeights = true
            )

            val dLayerInput = Array(seqLen) { FloatArray(dModel) }
            for (time in 0 until attentionBackwardRows) {
                val queryInputGradient = inputGradientByProjection[0][time]
                val keyInputGradient = inputGradientByProjection[1][time]
                val valueInputGradient = inputGradientByProjection[2][time]
                for (dimension in 0 until dModel) {
                    dLayerInput[time][dimension] =
                        dL_dh1PreLN[time][dimension] +
                        queryInputGradient[dimension] +
                        keyInputGradient[dimension] +
                        valueInputGradient[dimension] +
                        queryLoRaInputGradient[time][dimension] +
                        valueLoRaInputGradient[time][dimension]
                }
            }

            if (layer > 0) {
                incomingGrad = dLayerInput
            } else {
                val embeddingBackwardTokens = FloatArray(attentionBackwardRows) { row ->
                    seq[row].toFloat()
                }
                val embeddingInputGradients = flattenRows(
                    Array(attentionBackwardRows) { row -> dLayerInput[row] }
                )
                val embeddingGradientOnGpu = VulkanTrainingBackend.isResidentBackwardAvailable &&
                    VulkanTrainingBackend.tryRunResidentBackward(
                        operation = VulkanTrainingBackend.BACKWARD_EMBEDDING_GRADIENT,
                        rows = vocabCap,
                        columns = dModel,
                        sequenceLength = attentionBackwardRows,
                        inputs = arrayOf(
                            embeddingBackwardTokens,
                            embeddingInputGradients,
                            null
                        ),
                        outputs = arrayOf(gradWEmbed),
                        scale = embedScale
                    )
                if (!embeddingGradientOnGpu) {
                    for (time in 0 until attentionBackwardRows) {
                        val token = seq[time]
                        if (token !in 0 until vocabCap) continue
                        val embeddingOffset = token * dModel
                        for (dimension in 0 until dModel) {
                            gradWEmbed[embeddingOffset + dimension] +=
                                dLayerInput[time][dimension] * embedScale
                        }
                    }
                }
            }
        }

        // ==================== GLOBAL GRADIENT CLIPPING & ADAM STEP ====================
        val stepDiv = max(validSteps, 1).toFloat()
        normalizeGradient(gradWHead, stepDiv)
        normalizeGradient(gradWEmbed, stepDiv)
        for (lg in layerGrads) {
            normalizeGradient(lg.gradWDown, stepDiv)
            normalizeGradient(lg.gradWUp, stepDiv)
            normalizeGradient(lg.gradWGate, stepDiv)
            normalizeGradient(lg.gradWO, stepDiv)
            normalizeGradient(lg.gradWQ, stepDiv)
            normalizeGradient(lg.gradWK, stepDiv)
            normalizeGradient(lg.gradWV, stepDiv)
            normalizeGradient(lg.gradLoraQa, stepDiv)
            normalizeGradient(lg.gradLoraQb, stepDiv)
            normalizeGradient(lg.gradLoraVa, stepDiv)
            normalizeGradient(lg.gradLoraVb, stepDiv)
        }

        if (gradientProbe != null) {
            val selectedGradient = when {
                gradientProbe.parameterName == "wHead" -> gradWHead
                gradientProbe.parameterName == "wEmbed" -> gradWEmbed
                gradientProbe.parameterName.startsWith("layers.") -> {
                    val parts = gradientProbe.parameterName.split(".", limit = 3)
                    val lIdx = parts[1].toIntOrNull() ?: 0
                    val subName = parts[2]
                    val lg = layerGrads[lIdx]
                    when (subName) {
                        "wDown", "feed_forward.down.weight" -> lg.gradWDown
                        "wUp", "feed_forward.up.weight" -> lg.gradWUp
                        "wGate", "feed_forward.gate.weight" -> lg.gradWGate
                        "wO", "attention.output.weight" -> lg.gradWO
                        "wQ", "attention.query.weight" -> lg.gradWQ
                        "wK", "attention.key.weight" -> lg.gradWK
                        "wV", "attention.value.weight" -> lg.gradWV
                        "loraQa", "adapter.q.a" -> lg.gradLoraQa
                        "loraQb", "adapter.q.b" -> lg.gradLoraQb
                        "loraVa", "adapter.v.a" -> lg.gradLoraVa
                        "loraVb", "adapter.v.b" -> lg.gradLoraVb
                        else -> throw IllegalArgumentException("Unsupported gradient-check parameter: ${gradientProbe.parameterName}")
                    }
                }
                else -> {
                    val lg = layerGrads[0]
                    when (gradientProbe.parameterName) {
                        "wDown" -> lg.gradWDown
                        "wUp" -> lg.gradWUp
                        "wGate" -> lg.gradWGate
                        "wO" -> lg.gradWO
                        "wQ" -> lg.gradWQ
                        "wK" -> lg.gradWK
                        "wV" -> lg.gradWV
                        "loraQa" -> lg.gradLoraQa
                        "loraQb" -> lg.gradLoraQb
                        "loraVa" -> lg.gradLoraVa
                        "loraVb" -> lg.gradLoraVb
                        else -> throw IllegalArgumentException("Unsupported gradient-check parameter: ${gradientProbe.parameterName}")
                    }
                }
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

        if (gradientAccumulator != null) {
            if (validSteps > 0) {
                val sampleGradients = ArrayList<FloatArray>(2 + layerGrads.size * 11)
                sampleGradients += gradWHead
                sampleGradients += gradWEmbed
                for (layerGradient in layerGrads) {
                    sampleGradients += layerGradient.gradWDown
                    sampleGradients += layerGradient.gradWUp
                    sampleGradients += layerGradient.gradWGate
                    sampleGradients += layerGradient.gradWO
                    sampleGradients += layerGradient.gradWQ
                    sampleGradients += layerGradient.gradWK
                    sampleGradients += layerGradient.gradWV
                    sampleGradients += layerGradient.gradLoraQa
                    sampleGradients += layerGradient.gradLoraQb
                    sampleGradients += layerGradient.gradLoraVa
                    sampleGradients += layerGradient.gradLoraVb
                }
                require(sampleGradients.size == gradientAccumulator.gradients.size) {
                    "Gradient accumulator tensor count does not match the active architecture."
                }
                for (tensorIndex in sampleGradients.indices) {
                    require(sampleGradients[tensorIndex].size <= gradientAccumulator.gradients[tensorIndex].size) {
                        "Sample gradient exceeds the resident model tensor."
                    }
                }

                val residentAccumulator = gradientAccumulator.resident
                if (residentAccumulator != null &&
                    VulkanTrainingBackend.accumulateResidentGradients(
                        residentAccumulator,
                        sampleGradients.toTypedArray(),
                        validSteps
                    )) {
                    gradientAccumulator.targetCount = Math.addExact(
                        gradientAccumulator.targetCount,
                        validSteps
                    )
                    return totalLoss / validSteps
                }

                if (residentAccumulator != null) {
                    // Failed dispatches do not swap active/spare; recover the committed
                    // prefix before appending this sample with the CPU fallback.
                    if (!VulkanTrainingBackend.downloadResidentGradientAccumulator(
                            residentAccumulator,
                            gradientAccumulator.gradients
                        )) {
                        VulkanTrainingBackend.discardResidentGradientAccumulator(residentAccumulator)
                        gradientAccumulator.resident = null
                        throw IllegalStateException(
                            "Vulkan gradient accumulation failed and its committed sums could not be recovered."
                        )
                    }
                    gradientAccumulator.resident = null
                }

                for (tensorIndex in sampleGradients.indices) {
                    val source = sampleGradients[tensorIndex]
                    val destination = gradientAccumulator.gradients[tensorIndex]
                    for (elementIndex in source.indices) {
                        destination[elementIndex] += source[elementIndex] * validSteps
                    }
                }
                gradientAccumulator.targetCount = Math.addExact(
                    gradientAccumulator.targetCount,
                    validSteps
                )
            }
            return if (validSteps > 0) totalLoss / validSteps else Float.NaN
        }

        val beta1Pow = Math.pow(beta1.toDouble(), adamStep.toDouble()).toFloat()
        val beta2Pow = Math.pow(beta2.toDouble(), adamStep.toDouble()).toFloat()
        val biasCorr1 = 1f - beta1Pow
        val biasCorr2 = 1f - beta2Pow

        val parameterList = mutableListOf<Float16ParameterArray>()
        val gradientList = mutableListOf<FloatArray>()
        val firstMomentList = mutableListOf<FloatArray>()
        val secondMomentList = mutableListOf<FloatArray>()
        val weightDecayList = mutableListOf<Float>()

        parameterList.add(wHead); gradientList.add(gradWHead); firstMomentList.add(mHead); secondMomentList.add(vHead); weightDecayList.add(0f)
        parameterList.add(wEmbed); gradientList.add(gradWEmbed); firstMomentList.add(mEmbed); secondMomentList.add(vEmbed); weightDecayList.add(0f)

        for (l in 0 until layerCount) {
            val b = blocks[l]
            val lg = layerGrads[l]

            parameterList.add(b.wDown); gradientList.add(lg.gradWDown); firstMomentList.add(b.mWDown); secondMomentList.add(b.vWDown); weightDecayList.add(weightDecay)
            parameterList.add(b.wUp); gradientList.add(lg.gradWUp); firstMomentList.add(b.mWUp); secondMomentList.add(b.vWUp); weightDecayList.add(weightDecay)
            parameterList.add(b.wGate); gradientList.add(lg.gradWGate); firstMomentList.add(b.mWGate); secondMomentList.add(b.vWGate); weightDecayList.add(weightDecay)
            parameterList.add(b.wO); gradientList.add(lg.gradWO); firstMomentList.add(b.mWO); secondMomentList.add(b.vWO); weightDecayList.add(weightDecay)
            parameterList.add(b.wQ); gradientList.add(lg.gradWQ); firstMomentList.add(b.mWQ); secondMomentList.add(b.vWQ); weightDecayList.add(weightDecay)
            parameterList.add(b.wK); gradientList.add(lg.gradWK); firstMomentList.add(b.mWK); secondMomentList.add(b.vWK); weightDecayList.add(weightDecay)
            parameterList.add(b.wV); gradientList.add(lg.gradWV); firstMomentList.add(b.mWV); secondMomentList.add(b.vWV); weightDecayList.add(weightDecay)

            parameterList.add(b.loraQa); gradientList.add(lg.gradLoraQa); firstMomentList.add(b.mLoraQa); secondMomentList.add(b.vLoraQa); weightDecayList.add(0f)
            parameterList.add(b.loraQb); gradientList.add(lg.gradLoraQb); firstMomentList.add(b.mLoraQb); secondMomentList.add(b.vLoraQb); weightDecayList.add(0f)
            parameterList.add(b.loraVa); gradientList.add(lg.gradLoraVa); firstMomentList.add(b.mLoraVa); secondMomentList.add(b.vLoraVa); weightDecayList.add(0f)
            parameterList.add(b.loraVb); gradientList.add(lg.gradLoraVb); firstMomentList.add(b.mLoraVb); secondMomentList.add(b.vLoraVb); weightDecayList.add(0f)
        }

        val gradientGroups = gradientList.toTypedArray()
        val clippedOnGpu = try {
            tryNormalizeAndClipGradientsOnVulkan(gradientGroups, maxGradNorm)
        } catch (error: RuntimeException) {
            adamStep--
            throw error
        }
        val clipFactor = if (clippedOnGpu) {
            1.0f
        } else {
            var sumSquares = 0f
            for (gradient in gradientGroups) {
                sumSquares += computeSumSquares(gradient)
            }
            val totalNorm = sqrt(sumSquares)
            if (!totalNorm.isFinite()) {
                adamStep--
                return Float.NaN
            }
            if (totalNorm > maxGradNorm) {
                maxGradNorm / (totalNorm + 1e-6f)
            } else {
                1.0f
            }
        }

        applyAdamWUpdate(
            parameters = parameterList.toTypedArray(),
            gradients = gradientGroups,
            firstMoments = firstMomentList.toTypedArray(),
            secondMoments = secondMomentList.toTypedArray(),
            weightDecayByGroup = weightDecayList.toFloatArray(),
            learningRate = lr,
            biasCorrection1 = biasCorr1,
            biasCorrection2 = biasCorr2,
            gradientClipFactor = clipFactor
        )

        return if (validSteps > 0) totalLoss / validSteps else Float.NaN
    }

    private fun currentNamedParameterTensors(): List<VulkanModelParameterTensor> =
        buildList(2 + layerCount * 11) {
            add(VulkanModelParameterTensor("token.embedding.weight", wEmbed))
            add(VulkanModelParameterTensor("lm_head.weight", wHead))
            for ((layerIndex, block) in blocks.withIndex()) {
                val prefix = "layers.$layerIndex"
                add(VulkanModelParameterTensor("$prefix.attention.query.weight", block.wQ))
                add(VulkanModelParameterTensor("$prefix.attention.key.weight", block.wK))
                add(VulkanModelParameterTensor("$prefix.attention.value.weight", block.wV))
                add(VulkanModelParameterTensor("$prefix.attention.output.weight", block.wO))
                add(VulkanModelParameterTensor("$prefix.feed_forward.gate.weight", block.wGate))
                add(VulkanModelParameterTensor("$prefix.feed_forward.up.weight", block.wUp))
                add(VulkanModelParameterTensor("$prefix.feed_forward.down.weight", block.wDown))
                add(VulkanModelParameterTensor("$prefix.adapter.q.a", block.loraQa))
                add(VulkanModelParameterTensor("$prefix.adapter.q.b", block.loraQb))
                add(VulkanModelParameterTensor("$prefix.adapter.v.a", block.loraVa))
                add(VulkanModelParameterTensor("$prefix.adapter.v.b", block.loraVb))
            }
        }

    private fun currentParameterArrays(): Array<Float16ParameterArray> {
        val arrays = ArrayList<Float16ParameterArray>(2 + layerCount * 11)
        arrays += wHead
        arrays += wEmbed
        for (block in blocks) {
            arrays += block.wDown
            arrays += block.wUp
            arrays += block.wGate
            arrays += block.wO
            arrays += block.wQ
            arrays += block.wK
            arrays += block.wV
            arrays += block.loraQa
            arrays += block.loraQb
            arrays += block.loraVa
            arrays += block.loraVb
        }
        return arrays.toTypedArray()
    }

    private fun currentFirstMomentArrays(): Array<FloatArray> {
        val arrays = ArrayList<FloatArray>(2 + layerCount * 11)
        arrays += mHead
        arrays += mEmbed
        for (block in blocks) {
            arrays += block.mWDown
            arrays += block.mWUp
            arrays += block.mWGate
            arrays += block.mWO
            arrays += block.mWQ
            arrays += block.mWK
            arrays += block.mWV
            arrays += block.mLoraQa
            arrays += block.mLoraQb
            arrays += block.mLoraVa
            arrays += block.mLoraVb
        }
        return arrays.toTypedArray()
    }

    private fun currentSecondMomentArrays(): Array<FloatArray> {
        val arrays = ArrayList<FloatArray>(2 + layerCount * 11)
        arrays += vHead
        arrays += vEmbed
        for (block in blocks) {
            arrays += block.vWDown
            arrays += block.vWUp
            arrays += block.vWGate
            arrays += block.vWO
            arrays += block.vWQ
            arrays += block.vWK
            arrays += block.vWV
            arrays += block.vLoraQa
            arrays += block.vLoraQb
            arrays += block.vLoraVa
            arrays += block.vLoraVb
        }
        return arrays.toTypedArray()
    }

    private fun tryNormalizeAndClipGradientsOnVulkan(
        gradients: Array<FloatArray>,
        maximumGradientNorm: Float
    ): Boolean {
        if (!VulkanTrainingBackend.isResidentGradientAccumulatorAvailable) return false
        val accumulator = VulkanTrainingBackend.createResidentGradientAccumulator(gradients)
            ?: return false
        try {
            if (!VulkanTrainingBackend.accumulateResidentGradients(
                    accumulator,
                    gradients,
                    targetWeight = 1
                )) {
                return false
            }
            if (VulkanTrainingBackend.finalizeResidentGradientAccumulator(
                    accumulator,
                    gradients,
                    targetCount = 1,
                    maximumGradientNorm = maximumGradientNorm
                )) {
                return true
            }
            if (!VulkanTrainingBackend.downloadResidentGradientAccumulator(
                    accumulator,
                    gradients
                )) {
                throw IllegalStateException(
                    "Vulkan gradient clipping failed and its original gradients could not be recovered."
                )
            }
            return false
        } finally {
            if (!accumulator.released) {
                VulkanTrainingBackend.discardResidentGradientAccumulator(accumulator)
            }
        }
    }

    private fun createBatchGradientAccumulator(): BatchGradientAccumulator {
        val parameters = currentParameterArrays()
        val gradients = Array(parameters.size) { index ->
            FloatArray(parameters[index].size)
        }
        val resident = if (VulkanTrainingBackend.isResidentGradientAccumulatorAvailable) {
            VulkanTrainingBackend.createResidentGradientAccumulator(gradients)
        } else {
            null
        }
        return BatchGradientAccumulator(
            gradients = gradients,
            resident = resident
        )
    }

    /** Applies one target-weighted AdamW update after every micro-batch sample is processed. */
    private fun applyAccumulatedGradients(accumulator: BatchGradientAccumulator) {
        require(accumulator.targetCount > 0) {
            "Cannot apply an empty gradient accumulation batch."
        }

        val parameters = currentParameterArrays()
        val gradients = accumulator.gradients
        val firstMoments = currentFirstMomentArrays()
        val secondMoments = currentSecondMomentArrays()
        require(
            parameters.size == gradients.size &&
                parameters.size == firstMoments.size &&
                parameters.size == secondMoments.size
        ) { "Optimizer tensor groups do not match the active architecture." }

        for (index in gradients.indices) {
            require(
                gradients[index].size <= parameters[index].size &&
                    parameters[index].size == firstMoments[index].size &&
                    parameters[index].size == secondMoments[index].size
            ) { "Optimizer tensor shape mismatch at index $index." }
        }

        var clippedOnGpu = false
        val residentAccumulator = accumulator.resident
        if (residentAccumulator != null) {
            clippedOnGpu = VulkanTrainingBackend.finalizeResidentGradientAccumulator(
                residentAccumulator,
                gradients,
                accumulator.targetCount,
                maxGradNorm
            )
            if (!clippedOnGpu &&
                !VulkanTrainingBackend.downloadResidentGradientAccumulator(
                    residentAccumulator,
                    gradients
                )) {
                VulkanTrainingBackend.discardResidentGradientAccumulator(residentAccumulator)
                accumulator.resident = null
                throw IllegalStateException(
                    "Vulkan gradient norm/clipping failed and its committed sums could not be recovered."
                )
            }
            accumulator.resident = null
        }

        var sumSquares = 0f
        if (!clippedOnGpu) {
            for (gradient in gradients) {
                normalizeGradient(gradient, accumulator.targetCount.toFloat())
                sumSquares += computeSumSquares(gradient)
            }
        }

        val totalNorm = if (clippedOnGpu) 0.0f else sqrt(sumSquares)
        if (!clippedOnGpu && !totalNorm.isFinite()) {
            throw IllegalStateException("Accumulated gradient norm is not finite.")
        }
        val clipFactor = if (clippedOnGpu) {
            1.0f
        } else if (totalNorm > maxGradNorm) {
            maxGradNorm / (totalNorm + 1e-6f)
        } else {
            1.0f
        }

        adamStep++
        val learningRate = baseLearningRate / (1f + 0.0002f * adamStep)
        val biasCorrection1 = 1f -
            Math.pow(beta1.toDouble(), adamStep.toDouble()).toFloat()
        val biasCorrection2 = 1f -
            Math.pow(beta2.toDouble(), adamStep.toDouble()).toFloat()
        val weightDecayByGroup = FloatArray(parameters.size).also { values ->
            for (layer in 0 until layerCount) {
                val offset = 2 + layer * 11
                for (index in 0 until 7) values[offset + index] = weightDecay
            }
        }

        applyAdamWUpdate(
            parameters = parameters,
            gradients = gradients,
            firstMoments = firstMoments,
            secondMoments = secondMoments,
            weightDecayByGroup = weightDecayByGroup,
            learningRate = learningRate,
            biasCorrection1 = biasCorrection1,
            biasCorrection2 = biasCorrection2,
            gradientClipFactor = clipFactor
        )
    }

    /**
     * Executes one logical AdamW update. The preferred Vulkan path updates resident
     * FP16 weights and FP32 moments in place; staged Vulkan and Kotlin remain fallbacks.
     */
    private fun applyAdamWUpdate(
        parameters: Array<Float16ParameterArray>,
        gradients: Array<FloatArray>,
        firstMoments: Array<FloatArray>,
        secondMoments: Array<FloatArray>,
        weightDecayByGroup: FloatArray,
        learningRate: Float,
        biasCorrection1: Float,
        biasCorrection2: Float,
        gradientClipFactor: Float
    ) {
        require(parameters.isNotEmpty()) {
            "AdamW requires at least one parameter group."
        }
        require(
            parameters.size == gradients.size &&
                parameters.size == firstMoments.size &&
                parameters.size == secondMoments.size &&
                parameters.size == weightDecayByGroup.size
        ) {
            "AdamW tensor and weight-decay group counts must match."
        }

        val gpuOrStagedUpdateCompleted = try {
            VulkanTrainingBackend.tryApplyFp16AdamWUpdate(
                parameters = parameters,
                gradients = gradients,
                firstMoments = firstMoments,
                secondMoments = secondMoments,
                weightDecayByGroup = weightDecayByGroup,
                learningRate = learningRate,
                beta1 = beta1,
                beta2 = beta2,
                biasCorrection1 = biasCorrection1,
                biasCorrection2 = biasCorrection2,
                epsilon = epsilon,
                gradientClipFactor = gradientClipFactor
            )
        } catch (error: RuntimeException) {
            // The caller incremented the logical optimizer step before dispatch.
            // A failed state recovery must not advance that step or continue on stale state.
            adamStep = (adamStep - 1).coerceAtLeast(0)
            throw error
        }
        if (gpuOrStagedUpdateCompleted) {
            synchronizeVulkanModelTensorResidency("Vulkan AdamW parameter update")
            return
        }

        for (index in parameters.indices) {
            require(
                gradients[index].size <= parameters[index].size &&
                    parameters[index].size == firstMoments[index].size &&
                    parameters[index].size == secondMoments[index].size
            ) {
                "AdamW tensor shape mismatch at parameter group $index."
            }
            applyClip(gradients[index], gradientClipFactor)
            adamUpdate(
                param = parameters[index],
                grad = gradients[index],
                m = firstMoments[index],
                v = secondMoments[index],
                lr = learningRate,
                biasCorr1 = biasCorrection1,
                biasCorr2 = biasCorrection2,
                wd = weightDecayByGroup[index]
            )
        }
        synchronizeVulkanModelTensorResidency("Kotlin AdamW fallback parameter update")
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
        param: Float16ParameterArray,
        grad: FloatArray,
        m: FloatArray,
        v: FloatArray,
        lr: Float,
        biasCorr1: Float,
        biasCorr2: Float,
        wd: Float = 0f
    ) {
        require(param.size == m.size && param.size == v.size) {
            "FP16 parameters and FP32 AdamW moments must have matching tensor sizes."
        }
        val limit = min(param.size, grad.size)
        for (index in 0 until limit) {
            val gradient = grad[index]
            val parameter = param[index]
            m[index] = beta1 * m[index] + (1f - beta1) * gradient
            v[index] = beta2 * v[index] + (1f - beta2) * gradient * gradient
            val correctedFirstMoment = m[index] / biasCorr1
            val correctedSecondMoment = v[index] / biasCorr2
            val update = correctedFirstMoment / (sqrt(correctedSecondMoment) + epsilon) +
                wd * parameter
            val updatedParameter = parameter - lr * update
            require(updatedParameter.isFinite()) {
                "FP16 AdamW produced a non-finite parameter update."
            }
            param[index] = updatedParameter
        }
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

    private fun computeLinear(
        x: FloatArray,
        weights: Float16ParameterArray,
        inputColumns: Int,
        outputColumns: Int,
        output: FloatArray
    ) {
        require(weights.size == Math.multiplyExact(inputColumns, outputColumns)) {
            "FP16 linear matrix shape does not match its storage."
        }
        for (column in 0 until outputColumns) {
            var sum = 0f
            for (row in 0 until inputColumns) {
                sum += x[row] * weights[row * outputColumns + column]
            }
            output[column] = sum
        }
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
        weights: Float16ParameterArray,
        inputColumns: Int,
        outputColumns: Int,
        outputs: Array<FloatArray>,
        transposeWeights: Boolean = false
    ) {
        val fp16Dispatched = VulkanTrainingBackend.tryApplyFp16Linear(
            inputRows = inputs,
            weights = weights,
            outputRows = outputs,
            inputColumns = inputColumns,
            outputColumns = outputColumns,
            transposeWeights = transposeWeights
        )
        if (fp16Dispatched) return

        computeLinearBatch(
            inputs = inputs,
            weights = weights.toFloatArray(),
            inputColumns = inputColumns,
            outputColumns = outputColumns,
            outputs = outputs,
            transposeWeights = transposeWeights
        )
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
        weights: Array<Float16ParameterArray>,
        outputs: Array<Array<FloatArray>>,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean = false
    ) {
        val sharedInput = inputs.firstOrNull()
        val residentDispatched = sharedInput != null &&
            inputs.all { batchInput -> batchInput === sharedInput } &&
            VulkanTrainingBackend.tryApplyResidentProjectionSet(
                inputRows = sharedInput,
                weightTensors = weights,
                outputRows = outputs,
                inputColumns = inputColumns,
                outputColumns = outputColumns,
                transposeWeights = transposeWeights
            )
        if (residentDispatched) return

        val fp16Dispatched = VulkanTrainingBackend.tryApplyFp16BatchedLinear(
            inputBatches = inputs,
            weightBatches = weights,
            outputBatches = outputs,
            inputColumns = inputColumns,
            outputColumns = outputColumns,
            transposeWeights = transposeWeights
        )
        if (fp16Dispatched) return

        val fp32Weights = Array(weights.size) { index -> weights[index].toFloatArray() }
        computeBatchedLinear(
            inputs = inputs,
            weights = fp32Weights,
            outputs = outputs,
            inputColumns = inputColumns,
            outputColumns = outputColumns,
            transposeWeights = transposeWeights
        )
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
        weights: Float16ParameterArray,
        inputColumns: Int,
        outputColumns: Int,
        output: FloatArray
    ) {
        require(weights.size == Math.multiplyExact(inputColumns, outputColumns)) {
            "FP16 transposed matrix shape does not match its storage."
        }
        for (column in 0 until outputColumns) {
            var sum = 0f
            val weightOffset = column * inputColumns
            for (inner in 0 until inputColumns) {
                sum += input[inner] * weights[weightOffset + inner]
            }
            output[column] = sum
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
        x: FloatArray, wBase: Float16ParameterArray,
        loraA: Float16ParameterArray, loraB: Float16ParameterArray,
        inDim: Int, outDim: Int, rank: Int,
        midBuf: FloatArray, finalOut: FloatArray
    ) {
        computeLinear(x, wBase, inDim, outDim, finalOut)
        addLoraProjection(x, loraA, loraB, inDim, outDim, rank, midBuf, finalOut)
    }

    private fun addLoraProjection(
        input: FloatArray,
        loraA: Float16ParameterArray,
        loraB: Float16ParameterArray,
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

    private fun layerNormBackwardBatch(
        incomingGradients: Array<FloatArray>,
        preLayerNormRows: Array<FloatArray>,
        outputGradients: Array<FloatArray>,
        rowCount: Int
    ) {
        require(rowCount in 1..incomingGradients.size &&
            rowCount <= preLayerNormRows.size &&
            rowCount <= outputGradients.size) {
            "LayerNorm backward batch dimensions are invalid."
        }
        require(
            (0 until rowCount).all { row ->
                incomingGradients[row].size == dModel &&
                    preLayerNormRows[row].size == dModel &&
                    outputGradients[row].size == dModel
            }
        ) {
            "LayerNorm backward tensors must match the active model width."
        }

        val flattenedIncoming = flattenRows(
            Array(rowCount) { row -> incomingGradients[row] }
        )
        val flattenedInputs = flattenRows(
            Array(rowCount) { row -> preLayerNormRows[row] }
        )
        val flattenedOutput = FloatArray(Math.multiplyExact(rowCount, dModel))
        val appliedOnGpu = VulkanTrainingBackend.isResidentBackwardAvailable &&
            VulkanTrainingBackend.tryRunResidentBackward(
                operation = VulkanTrainingBackend.BACKWARD_LAYER_NORM,
                rows = rowCount,
                columns = dModel,
                sequenceLength = 0,
                inputs = arrayOf(flattenedIncoming, flattenedInputs, null),
                outputs = arrayOf(flattenedOutput),
                epsilon = 1.0e-5f
            )

        if (appliedOnGpu) {
            for (row in 0 until rowCount) {
                System.arraycopy(
                    flattenedOutput,
                    row * dModel,
                    outputGradients[row],
                    0,
                    dModel
                )
            }
        } else {
            for (row in 0 until rowCount) {
                layerNormBackward(
                    incomingGradients[row],
                    preLayerNormRows[row],
                    outputGradients[row],
                    dModel
                )
            }
        }
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
        val memMb = (countTrainableParameters() * Short.SIZE_BYTES.toLong()) / (1024f * 1024f)
        BenchmarkResult(tps, msPerToken, memMb, tokens, elapsedMs)
    }

    // =====================================================================
    // Checkpoint (ARG4 streaming writer with ARG3 compatibility reader)
    // =====================================================================
    fun saveCheckpoint(): Boolean {
        return try {
            synchronized(modelLock) {
                check(VulkanTrainingBackend.synchronizeTrainingStateMirrors()) {
                    "GPU model or optimizer state could not be recovered before checkpoint serialization."
                }
                writeArg4Checkpoint()
            }
            val file = File(context.filesDir, "argus_model_weights.bin")
            file.isFile && file.length() >= ARG4_HEADER_SIZE.toLong()
        } catch (t: Throwable) {
            Log.e(TAG, "Checkpoint save failed", t)
            false
        }
    }

    private class NamedCheckpointTensor private constructor(
        val name: String,
        private val floatValues: FloatArray?,
        private val fp16Values: Float16ParameterArray?
    ) {
        constructor(name: String, values: FloatArray) : this(name, values, null)
        constructor(name: String, values: Float16ParameterArray) : this(name, null, values)

        val size: Int
            get() = when {
                floatValues != null -> floatValues.size
                fp16Values != null -> fp16Values.size
                else -> throw IllegalStateException("Checkpoint tensor has no backing storage.")
            }

        operator fun get(index: Int): Float = when {
            floatValues != null -> floatValues[index]
            fp16Values != null -> fp16Values[index]
            else -> throw IllegalStateException("Checkpoint tensor has no backing storage.")
        }

        operator fun set(index: Int, value: Float) {
            when {
                floatValues != null -> floatValues[index] = value
                fp16Values != null -> fp16Values[index] = value
                else -> throw IllegalStateException("Checkpoint tensor has no backing storage.")
            }
        }
    }

    private data class CheckpointTensorDescriptor(
        val name: String,
        val elementCount: Long,
        val dataType: Int
    )

    private fun checkpointTensors(): List<NamedCheckpointTensor> {
        val list = ArrayList<NamedCheckpointTensor>(6 + layerCount * 32)
        list.add(NamedCheckpointTensor("token.embedding.weight", wEmbed))
        list.add(NamedCheckpointTensor("token.embedding.adam_m", mEmbed))
        list.add(NamedCheckpointTensor("token.embedding.adam_v", vEmbed))
        list.add(NamedCheckpointTensor("lm_head.weight", wHead))
        list.add(NamedCheckpointTensor("lm_head.adam_m", mHead))
        list.add(NamedCheckpointTensor("lm_head.adam_v", vHead))
        for (l in 0 until layerCount) {
            val b = blocks[l]
            val prefix = "layers.$l."
            list.add(NamedCheckpointTensor("${prefix}adapter.q.a", b.loraQa))
            list.add(NamedCheckpointTensor("${prefix}adapter.q.b", b.loraQb))
            list.add(NamedCheckpointTensor("${prefix}adapter.v.a", b.loraVa))
            list.add(NamedCheckpointTensor("${prefix}adapter.v.b", b.loraVb))
            list.add(NamedCheckpointTensor("${prefix}adapter.q.a.adam_m", b.mLoraQa))
            list.add(NamedCheckpointTensor("${prefix}adapter.q.b.adam_m", b.mLoraQb))
            list.add(NamedCheckpointTensor("${prefix}adapter.v.a.adam_m", b.mLoraVa))
            list.add(NamedCheckpointTensor("${prefix}adapter.v.b.adam_m", b.mLoraVb))
            list.add(NamedCheckpointTensor("${prefix}adapter.q.a.adam_v", b.vLoraQa))
            list.add(NamedCheckpointTensor("${prefix}adapter.q.b.adam_v", b.vLoraQb))
            list.add(NamedCheckpointTensor("${prefix}adapter.v.a.adam_v", b.vLoraVa))
            list.add(NamedCheckpointTensor("${prefix}adapter.v.b.adam_v", b.vLoraVb))
            list.add(NamedCheckpointTensor("${prefix}attention.query.weight", b.wQ))
            list.add(NamedCheckpointTensor("${prefix}attention.key.weight", b.wK))
            list.add(NamedCheckpointTensor("${prefix}attention.value.weight", b.wV))
            list.add(NamedCheckpointTensor("${prefix}attention.output.weight", b.wO))
            list.add(NamedCheckpointTensor("${prefix}attention.query.adam_m", b.mWQ))
            list.add(NamedCheckpointTensor("${prefix}attention.query.adam_v", b.vWQ))
            list.add(NamedCheckpointTensor("${prefix}attention.key.adam_m", b.mWK))
            list.add(NamedCheckpointTensor("${prefix}attention.key.adam_v", b.vWK))
            list.add(NamedCheckpointTensor("${prefix}attention.value.adam_m", b.mWV))
            list.add(NamedCheckpointTensor("${prefix}attention.value.adam_v", b.vWV))
            list.add(NamedCheckpointTensor("${prefix}attention.output.adam_m", b.mWO))
            list.add(NamedCheckpointTensor("${prefix}attention.output.adam_v", b.vWO))
            list.add(NamedCheckpointTensor("${prefix}feed_forward.gate.weight", b.wGate))
            list.add(NamedCheckpointTensor("${prefix}feed_forward.up.weight", b.wUp))
            list.add(NamedCheckpointTensor("${prefix}feed_forward.down.weight", b.wDown))
            list.add(NamedCheckpointTensor("${prefix}feed_forward.gate.adam_m", b.mWGate))
            list.add(NamedCheckpointTensor("${prefix}feed_forward.gate.adam_v", b.vWGate))
            list.add(NamedCheckpointTensor("${prefix}feed_forward.up.adam_m", b.mWUp))
            list.add(NamedCheckpointTensor("${prefix}feed_forward.up.adam_v", b.vWUp))
            list.add(NamedCheckpointTensor("${prefix}feed_forward.down.adam_m", b.mWDown))
            list.add(NamedCheckpointTensor("${prefix}feed_forward.down.adam_v", b.vWDown))
        }
        return list
    }

    private fun writeArg4Checkpoint() {
        ensureVocabCapacity(tokenizer.vocabSize)
        require(vocabCap >= tokenizer.vocabSize) {
            "Embedding capacity is smaller than the active tokenizer vocabulary."
        }

        val target = File(context.filesDir, "argus_model_weights.bin")
        val temporary = File(context.filesDir, "argus_model_weights.bin.tmp")
        val vocabEntries = tokenizer.exportVocab()
        val isLearnedBpe = config.tokenizerConfig.algorithm.usesBpeMerges
        val merges = if (isLearnedBpe) tokenizer.exportMerges() else emptyList()
        val tensors = checkpointTensors()
        val architecture = architectureConfig
        val architectureIdBytes = architecture.architectureId.toByteArray(Charsets.UTF_8)
        require(architecture.layerCount in 1..32) {
            "ARG4 writer supports 1..32 Transformer blocks."
        }
        require(architectureIdBytes.size in 1..512) {
            "Architecture ID length is invalid."
        }
        require(vocabEntries.size <= config.tokenizerConfig.maxVocabularySize) {
            "Tokenizer vocabulary exceeds its configured checkpoint capacity."
        }

        var payloadLength = architectureIdBytes.size.toLong()
        for ((token, _) in vocabEntries) {
            val tokenBytes = token.toByteArray(Charsets.UTF_8)
            require(tokenBytes.isNotEmpty() && tokenBytes.size <= config.tokenizerConfig.maxTokenLength * 4) {
                "Checkpoint token byte length is invalid."
            }
            payloadLength = Math.addExact(payloadLength, 8L + tokenBytes.size.toLong())
        }
        if (isLearnedBpe) {
            payloadLength = Math.addExact(payloadLength, 4L)
            for ((left, right) in merges) {
                val leftBytes = left.toByteArray(Charsets.UTF_8)
                val rightBytes = right.toByteArray(Charsets.UTF_8)
                payloadLength = Math.addExact(payloadLength, 8L + leftBytes.size.toLong() + rightBytes.size.toLong())
            }
        }
        for (tensor in tensors) {
            val nameBytes = tensor.name.toByteArray(Charsets.UTF_8)
            payloadLength = Math.addExact(payloadLength, 16L + nameBytes.size.toLong())
            payloadLength = Math.addExact(
                payloadLength,
                Math.multiplyExact(tensor.size.toLong(), 4L)
            )
        }

        val header = ByteBuffer.allocate(ARG4_HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN)
        header.put(MAGIC_0).put(MAGIC_1).put(MAGIC_2).put(MAGIC_4)
        header.putInt(ARG4_CHECKPOINT_VERSION)
        header.putInt(ARG4_HEADER_SIZE)
        header.putInt(architecture.schemaVersion)
        header.putInt(architectureIdBytes.size)
        header.putInt(architecture.layerCount)
        header.putInt(architecture.hiddenSize)
        header.putInt(architecture.attentionHeadCount)
        header.putInt(architecture.feedForwardSize)
        header.putInt(architecture.contextLength)
        header.putInt(architecture.adapterRank)
        header.putInt(vocabEntries.size) // Persist the actual tokenizer entry count.
        header.putInt(vocabCap)
        header.putInt(adamStep)
        header.putInt(config.tokenizerConfig.schemaVersion)
        header.putInt(config.tokenizerConfig.algorithm.checkpointId)
        header.putInt(config.tokenizerConfig.maxVocabularySize)
        header.putInt(config.tokenizerConfig.maxTokenLength)
        header.putInt(vocabEntries.size)
        header.putInt(tensors.size)
        header.putLong(payloadLength)
        header.putInt(0) // CRC32 of the payload is patched after streaming.
        header.putInt(0) // Reserved architecture flags.
        require(header.position() == ARG4_HEADER_SIZE) {
            "ARG4 fixed header layout is inconsistent."
        }

        val checksum = CRC32()
        try {
            FileOutputStream(temporary).use { fileOutput ->
                BufferedOutputStream(fileOutput).use { buffered ->
                    buffered.write(header.array())
                    val checked = CheckedOutputStream(buffered, checksum)

                    checked.write(architectureIdBytes)
                    for ((token, id) in vocabEntries) {
                        val tokenBytes = token.toByteArray(Charsets.UTF_8)
                        writeIntLittleEndian(checked, id)
                        writeIntLittleEndian(checked, tokenBytes.size)
                        checked.write(tokenBytes)
                    }

                    if (isLearnedBpe) {
                        writeIntLittleEndian(checked, merges.size)
                        for ((left, right) in merges) {
                            val leftBytes = left.toByteArray(Charsets.UTF_8)
                            val rightBytes = right.toByteArray(Charsets.UTF_8)
                            writeIntLittleEndian(checked, leftBytes.size)
                            checked.write(leftBytes)
                            writeIntLittleEndian(checked, rightBytes.size)
                            checked.write(rightBytes)
                        }
                    }

                    for (tensor in tensors) {
                        val nameBytes = tensor.name.toByteArray(Charsets.UTF_8)
                        writeIntLittleEndian(checked, nameBytes.size)
                        writeLongLittleEndian(checked, tensor.size.toLong())
                        writeIntLittleEndian(checked, FLOAT32_TENSOR_TYPE)
                        checked.write(nameBytes)
                    }

                    for (tensor in tensors) {
                        writeFloatArrayLittleEndian(checked, tensor)
                    }

                    checked.flush()
                    buffered.flush()
                    fileOutput.fd.sync()
                }
            }

            RandomAccessFile(temporary, "rw").use { randomAccessFile ->
                randomAccessFile.seek(ARG4_CHECKSUM_OFFSET.toLong())
                val checksumBytes = ByteBuffer.allocate(4)
                    .order(ByteOrder.LITTLE_ENDIAN)
                    .putInt(checksum.value.toInt())
                    .array()
                randomAccessFile.write(checksumBytes)
                randomAccessFile.fd.sync()
            }

            Files.move(
                temporary.toPath(),
                target.toPath(),
                StandardCopyOption.REPLACE_EXISTING,
                StandardCopyOption.ATOMIC_MOVE
            )
        } catch (t: Throwable) {
            temporary.delete()
            throw t
        }
    }

    private fun writeIntLittleEndian(output: OutputStream, value: Int) {
        output.write(value and 0xFF)
        output.write((value ushr 8) and 0xFF)
        output.write((value ushr 16) and 0xFF)
        output.write((value ushr 24) and 0xFF)
    }

    private fun writeLongLittleEndian(output: OutputStream, value: Long) {
        for (shift in 0 until 8) {
            output.write(((value ushr (shift * 8)) and 0xFFL).toInt())
        }
    }

    private fun writeFloatArrayLittleEndian(
        output: OutputStream,
        tensor: NamedCheckpointTensor
    ) {
        val buffer = ByteBuffer.allocate(TENSOR_WRITE_BUFFER_BYTES).order(ByteOrder.LITTLE_ENDIAN)
        for (index in 0 until tensor.size) {
            val value = tensor[index]
            require(value.isFinite()) {
                "Cannot save a non-finite value in tensor ${tensor.name}."
            }
            if (buffer.remaining() < Float.SIZE_BYTES) {
                output.write(buffer.array(), 0, buffer.position())
                buffer.clear()
            }
            buffer.putFloat(value)
        }
        if (buffer.position() > 0) {
            output.write(buffer.array(), 0, buffer.position())
        }
    }

    private fun readIntLittleEndian(input: InputStream): Int {
        val b0 = input.read()
        val b1 = input.read()
        val b2 = input.read()
        val b3 = input.read()
        if ((b0 or b1 or b2 or b3) < 0) {
            throw IllegalStateException("Unexpected end of little-endian integer.")
        }
        return b0 or (b1 shl 8) or (b2 shl 16) or (b3 shl 24)
    }

    private fun readLongLittleEndian(input: InputStream): Long {
        var value = 0L
        for (shift in 0 until 8) {
            val next = input.read()
            if (next < 0) {
                throw IllegalStateException("Unexpected end of little-endian long.")
            }
            value = value or (next.toLong() shl (shift * 8))
        }
        return value
    }

    private fun readFloatArrayLittleEndian(
        input: InputStream,
        tensor: NamedCheckpointTensor
    ) {
        val floatsPerChunk = TENSOR_WRITE_BUFFER_BYTES / Float.SIZE_BYTES
        var offset = 0
        while (offset < tensor.size) {
            val chunkCount = min(floatsPerChunk, tensor.size - offset)
            val byteCount = chunkCount * Float.SIZE_BYTES
            val chunk = ByteBuffer.wrap(input.readFully(byteCount))
                .order(ByteOrder.LITTLE_ENDIAN)
            for (index in 0 until chunkCount) {
                val value = chunk.float
                require(value.isFinite()) {
                    "Checkpoint contains a non-finite value in tensor ${tensor.name}."
                }
                tensor[offset + index] = value
            }
            offset += chunkCount
        }
    }

    private fun requireSafeVocabularyCapacity(targetVocabCap: Int) {
        require(targetVocabCap > 0) {
            "Checkpoint vocabulary capacity must be positive."
        }
        if (targetVocabCap == vocabCap) return

        val targetElements = Math.multiplyExact(targetVocabCap.toLong(), dModel.toLong())
        require(targetElements <= Int.MAX_VALUE.toLong()) {
            "Checkpoint vocabulary tensor dimensions exceed JVM array limits."
        }
        // Paired embedding/head rows: two FP16 weights, four FP32 moments,
        // two FP32 sample gradients and optionally two FP32 accumulation buffers.
        val bytesPerPairedElement = if (config.gradientAccumulationEnabled) 36L else 28L
        val newVocabularyArrayBytes = Math.multiplyExact(
            targetElements,
            bytesPerPairedElement
        )
        val runtime = Runtime.getRuntime()
        val availableHeap = runtime.maxMemory() -
            (runtime.totalMemory() - runtime.freeMemory())
        require(newVocabularyArrayBytes.toDouble() <= availableHeap.toDouble() * 0.7) {
            "Checkpoint vocabulary tensors exceed the safe available heap budget."
        }
    }

    private fun expectedCheckpointTensorSizes(targetVocabCap: Int): Map<String, Long> {
        val current = checkpointTensors()
        val dynamicVocabularyTensors = setOf(
            "token.embedding.weight",
            "token.embedding.adam_m",
            "token.embedding.adam_v",
            "lm_head.weight",
            "lm_head.adam_m",
            "lm_head.adam_v"
        )
        val vocabularyElementCount = Math.multiplyExact(targetVocabCap.toLong(), dModel.toLong())
        return current.associate { tensor ->
            tensor.name to if (tensor.name in dynamicVocabularyTensors) {
                vocabularyElementCount
            } else {
                tensor.size.toLong()
            }
        }
    }

    private fun loadArg4Checkpoint(file: File): Boolean {
        return try {
            require(file.length() >= ARG4_HEADER_SIZE.toLong()) {
                "ARG4 checkpoint is shorter than its fixed header."
            }
            val headerBytes = FileInputStream(file).use { it.readFully(ARG4_HEADER_SIZE) }
            val header = ByteBuffer.wrap(headerBytes).order(ByteOrder.LITTLE_ENDIAN)
            require(
                header.get() == MAGIC_0 && header.get() == MAGIC_1 &&
                    header.get() == MAGIC_2 && header.get() == MAGIC_4
            ) { "Invalid ARG4 checkpoint magic." }
            require(header.int == ARG4_CHECKPOINT_VERSION) {
                "Unsupported ARG4 checkpoint version."
            }
            require(header.int == ARG4_HEADER_SIZE) {
                "ARG4 checkpoint header size does not match."
            }

            val savedArchitectureSchema = header.int
            val architectureIdLength = header.int
            val savedLayerCount = header.int
            val savedDModel = header.int
            val savedNHeads = header.int
            val savedDFf = header.int
            val savedMaxSeq = header.int
            val savedLoraRank = header.int
            val savedVocabSize = header.int
            val savedVocabCap = header.int
            val savedAdamStep = header.int
            val savedTokenizerSchema = header.int
            val savedTokenizerAlgorithm = header.int
            val savedTokenizerMaxVocab = header.int
            val savedTokenizerMaxTokenLength = header.int
            val savedVocabularyEntryCount = header.int
            val savedTensorCount = header.int
            val savedPayloadBytes = header.long
            val savedChecksum = header.int.toLong() and 0xFFFF_FFFFL
            val flags = header.int

            require(savedArchitectureSchema == architectureConfig.schemaVersion) {
                "ARG4 architecture schema mismatch."
            }
            require(savedLayerCount in 1..32) {
                "ARG4 layer count must be between 1 and 32: $savedLayerCount."
            }
            require(savedLayerCount == layerCount) {
                "ARG4 layer count ($savedLayerCount) does not match the active engine ($layerCount)."
            }
            require(
                savedDModel == dModel && savedNHeads == nHeads &&
                    savedDFf == dFf && savedMaxSeq == maxSeqLen &&
                    savedLoraRank == loraRank
            ) { "ARG4 architecture dimensions do not match the active engine." }
            require(savedVocabSize > 0) {
                "ARG4 vocabulary size is invalid."
            }
            require(savedVocabCap >= savedVocabSize) {
                "ARG4 vocabulary capacity is invalid."
            }
            requireSafeVocabularyCapacity(savedVocabCap)
            require(savedAdamStep >= 0) { "ARG4 optimizer step cannot be negative." }
            require(savedTokenizerSchema == config.tokenizerConfig.schemaVersion) {
                "ARG4 tokenizer schema mismatch."
            }
            require(savedTokenizerAlgorithm == config.tokenizerConfig.algorithm.checkpointId) {
                "ARG4 tokenizer algorithm mismatch."
            }
            require(savedTokenizerMaxVocab == config.tokenizerConfig.maxVocabularySize) {
                "ARG4 tokenizer vocabulary capacity mismatch."
            }
            require(savedTokenizerMaxTokenLength == config.tokenizerConfig.maxTokenLength) {
                "ARG4 tokenizer token length mismatch."
            }
            require(
                savedVocabularyEntryCount in 1..config.tokenizerConfig.maxVocabularySize
            ) { "ARG4 vocabulary entry count is invalid." }
            require(savedTensorCount in 1..2048) { "ARG4 tensor count is invalid: $savedTensorCount." }
            require(architectureIdLength in 1..512) { "ARG4 architecture ID length is invalid." }
            require(savedPayloadBytes >= architectureIdLength.toLong()) {
                "ARG4 payload length is invalid."
            }
            require(flags == 0) { "ARG4 checkpoint contains unsupported flags." }
            require(savedPayloadBytes >= 0L) {
                "ARG4 payload byte count cannot be negative."
            }
            val expectedFileLength = Math.addExact(
                ARG4_HEADER_SIZE.toLong(),
                savedPayloadBytes
            )
            require(file.length() == expectedFileLength) {
                "ARG4 checkpoint length does not match its payload metadata."
            }

            val checksum = CRC32()
            BufferedInputStream(FileInputStream(file)).use { input ->
                skipFully(input, ARG4_HEADER_SIZE.toLong())
                val buffer = ByteArray(TENSOR_WRITE_BUFFER_BYTES)
                var remaining = savedPayloadBytes
                while (remaining > 0L) {
                    val requested = min(remaining, buffer.size.toLong()).toInt()
                    val count = input.read(buffer, 0, requested)
                    if (count < 0) throw IllegalStateException("ARG4 payload ended unexpectedly.")
                    checksum.update(buffer, 0, count)
                    remaining -= count.toLong()
                }
                require(input.read() == -1) { "ARG4 checkpoint contains trailing bytes." }
            }
            require(checksum.value == savedChecksum) {
                "ARG4 payload checksum mismatch."
            }

            BufferedInputStream(FileInputStream(file)).use { input ->
                input.readFully(ARG4_HEADER_SIZE)
                val architectureIdBytes = input.readFully(architectureIdLength)
                val savedArchitectureId = String(architectureIdBytes, Charsets.UTF_8)
                require(savedArchitectureId == architectureConfig.architectureId) {
                    "ARG4 architecture ID mismatch."
                }

                var measuredPayloadBytes = architectureIdLength.toLong()
                val vocabulary = ArrayList<Pair<String, Int>>(savedVocabularyEntryCount)
                repeat(savedVocabularyEntryCount) {
                    val tokenId = readIntLittleEndian(input)
                    val tokenLength = readIntLittleEndian(input)
                    require(
                        tokenLength in 1..(config.tokenizerConfig.maxTokenLength * 4)
                    ) { "ARG4 token byte length is invalid." }
                    val tokenBytes = input.readFully(tokenLength)
                    vocabulary.add(String(tokenBytes, Charsets.UTF_8) to tokenId)
                    measuredPayloadBytes = Math.addExact(
                        measuredPayloadBytes,
                        8L + tokenLength.toLong()
                    )
                }

                val isLearnedBpe = config.tokenizerConfig.algorithm.usesBpeMerges
                val loadedMerges = if (isLearnedBpe) {
                    val mergeCount = readIntLittleEndian(input)
                    require(mergeCount in 0..config.tokenizerConfig.maxVocabularySize) {
                        "ARG4 merge count is invalid: $mergeCount"
                    }
                    measuredPayloadBytes = Math.addExact(measuredPayloadBytes, 4L)
                    val mergeList = ArrayList<Pair<String, String>>(mergeCount)
                    repeat(mergeCount) {
                        val leftLen = readIntLittleEndian(input)
                        require(leftLen in 1..(config.tokenizerConfig.maxTokenLength * 4)) {
                            "ARG4 merge left component length is invalid."
                        }
                        val leftBytes = input.readFully(leftLen)
                        val rightLen = readIntLittleEndian(input)
                        require(rightLen in 1..(config.tokenizerConfig.maxTokenLength * 4)) {
                            "ARG4 merge right component length is invalid."
                        }
                        val rightBytes = input.readFully(rightLen)
                        measuredPayloadBytes = Math.addExact(measuredPayloadBytes, 8L + leftLen.toLong() + rightLen.toLong())
                        mergeList.add(
                            String(leftBytes, Charsets.UTF_8) to String(rightBytes, Charsets.UTF_8)
                        )
                    }
                    mergeList
                } else emptyList()

                val candidateTokenizer = Tokenizer(config.tokenizerConfig)
                candidateTokenizer.importVocab(vocabulary)
                if (isLearnedBpe) {
                    candidateTokenizer.importMerges(loadedMerges)
                }
                require(candidateTokenizer.vocabSize == savedVocabularyEntryCount) {
                    "ARG4 tokenizer vocabulary entry count does not match its metadata."
                }
                // Earlier ARG4 writers stored vocabCap in the savedVocabSize field.
                // Accept that legacy convention while validating the serialized entry count.
                require(
                    savedVocabSize == candidateTokenizer.vocabSize ||
                        savedVocabSize == savedVocabCap
                ) {
                    "ARG4 tokenizer vocabulary size does not match its metadata."
                }

                val expectedTensorSizes = expectedCheckpointTensorSizes(savedVocabCap)
                require(savedTensorCount == expectedTensorSizes.size) {
                    "ARG4 tensor count ($savedTensorCount) does not match the active engine (${expectedTensorSizes.size})."
                }
                val descriptors = ArrayList<CheckpointTensorDescriptor>(savedTensorCount)
                val descriptorNames = HashSet<String>(savedTensorCount)
                repeat(savedTensorCount) {
                    val nameLength = readIntLittleEndian(input)
                    val elementCount = readLongLittleEndian(input)
                    val dataType = readIntLittleEndian(input)
                    require(nameLength in 1..128) { "ARG4 tensor name length is invalid." }
                    require(elementCount >= 0L) { "ARG4 tensor element count is invalid." }
                    require(dataType == FLOAT32_TENSOR_TYPE) {
                        "ARG4 tensor data type is not supported by the ARG4 trainer."
                    }
                    val rawName = String(input.readFully(nameLength), Charsets.UTF_8)
                    val name = if (!rawName.startsWith("layers.") && !rawName.startsWith("token.") && !rawName.startsWith("lm_head.")) {
                        "layers.0.$rawName"
                    } else {
                        rawName
                    }
                    require(descriptorNames.add(name)) { "ARG4 tensor names must be unique: $name" }
                    val expectedCount = expectedTensorSizes[name]
                        ?: throw IllegalArgumentException("ARG4 contains an unknown tensor: $rawName (resolved as $name).")
                    require(elementCount == expectedCount) {
                        "ARG4 tensor element count for $name does not match the active architecture: got $elementCount, expected $expectedCount."
                    }
                    descriptors.add(
                        CheckpointTensorDescriptor(name, elementCount, dataType)
                    )
                    measuredPayloadBytes = Math.addExact(
                        measuredPayloadBytes,
                        16L + nameLength.toLong()
                    )
                    measuredPayloadBytes = Math.addExact(
                        measuredPayloadBytes,
                        Math.multiplyExact(elementCount, Float.SIZE_BYTES.toLong())
                    )
                }
                require(descriptorNames == expectedTensorSizes.keys) {
                    "ARG4 tensor manifest is incomplete."
                }
                require(measuredPayloadBytes == savedPayloadBytes) {
                    "ARG4 payload size does not match its tensor manifest: measured $measuredPayloadBytes, saved $savedPayloadBytes."
                }

                synchronized(modelLock) {
                    resizeVocabCapacityTo(savedVocabCap)
                    tokenizer.importVocab(vocabulary)
                    if (isLearnedBpe) {
                        tokenizer.importMerges(loadedMerges)
                    }
                    val tensorsByName = checkpointTensors().associateBy { it.name }
                    for (descriptor in descriptors) {
                        val targetTensor = tensorsByName[descriptor.name]
                            ?: throw IllegalStateException("ARG4 tensor target is unavailable: ${descriptor.name}")
                        require(targetTensor.size.toLong() == descriptor.elementCount) {
                            "ARG4 tensor target size changed during restoration."
                        }
                        readFloatArrayLittleEndian(input, targetTensor)
                    }
                    adamStep = savedAdamStep
                }
                require(input.read() == -1) { "ARG4 tensor payload has trailing bytes." }
            }
            true
        } catch (e: Exception) {
            Log.e(TAG, "ARG4 checkpoint load failed", e)
            false
        }
    }

    private fun skipFully(input: InputStream, count: Long) {
        var remaining = count
        while (remaining > 0L) {
            val skipped = input.skip(remaining)
            if (skipped > 0L) {
                remaining -= skipped
            } else {
                if (input.read() < 0) {
                    throw IllegalStateException("Unexpected end of checkpoint stream.")
                }
                remaining--
            }
        }
    }

    private fun loadLegacyCheckpoint(file: File): Boolean {
        return try {
            require(file.length() >= LEGACY_HEADER_SIZE.toLong()) {
                "ARG3 checkpoint is shorter than its fixed header."
            }
            FileInputStream(file).use { fileInput ->
                BufferedInputStream(fileInput).use { input ->
                    val headerBytes = input.readFully(LEGACY_HEADER_SIZE)
                    val header = ByteBuffer.wrap(headerBytes).order(ByteOrder.LITTLE_ENDIAN)
                    require(
                        header.get() == MAGIC_0 && header.get() == MAGIC_1 &&
                            header.get() == MAGIC_2 && header.get() == MAGIC_3
                    ) { "Invalid ARG3 checkpoint magic." }
                    require(header.int == LEGACY_CHECKPOINT_VERSION) {
                        "Unsupported ARG3 checkpoint version."
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

                    require(
                        savedDModel == dModel && savedLoraRank == loraRank &&
                            savedNHeads == nHeads && savedDFf == dFf &&
                            savedMaxSeq == maxSeqLen
                    ) { "ARG3 architecture dimensions do not match the active engine." }
                    require(layerCount == 1) {
                        "ARG3 legacy checkpoints can only be loaded into single-layer engines."
                    }
                    require(savedVocabSize > 0) {
                        "ARG3 vocabulary size is invalid."
                    }
                    require(savedVocabCap >= savedVocabSize) {
                        "ARG3 vocabulary capacity is invalid."
                    }
                    require(savedAdamStep >= 0) { "ARG3 optimizer step cannot be negative." }
                    requireSafeVocabularyCapacity(savedVocabCap)

                    val vocabCount = readIntLittleEndian(input)
                    require(vocabCount in 1..config.tokenizerConfig.maxVocabularySize) {
                        "ARG3 vocabulary entry count is invalid."
                    }
                    val entries = ArrayList<Pair<String, Int>>(vocabCount)
                    repeat(vocabCount) {
                        val id = readIntLittleEndian(input)
                        val length = readIntLittleEndian(input)
                        require(length in 1..(config.tokenizerConfig.maxTokenLength * 4)) {
                            "ARG3 token byte length is invalid."
                        }
                        entries.add(
                            String(input.readFully(length), Charsets.UTF_8) to id
                        )
                    }
                    val candidateTokenizer = Tokenizer(config.tokenizerConfig)
                    candidateTokenizer.importVocab(entries)
                    require(candidateTokenizer.vocabSize == savedVocabSize) {
                        "ARG3 tokenizer vocabulary size does not match its metadata."
                    }

                    synchronized(modelLock) {
                        tokenizer.importVocab(entries)
                        resizeVocabCapacityTo(savedVocabCap)
                        val tensors = listOf(
                            NamedCheckpointTensor("token.embedding.weight", wEmbed),
                            NamedCheckpointTensor("token.embedding.adam_m", mEmbed),
                            NamedCheckpointTensor("token.embedding.adam_v", vEmbed),
                            NamedCheckpointTensor("lm_head.weight", wHead),
                            NamedCheckpointTensor("lm_head.adam_m", mHead),
                            NamedCheckpointTensor("lm_head.adam_v", vHead),
                            NamedCheckpointTensor("adapter.q.a", loraQa),
                            NamedCheckpointTensor("adapter.q.b", loraQb),
                            NamedCheckpointTensor("adapter.v.a", loraVa),
                            NamedCheckpointTensor("adapter.v.b", loraVb),
                            NamedCheckpointTensor("adapter.q.a.adam_m", mLoraQa),
                            NamedCheckpointTensor("adapter.q.b.adam_m", mLoraQb),
                            NamedCheckpointTensor("adapter.v.a.adam_m", mLoraVa),
                            NamedCheckpointTensor("adapter.v.b.adam_m", mLoraVb),
                            NamedCheckpointTensor("adapter.q.a.adam_v", vLoraQa),
                            NamedCheckpointTensor("adapter.q.b.adam_v", vLoraQb),
                            NamedCheckpointTensor("adapter.v.a.adam_v", vLoraVa),
                            NamedCheckpointTensor("adapter.v.b.adam_v", vLoraVb),
                            NamedCheckpointTensor("attention.query.weight", wQ),
                            NamedCheckpointTensor("attention.key.weight", wK),
                            NamedCheckpointTensor("attention.value.weight", wV),
                            NamedCheckpointTensor("attention.output.weight", wO),
                            NamedCheckpointTensor("attention.query.adam_m", mWQ),
                            NamedCheckpointTensor("attention.query.adam_v", vWQ),
                            NamedCheckpointTensor("attention.key.adam_m", mWK),
                            NamedCheckpointTensor("attention.key.adam_v", vWK),
                            NamedCheckpointTensor("attention.value.adam_m", mWV),
                            NamedCheckpointTensor("attention.value.adam_v", vWV),
                            NamedCheckpointTensor("attention.output.adam_m", mWO),
                            NamedCheckpointTensor("attention.output.adam_v", vWO),
                            NamedCheckpointTensor("feed_forward.gate.weight", wGate),
                            NamedCheckpointTensor("feed_forward.up.weight", wUp),
                            NamedCheckpointTensor("feed_forward.down.weight", wDown),
                            NamedCheckpointTensor("feed_forward.gate.adam_m", mWGate),
                            NamedCheckpointTensor("feed_forward.gate.adam_v", vWGate),
                            NamedCheckpointTensor("feed_forward.up.adam_m", mWUp),
                            NamedCheckpointTensor("feed_forward.up.adam_v", vWUp),
                            NamedCheckpointTensor("feed_forward.down.adam_m", mWDown),
                            NamedCheckpointTensor("feed_forward.down.adam_v", vWDown)
                        )
                        tensors.forEach { tensor ->
                            readFloatArrayLittleEndian(input, tensor)
                        }
                        adamStep = savedAdamStep
                    }
                    true
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "ARG3 checkpoint load failed", e)
            false
        }
    }

    fun loadCheckpointIfExists(): Boolean {
        val file = File(context.filesDir, "argus_model_weights.bin")
        if (!file.isFile || file.length() < 4L) return false
        return try {
            val magic = FileInputStream(file).use { it.readFully(4) }
            when {
                magic[0] == MAGIC_0 && magic[1] == MAGIC_1 &&
                    magic[2] == MAGIC_2 && magic[3] == MAGIC_3 ->
                    loadLegacyCheckpoint(file)

                magic[0] == MAGIC_0 && magic[1] == MAGIC_1 &&
                    magic[2] == MAGIC_2 && magic[3] == MAGIC_4 ->
                    loadArg4Checkpoint(file)

                else -> {
                    Log.w(TAG, "Checkpoint magic is not supported.")
                    false
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Checkpoint header read failed", e)
            false
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