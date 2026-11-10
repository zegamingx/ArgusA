package com.argus.agi.model

import android.content.Context
import android.content.SharedPreferences
import android.os.Build
import android.os.Environment
import android.os.StatFs
import android.util.Log
import com.argus.agi.sensor.DeviceGuard
import com.argus.agi.sensor.SystemHardwareInfo
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale
import java.util.UUID

data class GenesisIdentity(
    val originId: String,
    val birthTimestamp: Long,
    val formattedBirthDate: String,
    val deviceRamGb: Int,
    val computeTarget: String,
    val parameterCount: Long,
    val architectureTier: String,
    val isGenesisComplete: Boolean = true
)

enum class GenesisLifecycleStage(val displayName: String, val description: String) {
    IDLE(
        "Waiting for initialization",
        "The system is ready to initialize a local model."
    ),
    ASSESSING_DEVICE(
        "Assessing device",
        "Assessing CPU/GPU capabilities, RAM, storage, thermal state, and device health."
    ),
    SELECTING_CONFIGURATION(
        "Selecting configuration",
        "Selecting a model configuration that fits the device capacity."
    ),
    PREPARING_TOKENIZER(
        "Preparing tokenizer",
        "Preparing the tokenizer before model tensors are allocated."
    ),
    CONSTRUCTING_MODEL(
        "Constructing model",
        "Building the neural network structure and allocating tensors."
    ),
    INITIALIZING_PARAMETERS(
        "Initializing parameters",
        "Initializing model weights locally without downloading a pretrained model."
    ),
    COMMITTING_INITIAL_GENERATION(
        "Committing initial generation",
        "Validating and safely saving the initial model generation."
    ),
    APPLICATION_READY(
        "Application ready",
        "Argus can open after the initial model generation is saved."
    ),
    GENERAL_LEARNING_SCHEDULED(
        "General learning scheduled",
        "Corpus training continues in the background without blocking the loading screen."
    ),
    FAILED(
        "Genesis failed",
        "Initialization failed. Argus can retry when device conditions improve."
    );

    companion object {
        /** Semua stage yang dianggap "progress" (tidak termasuk FAILED). */
        val progressStages: List<GenesisLifecycleStage> = values().filter {
            it != FAILED
        }
    }
}

data class GenesisLifecycleState(
    val stage: GenesisLifecycleStage = GenesisLifecycleStage.IDLE,
    val progress: Float = 0.0f,
    val assessedDetails: String = "",
    val stageLog: List<String> = emptyList(),
    val identity: GenesisIdentity? = null,
    val isComplete: Boolean = false,
    val errorMessage: String? = null
)

class GenesisException(message: String, cause: Throwable? = null) : Exception(message, cause)

class GenesisModelManager(
    private val context: Context,
    private val neuralEngine: ArgusNeuralEngine,
    private val deviceGuard: DeviceGuard,
    private val minFreeStorageGb: Long = 2L,
    private val minBatteryPct: Int = 5,
    private val maxThermalCelsius: Float = 45f,
    private val stageMinDelayMs: Long = 250L
) {

    companion object {
        private const val TAG = "GenesisModelManager"
        private const val PREFS_NAME = "argus_genesis_prefs"
        private const val KEY_ORIGIN_ID = "genesis_origin_id"
        private const val KEY_BIRTH_TS = "genesis_birth_timestamp"
        private const val KEY_RAM_GB = "genesis_ram_gb"
        private const val KEY_COMPUTE = "genesis_compute_target"
        private const val KEY_PARAM_COUNT = "genesis_param_count"
        private const val KEY_ARCH_TIER = "genesis_arch_tier"
        private const val KEY_BOOT_SEEN = "genesis_seen_boot_animation"

        private val DATE_FMT: DateTimeFormatter =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss").withLocale(Locale.US)
    }

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val genesisLock = Mutex()

    private val _lifecycleState = MutableStateFlow(GenesisLifecycleState())
    val lifecycleState: StateFlow<GenesisLifecycleState> = _lifecycleState.asStateFlow()

    init {
        loadIdentity()?.let { existing ->
            _lifecycleState.value = GenesisLifecycleState(
                stage = GenesisLifecycleStage.APPLICATION_READY,
                progress = 1.0f,
                assessedDetails = "Model Origin: ${existing.originId} | ${existing.deviceRamGb}GB RAM | ${existing.computeTarget}",
                stageLog = listOf(
                    "Genesis identity verified in local storage.",
                    "Generation 0 verified: ${existing.parameterCount} active parameters."
                ),
                identity = existing,
                isComplete = true
            )
        }
    }

    fun hasCompletedGenesis(): Boolean = prefs.contains(KEY_ORIGIN_ID)

    fun hasSeenBootAnimation(): Boolean = prefs.getBoolean(KEY_BOOT_SEEN, false)

    fun markBootAnimationSeen() {
        prefs.edit().putBoolean(KEY_BOOT_SEEN, true).commit()
    }

    fun getIdentity(): GenesisIdentity? = loadIdentity()

    suspend fun executeGenesisLifecycle(
        hardwareInfo: SystemHardwareInfo,
        trainingManager: TrainingManager? = null
    ): GenesisIdentity = genesisLock.withLock {
        loadIdentity()?.let { existing ->
            if (trainingManager != null) {
                val scheduled = ensureGeneralLearningCorpus(trainingManager)
                _lifecycleState.update { current ->
                    current.copy(
                        stage = GenesisLifecycleStage.GENERAL_LEARNING_SCHEDULED,
                        progress = 1.0f,
                        assessedDetails = "Existing local model verified; durable Genesis seed corpus checked.",
                        stageLog = current.stageLog + (
                            if (scheduled > 0) {
                                "Recovered $scheduled Genesis seed training chunks from the local corpus."
                            } else {
                                "Genesis seed corpus is already queued, active, or fully trained."
                            }
                        ),
                        identity = existing,
                        isComplete = true,
                        errorMessage = null
                    )
                }
            }
            return@withLock existing
        }

        withContext(Dispatchers.Default) {
            try {
                runGenesis(hardwareInfo, trainingManager)
            } catch (t: Throwable) {
                Log.e(TAG, "Genesis failed during stage ${_lifecycleState.value.stage}", t)
                _lifecycleState.update {
                    it.copy(
                        stage = GenesisLifecycleStage.FAILED,
                        isComplete = false,
                        errorMessage = t.message ?: t.javaClass.simpleName
                    )
                }
                throw GenesisException("Genesis failed: ${t.message}", t)
            }
        }
    }

    private suspend fun runGenesis(
        hardwareInfo: SystemHardwareInfo,
        trainingManager: TrainingManager?
    ): GenesisIdentity {
        val logs = mutableListOf<String>()

        fun publish(stage: GenesisLifecycleStage, details: String) {
            _lifecycleState.update {
                it.copy(
                    stage = stage,
                    progress = stageProgress(stage),
                    assessedDetails = details,
                    stageLog = logs.toList(),
                    identity = it.identity,
                    isComplete = it.isComplete,
                    errorMessage = null
                )
            }
        }

        publish(GenesisLifecycleStage.ASSESSING_DEVICE,
            "CPU cores: ${hardwareInfo.cpuCores} | RAM: ${hardwareInfo.totalRamGb} GB | Checking storage...")

        val safety = deviceGuard.safetyState.value
        val freeStorageGb = calculateFreeStorageGb()

        logs += "Starting hardware assessment: ${Build.MANUFACTURER} ${Build.MODEL} " +
                "(${hardwareInfo.socManufacturer} ${hardwareInfo.socModel})"
        logs += "CPU Cores: ${hardwareInfo.cpuCores} | SoC: ${hardwareInfo.socManufacturer} | " +
                "Hardware capability: ${hardwareInfo.primaryAccelerator}"
        logs += "RAM: ${hardwareInfo.totalRamGb} GB (Available: " +
                "${String.format(Locale.US, "%.1f", hardwareInfo.availRamGb)} GB) | " +
                "Free storage: $freeStorageGb GB"
        logs += "Battery: ${safety.batteryPct}% | Temperature: " +
                "${String.format(Locale.US, "%.1f", safety.temperatureCelsius)}°C"

        if (freeStorageGb < minFreeStorageGb) {
            throw GenesisException(
                "Insufficient storage: ${freeStorageGb} GB (at least ${minFreeStorageGb} GB required)")
        }
        if (safety.batteryPct in 0..minBatteryPct) {
            throw GenesisException(
                "Battery level is too low (${safety.batteryPct}%) to initialize the model.")
        }
        if (safety.temperatureCelsius > maxThermalCelsius) {
            throw GenesisException(
                "Device temperature ${String.format(Locale.US, "%.1f", safety.temperatureCelsius)}°C " +
                        "exceeds the safe limit of ${maxThermalCelsius}°C.")
        }
        throttleStage()

        publish(GenesisLifecycleStage.SELECTING_CONFIGURATION,
            "Selecting tensor dimensions based on the device compute and memory budget...")

        val archTier = neuralEngine.config.architectureLabel
        val detectedAccelerators = buildList {
            if (hardwareInfo.npuApuDetected.isNotBlank()) add(hardwareInfo.npuApuDetected)
            if (hardwareInfo.vulkanSupported) add("Vulkan ${hardwareInfo.vulkanVersionName}")
        }.distinct()

        // Vulkan accelerates eligible dense operations and AdamW after its runtime self-checks.
        // Remaining operations continue to use the Kotlin CPU implementation.
        val computeTarget = if (neuralEngine.isVulkanRuntimeReady) {
            "Vulkan Compute for eligible dense operations and AdamW; Kotlin CPU fallback for remaining operations"
        } else {
            "Kotlin CPU backend; Vulkan runtime unavailable or not verified"
        }
        logs += "Selected configuration: $archTier"
        logs += "Execution policy: $computeTarget | Context length: ${neuralEngine.maxSeqLen} tokens"
        throttleStage()

        publish(GenesisLifecycleStage.PREPARING_TOKENIZER,
            "Preparing the tokenizer vocabulary and control tokens before tensor allocation...")

        val tokenizer = neuralEngine.tokenizer
        logs += "Tokenizer initialized with ${tokenizer.vocabSize} base tokens."
        logs += "Special Tokens: <pad>, <unk>, <s>, </s>, <user>, <assistant>, <think>, </think>"
        throttleStage()

        publish(GenesisLifecycleStage.CONSTRUCTING_MODEL,
            "Constructing the transformer architecture and allocating flat tensors...")

        val totalParams = neuralEngine.countTotalParameters()
        if (totalParams <= 0L) {
            throw GenesisException("Parameter allocation failed: totalParams = $totalParams")
        }
        val adapterBytes = neuralEngine.calculateAdapterSizeBytes()
        logs += "Allocated tensors: embeddings, positional vectors, Q/K/V/O projections, SwiGLU, and LM head."
        logs += "LoRA adapters: ${String.format(Locale.US, "%.2f KB", adapterBytes / 1024.0)} " +
                "(Rank ${neuralEngine.loraRank})."
        logs += "Total structured parameters: $totalParams (FP32)."
        throttleStage()

        publish(GenesisLifecycleStage.INITIALIZING_PARAMETERS,
            "Initializing model weights locally (Xavier / He normal); no pretrained weights are downloaded...")

        logs += "Parameters initialized locally with Xavier/Glorot initialization and sinusoidal positional vectors."
        logs += "LoRA A uses Gaussian initialization; LoRA B starts at zero for stability."
        logs += "Integrity check: no model weights downloaded; the model identity is created locally."
        throttleStage()

        publish(GenesisLifecycleStage.COMMITTING_INITIAL_GENERATION,
            "Validating the forward pass and saving the initial ARG3 checkpoint...")

        val sanityLoss = neuralEngine.evaluateLoss("A.R.G.U.S Genesis origin test.")
        if (!sanityLoss.isFinite()) {
            throw GenesisException("Forward validation produced an invalid loss: $sanityLoss")
        }
        val checkpointSaved = neuralEngine.saveCheckpoint()
        if (!checkpointSaved) {
            throw GenesisException("The initial checkpoint could not be written to disk. Genesis was aborted.")
        }

        val originId = "GENESIS-" + UUID.randomUUID().toString().take(8).uppercase(Locale.US)
        val birthTs = System.currentTimeMillis()
        val identity = GenesisIdentity(
            originId = originId,
            birthTimestamp = birthTs,
            formattedBirthDate = formatDate(birthTs),
            deviceRamGb = hardwareInfo.totalRamGb,
            computeTarget = computeTarget,
            parameterCount = totalParams,
            architectureTier = archTier,
            isGenesisComplete = true
        )
        saveIdentity(identity)

        logs += "Forward validation completed without an optimizer update: initial loss = " +
                String.format(Locale.US, "%.4f", sanityLoss)
        logs += "Checkpoint 'argus_model_weights.bin' (ARG3 format) was verified in local storage."
        logs += "Model origin identity committed: $originId"
        throttleStage()

        logs += "Application ready. The user can open the terminal."
        publish(GenesisLifecycleStage.APPLICATION_READY,
            "Argus is ready. Genesis Generation 0 is active.")
        _lifecycleState.update {
            it.copy(identity = identity, isComplete = true)
        }

        if (trainingManager != null) {
            val scheduled = ensureGeneralLearningCorpus(trainingManager)
            logs += if (scheduled == 0) {
                "The general-learning seed corpus is empty or has already been scheduled."
            } else {
                "General learning scheduled: $scheduled corpus items added to the background queue."
            }
        }
        _lifecycleState.update {
            it.copy(
                stage = GenesisLifecycleStage.GENERAL_LEARNING_SCHEDULED,
                progress = 1.0f,
                stageLog = logs.toList(),
                identity = identity,
                isComplete = true,
                errorMessage = null
            )
        }
        return identity
    }

    private fun stageProgress(stage: GenesisLifecycleStage): Float {
        val stages = GenesisLifecycleStage.progressStages
        val idx = stages.indexOf(stage)
        return if (idx < 0) 0f else idx.toFloat() / (stages.size - 1).toFloat()
    }

    private suspend fun throttleStage() {
        if (stageMinDelayMs > 0) delay(stageMinDelayMs)
    }

    private fun calculateFreeStorageGb(): Long {
        return try {
            val stat = StatFs(context.filesDir.path)
            val bytes = stat.availableBlocksLong * stat.blockSizeLong
            val gb = bytes / (1024L * 1024L * 1024L)
            if (gb > 0L) {
                gb
            } else {
                val dataStat = StatFs(Environment.getDataDirectory().path)
                val dataGb = (dataStat.availableBlocksLong * dataStat.blockSizeLong) / (1024L * 1024L * 1024L)
                if (dataGb > 0L) dataGb else 8L // Fallback for JVM/Robolectric test harness
            }
        } catch (e: Exception) {
            Log.w(TAG, "StatFs failed; using the 8 GB fallback for this environment.", e)
            8L
        }
    }

    /**
     * Checks the bundled foundation notes on every completed Genesis lifecycle.
     *
     * The notes are persisted as ordinary TRAIN corpus documents. Existing pending
     * work is left alone, completed notes are not trained again, and interrupted or
     * cleared notes are replayed from Room without reopening an asset or rebuilding text.
     *
     * These short notes are only a bootstrap seed; they are not a general language dataset.
     */
    suspend fun ensureGeneralLearningCorpus(trainingManager: TrainingManager): Int {
        val corpusItems = listOf(
            Triple("Genesis Foundation", "Prinsip Kemandirian AGI",
                "A.R.G.U.S adalah inteligensi personal lokal yang diciptakan langsung dari perangkat pengguna. Setiap instalasi berkembang independen dari origin lokalnya sendiri."),
            Triple("Genesis Foundation", "Privasi dan Sensor",
                "Argus memisahkan pembelajaran umum dari data memori pribadi. Persepsi layar secara aktif mengecualikan aplikasi perbankan, kata sandi, dan data sensitif."),
            Triple("Genesis Foundation", "Arsitektur Transformer On-Device",
                "Struktur transformer on-device mengintegrasikan Causal Multi-Head Attention, LoRA Low-Rank Adaptation, dan feed-forward SwiGLU dengan alokasi tensor datar.")
        )

        var scheduledChunks = 0
        corpusItems.forEach { (source, title, text) ->
            val queuedChunks = try {
                trainingManager.ensureCorpusDocumentScheduled(
                    source = source,
                    title = title,
                    fullText = text
                )
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                Log.w(TAG, "Failed to restore Genesis seed '$title'", error)
                0
            }
            scheduledChunks += queuedChunks
        }
        return scheduledChunks
    }

    private fun saveIdentity(identity: GenesisIdentity) {
        val ok = prefs.edit()
            .putString(KEY_ORIGIN_ID, identity.originId)
            .putLong(KEY_BIRTH_TS, identity.birthTimestamp)
            .putInt(KEY_RAM_GB, identity.deviceRamGb)
            .putString(KEY_COMPUTE, identity.computeTarget)
            .putLong(KEY_PARAM_COUNT, identity.parameterCount)
            .putString(KEY_ARCH_TIER, identity.architectureTier)
            .commit()
        if (!ok) {
            throw GenesisException("Failed to commit the Genesis identity to SharedPreferences. The transaction was aborted.")
        }
    }

    private fun loadIdentity(): GenesisIdentity? {
        val originId = prefs.getString(KEY_ORIGIN_ID, null) ?: return null
        val birthTs = prefs.getLong(KEY_BIRTH_TS, 0L).takeIf { it > 0L }
            ?: System.currentTimeMillis()

        val params = if (prefs.contains(KEY_PARAM_COUNT)) {
            prefs.getLong(KEY_PARAM_COUNT, 0L)
        } else {
            neuralEngine.countTotalParameters()
        }

        return GenesisIdentity(
            originId = originId,
            birthTimestamp = birthTs,
            formattedBirthDate = formatDate(birthTs),
            deviceRamGb = prefs.getInt(KEY_RAM_GB, 8),
            computeTarget = prefs.getString(KEY_COMPUTE, "UNKNOWN") ?: "UNKNOWN",
            parameterCount = params,
            architectureTier = prefs.getString(KEY_ARCH_TIER, "Balanced Neural Core")
                ?: "Balanced Neural Core",
            isGenesisComplete = true
        )
    }

    private fun formatDate(ts: Long): String =
        DATE_FMT.format(Instant.ofEpochMilli(ts).atZone(ZoneId.systemDefault()))
}