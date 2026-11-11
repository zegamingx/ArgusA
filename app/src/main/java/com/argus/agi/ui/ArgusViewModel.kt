package com.argus.agi.ui

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.argus.agi.ArgusApp
import com.argus.agi.model.BenchmarkResult
import com.argus.agi.model.GenesisLifecycleStage
import com.argus.agi.model.TrainingData
import com.argus.agi.model.TrainingManager
import com.argus.agi.model.ReasoningOutputParser
import com.argus.agi.model.TrainingSessionMetrics
import com.argus.agi.model.WorldEntity
import android.media.MediaMetadataRetriever
import com.argus.agi.sensor.ArticleExtractor
import com.argus.agi.sensor.ExtractedWebArticle
import com.argus.agi.sensor.ExtractedDocument
import com.argus.agi.sensor.StructuredDocumentExtractor
import com.argus.agi.sensor.AudioVoiceManager
import com.argus.agi.sensor.DeviceSafetyState
import com.argus.agi.sensor.ExcludedAppsManager
import com.argus.agi.sensor.InstalledAppItem
import com.argus.agi.sensor.SearchResultItem
import com.argus.agi.sensor.SystemHardwareInfo
import com.argus.agi.sensor.WebSearchEngine
import com.argus.agi.service.ArgusAccessibilityService
import com.argus.agi.service.ArgusForegroundService
import com.argus.agi.skill.SkillRegistry
import com.argus.agi.skill.WebArticleExtractionSkill
import com.argus.agi.skill.WebArticleExtractionSkillOutput
import com.argus.agi.skill.WebSearchIntentRouter
import com.argus.agi.skill.WebSearchSkill
import com.argus.agi.skill.WebSearchSkillOutput
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.BufferedInputStream
import java.io.BufferedReader
import java.io.FilterInputStream
import java.io.InputStream
import java.io.InputStreamReader
import java.io.StringReader
import java.io.IOException
import java.security.MessageDigest
import java.net.URLEncoder
import java.nio.charset.StandardCharsets
import java.util.Locale
import java.util.UUID
import java.util.concurrent.atomic.AtomicBoolean
import java.util.zip.ZipInputStream

data class ChatMessage(
    val id: String = UUID.randomUUID().toString(),
    val isUser: Boolean,
    val text: String,
    val thinkReasoning: String? = null,
    val attachedFileName: String? = null,
    val attachedImageUri: Uri? = null,
    val timestamp: Long = System.currentTimeMillis()
)

class ArgusViewModel(application: Application) : AndroidViewModel(application) {

    companion object {
        private const val TAG = "ArgusViewModel"
        private const val MAX_MESSAGES = 200
        private const val ABSOLUTE_MAX_FILE_SIZE_BYTES = 100L * 1024L * 1024L
        private const val MAX_IMAGE_SIDE = 2048
        private const val MAX_ZIP_ENTRIES = 1_000
        private const val ABSOLUTE_MAX_ZIP_ENTRY_EXPANDED_BYTES = 20L * 1024L * 1024L
        private const val ABSOLUTE_MAX_ZIP_EXPANDED_TOTAL_BYTES = 40L * 1024L * 1024L
    }

    private val neuralEngine = ArgusApp.neuralEngine
    private val worldModel = ArgusApp.worldModel
    private val trainingManager = ArgusApp.trainingManager
    private val deviceGuard = ArgusApp.deviceGuard
    private val hardwareProfile = ArgusApp.hardwareProfile
    private val genesisModelManager = ArgusApp.genesisModelManager
    private val excludedAppsManager = ExcludedAppsManager(application)
    private val webSearchEngine = WebSearchEngine()
    private val articleExtractor = ArticleExtractor()
    private val skillRegistry = SkillRegistry(
        listOf(
            WebSearchSkill(webSearchEngine),
            WebArticleExtractionSkill(articleExtractor)
        )
    )

    private val repository = ArgusApp.repository

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    private val _isMessagesLoaded = MutableStateFlow(false)
    val isMessagesLoaded: StateFlow<Boolean> = _isMessagesLoaded.asStateFlow()

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    private val _hardwareInfo = MutableStateFlow(hardwareProfile.getHardwareInfo())
    val hardwareInfo: StateFlow<SystemHardwareInfo> = _hardwareInfo.asStateFlow()

    val safetyState: StateFlow<DeviceSafetyState> = deviceGuard.safetyState
    val isForceOverrideActive: StateFlow<Boolean> = deviceGuard.forceTrainingOverride
    val currentTrainingItem: StateFlow<TrainingData?> = trainingManager.currentTrainingItem
    val nextQueue: StateFlow<List<TrainingData>> = trainingManager.nextQueue
    val completedQueue: StateFlow<List<TrainingData>> = trainingManager.completedQueue
    val lossHistory: StateFlow<List<Float>> = trainingManager.lossHistory
    val isTrainingPaused: StateFlow<Boolean> = trainingManager.isTrainingPaused
    val totalStepsExecuted: StateFlow<Int> = trainingManager.totalStepsExecuted
    val trainingMetrics: StateFlow<TrainingSessionMetrics> = trainingManager.trainingMetrics
    val engineConfig get() = neuralEngine.config
    val activeComputeBackend get() = neuralEngine.optimizerBackend
    val isVulkanRuntimeReady get() = neuralEngine.isVulkanRuntimeReady
    val checkpointInterval get() = neuralEngine.checkpointInterval
    val worldEntities: StateFlow<List<WorldEntity>> = worldModel.entitiesFlow
    val currentContextSummary: StateFlow<String> = worldModel.currentContextSummary
    val genesisState: StateFlow<com.argus.agi.model.GenesisLifecycleState> = genesisModelManager.lifecycleState
    val activeDocument: StateFlow<com.argus.agi.model.ActiveDocumentTraining?> = trainingManager.activeDocument

    private val _benchmarkState = MutableStateFlow<BenchmarkResult?>(null)
    val benchmarkState: StateFlow<BenchmarkResult?> = _benchmarkState.asStateFlow()

    private val _isBenchmarking = MutableStateFlow(false)
    val isBenchmarking: StateFlow<Boolean> = _isBenchmarking.asStateFlow()

    private val _installedApps = MutableStateFlow<List<InstalledAppItem>>(emptyList())
    val installedApps: StateFlow<List<InstalledAppItem>> = _installedApps.asStateFlow()

    private val _cameraLuminance = MutableStateFlow(0f)
    val cameraLuminance: StateFlow<Float> = _cameraLuminance.asStateFlow()

    private val _searchResults = MutableStateFlow<List<SearchResultItem>>(emptyList())
    val searchResults: StateFlow<List<SearchResultItem>> = _searchResults.asStateFlow()

    private val _articlePreview = MutableStateFlow<ExtractedWebArticle?>(null)
    val articlePreview: StateFlow<ExtractedWebArticle?> = _articlePreview.asStateFlow()

    private val _isExtractingArticle = MutableStateFlow(false)
    val isExtractingArticle: StateFlow<Boolean> = _isExtractingArticle.asStateFlow()

    private val _articleExtractionError = MutableStateFlow<String?>(null)
    val articleExtractionError: StateFlow<String?> = _articleExtractionError.asStateFlow()

    private val _isQueueingArticle = MutableStateFlow(false)
    val isQueueingArticle: StateFlow<Boolean> = _isQueueingArticle.asStateFlow()

    private val _articleQueueStatus = MutableStateFlow<String?>(null)
    val articleQueueStatus: StateFlow<String?> = _articleQueueStatus.asStateFlow()

    private val _isNotificationPermissionGranted = MutableStateFlow(false)
    val isNotificationPermissionGranted: StateFlow<Boolean> = _isNotificationPermissionGranted.asStateFlow()

    val audioVoiceManager = AudioVoiceManager(application, viewModelScope)

    val isSpeaking: StateFlow<Boolean> = audioVoiceManager.isSpeaking
    val isListeningVoice: StateFlow<Boolean> = audioVoiceManager.isListening
    val isAutoSpeakEnabled: StateFlow<Boolean> = audioVoiceManager.isAutoSpeakEnabled
    val speechRmsDb: StateFlow<Float> = audioVoiceManager.speechRmsDb
    val ambientDecibel: StateFlow<Float> = audioVoiceManager.ambientDecibel
    val isAcousticMonitoring: StateFlow<Boolean> = audioVoiceManager.isAcousticMonitoring
    val partialTranscription: StateFlow<String> = audioVoiceManager.partialTranscription

    fun toggleAutoSpeak() = audioVoiceManager.toggleAutoSpeak()
    fun speakMessage(text: String) = audioVoiceManager.speak(text)
    fun stopSpeaking() = audioVoiceManager.stopSpeaking()

    fun startVoiceInput(onResult: (String) -> Unit, onError: (String) -> Unit) {
        audioVoiceManager.startListening(
            onFinalResult = { text ->
                if (text.isNotBlank()) {
                    onResult(text)
                }
            },
            onError = onError
        )
    }

    fun stopVoiceInput() {
        audioVoiceManager.stopListening()
    }

    fun toggleAcousticMonitoring() {
        if (audioVoiceManager.isAcousticMonitoring.value) {
            audioVoiceManager.stopAcousticMonitoring()
        } else {
            audioVoiceManager.startAcousticMonitoring { db ->
                worldModel.updateTransientEntities(
                    WorldEntity(
                        "ambient_acoustic",
                        String.format(Locale.US, "%.1f dB relative (uncalibrated)", db),
                        "Microphone Acoustic Sensor"
                    )
                )
            }
        }
    }

    private val isSending = AtomicBoolean(false)
    private var generationJob: Job? = null

    init {
        runCatching { ArgusForegroundService.startService(application) }
            .onFailure { Log.e(TAG, "Failed to start foreground service", it) }

        // Muat riwayat percakapan dari Room database on-device
        viewModelScope.launch(Dispatchers.IO) {
            val savedMessages = runCatching { repository.getChatMessagesSnapshot() }.getOrDefault(emptyList())
            if (savedMessages.isNotEmpty()) {
                withContext(Dispatchers.Main) {
                    _messages.value = savedMessages.takeLast(MAX_MESSAGES)
                }
            } else {
                val existing = genesisModelManager.getIdentity()
                if (existing == null) {
                    try {
                        val identity = genesisModelManager.executeGenesisLifecycle(
                            _hardwareInfo.value, trainingManager
                        )
                        val initialMsg = ChatMessage(
                            isUser = false,
                            text = "A.R.G.U.S local model initialized on this device (${identity.originId}). " +
                                    "Model parameters are initialized locally and can continue to evolve through on-device training.",
                            thinkReasoning = "Genesis Origin: ${identity.originId} | " +
                                    "Created: ${identity.formattedBirthDate} | " +
                                    "Architecture: ${identity.architectureTier}. " +
                                    "Total parameters: ${identity.parameterCount} (FP32)."
                        )
                        withContext(Dispatchers.Main) {
                            appendMessage(initialMsg)
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Genesis lifecycle failed", e)
                    }
                } else {
                    val initialMsg = ChatMessage(
                        isUser = false,
                        text = "A.R.G.U.S local model initialized on this device (${existing.originId}). " +
                                "Model parameters are initialized locally and can continue to evolve through on-device training.",
                        thinkReasoning = "Genesis Origin: ${existing.originId} | " +
                                "Created: ${existing.formattedBirthDate} | " +
                                "Architecture: ${existing.architectureTier}. " +
                                "Total parameters: ${existing.parameterCount} (FP32)."
                    )
                    withContext(Dispatchers.Main) {
                        appendMessage(initialMsg)
                    }
                }
            }
            withContext(Dispatchers.Main) {
                _isMessagesLoaded.value = true
            }
        }

        // Existing installations can have a Genesis identity and chat history while
        // their original seed work was never persisted in the retained corpus.
        if (genesisModelManager.hasCompletedGenesis()) {
            viewModelScope.launch(Dispatchers.IO) {
                runCatching {
                    genesisModelManager.ensureGeneralLearningCorpus(trainingManager)
                }.onFailure { error ->
                    Log.w(TAG, "Genesis seed corpus recovery failed", error)
                }
            }
        }

        loadInstalledApps()
    }

    private fun appendMessage(msg: ChatMessage) {
        _messages.update { (it + msg).takeLast(MAX_MESSAGES) }
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { repository.saveChatMessage(msg) }
        }
    }

    fun clearChatHistory() {
        _messages.value = emptyList()
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { repository.clearChatMessages() }
        }
    }

    /**
     * Refreshes the volatile free-memory snapshot without adding a chat message.
     * Model capacity is based on currently available RAM, so this is called when
     * the Neural Core screen becomes visible.
     */
    fun refreshDeviceMemorySnapshot() {
        viewModelScope.launch(Dispatchers.Default) {
            runCatching { hardwareProfile.getHardwareInfo() }
                .onSuccess { info -> _hardwareInfo.value = info }
                .onFailure { error ->
                    Log.w(TAG, "Device memory snapshot refresh failed", error)
                }
        }
    }

    fun refreshHardwareAssessment() {
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val info = hardwareProfile.getHardwareInfo()
                _hardwareInfo.value = info
                appendMessage(
                    ChatMessage(
                        isUser = false,
                        text = "Hardware assessment refreshed. SoC: ${info.socManufacturer} ${info.socModel}, " +
                                "RAM: ${info.totalRamGb} GB, reported capability: ${info.primaryAccelerator}.",
                        thinkReasoning = "Hardware assessment refreshed. The model origin identity was not changed."
                    )
                )
            } catch (e: Exception) {
                Log.e(TAG, "Hardware refresh failed", e)
            }
        }
    }

    fun rerunGenesisAssessment() = refreshHardwareAssessment()

    /** Retry genesis lifecycle. Aman dipanggil berkali-kali (idempotent via Mutex). */
    fun retryGenesis() {
        viewModelScope.launch {
            runCatching {
                genesisModelManager.executeGenesisLifecycle(
                    _hardwareInfo.value, trainingManager
                )
            }.onFailure { e ->
                Log.e(TAG, "Genesis retry failed", e)
            }
        }
    }

    fun getTotalParameters(): Long = runCatching { neuralEngine.countTotalParameters() }.getOrDefault(0L)
    fun getAdapterSizeBytes(): Long = runCatching { neuralEngine.calculateAdapterSizeBytes() }.getOrDefault(0L)
    fun getVocabSize(): Int = runCatching { neuralEngine.tokenizer.vocabSize }.getOrDefault(0)

    fun hasSeenGenesisBoot(): Boolean = genesisModelManager.hasSeenBootAnimation()
    fun markGenesisBootSeen() { genesisModelManager.markBootAnimationSeen() }

    fun setForceTrainingOverride(enabled: Boolean) = deviceGuard.setForceTrainingOverride(enabled)
    fun triggerImmediateTrainingStep() = trainingManager.triggerImmediateTrainingStep()
    fun toggleTrainingPause() = trainingManager.togglePauseTraining()

    fun onNotificationPermissionResult(granted: Boolean) {
        _isNotificationPermissionGranted.value = granted
    }

    fun runBenchmark() {
        if (!_isBenchmarking.compareAndSet(false, true)) return
        viewModelScope.launch {
            try {
                _benchmarkState.value = neuralEngine.runBenchmark(iterations = 12)
            } catch (e: Exception) {
                Log.e(TAG, "Benchmark failed", e)
                appendMessage(
                    ChatMessage(
                        isUser = false,
                        text = "Benchmark failed: ${e.message}",
                        thinkReasoning = "Exception while running the benchmark."
                    )
                )
            } finally {
                _isBenchmarking.value = false
            }
        }
    }

    fun sendMessage(text: String, attachedUri: Uri? = null, isImage: Boolean = false) {
        if (text.isBlank() && attachedUri == null) return

        val stage = genesisState.value.stage
        if (!genesisModelManager.hasCompletedGenesis() &&
            stage != GenesisLifecycleStage.APPLICATION_READY &&
            stage != GenesisLifecycleStage.GENERAL_LEARNING_SCHEDULED) {
            appendMessage(
                ChatMessage(
                    isUser = false,
                    text = "Genesis is running (${stage.displayName}). " +
                            "Wait for initialization to finish before starting a conversation."
                )
            )
            return
        }

        if (attachedUri != null && !isImage) {
            importFileContent(attachedUri)
            return
        }

        if (!isSending.compareAndSet(false, true)) {
            appendMessage(
                ChatMessage(
                    isUser = false,
                    text = "Wait for the previous response to finish, or stop it."
                )
            )
            return
        }

        var attachedImage: Uri? = null
        if (attachedUri != null && isImage) attachedImage = attachedUri

        val userMsg = ChatMessage(
            isUser = true,
            text = text,
            attachedImageUri = attachedImage
        )
        appendMessage(userMsg)

        _isGenerating.value = true
        generationJob = viewModelScope.launch {
            try {
                val worldSummary = worldModel.currentContextSummary.value
                val priorMessages = _messages.value.takeLast(6).filter { it.text.isNotBlank() }
                val multiTurnPrompt = buildString {
                    if (worldSummary.isNotBlank()) {
                        append("<SYSTEM> ").append(worldSummary.take(120)).append(" ")
                    }
                    priorMessages.forEach { msg ->
                        if (msg.isUser) append("<USER> ").append(msg.text).append(" ")
                        else append("<ASSISTANT> ").append(msg.text).append(" <EOS> ")
                    }
                    append("<USER> ").append(text).append(" <ASSISTANT>")
                }

                val explicitWebQuery = WebSearchIntentRouter.extractQuery(text)
                val mathResult = tryEvaluateMath(text)

                val rawOutput = try {
                    when {
                        explicitWebQuery != null ->
                            "Web search request: $explicitWebQuery"
                        mathResult != null -> mathResult
                        else -> neuralEngine.generate(
                            multiTurnPrompt,
                            maxNewTokens = 60,
                            includeReasoningMarkers = true
                        )
                    }
                } catch (e: Exception) {
                    Log.e(TAG, "Generation failed", e)
                    "The local model is busy or encountered an error (${e.javaClass.simpleName}). " +
                            "Please try again shortly."
                }

                val parsedOutput = ReasoningOutputParser.parse(rawOutput)
                val reflection = worldModel.reflectOnResponse(text, parsedOutput.answer)

                var finalAnswer = parsedOutput.answer
                val thinking = StringBuilder()
                parsedOutput.reasoning?.takeIf { it.isNotBlank() }?.let { modelReasoning ->
                    thinking.append("Model reasoning:\n")
                        .append(modelReasoning)
                        .append("\n\n")
                }
                thinking.append("Self-Reflection: ").append(reflection.reasoningNote)
                thinking.append(" | Confidence: ")
                    .append(String.format("%.0f%%", reflection.confidenceScore * 100))

                val shouldSearchWeb = explicitWebQuery != null ||
                    !reflection.isConfident ||
                    reflection.suggestedAction == "AUTO_LEARN_QUERY"

                if (shouldSearchWeb) {
                    val searchQuery = explicitWebQuery ?: text
                    thinking.append(
                        if (explicitWebQuery != null) {
                            "\n→ Explicit web-search request matched; Web Search skill invoked directly."
                        } else {
                            "\n→ Confidence-based web retrieval initiated; sources require article review before local training."
                        }
                    )

                    val webItems = runCatching { executeWebSearchSkill(searchQuery) }
                        .onFailure { Log.w(TAG, "Web search failed", it) }
                        .getOrDefault(emptyList())

                    _searchResults.value = webItems
                    if (webItems.isNotEmpty()) {
                        finalAnswer = "I found several web sources for \"$searchQuery\". " +
                                "The items below are search snippets, not full articles, and have not been added to training:\n\n" +
                                webItems.take(2).joinToString("\n\n") {
                                    "• ${it.title}\n${it.snippet}\n${it.sourceUrl}"
                                } +
                                "\n\nOpen Skills & Settings to extract an article, review its source, and confirm usage rights before adding it to the local training queue."
                    } else {
                        finalAnswer = "Web search did not find usable sources for \"$searchQuery\". " +
                                "Search queries alone are not stored as training data."
                    }
                }

                appendMessage(
                    ChatMessage(
                        isUser = false,
                        text = finalAnswer,
                        thinkReasoning = thinking.toString()
                    )
                )

                if (audioVoiceManager.isAutoSpeakEnabled.value) {
                    audioVoiceManager.speak(finalAnswer)
                }
            } finally {
                isSending.set(false)
                _isGenerating.value = false
                generationJob = null
            }
        }
    }

    fun stopGeneration() {
        generationJob?.cancel()
        generationJob = null
        isSending.set(false)
        _isGenerating.value = false
    }

    fun importFileContent(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val cr = getApplication<Application>().contentResolver

                val sizeBytes: Long? = runCatching {
                    cr.openFileDescriptor(uri, "r")?.use { it.statSize }
                }.getOrNull()

                val maxFileSizeBytes = maxImportFileSizeBytes()
                if (sizeBytes != null && sizeBytes > maxFileSizeBytes) {
                    withContext(Dispatchers.Main) {
                        appendMessage(
                            ChatMessage(
                                isUser = false,
                                text = "File is too large (${sizeBytes / 1024 / 1024} MB). " +
                                    "The current device-memory-based import limit is " +
                                    "${maxFileSizeBytes / 1024 / 1024} MB.",
                                thinkReasoning = "Device-memory-aware import size guard triggered."
                            )
                        )
                    }
                    return@launch
                }

                val docDisplayName = runCatching {
                    var name: String? = null
                    cr.query(uri, null, null, null, null)?.use { cursor ->
                        val idx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                        if (idx != -1 && cursor.moveToFirst()) name = cursor.getString(idx)
                    }
                    name ?: "Document_${System.currentTimeMillis() % 10000}.txt"
                }.getOrDefault("Document_${System.currentTimeMillis() % 10000}.txt")

                val mimeType = cr.getType(uri).orEmpty().lowercase(Locale.ROOT)
                val extension = docDisplayName.substringAfterLast('.', "").lowercase(Locale.ROOT)
                val structuredFormat = when {
                    extension == "pdf" || mimeType in setOf("application/pdf", "application/x-pdf") -> "pdf"
                    extension == "docx" || mimeType.contains("wordprocessingml") -> "docx"
                    extension == "epub" || mimeType.contains("epub+zip") -> "epub"
                    extension in setOf("html", "htm", "xhtml", "xml") ||
                        mimeType in setOf("text/html", "application/xhtml+xml", "application/xml", "text/xml") -> "html"
                    else -> null
                }

                if (structuredFormat != null) {
                    val extracted = cr.openInputStream(uri)?.use { stream ->
                        when (structuredFormat) {
                            "pdf" -> StructuredDocumentExtractor.extractPdf(
                                context = getApplication<Application>(),
                                input = stream,
                                title = docDisplayName,
                                maxInputBytes = maxImportFileSizeBytes(),
                                maxTextCharacters = maxImportTextCharacters().toInt()
                            )
                            "docx" -> StructuredDocumentExtractor.extractDocx(
                                input = stream,
                                title = docDisplayName,
                                maxEntryBytes = maxQueuedTextMemoryBudgetBytes(),
                                maxExpandedBytes = maxQueuedTextMemoryBudgetBytes(),
                                maxTextCharacters = maxImportTextCharacters().toInt()
                            )
                            "epub" -> StructuredDocumentExtractor.extractEpub(
                                input = stream,
                                fallbackTitle = docDisplayName,
                                maxEntryBytes = maxQueuedTextMemoryBudgetBytes(),
                                maxExpandedBytes = maxQueuedTextMemoryBudgetBytes(),
                                maxStoredTextBytes = maxQueuedTextMemoryBudgetBytes(),
                                maxTextCharacters = maxImportTextCharacters().toInt()
                            )
                            else -> StructuredDocumentExtractor.extractHtml(
                                input = stream,
                                title = docDisplayName,
                                maxInputBytes = minOf(
                                    maxImportFileSizeBytes(),
                                    maxQueuedTextMemoryBudgetBytes() / 2L
                                ),
                                maxTextCharacters = maxImportTextCharacters().toInt()
                            )
                        }
                    } ?: throw IOException("Unable to open the file for text extraction.")

                    val (chunkCount, charsRead) = enqueueExtractedDocument(
                        extracted = extracted,
                        source = "local_structured_file|name=" +
                            encodeProvenanceField(docDisplayName) +
                            "|format=" + extracted.format
                    )

                    if (chunkCount == 0 || charsRead == 0L) {
                        withContext(Dispatchers.Main) {
                            appendMessage(
                                ChatMessage(
                                    isUser = false,
                                    text = "File '" + docDisplayName + "' did not produce any trainable text chunks.",
                                    thinkReasoning = "Structured document extractor returned empty content."
                                )
                            )
                        }
                        return@launch
                    }

                    withContext(Dispatchers.Main) {
                        appendMessage(
                            ChatMessage(
                                isUser = true,
                                text = "Importing " + extracted.format + " '" + extracted.title +
                                    "' (" + extracted.characterCount + " characters) for on-device training.",
                                attachedFileName = docDisplayName
                            )
                        )
                        appendMessage(
                            ChatMessage(
                                isUser = false,
                                text = extracted.format + " '" + extracted.title +
                                    "' was extracted into " + chunkCount +
                                    " training chunks. Text content remains on this device.",
                                thinkReasoning = extracted.format + " extraction: " +
                                    extracted.characterCount + " characters → " + chunkCount + " blocks."
                            )
                        )
                    }
                    return@launch
                }

                val isZip = extension == "zip" || mimeType.contains("zip")
                if (isZip) {
                    importZipContent(uri, docDisplayName)
                    return@launch
                }

                val isVideo = docDisplayName.endsWith(".mp4", ignoreCase = true) ||
                        docDisplayName.endsWith(".mkv", ignoreCase = true) ||
                        docDisplayName.endsWith(".webm", ignoreCase = true) ||
                        docDisplayName.endsWith(".mov", ignoreCase = true) ||
                        docDisplayName.endsWith(".3gp", ignoreCase = true) ||
                        cr.getType(uri)?.startsWith("video/") == true

                if (isVideo) {
                    importVideoContent(uri)
                    return@launch
                }

                // Streaming chunk ingestion: membaca langsung secara buffered (4KB)
                // tanpa pernah memuat seluruh file ke dalam sebuah String raksasa di heap RAM.
                val (chunkCount, charsRead) = cr.openInputStream(uri)?.use { stream ->
                    val boundedReader = BoundedReader(
                        InputStreamReader(stream, Charsets.UTF_8),
                        maxImportTextCharacters()
                    )
                    val reader = BufferedReader(boundedReader, 16_384)
                    trainingManager.enqueueDocumentStream(
                        source = "text_file|name=${encodeProvenanceField(docDisplayName)}",
                        title = docDisplayName,
                        reader = reader,
                        epochs = recommendedTrainingPasses(sizeBytes ?: 0L)
                    )
                } ?: Pair(0, 0L)

                if (chunkCount == 0 || charsRead == 0L) {
                    val reason = when {
                        charsRead == 0L -> "The file is empty or could not be read."
                        else -> "The content produced no new chunks. An identical document may already exist in the local corpus."
                    }
                    withContext(Dispatchers.Main) {
                        appendMessage(
                            ChatMessage(
                                isUser = false,
                                text = "Skipped '$docDisplayName'. $reason",
                                thinkReasoning = "Buffered import produced $charsRead characters and $chunkCount new training chunks."
                            )
                        )
                    }
                    return@launch
                }

                val sizeDisplay = if (sizeBytes != null && sizeBytes > 0) {
                    val mb = sizeBytes / (1024.0 * 1024.0)
                    if (mb >= 1.0) String.format(Locale.US, "%.1f MB", mb)
                    else String.format(Locale.US, "%.1f KB", sizeBytes / 1024.0)
                } else {
                    "$charsRead characters"
                }

                withContext(Dispatchers.Main) {
                    appendMessage(
                        ChatMessage(
                            isUser = true,
                            text = "Importing '$docDisplayName' ($sizeDisplay, $charsRead characters) for on-device training.",
                            attachedFileName = docDisplayName
                        )
                    )
                    appendMessage(
                        ChatMessage(
                            isUser = false,
                            text = "Document '$docDisplayName' ($sizeDisplay) was read incrementally.\n\n" +
                                    "The text was split into $chunkCount chunks and scheduled for local training. " +
                                    "Queued chunks are currently held in application memory, so large imports require sufficient available RAM.\n\n" +
                                    "Track progress in the Neural Core tab.",
                            thinkReasoning = "Buffered document ingestion: $charsRead characters → $chunkCount queued blocks."
                        )
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "importFileContent failed", e)
                withContext(Dispatchers.Main) {
                    appendMessage(
                        ChatMessage(
                            isUser = false,
                            text = "Failed to read the file: ${e.message}",
                            thinkReasoning = "I/O exception: ${e.javaClass.simpleName}"
                        )
                    )
                }
            }
        }
    }

    private fun maxQueuedTextMemoryBudgetBytes(): Long {
        val profile = _hardwareInfo.value
        val heapBytes = profile.maxHeapMb.coerceAtLeast(96L) * 1024L * 1024L
        val availableRamBytes = (
            profile.availRamGb.coerceAtLeast(0.25f).toDouble() *
                1024.0 * 1024.0 * 1024.0
            ).toLong()

        return minOf(
            heapBytes / 20L,
            availableRamBytes / 32L,
            32L * 1024L * 1024L
        ).coerceAtLeast(1L * 1024L * 1024L)
    }

    private fun maxImportFileSizeBytes(): Long =
        minOf(
            ABSOLUTE_MAX_FILE_SIZE_BYTES,
            maxQueuedTextMemoryBudgetBytes() * 2L
        )

    private fun maxImportTextCharacters(): Long =
        maxQueuedTextMemoryBudgetBytes() / 4L

    private fun maxZipEntryExpandedBytes(): Long =
        minOf(
            ABSOLUTE_MAX_ZIP_ENTRY_EXPANDED_BYTES,
            maxQueuedTextMemoryBudgetBytes()
        )

    private fun maxZipExpandedTotalBytes(): Long =
        minOf(
            ABSOLUTE_MAX_ZIP_EXPANDED_TOTAL_BYTES,
            maxQueuedTextMemoryBudgetBytes()
        )

    private suspend fun enqueueExtractedDocument(
        extracted: ExtractedDocument,
        source: String,
        epochs: Int = recommendedTrainingPasses(extracted.characterCount.toLong())
    ): Pair<Int, Long> {
        val provenance = buildString {
            append(source)
            append("|document_title=").append(encodeProvenanceField(extracted.title))
            append("|format=").append(extracted.format)
            append("|characters=").append(extracted.characterCount)
            append("|sha256=").append(sha256Hex(extracted.text))
            append("|extracted_at=").append(extracted.extractedAtEpochMs)
            append("|rights_status=unverified")
        }
        val reader = BufferedReader(
            BoundedReader(StringReader(extracted.text), maxImportTextCharacters()),
            16_384
        )
        return reader.use {
            trainingManager.enqueueDocumentStream(
                source = provenance,
                title = extracted.title,
                reader = it,
                chunkSize = 700,
                overlap = 80,
                epochs = epochs
            )
        }
    }

    private fun encodeProvenanceField(value: String): String =
        URLEncoder.encode(value, StandardCharsets.UTF_8.name())

    private fun sha256Hex(text: String): String {
        val digest = MessageDigest.getInstance("SHA-256")
            .digest(text.toByteArray(StandardCharsets.UTF_8))
        val alphabet = "0123456789abcdef"

        return buildString(digest.size * 2) {
            digest.forEach { byte ->
                val value = byte.toInt() and 0xFF
                append(alphabet[value ushr 4])
                append(alphabet[value and 0x0F])
            }
        }
    }

    fun importImageContent(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            var bitmap: Bitmap? = null
            try {
                val cr = getApplication<Application>().contentResolver
                cr.openInputStream(uri)?.use { stream ->
                    val opts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                    BitmapFactory.decodeStream(stream, null, opts)
                    val w = opts.outWidth
                    val h = opts.outHeight
                    if (w <= 0 || h <= 0) {
                        withContext(Dispatchers.Main) {
                            appendMessage(
                                ChatMessage(
                                    isUser = false,
                                    text = "The image is invalid or could not be read.",
                                    thinkReasoning = "BitmapFactory bounds decode failed."
                                )
                            )
                        }
                        return@use
                    }

                    val sample = maxOf(1, maxOf(w, h) / MAX_IMAGE_SIDE)
                    val decodeOpts = BitmapFactory.Options().apply { inSampleSize = sample }
                    val stream2 = cr.openInputStream(uri)
                    bitmap = stream2?.use { BitmapFactory.decodeStream(it, null, decodeOpts) }
                    stream2?.close()

                    val finalW = bitmap?.width ?: w
                    val finalH = bitmap?.height ?: h
                    val (avgLum, dominantTone, contrast) = bitmap?.let { bmp ->
                        analyzeBitmapMetrics(bmp)
                    } ?: Triple(50f, "Neutral", 1.0f)

                    val desc = "Image ${finalW}×${finalH} px imported for on-device visual perception. " +
                            "Average luminance: ${String.format(Locale.US, "%.1f", avgLum)}%, palette: $dominantTone, contrast: ${String.format(Locale.US, "%.1f", contrast)}:1."

                    withContext(Dispatchers.Main) {
                        sendMessage(desc, uri, isImage = true)
                    }

                    trainingManager.enqueueData(
                        source = "Image Vision Import",
                        title = "Visual $finalW×$finalH",
                        content = "On-device visual input: image resolution ${finalW}×${finalH} px. " +
                                "Average luminance: ${String.format(Locale.US, "%.1f", avgLum)}%, " +
                                "color palette: $dominantTone, " +
                                "contrast ratio: ${String.format(Locale.US, "%.1f", contrast)}:1."
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "importImageContent failed", e)
                withContext(Dispatchers.Main) {
                    appendMessage(
                        ChatMessage(
                            isUser = false,
                            text = "Failed to load the image: ${e.message}",
                            thinkReasoning = "Image decode exception."
                        )
                    )
                }
            } finally {
                bitmap?.recycle()
            }
        }
    }

    private fun analyzeBitmapMetrics(bmp: Bitmap): Triple<Float, String, Float> {
        val scaled = Bitmap.createScaledBitmap(bmp, 16, 16, true)
        var totalLum = 0f
        var rSum = 0L
        var gSum = 0L
        var bSum = 0L
        var minLum = 255f
        var maxLum = 0f

        for (y in 0 until 16) {
            for (x in 0 until 16) {
                val px = scaled.getPixel(x, y)
                val r = (px shr 16) and 0xFF
                val g = (px shr 8) and 0xFF
                val b = px and 0xFF
                rSum += r
                gSum += g
                bSum += b
                val lum = 0.299f * r + 0.587f * g + 0.114f * b
                totalLum += lum
                if (lum < minLum) minLum = lum
                if (lum > maxLum) maxLum = lum
            }
        }
        if (scaled != bmp) scaled.recycle()

        val avgLum = (totalLum / 256f) / 255f * 100f
        val contrast = if (minLum > 0f) maxLum / minLum else maxLum
        val dominant = when {
            rSum > gSum * 1.25f && rSum > bSum * 1.25f -> "Dominant red/warm"
            bSum > rSum * 1.25f && bSum > gSum * 1.25f -> "Dominant blue/cool"
            gSum > rSum * 1.25f && gSum > bSum * 1.25f -> "Dominant green/natural"
            kotlin.math.abs(rSum - gSum) < rSum * 0.15f && kotlin.math.abs(rSum - bSum) < rSum * 0.15f -> "Monochrome/balanced"
            else -> "Mixed spectrum"
        }
        return Triple(avgLum, dominant, contrast)
    }

    fun importZipContent(uri: Uri, displayName: String) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val contentResolver = getApplication<Application>().contentResolver
                var processedFiles = 0
                var failedFiles = 0
                var duplicateFilesSkipped = 0
                var totalChunksEnqueued = 0
                var totalCharsExtracted = 0L
                var totalExpandedBytes = 0L
                var entryCount = 0
                val maxZipEntryBytes = maxZipEntryExpandedBytes()
                val maxZipTotalBytes = maxZipExpandedTotalBytes()
                val maxTextCharacters = maxImportTextCharacters()

                val plainTextExtensions = setOf(
                    "txt", "md", "json", "jsonl", "csv", "tsv", "log",
                    "kt", "java", "py"
                )
                val structuredExtensions = setOf(
                    "pdf", "docx", "epub", "html", "htm", "xhtml", "xml"
                )

                contentResolver.openInputStream(uri)?.use { rawStream ->
                    ZipInputStream(BufferedInputStream(rawStream)).use { zipStream ->
                        var entry = zipStream.nextEntry
                        while (entry != null) {
                            entryCount++
                            if (entryCount > MAX_ZIP_ENTRIES) {
                                throw IOException(
                                    "ZIP archive exceeds the limit of " + MAX_ZIP_ENTRIES + " entri."
                                )
                            }

                            val normalizedName = entry.name.replace('\\', '/')
                            val pathSegments = normalizedName.split('/')
                            val unsafePath = normalizedName.startsWith("/") ||
                                pathSegments.any { it == ".." }
                            val extension = normalizedName
                                .substringAfterLast('.', "")
                                .lowercase(Locale.ROOT)
                            val isSupported = extension in plainTextExtensions ||
                                extension in structuredExtensions

                            val boundedEntry = LimitedNonClosingInputStream(
                                zipStream,
                                maxZipEntryBytes
                            )

                            if (unsafePath) {
                                Log.w(TAG, "Zip entry mencurigakan diabaikan: " + normalizedName)
                            } else if (!entry.isDirectory && isSupported) {
                                val entryTitle = normalizedName.substringAfterLast('/')
                                val source = "zip_file|archive=" +
                                    encodeProvenanceField(displayName) +
                                    "|entry=" + encodeProvenanceField(normalizedName) +
                                    "|format=" + extension

                                val extraction = runCatching {
                                    when (extension) {
                                        "pdf" -> enqueueExtractedDocument(
                                            StructuredDocumentExtractor.extractPdf(
                                                context = getApplication<Application>(),
                                                input = boundedEntry,
                                                title = entryTitle,
                                                maxInputBytes = maxZipEntryBytes / 2L,
                                                maxTextCharacters = maxTextCharacters.toInt()
                                            ),
                                            source
                                        )
                                        "docx" -> enqueueExtractedDocument(
                                            StructuredDocumentExtractor.extractDocx(
                                                input = boundedEntry,
                                                title = entryTitle,
                                                maxEntryBytes = maxZipEntryBytes,
                                                maxExpandedBytes = maxZipEntryBytes,
                                                maxTextCharacters = maxTextCharacters.toInt()
                                            ),
                                            source
                                        )
                                        "epub" -> enqueueExtractedDocument(
                                            StructuredDocumentExtractor.extractEpub(
                                                input = boundedEntry,
                                                fallbackTitle = entryTitle,
                                                maxEntryBytes = maxZipEntryBytes,
                                                maxExpandedBytes = maxZipEntryBytes,
                                                maxStoredTextBytes = maxZipEntryBytes,
                                                maxTextCharacters = maxTextCharacters.toInt()
                                            ),
                                            source
                                        )
                                        "html", "htm", "xhtml", "xml" -> enqueueExtractedDocument(
                                            StructuredDocumentExtractor.extractHtml(
                                                input = boundedEntry,
                                                title = entryTitle,
                                                maxInputBytes = maxZipEntryBytes / 2L,
                                                maxTextCharacters = maxTextCharacters.toInt()
                                            ),
                                            source
                                        )
                                        else -> {
                                            val reader = BufferedReader(
                                                BoundedReader(
                                                    InputStreamReader(boundedEntry, Charsets.UTF_8),
                                                    maxTextCharacters
                                                ),
                                                8_192
                                            )
                                            reader.use {
                                                trainingManager.enqueueDocumentStream(
                                                    source = source,
                                                    title = entryTitle,
                                                    reader = it,
                                                    epochs = recommendedTrainingPasses(entry.size)
                                                )
                                            }
                                        }
                                    }
                                }

                                if (boundedEntry.limitExceeded) {
                                    throw IOException(
                                        "A ZIP entry exceeds the limit of " +
                                            (maxZipEntryBytes / 1024L / 1024L) + " MB: " + normalizedName
                                    )
                                }

                                extraction.onSuccess { result ->
                                    when {
                                        result.first > 0 && result.second > 0L -> {
                                            processedFiles++
                                            totalChunksEnqueued += result.first
                                            totalCharsExtracted += result.second
                                        }
                                        result.first == 0 && result.second > 0L -> {
                                            duplicateFilesSkipped++
                                            totalCharsExtracted += result.second
                                            Log.i(TAG, "Corpus duplicate skipped: $normalizedName")
                                        }
                                        else -> {
                                            failedFiles++
                                            Log.w(TAG, "No importable text was found in " + normalizedName)
                                        }
                                    }
                                }.onFailure { error ->
                                    failedFiles++
                                    Log.w(TAG, "Failed to process ZIP entry " + normalizedName, error)
                                }
                            }

                            drainLimitedInput(boundedEntry)
                            if (boundedEntry.limitExceeded) {
                                throw IOException(
                                    "Arsip ZIP melewati batas " +
                                        (maxZipEntryBytes / 1024L / 1024L) + " MB pada entri " + normalizedName + "."
                                )
                            }

                            totalExpandedBytes += boundedEntry.bytesRead
                            if (totalExpandedBytes > maxZipTotalBytes) {
                                throw IOException(
                                    "ZIP archive exceeds the total expanded-size limit of " +
                                        (maxZipTotalBytes / 1024L / 1024L) + " MB."
                                )
                            }

                            zipStream.closeEntry()
                            entry = zipStream.nextEntry
                        }
                    }
                } ?: throw IOException("Unable to open the ZIP archive.")

                withContext(Dispatchers.Main) {
                    if (processedFiles == 0 && duplicateFilesSkipped == 0) {
                        appendMessage(
                            ChatMessage(
                                isUser = false,
                                text = "ZIP archive '" + displayName +
                                    "' did not contain importable text. " +
                                    "Supported formats: TXT, Markdown, JSON/JSONL, CSV/TSV, HTML, PDF, DOCX, and EPUB.",
                                thinkReasoning = "No supported archive entries produced extractable text."
                            )
                        )
                    } else {
                        val charsDisplay = String.format(Locale.US, "%,d", totalCharsExtracted)
                        val duplicateSummary = if (duplicateFilesSkipped > 0) {
                            "\nIdentical documents already stored: " + duplicateFilesSkipped + "."
                        } else {
                            ""
                        }
                        val failureSummary = if (failedFiles > 0) {
                            "\nFiles that failed or could not be extracted: " + failedFiles + "."
                        } else {
                            ""
                        }
                        appendMessage(
                            ChatMessage(
                                isUser = true,
                                text = "Importing ZIP dataset archive '" + displayName +
                                    "' (" + processedFiles + " files processed, " +
                                    charsDisplay + " characters).",
                                attachedFileName = displayName
                            )
                        )
                        appendMessage(
                            ChatMessage(
                                isUser = false,
                                text = "ZIP archive '" + displayName + "' was processed locally.\n\n" +
                                    "• Files processed: " + processedFiles + "\n" +
                                    "• Files failed or not extracted: " + failedFiles + "\n" +
                                    "• Duplicates skipped: " + duplicateFilesSkipped + "\n" +
                                    "• Total characters: " + charsDisplay + "\n" +
                                    "• Training chunks scheduled: " + totalChunksEnqueued +
                                    duplicateSummary +
                                    failureSummary,
                                thinkReasoning = "Bounded ZIP extraction: " +
                                    processedFiles + " files → " + totalChunksEnqueued + " blocks."
                            )
                        )
                    }
                }
            } catch (error: Exception) {
                Log.e(TAG, "importZipContent failed", error)
                withContext(Dispatchers.Main) {
                    appendMessage(
                        ChatMessage(
                            isUser = false,
                            text = "Failed to extract the ZIP archive: " + error.message,
                            thinkReasoning = "Bounded ZIP extraction exception: " +
                                error.javaClass.simpleName
                        )
                    )
                }
            }
        }
    }

    private fun drainLimitedInput(input: InputStream) {
        val buffer = ByteArray(8_192)
        while (true) {
            val count = input.read(buffer)
            if (count < 0) break
        }
    }

    fun importVideoContent(uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            val retriever = MediaMetadataRetriever()
            try {
                val cr = getApplication<Application>().contentResolver
                cr.openFileDescriptor(uri, "r")?.use { pfd ->
                    retriever.setDataSource(pfd.fileDescriptor)
                } ?: run {
                    retriever.setDataSource(getApplication(), uri)
                }

                val durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                val durationMs = durationStr?.toLongOrNull() ?: 0L
                val widthStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH)
                val heightStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT)
                val mime = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_MIMETYPE) ?: "video/*"

                val w = widthStr?.toIntOrNull() ?: 0
                val h = heightStr?.toIntOrNull() ?: 0
                val durationSec = durationMs / 1000.0

                val timeUs1 = (durationMs * 200).coerceAtLeast(0L)
                val frame1 = runCatching {
                    retriever.getFrameAtTime(timeUs1, MediaMetadataRetriever.OPTION_CLOSEST_SYNC)
                }.getOrNull()

                val (avgLum, dominantTone, contrast) = frame1?.let {
                    analyzeBitmapMetrics(it)
                } ?: Triple(50f, "Neutral", 1.0f)
                frame1?.recycle()

                val videoDisplayName = runCatching {
                    var name: String? = null
                    cr.query(uri, null, null, null, null)?.use { cursor ->
                        val idx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                        if (idx != -1 && cursor.moveToFirst()) name = cursor.getString(idx)
                    }
                    name ?: "Video_${System.currentTimeMillis() % 10000}.mp4"
                }.getOrDefault("Video_${System.currentTimeMillis() % 10000}.mp4")

                val desc = "Video '$videoDisplayName' (${String.format(Locale.US, "%.1f", durationSec)}s, ${w}×${h} px, $mime) was imported for on-device visual perception."

                withContext(Dispatchers.Main) {
                    appendMessage(
                        ChatMessage(
                            isUser = true,
                            text = desc,
                            attachedFileName = videoDisplayName
                        )
                    )
                }

                val perceptualStimulus = "On-device video perception: '$videoDisplayName', " +
                        "duration: ${String.format(Locale.US, "%.1f", durationSec)} seconds, " +
                        "frame resolution: ${w}×${h} px, " +
                        "format: $mime. " +
                        "Keyframe analysis: average luminance ${String.format(Locale.US, "%.1f", avgLum)}%, " +
                        "color palette: $dominantTone, " +
                        "contrast ratio: ${String.format(Locale.US, "%.1f", contrast)}:1."

                trainingManager.enqueueData(
                    source = "Video Perception Import",
                    title = "Video: $videoDisplayName",
                    content = perceptualStimulus
                )

                worldModel.storeEntities(
                    WorldEntity("video_perception", perceptualStimulus.take(200), "Video Sensory Ingestion")
                )

                withContext(Dispatchers.Main) {
                    appendMessage(
                        ChatMessage(
                            isUser = false,
                            text = "Video '$videoDisplayName' was analyzed on-device using memory-bounded sampling.\n\n" +
                                    "• Duration: ${String.format(Locale.US, "%.1f", durationSec)} seconds\n" +
                                    "• Resolution: ${w}×${h} px ($mime)\n" +
                                    "• Keyframe analysis: luminance ${String.format(Locale.US, "%.1f", avgLum)}%, palette $dominantTone, contrast ${String.format(Locale.US, "%.1f", contrast)}:1\n\n" +
                                    "The visual summary was stored in the World Model and added to the local training queue.",
                            thinkReasoning = "Video keyframe & metadata extraction (OOM-safe sample)."
                        )
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "importVideoContent failed", e)
                withContext(Dispatchers.Main) {
                    appendMessage(
                        ChatMessage(
                            isUser = false,
                            text = "Failed to process the video: ${e.message}",
                            thinkReasoning = "MediaMetadataRetriever error."
                        )
                    )
                }
            } finally {
                runCatching { retriever.release() }
            }
        }
    }

    fun updateCameraFrame(luminance: Float) {
        val safeLuma = luminance.takeIf(Float::isFinite)?.coerceIn(0f, 255f) ?: 0f
        _cameraLuminance.value = safeLuma
        worldModel.updateCameraContext(
            lumaLevel = safeLuma,
            visualSummary = "Camera active. Average Y-plane luminance: ${String.format(Locale.US, "%.1f", safeLuma)}/255 (relative, not calibrated lux)."
        )
    }

    private suspend fun executeWebSearchSkill(query: String): List<SearchResultItem> {
        return when (val output = skillRegistry.execute(WebSearchSkill.ID, query)) {
            is WebSearchSkillOutput -> output.results
            else -> throw IllegalStateException("Unexpected output from Web Search skill.")
        }
    }

    private suspend fun executeWebArticleExtractionSkill(sourceUrl: String): ExtractedWebArticle {
        return when (
            val output = skillRegistry.execute(WebArticleExtractionSkill.ID, sourceUrl)
        ) {
            is WebArticleExtractionSkillOutput -> output.article
            else -> throw IllegalStateException("Unexpected output from Web Article Extraction skill.")
        }
    }

    fun extractWebArticle(result: SearchResultItem) {
        if (_isExtractingArticle.value) return

        viewModelScope.launch {
            _isExtractingArticle.value = true
            _articleExtractionError.value = null
            _articleQueueStatus.value = null
            _articlePreview.value = null

            try {
                _articlePreview.value = executeWebArticleExtractionSkill(result.sourceUrl)
            } catch (error: Exception) {
                Log.w(TAG, "Article extraction failed for ${result.sourceUrl}", error)
                _articleExtractionError.value = error.message ?: "The article could not be extracted."
            } finally {
                _isExtractingArticle.value = false
            }
        }
    }

    private fun recommendedTrainingPasses(contentSize: Long): Int {
        if (contentSize <= 0L) return 1
        val profile = _hardwareInfo.value

        return when {
            contentSize <= 250_000L &&
                profile.totalRamGb >= 8 &&
                profile.availRamGb >= 2.0f &&
                profile.maxHeapMb >= 384L -> 3

            contentSize <= 600_000L &&
                profile.totalRamGb >= 6 &&
                profile.availRamGb >= 1.0f &&
                profile.maxHeapMb >= 256L -> 2

            else -> 1
        }
    }

    fun enqueueReviewedWebArticle() {
        val article = _articlePreview.value ?: return
        if (_isQueueingArticle.value) return

        val epochs = recommendedTrainingPasses(article.characterCount.toLong())
        _isQueueingArticle.value = true
        _articleQueueStatus.value = null

        viewModelScope.launch(Dispatchers.Default) {
            val result = runCatching {
                trainingManager.enqueueDocument(
                    source = buildString {
                        append("web_article")
                        append("|site=").append(encodeProvenanceField(article.siteName))
                        append("|domain=").append(article.domain)
                        append("|source_url=").append(encodeProvenanceField(article.sourceUrl))
                        append("|canonical=").append(encodeProvenanceField(article.canonicalUrl))
                        append("|published=").append(encodeProvenanceField(article.publishedAt.orEmpty()))
                        append("|language=").append(encodeProvenanceField(article.language.orEmpty()))
                        append("|characters=").append(article.characterCount)
                        append("|sha256=").append(sha256Hex(article.text))
                        append("|extracted_at=").append(article.extractedAtEpochMs)
                        append("|rights_declared_by_user=true")
                    },
                    title = article.title,
                    fullText = article.text,
                    chunkSize = 700,
                    overlap = 80,
                    epochs = epochs,
                    language = article.language,
                    canonicalUrl = article.canonicalUrl,
                    rightsStatus = "USER_DECLARED"
                )
            }

            withContext(Dispatchers.Main) {
                _isQueueingArticle.value = false
                result.onSuccess { queuedChunks ->
                    if (queuedChunks > 0) {
                        _articlePreview.value = null
                        _articleExtractionError.value = null
                        _articleQueueStatus.value = "Article queued as $queuedChunks training chunks across $epochs passes, based on available device RAM and app heap."
                    } else {
                        _articleQueueStatus.value =
                            "This article is already in the local corpus. No new training copy was created."
                    }
                }.onFailure { error ->
                    Log.e(TAG, "Failed to queue reviewed web article", error)
                    _articleQueueStatus.value = error.message ?: "The article could not be added to the queue."
                }
            }
        }
    }

    fun dismissWebArticlePreview() {
        _articlePreview.value = null
        _articleExtractionError.value = null
        _articleQueueStatus.value = null
    }

    fun searchWebDirectly(query: String) {
        val normalizedQuery = query.trim()
        if (normalizedQuery.isBlank()) return

        viewModelScope.launch {
            _articlePreview.value = null
            _articleExtractionError.value = null
            _articleQueueStatus.value = null

            try {
                _searchResults.value = executeWebSearchSkill(normalizedQuery)
            } catch (error: Exception) {
                Log.e(TAG, "searchWebDirectly failed", error)
                appendMessage(
                    ChatMessage(
                        isUser = false,
                        text = "Web search failed: ${error.message}",
                        thinkReasoning = "Web search exception."
                    )
                )
            }
        }
    }

    fun executeActionClick(targetText: String): Boolean =
        runCatching { ArgusAccessibilityService.performClickAction(targetText) }
            .getOrDefault(false)

    fun toggleAppExclusion(packageName: String) {
        excludedAppsManager.toggleAppExclusion(packageName)
        loadInstalledApps()
    }

    fun clearChat() {
        stopGeneration()
        val resetMsg = ChatMessage(
            isUser = false,
            text = "The conversation was cleared. Local model weights and stored World Model records remain unchanged.",
            thinkReasoning = "The message history was reset. Model weights were not changed."
        )
        _messages.value = listOf(resetMsg)
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                repository.clearChatMessages()
                repository.saveChatMessage(resetMsg)
            }
        }
    }

    fun clearTrainingQueue() {
        trainingManager.clearQueue()
    }

    fun clearCompletedTraining() {
        trainingManager.clearCompleted()
    }

    private fun loadInstalledApps() {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                _installedApps.value = excludedAppsManager.getInstalledApps()
            } catch (e: Exception) {
                Log.e(TAG, "loadInstalledApps failed", e)
            }
        }
    }

    private fun tryEvaluateMath(text: String): String? {
        val cleaned = text.trim().lowercase()
            .replace("×", "*")
            .replace("÷", "/")
            .replace(Regex("^(hitung|berapa|calculate|compute|nilai dari)\\s+"), "")
            .replace(Regex("\\s*(sama dengan|=)\\s*\\??$"), "")
            .trim()

        if (cleaned.none { it.isDigit() }) return null
        if (cleaned.none { it in "+-*/^" }) return null
        if (!cleaned.matches(Regex("^[0-9.()+\\-*/^\\s]+$"))) return null

        return try {
            val parser = ExpressionParser(cleaned)
            val result = parser.parse()
            if (!result.isFinite()) null
            else "Arithmetic evaluation result: ${parser.normalized} = " +
                    formatNumber(result)
        } catch (_: Exception) {
            null
        }
    }

    private fun formatNumber(v: Double): String =
        if (v == v.toLong().toDouble()) v.toLong().toString()
        else String.format(java.util.Locale.US, "%.6f", v).trimEnd('0').trimEnd('.')

    private class ExpressionParser(private val expr: String) {
        private var pos = 0
        val normalized: String get() = expr.replace(Regex("\\s+"), "")

        fun parse(): Double {
            val v = parseExpr()
            skipWs()
            require(pos >= expr.length) { "Unexpected char at $pos" }
            return v
        }

        private fun parseExpr(): Double {
            var left = parseTerm()
            while (true) {
                skipWs()
                if (pos >= expr.length) break
                val op = expr[pos]
                if (op != '+' && op != '-') break
                pos++
                val right = parseTerm()
                left = if (op == '+') left + right else left - right
            }
            return left
        }

        private fun parseTerm(): Double {
            var left = parsePower()
            while (true) {
                skipWs()
                if (pos >= expr.length) break
                val op = expr[pos]
                if (op != '*' && op != '/') break
                pos++
                val right = parsePower()
                left = when (op) {
                    '*' -> left * right
                    '/' -> {
                        require(right != 0.0) { "Division by zero" }
                        left / right
                    }
                    else -> error("unreachable")
                }
            }
            return left
        }

        private fun parsePower(): Double {
            val base = parseFactor()
            skipWs()
            if (pos < expr.length && expr[pos] == '^') {
                pos++
                val exponent = parsePower()
                return Math.pow(base, exponent)
            }
            return base
        }

        private fun parseFactor(): Double {
            skipWs()
            require(pos < expr.length) { "Unexpected end" }
            val c = expr[pos]
            if (c == '(') {
                pos++
                val v = parseExpr()
                skipWs()
                require(pos < expr.length && expr[pos] == ')') { "Expected )" }
                pos++
                return v
            }
            if (c == '-') { pos++; return -parseFactor() }
            if (c == '+') { pos++; return parseFactor() }

            val start = pos
            var dotSeen = false
            while (pos < expr.length) {
                val ch = expr[pos]
                if (ch.isDigit()) pos++
                else if (ch == '.' && !dotSeen) { dotSeen = true; pos++ }
                else break
            }
            require(pos > start) { "Expected number at $start" }
            return expr.substring(start, pos).toDouble()
        }

        private fun skipWs() {
            while (pos < expr.length && expr[pos].isWhitespace()) pos++
        }
    }

    override fun onCleared() {
        super.onCleared()
        audioVoiceManager.close()
    }

    private class BoundedReader(
        private val delegate: java.io.Reader,
        private val maximumCharacters: Long
    ) : java.io.Reader() {
        private var charactersRead = 0L

        override fun read(buffer: CharArray, offset: Int, length: Int): Int {
            require(offset >= 0 && length >= 0 && offset + length <= buffer.size)
            if (length == 0) return 0

            val remaining = maximumCharacters - charactersRead
            if (remaining <= 0L) {
                if (delegate.read() >= 0) {
                    throw IOException(
                        "Text document exceeds the limit of " + maximumCharacters + " characters."
                    )
                }
                return -1
            }

            val allowedLength = minOf(length.toLong(), remaining).toInt()
            val count = delegate.read(buffer, offset, allowedLength)
            if (count > 0) charactersRead += count.toLong()
            return count
        }

        override fun close() {
            delegate.close()
        }
    }

    private class LimitedNonClosingInputStream(
        inputStream: InputStream,
        private val maximumBytes: Long
    ) : FilterInputStream(inputStream) {
        var bytesRead: Long = 0L
            private set

        var limitExceeded: Boolean = false
            private set

        override fun read(): Int {
            val value = super.read()
            if (value >= 0) countBytes(1L)
            return value
        }

        override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
            val count = super.read(buffer, offset, length)
            if (count > 0) countBytes(count.toLong())
            return count
        }

        private fun countBytes(count: Long) {
            bytesRead += count
            if (bytesRead > maximumBytes) {
                limitExceeded = true
                throw IOException("ZIP entry expanded beyond the configured limit.")
            }
        }

        override fun close() {
            // Keep the enclosing ZipInputStream open until closeEntry().
        }
    }
}