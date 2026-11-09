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
import com.argus.agi.model.WorldEntity
import android.media.MediaMetadataRetriever
import com.argus.agi.sensor.AudioVoiceManager
import com.argus.agi.sensor.DeviceSafetyState
import com.argus.agi.sensor.ExcludedAppsManager
import com.argus.agi.sensor.InstalledAppItem
import com.argus.agi.sensor.SearchResultItem
import com.argus.agi.sensor.SystemHardwareInfo
import com.argus.agi.sensor.WebSearchEngine
import com.argus.agi.service.ArgusAccessibilityService
import com.argus.agi.service.ArgusForegroundService
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
        private const val MAX_FILE_SIZE_BYTES = 100L * 1024L * 1024L // Kuota streaming OOM-safe hingga 100 MB
        private const val MAX_IMAGE_SIDE = 2048
    }

    private val neuralEngine = ArgusApp.neuralEngine
    private val worldModel = ArgusApp.worldModel
    private val trainingManager = ArgusApp.trainingManager
    private val deviceGuard = ArgusApp.deviceGuard
    private val hardwareProfile = ArgusApp.hardwareProfile
    private val genesisModelManager = ArgusApp.genesisModelManager
    private val excludedAppsManager = ExcludedAppsManager(application)
    private val webSearchEngine = WebSearchEngine()

    private val repository = ArgusApp.repository

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

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
                worldModel.storeEntities(
                    WorldEntity(
                        "ambient_acoustic",
                        String.format(Locale.US, "%.1f dB", db),
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
                            text = "A.R.G.U.S Genesis Core aktif (${identity.originId}). " +
                                    "100% on-device neural intelligence. Parameter diinisialisasi " +
                                    "dan dievolusikan langsung pada perangkat ini.",
                            thinkReasoning = "Genesis Origin: ${identity.originId} | " +
                                    "Dibuat: ${identity.formattedBirthDate} | " +
                                    "Arsitektur: ${identity.architectureTier}. " +
                                    "Parameter total: ${identity.parameterCount} (FP32)."
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
                        text = "A.R.G.U.S Genesis Core aktif (${existing.originId}). " +
                                "100% on-device neural intelligence. Parameter diinisialisasi " +
                                "dan dievolusikan langsung pada perangkat ini.",
                        thinkReasoning = "Genesis Origin: ${existing.originId} | " +
                                "Dibuat: ${existing.formattedBirthDate} | " +
                                "Arsitektur: ${existing.architectureTier}. " +
                                "Parameter total: ${existing.parameterCount} (FP32)."
                    )
                    withContext(Dispatchers.Main) {
                        appendMessage(initialMsg)
                    }
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

    fun refreshHardwareAssessment() {
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val info = hardwareProfile.getHardwareInfo()
                _hardwareInfo.value = info
                appendMessage(
                    ChatMessage(
                        isUser = false,
                        text = "Assessment ulang selesai. SoC ${info.socManufacturer} ${info.socModel}, " +
                                "RAM ${info.totalRamGb} GB, akselerator ${info.primaryAccelerator}.",
                        thinkReasoning = "Hardware re-fetch. Tidak mengubah origin identity."
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
                Log.e(TAG, "retryGenesis gagal", e)
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
                        text = "Benchmark gagal: ${e.message}",
                        thinkReasoning = "Exception saat runBenchmark"
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
                    text = "Genesis sedang berjalan (${stage.displayName}). " +
                            "Tunggu hingga selesai sebelum mulai chat."
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
                    text = "Tunggu respon sebelumnya selesai, atau tekan stop."
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

                val mathResult = tryEvaluateMath(text)

                val rawOutput = try {
                    if (mathResult != null) mathResult
                    else neuralEngine.generate(multiTurnPrompt, maxNewTokens = 60)
                } catch (e: Exception) {
                    Log.e(TAG, "Generation failed", e)
                    "Model lokal sedang sibuk atau mengalami error (${e.javaClass.simpleName}). " +
                            "Coba lagi sebentar."
                }

                val reflection = worldModel.reflectOnResponse(text, rawOutput)

                var finalAnswer = rawOutput
                val thinking = StringBuilder()
                thinking.append("Self-Reflection: ").append(reflection.reasoningNote)
                thinking.append(" | Keyakinan: ")
                    .append(String.format("%.0f%%", reflection.confidenceScore * 100))

                if (!reflection.isConfident || reflection.suggestedAction == "AUTO_LEARN_QUERY") {
                    thinking.append("\n→ Autonomous On-Device Web Retrieval & Training Queue.")

                    val webItems = runCatching { webSearchEngine.search(text) }
                        .onFailure { Log.w(TAG, "Web search failed", it) }
                        .getOrDefault(emptyList())

                    if (webItems.isNotEmpty()) {
                        val learned = webItems.joinToString("\n") {
                            "${it.title}: ${it.snippet}"
                        }
                        trainingManager.enqueueData(
                            source = "Chat Inquiry Learner",
                            title = "Pertanyaan: ${text.take(30)}",
                            content = "Tanya: $text. Pengetahuan: $learned"
                        )
                        finalAnswer = "Saya sedang mempelajari topik ini langsung dari web on-device:\n\n" +
                                webItems.take(2).joinToString("\n\n") {
                                    "• **${it.title}**\n${it.snippet}"
                                } +
                                "\n\n*Data ini telah dimasukkan ke antrean training lokal.*"
                    } else {
                        trainingManager.enqueueData(
                            source = "Chat Unresolved",
                            title = "Inquiry: ${text.take(30)}",
                            content = text
                        )
                        finalAnswer = "Pola ini baru bagi model. Pertanyaan ini telah langsung " +
                                "dijadwalkan ke antrean training on-device."
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

                if (sizeBytes != null && sizeBytes > MAX_FILE_SIZE_BYTES) {
                    withContext(Dispatchers.Main) {
                        appendMessage(
                            ChatMessage(
                                isUser = false,
                                text = "File terlalu besar (${sizeBytes / 1024 / 1024} MB). " +
                                        "Batas maksimal ${MAX_FILE_SIZE_BYTES / 1024 / 1024} MB " +
                                        "untuk menjaga memori perangkat.",
                                thinkReasoning = "File size guard triggered."
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
                    name ?: "Dokumen_${System.currentTimeMillis() % 10000}.txt"
                }.getOrDefault("Dokumen_${System.currentTimeMillis() % 10000}.txt")

                val isZip = docDisplayName.endsWith(".zip", ignoreCase = true) ||
                        cr.getType(uri)?.contains("zip") == true

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
                    val reader = BufferedReader(InputStreamReader(stream, Charsets.UTF_8), 16384)
                    trainingManager.enqueueDocumentStream(
                        source = "Dokumen File",
                        title = docDisplayName,
                        reader = reader
                    )
                } ?: Pair(0, 0L)

                if (chunkCount == 0 || charsRead == 0L) {
                    withContext(Dispatchers.Main) {
                        appendMessage(
                            ChatMessage(
                                isUser = false,
                                text = "File '$docDisplayName' kosong atau tidak bisa dibaca.",
                                thinkReasoning = "Empty content or stream read returned 0 chunks."
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
                    "$charsRead karakter"
                }

                withContext(Dispatchers.Main) {
                    appendMessage(
                        ChatMessage(
                            isUser = true,
                            text = "Mengunggah dokumen '$docDisplayName' ($sizeDisplay, $charsRead karakter) untuk pelatihan on-device.",
                            attachedFileName = docDisplayName
                        )
                    )
                    appendMessage(
                        ChatMessage(
                            isUser = false,
                            text = "Dokumen '$docDisplayName' ($sizeDisplay) berhasil diimpor secara streaming (kebal OOM).\n\n" +
                                    "Sistem mempartisi teks menjadi $chunkCount blok latihan berurutan secara dinamis. " +
                                    "Backpropagation melatih bobot model lokal bertahap tanpa mengirim data ke server.\n\n" +
                                    "Pantau progres latihan & live ETA di tab Neural Core.",
                            thinkReasoning = "Streaming chunk ingestion: $charsRead char → $chunkCount blok (Heap OOM-safe)."
                        )
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "importFileContent failed", e)
                withContext(Dispatchers.Main) {
                    appendMessage(
                        ChatMessage(
                            isUser = false,
                            text = "Gagal membaca file: ${e.message}",
                            thinkReasoning = "I/O exception: ${e.javaClass.simpleName}"
                        )
                    )
                }
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
                                    text = "Gambar tidak valid atau tidak bisa dibaca.",
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
                    } ?: Triple(50f, "Netral", 1.0f)

                    val desc = "Gambar ${finalW}×${finalH} px diimpor untuk persepsi visual on-device. " +
                            "Kecerahan: ${String.format(Locale.US, "%.1f", avgLum)}%, Palet: $dominantTone, Kontras: ${String.format(Locale.US, "%.1f", contrast)}:1."

                    withContext(Dispatchers.Main) {
                        sendMessage(desc, uri, isImage = true)
                    }

                    trainingManager.enqueueData(
                        source = "Image Vision Import",
                        title = "Visual $finalW×$finalH",
                        content = "Persepsi Visual On-Device: Gambar resolusi ${finalW}×${finalH} px. " +
                                "Tingkat pencahayaan rata-rata: ${String.format(Locale.US, "%.1f", avgLum)}%, " +
                                "Karakteristik spektrum warna: $dominantTone, " +
                                "Rasio kontras: ${String.format(Locale.US, "%.1f", contrast)}:1."
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "importImageContent failed", e)
                withContext(Dispatchers.Main) {
                    appendMessage(
                        ChatMessage(
                            isUser = false,
                            text = "Gagal memuat gambar: ${e.message}",
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
            rSum > gSum * 1.25f && rSum > bSum * 1.25f -> "Dominan Merah/Hangat"
            bSum > rSum * 1.25f && bSum > gSum * 1.25f -> "Dominan Biru/Sejuk"
            gSum > rSum * 1.25f && gSum > bSum * 1.25f -> "Dominan Hijau/Alami"
            kotlin.math.abs(rSum - gSum) < rSum * 0.15f && kotlin.math.abs(rSum - bSum) < rSum * 0.15f -> "Monokrom/Seimbang"
            else -> "Spektrum Campuran"
        }
        return Triple(avgLum, dominant, contrast)
    }

    fun importZipContent(uri: Uri, displayName: String) {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                val cr = getApplication<Application>().contentResolver
                var totalFiles = 0
                var totalChunksEnqueued = 0
                var totalCharsExtracted = 0L
                val textExtensions = listOf(".txt", ".md", ".json", ".jsonl", ".csv", ".tsv", ".log", ".xml", ".html", ".kt", ".java", ".py")

                cr.openInputStream(uri)?.use { rawStream ->
                    ZipInputStream(BufferedInputStream(rawStream)).use { zis ->
                        var entry = zis.nextEntry
                        while (entry != null) {
                            val name = entry.name
                            val isDirectory = entry.isDirectory

                            // Proteksi Zip Slip (Path Traversal)
                            if (name.contains("..") || name.startsWith("/")) {
                                Log.w(TAG, "Zip entry mencurigakan diabaikan: $name")
                                zis.closeEntry()
                                entry = zis.nextEntry
                                continue
                            }

                            val isText = textExtensions.any { name.endsWith(it, ignoreCase = true) }

                            if (!isDirectory && isText) {
                                totalFiles++
                                val nonClosing = NonClosingInputStream(zis)
                                val entryReader = BufferedReader(InputStreamReader(nonClosing, Charsets.UTF_8), 8192)
                                val (chunks, chars) = trainingManager.enqueueDocumentStream(
                                    source = "ZIP: $displayName",
                                    title = name.substringAfterLast('/'),
                                    reader = entryReader
                                )
                                totalChunksEnqueued += chunks
                                totalCharsExtracted += chars
                            }
                            zis.closeEntry()
                            entry = zis.nextEntry
                        }
                    }
                }

                withContext(Dispatchers.Main) {
                    if (totalFiles == 0) {
                        appendMessage(
                            ChatMessage(
                                isUser = false,
                                text = "Arsip ZIP '$displayName' tidak berisi file teks yang didukung (.txt, .md, .json, .csv, dll).",
                                thinkReasoning = "No valid text entries in zip archive."
                            )
                        )
                    } else {
                        val charsDisplay = String.format(Locale.US, "%,d", totalCharsExtracted)
                        appendMessage(
                            ChatMessage(
                                isUser = true,
                                text = "Mengunggah arsip dataset ZIP '$displayName' ($totalFiles file teks, $charsDisplay karakter).",
                                attachedFileName = displayName
                            )
                        )
                        appendMessage(
                            ChatMessage(
                                isUser = false,
                                text = "Arsip ZIP '$displayName' berhasil diekstraksi secara streaming tanpa memicu OOM!\n\n" +
                                        "• File teks diproses: $totalFiles file\n" +
                                        "• Total karakter: $charsDisplay karakter\n" +
                                        "• Blok training dihasilkan: $totalChunksEnqueued blok\n\n" +
                                        "Seluruh blok teks telah dijadwalkan ke antrean training on-device. Pantau progres di tab Neural Core.",
                                thinkReasoning = "ZIP streaming extraction: $totalFiles files → $totalChunksEnqueued chunks."
                            )
                        )
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "importZipContent failed", e)
                withContext(Dispatchers.Main) {
                    appendMessage(
                        ChatMessage(
                            isUser = false,
                            text = "Gagal mengekstraksi file ZIP: ${e.message}",
                            thinkReasoning = "ZipInputStream extraction exception."
                        )
                    )
                }
            }
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
                } ?: Triple(50f, "Netral", 1.0f)
                frame1?.recycle()

                val videoDisplayName = runCatching {
                    var name: String? = null
                    cr.query(uri, null, null, null, null)?.use { cursor ->
                        val idx = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                        if (idx != -1 && cursor.moveToFirst()) name = cursor.getString(idx)
                    }
                    name ?: "Video_${System.currentTimeMillis() % 10000}.mp4"
                }.getOrDefault("Video_${System.currentTimeMillis() % 10000}.mp4")

                val desc = "Video '$videoDisplayName' (${String.format(Locale.US, "%.1f", durationSec)}s, ${w}×${h} px, $mime) diimpor untuk persepsi visual on-device."

                withContext(Dispatchers.Main) {
                    appendMessage(
                        ChatMessage(
                            isUser = true,
                            text = desc,
                            attachedFileName = videoDisplayName
                        )
                    )
                }

                val perceptualStimulus = "Persepsi Temporal Video On-Device: File '$videoDisplayName', " +
                        "Durasi: ${String.format(Locale.US, "%.1f", durationSec)} detik, " +
                        "Resolusi frame: ${w}×${h} px, " +
                        "Format: $mime. " +
                        "Analisis visual keyframe: Kecerahan rata-rata ${String.format(Locale.US, "%.1f", avgLum)}%, " +
                        "Karakteristik warna: $dominantTone, " +
                        "Rasio kontras: ${String.format(Locale.US, "%.1f", contrast)}:1."

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
                            text = "Video '$videoDisplayName' berhasil dianalisis on-device secara hemat RAM.\n\n" +
                                    "• Durasi: ${String.format(Locale.US, "%.1f", durationSec)} detik\n" +
                                    "• Resolusi: ${w}×${h} px ($mime)\n" +
                                    "• Analisis Keyframe: Luminansi ${String.format(Locale.US, "%.1f", avgLum)}%, Palet $dominantTone, Kontras ${String.format(Locale.US, "%.1f", contrast)}:1\n\n" +
                                    "Stimulus visual temporal telah dimasukkan ke World Model dan antrean training lokal.",
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
                            text = "Gagal memproses video: ${e.message}",
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
        _cameraLuminance.value = luminance
        worldModel.updateCameraContext(
            lightLevel = luminance,
            visualSummary = "Kamera aktif. Luminansi: ${String.format("%.1f", luminance)} lx"
        )
    }

    fun searchWebDirectly(query: String) {
        viewModelScope.launch {
            try {
                val results = webSearchEngine.search(query)
                _searchResults.value = results
                if (results.isNotEmpty()) {
                    trainingManager.enqueueData(
                        source = "Web Search Tool",
                        title = "Web: $query",
                        content = results.joinToString(" ") { it.snippet }
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "searchWebDirectly failed", e)
                appendMessage(
                    ChatMessage(
                        isUser = false,
                        text = "Pencarian web gagal: ${e.message}",
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
            text = "Terminal percakapan dibersihkan. Memori episodik dan bobot model tetap aman.",
            thinkReasoning = "Histori pesan di-reset. Bobot model tidak terpengaruh."
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
            else "Hasil penalaran logika matematika: ${parser.normalized} = " +
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

    private class NonClosingInputStream(inputStream: InputStream) : FilterInputStream(inputStream) {
        override fun close() {
            // Keep underlying ZipInputStream open across entries
        }
    }
}