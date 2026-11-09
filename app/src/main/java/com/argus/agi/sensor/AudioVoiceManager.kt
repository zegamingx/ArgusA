package com.argus.agi.sensor

import android.content.Context
import android.content.Intent
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import java.io.Closeable
import java.util.Locale
import kotlin.math.log10
import kotlin.math.sqrt

/**
 * Mengelola interaksi suara offline dua arah:
 * 1. Text-to-Speech (TTS) on-device: Argus berbicara langsung kepada pengguna tanpa koneksi internet.
 * 2. Speech-to-Text (STT) on-device: Mendengarkan suara pengguna dan mentranskripsikannya ke teks.
 * 3. Acoustic Environment Sensor: Mengukur tingkat kebisingan/desibel suara sekitar secara real-time.
 */
class AudioVoiceManager(
    private val context: Context,
    private val scope: CoroutineScope
) : Closeable, TextToSpeech.OnInitListener {

    companion object {
        private const val TAG = "AudioVoiceManager"
        private const val UTTERANCE_ID = "ARGUS_VOICE_OUT"
    }

    // =========================================================================
    // 1. Text-To-Speech (Offline TTS)
    // =========================================================================
    private var tts: TextToSpeech? = null
    private val _isTtsReady = MutableStateFlow(false)
    val isTtsReady: StateFlow<Boolean> = _isTtsReady.asStateFlow()

    private val _isSpeaking = MutableStateFlow(false)
    val isSpeaking: StateFlow<Boolean> = _isSpeaking.asStateFlow()

    private val _isAutoSpeakEnabled = MutableStateFlow(false)
    val isAutoSpeakEnabled: StateFlow<Boolean> = _isAutoSpeakEnabled.asStateFlow()

    // =========================================================================
    // 2. Speech-To-Text (Offline STT)
    // =========================================================================
    private var speechRecognizer: SpeechRecognizer? = null
    private val _isListening = MutableStateFlow(false)
    val isListening: StateFlow<Boolean> = _isListening.asStateFlow()

    private val _partialTranscription = MutableStateFlow("")
    val partialTranscription: StateFlow<String> = _partialTranscription.asStateFlow()

    private val _speechRmsDb = MutableStateFlow(0f)
    val speechRmsDb: StateFlow<Float> = _speechRmsDb.asStateFlow()

    // =========================================================================
    // 3. Acoustic Environment Monitor (Ambient Decibel via AudioRecord)
    // =========================================================================
    private var acousticJob: Job? = null
    private val _ambientDecibel = MutableStateFlow(0f)
    val ambientDecibel: StateFlow<Float> = _ambientDecibel.asStateFlow()

    private val _isAcousticMonitoring = MutableStateFlow(false)
    val isAcousticMonitoring: StateFlow<Boolean> = _isAcousticMonitoring.asStateFlow()

    init {
        initTts()
    }

    private fun initTts() {
        try {
            tts = TextToSpeech(context.applicationContext, this)
        } catch (e: Exception) {
            Log.e(TAG, "Gagal menginisialisasi TextToSpeech", e)
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) {
            val t = tts ?: return
            val idLocale = Locale("id", "ID")
            val result = t.setLanguage(idLocale)
            if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                // Fallback ke default perangkat
                t.setLanguage(Locale.getDefault())
            }
            t.setPitch(1.0f)
            t.setSpeechRate(1.05f)

            t.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
                override fun onStart(utteranceId: String?) {
                    _isSpeaking.value = true
                }
                override fun onDone(utteranceId: String?) {
                    _isSpeaking.value = false
                }
                override fun onError(utteranceId: String?) {
                    _isSpeaking.value = false
                }
            })
            _isTtsReady.value = true
            Log.i(TAG, "TTS on-device siap digunakan.")
        } else {
            Log.w(TAG, "Inisialisasi TTS gagal dengan kode status: $status")
            _isTtsReady.value = false
        }
    }

    fun toggleAutoSpeak() {
        _isAutoSpeakEnabled.value = !_isAutoSpeakEnabled.value
    }

    fun setAutoSpeak(enabled: Boolean) {
        _isAutoSpeakEnabled.value = enabled
    }

    /**
     * Mendeteksi bahasa teks secara otomatis (Indonesian vs English vs default)
     * agar mesin TTS berbicara dengan pelafalan dan fonem yang sesuai dengan bahasa kalimat.
     */
    fun detectLanguageLocale(text: String): Locale {
        val words = text.lowercase(Locale.ROOT).split(Regex("[\\s.,!?:;\"'()\\[\\]{}]+")).filter { it.length >= 2 }
        if (words.isEmpty()) return Locale("id", "ID")

        val idMarkers = setOf(
            "yang", "dan", "di", "ini", "itu", "ke", "dari", "untuk", "pada", "adalah", "dengan",
            "saya", "kamu", "bisa", "ada", "tidak", "apa", "bagaimana", "kata", "dasar", "sudah",
            "akan", "mereka", "kita", "kami", "sangat", "lebih", "telah", "oleh", "hal", "saat",
            "karena", "bahwa", "sebagai", "dalam", "bagi", "juga", "hanya", "namun", "tetapi"
        )
        val enMarkers = setOf(
            "the", "and", "is", "in", "to", "of", "for", "with", "that", "this", "you",
            "are", "have", "from", "not", "what", "how", "word", "root", "been", "will",
            "they", "we", "very", "more", "has", "by", "model", "data", "training", "it",
            "can", "about", "which", "their", "there", "when", "also", "your", "an", "as"
        )

        var idCount = 0
        var enCount = 0
        for (w in words) {
            if (w in idMarkers) idCount++
            if (w in enMarkers) enCount++
        }

        return when {
            enCount > idCount -> Locale.US
            idCount > enCount -> Locale("id", "ID")
            else -> Locale("id", "ID")
        }
    }

    /**
     * Berbicara secara offline menggunakan sintetis vokal lokal perangkat.
     * Secara otomatis mendeteksi bahasa teks yang dilatih/dihasilkan dan mengadaptasikan fonem suara TTS.
     * Membersihkan tag pemikiran internal (<THINK>...</THINK>) agar ucapan natural.
     */
    fun speak(rawText: String) {
        val t = tts ?: return
        if (!_isTtsReady.value) return

        // Bersihkan reasoning internal dan markdown
        val clean = rawText
            .replace(Regex("(?s)<THINK>.*?</THINK>"), "")
            .replace(Regex("(?s)<think>.*?</think>"), "")
            .replace(Regex("[*#`_~]"), "")
            .trim()

        if (clean.isBlank()) return

        // Adaptasikan bahasa suara TTS sesuai bahasa kalimat yang dilatih/dihasilkan model
        val targetLocale = detectLanguageLocale(clean)
        val res = t.setLanguage(targetLocale)
        if (res == TextToSpeech.LANG_MISSING_DATA || res == TextToSpeech.LANG_NOT_SUPPORTED) {
            t.setLanguage(Locale.getDefault())
        }

        val params = Bundle().apply {
            putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, UTTERANCE_ID)
        }
        t.speak(clean, TextToSpeech.QUEUE_FLUSH, params, UTTERANCE_ID)
    }

    fun stopSpeaking() {
        tts?.stop()
        _isSpeaking.value = false
    }

    // =========================================================================
    // =========================================================================
    // STT Control
    // =========================================================================

    fun startListening(
        onFinalResult: (String) -> Unit,
        onError: (String) -> Unit
    ) {
        startListeningInternal(preferOffline = true, isFallback = false, onFinalResult = onFinalResult, onError = onError)
    }

    private fun startListeningInternal(
        preferOffline: Boolean,
        isFallback: Boolean,
        onFinalResult: (String) -> Unit,
        onError: (String) -> Unit
    ) {
        scope.launch(Dispatchers.Main) {
            try {
                if (!SpeechRecognizer.isRecognitionAvailable(context)) {
                    onError("Pengenal suara on-device tidak tersedia di perangkat ini.")
                    return@launch
                }

                // Bersihkan sesi pengenal sebelumnya secara tuntas demi mencegah Kode 5 (ERROR_CLIENT)
                try {
                    speechRecognizer?.cancel()
                    speechRecognizer?.destroy()
                } catch (e: Exception) {
                    Log.w(TAG, "Error cleaning previous SpeechRecognizer", e)
                }
                speechRecognizer = null

                val recognizer = SpeechRecognizer.createSpeechRecognizer(context.applicationContext)
                speechRecognizer = recognizer

                recognizer.setRecognitionListener(object : RecognitionListener {
                    override fun onReadyForSpeech(params: Bundle?) {
                        _isListening.value = true
                        _partialTranscription.value = ""
                    }
                    override fun onBeginningOfSpeech() {
                        _isListening.value = true
                    }
                    override fun onRmsChanged(rmsdB: Float) {
                        _speechRmsDb.value = rmsdB.coerceAtLeast(0f)
                    }
                    override fun onBufferReceived(buffer: ByteArray?) {}
                    override fun onEndOfSpeech() {
                        _isListening.value = false
                    }
                    override fun onError(error: Int) {
                        _isListening.value = false
                        Log.w(TAG, "SpeechRecognizer onError: $error (preferOffline=$preferOffline, isFallback=$isFallback)")

                        // Kode 11 (ERROR_SERVER_DISCONNECTED / model offline belum terpasang) atau Kode 4 / 14:
                        // Otomatis fallback ke pengenal standar perangkat tanpa paksaan offline model
                        if ((error == 11 || error == 4 || error == 14) && preferOffline && !isFallback) {
                            Log.i(TAG, "Mencoba fallback speech recognition tanpa paksaan offline model...")
                            startListeningInternal(preferOffline = false, isFallback = true, onFinalResult = onFinalResult, onError = onError)
                            return
                        }

                        val msg = when (error) {
                            SpeechRecognizer.ERROR_AUDIO -> "Gangguan perekaman mikrofon audio. Pastikan tidak ada aplikasi lain memakai mic."
                            SpeechRecognizer.ERROR_CLIENT -> "Sesi mikrofon sebelumnya sedang diatur ulang. Silakan tekan tombol mic sekali lagi."
                            SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS -> "Izin mikrofon diperlukan untuk perintah suara."
                            SpeechRecognizer.ERROR_NETWORK -> "Koneksi jaringan suara terputus. Pastikan model bahasa offline telah terpasang di pengaturan perangkat."
                            SpeechRecognizer.ERROR_NETWORK_TIMEOUT -> "Batas waktu koneksi pengenal suara habis."
                            SpeechRecognizer.ERROR_NO_MATCH -> "Tidak ada suara atau kata yang terdengar jelas. Silakan bicara lebih dekat."
                            SpeechRecognizer.ERROR_RECOGNIZER_BUSY -> "Layanan suara sedang sibuk. Mengatur ulang..."
                            SpeechRecognizer.ERROR_SERVER -> "Layanan pengenal suara perangkat belum siap."
                            SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> "Waktu bicara selesai (tidak ada suara terdeteksi)."
                            10 -> "Terlalu banyak permintaan suara dalam waktu singkat."
                            11 -> "Model bahasa suara offline belum terunduh di perangkat. Silakan instal paket bahasa Indonesia di Pengaturan -> Google Suara."
                            12 -> "Bahasa Indonesia belum didukung oleh mesin suara perangkat."
                            13 -> "Paket bahasa suara sedang tidak tersedia."
                            14 -> "Gagal memeriksa dukungan suara on-device."
                            else -> "Pengenalan suara gagal (kode $error). Silakan coba lagi."
                        }
                        onError(msg)
                    }
                    override fun onResults(results: Bundle?) {
                        _isListening.value = false
                        val matches = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        val recognized = matches?.firstOrNull()?.trim() ?: ""
                        _partialTranscription.value = recognized
                        if (recognized.isNotBlank()) {
                            onFinalResult(recognized)
                        }
                    }
                    override fun onPartialResults(partialResults: Bundle?) {
                        val matches = partialResults?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                        val text = matches?.firstOrNull()?.trim() ?: ""
                        if (text.isNotBlank()) {
                            _partialTranscription.value = text
                        }
                    }
                    override fun onEvent(eventType: Int, params: Bundle?) {}
                })

                val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE, "id-ID")
                    putExtra(RecognizerIntent.EXTRA_LANGUAGE_PREFERENCE, "id-ID")
                    putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
                    if (preferOffline) {
                        putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
                    }
                }

                recognizer.startListening(intent)
            } catch (e: Exception) {
                Log.e(TAG, "Gagal memulai speech listening", e)
                _isListening.value = false
                onError("Gagal mengaktifkan mikrofon: ${e.message}")
            }
        }
    }

    fun stopListening() {
        scope.launch(Dispatchers.Main) {
            try {
                speechRecognizer?.stopListening()
                speechRecognizer?.cancel()
            } catch (e: Exception) {
                Log.w(TAG, "stopListening exception", e)
            } finally {
                _isListening.value = false
            }
        }
    }

    // =========================================================================
    // Acoustic Sensor (Mengukur desibel kebisingan suara lingkungan nyata)
    // =========================================================================

    fun startAcousticMonitoring(onDecibelSample: (Float) -> Unit) {
        if (_isAcousticMonitoring.value) return
        _isAcousticMonitoring.value = true

        acousticJob = scope.launch(Dispatchers.IO) {
            val sampleRate = 8000
            val channelConfig = AudioFormat.CHANNEL_IN_MONO
            val audioFormat = AudioFormat.ENCODING_PCM_16BIT
            val minBufSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat)

            if (minBufSize <= 0) {
                _isAcousticMonitoring.value = false
                return@launch
            }

            var recorder: AudioRecord? = null
            try {
                recorder = AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    sampleRate,
                    channelConfig,
                    audioFormat,
                    minBufSize * 2
                )

                if (recorder.state != AudioRecord.STATE_INITIALIZED) {
                    _isAcousticMonitoring.value = false
                    return@launch
                }

                recorder.startRecording()
                val buffer = ShortArray(minBufSize)

                while (isActive && _isAcousticMonitoring.value) {
                    val read = recorder.read(buffer, 0, buffer.size)
                    if (read > 0) {
                        var sumSq = 0.0
                        for (i in 0 until read) {
                            sumSq += buffer[i] * buffer[i]
                        }
                        val rms = sqrt(sumSq / read)
                        val db = if (rms > 1.0) (20.0 * log10(rms)).toFloat().coerceIn(0f, 120f) else 0f
                        _ambientDecibel.value = db
                        onDecibelSample(db)
                    }
                    kotlinx.coroutines.delay(500)
                }
            } catch (e: SecurityException) {
                Log.w(TAG, "Izin AudioRecord tidak tersedia untuk monitoring akustik", e)
            } catch (e: Exception) {
                Log.e(TAG, "Acoustic monitoring error", e)
            } finally {
                runCatching {
                    recorder?.stop()
                    recorder?.release()
                }
                _isAcousticMonitoring.value = false
            }
        }
    }

    fun stopAcousticMonitoring() {
        _isAcousticMonitoring.value = false
        acousticJob?.cancel()
        acousticJob = null
    }

    override fun close() {
        stopSpeaking()
        stopListening()
        stopAcousticMonitoring()
        try {
            tts?.shutdown()
            tts = null
        } catch (e: Exception) {
            Log.w(TAG, "Error shutting down TTS", e)
        }
        try {
            speechRecognizer?.destroy()
            speechRecognizer = null
        } catch (e: Exception) {
            Log.w(TAG, "Error destroying speech recognizer", e)
        }
    }
}
