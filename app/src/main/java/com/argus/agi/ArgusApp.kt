package com.argus.agi

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import android.util.Log
import androidx.core.content.ContextCompat
import com.argus.agi.model.ArgusNeuralEngine
import com.argus.agi.model.EngineConfig
import com.argus.agi.model.GenesisModelManager
import com.argus.agi.model.TrainingManager
import com.argus.agi.model.WorldModel
import com.argus.agi.sensor.DeviceGuard
import com.argus.agi.sensor.HardwareProfile
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch

class ArgusApp : Application() {

    companion object {
        private const val TAG = "ArgusApp"
        const val CHANNEL_ID = "argus_core_service_channel"

        @Volatile
        private var _instance: ArgusApp? = null

        /** Akses aman: throw jelas jika dipakai sebelum onCreate. */
        val instance: ArgusApp
            get() = _instance ?: error(
                "ArgusApp.instance diakses sebelum Application.onCreate()"
            )

        // === Lazy singletons (SYNCHRONIZED = default, dibuat eksplisit untuk dokumentasi) ===

        private val _neuralEngine = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
            // Stable device capacity selects a fresh model tier. Existing checkpoints
            // keep their original tensor dimensions to avoid silently discarding weights.
            val hardwareInfo = hardwareProfile.getHardwareInfo()
            val deviceConfig = EngineConfig.forDevice(
                totalRamGb = hardwareInfo.totalRamGb,
                maxHeapMb = hardwareInfo.maxHeapMb
            )
            val resolvedConfig = ArgusNeuralEngine.resolveConfig(
                context = instance,
                totalRamGb = hardwareInfo.totalRamGb,
                maxHeapMb = hardwareInfo.maxHeapMb
            )

            if (resolvedConfig != deviceConfig) {
                Log.i(
                    TAG,
                    "Preserving saved model architecture: ${resolvedConfig.architectureLabel}"
                )
            } else {
                Log.i(TAG, "Fresh device model architecture: ${resolvedConfig.architectureLabel}")
            }

            ArgusNeuralEngine.createBlocking(instance, resolvedConfig)
        }
        val neuralEngine: ArgusNeuralEngine get() = _neuralEngine.value

        private val _worldModel = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
            WorldModel(instance)
        }
        val worldModel: WorldModel get() = _worldModel.value

        private val _deviceGuard = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
            DeviceGuard(instance)
        }
        val deviceGuard: DeviceGuard get() = _deviceGuard.value

        private val _hardwareProfile = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
            HardwareProfile(instance)
        }
        val hardwareProfile: HardwareProfile get() = _hardwareProfile.value

        private val _genesisModelManager = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
            GenesisModelManager(instance, neuralEngine, deviceGuard)
        }
        val genesisModelManager: GenesisModelManager get() = _genesisModelManager.value

        private val _trainingManager = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
            TrainingManager(instance, neuralEngine, deviceGuard, worldModel)
        }
        val trainingManager: TrainingManager get() = _trainingManager.value

        private val _repository = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
            com.argus.agi.data.repository.ArgusRepository(instance)
        }
        val repository: com.argus.agi.data.repository.ArgusRepository get() = _repository.value

        /** Untuk test / shutdown: cek apakah singleton sudah diinisialisasi. */
        fun isTrainingManagerInitialized(): Boolean = _trainingManager.isInitialized()
    }

    /** Scope aplikasi untuk pre-warm singleton. Di-cancel saat onTerminate (test). */
    private val applicationScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    override fun onCreate() {
        super.onCreate()
        _instance = this

        createNotificationChannel()

        // Pre-warm: picu lazy pada IO thread agar main thread tidak kena I/O.
        // Ini juga memicu load checkpoint neural engine di background.
        applicationScope.launch {
            runCatching { neuralEngine }
                .onFailure { Log.e(TAG, "Gagal init neuralEngine", it) }

            runCatching { deviceGuard }
                .onFailure { Log.e(TAG, "Gagal init deviceGuard", it) }

            runCatching { worldModel }
                .onFailure { Log.e(TAG, "Gagal init worldModel", it) }

            runCatching { hardwareProfile }
                .onFailure { Log.e(TAG, "Gagal init hardwareProfile", it) }

            runCatching { genesisModelManager }
                .onFailure { Log.e(TAG, "Gagal init genesisModelManager", it) }

            // trainingManager memulai worker coroutine. Pre-warm setelah
            // deviceGuard siap agar safetyState sudah terisi.
            runCatching { trainingManager }
                .onFailure { Log.e(TAG, "Gagal init trainingManager", it) }
        }
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return

        val manager = ContextCompat.getSystemService(this, NotificationManager::class.java)
            ?: return

        val channel = NotificationChannel(
            CHANNEL_ID,
            "A.R.G.U.S Neural Core & Perception",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "Status proses training on-device dan persepsi lingkungan A.R.G.U.S"
            setShowBadge(false)
            enableLights(false)
            enableVibration(false)
        }
        manager.createNotificationChannel(channel)
    }

    override fun onTerminate() {
        // Tidak dipanggil di production Android; berguna untuk instrumentation test.
        if (isTrainingManagerInitialized()) {
            runCatching { trainingManager.close() }
                .onFailure { Log.w(TAG, "Gagal menutup trainingManager", it) }
        }
        applicationScope.cancel()
        super.onTerminate()
    }
}