package com.argus.agi

import android.app.ActivityManager
import android.app.Application
import android.app.ApplicationExitInfo
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build
import android.util.Log
import androidx.core.content.ContextCompat
import com.argus.agi.model.ArgusNeuralEngine
import com.argus.agi.model.GenesisModelManager
import com.argus.agi.model.TrainingManager
import com.argus.agi.model.WorldModel
import com.argus.agi.sensor.DeviceGuard
import com.argus.agi.sensor.HardwareProfile
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.collect

sealed interface ArgusCoreInitializationState {
    data object Initializing : ArgusCoreInitializationState
    data object Ready : ArgusCoreInitializationState
    data class Failed(val message: String) : ArgusCoreInitializationState
}

data class ArgusCoreInitializationProgress(
    val percent: Int = 0,
    val status: String = "Assessing device",
    val driverInfo: String = "Vulkan driver detection pending"
)

class ArgusApp : Application() {

    companion object {
        private const val TAG = "ArgusApp"
        const val CHANNEL_ID = "argus_core_service_channel"
        private const val DIAGNOSTICS_FILE_NAME = "argus_runtime_diagnostics.txt"
        private const val MAX_DIAGNOSTICS_CHARS = 12_000

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
            val resolvedConfig = ArgusNeuralEngine.resolveConfig(
                context = instance,
                totalRamGb = hardwareInfo.totalRamGb,
                maxHeapMb = hardwareInfo.maxHeapMb,
                availableRamGb = hardwareInfo.availRamGb,
                cpuCores = hardwareInfo.cpuCores
            )

            Log.i(TAG, "Resolved neural engine architecture: ${resolvedConfig.architectureLabel}")

            ArgusNeuralEngine.createBlocking(instance, resolvedConfig)
        }
        val neuralEngine: ArgusNeuralEngine get() = _neuralEngine.value

        private val _worldModel = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
            WorldModel(instance, repository)
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
            TrainingManager(
                context = instance,
                neuralEngine = neuralEngine,
                deviceGuard = deviceGuard,
                worldModel = worldModel,
                repository = repository
            )
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
    private val _coreInitializationState =
        MutableStateFlow<ArgusCoreInitializationState>(ArgusCoreInitializationState.Initializing)
    val coreInitializationState: StateFlow<ArgusCoreInitializationState> =
        _coreInitializationState.asStateFlow()

    private val _coreInitializationProgress =
        MutableStateFlow(ArgusCoreInitializationProgress())
    val coreInitializationProgress: StateFlow<ArgusCoreInitializationProgress> =
        _coreInitializationProgress.asStateFlow()

    private val _runtimeDiagnostics = MutableStateFlow<String?>(null)
    val runtimeDiagnostics: StateFlow<String?> = _runtimeDiagnostics.asStateFlow()

    private val diagnosticsLock = Any()
    private val diagnosticsFile: File by lazy(LazyThreadSafetyMode.NONE) {
        File(noBackupFilesDir, DIAGNOSTICS_FILE_NAME)
    }

    @Volatile
    private var initializationStage: String = "Application startup"

    @Volatile
    private var coreInitializationJob: Job? = null

    override fun onCreate() {
        super.onCreate()
        _instance = this
        capturePreviousProcessExit()
        installUncaughtExceptionHandler()
        createNotificationChannel()
        initializeCoreRuntime()
    }

    private fun publishCoreProgress(
        percent: Int,
        status: String,
        driverInfo: String? = null
    ) {
        val current = _coreInitializationProgress.value
        _coreInitializationProgress.value = ArgusCoreInitializationProgress(
            percent = percent.coerceIn(0, 100),
            status = status,
            driverInfo = driverInfo ?: current.driverInfo
        )
    }

    fun recordRuntimeDiagnostic(message: String) {
        val timestamp = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ", Locale.US).format(Date())
        synchronized(diagnosticsLock) {
            val updated = (_runtimeDiagnostics.value.orEmpty() + "\n[" + timestamp + "] " + message)
                .trim()
                .takeLast(MAX_DIAGNOSTICS_CHARS)
            runCatching { diagnosticsFile.writeText(updated) }
            _runtimeDiagnostics.value = updated
        }
    }

    private fun capturePreviousProcessExit() {
        val savedReport = runCatching {
            if (diagnosticsFile.isFile) diagnosticsFile.readText() else ""
        }.getOrDefault("")
        _runtimeDiagnostics.value = savedReport.takeIf { it.isNotBlank() }

        val activityManager = getSystemService(ActivityManager::class.java) ?: return
        val previousExit = runCatching {
            activityManager.getHistoricalProcessExitReasons(packageName, 0, 5).firstOrNull()
        }.getOrNull() ?: return

        val exitReport = buildString {
            appendLine("Previous process exit")
            appendLine("Reason: " + exitReasonLabel(previousExit.reason) + " (#" + previousExit.reason + ")")
            appendLine("Timestamp: " + SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ", Locale.US).format(Date(previousExit.timestamp)))
            appendLine("Exit status: " + previousExit.status)
            appendLine("Process importance: " + previousExit.importance)
            appendLine("PSS: " + previousExit.pss)
            appendLine("RSS: " + previousExit.rss)
            append("System description: " + (previousExit.description ?: "Not provided"))
        }
        recordRuntimeDiagnostic(exitReport)
    }

    private fun exitReasonLabel(reason: Int): String = when (reason) {
        ApplicationExitInfo.REASON_EXIT_SELF -> "App exited itself"
        ApplicationExitInfo.REASON_SIGNALED -> "Process terminated by signal"
        ApplicationExitInfo.REASON_LOW_MEMORY -> "System killed process for low memory"
        ApplicationExitInfo.REASON_CRASH -> "Java/Kotlin crash"
        ApplicationExitInfo.REASON_CRASH_NATIVE -> "Native crash"
        ApplicationExitInfo.REASON_ANR -> "Application not responding"
        ApplicationExitInfo.REASON_INITIALIZATION_FAILURE -> "Process initialization failure"
        ApplicationExitInfo.REASON_PERMISSION_CHANGE -> "Permission change"
        ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "Excessive resource usage"
        ApplicationExitInfo.REASON_USER_REQUESTED -> "User-requested process stop"
        ApplicationExitInfo.REASON_USER_STOPPED -> "App stopped by system user action"
        ApplicationExitInfo.REASON_DEPENDENCY_DIED -> "Dependency process died"
        ApplicationExitInfo.REASON_OTHER -> "Other process exit"
        else -> "Unknown process exit"
    }

    private fun installUncaughtExceptionHandler() {
        val previousHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, error ->
            runCatching {
                recordRuntimeDiagnostic(
                    "Uncaught exception on thread '" + thread.name + "': " +
                        error.javaClass.name + ": " + (error.message ?: "No message") +
                        "\n" + Log.getStackTraceString(error)
                )
            }.onFailure { reportingError ->
                Log.e(TAG, "Could not persist the uncaught exception report.", reportingError)
            }
            if (previousHandler != null) {
                previousHandler.uncaughtException(thread, error)
            } else {
                android.os.Process.killProcess(android.os.Process.myPid())
                kotlin.system.exitProcess(10)
            }
        }
    }

    /**
     * Initializes blocking singletons on the application IO scope.
     * The activity must not create its ViewModel until this state becomes Ready.
     */
    private fun initializeCoreRuntime() {
        synchronized(this) {
            if (coreInitializationJob?.isActive == true) return

            _coreInitializationState.value = ArgusCoreInitializationState.Initializing
            _coreInitializationProgress.value = ArgusCoreInitializationProgress()
            coreInitializationJob = applicationScope.launch {
                try {
                    initializationStage = "Reading hardware profile"
                    publishCoreProgress(8, "Assessing device")
                    hardwareProfile

                    initializationStage = "Registering device safety monitor"
                    publishCoreProgress(16, "Preparing device safety")
                    deviceGuard

                    initializationStage = "Opening local database"
                    publishCoreProgress(24, "Opening local database")
                    repository

                    initializationStage = "Allocating neural engine and restoring model tensors"
                    publishCoreProgress(42, "Constructing and restoring the local model")
                    val engine = neuralEngine

                    initializationStage = "Restoring world model"
                    publishCoreProgress(58, "Restoring local world memory")
                    worldModel

                    initializationStage = "Preparing Genesis manager"
                    publishCoreProgress(66, "Preparing model lifecycle")
                    genesisModelManager

                    // The model constructor has initialized its tensors and loaded any
                    // compatible checkpoint. Vulkan now runs on this IO worker, while
                    // the activity keeps rendering the startup progress screen.
                    initializationStage = "Initializing Vulkan after model readiness"
                    publishCoreProgress(
                        72,
                        "Initializing Vulkan",
                        "Detecting Vulkan compute driver..."
                    )
                    val progressObserver = launch {
                        engine.vulkanInitializationProgress.collect { progress ->
                            if (progress.isInitializing || progress.isVerified || progress.isCpuFallback) {
                                publishCoreProgress(
                                    percent = progress.percent,
                                    status = progress.status,
                                    driverInfo = progress.driverInfo
                                )
                            }
                        }
                    }
                    val vulkanReady = try {
                        engine.initializeOptionalVulkanRuntime()
                    } finally {
                        progressObserver.cancel()
                    }
                    val vulkanProgress = engine.vulkanInitializationProgress.value
                    publishCoreProgress(
                        percent = if (vulkanReady) 92 else 88,
                        status = if (vulkanReady) {
                            "Vulkan runtime verified"
                        } else {
                            "Vulkan unavailable — CPU fallback"
                        },
                        driverInfo = vulkanProgress.driverInfo
                    )

                    initializationStage = "Starting training manager"
                    publishCoreProgress(96, "Preparing background training runtime")
                    trainingManager

                    initializationStage = "Ready"
                    publishCoreProgress(100, "ARGUS READY", vulkanProgress.driverInfo)
                    _coreInitializationState.value = ArgusCoreInitializationState.Ready
                    Log.i(TAG, "Core runtime initialization completed.")
                } catch (cancelled: CancellationException) {
                    throw cancelled
                } catch (error: OutOfMemoryError) {
                    val message = "Insufficient memory while initializing Argus at: " +
                        initializationStage + ". The model was not started."
                    _coreInitializationState.value = ArgusCoreInitializationState.Failed(message)
                    runCatching { recordRuntimeDiagnostic(message) }
                } catch (error: LinkageError) {
                    val message = "Core initialization failed at " + initializationStage +
                        ": " + error.javaClass.simpleName + ": " + (error.message ?: "No message")
                    Log.e(TAG, message, error)
                    recordRuntimeDiagnostic(message + "\n" + Log.getStackTraceString(error))
                    _coreInitializationState.value = ArgusCoreInitializationState.Failed(message)
                } catch (error: Exception) {
                    val message = "Core initialization failed at " + initializationStage +
                        ": " + error.javaClass.simpleName + ": " + (error.message ?: "No message")
                    Log.e(TAG, message, error)
                    recordRuntimeDiagnostic(message + "\n" + Log.getStackTraceString(error))
                    _coreInitializationState.value = ArgusCoreInitializationState.Failed(message)
                }
            }
        }
    }

    fun retryCoreInitialization() {
        initializeCoreRuntime()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return

        val manager = ContextCompat.getSystemService(this, NotificationManager::class.java)
            ?: return

        val channel = NotificationChannel(
            CHANNEL_ID,
            "A.R.G.U.S Neural Core and Perception",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = "On-device training status and A.R.G.U.S environment perception."
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
                .onFailure { Log.w(TAG, "Failed to close the training manager", it) }
        }
        applicationScope.cancel()
        super.onTerminate()
    }
}