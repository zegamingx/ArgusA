package com.argus.agi

import android.app.Activity
import android.app.ActivityManager
import android.app.Application
import android.app.ApplicationExitInfo
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.content.ComponentCallbacks2
import android.os.Build
import android.os.Bundle
import android.util.Log
import androidx.core.content.ContextCompat
import com.argus.agi.diagnostics.ArgusDiagnosticLogger
import com.argus.agi.diagnostics.ArgusDiagnosticSeverity
import com.argus.agi.diagnostics.ArgusDiagnosticSubsystem
import com.argus.agi.model.ArgusNeuralEngine
import com.argus.agi.model.EngineConfig
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
    val status: String = "Assessing device",
    val driverInfo: String = "System Vulkan capability profile pending"
)

class ArgusApp : Application() {

    companion object {
        private const val TAG = "ArgusApp"
        const val CHANNEL_ID = "argus_core_service_channel"
        private const val MAX_DIAGNOSTICS_CHARS = 12_000

        @Volatile
        private var _instance: ArgusApp? = null

        /** Akses aman: throw jelas jika dipakai sebelum onCreate. */
        val instance: ArgusApp
            get() = _instance ?: error(
                "ArgusApp.instance diakses sebelum Application.onCreate()"
            )

        // === Lazy singletons (SYNCHRONIZED = default, dibuat eksplisit untuk dokumentasi) ===

        private val _hardwareInfo = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
            hardwareProfile.getHardwareInfo()
        }
        val hardwareInfo: com.argus.agi.sensor.SystemHardwareInfo
            get() = _hardwareInfo.value

        // Resolve model capacity before allocating tensors. Existing compatible checkpoints
        // retain their architecture; fresh models use this device's current memory budget.
        private val _neuralEngineConfig = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
            val profile = hardwareInfo
            ArgusNeuralEngine.resolveConfig(
                context = instance,
                totalRamGb = profile.totalRamGb,
                maxHeapMb = profile.maxHeapMb,
                availableRamGb = profile.availRamGb,
                cpuCores = profile.cpuCores,
                vulkanSupported = profile.vulkanSupported
            )
        }
        val neuralEngineConfig: EngineConfig
            get() = _neuralEngineConfig.value

        private val _neuralEngine = lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
            val resolvedConfig = neuralEngineConfig
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

        /** For lifecycle cleanup without triggering lazy singleton construction. */
        fun isWorldModelInitialized(): Boolean = _worldModel.isInitialized()

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

    private val _sharedDiagnosticsReady = MutableStateFlow(false)
    val sharedDiagnosticsReady: StateFlow<Boolean> = _sharedDiagnosticsReady.asStateFlow()

    private val diagnosticsPreviewLock = Any()
    private val diagnosticLogger: ArgusDiagnosticLogger by lazy(LazyThreadSafetyMode.SYNCHRONIZED) {
        ArgusDiagnosticLogger(this)
    }

    @Volatile
    private var initializationStage: String = "Application startup"

    @Volatile
    private var coreInitializationJob: Job? = null

    @Volatile
    private var residencyTrimJob: Job? = null

    override fun onCreate() {
        super.onCreate()

        // The process-death instrumentation service runs in its own Android process.
        // It constructs an isolated engine/repository for each crash scenario and must
        // not initialize a second application-wide model, Vulkan runtime, or worker.
        val processName = Application.getProcessName()
        if (processName != packageName) {
            Log.i(
                TAG,
                "Skipping main-process core initialization in secondary process '$processName'."
            )
            return
        }

        _instance = this
        diagnosticLogger.initialize()
        _sharedDiagnosticsReady.value = diagnosticLogger.isSharedDirectoryReady()
        if (diagnosticLogger.hasSharedStorageAccess()) {
            refreshSharedDiagnosticsStorage()
        }
        capturePreviousProcessExit()
        installUncaughtExceptionHandler()
        createNotificationChannel()
        registerActivityLifecycleCallbacks(object : ActivityLifecycleCallbacks {
            override fun onActivityCreated(activity: Activity, savedInstanceState: Bundle?) = Unit
            override fun onActivityStarted(activity: Activity) = Unit
            override fun onActivityPaused(activity: Activity) = Unit
            override fun onActivityStopped(activity: Activity) = Unit
            override fun onActivitySaveInstanceState(activity: Activity, outState: Bundle) = Unit
            override fun onActivityDestroyed(activity: Activity) = Unit

            override fun onActivityResumed(activity: Activity) {
                val engine = if (_neuralEngine.isInitialized()) _neuralEngine.value else return
                val pendingTrim = residencyTrimJob
                applicationScope.launch(Dispatchers.IO) {
                    pendingTrim?.join()
                    engine.resumeVulkanTensorResidencyAfterForeground()
                }
            }
        })
        initializeCoreRuntime()
    }

    private fun publishCoreProgress(
        status: String,
        driverInfo: String? = null
    ) {
        val current = _coreInitializationProgress.value
        _coreInitializationProgress.value = ArgusCoreInitializationProgress(
            status = status,
            driverInfo = driverInfo ?: current.driverInfo
        )
    }

    fun recordRuntimeDiagnostic(message: String) {
        recordDiagnostic(
            message = message,
            severity = ArgusDiagnosticSeverity.ERROR,
            subsystem = ArgusDiagnosticSubsystem.APP
        )
    }

    fun recordDiagnostic(
        message: String,
        severity: ArgusDiagnosticSeverity = ArgusDiagnosticSeverity.ERROR,
        subsystem: ArgusDiagnosticSubsystem = ArgusDiagnosticSubsystem.APP,
        throwable: Throwable? = null,
        immediate: Boolean = false
    ) {
        val timestamp = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSSZ", Locale.US).format(Date())
        synchronized(diagnosticsPreviewLock) {
            val updated = (_runtimeDiagnostics.value.orEmpty() + "\n[" + timestamp + "] " + message)
                .trim()
                .takeLast(MAX_DIAGNOSTICS_CHARS)
            _runtimeDiagnostics.value = updated
        }
        diagnosticLogger.record(
            subsystem = subsystem,
            severity = severity,
            message = message,
            throwable = throwable,
            immediate = immediate
        )
    }

    fun hasSharedDiagnosticsAccess(): Boolean =
        diagnosticLogger.hasSharedStorageAccess()

    fun isSharedDiagnosticsDirectoryReady(): Boolean =
        diagnosticLogger.isSharedDirectoryReady()

    fun refreshSharedDiagnosticsStorage() {
        applicationScope.launch(Dispatchers.IO) {
            val ready = runCatching {
                diagnosticLogger.synchronizeSharedStorage()
            }.getOrDefault(false)
            _sharedDiagnosticsReady.value = ready
        }
    }

    fun sharedDiagnosticsDirectoryPath(): String =
        diagnosticLogger.sharedDirectoryPath()

    private fun capturePreviousProcessExit() {
        val savedReport = diagnosticLogger.readRecent(
            subsystem = ArgusDiagnosticSubsystem.APP,
            maximumCharacters = MAX_DIAGNOSTICS_CHARS
        ).orEmpty()
        _runtimeDiagnostics.value = savedReport.takeIf { it.isNotBlank() }

        val activityManager = getSystemService(ActivityManager::class.java) ?: return
        val previousExit = runCatching {
            activityManager.getHistoricalProcessExitReasons(packageName, 0, 5).firstOrNull()
        }.getOrNull() ?: return

        // Normal app exits and explicit user stops are not diagnostic warnings.
        if (previousExit.reason == ApplicationExitInfo.REASON_EXIT_SELF ||
            previousExit.reason == ApplicationExitInfo.REASON_USER_REQUESTED ||
            previousExit.reason == ApplicationExitInfo.REASON_USER_STOPPED
        ) {
            return
        }

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
        val severity = when (previousExit.reason) {
            ApplicationExitInfo.REASON_CRASH,
            ApplicationExitInfo.REASON_CRASH_NATIVE,
            ApplicationExitInfo.REASON_ANR,
            ApplicationExitInfo.REASON_LOW_MEMORY,
            ApplicationExitInfo.REASON_INITIALIZATION_FAILURE -> ArgusDiagnosticSeverity.ERROR
            else -> ArgusDiagnosticSeverity.WARNING
        }
        recordDiagnostic(
            message = exitReport,
            severity = severity,
            subsystem = ArgusDiagnosticSubsystem.RECOVERY
        )
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
                recordDiagnostic(
                    message = "Uncaught exception on thread '" + thread.name + "'.",
                    severity = ArgusDiagnosticSeverity.ERROR,
                    subsystem = ArgusDiagnosticSubsystem.APP,
                    throwable = error,
                    immediate = true
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
    private fun initializeCoreRuntime(resetVulkanRuntime: Boolean = false) {
        synchronized(this) {
            if (coreInitializationJob?.isActive == true) return

            _coreInitializationState.value = ArgusCoreInitializationState.Initializing
            _coreInitializationProgress.value = ArgusCoreInitializationProgress()
            coreInitializationJob = applicationScope.launch {
                try {
                    if (resetVulkanRuntime && _neuralEngine.isInitialized()) {
                        initializationStage = "Resetting Vulkan runtime for retry"
                        _neuralEngine.value.resetOptionalVulkanRuntimeForRetry()
                    }

                    initializationStage = "Reading system device profile"
                    publishCoreProgress("Reading device specifications")
                    val profile = hardwareInfo
                    val advertisedVulkan = if (profile.vulkanSupported) {
                        "Vulkan API ${profile.vulkanVersionName} advertised"
                    } else {
                        "Vulkan is not advertised by the system"
                    }
                    publishCoreProgress("System device profile captured",
                        "${profile.socManufacturer} ${profile.socModel} · $advertisedVulkan"
                    )

                    initializationStage = "Registering device safety monitor"
                    publishCoreProgress("Assessing device safety")
                    deviceGuard.refreshState()
                    val safety = deviceGuard.safetyState.value

                    initializationStage = "Opening local database"
                    publishCoreProgress("Opening local database")
                    repository

                    initializationStage = "Assessing device compute and memory capacity"
                    publishCoreProgress("Assessing device compute and memory budget")
                    val modelConfig = neuralEngineConfig
                    Log.i(
                        TAG,
                        "Device assessment resolved ${modelConfig.architectureLabel}; " +
                            "available RAM ${profile.availRamGb} GB; " +
                            "training safety: ${safety.statusMessage}"
                    )
                    publishCoreProgress("Device assessment complete",
                        "${modelConfig.modelTier.displayName} · ${modelConfig.layerCount} layers · " +
                            "batch ${modelConfig.trainingBatchSize}"
                    )

                    initializationStage = "Constructing model and restoring checkpoint"
                    publishCoreProgress("Constructing or restoring model tensors")
                    val engine = neuralEngine

                    initializationStage = "Restoring world model"
                    publishCoreProgress("Restoring local world memory")
                    worldModel

                    initializationStage = "Preparing Genesis manager"
                    publishCoreProgress("Preparing model lifecycle")
                    genesisModelManager

                    // The model constructor has initialized its tensors and loaded any
                    // compatible checkpoint. Vulkan now runs on this IO worker, while
                    // the activity keeps rendering the startup progress screen.
                    initializationStage = "Initializing Vulkan after model readiness"
                    publishCoreProgress("Initializing Vulkan",
                        "Detecting Vulkan compute driver..."
                    )
                    val progressObserver = launch {
                        engine.vulkanInitializationProgress.collect { progress ->
                            if (progress.isInitializing || progress.isVerified || progress.isCpuFallback) {
                                publishCoreProgress(
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
                        status = if (vulkanReady) {
                            "Vulkan runtime verified"
                        } else {
                            "Vulkan unavailable — CPU fallback"
                        },
                        driverInfo = vulkanProgress.driverInfo
                    )

                    initializationStage = "Starting training manager"
                    publishCoreProgress("Preparing background training runtime")
                    trainingManager

                    initializationStage = "Ready"
                    publishCoreProgress("ARGUS READY", vulkanProgress.driverInfo)
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
                        ": " + error.javaClass.simpleName
                    Log.e(TAG, message, error)
                    recordDiagnostic(message, throwable = error)
                    _coreInitializationState.value = ArgusCoreInitializationState.Failed(message)
                } catch (error: Exception) {
                    val message = "Core initialization failed at " + initializationStage +
                        ": " + error.javaClass.simpleName + ": " + (error.message ?: "No message")
                    Log.e(TAG, message, error)
                    recordDiagnostic(message, throwable = error)
                    _coreInitializationState.value = ArgusCoreInitializationState.Failed(message)
                }
            }
        }
    }

    fun retryCoreInitialization() {
        initializeCoreRuntime(resetVulkanRuntime = true)
    }

    override fun onTrimMemory(level: Int) {
    super.onTrimMemory(level)
    if (level < ComponentCallbacks2.TRIM_MEMORY_BACKGROUND) return
    val engine = if (_neuralEngine.isInitialized()) _neuralEngine.value else return
    val previousTrim = residencyTrimJob
    residencyTrimJob = applicationScope.launch(Dispatchers.IO) {
        previousTrim?.join()
        val released = engine.releaseVulkanTensorResidencyForMemoryPressure()
        if (released) {
            Log.i(
                TAG,
                "Persistent Vulkan model residency released after memory-trim callback (level=$level)."
            )
        }
    }
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
        diagnosticLogger.close()
        applicationScope.cancel()
        super.onTerminate()
    }
}
