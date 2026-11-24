package com.argus.agi.service

import com.argus.agi.diagnostics.ArgusDiagnosticBridge

import android.app.Notification
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.argus.agi.ArgusApp
import com.argus.agi.ArgusCoreInitializationState
import com.argus.agi.MainActivity
import com.argus.agi.R
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch

/**
 * Training state as presented by the foreground-service notification. The notification
 * must describe the real TrainingManager state (AGENTS.md §5.2/§14): it never claims
 * training is running when it is paused or idle.
 */
internal enum class ForegroundTrainingPresentation {
    STARTING,
    ACTIVE,
    PAUSED,
    IDLE
}

class ArgusForegroundService : Service() {

    companion object {
        const val NOTIFICATION_ID = 1001
        const val ACTION_START = "com.argus.agi.START_SERVICE"
        const val ACTION_STOP = "com.argus.agi.STOP_SERVICE"
        const val ACTION_TOGGLE_PAUSE = "com.argus.agi.TOGGLE_PAUSE"

        private const val TAG = "ArgusForegroundService"

        // The user-stop latch lives in the app's existing SharedPreferences pattern so an
        // explicit "stop the background service" choice survives process death and is
        // honoured by START_STICKY restarts. It is cleared only when the user explicitly
        // resumes/starts training from the UI (ArgusViewModel).
        private const val PREFERENCES_NAME = "argus_foreground_service"
        private const val KEY_USER_STOPPED = "user_stopped_background_service"

        /**
         * minSdk is 34, where specialUse is available, so the runtime type is
         * unconditionally SPECIAL_USE. There is no dataSync variant: API 29-33 devices
         * cannot install the app.
         */
        internal val foregroundServiceType: Int =
            ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE

        internal fun presentationFor(
            isPaused: Boolean,
            hasActiveWork: Boolean,
            hasQueuedWork: Boolean
        ): ForegroundTrainingPresentation = when {
            isPaused -> ForegroundTrainingPresentation.PAUSED
            hasActiveWork || hasQueuedWork -> ForegroundTrainingPresentation.ACTIVE
            else -> ForegroundTrainingPresentation.IDLE
        }

        internal fun isUserStopped(context: Context): Boolean =
            context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)
                .getBoolean(KEY_USER_STOPPED, false)

        internal fun setUserStopped(context: Context) {
            context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)
                .edit()
                .putBoolean(KEY_USER_STOPPED, true)
                .apply()
        }

        internal fun clearUserStopped(context: Context) {
            context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)
                .edit()
                .putBoolean(KEY_USER_STOPPED, false)
                .apply()
        }

        fun startService(context: Context) {
            if (isUserStopped(context)) {
                ArgusDiagnosticBridge.warning(
                    TAG,
                    "Foreground service start skipped: the user stopped the background service."
                )
                return
            }
            val intent = Intent(context, ArgusForegroundService::class.java).apply {
                action = ACTION_START
            }
            context.startForegroundService(intent)
        }

        /**
         * User-initiated stop, reachable from the app UI: latch the choice, pause training
         * if it is running, then stop the service. The latch keeps ViewModel recreation
         * and START_STICKY restarts from silently resurrecting the service.
         */
        fun requestStop(context: Context) {
            setUserStopped(context)
            pauseTrainingIfRunning(context)
            val intent = Intent(context, ArgusForegroundService::class.java)
            context.stopService(intent)
        }

        private fun pauseTrainingIfRunning(context: Context) {
            val app = context.applicationContext as? ArgusApp ?: return
            if (app.coreInitializationState.value !is ArgusCoreInitializationState.Ready) return
            if (!ArgusApp.isTrainingManagerInitialized()) return
            runCatching {
                val trainingManager = ArgusApp.trainingManager
                if (!trainingManager.isTrainingPaused.value) {
                    trainingManager.pauseTraining()
                }
            }.onFailure { error ->
                ArgusDiagnosticBridge.error(TAG, "Could not pause training while stopping the foreground service.", error)
            }
        }
    }

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var trainingObservationJob: Job? = null

    @Volatile
    private var presentation: ForegroundTrainingPresentation =
        ForegroundTrainingPresentation.STARTING

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            // User stop from the notification action: latch the choice and pause training
            // before tearing the service down, so the stop is real and stays stopped.
            setUserStopped(this)
            pauseTrainingIfRunning(this)
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }

        if (isUserStopped(this)) {
            // START_STICKY restart after process death, or a stray start intent: the user
            // turned the background service off, so do not re-promote.
            stopSelfResult(startId)
            return START_NOT_STICKY
        }

        // Android requires immediate foreground promotion after startForegroundService().
        // This notification does not access model singletons, Room, JNI, or GPU drivers.
        try {
            startForegroundCompat()
        } catch (error: Exception) {
            val message = "Foreground service promotion failed: " + error.javaClass.simpleName
            ArgusDiagnosticBridge.error(TAG, message, error)
            stopSelfResult(startId)
            return START_NOT_STICKY
        }

        startObservingTrainingState()

        if (intent?.action == ACTION_TOGGLE_PAUSE) {
            val app = application as? ArgusApp
            if (app?.coreInitializationState?.value is ArgusCoreInitializationState.Ready) {
                runCatching {
                    ArgusApp.trainingManager.togglePauseTraining()
                }.onFailure { error ->
                    ArgusDiagnosticBridge.error(TAG, "Could not toggle training from the foreground notification.", error)
                }
            } else {
                ArgusDiagnosticBridge.warning(TAG, "Ignored notification training action while the core runtime is not ready.")
            }
        }
        return START_STICKY
    }

    override fun onDestroy() {
        serviceScope.cancel()
        super.onDestroy()
    }

    private fun startForegroundCompat() {
        startForeground(NOTIFICATION_ID, buildNotification(), foregroundServiceType)
    }

    /**
     * Observes the real training state and keeps the notification honest: its text and
     * the pause/resume action label follow TrainingManager instead of a static string.
     * Runs on the service scope after promotion; never blocks onStartCommand.
     */
    private fun startObservingTrainingState() {
        if (trainingObservationJob?.isActive == true) return
        val app = application as? ArgusApp ?: return
        trainingObservationJob = serviceScope.launch {
            app.coreInitializationState.first {
                it is ArgusCoreInitializationState.Ready
            }
            if (!ArgusApp.isTrainingManagerInitialized()) return@launch
            val trainingManager = ArgusApp.trainingManager
            combine(
                trainingManager.isTrainingPaused,
                trainingManager.currentTrainingItem,
                trainingManager.activeDocument,
                trainingManager.nextQueue
            ) { isPaused, currentItem, activeDocument, queue ->
                presentationFor(
                    isPaused = isPaused,
                    hasActiveWork = currentItem != null || activeDocument != null,
                    hasQueuedWork = queue.isNotEmpty()
                )
            }
                .distinctUntilChanged()
                .collect { updated ->
                    presentation = updated
                    runCatching {
                        getSystemService(NotificationManager::class.java)
                            ?.notify(NOTIFICATION_ID, buildNotification())
                    }.onFailure { error ->
                        ArgusDiagnosticBridge.error(TAG, "Could not refresh the foreground notification for the new training state.", error)
                    }
                }
        }
    }

    /**
     * Keep this notification lightweight. The Android foreground-service deadline must not
     * depend on model allocation, Room restoration, native-library loading, or a GPU driver.
     */
    private fun buildNotification(): Notification {
        val openAppIntent = Intent(this, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val openAppPendingIntent = PendingIntent.getActivity(
            this,
            0,
            openAppIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val toggleIntent = Intent(this, ArgusForegroundService::class.java).apply {
            action = ACTION_TOGGLE_PAUSE
        }
        val togglePendingIntent = PendingIntent.getService(
            this,
            1,
            toggleIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val stopIntent = Intent(this, ArgusForegroundService::class.java).apply {
            action = ACTION_STOP
        }
        val stopPendingIntent = PendingIntent.getService(
            this,
            2,
            stopIntent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )

        val currentPresentation = presentation
        val contentText = when (currentPresentation) {
            ForegroundTrainingPresentation.STARTING ->
                "On-device compute service is starting"
            ForegroundTrainingPresentation.ACTIVE ->
                "Training in progress"
            ForegroundTrainingPresentation.PAUSED ->
                "Training paused"
            ForegroundTrainingPresentation.IDLE ->
                "Neural core ready — training idle"
        }
        val toggleLabel = when (currentPresentation) {
            ForegroundTrainingPresentation.PAUSED -> "Resume training"
            else -> "Pause training"
        }

        return NotificationCompat.Builder(this, ArgusApp.CHANNEL_ID)
            .setContentTitle("A.R.G.U.S Neural Core")
            .setContentText(contentText)
            .setSmallIcon(R.drawable.ic_argus_logo)
            .setContentIntent(openAppPendingIntent)
            .setOngoing(true)
            .addAction(0, toggleLabel, togglePendingIntent)
            .addAction(0, "Stop", stopPendingIntent)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }
}
