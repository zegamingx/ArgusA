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
import android.os.PowerManager
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
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.isActive
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

/**
 * Chunk progress of the document currently being trained, mirrored into the
 * foreground-service notification so background progress is visible without
 * opening the app — and a stalled count is just as visible. Values come
 * straight from TrainingManager's active-document state (AGENTS.md §14):
 * real training state only, never an estimate.
 */
internal data class ForegroundTrainingProgress(
    val completedChunks: Int,
    val totalChunks: Int
) {
    val percent: Int
        get() = if (totalChunks > 0) {
            (completedChunks.coerceIn(0, totalChunks) * 100) / totalChunks
        } else {
            0
        }

    /** Same wording as the in-app training header: "2311 of 10000 chunks (23%)". */
    val summary: String
        get() = "$completedChunks of $totalChunks chunks ($percent%)"
}

/**
 * Owns the partial WakeLock that keeps the CPU awake while training is actually
 * running in the background (owner requirement: background training must not be
 * throttled to a stop by Doze when the screen is off).
 *
 * The lock is held ONLY in the ACTIVE presentation state — never merely because
 * the foreground service exists — so an idle or paused Argus does not drain the
 * battery (AGENTS.md §13). All transitions go through [update]/[renew]/[release],
 * which are synchronized and check `isHeld` first: duplicate state emissions can
 * neither double-acquire nor over-release. The underlying lock is not reference
 * counted, so the held state is a single boolean owned by this holder.
 *
 * Acquisition uses a bounded timeout ([ACQUIRE_TIMEOUT_MS]) as a safety net: if
 * the service is killed mid-hold the system reclaims the lock instead of it
 * leaking forever. While training stays ACTIVE the service periodically calls
 * [renew] well before the timeout expires, so a long run never loses the lock.
 *
 * Any acquisition failure is reported through [onFailure] and swallowed:
 * training continues without the lock (degraded, logged), never crashes.
 */
internal class TrainingWakeLock(
    private val wakeLock: PowerManager.WakeLock?,
    private val onFailure: (String, Throwable?) -> Unit = { _, _ -> }
) {

    init {
        wakeLock?.setReferenceCounted(false)
    }

    val isHeld: Boolean
        get() = wakeLock?.isHeld == true

    /** Single transition point driven by the training-state collector. */
    @Synchronized
    fun update(trainingActive: Boolean) {
        if (trainingActive) {
            acquireLocked()
        } else {
            releaseLocked()
        }
    }

    /**
     * Extends the hold while training remains ACTIVE: releases and re-acquires
     * with a fresh timeout (or acquires outright if the previous hold already
     * timed out), so the bounded timeout never cuts a long run short.
     */
    @Synchronized
    fun renew() {
        if (isHeld) {
            releaseLocked()
        }
        acquireLocked()
    }

    /** Unconditional release for service teardown. Safe to call when not held. */
    @Synchronized
    fun release() {
        releaseLocked()
    }

    private fun acquireLocked() {
        val lock = wakeLock ?: return
        if (lock.isHeld) return
        runCatching { lock.acquire(ACQUIRE_TIMEOUT_MS) }
            .onFailure { error ->
                onFailure("Could not acquire the training WakeLock; training continues without it.", error)
            }
    }

    private fun releaseLocked() {
        val lock = wakeLock ?: return
        if (!lock.isHeld) return
        runCatching { lock.release() }
            .onFailure { error ->
                onFailure("Could not release the training WakeLock.", error)
            }
    }

    companion object {
        /** Bounded hold; renewed by the service while training stays ACTIVE. */
        const val ACQUIRE_TIMEOUT_MS: Long = 10 * 60 * 1000L

        /** Renewal cadence: comfortably before [ACQUIRE_TIMEOUT_MS] expires. */
        const val RENEWAL_INTERVAL_MS: Long = 5 * 60 * 1000L
    }
}

class ArgusForegroundService : Service() {

    companion object {
        const val NOTIFICATION_ID = 1001
        const val ACTION_START = "com.argus.agi.START_SERVICE"
        const val ACTION_STOP = "com.argus.agi.STOP_SERVICE"
        const val ACTION_TOGGLE_PAUSE = "com.argus.agi.TOGGLE_PAUSE"

        private const val TAG = "ArgusForegroundService"

        /** Partial WakeLock tag; held only while training is ACTIVE. */
        internal const val WAKE_LOCK_TAG = "Argus:TrainingWakeLock"

        // The user-stop latch lives in the app's existing SharedPreferences pattern so an
        // explicit "stop the background service" choice survives process death and is
        // honoured by START_STICKY restarts. It is cleared only when the user explicitly
        // resumes/starts training from the UI (ArgusViewModel).
        private const val PREFERENCES_NAME = "argus_foreground_service"
        private const val KEY_USER_STOPPED = "user_stopped_background_service"

        /**
         * Let Android apply the service types declared in AndroidManifest.xml.
         * The manifest owns the specialUse declaration and its subtype description.
         */
        internal val foregroundServiceType: Int =
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MANIFEST

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
    private var wakeLockRenewalJob: Job? = null

    /**
     * Created on the first training-state emission, i.e. always after foreground
     * promotion, so WakeLock setup can never delay promotion. A null underlying
     * lock (no PowerManager, or creation failure) degrades to logged no-ops
     * inside [TrainingWakeLock].
     */
    @Volatile
    private var trainingWakeLock: TrainingWakeLock? = null

    private fun obtainTrainingWakeLock(): TrainingWakeLock =
        trainingWakeLock ?: synchronized(this) {
            trainingWakeLock ?: run {
                val lock = runCatching {
                    getSystemService(PowerManager::class.java)
                        ?.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKE_LOCK_TAG)
                }.onFailure { error ->
                    ArgusDiagnosticBridge.error(TAG, "Could not create the training WakeLock; training continues without it.", error)
                }.getOrNull()
                TrainingWakeLock(lock) { message, error ->
                    if (error != null) {
                        ArgusDiagnosticBridge.error(TAG, message, error)
                    } else {
                        ArgusDiagnosticBridge.warning(TAG, message)
                    }
                }.also { trainingWakeLock = it }
            }
        }

    @Volatile
    private var presentation: ForegroundTrainingPresentation =
        ForegroundTrainingPresentation.STARTING

    @Volatile
    private var progress: ForegroundTrainingProgress? = null

    /** One notification refresh unit: re-post only when presentation or progress moves. */
    private data class NotificationSnapshot(
        val presentation: ForegroundTrainingPresentation,
        val progress: ForegroundTrainingProgress?
    )

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            // User stop from the notification action: latch the choice and pause training
            // before tearing the service down, so the stop is real and stays stopped.
            setUserStopped(this)
            pauseTrainingIfRunning(this)
            trainingWakeLock?.release()
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
        // Cancel the scope first so the renewal loop cannot re-acquire, then
        // release unconditionally: the lock must never outlive the service.
        serviceScope.cancel()
        trainingWakeLock?.release()
        super.onDestroy()
    }

    private fun startForegroundCompat() {
        startForeground(
            NOTIFICATION_ID,
            buildNotification(),
            foregroundServiceType
        )
    }

    /**
     * Observes the real training state and keeps the notification honest and live:
     * its text and the pause/resume action label follow TrainingManager instead of
     * a static string, and while a document is being trained the notification also
     * carries that document's chunk progress (text + determinate progress bar),
     * re-posted every time a chunk completes. Because the snapshot is
     * distinct-until-changed, an idle or paused Argus never spams the shade, and a
     * background stall is directly visible as a frozen count. Runs on the service
     * scope after promotion; never blocks onStartCommand.
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
                NotificationSnapshot(
                    presentation = presentationFor(
                        isPaused = isPaused,
                        hasActiveWork = currentItem != null || activeDocument != null,
                        hasQueuedWork = queue.isNotEmpty()
                    ),
                    progress = activeDocument
                        ?.takeIf { it.totalChunks > 0 }
                        ?.let { document ->
                            ForegroundTrainingProgress(
                                completedChunks = document.completedChunks,
                                totalChunks = document.totalChunks
                            )
                        }
                )
            }
                .distinctUntilChanged()
                .collect { snapshot ->
                    presentation = snapshot.presentation
                    progress = snapshot.progress
                    updateWakeLockForPresentation(snapshot.presentation)
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
     * Drives the training WakeLock from the same presentation state as the
     * notification: held only while ACTIVE. While ACTIVE, a renewal loop in the
     * service scope re-acquires the lock every [TrainingWakeLock.RENEWAL_INTERVAL_MS]
     * so the bounded acquire timeout never cuts a long training run short; the
     * loop is cancelled (and the lock released) on the first non-ACTIVE state.
     */
    private fun updateWakeLockForPresentation(state: ForegroundTrainingPresentation) {
        val holder = obtainTrainingWakeLock()
        if (state == ForegroundTrainingPresentation.ACTIVE) {
            holder.update(trainingActive = true)
            if (wakeLockRenewalJob?.isActive != true) {
                wakeLockRenewalJob = serviceScope.launch {
                    while (isActive) {
                        delay(TrainingWakeLock.RENEWAL_INTERVAL_MS)
                        if (presentation == ForegroundTrainingPresentation.ACTIVE) {
                            holder.renew()
                        }
                    }
                }
            }
        } else {
            wakeLockRenewalJob?.cancel()
            wakeLockRenewalJob = null
            holder.update(trainingActive = false)
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
        val currentProgress = progress
        val progressSuffix = currentProgress?.let { " • ${it.summary}" } ?: ""
        val contentText = when (currentPresentation) {
            ForegroundTrainingPresentation.STARTING ->
                "On-device compute service is starting"
            ForegroundTrainingPresentation.ACTIVE ->
                "Training in progress$progressSuffix"
            ForegroundTrainingPresentation.PAUSED ->
                "Training paused$progressSuffix"
            ForegroundTrainingPresentation.IDLE ->
                "Neural core ready — training idle"
        }
        val toggleLabel = when (currentPresentation) {
            ForegroundTrainingPresentation.PAUSED -> "Resume training"
            else -> "Pause training"
        }

        val builder = NotificationCompat.Builder(this, ArgusApp.CHANNEL_ID)
            .setContentTitle("A.R.G.U.S Neural Core")
            .setContentText(contentText)
            .setStyle(NotificationCompat.BigTextStyle().bigText(contentText))
            .setSmallIcon(R.drawable.ic_argus_logo)
            .setContentIntent(openAppPendingIntent)
            .setOngoing(true)
            .addAction(0, toggleLabel, togglePendingIntent)
            .addAction(0, "Stop", stopPendingIntent)
            .setPriority(NotificationCompat.PRIORITY_LOW)
        if (currentProgress != null &&
            (currentPresentation == ForegroundTrainingPresentation.ACTIVE ||
                currentPresentation == ForegroundTrainingPresentation.PAUSED)
        ) {
            builder.setProgress(
                currentProgress.totalChunks,
                currentProgress.completedChunks.coerceIn(0, currentProgress.totalChunks),
                false
            )
        }
        return builder.build()
    }
}
