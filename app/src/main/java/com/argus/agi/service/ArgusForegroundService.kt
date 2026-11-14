package com.argus.agi.service

import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.argus.agi.ArgusApp
import com.argus.agi.ArgusCoreInitializationState
import com.argus.agi.MainActivity
import com.argus.agi.R

class ArgusForegroundService : Service() {

    companion object {
        const val NOTIFICATION_ID = 1001
        const val ACTION_START = "com.argus.agi.START_SERVICE"
        const val ACTION_STOP = "com.argus.agi.STOP_SERVICE"
        const val ACTION_TOGGLE_PAUSE = "com.argus.agi.TOGGLE_PAUSE"

        private const val TAG = "ArgusForegroundService"

        fun startService(context: Context) {
            val intent = Intent(context, ArgusForegroundService::class.java).apply {
                action = ACTION_START
            }
            context.startForegroundService(intent)
        }

        fun stopService(context: Context) {
            val intent = Intent(context, ArgusForegroundService::class.java).apply {
                action = ACTION_STOP
            }
            context.stopService(intent)
        }

        internal fun foregroundServiceTypeForApi(sdkInt: Int): Int =
            if (sdkInt >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE
            } else {
                ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC
            }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
                return START_NOT_STICKY
            }

            else -> {
                // Android requires immediate foreground promotion after startForegroundService().
                // This notification does not access model singletons, Room, JNI, or GPU drivers.
                startForegroundCompat()

                if (intent?.action == ACTION_TOGGLE_PAUSE) {
                    val app = application as? ArgusApp
                    if (app?.coreInitializationState?.value is ArgusCoreInitializationState.Ready) {
                        runCatching {
                            ArgusApp.trainingManager.togglePauseTraining()
                        }.onFailure { error ->
                            Log.e(TAG, "Could not toggle training from the foreground notification.", error)
                        }
                    } else {
                        Log.w(TAG, "Ignored notification training action while the core runtime is not ready.")
                    }
                }
            }
        }
        return START_STICKY
    }

    private fun startForegroundCompat() {
        val notification = buildNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                foregroundServiceTypeForApi(Build.VERSION.SDK_INT)
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
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

        return NotificationCompat.Builder(this, ArgusApp.CHANNEL_ID)
            .setContentTitle("A.R.G.U.S Neural Core")
            .setContentText("On-device compute service is active")
            .setSmallIcon(R.drawable.ic_argus_logo)
            .setContentIntent(openAppPendingIntent)
            .setOngoing(true)
            .addAction(0, "Pause / resume training", togglePendingIntent)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }
}
