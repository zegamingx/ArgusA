package com.argus.agi.sensor

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.util.Log
import androidx.core.content.ContextCompat
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.Locale

data class DeviceSafetyState(
    val batteryPct: Int = 100,
    val isCharging: Boolean = true,
    val temperatureCelsius: Float = 30.0f,
    val thermalStatus: Int = 0,
    val isPhysicalThermalHazard: Boolean = false,
    val isBatteryLow: Boolean = false,
    val isBatteryOptimizedIgnored: Boolean = false,
    val isForceOverrideActive: Boolean = false,
    val isSafeToTrain: Boolean = true,
    val statusMessage: String = "Device is safe"
)

/**
 * Pengawal Keselamatan Perangkat Keras:
 * - Mengintegrasikan PowerManager.getCurrentThermalStatus() (API 29+) dan OnThermalStatusChangedListener
 * - Memisahkan secara tegas:
 *   1. Physical Thermal Hazard (Suhu SoC/baterai >= 48°C atau thermal status SEVERE): TIDAK BOLEH dilewati mode paksa demi perlindungan fisik HP.
 *   2. Battery Safety: Baterai rendah (< 10%) dapat dilewati bila user mengaktifkan Force Override.
 */
class DeviceGuard(private val context: Context) {

    companion object {
        private const val TAG = "DeviceGuard"
        const val THERMAL_CEILING_CELSIUS = 48.0f
        const val MIN_BATTERY_PCT = 10
    }

    private val _forceTrainingOverride = MutableStateFlow(false)
    val forceTrainingOverride: StateFlow<Boolean> = _forceTrainingOverride.asStateFlow()

    private val _safetyState = MutableStateFlow(DeviceSafetyState())
    val safetyState: StateFlow<DeviceSafetyState> = _safetyState.asStateFlow()

    private val powerManager = context.getSystemService(Context.POWER_SERVICE) as? PowerManager

    private val batteryReceiver = object : BroadcastReceiver() {
        override fun onReceive(c: Context?, intent: Intent?) {
            intent?.let { updateState(it) }
        }
    }

    init {
        val filter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        val initialIntent = ContextCompat.registerReceiver(
            context,
            batteryReceiver,
            filter,
            ContextCompat.RECEIVER_EXPORTED
        )
        if (initialIntent != null) {
            updateState(initialIntent)
        }

        // Listener status thermal resmi Android OS (API 29+)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q && powerManager != null) {
            runCatching {
                powerManager?.addThermalStatusListener { status ->
                    Log.d(TAG, "Android OS Thermal status changed: $status")
                    refreshState()
                }
            }.onFailure { Log.w(TAG, "Gagal mendaftarkan ThermalStatusListener", it) }
        }
    }

    fun setForceTrainingOverride(force: Boolean) {
        _forceTrainingOverride.value = force
        refreshState()
    }

    fun consumeForceOverride(): Boolean {
        val was = _forceTrainingOverride.value
        if (was) {
            _forceTrainingOverride.value = false
            refreshState()
        }
        return was
    }

    fun isSafeToTrain(): Boolean {
        refreshState()
        return _safetyState.value.isSafeToTrain
    }

    fun isIgnoringBatteryOptimizations(): Boolean {
        return powerManager?.isIgnoringBatteryOptimizations(context.packageName) ?: false
    }

    fun refreshState() {
        val filter = IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        val batteryStatus = ContextCompat.registerReceiver(
            context,
            null,
            filter,
            ContextCompat.RECEIVER_EXPORTED
        )
        if (batteryStatus != null) {
            updateState(batteryStatus)
        } else {
            val current = _safetyState.value
            val isForce = _forceTrainingOverride.value
            val safe = (!current.isPhysicalThermalHazard) && (isForce || !current.isBatteryLow)
            _safetyState.value = current.copy(
                isForceOverrideActive = isForce,
                isSafeToTrain = safe
            )
        }
    }

    private fun updateState(intent: Intent) {
        val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
        val batteryPct = if (level >= 0 && scale > 0) ((level / scale.toFloat()) * 100).toInt() else 100

        val status = intent.getIntExtra(BatteryManager.EXTRA_STATUS, -1)
        val isCharging = status == BatteryManager.BATTERY_STATUS_CHARGING ||
                status == BatteryManager.BATTERY_STATUS_FULL

        val tempTenths = intent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0)
        val tempCelsius = if (tempTenths > 0) tempTenths / 10.0f else 30.0f

        val thermalStatus = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            powerManager?.currentThermalStatus ?: 0
        } else 0

        val isIgnored = isIgnoringBatteryOptimizations()
        val isForce = _forceTrainingOverride.value

        // 1. Physical Thermal Hazard:
        // Status SEVERE (3), CRITICAL (4), EMERGENCY (5), SHUTDOWN (6) atau suhu >= 48°C
        val isThermalSevere = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            thermalStatus >= PowerManager.THERMAL_STATUS_SEVERE
        } else false
        val isPhysicalThermalHazard = tempCelsius >= THERMAL_CEILING_CELSIUS || isThermalSevere

        // 2. Battery Safety:
        val isBatteryLow = !isCharging && batteryPct in 0 until MIN_BATTERY_PCT

        // ATURAN INTEGRITAS FISIK:
        // Force Override HANYA boleh melewati batas baterai rendah,
        // TIDAK PERNAH boleh mengabaikan bahaya thermal fisik (Physical Thermal Hazard).
        val isSafe = if (isPhysicalThermalHazard) {
            false
        } else {
            if (isBatteryLow) isForce else true
        }

        val statusMsg = when {
            isPhysicalThermalHazard ->
                "THERMAL HAZARD: ${String.format(Locale.US, "%.1f", tempCelsius)}°C (limit ${THERMAL_CEILING_CELSIUS}°C). " +
                        "Training is suspended to protect the device; force override cannot bypass thermal safety."
            isBatteryLow && isForce ->
                "Force override active: the user allowed training to continue at low battery ($batteryPct%)."
            isBatteryLow ->
                "Battery is low ($batteryPct% < ${MIN_BATTERY_PCT}%). Connect a charger or enable force override."
            isForce ->
                "Force override active (${batteryPct}%, ${String.format(Locale.US, "%.1f", tempCelsius)}°C)"
            else ->
                "Device is safe (${batteryPct}%, ${String.format(Locale.US, "%.1f", tempCelsius)}°C)" +
                        (if (isCharging) " [Charging]" else "")
        }

        _safetyState.value = DeviceSafetyState(
            batteryPct = batteryPct,
            isCharging = isCharging,
            temperatureCelsius = tempCelsius,
            thermalStatus = thermalStatus,
            isPhysicalThermalHazard = isPhysicalThermalHazard,
            isBatteryLow = isBatteryLow,
            isBatteryOptimizedIgnored = isIgnored,
            isForceOverrideActive = isForce,
            isSafeToTrain = isSafe,
            statusMessage = statusMsg
        )
    }
}
