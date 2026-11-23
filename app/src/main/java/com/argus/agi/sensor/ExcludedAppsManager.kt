package com.argus.agi.sensor

import com.argus.agi.diagnostics.ArgusDiagnosticBridge

import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.drawable.Drawable
import android.os.Build
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext

data class InstalledAppItem(
    val appName: String,
    val packageName: String,
    val icon: Drawable?,
    val isExcluded: Boolean,
    val isSystemApp: Boolean = false,
    val isTrainingAllowed: Boolean = false
)

/**
 * Owns the user's local screen-perception policy.
 *
 * Screen perception, screen-derived training, and accessibility actions are
 * separate opt-ins. App access is default-deny and protected categories cannot
 * be added to the allowlist.
 */
class ExcludedAppsManager(private val context: Context) {

    companion object {
        private const val TAG = "ExcludedAppsManager"
        private const val PREFS_NAME = "argus_excluded_apps"
        private const val KEY_ALLOWED_PACKAGES = "allowed_screen_packages"
        private const val KEY_TRAINING_ALLOWED_PACKAGES = "allowed_screen_training_packages"
        private const val KEY_PROTECTED_PACKAGES = "protected_screen_packages"
        private const val KEY_SCREEN_PERCEPTION = "screen_perception_enabled"
        private const val KEY_SCREEN_TRAINING = "screen_training_enabled"
        private const val KEY_ACCESSIBILITY_ACTIONS = "accessibility_actions_enabled"
    }

    private val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    private val _excludedPackages = MutableStateFlow<Set<String>>(emptySet())
    val excludedPackages: StateFlow<Set<String>> = _excludedPackages.asStateFlow()

    private val _screenPerceptionEnabled = MutableStateFlow(
        prefs.getBoolean(KEY_SCREEN_PERCEPTION, false)
    )
    val screenPerceptionEnabled: StateFlow<Boolean> = _screenPerceptionEnabled.asStateFlow()

    private val _screenTrainingEnabled = MutableStateFlow(
        prefs.getBoolean(KEY_SCREEN_TRAINING, false)
    )
    val screenTrainingEnabled: StateFlow<Boolean> = _screenTrainingEnabled.asStateFlow()

    private val _accessibilityActionsEnabled = MutableStateFlow(
        prefs.getBoolean(KEY_ACCESSIBILITY_ACTIONS, false)
    )
    val accessibilityActionsEnabled: StateFlow<Boolean> = _accessibilityActionsEnabled.asStateFlow()

    fun isScreenPerceptionEnabled(): Boolean =
        prefs.getBoolean(KEY_SCREEN_PERCEPTION, false)

    fun isScreenTrainingEnabled(): Boolean =
        isScreenPerceptionEnabled() && prefs.getBoolean(KEY_SCREEN_TRAINING, false)

    fun isAccessibilityActionsEnabled(): Boolean =
        prefs.getBoolean(KEY_ACCESSIBILITY_ACTIONS, false)

    /**
     * Accessibility actions require independent global consent and an explicitly
     * allowed target package. Unknown package identities always fail closed.
     */
    fun isAccessibilityActionAllowed(packageName: String?): Boolean {
        if (!isAccessibilityActionsEnabled()) return false
        val normalizedPackage = packageName?.trim()?.takeIf { it.isNotEmpty() } ?: return false
        return !isAppExcluded(normalizedPackage)
    }

    fun isAppTrainingAllowed(packageName: String): Boolean {
        if (!isScreenTrainingEnabled() || isAppExcluded(packageName)) return false
        val trainingAllowed = prefs.getStringSet(KEY_TRAINING_ALLOWED_PACKAGES, emptySet()).orEmpty()
        return packageName in trainingAllowed
    }

    fun toggleAppTrainingAccess(packageName: String) {
        val protected = prefs.getStringSet(KEY_PROTECTED_PACKAGES, emptySet()).orEmpty()
        if (packageName in protected ||
            PrivacyAppHeuristics.isPrivacySensitiveApp("", packageName)
        ) {
            return
        }

        val allowedForScreen = prefs.getStringSet(KEY_ALLOWED_PACKAGES, emptySet()).orEmpty()
        if (packageName !in allowedForScreen) return

        val current = prefs.getStringSet(KEY_TRAINING_ALLOWED_PACKAGES, emptySet())
            .orEmpty()
            .toMutableSet()
        if (!current.add(packageName)) current.remove(packageName)
        prefs.edit().putStringSet(KEY_TRAINING_ALLOWED_PACKAGES, current.toSet()).apply()
    }

    fun setScreenPerceptionEnabled(enabled: Boolean) {
        prefs.edit()
            .putBoolean(KEY_SCREEN_PERCEPTION, enabled)
            .apply()
        _screenPerceptionEnabled.value = enabled

        if (!enabled) {
            prefs.edit().putBoolean(KEY_SCREEN_TRAINING, false).apply()
            _screenTrainingEnabled.value = false
        }
    }

    fun setScreenTrainingEnabled(enabled: Boolean) {
        val permitted = enabled && isScreenPerceptionEnabled()
        prefs.edit()
            .putBoolean(KEY_SCREEN_TRAINING, permitted)
            .apply()
        _screenTrainingEnabled.value = permitted
    }

    fun setAccessibilityActionsEnabled(enabled: Boolean) {
        prefs.edit()
            .putBoolean(KEY_ACCESSIBILITY_ACTIONS, enabled)
            .apply()
        _accessibilityActionsEnabled.value = enabled
    }

    /**
     * Revokes all screen-reading and screen-training app grants.
     * The independent accessibility-action consent is intentionally preserved.
     */
    fun clearScreenAppAllowlist() {
        prefs.edit()
            .putStringSet(KEY_ALLOWED_PACKAGES, emptySet())
            .putStringSet(KEY_TRAINING_ALLOWED_PACKAGES, emptySet())
            .apply()
        _excludedPackages.value = emptySet()
    }

    /**
     * Every package is blocked until the user explicitly allows it.
     * Sensitive categories remain blocked even if stale preferences contain
     * their package names.
     */
    fun isAppExcluded(packageName: String): Boolean {
        val protected = prefs.getStringSet(KEY_PROTECTED_PACKAGES, emptySet()).orEmpty()
        if (packageName in protected ||
            PrivacyAppHeuristics.isPrivacySensitiveApp("", packageName)
        ) {
            return true
        }
        val allowed = prefs.getStringSet(KEY_ALLOWED_PACKAGES, emptySet()).orEmpty()
        return packageName !in allowed
    }

    fun toggleAppAccess(packageName: String) {
        val protected = prefs.getStringSet(KEY_PROTECTED_PACKAGES, emptySet()).orEmpty()
        if (packageName in protected ||
            PrivacyAppHeuristics.isPrivacySensitiveApp("", packageName)
        ) {
            return
        }

        val current = prefs.getStringSet(KEY_ALLOWED_PACKAGES, emptySet())
            .orEmpty()
            .toMutableSet()
        val isNowAllowed = current.add(packageName)
        if (!isNowAllowed) {
            current.remove(packageName)
            val trainingAllowed = prefs.getStringSet(KEY_TRAINING_ALLOWED_PACKAGES, emptySet())
                .orEmpty()
                .toMutableSet()
            trainingAllowed.remove(packageName)
            prefs.edit()
                .putStringSet(KEY_TRAINING_ALLOWED_PACKAGES, trainingAllowed.toSet())
                .apply()
        }

        prefs.edit().putStringSet(KEY_ALLOWED_PACKAGES, current.toSet()).apply()
        updateExcludedPackageSnapshot(packageName, isExcluded = !isNowAllowed)
    }

    /**
     * Only launchable applications are listed. This is sufficient for user
     * selection and avoids broad installed-package inventory access.
     */
    suspend fun getInstalledApps(): List<InstalledAppItem> = withContext(Dispatchers.IO) {
        val packageManager = context.packageManager
        val launcherIntent = Intent(Intent.ACTION_MAIN).apply {
            addCategory(Intent.CATEGORY_LAUNCHER)
        }

        try {
            val launchers = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                packageManager.queryIntentActivities(
                    launcherIntent,
                    PackageManager.ResolveInfoFlags.of(0)
                )
            } else {
                @Suppress("DEPRECATION")
                packageManager.queryIntentActivities(launcherIntent, 0)
            }

            val protectedPackages = launchers.mapNotNull { resolveInfo ->
                val activityInfo = resolveInfo.activityInfo ?: return@mapNotNull null
                val packageName = activityInfo.packageName
                val appName = resolveInfo.loadLabel(packageManager).toString()
                packageName.takeIf {
                    PrivacyAppHeuristics.isPrivacySensitiveApp(appName, packageName)
                }
            }.toSet()
            prefs.edit()
                .putStringSet(KEY_PROTECTED_PACKAGES, protectedPackages)
                .apply()

            val appsByPackage = linkedMapOf<String, InstalledAppItem>()
            for (resolveInfo in launchers) {
                val activityInfo = resolveInfo.activityInfo ?: continue
                val packageName = activityInfo.packageName
                if (packageName == context.packageName || appsByPackage.containsKey(packageName)) {
                    continue
                }

                val appName = resolveInfo.loadLabel(packageManager)
                    .toString()
                    .trim()
                    .ifBlank { packageName }
                val icon = runCatching { resolveInfo.loadIcon(packageManager) }
                    .getOrNull()
                val isSystemApp = (activityInfo.applicationInfo.flags and
                    android.content.pm.ApplicationInfo.FLAG_SYSTEM) != 0

                appsByPackage[packageName] = InstalledAppItem(
                    appName = appName,
                    packageName = packageName,
                    icon = icon,
                    isExcluded = packageName in protectedPackages || isAppExcluded(packageName),
                    isSystemApp = isSystemApp,
                    isTrainingAllowed = packageName in prefs.getStringSet(
                        KEY_TRAINING_ALLOWED_PACKAGES,
                        emptySet()
                    ).orEmpty()
                )
            }

            val result = appsByPackage.values.sortedWith(
                compareByDescending<InstalledAppItem> { it.isExcluded }
                    .thenBy { it.appName.lowercase() }
            )
            _excludedPackages.value = result
                .asSequence()
                .filter { it.isExcluded }
                .map { it.packageName }
                .toSet()
            result
        } catch (error: Exception) {
            ArgusDiagnosticBridge.error(TAG, "Failed to query launchable applications.", error)
            emptyList()
        }
    }

    private fun updateExcludedPackageSnapshot(
        packageName: String,
        isExcluded: Boolean
    ) {
        _excludedPackages.value = _excludedPackages.value.toMutableSet().also { snapshot ->
            if (isExcluded) {
                snapshot.add(packageName)
            } else {
                snapshot.remove(packageName)
            }
        }
    }
}
