package com.argus.agi.sensor

import android.content.Context
import android.content.Intent
import android.content.pm.ApplicationInfo
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
    val isSystemApp: Boolean = false
)

class ExcludedAppsManager(private val context: Context) {

    companion object {
        private const val TAG = "ExcludedAppsManager"
    }

    private val prefs = context.getSharedPreferences("argus_excluded_apps", Context.MODE_PRIVATE)
    private val _excludedPackages = MutableStateFlow<Set<String>>(emptySet())
    val excludedPackages: StateFlow<Set<String>> = _excludedPackages.asStateFlow()

    init {
        val saved = prefs.getStringSet("excluded_set", null)
        if (saved == null) {
            // Seed sensitive package patterns by default
            val defaults = setOf(
                "com.google.android.apps.authenticator2",
                "com.whatsapp",
                "org.telegram.messenger",
                "com.bca",
                "id.co.bri.brimo",
                "com.bankmandiri.livin",
                "id.bca.mybca",
                "com.telkom.indonesia.digitallife",
                "com.binance.dev",
                "com.tokocrypto.app"
            )
            prefs.edit().putStringSet("excluded_set", defaults).apply()
            _excludedPackages.value = defaults
        } else {
            _excludedPackages.value = saved
        }
    }

    fun isAppExcluded(packageName: String): Boolean {
        if (_excludedPackages.value.contains(packageName)) return true
        val lower = packageName.lowercase()
        return lower.contains("authenticator") ||
                lower.contains("password") ||
                lower.contains("bitwarden") ||
                lower.contains("1password") ||
                lower.contains("keepass") ||
                lower.contains("autofill") ||
                lower.contains("keychain") ||
                lower.contains("bank")
    }

    fun toggleAppExclusion(packageName: String) {
        val current = _excludedPackages.value.toMutableSet()
        if (current.contains(packageName)) {
            current.remove(packageName)
        } else {
            current.add(packageName)
        }
        prefs.edit().putStringSet("excluded_set", current).apply()
        _excludedPackages.value = current
    }

    /**
     * Discovers ALL applications installed on device by combining multiple package manager queries:
     * 1. Querying all launcher intent activities (guarantees every app the user can open on home screen).
     * 2. Querying all installed applications (including user utilities, social media, banking, and services).
     * 3. Querying installed packages.
     * With QUERY_ALL_PACKAGES declared in AndroidManifest, this enumerates 100% of real device apps.
     */
    suspend fun getInstalledApps(): List<InstalledAppItem> = withContext(Dispatchers.IO) {
        val pm = context.packageManager
        val appMap = mutableMapOf<String, InstalledAppItem>()
        val currentExcluded = _excludedPackages.value

        // Strategy 1: All launchable apps (Every user-facing app on the phone)
        try {
            val launcherIntent = Intent(Intent.ACTION_MAIN, null).apply {
                addCategory(Intent.CATEGORY_LAUNCHER)
            }
            val launcherApps = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                pm.queryIntentActivities(launcherIntent, PackageManager.ResolveInfoFlags.of(0))
            } else {
                @Suppress("DEPRECATION")
                pm.queryIntentActivities(launcherIntent, 0)
            }
            for (resolveInfo in launcherApps) {
                val pkg = resolveInfo.activityInfo.packageName
                if (pkg == context.packageName) continue // Skip ARGUS itself
                val name = resolveInfo.loadLabel(pm).toString().trim().ifBlank { pkg }
                val icon = try { resolveInfo.loadIcon(pm) } catch (_: Exception) { null }
                val isExcluded = currentExcluded.contains(pkg) || isAppExcluded(pkg)
                val isSys = (resolveInfo.activityInfo.applicationInfo.flags and ApplicationInfo.FLAG_SYSTEM) != 0
                appMap[pkg] = InstalledAppItem(
                    appName = name,
                    packageName = pkg,
                    icon = icon,
                    isExcluded = isExcluded,
                    isSystemApp = isSys
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to query launcher activities: ${e.message}")
        }

        // Strategy 2: Query all installed applications via getInstalledApplications
        try {
            val allApps = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                pm.getInstalledApplications(PackageManager.ApplicationInfoFlags.of(0))
            } else {
                @Suppress("DEPRECATION")
                pm.getInstalledApplications(0)
            }
            for (appInfo in allApps) {
                val pkg = appInfo.packageName
                if (pkg == context.packageName) continue // Skip ARGUS
                if (!appMap.containsKey(pkg)) {
                    val isSys = (appInfo.flags and ApplicationInfo.FLAG_SYSTEM) != 0
                    val launchIntent = pm.getLaunchIntentForPackage(pkg)
                    // If not system app OR has a launch intent OR matches a known pattern, include it
                    val name = try { pm.getApplicationLabel(appInfo).toString().trim().ifBlank { pkg } } catch (_: Exception) { pkg }
                    val icon = try { pm.getApplicationIcon(appInfo) } catch (_: Exception) { null }
                    val isExcluded = currentExcluded.contains(pkg) || isAppExcluded(pkg)

                    appMap[pkg] = InstalledAppItem(
                        appName = name,
                        packageName = pkg,
                        icon = icon,
                        isExcluded = isExcluded,
                        isSystemApp = isSys
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to query installed applications: ${e.message}")
        }

        // Strategy 3: Query getInstalledPackages as a comprehensive fallback
        try {
            val allPackages = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                pm.getInstalledPackages(PackageManager.PackageInfoFlags.of(0))
            } else {
                @Suppress("DEPRECATION")
                pm.getInstalledPackages(0)
            }
            for (pkgInfo in allPackages) {
                val pkg = pkgInfo.packageName
                if (pkg == context.packageName) continue
                if (!appMap.containsKey(pkg)) {
                    val appInfo = pkgInfo.applicationInfo ?: continue
                    val name = try { pm.getApplicationLabel(appInfo).toString().trim().ifBlank { pkg } } catch (_: Exception) { pkg }
                    val icon = try { pm.getApplicationIcon(appInfo) } catch (_: Exception) { null }
                    val isExcluded = currentExcluded.contains(pkg) || isAppExcluded(pkg)
                    val isSys = (appInfo.flags and ApplicationInfo.FLAG_SYSTEM) != 0

                    appMap[pkg] = InstalledAppItem(
                        appName = name,
                        packageName = pkg,
                        icon = icon,
                        isExcluded = isExcluded,
                        isSystemApp = isSys
                    )
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to query installed packages: ${e.message}")
        }

        // Sort: Protected (isExcluded) first, then alphabetical by App Name
        appMap.values.sortedWith(
            compareByDescending<InstalledAppItem> { it.isExcluded }
                .thenBy { it.appName.lowercase() }
        )
    }
}
