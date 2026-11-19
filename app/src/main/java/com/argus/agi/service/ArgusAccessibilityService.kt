package com.argus.agi.service

import android.accessibilityservice.AccessibilityService
import android.content.Intent
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.text.InputType
import android.util.Log
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import com.argus.agi.ArgusApp
import com.argus.agi.sensor.ExcludedAppsManager
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

@Suppress("DEPRECATION")
class ArgusAccessibilityService : AccessibilityService() {

    companion object {
        private const val TAG = "ArgusAccessibility"

        const val THROTTLE_MS = 2500L
        const val MIN_TEXT_LENGTH = 20
        const val MAX_TRAVERSAL_NODES = 350
        const val MAX_TRAVERSAL_DEPTH = 25

        @Volatile
        var instance: ArgusAccessibilityService? = null
            private set

        private val mainHandler = Handler(Looper.getMainLooper())

        /**
         * Performs a user-requested click only when the user separately enabled
         * accessibility actions. Screen-reading consent does not imply action consent.
         */
        fun performClickAction(query: String): Boolean {
            val service = instance ?: return false
            var result = false

            if (Looper.myLooper() == Looper.getMainLooper()) {
                result = service.internalPerformClick(query)
            } else {
                val latch = CountDownLatch(1)
                mainHandler.post {
                    try {
                        result = service.internalPerformClick(query)
                    } finally {
                        latch.countDown()
                    }
                }
                latch.await(1000, TimeUnit.MILLISECONDS)
            }
            return result
        }
    }

    private lateinit var excludedAppsManager: ExcludedAppsManager
    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val isTraversing = AtomicBoolean(false)

    @Volatile
    private var currentScreenPackage: String? = null

    private var lastScreenTextHash: Int? = null
    private var lastPerceptionTime = 0L

    override fun onCreate() {
        super.onCreate()
        excludedAppsManager = ExcludedAppsManager(this)
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        Log.i(TAG, "ArgusAccessibilityService connected. Screen reading remains opt-in and default-off.")
    }

    override fun onUnbind(intent: Intent?): Boolean {
        if (instance == this) {
            instance = null
        }
        invalidateScreenContext()
        return super.onUnbind(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
        if (instance == this) {
            instance = null
        }
        invalidateScreenContext()
        serviceScope.cancel()
        Log.i(TAG, "ArgusAccessibilityService destroyed.")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        if (event == null) {
            invalidateScreenContext()
            return
        }
        if (!excludedAppsManager.isScreenPerceptionEnabled()) {
            invalidateScreenContext()
            return
        }

        val eventType = event.eventType
        if (eventType != AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED &&
            eventType != AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED
        ) {
            return
        }

        val eventPackage = event.packageName?.toString()?.takeIf { it.isNotBlank() }
        if (eventPackage == null) {
            invalidateScreenContext()
            return
        }

        if (currentScreenPackage != eventPackage) {
            invalidateScreenContext(eventPackage)
        }
        if (isInputMethodPackage(eventPackage) ||
            excludedAppsManager.isAppExcluded(eventPackage)
        ) {
            invalidateScreenContext(eventPackage)
            return
        }

        val now = SystemClock.elapsedRealtime()
        if (now - lastPerceptionTime < THROTTLE_MS) return
        lastPerceptionTime = now

        if (!isTraversing.compareAndSet(false, true)) return

        try {
            val rootNode = rootInActiveWindow ?: return
            try {
                val rootPackage = rootNode.packageName?.toString()
                    ?.takeIf { it.isNotBlank() }
                    ?: eventPackage
                if (rootPackage != eventPackage || excludedAppsManager.isAppExcluded(rootPackage)) {
                    invalidateScreenContext()
                    return
                }

                val gatheredText = StringBuilder()
                val visitedCount = IntArray(1)
                traverseNodes(rootNode, gatheredText, visitedCount, depth = 0)

                val fullText = gatheredText.toString().trim()
                if (fullText.length < MIN_TEXT_LENGTH) return

                val currentHash = fullText.hashCode()
                if (currentHash == lastScreenTextHash) return
                lastScreenTextHash = currentHash

                serviceScope.launch {
                    val worldModel = ArgusApp.worldModel
                    try {
                        // A delayed job from a previous app must not publish after the
                        // active package changed or its screen-reading grant was revoked.
                        if (currentScreenPackage != rootPackage ||
                            !excludedAppsManager.isScreenPerceptionEnabled() ||
                            excludedAppsManager.isAppExcluded(rootPackage)
                        ) {
                            worldModel.clearScreenContext(rootPackage)
                            return@launch
                        }

                        worldModel.updateScreenContext(rootPackage, fullText)

                        // Package transitions can race this asynchronous dispatch. Clear
                        // only this package's context so a newer allowed app is untouched.
                        if (currentScreenPackage != rootPackage ||
                            !excludedAppsManager.isScreenPerceptionEnabled() ||
                            excludedAppsManager.isAppExcluded(rootPackage)
                        ) {
                            worldModel.clearScreenContext(rootPackage)
                            return@launch
                        }

                        if (currentScreenPackage == rootPackage &&
                            excludedAppsManager.isAppTrainingAllowed(rootPackage)
                        ) {
                            ArgusApp.trainingManager.enqueueScreenData(
                                packageName = rootPackage,
                                title = "Layar: ${rootPackage.substringAfterLast('.')}",
                                content = fullText.take(500)
                            )
                        }
                    } catch (error: Exception) {
                        worldModel.clearScreenContext(rootPackage)
                        Log.e(TAG, "Screen perception dispatch failed.", error)
                    }
                }
            } finally {
                if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
                    rootNode.recycle()
                }
            }
        } finally {
            isTraversing.set(false)
        }
    }

    override fun onInterrupt() {
        invalidateScreenContext()
    }

    private fun invalidateScreenContext(nextPackage: String? = null) {
        currentScreenPackage = nextPackage
        lastScreenTextHash = null
        ArgusApp.worldModel.clearScreenContext()
    }

    private fun traverseNodes(
        node: AccessibilityNodeInfo?,
        output: StringBuilder,
        visitedCount: IntArray,
        depth: Int
    ) {
        if (node == null || depth > MAX_TRAVERSAL_DEPTH ||
            visitedCount[0] >= MAX_TRAVERSAL_NODES
        ) {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
                node?.recycle()
            }
            return
        }

        try {
            visitedCount[0]++
            if (isSensitiveNode(node)) return

            val text = node.text?.toString()?.trim()
            val description = node.contentDescription?.toString()?.trim()
            if (!text.isNullOrBlank()) {
                output.append(text).append(' ')
            } else if (!description.isNullOrBlank()) {
                output.append(description).append(' ')
            }

            for (index in 0 until node.childCount) {
                if (visitedCount[0] >= MAX_TRAVERSAL_NODES) break
                traverseNodes(node.getChild(index), output, visitedCount, depth + 1)
            }
        } finally {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
                node.recycle()
            }
        }
    }

    private fun isSensitiveNode(node: AccessibilityNodeInfo): Boolean {
        if (node.isPassword) return true

        val inputType = node.inputType
        val passwordVariation =
            (inputType and InputType.TYPE_TEXT_VARIATION_PASSWORD) != 0 ||
                (inputType and InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD) != 0 ||
                (inputType and InputType.TYPE_NUMBER_VARIATION_PASSWORD) != 0 ||
                (inputType and InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD) != 0
        if (passwordVariation) return true

        val text = node.text?.toString()?.lowercase().orEmpty()
        val description = node.contentDescription?.toString()?.lowercase().orEmpty()
        val hint = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            node.hintText?.toString()?.lowercase().orEmpty()
        } else {
            ""
        }

        val sensitiveKeywords = listOf(
            "password", "kata sandi", "passcode", "pin ", "otp", "cvv",
            "kode verifikasi", "security code", "token rahasia"
        )
        return node.isEditable &&
            sensitiveKeywords.any { keyword ->
                text.contains(keyword) || hint.contains(keyword) || description.contains(keyword)
            }
    }

    private fun isInputMethodPackage(packageName: String): Boolean {
        val lower = packageName.lowercase()
        return lower.contains("inputmethod") ||
            lower.contains("keyboard") ||
            lower.contains("ime") ||
            lower == "com.google.android.inputmethod.latin" ||
            lower == "com.touchtype.swiftkey"
    }

    private fun internalPerformClick(query: String): Boolean {
        val root = rootInActiveWindow ?: return false
        try {
            // A missing active package is not an allowed target. The manager checks
            // independent action consent, the explicit app allowlist, and protected apps.
            if (!excludedAppsManager.isAccessibilityActionAllowed(root.packageName?.toString())) {
                return false
            }

            val nodes = root.findAccessibilityNodeInfosByText(query) ?: return false
            try {
                for (node in nodes) {
                    val exactMatch =
                        node.text?.toString()?.trim()?.equals(query.trim(), ignoreCase = true) == true ||
                            node.contentDescription?.toString()?.trim()
                                ?.equals(query.trim(), ignoreCase = true) == true

                    if (!exactMatch || !node.isVisibleToUser || !node.isEnabled) continue
                    if (node.isClickable) {
                        return node.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                    }

                    var parent: AccessibilityNodeInfo? = node.parent
                    while (parent != null) {
                        try {
                            if (parent.isClickable && parent.isEnabled) {
                                return parent.performAction(AccessibilityNodeInfo.ACTION_CLICK)
                            }
                            val nextParent = parent.parent
                            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
                                parent.recycle()
                            }
                            parent = nextParent
                        } catch (_: Exception) {
                            break
                        }
                    }
                }
            } finally {
                if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
                    nodes.forEach { node ->
                        runCatching { node.recycle() }
                    }
                }
            }
        } finally {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
                root.recycle()
            }
        }
        return false
    }
}
