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
         * Safely performs a click on a visible, enabled node matching the target query.
         * Dispatches safely to the main thread and properly recycles nodes on API < 33.
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

    private var lastScreenTextHash = 0
    private var lastPerceptionTime = 0L

    override fun onCreate() {
        super.onCreate()
        excludedAppsManager = ExcludedAppsManager(this)
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        Log.i(TAG, "ArgusAccessibilityService connected and ready.")
    }

    override fun onUnbind(intent: Intent?): Boolean {
        if (instance == this) {
            instance = null
        }
        return super.onUnbind(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
        if (instance == this) {
            instance = null
        }
        serviceScope.cancel()
        Log.i(TAG, "ArgusAccessibilityService destroyed.")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        if (event == null) return

        // 1. EVENT TYPE GATING:
        // Strictly ignore keystroke events (TYPE_VIEW_TEXT_CHANGED) to prevent keylogging.
        // Only process high-level window state or significant content changes.
        val eventType = event.eventType
        if (eventType != AccessibilityEvent.TYPE_WINDOW_STATE_CHANGED &&
            eventType != AccessibilityEvent.TYPE_WINDOW_CONTENT_CHANGED
        ) {
            return
        }

        val eventPkg = event.packageName?.toString() ?: return

        // 2. PRIVACY & EXCLUSION CHECK (Event Package):
        // Reject excluded apps and keyboard / IME packages from triggering window captures.
        if (isInputMethodPackage(eventPkg) || excludedAppsManager.isAppExcluded(eventPkg)) {
            return
        }

        // 3. SEPARATED THROTTLE:
        // Update throttle timestamp immediately to avoid redundant full-tree walks on static screens.
        val now = SystemClock.elapsedRealtime()
        if (now - lastPerceptionTime < THROTTLE_MS) {
            return
        }
        lastPerceptionTime = now

        // 4. REENTRANCY GUARD:
        if (!isTraversing.compareAndSet(false, true)) {
            return
        }

        try {
            // Retrieve root node of the active window
            val rootNode = rootInActiveWindow ?: return

            try {
                // 5. PRIVACY & EXCLUSION CHECK (Root Active Window Package):
                // Guarantees that system UI or overlay events cannot leak content from an excluded background app.
                val rootPkg = rootNode.packageName?.toString()
                if (rootPkg != null && excludedAppsManager.isAppExcluded(rootPkg)) {
                    Log.d(TAG, "Skipping perception: Active root window '$rootPkg' is excluded.")
                    return
                }

                // 6. TRAVERSE ACCESSIBILITY TREE WITH DEPTH CAP & SENSITIVE FIELD FILTERING
                val gatheredText = StringBuilder()
                val visitedCount = IntArray(1) { 0 }
                traverseNodes(rootNode, gatheredText, visitedCount, depth = 0)

                val fullText = gatheredText.toString().trim()
                if (fullText.length >= MIN_TEXT_LENGTH) {
                    val currentHash = fullText.hashCode()
                    if (currentHash != lastScreenTextHash) {
                        lastScreenTextHash = currentHash
                        val targetPkg = rootPkg ?: eventPkg

                        // Offload model dispatch & training ingestion to background coroutine
                        serviceScope.launch {
                            try {
                                ArgusApp.worldModel.updateScreenContext(targetPkg, fullText)
                                ArgusApp.trainingManager.enqueueData(
                                    source = "Screen ($targetPkg)",
                                    title = "Layar: ${targetPkg.substringAfterLast('.')}",
                                    content = fullText.take(500)
                                )
                            } catch (e: Exception) {
                                Log.e(TAG, "Error passing perception to WorldModel/TrainingManager: ${e.message}", e)
                            }
                        }
                    }
                }
            } finally {
                // Prevent native memory leak on API < 33
                if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
                    rootNode.recycle()
                }
            }
        } finally {
            isTraversing.set(false)
        }
    }

    override fun onInterrupt() {}

    /**
     * Recursively traverses nodes while:
     * 1. Skipping password fields, OTPs, and sensitive input fields.
     * 2. Enforcing a maximum node cap (MAX_TRAVERSAL_NODES) to avoid ANRs.
     * 3. Enforcing a maximum depth cap (MAX_TRAVERSAL_DEPTH).
     * 4. Recycling all intermediate AccessibilityNodeInfo objects on API < 33.
     */
    private fun traverseNodes(
        node: AccessibilityNodeInfo?,
        sb: StringBuilder,
        visitedCount: IntArray,
        depth: Int
    ) {
        if (node == null || depth > MAX_TRAVERSAL_DEPTH || visitedCount[0] >= MAX_TRAVERSAL_NODES) {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
                node?.recycle()
            }
            return
        }

        try {
            visitedCount[0]++

            // SENSITIVE INPUT FILTER: Never ingest passwords, PINs, or secret credentials
            if (isSensitiveNode(node)) {
                return
            }

            val text = node.text?.toString()?.trim()
            val desc = node.contentDescription?.toString()?.trim()

            if (!text.isNullOrBlank()) {
                sb.append(text).append(" ")
            } else if (!desc.isNullOrBlank()) {
                sb.append(desc).append(" ")
            }

            val childCount = node.childCount
            for (i in 0 until childCount) {
                if (visitedCount[0] >= MAX_TRAVERSAL_NODES) break
                val child = node.getChild(i)
                traverseNodes(child, sb, visitedCount, depth + 1)
            }
        } finally {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
                node.recycle()
            }
        }
    }

    /**
     * Detects if an AccessibilityNode represents password, PIN, OTP, or sensitive inputs.
     */
    private fun isSensitiveNode(node: AccessibilityNodeInfo): Boolean {
        if (node.isPassword) return true

        val inputType = node.inputType
        val isPasswordVariation = (inputType and InputType.TYPE_TEXT_VARIATION_PASSWORD) != 0 ||
                (inputType and InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD) != 0 ||
                (inputType and InputType.TYPE_NUMBER_VARIATION_PASSWORD) != 0 ||
                (inputType and InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD) != 0

        if (isPasswordVariation) return true

        val text = node.text?.toString()?.lowercase() ?: ""
        val desc = node.contentDescription?.toString()?.lowercase() ?: ""
        val hint = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            node.hintText?.toString()?.lowercase() ?: ""
        } else ""

        val sensitiveKeywords = listOf(
            "password", "kata sandi", "passcode", "pin ", "otp", "cvv",
            "kode verifikasi", "security code", "token rahasia"
        )

        if (node.isEditable && sensitiveKeywords.any { text.contains(it) || hint.contains(it) || desc.contains(it) }) {
            return true
        }

        return false
    }

    private fun isInputMethodPackage(pkg: String): Boolean {
        val lower = pkg.lowercase()
        return lower.contains("inputmethod") ||
                lower.contains("keyboard") ||
                lower.contains("ime") ||
                lower == "com.google.android.inputmethod.latin" ||
                lower == "com.touchtype.swiftkey"
    }

    private fun internalPerformClick(query: String): Boolean {
        val root = rootInActiveWindow ?: return false
        try {
            val nodes = root.findAccessibilityNodeInfosByText(query) ?: return false
            try {
                for (node in nodes) {
                    val textMatch = node.text?.toString()?.trim()?.equals(query.trim(), ignoreCase = true) == true ||
                            node.contentDescription?.toString()?.trim()?.equals(query.trim(), ignoreCase = true) == true

                    if (textMatch && node.isVisibleToUser && node.isEnabled) {
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
                }
            } finally {
                if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
                    for (n in nodes) {
                        try { n.recycle() } catch (_: Exception) {}
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
