package com.argus.agi.diagnostics

import android.util.Log
import com.argus.agi.ArgusApp

/**
 * Connects existing Android warning/error call sites to persistent diagnostics
 * without making logging failures affect the original operation.
 */
object ArgusDiagnosticBridge {

    fun error(
        tag: String,
        message: String,
        throwable: Throwable? = null
    ) {
        Log.e(tag, message, throwable)
        persist(
            tag = tag,
            message = message,
            severity = ArgusDiagnosticSeverity.ERROR,
            throwable = throwable
        )
    }

    fun warning(
        tag: String,
        message: String,
        throwable: Throwable? = null
    ) {
        Log.w(tag, message, throwable)
        persist(
            tag = tag,
            message = message,
            severity = ArgusDiagnosticSeverity.WARNING,
            throwable = throwable
        )
    }

    private fun persist(
        tag: String,
        message: String,
        severity: ArgusDiagnosticSeverity,
        throwable: Throwable?
    ) {
        val subsystem = when (tag) {
            "VulkanTrainingBackend" -> ArgusDiagnosticSubsystem.VULKAN
            "TrainingManager",
            "GenesisModelManager",
            "ArgusNeuralEngine" -> ArgusDiagnosticSubsystem.TRAINING
            else -> ArgusDiagnosticSubsystem.APP
        }

        runCatching {
            ArgusApp.instance.recordDiagnostic(
                message = message,
                severity = severity,
                subsystem = subsystem,
                throwable = throwable
            )
        }
    }
}
