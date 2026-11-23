package com.argus.agi.diagnostics

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.Environment
import android.os.storage.StorageManager
import android.util.Log
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.nio.charset.StandardCharsets
import java.time.Instant

enum class ArgusDiagnosticSeverity {
    WARNING,
    ERROR
}

enum class ArgusDiagnosticSubsystem(val fileName: String) {
    APP("app-errors.log"),
    TRAINING("training-errors.log"),
    VULKAN("vulkan-errors.log"),
    RECOVERY("recovery.log")
}

/**
 * Persists warning/error diagnostics locally and mirrors them to the visible
 * shared-storage .Argus folder only when the user granted All files access.
 *
 * The private copy is authoritative. Failure to write or mirror diagnostics
 * must never interrupt application, training, or recovery behavior.
 */
class ArgusDiagnosticLogger(context: Context) {

    companion object {
        private const val TAG = "ArgusDiagnosticLogger"
        private const val LEGACY_FILE_NAME = "argus_runtime_diagnostics.txt"
        private const val MAX_FILE_BYTES = 512L * 1024L
        private const val MAX_MESSAGE_CHARS = 4_096
        private const val MAX_STACK_TRACE_CHARS = 12_000
        private const val BACKUP_COUNT = 2
    }

    private val appContext = context.applicationContext
    private val lock = Any()
    private val writerScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val privateDirectory = File(appContext.noBackupFilesDir, ".Argus/logs")

    private val appVersion: String by lazy(LazyThreadSafetyMode.PUBLICATION) {
        runCatching {
            val packageInfo = appContext.packageManager.getPackageInfo(
                appContext.packageName,
                PackageManager.PackageInfoFlags.of(0)
            )
            "${packageInfo.versionName ?: "unknown"}+${packageInfo.longVersionCode}"
        }.getOrDefault("unknown")
    }

    fun initialize() {
        runCatching {
            synchronized(lock) {
                ensureDirectory(privateDirectory)
                migrateLegacyDiagnosticsLocked()
            }
        }.onFailure { error ->
            Log.w(TAG, "Could not initialize persistent diagnostic storage.", error)
        }
    }

    fun hasSharedStorageAccess(): Boolean =
        runCatching { Environment.isExternalStorageManager() }.getOrDefault(false)

    fun isSharedDirectoryReady(): Boolean =
        hasSharedStorageAccess() && sharedDirectory().isDirectory

    fun sharedDirectoryPath(): String =
        sharedDirectory().absolutePath

    fun privateDirectoryPath(): String =
        privateDirectory.absolutePath

    /**
     * Called after returning from Android Special app access settings.
     * It creates the root-level folder and copies existing private logs.
     */
    fun synchronizeSharedStorage(): Boolean {
        if (!hasSharedStorageAccess()) return false

        return runCatching {
            synchronized(lock) {
                synchronizeSharedStorageLocked()
            }
        }.onFailure { error ->
            Log.w(TAG, "Could not synchronize logs to shared storage.", error)
        }.getOrDefault(false)
    }

    fun record(
        subsystem: ArgusDiagnosticSubsystem,
        severity: ArgusDiagnosticSeverity,
        message: String,
        throwable: Throwable? = null,
        immediate: Boolean = false
    ) {
        val entry = createEntry(subsystem, severity, message, throwable)
        val privateWriteAction: () -> Unit = {
            synchronized(lock) {
                ensureDirectory(privateDirectory)
                appendEntryLocked(File(privateDirectory, subsystem.fileName), entry)
            }
        }
        val writeAction: () -> Unit = {
            privateWriteAction()
            if (hasSharedStorageAccess()) {
                synchronized(lock) {
                    ensureDirectory(sharedDirectory())
                    appendEntryLocked(File(sharedDirectory(), subsystem.fileName), entry)
                }
            }
        }

        if (immediate) {
            // Never block an uncaught-exception handler on removable/shared storage.
            // The next normal app startup will mirror the durable private record.
            runCatching(privateWriteAction).onFailure { error ->
                Log.e(TAG, "Could not persist an immediate diagnostic entry.", error)
            }
        } else {
            writerScope.launch {
                runCatching(writeAction).onFailure { error ->
                    Log.w(TAG, "Could not persist a diagnostic entry.", error)
                }
            }
        }
    }

    fun readRecent(
        subsystem: ArgusDiagnosticSubsystem,
        maximumCharacters: Int = 12_000
    ): String? = runCatching {
        synchronized(lock) {
            val privateFile = File(privateDirectory, subsystem.fileName)
            val source = when {
                privateFile.isFile -> privateFile
                hasSharedStorageAccess() -> File(sharedDirectory(), subsystem.fileName)
                else -> null
            } ?: return@synchronized null

            if (!source.isFile) return@synchronized null
            source.readText(StandardCharsets.UTF_8).takeLast(maximumCharacters)
        }
    }.getOrNull()

    fun close() {
        writerScope.cancel()
    }

    private fun synchronizeSharedStorageLocked(): Boolean {
        if (!hasSharedStorageAccess()) return false

        val destinationDirectory = sharedDirectory()
        ensureDirectory(destinationDirectory)
        val logFiles = privateDirectory.listFiles()
            .orEmpty()
            .filter { file ->
                file.isFile && (
                    file.name.endsWith(".log") ||
                        file.name.endsWith(".log.1") ||
                        file.name.endsWith(".log.2")
                    )
            }

        logFiles.forEach { source ->
            source.copyTo(File(destinationDirectory, source.name), overwrite = true)
        }
        return destinationDirectory.isDirectory
    }

    private fun sharedDirectory(): File {
        val storageManager = appContext.getSystemService(StorageManager::class.java)
        val primaryRoot = storageManager?.primaryStorageVolume?.directory
            ?: File("/storage/emulated/0")
        return File(File(primaryRoot, ".Argus"), "logs")
    }

    private fun ensureDirectory(directory: File) {
        if (directory.isDirectory) return
        if (!directory.mkdirs() && !directory.isDirectory) {
            throw IOException("Could not create the diagnostic directory.")
        }
    }

    private fun migrateLegacyDiagnosticsLocked() {
        val legacyFile = File(appContext.noBackupFilesDir, LEGACY_FILE_NAME)
        val destination = File(privateDirectory, ArgusDiagnosticSubsystem.APP.fileName)
        if (!legacyFile.isFile || destination.isFile) return

        val legacyContent = runCatching {
            legacyFile.readText(StandardCharsets.UTF_8).takeLast(MAX_MESSAGE_CHARS * 4)
        }.getOrDefault("")
        if (legacyContent.isBlank()) return

        val entry = createEntry(
            subsystem = ArgusDiagnosticSubsystem.APP,
            severity = ArgusDiagnosticSeverity.WARNING,
            message = "Diagnostics migrated from the previous runtime file:\n$legacyContent",
            throwable = null
        )
        appendEntryLocked(destination, entry)
    }

    private fun appendEntryLocked(file: File, entry: String) {
        val parent = file.parentFile
            ?: throw IOException("Diagnostic file has no parent directory.")
        ensureDirectory(parent)

        val bytes = (entry + "\n").toByteArray(StandardCharsets.UTF_8)
        if (file.isFile && file.length() + bytes.size > MAX_FILE_BYTES) {
            rotateFileLocked(file)
        }

        FileOutputStream(file, true).use { output ->
            output.write(bytes)
            output.flush()
            output.fd.sync()
        }
    }

    private fun rotateFileLocked(file: File) {
        val oldestBackup = File(file.parentFile, "${file.name}.$BACKUP_COUNT")
        oldestBackup.delete()

        for (index in BACKUP_COUNT - 1 downTo 1) {
            val current = File(file.parentFile, "${file.name}.$index")
            if (current.isFile) {
                val next = File(file.parentFile, "${file.name}.${index + 1}")
                if (!current.renameTo(next)) {
                    current.copyTo(next, overwrite = true)
                    current.delete()
                }
            }
        }

        val firstBackup = File(file.parentFile, "${file.name}.1")
        if (!file.renameTo(firstBackup)) {
            file.copyTo(firstBackup, overwrite = true)
            file.delete()
        }
    }

    private fun createEntry(
        subsystem: ArgusDiagnosticSubsystem,
        severity: ArgusDiagnosticSeverity,
        message: String,
        throwable: Throwable?
    ): String {
        val sanitizedMessage = sanitize(message).take(MAX_MESSAGE_CHARS)
        val sanitizedStack = throwable
            ?.let(::stackTraceFrames)
            .orEmpty()
            .take(MAX_STACK_TRACE_CHARS)
        val errorType = throwable?.javaClass?.name.orEmpty()

        return buildString {
            append('{')
            append("\"timestamp\":").append(json(Instant.now().toString())).append(',')
            append("\"severity\":").append(json(severity.name)).append(',')
            append("\"subsystem\":").append(json(subsystem.name.lowercase())).append(',')
            append("\"appVersion\":").append(json(appVersion)).append(',')
            append("\"androidApi\":").append(Build.VERSION.SDK_INT).append(',')
            append("\"device\":").append(json(sanitize("${Build.MANUFACTURER} ${Build.MODEL}"))).append(',')
            append("\"thread\":").append(json(sanitize(Thread.currentThread().name))).append(',')
            append("\"message\":").append(json(sanitizedMessage)).append(',')
            append("\"errorType\":").append(json(errorType)).append(',')
            append("\"stackTrace\":").append(json(sanitizedStack))
            append('}')
        }
    }

    private fun stackTraceFrames(throwable: Throwable): String = buildString {
        val visited = HashSet<Throwable>()
        var current: Throwable? = throwable
        var causeDepth = 0
        while (current != null && visited.add(current) && causeDepth < 8) {
            appendLine(current.javaClass.name)
            current.stackTrace.take(40).forEach { frame ->
                append("  at ")
                    .append(frame.className)
                    .append('.')
                    .append(frame.methodName)
                    .append('(')
                    .append(frame.fileName ?: "Unknown Source")
                    .append(':')
                    .append(frame.lineNumber)
                    .append(')')
                    .append('\n')
            }
            current = current.cause
            if (current != null) appendLine("Caused by:")
            causeDepth++
        }
    }

    private fun sanitize(value: String): String {
        var sanitized = value
            .replace(Regex("(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|secret)\\s*[:=]\\s*[^\\s,;]+")) {
                "${it.groupValues[1]}=[REDACTED]"
            }
            .replace(Regex("(?i)Bearer\\s+[^\\s,;]+"), "Bearer [REDACTED]")
            .replace(
                Regex("(?i)(content|text|prompt|dataset|query|input|body|payload)\\s*[:=]\\s*([^\\r\\n]+)")
            ) { match ->
                "${match.groupValues[1]}=[REDACTED]"
            }
            .replace(Regex("'[^']{16,}'"), "'[REDACTED]'")
            .replace(Regex("\"[^\"]{16,}\""), "\"[REDACTED]\"")
            .replace(
                Regex("/(?:storage/emulated/0|data/user/0|data/data)/[^\\s\"']+"),
                "[PRIVATE_PATH]"
            )
        sanitized = sanitized.filter { character ->
            character == '\n' || character == '\r' || character == '\t' || character >= ' '
        }
        return sanitized
    }

    private fun json(value: String): String = buildString(value.length + 2) {
        append('"')
        value.forEach { character ->
            when (character) {
                '"' -> append("\\\"")
                '\\' -> append("\\\\")
                '\n' -> append("\\n")
                '\r' -> append("\\r")
                '\t' -> append("\\t")
                else -> if (character < ' ') {
                    append(String.format("\\u%04x", character.code))
                } else {
                    append(character)
                }
            }
        }
        append('"')
    }
}
