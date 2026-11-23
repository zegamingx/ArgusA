package com.argus.agi.diagnostics

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.junit.After
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ArgusDiagnosticLoggerTest {

    private lateinit var context: Context
    private lateinit var logger: ArgusDiagnosticLogger
    private lateinit var privateLogDirectory: File

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        privateLogDirectory = File(context.noBackupFilesDir, ".Argus/logs")
        privateLogDirectory.deleteRecursively()
        logger = ArgusDiagnosticLogger(context)
        logger.initialize()
    }

    @After
    fun tearDown() {
        logger.close()
        privateLogDirectory.deleteRecursively()
    }

    @Test
    fun createsPrivateFallbackWithoutSharedStoragePermission() {
        logger.record(
            subsystem = ArgusDiagnosticSubsystem.TRAINING,
            severity = ArgusDiagnosticSeverity.ERROR,
            message = "Training worker failed to recover its durable queue.",
            immediate = true
        )

        val logFile = File(privateLogDirectory, ArgusDiagnosticSubsystem.TRAINING.fileName)

        assertTrue(privateLogDirectory.isDirectory)
        assertTrue(logFile.isFile)
        assertTrue(logFile.readText().contains("Training worker failed to recover"))
    }

    @Test
    fun redactsCredentialsAndPrivateStoragePaths() {
        logger.record(
            subsystem = ArgusDiagnosticSubsystem.APP,
            severity = ArgusDiagnosticSeverity.WARNING,
            message = "api_key=abc123 Bearer token-value path=/storage/emulated/0/private/corpus.json",
            immediate = true
        )

        val logFile = File(privateLogDirectory, ArgusDiagnosticSubsystem.APP.fileName)
        val savedLog = logFile.readText()

        assertFalse(savedLog.contains("abc123"))
        assertFalse(savedLog.contains("token-value"))
        assertFalse(savedLog.contains("/storage/emulated/0/private"))
        assertTrue(savedLog.contains("[REDACTED]"))
        assertTrue(savedLog.contains("[PRIVATE_PATH]"))
    }

    @Test
    fun rotatesLargeLogFilesAndKeepsCurrentFileBounded() {
        val largeMessage = "diagnostic-event-".repeat(240)

        repeat(180) {
            logger.record(
                subsystem = ArgusDiagnosticSubsystem.VULKAN,
                severity = ArgusDiagnosticSeverity.ERROR,
                message = largeMessage,
                immediate = true
            )
        }

        val activeFile = File(privateLogDirectory, ArgusDiagnosticSubsystem.VULKAN.fileName)
        val firstBackup = File(privateLogDirectory, "${ArgusDiagnosticSubsystem.VULKAN.fileName}.1")

        assertTrue(activeFile.isFile)
        assertTrue(firstBackup.isFile)
        assertTrue(activeFile.length() <= 512L * 1024L)
    }

    @Test
    fun readsRecentSubsystemEntriesFromPrivateStorage() {
        logger.record(
            subsystem = ArgusDiagnosticSubsystem.RECOVERY,
            severity = ArgusDiagnosticSeverity.WARNING,
            message = "Vulkan recovery completed using the CPU fallback.",
            immediate = true
        )

        val report = logger.readRecent(ArgusDiagnosticSubsystem.RECOVERY)

        assertTrue(report.orEmpty().contains("recovery"))
        assertTrue(report.orEmpty().contains("Vulkan recovery completed"))
    }
}
