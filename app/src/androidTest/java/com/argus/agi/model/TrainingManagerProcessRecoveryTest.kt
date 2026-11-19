package com.argus.agi.model

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.IBinder
import android.os.Message
import android.os.Messenger
import android.os.SystemClock
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

@RunWith(AndroidJUnit4::class)
class TrainingManagerProcessRecoveryTest {

    private val context: Context
        get() = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun processDeathAtEveryDurableBoundaryRecoversFromDiskOnANewAndroidProcess() {
        val cases = listOf(
            BoundaryCase(
                id = "prepared",
                boundary = "PREPARED_JOURNAL_DURABLE",
                expectedStatus = "QUEUED",
                expectedCommittedCheckpoint = false,
                expectedJournalCommitted = false,
                expectedJournal = false
            ),
            BoundaryCase(
                id = "optimizer_applied",
                boundary = "OPTIMIZER_UPDATE_APPLIED",
                expectedStatus = "QUEUED",
                expectedCommittedCheckpoint = false,
                expectedJournalCommitted = false,
                expectedJournal = false
            ),
            BoundaryCase(
                id = "committed_journal",
                boundary = "COMMITTED_JOURNAL_DURABLE",
                expectedStatus = "QUEUED",
                expectedCommittedCheckpoint = false,
                expectedJournalCommitted = true,
                expectedJournal = false
            ),
            BoundaryCase(
                id = "checkpoint_replaced",
                boundary = "CHECKPOINT_FILE_REPLACED",
                expectedStatus = "OPTIMIZER_COMMITTED",
                expectedCommittedCheckpoint = true,
                expectedJournalCommitted = true,
                expectedJournal = true
            ),
            BoundaryCase(
                id = "checkpoint_bound",
                boundary = "CHECKPOINT_BOUND_JOURNAL_DURABLE",
                expectedStatus = "OPTIMIZER_COMMITTED",
                expectedCommittedCheckpoint = true,
                expectedJournalCommitted = true,
                expectedJournal = true
            ),
            BoundaryCase(
                id = "room_committed",
                boundary = "ROOM_OPTIMIZER_STATE_COMMITTED",
                expectedStatus = "OPTIMIZER_COMMITTED",
                expectedCommittedCheckpoint = true,
                expectedJournalCommitted = true,
                expectedJournal = true
            ),
            BoundaryCase(
                id = "queue_finalized",
                boundary = "QUEUE_ITEM_FINALIZED",
                expectedStatus = "COMPLETED",
                expectedCommittedCheckpoint = true,
                expectedJournalCommitted = true,
                expectedJournal = false
            )
        )

        for (case in cases) {
            runBoundaryScenario(case)
        }
    }

    private fun runBoundaryScenario(case: BoundaryCase) {
        val controlDirectory = File(
            context.filesDir,
            TrainingRecoveryProcessTestProtocol.CONTROL_DIRECTORY
        )
        check(controlDirectory.isDirectory || controlDirectory.mkdirs()) {
            "Could not create the process-recovery control directory."
        }

        val crashMarker = TrainingRecoveryProcessTestProtocol.crashMarker(context, case.id)
        val recoveryMarker = TrainingRecoveryProcessTestProtocol.recoveryMarker(context, case.id)
        val errorMarker = TrainingRecoveryProcessTestProtocol.errorMarker(context, case.id)
        crashMarker.delete()
        recoveryMarker.delete()
        errorMarker.delete()

        val isolatedRoot = File(
            File(context.filesDir, TrainingRecoveryProcessTestProtocol.TEST_DATA_DIRECTORY),
            case.id
        )
        check(!isolatedRoot.exists() || isolatedRoot.deleteRecursively()) {
            "The prior isolated test data could not be removed for " + case.id
        }

        val crashService = bindRecoveryService()
        val crash: JSONObject
        try {
            sendCommand(
                crashService.messenger,
                TrainingRecoveryProcessTestProtocol.MESSAGE_CRASH_AT_BOUNDARY,
                case.id,
                case.boundary
            )
            crash = awaitMarker(crashMarker, errorMarker, 90_000L)
            assertEquals(case.id, crash.getString("caseId"))
            assertEquals(case.boundary, crash.getString("boundary"))
            assertTrue(
                "The crash point must leave its durable transaction journal on disk.",
                crash.getBoolean("journalExists")
            )
            assertEquals(
                "The journal state at the kill point must match the selected transaction boundary.",
                case.expectedJournalCommitted,
                crash.getBoolean("journalIsCommitted")
            )
            assertTrue(
                "The crash marker must identify its remote Android process.",
                crash.getInt("pid") > 0
            )
            assertTrue(
                "The worker process did not terminate after the requested boundary.",
                crashService.disconnected.await(15L, TimeUnit.SECONDS)
            )
        } finally {
            context.unbindService(crashService.connection)
        }

        val recoveryService = bindRecoveryService()
        try {
            sendCommand(
                recoveryService.messenger,
                TrainingRecoveryProcessTestProtocol.MESSAGE_RECOVER_AFTER_RESTART,
                case.id,
                case.expectedStatus
            )
            val recovery = awaitMarker(recoveryMarker, errorMarker, 60_000L)
            assertEquals(case.id, recovery.getString("caseId"))
            assertNotEquals(
                "Cold-start recovery must execute in a different Android process.",
                crash.getInt("pid"),
                recovery.getInt("pid")
            )
            assertEquals(case.expectedStatus, recovery.getString("itemStatus"))
            assertEquals(case.expectedJournal, recovery.getBoolean("journalExists"))
            assertTrue(recovery.getInt("optimizerStep") >= 0)
            assertTrue(recovery.getString("checkpointSha256Fingerprint").isNotBlank())

            val optimizerStepAtBoundary = crash.getInt("optimizerStepAtBoundary")
            val baselineOptimizerStep = crash.getInt("baselineOptimizerStep")
            when (case.boundary) {
                "PREPARED_JOURNAL_DURABLE" -> assertEquals(
                    "PREPARED must be durable before any optimizer update.",
                    baselineOptimizerStep,
                    optimizerStepAtBoundary
                )

                "OPTIMIZER_UPDATE_APPLIED",
                "COMMITTED_JOURNAL_DURABLE" -> assertTrue(
                    "The optimizer update must already be applied at this boundary.",
                    optimizerStepAtBoundary > baselineOptimizerStep
                )

                else -> assertEquals(
                    "The checkpoint/Room/finalization boundary must match the committed optimizer step.",
                    crash.getInt("journalCommittedOptimizerStep"),
                    optimizerStepAtBoundary
                )
            }

            if (!case.expectedCommittedCheckpoint || case.expectedStatus == "COMPLETED") {
                assertEquals(
                    "A rollback or already-finalized transaction must retain the exact checkpoint bytes.",
                    crash.getString("checkpointSha256Fingerprint"),
                    recovery.getString("checkpointSha256Fingerprint")
                )
            }

            if (case.expectedCommittedCheckpoint) {
                val committedStep = crash.getInt("journalCommittedOptimizerStep")
                assertTrue(
                    "The crash point must contain a committed optimizer journal step.",
                    committedStep >= crash.getInt("baselineOptimizerStep")
                )
                assertEquals(committedStep, recovery.getInt("optimizerStep"))
            } else {
                assertEquals(
                    "A pre-checkpoint crash must restore the exact baseline optimizer step.",
                    crash.getInt("baselineOptimizerStep"),
                    recovery.getInt("optimizerStep")
                )
            }

            if (case.expectedJournal) {
                assertEquals(
                    recovery.getString("checkpointSha256Fingerprint"),
                    recovery.getString("journalCheckpointSha256Fingerprint")
                )
                assertEquals(
                    crash.getInt("journalCommittedOptimizerStep"),
                    recovery.getInt("journalCommittedOptimizerStep")
                )
                assertTrue(recovery.getBoolean("queueContainsItem"))
            } else {
                assertEquals(
                    JSONObject.NULL,
                    recovery.opt("journalCheckpointSha256Fingerprint")
                )
                if (case.expectedStatus == "QUEUED") {
                    assertTrue(recovery.getBoolean("queueContainsItem"))
                } else {
                    assertFalse(recovery.getBoolean("queueContainsItem"))
                    assertEquals(
                        "The finalized item must not apply its optimizer update again.",
                        crash.getInt("journalCommittedOptimizerStep"),
                        recovery.getInt("optimizerStep")
                    )
                }
            }

            assertTrue(
                "The recovery probe process did not terminate after writing its marker.",
                recoveryService.disconnected.await(15L, TimeUnit.SECONDS)
            )
        } finally {
            context.unbindService(recoveryService.connection)
        }

        check(!isolatedRoot.exists() || isolatedRoot.deleteRecursively()) {
            "The isolated process-recovery data could not be cleaned for " + case.id
        }
        crashMarker.delete()
        recoveryMarker.delete()
        errorMarker.delete()
    }

    private fun bindRecoveryService(): BoundService {
        val connected = CountDownLatch(1)
        val disconnected = CountDownLatch(1)
        val messengerReference = AtomicReference<Messenger>()
        val connection = object : ServiceConnection {
            override fun onServiceConnected(name: ComponentName, service: IBinder) {
                messengerReference.set(Messenger(service))
                connected.countDown()
            }

            override fun onServiceDisconnected(name: ComponentName) {
                disconnected.countDown()
            }

            override fun onBindingDied(name: ComponentName) {
                disconnected.countDown()
            }

            override fun onNullBinding(name: ComponentName) {
                disconnected.countDown()
            }
        }
        val intent = Intent().setClassName(
            context,
            TrainingRecoveryProcessTestProtocol.SERVICE_CLASS_NAME
        )
        assertTrue(
            "The process-recovery service could not be bound.",
            context.bindService(intent, connection, Context.BIND_AUTO_CREATE)
        )
        assertTrue(
            "The process-recovery service did not connect.",
            connected.await(15L, TimeUnit.SECONDS)
        )
        return BoundService(
            messenger = requireNotNull(messengerReference.get()),
            connection = connection,
            disconnected = disconnected
        )
    }

    private fun sendCommand(
        messenger: Messenger,
        messageCode: Int,
        caseId: String,
        value: String
    ) {
        val message = Message.obtain(null, messageCode)
        message.data = Bundle().apply {
            putString(TrainingRecoveryProcessTestProtocol.EXTRA_CASE_ID, caseId)
            when (messageCode) {
                TrainingRecoveryProcessTestProtocol.MESSAGE_CRASH_AT_BOUNDARY ->
                    putString(TrainingRecoveryProcessTestProtocol.EXTRA_BOUNDARY, value)

                TrainingRecoveryProcessTestProtocol.MESSAGE_RECOVER_AFTER_RESTART ->
                    putString(TrainingRecoveryProcessTestProtocol.EXTRA_EXPECTED_STATUS, value)
            }
        }
        messenger.send(message)
    }

    private fun awaitMarker(
        marker: File,
        error: File,
        timeoutMs: Long
    ): JSONObject {
        val deadline = SystemClock.elapsedRealtime() + timeoutMs
        while (SystemClock.elapsedRealtime() < deadline) {
            if (error.isFile) {
                throw AssertionError(
                    "Remote process-recovery service failed: " + error.readText()
                )
            }
            if (marker.isFile) return JSONObject(marker.readText())
            SystemClock.sleep(50L)
        }
        throw AssertionError(
            "Timed out waiting for process-recovery marker: " + marker.name
        )
    }

    private data class BoundaryCase(
        val id: String,
        val boundary: String,
        val expectedStatus: String,
        val expectedCommittedCheckpoint: Boolean,
        val expectedJournalCommitted: Boolean,
        val expectedJournal: Boolean
    )

    private data class BoundService(
        val messenger: Messenger,
        val connection: ServiceConnection,
        val disconnected: CountDownLatch
    )
}
