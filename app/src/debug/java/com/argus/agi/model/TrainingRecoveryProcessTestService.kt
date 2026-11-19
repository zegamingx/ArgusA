package com.argus.agi.model

import android.app.Service
import android.content.Context
import android.content.ContextWrapper
import android.content.Intent
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Message
import android.os.Messenger
import android.os.Process
import android.util.Log
import androidx.room.Room
import com.argus.agi.data.local.ArgusDatabase
import com.argus.agi.data.repository.ArgusRepository
import com.argus.agi.sensor.DeviceGuard
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeout
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream

object TrainingRecoveryProcessTestProtocol {
    const val SERVICE_CLASS_NAME =
        "com.argus.agi.model.TrainingRecoveryProcessTestService"

    const val MESSAGE_CRASH_AT_BOUNDARY = 1
    const val MESSAGE_RECOVER_AFTER_RESTART = 2

    const val EXTRA_CASE_ID = "case_id"
    const val EXTRA_BOUNDARY = "boundary"
    const val EXTRA_EXPECTED_STATUS = "expected_status"

    const val CONTROL_DIRECTORY = "training_process_recovery_control"
    const val TEST_DATA_DIRECTORY = "training_process_recovery"
    const val DATABASE_NAME = "argus_process_recovery.db"

    fun itemId(caseId: String): String = "training-recovery-" + caseId

    fun crashMarker(context: Context, caseId: String): File =
        File(controlDirectory(context), caseId + ".crashed.json")

    fun recoveryMarker(context: Context, caseId: String): File =
        File(controlDirectory(context), caseId + ".recovered.json")

    fun errorMarker(context: Context, caseId: String): File =
        File(controlDirectory(context), caseId + ".error")

    private fun controlDirectory(context: Context): File =
        File(context.applicationContext.filesDir, CONTROL_DIRECTORY)
}

class TrainingRecoveryProcessTestService : Service() {

    private lateinit var serviceScope: CoroutineScope
    private lateinit var commandMessenger: Messenger

    override fun onCreate() {
        super.onCreate()
        serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        commandMessenger = Messenger(
            object : Handler(Looper.getMainLooper()) {
                override fun handleMessage(message: Message) {
                    val data = message.data
                    val caseId = data.getString(
                        TrainingRecoveryProcessTestProtocol.EXTRA_CASE_ID
                    ).orEmpty()
                    if (caseId.isBlank()) return

                    when (message.what) {
                        TrainingRecoveryProcessTestProtocol.MESSAGE_CRASH_AT_BOUNDARY -> {
                            val boundary = data.getString(
                                TrainingRecoveryProcessTestProtocol.EXTRA_BOUNDARY
                            ).orEmpty()
                            serviceScope.launch {
                                crashAtBoundary(caseId, boundary)
                            }
                        }

                        TrainingRecoveryProcessTestProtocol.MESSAGE_RECOVER_AFTER_RESTART -> {
                            val expectedStatus = data.getString(
                                TrainingRecoveryProcessTestProtocol.EXTRA_EXPECTED_STATUS
                            ).orEmpty()
                            serviceScope.launch {
                                recoverAfterRestart(caseId, expectedStatus)
                            }
                        }
                    }
                }
            }
        )
    }

    override fun onBind(intent: Intent?): IBinder = commandMessenger.binder

    override fun onDestroy() {
        serviceScope.cancel()
        super.onDestroy()
    }

    private suspend fun crashAtBoundary(caseId: String, boundaryName: String) {
        var database: ArgusDatabase? = null
        var manager: TrainingManager? = null
        try {
            val isolated = isolatedContext(caseId)
            database = openDatabase(isolated)
            val repository = ArgusRepository(isolated, database)
            val engine = createEngine(isolated)
            check(engine.saveCheckpoint()) {
                "The process-recovery baseline checkpoint could not be saved."
            }

            val activeManager = TrainingManager(
                context = isolated,
                neuralEngine = engine,
                deviceGuard = DeviceGuard(isolated),
                worldModel = WorldModel(isolated, repository),
                repository = repository,
                onCommitBoundary = { reachedBoundary ->
                    if (reachedBoundary.name == boundaryName) {
                        val journalFile = File(
                            isolated.filesDir,
                            TrainingCommitJournal.FILE_NAME
                        )
                        val snapshot = TrainingCommitJournal.read(journalFile)
                        val marker = JSONObject()
                            .put("caseId", caseId)
                            .put("boundary", reachedBoundary.name)
                            .put("pid", Process.myPid())
                            .put("itemId", queuedItem.id)
                            .put(
                                "baselineOptimizerStep",
                                snapshot?.baselineOptimizerStep ?: engine.optimizerStep
                            )
                            .put("optimizerStepAtBoundary", engine.optimizerStep)
                            .put("journalExists", journalFile.isFile)
                            .put("journalIsCommitted", snapshot?.isCommitted ?: false)
                            .put(
                                "journalCommittedOptimizerStep",
                                snapshot?.committedOptimizerStep ?: -1
                            )
                            .put(
                                "journalCheckpointSha256Fingerprint",
                                snapshot?.checkpointSha256Fingerprint ?: JSONObject.NULL
                            )
                            .put(
                                "checkpointSha256Fingerprint",
                                engine.checkpointFileSha256Fingerprint()
                            )

                        writeJson(
                            TrainingRecoveryProcessTestProtocol.crashMarker(
                                applicationContext,
                                caseId
                            ),
                            marker
                        )

                        // Deliberately terminate the remote Android process. No
                        // TrainingManager finally block or Room close can run.
                        Process.killProcess(Process.myPid())
                    }
                }
            )
            manager = activeManager
            activeManager.pauseTraining()
            val itemTitle = "Cold-start recovery boundary " + boundaryName
            activeManager.enqueueData(
                source = "process-death-recovery-test",
                title = itemTitle,
                content = "The recovery test must preserve exactly one durable optimizer transaction after process death."
            )
            activeManager.awaitQueuePersistenceIdle()

            val queuedItem = withTimeout(20_000L) {
                while (activeManager.nextQueue.value.none { it.title == itemTitle }) {
                    delay(25L)
                }
                activeManager.nextQueue.value.single { it.title == itemTitle }
            }
            activeManager.triggerImmediateTrainingStep()

            // If the requested boundary is never observed, leave a durable failure
            // marker for the instrumentation process rather than reporting a pass.
            delay(60_000L)
            throw IllegalStateException(
                "Training did not reach the requested process-death boundary: " + boundaryName
            )
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Throwable) {
            writeFailure(caseId, error)
            runCatching { manager?.close() }
            runCatching { database?.close() }
            Process.killProcess(Process.myPid())
        }
    }

    private suspend fun recoverAfterRestart(caseId: String, expectedStatusName: String) {
        var database: ArgusDatabase? = null
        var manager: TrainingManager? = null
        try {
            val expectedStatus = TrainingStatus.valueOf(expectedStatusName)
            val isolated = isolatedContext(caseId)
            database = openDatabase(isolated)
            val repository = ArgusRepository(isolated, database)
            val engine = createEngine(isolated)
            val crashMarker = TrainingRecoveryProcessTestProtocol.crashMarker(
                applicationContext,
                caseId
            )
            check(crashMarker.isFile) {
                "The preceding process did not leave its durable crash marker."
            }
            val itemId = JSONObject(crashMarker.readText()).getString("itemId")
            val journalFile = File(isolated.filesDir, TrainingCommitJournal.FILE_NAME)

            val activeManager = TrainingManager(
                context = isolated,
                neuralEngine = engine,
                deviceGuard = DeviceGuard(isolated),
                worldModel = WorldModel(isolated, repository),
                repository = repository
            )
            manager = activeManager
            activeManager.pauseTraining()

            var recoveredItem: TrainingData? = null
            withTimeout(45_000L) {
                while (recoveredItem == null) {
                    val candidate = repository.getTrainingItem(itemId)
                        ?: throw IllegalStateException(
                            "The training item disappeared during cold-start recovery."
                        )
                    val journal = TrainingCommitJournal.read(journalFile)
                    val queueContainsItem = activeManager.nextQueue.value.any {
                        it.id == itemId
                    }
                    val checkpointFingerprint = engine.checkpointFileSha256Fingerprint()

                    val ready = when (expectedStatus) {
                        TrainingStatus.QUEUED -> candidate.status == TrainingStatus.QUEUED &&
                            journal == null && !journalFile.exists() && queueContainsItem

                        TrainingStatus.OPTIMIZER_COMMITTED ->
                            candidate.status == TrainingStatus.OPTIMIZER_COMMITTED &&
                                journal?.isCommitted == true &&
                                !journal.checkpointSha256Fingerprint.isNullOrBlank() &&
                                journal.checkpointSha256Fingerprint == checkpointFingerprint &&
                                queueContainsItem

                        TrainingStatus.COMPLETED -> candidate.status == TrainingStatus.COMPLETED &&
                            journal == null && !journalFile.exists() && !queueContainsItem

                        else -> false
                    }

                    if (ready) recoveredItem = candidate else delay(25L)
                }
            }

            val finalItem = requireNotNull(recoveredItem)
            val finalJournal = TrainingCommitJournal.read(journalFile)
            val checkpointFingerprint = engine.checkpointFileSha256Fingerprint()
            val queueContainsItem = activeManager.nextQueue.value.any {
                it.id == itemId
            }
            val probe = JSONObject()
                .put("caseId", caseId)
                .put("pid", Process.myPid())
                .put("itemStatus", finalItem.status.name)
                .put("optimizerStep", engine.optimizerStep)
                .put("queueContainsItem", queueContainsItem)
                .put("journalExists", journalFile.isFile)
                .put("journalIsCommitted", finalJournal?.isCommitted ?: false)
                .put(
                    "journalCommittedOptimizerStep",
                    finalJournal?.committedOptimizerStep ?: -1
                )
                .put(
                    "journalCheckpointSha256Fingerprint",
                    finalJournal?.checkpointSha256Fingerprint ?: JSONObject.NULL
                )
                .put("checkpointSha256Fingerprint", checkpointFingerprint)

            activeManager.close()
            manager = null
            database.close()
            database = null

            writeJson(
                TrainingRecoveryProcessTestProtocol.recoveryMarker(
                    applicationContext,
                    caseId
                ),
                probe
            )
            // Finish the probe in this remote process so the next case cannot
            // reuse static Room, engine, Vulkan or coroutine state.
            Process.killProcess(Process.myPid())
        } catch (cancelled: CancellationException) {
            throw cancelled
        } catch (error: Throwable) {
            writeFailure(caseId, error)
            runCatching { manager?.close() }
            runCatching { database?.close() }
            Process.killProcess(Process.myPid())
        }
    }

    private fun isolatedContext(caseId: String): Context {
        require(caseId.matches(Regex("[a-z0-9_-]{1,48}"))) {
            "The process-recovery case identifier is invalid."
        }
        return TrainingRecoveryIsolatedContext(applicationContext, caseId)
    }

    private fun openDatabase(context: Context): ArgusDatabase =
        Room.databaseBuilder(
            context,
            ArgusDatabase::class.java,
            TrainingRecoveryProcessTestProtocol.DATABASE_NAME
        )
            .addMigrations(
                ArgusDatabase.MIGRATION_1_2,
                ArgusDatabase.MIGRATION_2_3,
                ArgusDatabase.MIGRATION_3_4
            )
            .build()

    private fun createEngine(context: Context): ArgusNeuralEngine =
        ArgusNeuralEngine.createBlocking(
            context,
            EngineConfig(
                dModel = 16,
                nHeads = 4,
                dFf = 32,
                maxSeqLen = 32,
                loraRank = 2,
                tokenizerConfig = TokenizerConfig(
                    maxVocabularySize = 512,
                    maxTokenLength = 32
                )
            )
        )

    private fun writeFailure(caseId: String, error: Throwable) {
        val marker = JSONObject()
            .put("caseId", caseId)
            .put("pid", Process.myPid())
            .put("errorType", error.javaClass.name)
            .put("message", error.message ?: "Unknown process-recovery test failure.")
        runCatching {
            writeJson(
                TrainingRecoveryProcessTestProtocol.errorMarker(
                    applicationContext,
                    caseId
                ),
                marker
            )
        }
        Log.e(TAG, "Training process-recovery test failed for " + caseId, error)
    }

    private fun writeJson(file: File, value: JSONObject) {
        val parent = file.parentFile
            ?: throw IllegalStateException(
                "A process-recovery marker has no parent directory."
            )
        check(parent.isDirectory || parent.mkdirs()) {
            "The process-recovery marker directory could not be created."
        }
        FileOutputStream(file, false).use { stream ->
            stream.write(value.toString().toByteArray(Charsets.UTF_8))
            stream.flush()
            stream.fd.sync()
        }
    }

    private class TrainingRecoveryIsolatedContext(
        base: Context,
        caseId: String
    ) : ContextWrapper(base.applicationContext) {

        private val rootDirectory = File(
            File(
                base.applicationContext.filesDir,
                TrainingRecoveryProcessTestProtocol.TEST_DATA_DIRECTORY
            ),
            caseId
        )

        override fun getApplicationContext(): Context = this

        override fun getFilesDir(): File = ensureDirectory("files")

        override fun getCacheDir(): File = ensureDirectory("cache")

        override fun getDatabasePath(name: String): File =
            File(ensureDirectory("databases"), name)

        private fun ensureDirectory(name: String): File {
            val directory = File(rootDirectory, name)
            check(directory.isDirectory || directory.mkdirs()) {
                "The isolated process-recovery directory could not be created."
            }
            return directory
        }
    }

    private companion object {
        const val TAG = "ArgusRecoveryProcessTest"
    }
}
