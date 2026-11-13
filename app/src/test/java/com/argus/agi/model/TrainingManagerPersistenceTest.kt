package com.argus.agi.model

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.argus.agi.data.local.ArgusDatabase
import com.argus.agi.data.repository.ArgusRepository
import com.argus.agi.sensor.DeviceGuard
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File
import java.util.concurrent.Executors

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class TrainingManagerPersistenceTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun failedStartPersistenceRestoresItemWithoutTraining() = runBlocking {
        withTrainingManager { database, repository, manager ->
            manager.pauseTraining()
            manager.enqueueData(
                source = "persistence-test",
                title = "Start persistence failure",
                content = "A short queue item used to verify durable training state."
            )
            awaitPersistence(manager)

            val queuedItem = manager.nextQueue.value.single()
            assertEquals(TrainingStatus.QUEUED, queuedItem.status)
            assertEquals(queuedItem.id, repository.getPendingQueueSnapshot().single().id)

            database.close()

            val nextItem = withTimeout(5_000L) {
                manager.popNext()
            }
            assertNull(
                "A Room start-state failure must resolve instead of leaving popNext suspended.",
                nextItem
            )
            assertEquals(listOf(queuedItem.id), manager.nextQueue.value.map(TrainingData::id))
            assertEquals(TrainingStatus.QUEUED, manager.nextQueue.value.single().status)
            assertEquals(0, manager.totalStepsExecuted.value)
        }
    }

    @Test
    fun failedSavePersistenceDoesNotLeavePhantomQueuedItem() = runBlocking {
        withTrainingManager { database, _, manager ->
            manager.pauseTraining()
            database.close()

            manager.enqueueData(
                source = "persistence-test",
                title = "Save persistence failure",
                content = "A short queue item whose Room insert is expected to fail."
            )
            awaitPersistence(manager)

            assertTrue(manager.nextQueue.value.isEmpty())
            assertEquals(1, manager.completedQueue.value.size)
            assertEquals(TrainingStatus.FAILED, manager.completedQueue.value.single().status)
            assertEquals(0, manager.totalStepsExecuted.value)
        }
    }

    private suspend fun awaitPersistence(manager: TrainingManager) {
        withTimeout(5_000L) {
            manager.awaitQueuePersistenceIdle()
        }
    }

    private suspend fun withTrainingManager(
        test: suspend (
            ArgusDatabase,
            ArgusRepository,
            TrainingManager
        ) -> Unit
    ) {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            ArgusDatabase::class.java
        ).build()
        val repository = ArgusRepository(context, database)
        val trainingDispatcher = Executors.newSingleThreadExecutor { runnable ->
            Thread(runnable, "ArgusTrainingManagerPersistenceTest").apply {
                isDaemon = true
            }
        }.asCoroutineDispatcher()
        var manager: TrainingManager? = null

        try {
            checkpoint.delete()
            val engine = ArgusNeuralEngine.createBlocking(
                context,
                EngineConfig(
                    dModel = 16,
                    nHeads = 4,
                    dFf = 32,
                    maxSeqLen = 32,
                    loraRank = 2,
                    tokenizerConfig = TokenizerConfig(
                        maxVocabularySize = 512,
                        maxTokenLength = 16
                    )
                )
            )
            val deviceGuard = DeviceGuard(context)
            val worldModel = WorldModel(context, repository)
            val activeManager = TrainingManager(
                context = context,
                neuralEngine = engine,
                deviceGuard = deviceGuard,
                worldModel = worldModel,
                trainingDispatcher = trainingDispatcher,
                repository = repository
            )
            manager = activeManager
            test(database, repository, activeManager)
        } finally {
            try {
                manager?.close()
            } finally {
                runCatching { database.close() }
                trainingDispatcher.close()
            }
            if (previousCheckpoint == null) {
                checkpoint.delete()
            } else {
                checkpoint.writeBytes(previousCheckpoint)
            }
            File(context.filesDir, "$CHECKPOINT_FILE_NAME.tmp").delete()
        }
    }

    private companion object {
        const val CHECKPOINT_FILE_NAME = "argus_model_weights.bin"
    }
}
