package com.argus.agi.model

import android.app.Application
import android.content.Context
import android.content.SharedPreferences
import androidx.test.core.app.ApplicationProvider
import com.argus.agi.sensor.DeviceGuard
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34], application = Application::class)
class GenesisModelManagerTest {

    @Test
    fun storedIdentityIsRejectedWhenNoCompatibleCheckpointWasRestored() {
        val context: Context = ApplicationProvider.getApplicationContext()
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()
        val preferences = context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)
        val previousPreferences = preferences.all

        try {
            checkpoint.delete()
            val committed = preferences.edit()
                .clear()
                .putString(KEY_ORIGIN_ID, "GENESIS-STALE")
                .putLong(KEY_BIRTH_TS, 1_700_000_000_000L)
                .putInt(KEY_RAM_GB, 8)
                .putString(KEY_COMPUTE, "Kotlin CPU backend")
                .putLong(KEY_PARAM_COUNT, 10_000L)
                .putString(KEY_ARCH_TIER, "Test model")
                .commit()
            assertTrue("The stale test identity must be persisted.", committed)

            val engine = ArgusNeuralEngine.createBlocking(
                context,
                EngineConfig(
                    dModel = 16,
                    nHeads = 4,
                    dFf = 32,
                    maxSeqLen = 16,
                    loraRank = 2,
                    tokenizerConfig = TokenizerConfig(
                        maxVocabularySize = 512,
                        maxTokenLength = 32
                    )
                )
            )
            assertFalse(engine.isCheckpointStateVerified)

            val manager = GenesisModelManager(
                context = context,
                neuralEngine = engine,
                deviceGuard = DeviceGuard(context),
                stageMinDelayMs = 0L
            )

            assertFalse(manager.hasCompletedGenesis())
            assertNull(manager.getIdentity())
            assertFalse("Stale origin IDs must be removed, not merely hidden.", preferences.contains(KEY_ORIGIN_ID))
            assertEquals(GenesisLifecycleStage.IDLE, manager.lifecycleState.value.stage)

            // A later successful save of freshly initialized weights must not revive
            // the identity belonging to the checkpoint that failed restoration.
            assertTrue(engine.saveCheckpoint())
            assertTrue(engine.isCheckpointStateVerified)
            assertFalse(manager.hasCompletedGenesis())
            assertNull(manager.getIdentity())
        } finally {
            restorePreferences(preferences, previousPreferences)
            if (previousCheckpoint == null) {
                checkpoint.delete()
            } else {
                checkpoint.writeBytes(previousCheckpoint)
            }
        }
    }

    private fun restorePreferences(
        preferences: SharedPreferences,
        values: Map<String, *>
    ) {
        val editor = preferences.edit().clear()
        values.forEach { (key, value) ->
            when (value) {
                is String -> editor.putString(key, value)
                is Boolean -> editor.putBoolean(key, value)
                is Int -> editor.putInt(key, value)
                is Long -> editor.putLong(key, value)
                is Float -> editor.putFloat(key, value)
                is Set<*> -> editor.putStringSet(key, value.filterIsInstance<String>().toSet())
            }
        }
        check(editor.commit()) {
            "Previous Genesis preferences could not be restored after the test."
        }
    }

    private companion object {
        const val CHECKPOINT_FILE_NAME = "argus_model_weights.bin"
        const val PREFERENCES_NAME = "argus_genesis_prefs"
        const val KEY_ORIGIN_ID = "genesis_origin_id"
        const val KEY_BIRTH_TS = "genesis_birth_timestamp"
        const val KEY_RAM_GB = "genesis_ram_gb"
        const val KEY_COMPUTE = "genesis_compute_target"
        const val KEY_PARAM_COUNT = "genesis_param_count"
        const val KEY_ARCH_TIER = "genesis_arch_tier"
    }
}
