package com.argus.agi.model

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import java.io.File

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ArgusNeuralEngineCancellationTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun cancellingLongPrefillStopsGenerationPromptly() = runBlocking {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            checkpoint.delete()
            val engine = ArgusNeuralEngine.createBlocking(
                context,
                EngineConfig(
                    dModel = 64,
                    nHeads = 4,
                    dFf = 256,
                    maxSeqLen = 1_024,
                    loraRank = 4,
                    tokenizerConfig = TokenizerConfig(
                        maxVocabularySize = 512,
                        maxTokenLength = 32
                    )
                )
            )
            val prompt = buildString {
                repeat(600) { index ->
                    append("argus-context-")
                    append(index)
                    append(' ')
                }
            }

            val generationJob = CoroutineScope(Dispatchers.Default).launch {
                engine.generate(prompt, maxNewTokens = 512)
            }

            // The 1,024-token prefill is deliberately large enough to keep this
            // CPU-only inference active while the test requests cancellation.
            delay(20)
            assertTrue(
                "Generation completed before cancellation could be exercised.",
                generationJob.isActive
            )

            withTimeout(2_000L) {
                generationJob.cancelAndJoin()
            }
            assertTrue("Generation job must finish in the cancelled state.", generationJob.isCancelled)
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    private fun restoreCheckpoint(checkpoint: File, previousCheckpoint: ByteArray?) {
        if (previousCheckpoint == null) {
            checkpoint.delete()
        } else {
            checkpoint.writeBytes(previousCheckpoint)
        }
        File(context.filesDir, "$CHECKPOINT_FILE_NAME.tmp").delete()
    }

    private companion object {
        const val CHECKPOINT_FILE_NAME = "argus_model_weights.bin"
    }
}
