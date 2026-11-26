package com.argus.agi.model

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import com.argus.agi.backend.ComputeBackend
import com.argus.agi.backend.ComputeBackendKind
import com.argus.agi.device.DeviceProfile
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ArgusNeuralEngineComputeBackendTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun lifecycleAndCheckpointSynchronizationUseTheInjectedBackend() {
        val checkpoint = File(context.filesDir, CHECKPOINT_FILE_NAME)
        val temporaryCheckpoint = File(context.filesDir, "$CHECKPOINT_FILE_NAME.tmp")
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()
        val previousTemporaryCheckpoint = temporaryCheckpoint.takeIf(File::isFile)?.readBytes()

        try {
            checkpoint.delete()
            temporaryCheckpoint.delete()

            val backend = RecordingComputeBackend()
            val engine = ArgusNeuralEngine.createBlockingWithBackend(
                context = context,
                config = EngineConfig(
                    dModel = 64,
                    nHeads = 4,
                    dFf = 256,
                    maxSeqLen = 64,
                    loraRank = 4,
                    tokenizerConfig = TokenizerConfig(
                        maxVocabularySize = 512,
                        maxTokenLength = 32
                    )
                ),
                computeBackend = backend
            )

            assertFalse(engine.isVulkanRuntimeReady)
            assertTrue(engine.initializeOptionalVulkanRuntime())
            assertEquals(1, backend.initializeCalls)
            assertTrue(engine.isVulkanRuntimeReady)

            backend.synchronizeSucceeds = false
            assertFalse(engine.saveCheckpoint())
            assertEquals(1, backend.synchronizeCalls)
            assertFalse(engine.isCheckpointStateVerified)

            backend.synchronizeSucceeds = true
            assertTrue(engine.saveCheckpoint())
            assertEquals(2, backend.synchronizeCalls)
            assertTrue(engine.isCheckpointStateVerified)
            assertTrue(checkpoint.isFile)

            backend.synchronizeSucceeds = false
            assertTrue(engine.loadCheckpointIfExists())
            assertEquals(1, backend.checkpointRestoreCalls)
            // A verified checkpoint replaces stale GPU mirrors and is sufficient
            // to permit a subsequent checkpoint without another GPU readback.
            assertTrue(engine.saveCheckpoint())
            assertEquals(3, backend.synchronizeCalls)

            assertTrue(engine.releaseVulkanTensorResidencyForMemoryPressure())
            assertEquals(1, backend.releaseCalls)

            engine.resumeVulkanTensorResidencyAfterForeground()
            assertEquals(1, backend.resumeCalls)

            engine.resetOptionalVulkanRuntimeForRetry()
            assertEquals(1, backend.resetCalls)
            assertEquals(3, backend.synchronizeCalls)
            assertFalse(engine.isVulkanRuntimeReady)
        } finally {
            restoreFile(checkpoint, previousCheckpoint)
            restoreFile(temporaryCheckpoint, previousTemporaryCheckpoint)
        }
    }

    private fun restoreFile(file: File, previousBytes: ByteArray?) {
        if (previousBytes == null) {
            file.delete()
        } else {
            file.writeBytes(previousBytes)
        }
    }

    private class RecordingComputeBackend : ComputeBackend {

        private val profile = MutableStateFlow<DeviceProfile?>(null)

        override val kind: ComputeBackendKind = ComputeBackendKind.VULKAN
        override var isReady: Boolean = false
            private set
        override val isFp16ComputeReady: Boolean = false
        override val deviceProfile: StateFlow<DeviceProfile?> = profile

        var initializeCalls: Int = 0
            private set
        var resetCalls: Int = 0
            private set
        var synchronizeCalls: Int = 0
            private set
        var releaseCalls: Int = 0
            private set
        var resumeCalls: Int = 0
            private set
        var checkpointRestoreCalls: Int = 0
            private set
        var checkpointRestoreAuthoritative: Boolean = false
        var synchronizeSucceeds: Boolean = true

        override fun initializeAfterModelReady(): Boolean {
            initializeCalls += 1
            isReady = true
            return true
        }

        override fun resetForRetry() {
            resetCalls += 1
            if (!checkpointRestoreAuthoritative && !synchronizeAuthoritativeState()) {
                return
            }
            checkpointRestoreAuthoritative = false
            isReady = false
        }

        override fun synchronizeAuthoritativeState(): Boolean {
            synchronizeCalls += 1
            return checkpointRestoreAuthoritative || synchronizeSucceeds
        }

        override fun markCheckpointRestoredAsAuthoritative() {
            checkpointRestoreCalls += 1
            checkpointRestoreAuthoritative = true
        }

        override fun releaseDeviceResourcesForMemoryPressure(): Boolean {
            releaseCalls += 1
            return true
        }

        override fun resumeDeviceResourcesAfterForeground() {
            resumeCalls += 1
        }
    }

    private companion object {
        const val CHECKPOINT_FILE_NAME = "argus_model_weights.bin"
    }
}
