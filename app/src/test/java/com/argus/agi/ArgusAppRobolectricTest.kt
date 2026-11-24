package com.argus.agi

import android.content.Context
import android.content.pm.ServiceInfo
import androidx.test.core.app.ApplicationProvider
import com.argus.agi.model.ArgusArchitectureConfig
import com.argus.agi.model.ArgusArchitectureParameterEstimator
import com.argus.agi.model.ArgusDeviceModelTier
import com.argus.agi.model.ArgusNeuralEngine
import com.argus.agi.model.EngineConfig
import com.argus.agi.model.ArgusTokenizerAlgorithm
import com.argus.agi.model.Tokenizer
import com.argus.agi.model.TokenizerConfig
import com.argus.agi.model.TrainingManager
import com.argus.agi.model.WorldModel
import com.argus.agi.sensor.DeviceGuard
import com.argus.agi.service.ArgusForegroundService
import com.argus.agi.service.ForegroundTrainingPresentation
import java.io.StringReader
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ArgusAppRobolectricTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @org.junit.Before
    fun setUp() = kotlinx.coroutines.runBlocking<Unit> {
        com.argus.agi.data.repository.ArgusRepository(context).clearPendingQueue()
    }

    @Test
    fun foregroundTrainingServiceUsesSpecialUseTypeOnly() {
        // minSdk is 34, where specialUse exists; there is no dataSync variant.
        assertEquals(
            ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE,
            ArgusForegroundService.foregroundServiceType
        )
    }

    @Test
    fun foregroundServiceUserStopLatchPersistsAndClears() {
        ArgusForegroundService.clearUserStopped(context)
        assertFalse(ArgusForegroundService.isUserStopped(context))

        ArgusForegroundService.setUserStopped(context)
        assertTrue(ArgusForegroundService.isUserStopped(context))

        ArgusForegroundService.clearUserStopped(context)
        assertFalse(ArgusForegroundService.isUserStopped(context))
    }

    @Test
    fun foregroundNotificationPresentationReflectsTrainingState() {
        assertEquals(
            ForegroundTrainingPresentation.PAUSED,
            ArgusForegroundService.presentationFor(
                isPaused = true,
                hasActiveWork = true,
                hasQueuedWork = true
            )
        )
        assertEquals(
            ForegroundTrainingPresentation.ACTIVE,
            ArgusForegroundService.presentationFor(
                isPaused = false,
                hasActiveWork = true,
                hasQueuedWork = false
            )
        )
        assertEquals(
            ForegroundTrainingPresentation.ACTIVE,
            ArgusForegroundService.presentationFor(
                isPaused = false,
                hasActiveWork = false,
                hasQueuedWork = true
            )
        )
        assertEquals(
            ForegroundTrainingPresentation.IDLE,
            ArgusForegroundService.presentationFor(
                isPaused = false,
                hasActiveWork = false,
                hasQueuedWork = false
            )
        )
    }

    @Test
    fun testDeviceModelConfigurationSelectsFourTiersAndSafeParameterBudgets() {
        val compact = EngineConfig.forDevice(
            totalRamGb = 4,
            maxHeapMb = 192L,
            cpuCores = 4
        )
        assertEquals(ArgusDeviceModelTier.COMPACT, compact.modelTier)
        assertEquals(1, compact.trainingBatchSize)

        val balanced = EngineConfig.forDevice(
            totalRamGb = 6,
            maxHeapMb = 256L,
            cpuCores = 6
        )
        assertEquals(ArgusDeviceModelTier.BALANCED, balanced.modelTier)
        assertEquals(2, balanced.trainingBatchSize)

        val performance = EngineConfig.forDevice(
            totalRamGb = 8,
            maxHeapMb = 384L,
            cpuCores = 8
        )
        assertEquals(ArgusDeviceModelTier.PERFORMANCE, performance.modelTier)
        assertEquals(4, performance.trainingBatchSize)

        val flagship = EngineConfig.forDevice(
            totalRamGb = 12,
            maxHeapMb = 512L,
            cpuCores = 8
        )
        assertEquals(ArgusDeviceModelTier.FLAGSHIP, flagship.modelTier)
        assertEquals(8, flagship.trainingBatchSize)

        val memoryConstrainedFlagship = EngineConfig.forDevice(
            totalRamGb = 12,
            maxHeapMb = 512L,
            availableRamGb = 1.5f,
            cpuCores = 8
        )
        assertEquals(ArgusDeviceModelTier.PERFORMANCE, memoryConstrainedFlagship.modelTier)
        assertEquals(4, memoryConstrainedFlagship.trainingBatchSize)

        listOf(compact, balanced, performance, flagship, memoryConstrainedFlagship).forEach { config ->
            val architecture = ArgusArchitectureConfig(
                layerCount = config.layerCount,
                hiddenSize = config.dModel,
                attentionHeadCount = config.nHeads,
                feedForwardSize = config.dFf,
                vocabularySize = TokenizerConfig.MINIMUM_VOCAB_SIZE,
                contextLength = config.maxSeqLen,
                adapterRank = config.loraRank
            )
            val parameters = ArgusArchitectureParameterEstimator
                .estimate(architecture)
                .totalTrainableParameters
            assertTrue(parameters >= 1_000_000L)
            assertTrue(config.trainingBatchSize in 1..8)
        }
    }

    @Test
    fun preReleaseCheckpointFallsBackToCurrentDeviceAndBpeContract() {
        val checkpoint = java.io.File(context.filesDir, "argus_model_weights.bin")
        val previousBytes = checkpoint.takeIf { it.isFile }?.readBytes()

        try {
            val legacyHeader = ByteBuffer.allocate(48).order(ByteOrder.LITTLE_ENDIAN)
            legacyHeader.put(byteArrayOf(0x41, 0x52, 0x47, 0x33))
            legacyHeader.putInt(3)
            checkpoint.writeBytes(legacyHeader.array())

            val resolved = ArgusNeuralEngine.resolveConfig(
                context = context,
                totalRamGb = 4,
                maxHeapMb = 192L,
                availableRamGb = 4.0f,
                cpuCores = 4
            )

            assertEquals(ArgusDeviceModelTier.COMPACT, resolved.modelTier)
            assertEquals(128, resolved.dModel)
            assertEquals(4, resolved.nHeads)
            assertEquals(384, resolved.dFf)
            assertEquals(64, resolved.maxSeqLen)
            assertEquals(4, resolved.loraRank)
            assertEquals(1_024, resolved.initialVocabularyCapacity)
            assertEquals(
                ArgusTokenizerAlgorithm.BYTE_LEVEL_BPE_V1,
                resolved.tokenizerConfig.algorithm
            )
        } finally {
            if (previousBytes == null) {
                checkpoint.delete()
            } else {
                checkpoint.writeBytes(previousBytes)
            }
        }
    }

    @Test
    fun testTokenizerEncodingDecoding() {
        val tokenizer = Tokenizer()
        val text = "A.R.G.U.S on device neural intelligence"
        val tokens = tokenizer.encode(text, autoRegister = true)
        assertTrue("Tokens should not be empty", tokens.isNotEmpty())
        val decoded = tokenizer.decode(tokens)
        assertEquals(text, decoded)
    }

    @Test
    fun testSpecialTokens() {
        val tokenizer = Tokenizer()
        val text = "${Tokenizer.USER_TOKEN}Hello${Tokenizer.ASSISTANT_TOKEN}"
        val tokens = tokenizer.encode(text, autoRegister = true)
        assertTrue("Tokens should contain user token", tokens.contains(Tokenizer.USER_ID))
        assertTrue("Tokens should contain assistant token", tokens.contains(Tokenizer.ASSISTANT_ID))
    }

    @Test
    fun testTokenizerCollisionRejectionAndMonotonicId() {
        val tokenizer = Tokenizer()
        val id1 = tokenizer.registerToken("alpha")
        val id2 = tokenizer.registerToken("beta")
        assertTrue("Monotonic IDs should be unique", id1 != id2)

        assertEquals(id1, tokenizer.registerToken("alpha"))

        try {
            tokenizer.registerToken("gamma", id1)
            fail("Should throw IllegalArgumentException when registering token with an occupied ID")
        } catch (_: IllegalArgumentException) {
        }
    }

    @Test
    fun testTokenizerImportVocabValidation() {
        val source = Tokenizer()
        val canonicalEntries = source.exportVocab()
        assertEquals(288, canonicalEntries.size)

        val restored = Tokenizer()
        restored.importVocab(canonicalEntries)
        assertEquals(canonicalEntries, restored.exportVocab())
        val sample = "Argus byte-level vocabulary round trip 🎯"
        assertArrayEquals(source.encodeToIds(sample), restored.encodeToIds(sample))

        val duplicateIdEntries = canonicalEntries.mapIndexed { index, entry ->
            if (index == 1) entry.first to canonicalEntries.first().second else entry
        }
        val duplicateIdFailure = runCatching {
            restored.importVocab(duplicateIdEntries)
        }.exceptionOrNull()
        assertTrue(duplicateIdFailure is IllegalArgumentException)
        assertEquals(canonicalEntries, restored.exportVocab())

        val duplicateTokenEntries = canonicalEntries.mapIndexed { index, entry ->
            if (index == 1) canonicalEntries.first().first to entry.second else entry
        }
        val duplicateTokenFailure = runCatching {
            restored.importVocab(duplicateTokenEntries)
        }.exceptionOrNull()
        assertTrue(duplicateTokenFailure is IllegalArgumentException)
        assertEquals(canonicalEntries, restored.exportVocab())
    }

    @Test
    fun testTokenizerEmojiSurrogatePair() {
        val tokenizer = Tokenizer()
        val emojiText = "Target 🎯 dan AGI 🚀"
        val tokens = tokenizer.encode(emojiText, autoRegister = true)
        val decoded = tokenizer.decode(tokens)
        assertEquals("Decoded text should preserve emoji code points cleanly", emojiText, decoded)
    }

    @Test
    fun testTokenizerUnicodeFallbackIsDeterministicAndDoesNotGrowVocabulary() {
        val tokenizer = Tokenizer()
        val input = "Argus 世界 🎯 ~u{1F680}~"
        val initialVocabSize = tokenizer.vocabSize

        val firstEncoding = tokenizer.encode(input, autoRegister = false)
        val secondEncoding = tokenizer.encode(input, autoRegister = false)

        assertEquals("Fallback encoding must be deterministic", firstEncoding, secondEncoding)
        assertEquals("Frozen encoding must not add vocabulary entries", initialVocabSize, tokenizer.vocabSize)
        assertEquals("Unicode and literal escape markers must round-trip", input, tokenizer.decode(firstEncoding))
    }

    @Test
    fun testTokenizerDefaultsToFrozenVocabulary() {
        val tokenizer = Tokenizer()
        val initialVocabSize = tokenizer.vocabSize

        tokenizer.encode("Unseen script 漢字 and emoji 🧭")

        assertEquals("Default encode must not mutate vocabulary", initialVocabSize, tokenizer.vocabSize)
    }

    @Test
    fun testOverfitOneBatchLossReduction() {
        val engine = ArgusNeuralEngine.createBlocking(context)
        val prompt = "A.R.G.U.S belajar on device terus menerus"

        val initialLoss = engine.trainStep(prompt)
        assertFalse("Initial loss should not be NaN", initialLoss.isNaN())
        assertTrue("Initial loss should be positive (> 0.5)", initialLoss > 0.5f)

        var lastLoss = initialLoss
        for (step in 1..35) {
            val loss = engine.trainStep(prompt)
            if (!loss.isNaN()) {
                lastLoss = loss
            }
        }

        // Verify that genuine backprop & Adam significantly reduced the cross-entropy loss
        assertTrue(
            "Loss should decrease by at least 40% after 35 steps (Initial: $initialLoss, Final: $lastLoss)",
            lastLoss < initialLoss * 0.6f
        )
    }

    @Test
    fun testEnqueueDocumentSchedulesCompleteEpochPassesInOrder() = runTest {
        val engine = ArgusNeuralEngine.createBlocking(context)
        val deviceGuard = DeviceGuard(context)
        val worldModel = WorldModel(context)
        val trainingManager = TrainingManager(context, engine, deviceGuard, worldModel)

        try {
            trainingManager.pauseTraining()

            val title = "Epoch Training Sample"
            val document = (
                "Argus processes a local article as ordered blocks. " +
                    "Each block contributes next-token targets from its text. "
                ).repeat(18)
            val chunkSize = 200
            val overlap = 30
            val chunksPerEpoch = com.argus.agi.model.TextModalityProcessor
                .process(document, title, chunkSize, overlap)
                .size

            val queued = trainingManager.enqueueDocument(
                source = "Test Corpus",
                title = title,
                fullText = document,
                chunkSize = chunkSize,
                overlap = overlap,
                epochs = 3
            )

            val items = trainingManager.nextQueue.value
            assertEquals(chunksPerEpoch * 3, queued)
            assertEquals(queued, items.size)
            assertEquals(items[0].content, items[chunksPerEpoch].content)
            assertEquals(items[chunksPerEpoch].content, items[chunksPerEpoch * 2].content)
            assertTrue(items[chunksPerEpoch].title.contains("[Epoch 2/3]"))
            assertTrue(items[chunksPerEpoch * 2].title.contains("[Epoch 3/3]"))
            assertEquals(queued, items.last().chunkIndex)
            assertEquals(queued, items.last().totalChunks)
            assertEquals(
                "Persisted queue order must use unique timestamps.",
                items.size,
                items.map { it.timestamp }.toSet().size
            )
            assertTrue(
                "Queue timestamps must preserve epoch-major order.",
                items.zipWithNext().all { (first, second) -> first.timestamp < second.timestamp }
            )
        } finally {
            trainingManager.close()
        }
    }

    @Test
    fun testEnqueueDocumentStreamSchedulesMultipleCompletePasses() = runTest {
        val engine = ArgusNeuralEngine.createBlocking(context)
        val guard = DeviceGuard(context)
        val worldModel = WorldModel(context)
        val manager = TrainingManager(context, engine, guard, worldModel)

        try {
            manager.pauseTraining()
            val document = (
                "Argus parses document streams incrementally and keeps each generated chunk ordered. " +
                    "Training passes repeat the document blocks without modifying their token IDs. "
                ).repeat(10)

            val multiPassDocument = document.replaceFirst("Argus parses", "Argus copies")

            val firstPass = manager.enqueueDocumentStream(
                source = "Single pass test",
                title = "Single pass",
                reader = StringReader(document),
                chunkSize = 180,
                overlap = 30,
                epochs = 1
            )
            val multiPass = manager.enqueueDocumentStream(
                source = "Multi pass test",
                title = "Multi pass",
                reader = StringReader(multiPassDocument),
                chunkSize = 180,
                overlap = 30,
                epochs = 3
            )

            val allItems = manager.nextQueue.value
            val multiPassItems = allItems.drop(firstPass.first)

            assertTrue("Streamed document should produce multiple chunks.", firstPass.first > 1)
            assertEquals(firstPass.first * 3, multiPass.first)
            assertEquals(multiPass.first, multiPassItems.size)
            assertEquals(
                multiPassItems[0].content,
                multiPassItems[firstPass.first].content
            )
            assertEquals(
                multiPassItems[firstPass.first].content,
                multiPassItems[firstPass.first * 2].content
            )
            assertTrue(multiPassItems[firstPass.first].title.contains("[Epoch 2/3]"))
            assertTrue(multiPassItems[firstPass.first * 2].title.contains("[Epoch 3/3]"))
            assertTrue(
                multiPassItems.zipWithNext().all { (first, second) ->
                    first.timestamp < second.timestamp
                }
            )
        } finally {
            manager.close()
        }
    }

    @Test
    fun testTrainingWithUnseenUnicodeDoesNotExpandTokenizerVocabulary() {
        val engine = ArgusNeuralEngine.createBlocking(context)
        val initialVocabularySize = engine.tokenizer.vocabSize

        val loss = engine.trainStep("Argus memproses bahasa baru: 世界 🌌 مرحبا")

        assertTrue("Training loss must be finite", loss.isFinite())
        assertEquals(
            "Training must use stable Unicode fallback instead of registering tokens.",
            initialVocabularySize,
            engine.tokenizer.vocabSize
        )
    }

    @Test
    fun testEvaluateLossDoesNotMutateCheckpointState() {
        val engine = ArgusNeuralEngine.createBlocking(context)
        val sample = (
            "This is held-out validation text. It must be measured without updating " +
                "the Argus neural weights, optimizer moments, tokenizer, or training step."
            ).repeat(25)

        assertTrue("Checkpoint must be written before evaluation", engine.saveCheckpoint())
        val checkpointFile = java.io.File(context.filesDir, "argus_model_weights.bin")
        val beforeEvaluation = checkpointFile.readBytes()

        val validationLoss = engine.evaluateLoss(sample)
        val validationPerplexity = engine.evaluatePerplexity(sample)

        assertTrue("Validation loss must be finite", validationLoss.isFinite())
        assertTrue("Validation perplexity must be finite", validationPerplexity.isFinite())
        assertEquals(
            "Perplexity must be exp(cross-entropy loss)",
            kotlin.math.exp(validationLoss),
            validationPerplexity,
            0.0001f
        )
        assertTrue("Checkpoint must still be writable", engine.saveCheckpoint())
        val afterEvaluation = checkpointFile.readBytes()
        assertArrayEquals(
            "Evaluation must not mutate model or optimizer checkpoint state",
            beforeEvaluation,
            afterEvaluation
        )
    }

    @Test
    fun testAutoregressiveGenerationWithKVCache() = runTest {
        val engine = ArgusNeuralEngine.createBlocking(context)
        val output = engine.generate("Halo AGI", maxNewTokens = 15, temperature = 0.7f)
        assertNotNull("Generated output should not be null", output)
        assertTrue("Generated output should not be empty", output.isNotBlank())
        assertFalse(
            "Output must not be hardcoded template string",
            output == "A.R.G.U.S telah menganalisis input dengan model on-device lokal."
        )
    }

    @Test
    fun testAtomicCheckpointPersistence() {
        val engine1 = ArgusNeuralEngine.createBlocking(context)
        val prompt = "Uji bobot memori model"
        val loss = engine1.trainStep(prompt)
        assertFalse(loss.isNaN())

        engine1.saveCheckpoint()

        // Load into engine2
        val engine2 = ArgusNeuralEngine.createBlocking(context)
        val params2 = engine2.countTotalParameters()
        val adapterBytes2 = engine2.calculateAdapterSizeBytes()

        assertTrue("Params should be counted (> 10000)", params2 > 10000)
        assertTrue("Adapter size should be calculated (> 0 bytes)", adapterBytes2 > 0)
    }

    @Test
    fun testTrainingManagerQueueAndConcurrency() {
        val engine = ArgusNeuralEngine.createBlocking(context)
        val deviceGuard = DeviceGuard(context)
        val worldModel = WorldModel(context)
        val trainingManager = TrainingManager(context, engine, deviceGuard, worldModel)

        try {
            trainingManager.pauseTraining()

            // Test enqueue & deduplication
            trainingManager.enqueueData("Test Source", "Title 1", "Sample content for neural training")
            assertEquals(1, trainingManager.nextQueue.value.size)

            // Duplicate content should be rejected
            trainingManager.enqueueData("Test Source", "Duplicate", "Sample content for neural training")
            assertEquals(1, trainingManager.nextQueue.value.size)

            // Different content should be accepted
            trainingManager.enqueueData("Test Source", "Title 2", "Distinct content for neural training")
            assertEquals(2, trainingManager.nextQueue.value.size)

            // Clear queue
            trainingManager.clearQueue()
            assertEquals(0, trainingManager.nextQueue.value.size)

            // Clear completed
            trainingManager.clearCompleted()
            assertEquals(0, trainingManager.completedQueue.value.size)
        } finally {
            trainingManager.close()
        }
    }

    @Test
    fun testClearQueueSerializesBehindPendingPersistenceWrites() =
        kotlinx.coroutines.runBlocking<Unit> {
            val engine = ArgusNeuralEngine.createBlocking(context)
            val guard = DeviceGuard(context)
            val worldModel = WorldModel(context)
            val repository = com.argus.agi.data.repository.ArgusRepository(context)
            repository.clearPendingQueue()

            val manager = TrainingManager(
                context = context,
                neuralEngine = engine,
                deviceGuard = guard,
                worldModel = worldModel,
                repository = repository
            )

            try {
                manager.pauseTraining()
                repeat(12) { index ->
                    manager.enqueueData(
                        source = "Queue persistence test",
                        title = "Item $index",
                        content = "Unique training sample number $index for persistence ordering."
                    )
                }
                assertEquals(12, manager.nextQueue.value.size)

                manager.clearQueue()
                assertTrue(manager.nextQueue.value.isEmpty())

                manager.awaitQueuePersistenceIdle()

                assertTrue(
                    "Clearing the queue must win over earlier pending save operations.",
                    repository.getPendingQueueSnapshot().isEmpty()
                )
            } finally {
                manager.close()
            }
        }

    @Test
    fun testExcludedAppsManagerSensitivityHeuristic() {
        context.getSharedPreferences("argus_excluded_apps", Context.MODE_PRIVATE)
            .edit()
            .clear()
            .commit()
        val manager = com.argus.agi.sensor.ExcludedAppsManager(context)
        assertTrue("Authenticator must be excluded", manager.isAppExcluded("com.google.android.apps.authenticator2"))
        assertTrue("Bitwarden must be excluded by keyword", manager.isAppExcluded("com.x8bit.bitwarden"))
        assertTrue("1Password must be excluded by keyword", manager.isAppExcluded("com.agilebits.onepassword"))
        assertTrue("Banking apps must be excluded by keyword", manager.isAppExcluded("com.bca.mobilebanking"))
        assertTrue("PayPal must be excluded by keyword", manager.isAppExcluded("com.paypal.android.p2pmobile"))
        assertTrue("Venmo must be excluded by keyword", manager.isAppExcluded("com.venmo"))
        assertTrue("Unapproved apps must be blocked by default", manager.isAppExcluded("com.example.calculator"))
    }

    @Test
    fun testAccessibilityServiceSafeClickWhenDisconnected() {
        // When service is not bound/connected, performClickAction must return false cleanly
        val result = com.argus.agi.service.ArgusAccessibilityService.performClickAction("Submit")
        assertFalse("Click should return false cleanly when service instance is null", result)
    }

    @Test
    fun testDeviceGuardForceOverride() {
        val guard = DeviceGuard(context)
        guard.setForceTrainingOverride(true)
        assertTrue("When force override is enabled, isSafeToTrain must be true", guard.isSafeToTrain())
        assertTrue("Force override active flag must be true", guard.safetyState.value.isForceOverrideActive)

        guard.setForceTrainingOverride(false)
        assertFalse("Force override active flag must be false", guard.safetyState.value.isForceOverrideActive)
    }

    @Test
    fun testGenesisModelManagerPersistence() = kotlinx.coroutines.runBlocking {
        val engine = ArgusNeuralEngine.createBlocking(context)
        val guard = DeviceGuard(context)
        val genesisManager = com.argus.agi.model.GenesisModelManager(context, engine, guard)
        val hwInfo = com.argus.agi.sensor.HardwareProfile(context).getHardwareInfo()

        val identity = genesisManager.executeGenesisLifecycle(hwInfo)
        assertNotNull("Genesis identity must be generated", identity)
        assertTrue("Genesis origin ID must start with GENESIS-", identity.originId.startsWith("GENESIS-"))
        assertTrue("Device RAM must be > 0", identity.deviceRamGb > 0)
        assertTrue("Parameter count must match engine", identity.parameterCount > 0)

        // Loading from a new manager instance must preserve origin ID
        val genesisManager2 = com.argus.agi.model.GenesisModelManager(context, engine, guard)
        val identity2 = genesisManager2.getIdentity()
        assertNotNull("Identity must be persisted", identity2)
        assertEquals("Persisted origin ID must match across instances", identity.originId, identity2?.originId)
    }

    @Test
    fun testEnqueueDocumentMultiChunk() = runTest {
        val engine = ArgusNeuralEngine.createBlocking(context)
        val deviceGuard = DeviceGuard(context)
        val worldModel = WorldModel(context)
        val trainingManager = TrainingManager(context, engine, deviceGuard, worldModel)

        try {
            trainingManager.pauseTraining()

            // Text with ~600 chars should be broken into multiple distinct chunks
            val longDocument = "A.R.G.U.S adalah sistem kecerdasan on-device yang belajar secara mandiri langsung pada perangkat keras pengguna tanpa memerlukan koneksi cloud atau API pihak ketiga. Setiap dokumen yang diimpor akan dipartisi secara kontekstual menjadi beberapa blok latihan berurutan, sehingga proses backpropagation dan optimasi parameter LoRA dapat berlangsung secara terukur dan berkelanjutan tanpa memotong isi informasi penting yang terkandung di dalamnya. Sistem ini menghormati privasi penuh pengguna dan membatasi pemindaian hanya pada aplikasi yang diizinkan."

            val chunksCount = trainingManager.enqueueDocument("Document Source", "Buku Panduan", longDocument, chunkSize = 200, overlap = 30)
            assertTrue("Long document must generate at least 3 chunks", chunksCount >= 3)
            assertEquals("Queue size must equal number of enqueued chunks", chunksCount, trainingManager.nextQueue.value.size)

            val firstItem = trainingManager.nextQueue.value.first()
            assertEquals(1, firstItem.chunkIndex)
            assertEquals(chunksCount, firstItem.totalChunks)
            assertTrue(firstItem.title.contains("Blok 1/$chunksCount"))
        } finally {
            trainingManager.close()
        }
    }

    @Test
    fun testGenesisBootSeenPersistence() {
        val engine = ArgusNeuralEngine.createBlocking(context)
        val guard = DeviceGuard(context)
        val manager = com.argus.agi.model.GenesisModelManager(context, engine, guard)
        manager.markBootAnimationSeen()
        assertTrue("Boot animation flag must be persisted as seen", manager.hasSeenBootAnimation())
    }

    @Test
    fun testLargeCorpusSlidingWindowTrainStep() {
        val engine = ArgusNeuralEngine.createBlocking(context)
        // Multi-sentence text with > 150 tokens to verify multi-window sliding pass
        val largeCorpus = StringBuilder().apply {
            repeat(12) { i ->
                append("Paragraf $i: A.R.G.U.S memproses seluruh aliran token tanpa pemotongan sepihak. ")
                append("Setiap bobot neural dan parameter LoRA diperbarui secara langsung melalui gradien analitis on-device. ")
            }
        }.toString()

        val tokenCount = engine.tokenizer.encode(largeCorpus).size
        assertTrue("Corpus must have more than 128 tokens for sliding window", tokenCount > 128)

        val loss = engine.trainStep(largeCorpus)
        assertTrue("Loss must be finite for multi-window corpus", loss.isFinite())
        assertTrue("Loss must be positive", loss > 0f)
    }

    @Test
    fun testEnqueueDocumentStreamLargeCorpus() = runTest {
        val engine = ArgusNeuralEngine.createBlocking(context)
        val deviceGuard = DeviceGuard(context)
        val worldModel = WorldModel(context)
        val trainingManager = TrainingManager(context, engine, deviceGuard, worldModel)

        try {
            trainingManager.pauseTraining()
            trainingManager.clearQueue()
            val sb = StringBuilder()
            repeat(30) { i ->
                sb.append("Paragraf streaming $i: Argus membaca dokumen berurutan melalui buffered stream tanpa beban memori heap. ")
            }
            val reader = java.io.StringReader(sb.toString())
            val (chunks, charsRead) = trainingManager.enqueueDocumentStream(
                source = "Streaming Test",
                title = "Buku Besar",
                reader = reader,
                chunkSize = 250,
                overlap = 40
            )

            assertTrue("Streaming must generate multiple chunks", chunks >= 10)
            assertTrue("Total chars read must match input length", charsRead == sb.length.toLong())
            assertEquals("Next queue size must equal chunk count", chunks, trainingManager.nextQueue.value.size)
        } finally {
            trainingManager.close()
        }
    }

    @Test
    fun testZipStreamingExtractionInMemory() = runTest {
        val engine = ArgusNeuralEngine.createBlocking(context)
        val deviceGuard = DeviceGuard(context)
        val worldModel = WorldModel(context)
        val trainingManager = TrainingManager(context, engine, deviceGuard, worldModel)

        try {
            trainingManager.pauseTraining()
            trainingManager.clearQueue()
            val baos = java.io.ByteArrayOutputStream()
            java.util.zip.ZipOutputStream(baos).use { zos ->
                zos.putNextEntry(java.util.zip.ZipEntry("bab1.txt"))
                zos.write("Bab 1: A.R.G.U.S adalah model kecerdasan buatan personal lokal yang berkembang mandiri.".toByteArray(Charsets.UTF_8))
                zos.closeEntry()

                zos.putNextEntry(java.util.zip.ZipEntry("bab2.txt"))
                zos.write("Bab 2: Pembacaan arsip ZIP berlangsung secara streaming tanpa memicu OutOfMemory.".toByteArray(Charsets.UTF_8))
                zos.closeEntry()
            }

            val bais = java.io.ByteArrayInputStream(baos.toByteArray())
            var filesCount = 0
            var chunksCount = 0

            java.util.zip.ZipInputStream(bais).use { zis ->
                var entry = zis.nextEntry
                while (entry != null) {
                    if (!entry.isDirectory && entry.name.endsWith(".txt")) {
                        filesCount++
                        val nonClosing = object : java.io.FilterInputStream(zis) { override fun close() {} }
                        val reader = java.io.BufferedReader(java.io.InputStreamReader(nonClosing, Charsets.UTF_8))
                        val (c, _) = trainingManager.enqueueDocumentStream("ZIP Test", entry.name, reader)
                        chunksCount += c
                    }
                    zis.closeEntry()
                    entry = zis.nextEntry
                }
            }

            assertEquals("Must extract 2 text files from zip", 2, filesCount)
            assertTrue("Must enqueue at least 2 chunks", chunksCount >= 2)
            assertEquals("Next queue size must match enqueued chunks", chunksCount, trainingManager.nextQueue.value.size)
        } finally {
            trainingManager.close()
        }
    }

    @Test
    fun testTextModalityProcessorConversationFormat() {
        val rawChat = """
            User: Halo Argus, bisakah kamu mengenali lingkungan sekitar?
            Assistant: Ya, saya memiliki modul persepsi kamera dan layar yang bekerja secara on-device.
            User: Apakah data saya dikirim ke internet?
            Assistant: Tidak sama sekali. Semua pemrosesan berlangsung 100% lokal di perangkat Anda.
        """.trimIndent()

        val modality = com.argus.agi.model.TextModalityProcessor.detectModality(rawChat)
        assertEquals("Chat must be detected as CONVERSATION", com.argus.agi.model.TextModality.CONVERSATION, modality)

        val processed = com.argus.agi.model.TextModalityProcessor.process(rawChat, "Sesi Obrolan", chunkSize = 400)
        assertTrue("Processed chunks must not be empty", processed.isNotEmpty())

        val firstChunk = processed.first()
        assertTrue("Chunk must contain <USER> token", firstChunk.content.contains("<USER>"))
        assertTrue("Chunk must contain <ASSISTANT> token", firstChunk.content.contains("<ASSISTANT>"))
        assertTrue("Chunk must end with <EOS> token", firstChunk.content.contains("<EOS>"))
    }

    @Test
    fun testTextModalityProcessorQAPairsFormat() {
        val rawQA = """
            Tanya: Apa itu hukum inersia Newton?
            Jawab: Benda akan mempertahankan keadaan diam atau gerak lurus beraturannya kecuali ada gaya luar yang mempengaruhinya.
            Tanya: Siapa penemu kalkulus?
            Jawab: Isaac Newton dan Gottfried Wilhelm Leibniz secara independen mengembangkan kalkulus.
        """.trimIndent()

        val modality = com.argus.agi.model.TextModalityProcessor.detectModality(rawQA)
        assertEquals("Q&A text must be detected as QA_PAIR", com.argus.agi.model.TextModality.QA_PAIR, modality)

        val processed = com.argus.agi.model.TextModalityProcessor.process(rawQA, "Fisika & Matematika")
        assertEquals("Must produce 2 distinct QA training pairs", 2, processed.size)

        for (chunk in processed) {
            assertTrue("Must contain <USER> Tanya:", chunk.content.contains("<USER> Tanya:"))
            assertTrue("Must contain <ASSISTANT> Jawab:", chunk.content.contains("<ASSISTANT> Jawab:"))
            assertTrue("Must contain <EOS>", chunk.content.contains("<EOS>"))
        }
    }

    @Test
    fun testTextModalityProcessorStoryNarrativeAndTypography() {
        val rawStory = """
            Malam itu sangat dingin. “Apakah kita akan berhasil menyeberangi ngarai ini?” bisiknya cemas.
            “Tentu saja,” jawabnya dengan tenang—meskipun badai semakin menderu… Sang pengelana menatap bintang-bintang di kejauhan.
            “Jangan pernah menyerah pada kegelapan,” gumam sang ksatria seraya melangkah maju.
        """.trimIndent()

        // 1. Verify typography normalization
        val normalized = com.argus.agi.model.TextModalityProcessor.normalizeTypography(rawStory)
        assertFalse("Curly double quotes must be normalized to standard ASCII quotes", normalized.contains('“') || normalized.contains('”'))
        assertFalse("Em-dash must be normalized to hyphen", normalized.contains('—'))
        assertFalse("Single-character ellipsis must be normalized to three dots", normalized.contains('…'))
        assertTrue("Standard double quotes must exist", normalized.contains('"'))
        assertTrue("Hyphen must exist", normalized.contains('-'))
        assertTrue("Three dots must exist", normalized.contains("..."))

        // 2. Tokenize with Tokenizer and ensure clean token IDs
        val engine = ArgusNeuralEngine.createBlocking(context)
        val tokens = engine.tokenizer.encode(rawStory)
        assertTrue("Story tokens must be encoded", tokens.size > 10)
        // Ensure UNK is not spammed
        val unkCount = tokens.count { it == com.argus.agi.model.Tokenizer.UNK_ID }
        assertEquals("Standard story tokens must encode without producing UNK tokens", 0, unkCount)

        // 3. Train step verification
        val loss = engine.trainStep(rawStory)
        assertTrue("Loss must be valid finite number", loss.isFinite() && loss > 0f)
    }

    @Test
    fun testTextModalityProcessorArticleNewsFormat() {
        val rawNews = """
            JAKARTA — Pemerintah Indonesia menurut laporan resmi telah meresmikan pusat riset teknologi masa depan.

            Penelitian ini melibatkan berbagai perguruan tinggi negeri dan institusi riset independen. Menurut juru bicara kepresidenan, fasilitas ini diharapkan dapat mempercepat inovasi komputasi tepi di berbagai sektor vital nasional.

            Perusahaan teknologi swasta juga turut serta memberikan kontribusi berupa infrastruktur pengujian terdesentralisasi.
        """.trimIndent()

        val modality = com.argus.agi.model.TextModalityProcessor.detectModality(rawNews)
        assertEquals("Formal news text must be detected as ARTICLE_NEWS", com.argus.agi.model.TextModality.ARTICLE_NEWS, modality)

        val processed = com.argus.agi.model.TextModalityProcessor.process(rawNews, "Berita Teknologi", chunkSize = 250)
        assertTrue("News text must generate chunks partitioned by paragraph/sentence", processed.isNotEmpty())
        for (chunk in processed) {
            assertFalse("Chunk must not start or end with arbitrary broken punctuation", chunk.content.isBlank())
        }
    }

    @Test
    fun testDynamicEngineConfigAndFullTraining() {
        val customConfig = com.argus.agi.model.EngineConfig(
            dModel = 48,
            nHeads = 4,
            dFf = 96,
            maxSeqLen = 64,
            loraRank = 2,
            isFullTrainingEnabled = true
        )
        val engine = ArgusNeuralEngine.createBlocking(context, customConfig)
        assertEquals(48, engine.dModel)
        assertEquals(96, engine.dFf)
        assertEquals(2, engine.loraRank)

        val loss = engine.trainStep("Test training with dynamic compact configuration.")
        assertTrue("Loss must be valid finite number on custom config", loss.isFinite())
        assertTrue("Loss must be positive", loss > 0f)

        // Save checkpoint with custom dimensions and verify roundtrip
        val saved = engine.saveCheckpoint()
        assertTrue("Checkpoint must be successfully verified and saved", saved)
    }

    @Test
    fun testDeviceGuardThermalCeilingOverrideImmunity() {
        val guard = DeviceGuard(context)
        guard.setForceTrainingOverride(true)
        assertTrue("Force override can be toggled", guard.forceTrainingOverride.value)

        // Simulate physical thermal hazard (>= 48°C)
        val intent = android.content.Intent(android.content.IntentFilter(android.content.Intent.ACTION_BATTERY_CHANGED).actionsIterator().next()).apply {
            putExtra(android.os.BatteryManager.EXTRA_LEVEL, 80)
            putExtra(android.os.BatteryManager.EXTRA_SCALE, 100)
            putExtra(android.os.BatteryManager.EXTRA_TEMPERATURE, 490) // 49.0°C
        }
        val method = DeviceGuard::class.java.getDeclaredMethod("updateState", android.content.Intent::class.java).apply {
            isAccessible = true
        }
        method.invoke(guard, intent)

        val state = guard.safetyState.value
        assertTrue("Physical thermal hazard must be detected", state.isPhysicalThermalHazard)
        assertFalse("Physical thermal hazard must NOT allow training even with force override active", state.isSafeToTrain)
    }

    @Test
    fun testUnboundedDocumentAndStreamTrainingBlocks() = runTest {
        val engine = ArgusNeuralEngine.createBlocking(context)
        val guard = DeviceGuard(context)
        val world = WorldModel(context)
        val manager = TrainingManager(context, engine, guard, world)

        try {
            manager.pauseTraining()
            manager.clearQueue()
            manager.awaitQueuePersistenceIdle()
            val repository = com.argus.agi.data.repository.ArgusRepository(context)
            // 1. Validasi enqueueDocument memproses seluruh teks sampai selesai tanpa batas blok buatan
            val text = "Paragraf pengujian kelengkapan blok dataset besar. ".repeat(100)
            val count1 = manager.enqueueDocument("Test Source", "Buku Lengkap", text, chunkSize = 120, overlap = 20)
            assertTrue("Seluruh blok dokumen harus dihasilkan", count1 > 10)
            assertEquals("Semua blok yang dihasilkan harus masuk ke antrean", count1, manager.nextQueue.value.size)

            // 2. Validasi enqueueDocumentStream membaca tuntas sampai EOF
            val streamText = "Aliran teks dataset streaming on-device tanpa batasan 300 blok. ".repeat(150)
            val reader = java.io.StringReader(streamText)
            val (streamChunks, streamChars) = manager.enqueueDocumentStream(
                source = "Stream Source",
                title = "Stream Doc",
                reader = reader,
                chunkSize = 100,
                overlap = 20
            )
            assertTrue("Streaming harus membaca seluruh karakter sampai EOF", streamChars > 0)
            assertTrue("Streaming harus menghasilkan semua blok secara tuntas", streamChunks > 20)
            manager.awaitQueuePersistenceIdle()
            assertEquals(
                "Semua item antrean harus tersimpan secara durable, meskipun halaman RAM dibatasi.",
                count1 + streamChunks,
                repository.getPendingTrainingQueueCount()
            )
            assertTrue(
                "Halaman antrean di memori tidak boleh melebihi batas prefetched.",
                manager.nextQueue.value.size <= 128
            )
        } finally {
            manager.close()
        }
    }

    @Test
    fun testRoomDatabasePersistence() = runTest {
        val testDb = androidx.room.Room.inMemoryDatabaseBuilder(
            context,
            com.argus.agi.data.local.ArgusDatabase::class.java
        ).allowMainThreadQueries().build()

        try {
            val chatDao = testDb.chatMessageDao()
            val memoryDao = testDb.worldMemoryDao()
            val queueDao = testDb.trainingQueueDao()

            // 1. Chat Message persistence
            val chatEntity = com.argus.agi.data.local.entity.ChatMessageEntity(
                id = "test-msg-1",
                isUser = true,
                text = "Halo A.R.G.U.S on-device",
                thinkReasoning = "Testing Room persistence"
            )
            chatDao.insertMessage(chatEntity)
            val fetchedMessages = chatDao.getAllMessagesSnapshot()
            assertTrue("Messages list should contain saved message", fetchedMessages.any { it.id == "test-msg-1" && it.text == "Halo A.R.G.U.S on-device" })

            // 2. World Memory persistence
            val entity = com.argus.agi.data.local.entity.WorldMemoryEntity(
                key = "user_preference_test",
                value = "Bahasa Indonesia",
                source = "User Profile"
            )
            memoryDao.insertMemory(entity)
            val fetchedMemories = memoryDao.getAllMemoriesSnapshot()
            assertTrue("Memories list should contain saved entity", fetchedMemories.any { it.key == "user_preference_test" && it.value == "Bahasa Indonesia" })

            // 3. Training Queue persistence
            val trainingItem = com.argus.agi.data.local.entity.TrainingItemEntity(
                id = "train-item-1",
                source = "Document Test",
                title = "Test Chapter 1",
                content = "Isi teks untuk validasi persistence Room.",
                chunkIndex = 1,
                totalChunks = 1,
                status = "QUEUED"
            )
            queueDao.insertItem(trainingItem)
            val fetchedQueue = queueDao.getPendingQueueSnapshot()
            assertTrue("Pending queue should contain item", fetchedQueue.any { it.id == "train-item-1" })

            queueDao.updateStatus("train-item-1", status = "COMPLETED", loss = 1.234f, duration = 250L)
            val queueAfterCompletion = queueDao.getPendingQueueSnapshot()
            assertFalse("Pending queue should no longer contain completed item", queueAfterCompletion.any { it.id == "train-item-1" })
        } finally {
            testDb.close()
        }
    }

    @Test
    fun testJsonlLexiconProcessing() {
        val jsonlSample = """
            {"id": 1, "kata": "susu", "kata_dasar": "susu", "sumber": "kata_dasar"}
            {"id": 2, "kata": "daging", "kata_dasar": "daging", "sumber": "kata_dasar"}
            {"id": 3, "kata": "rentan", "kata_dasar": "rentan", "sumber": "kata_dasar"}
            {"id": 4, "kata": "awan", "kata_dasar": "awan (1), perawanan", "sumber": "kata_dasar"}
            {"id": 5, "kata": "perawanan", "kata_dasar": "awan (1), perawanan", "sumber": "kata_dasar"}
            {"id": 6, "kata": "pangsa", "kata_dasar": "pangsa", "sumber": "kata_dasar"}
            {"id": 7, "kata": "abu", "kata_dasar": "abu (1)", "sumber": "kata_dasar"}
            {"id": 8, "kata": "anjing", "kata_dasar": "anjing", "sumber": "kata_dasar"}
        """.trimIndent()

        val modality = com.argus.agi.model.TextModalityProcessor.detectModality(jsonlSample)
        assertEquals("JSONL lines must be recognized as STRUCTURED_DATA", com.argus.agi.model.TextModality.STRUCTURED_DATA, modality)

        val chunks = com.argus.agi.model.TextModalityProcessor.process(jsonlSample, "Kamus KBBI")
        assertTrue("Must generate at least one processed chunk", chunks.isNotEmpty())
        val firstChunkContent = chunks.first().content
        assertTrue("Processed chunk must contain semantic relation for susu", firstChunkContent.contains("Kata: susu -> Kata dasar: susu"))
        assertTrue("Processed chunk must contain semantic relation for perawanan", firstChunkContent.contains("Kata: perawanan -> Kata dasar: awan (1), perawanan"))

        // Verify Neural Engine can execute training step on this processed lexicon
        val engine = ArgusNeuralEngine.createBlocking(context)
        val loss = engine.trainStep(firstChunkContent)
        assertTrue("Training loss on lexicon must be finite", loss.isFinite())
        assertTrue("Training loss on lexicon must be positive", loss > 0f)
    }
}



