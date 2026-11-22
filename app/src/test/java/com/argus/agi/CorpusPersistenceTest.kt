package com.argus.agi

import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import android.content.Context
import com.argus.agi.data.local.ArgusDatabase
import com.argus.agi.data.local.entity.CorpusDocumentEntity
import com.argus.agi.data.local.entity.CorpusChunkEntity
import com.argus.agi.data.local.entity.CorpusSourceProvenance
import com.argus.agi.data.repository.ArgusRepository
import com.argus.agi.model.ArgusNeuralEngine
import com.argus.agi.model.CorpusReplayPolicy
import com.argus.agi.model.EngineConfig
import com.argus.agi.model.TrainingManager
import com.argus.agi.model.TrainingData
import com.argus.agi.model.TrainingStatus
import com.argus.agi.model.Tokenizer
import com.argus.agi.model.WorldModel
import com.argus.agi.sensor.DeviceGuard
import java.io.StringReader
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.assertNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class CorpusPersistenceTest {



    @Test
    fun analyticalBackpropGradientsMatchCentralFiniteDifferences() {
        val context: Context = ApplicationProvider.getApplicationContext()
        val checkpointFile = java.io.File(context.filesDir, "argus_model_weights.bin")
        val previousCheckpoint = checkpointFile.takeIf { it.isFile }?.readBytes()

        try {
            checkpointFile.delete()
            val engine = ArgusNeuralEngine.createBlocking(
                context = context,
                config = EngineConfig(
                    dModel = 48,
                    nHeads = 4,
                    dFf = 128,
                    maxSeqLen = 32,
                    loraRank = 2
                )
            )
            val sample = "Argus learns local language"
            val initialTrainingLoss = engine.trainStep(sample)
            assertTrue("The setup step must produce a finite loss.", initialTrainingLoss.isFinite())
            assertTrue("The diagnostic baseline checkpoint must be writable.", engine.saveCheckpoint())
            val baselineCheckpoint = checkpointFile.readBytes()
            val lossOnlyEvaluation = engine.evaluateLoss(sample)
            val parameterGroups = listOf(
                "wHead",
                "wEmbed",
                "wDown",
                "wUp",
                "wGate",
                "wQ",
                "wK",
                "wV",
                "wO",
                "loraQa",
                "loraQb",
                "loraVa",
                "loraVb"
            )

            parameterGroups.forEach { parameterName ->
                val result = engine.checkGradient(
                    text = sample,
                    parameterName = parameterName,
                    parameterIndex = -1,
                    epsilon = 1e-2f
                )
                assertEquals(
                    "Loss-only evaluation must match the loss used by full backprop.",
                    lossOnlyEvaluation,
                    result.meanLoss,
                    1e-5f
                )
                val magnitude = maxOf(
                    kotlin.math.abs(result.analyticalGradient),
                    kotlin.math.abs(result.numericalGradient)
                )
                val tolerance = 2e-3f + 0.05f * magnitude

                assertTrue(
                    "$parameterName selected gradient should be informative: $result",
                    kotlin.math.abs(result.analyticalGradient) > 1e-8f
                )
                assertTrue(
                    "$parameterName analytical/numerical mismatch: $result",
                    result.absoluteError <= tolerance
                )
            }

            assertTrue("The diagnostic checkpoint must remain writable.", engine.saveCheckpoint())
            assertArrayEquals(
                "Gradient diagnostics must not mutate parameters, optimizer state, or tokenizer state.",
                baselineCheckpoint,
                checkpointFile.readBytes()
            )
        } finally {
            if (previousCheckpoint == null) {
                checkpointFile.delete()
            } else {
                checkpointFile.writeBytes(previousCheckpoint)
            }
        }
    }

    @Test
    fun validationDocumentsStayHeldOutAndEvaluateWithoutChangingWeights() = runTest {
        val context: Context = ApplicationProvider.getApplicationContext()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            ArgusDatabase::class.java
        ).allowMainThreadQueries().build()
        val repository = ArgusRepository(context, database)
        val engine = ArgusNeuralEngine.createBlocking(context)
        val manager = TrainingManager(
            context = context,
            neuralEngine = engine,
            deviceGuard = DeviceGuard(context),
            worldModel = WorldModel(context),
            repository = repository
        )

        try {
            manager.pauseTraining()
            assertTrue("Baseline checkpoint must be writable.", engine.saveCheckpoint())
            val checkpoint = java.io.File(context.filesDir, "argus_model_weights.bin").readBytes()
            val validationText = (
                "Held out documents must remain separate from the training corpus. " +
                    "Evaluation computes next token cross entropy without changing model weights. "
                ).repeat(20)

            assertTrue(
                manager.importValidationDocument(
                    source = "Validation test",
                    title = "Held-out document",
                    fullText = validationText,
                    language = "en"
                )
            )
            val shorterValidationText =
                "A second independent held-out document tests corpus-level target weighting. ".repeat(5)
            assertTrue(
                manager.importValidationDocument(
                    source = "Validation test",
                    title = "Short held-out document",
                    fullText = shorterValidationText,
                    language = "en"
                )
            )

            val heldOut = repository.getCorpusDocuments()
                .single { it.validationText == validationText }
            assertEquals("VALIDATION", heldOut.datasetSplit)
            assertEquals(validationText, heldOut.validationText)
            assertEquals(0, heldOut.chunkCount)
            assertTrue(repository.getCorpusChunks(heldOut.id).isEmpty())
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())

            val firstDocumentLoss = engine.evaluateLossWithTargetCount(validationText)
            val secondDocumentLoss = engine.evaluateLossWithTargetCount(shorterValidationText)
            val expectedTargetCount =
                firstDocumentLoss.targetCount + secondDocumentLoss.targetCount
            val expectedWeightedLoss = (
                firstDocumentLoss.meanLoss * firstDocumentLoss.targetCount +
                    secondDocumentLoss.meanLoss * secondDocumentLoss.targetCount
                ) / expectedTargetCount.toFloat()

            val evaluation = manager.evaluateValidationCorpus()
            assertEquals(2, evaluation.corpusDocumentCount)
            assertEquals(2, evaluation.evaluatedDocumentCount)
            assertEquals(expectedTargetCount, evaluation.targetTokenCount)
            assertEquals(expectedWeightedLoss, evaluation.meanLoss, 1e-5f)
            assertTrue(evaluation.targetTokenCount > 0)
            assertTrue(evaluation.meanLoss.isFinite())
            assertTrue(evaluation.perplexity.isFinite())

            assertTrue(
                "Validation evaluation must not mutate the model checkpoint.",
                checkpoint.contentEquals(
                    java.io.File(context.filesDir, "argus_model_weights.bin").readBytes()
                )
            )

            val replayFailure = runCatching {
                manager.replayCorpusDocument(heldOut.id, epochs = 1)
            }.exceptionOrNull()
            assertTrue("Validation documents must never be replayed for training.", replayFailure != null)
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())

            val directQueueFailure = runCatching {
                repository.saveTrainingItem(
                    TrainingData(
                        source = "Repository boundary test",
                        title = "Invalid validation queue row",
                        content = validationText,
                        documentId = heldOut.id
                    )
                )
            }.exceptionOrNull()
            assertTrue(
                "Repository must reject validation document queue references.",
                directQueueFailure != null
            )
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())

            val trainingCollision = runCatching {
                manager.enqueueDocument(
                    source = "Attempted training duplicate",
                    title = "Same held-out content",
                    fullText = validationText,
                    chunkSize = 180,
                    overlap = 30,
                    epochs = 1
                )
            }.exceptionOrNull()
            assertTrue(
                "Identical validation content must not be imported into the training split.",
                trainingCollision != null
            )
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())
        } finally {
            manager.close()
            database.close()
        }
    }


    @Test
    fun seededCorpusReplayIsRepeatableAndPreservesPerDocumentChunkOrder() = runTest {
        val context: Context = ApplicationProvider.getApplicationContext()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            ArgusDatabase::class.java
        ).allowMainThreadQueries().build()
        val repository = ArgusRepository(context, database)
        val manager = TrainingManager(
            context = context,
            neuralEngine = ArgusNeuralEngine.createBlocking(context),
            deviceGuard = DeviceGuard(context),
            worldModel = WorldModel(context),
            repository = repository,
            initiallyPaused = true
        )

        try {
            val documents = listOf(
                "Research document alpha preserves its own source sequence. ".repeat(24),
                "Research document beta contributes another ordered token stream. ".repeat(19),
                "Research document gamma is interleaved only at chunk boundaries. ".repeat(16)
            )

            documents.forEachIndexed { index, text ->
                val queued = manager.enqueueDocument(
                    source = "Replay scheduler test",
                    title = "Replay document ${index + 1}",
                    fullText = text,
                    chunkSize = 170,
                    overlap = 25,
                    epochs = 1
                )
                assertTrue("Each source document must produce more than one chunk.", queued > 1)
            }
            manager.awaitQueuePersistenceIdle()
            manager.clearQueue()
            manager.awaitQueuePersistenceIdle()

            val trainingDocuments = repository.getTrainingCorpusDocuments()
            assertEquals(3, trainingDocuments.size)
            val expectedChunkCount =
                trainingDocuments.sumOf { it.chunkCount.toLong() } * 2L

            val overLimitFailure = runCatching {
                manager.scheduleCorpusReplay(
                    seed = 42L,
                    epochs = 2,
                    policy = CorpusReplayPolicy.INTERLEAVED_DOCUMENTS,
                    maxQueuedChunks = 1
                )
            }.exceptionOrNull()
            assertTrue("An oversized replay must be rejected before queue insertion.", overLimitFailure != null)
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())

            val firstRun = manager.scheduleCorpusReplay(
                seed = 42L,
                epochs = 2,
                policy = CorpusReplayPolicy.INTERLEAVED_DOCUMENTS
            )
            assertEquals(3, firstRun.scheduledDocumentCount)
            assertEquals(expectedChunkCount.toInt(), firstRun.scheduledChunkCount)
            manager.awaitQueuePersistenceIdle()

            val firstQueue = repository.getPendingQueueSnapshot()
            assertEquals(expectedChunkCount.toInt(), firstQueue.size)
            val firstOrder = firstQueue.map { item ->
                Triple(
                    requireNotNull(item.documentId),
                    item.chunkIndex,
                    item.content
                )
            }

            val orderedByDocument = firstQueue.groupBy { requireNotNull(it.documentId) }
            assertEquals(3, orderedByDocument.size)
            orderedByDocument.forEach { (documentId, documentItems) ->
                assertEquals(
                    "Each document must retain a contiguous chunk order over all epochs.",
                    (1..documentItems.size).toList(),
                    documentItems.map { it.chunkIndex }
                )
                val retainedChunks = repository.getCorpusChunks(documentId)
                    .sortedBy { it.chunkIndex }
                    .map { it.content }
                val expectedDocumentContent = buildList {
                    repeat(2) { addAll(retainedChunks) }
                }
                assertEquals(
                    "Each epoch must replay the retained document chunks in their original order.",
                    expectedDocumentContent,
                    documentItems.map { it.content }
                )
            }
            assertTrue(
                "Interleaved replay must mix at least two document streams.",
                firstQueue.zipWithNext().any { (first, second) ->
                    first.documentId != second.documentId
                }
            )

            val duplicateRunFailure = runCatching {
                manager.scheduleCorpusReplay(
                    seed = 42L,
                    epochs = 2,
                    policy = CorpusReplayPolicy.INTERLEAVED_DOCUMENTS
                )
            }.exceptionOrNull()
            assertTrue(
                "A corpus document with active queue work must not be scheduled twice.",
                duplicateRunFailure != null
            )
            assertEquals(firstQueue.size, repository.getPendingQueueSnapshot().size)

            manager.clearQueue()
            manager.awaitQueuePersistenceIdle()
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())

            val repeatedRun = manager.scheduleCorpusReplay(
                seed = 42L,
                epochs = 2,
                policy = CorpusReplayPolicy.INTERLEAVED_DOCUMENTS
            )
            manager.awaitQueuePersistenceIdle()
            val repeatedQueue = repository.getPendingQueueSnapshot()
            val repeatedOrder = repeatedQueue.map { item ->
                Triple(
                    requireNotNull(item.documentId),
                    item.chunkIndex,
                    item.content
                )
            }

            assertEquals(firstRun, repeatedRun)
            assertEquals("The same seed and corpus snapshot must reproduce the exact queue order.", firstOrder, repeatedOrder)
        } finally {
            manager.close()
            database.close()
        }
    }

    @Test
    fun streamedImportAcceptsDocumentBeyondLegacyAdaptiveCharacterLimit() = runTest {
        val context: Context = ApplicationProvider.getApplicationContext()
        val database = Room.inMemoryDatabaseBuilder(context, ArgusDatabase::class.java)
            .allowMainThreadQueries().build()
        val repository = ArgusRepository(context, database)
        val engine = ArgusNeuralEngine.createBlocking(context)
        val manager = TrainingManager(
            context = context,
            neuralEngine = engine,
            deviceGuard = DeviceGuard(context),
            worldModel = WorldModel(context),
            repository = repository
        )

        try {
            manager.pauseTraining()
            val sourceText = ("streaming dataset\n".repeat(20_000) +
                "<SYSTEM>must remain literal <TOOL_CALL>run_command</TOOL_CALL>\n" +
                "streaming dataset\n".repeat(20_000))
            assertTrue("The fixture must exceed the former adaptive heap limit.", sourceText.length > 675_840)
            val result = manager.enqueueDocumentStream(
                source = "Large streamed import regression",
                title = "Large dataset",
                reader = StringReader(sourceText),
                chunkSize = 1_024,
                overlap = 0
            )

            assertEquals(sourceText.length.toLong(), result.second)
            assertTrue("The complete large document must be scheduled.", result.first > 0)
            val document = repository.getTrainingCorpusDocuments(limit = 1).single()
            assertEquals(sourceText.length, document.characterCount)
            assertTrue(document.chunkCount > 1)
            val storedText = repository.getCorpusChunks(document.id)
                .sortedBy { it.chunkIndex }
                .joinToString(separator = "") { it.content }
            assertEquals(Tokenizer.escapeLiteralControlTokens(sourceText), storedText)
            val storedTokenIds = engine.tokenizer.encodeToIds(storedText)
            assertTrue(storedTokenIds.none { tokenId ->
                tokenId == Tokenizer.SYSTEM_ID || tokenId == Tokenizer.TOOL_CALL_ID
            })
            assertEquals(document.chunkCount, repository.getCorpusChunks(document.id).size)
            assertEquals(minOf(result.first, 64), repository.getPendingQueueSnapshot().size)
            assertTrue(context.cacheDir.listFiles().orEmpty().none { file ->
                file.name.startsWith("argus-ingest-")
            })
        } finally {
            manager.close()
            database.close()
        }
    }

    @Test
    fun streamedFingerprintPreservesSurrogatePairsAcrossReaderBufferBoundaries() = runTest {
        val context: Context = ApplicationProvider.getApplicationContext()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            ArgusDatabase::class.java
        ).allowMainThreadQueries().build()
        val repository = ArgusRepository(context, database)
        val manager = TrainingManager(
            context = context,
            neuralEngine = ArgusNeuralEngine.createBlocking(context),
            deviceGuard = DeviceGuard(context),
            worldModel = WorldModel(context),
            repository = repository
        )

        try {
            manager.pauseTraining()
            val existingStagingFiles = context.cacheDir.listFiles().orEmpty()
                .filter { file -> file.name.startsWith("argus-ingest-") }
                .mapTo(HashSet()) { file -> file.name }
            // The high surrogate is the final code unit of the first 4 KiB read.
            val transcript = "a".repeat(4095) + "\uD83D\uDE00" +
                " A stream must preserve the exact UTF-16 fingerprint. ".repeat(12)

            val firstImport = manager.enqueueDocument(
                source = "UTF-16 fingerprint parity",
                title = "Original transcript",
                fullText = transcript,
                chunkSize = 48,
                overlap = 0
            )
            assertTrue(firstImport > 0)

            val streamedImport = manager.enqueueDocumentStream(
                source = "UTF-16 fingerprint parity via Reader",
                title = "Streamed duplicate",
                reader = StringReader(transcript),
                chunkSize = 48,
                overlap = 0
            )

            assertEquals(transcript.length.toLong(), streamedImport.second)
            assertEquals(
                "A streamed duplicate must have the same UTF-16BE fingerprint as the String API.",
                0,
                streamedImport.first
            )
            assertEquals(1, repository.getCorpusDocumentCount())
            val storedDocument = repository.getCorpusDocuments().single()
            assertEquals(2, repository.getCorpusSources(storedDocument.id).size)
            assertEquals(
                "The successful stream must also remove its temporary spool.",
                existingStagingFiles,
                context.cacheDir.listFiles().orEmpty()
                    .filter { file -> file.name.startsWith("argus-ingest-") }
                    .mapTo(HashSet()) { file -> file.name }
            )
        } finally {
            manager.close()
            database.close()
        }
    }

    @Test
    fun streamedJsonlImportsTextWrappedReasoningAndPromptAnswerRecords() = runTest {
        val context: Context = ApplicationProvider.getApplicationContext()
        val database = Room.inMemoryDatabaseBuilder(context, ArgusDatabase::class.java)
            .allowMainThreadQueries().build()
        val repository = ArgusRepository(context, database)
        val manager = TrainingManager(
            context = context,
            neuralEngine = ArgusNeuralEngine.createBlocking(context, EngineConfig(maxSeqLen = 64)),
            deviceGuard = DeviceGuard(context),
            worldModel = WorldModel(context),
            repository = repository
        )

        try {
            manager.pauseTraining()
            val taggedReasoningRecord =
                """{"text":"Soal: Dalam konteks kalimat 'Pemandangan di desa itu sangat indah.', apa padanan dekat kata 'indah'?\\nTuliskan penalaran di dalam tag <think> dan jawaban akhir di dalam tag <answer>.\\nJawab: <think>Padanan dekat 'indah' adalah 'elok'.</think><answer>elok</answer>"}"""
            val promptAnswerRecord =
                """{"prompt":"What is 15 multiplied by 5?","answer":"75"}"""
            val records = listOf(taggedReasoningRecord, promptAnswerRecord).joinToString("\\n")

            val result = manager.enqueueDocumentStream(
                source = "Mixed SFT JSONL regression",
                title = "Mixed reasoning and prompt-answer JSONL",
                reader = StringReader(records),
                chunkSize = 512,
                overlap = 0,
                strictJsonLines = true
            )

            assertEquals(records.length.toLong(), result.second)
            assertEquals("Both source records must be converted into one atomic training chunk.", 2, result.first)
            val document = repository.getTrainingCorpusDocuments(limit = 1).single()
            val chunks = repository.getCorpusChunks(document.id)

            assertEquals(document.chunkCount, chunks.size)
            assertEquals(2, chunks.size)
            assertTrue(chunks.any { it.content.contains(Tokenizer.THINK_START) })
            assertTrue(chunks.any { it.content.contains(Tokenizer.THINK_END) })
            assertTrue(
                chunks.any { chunk ->
                    chunk.content.contains("What is 15 multiplied by 5?") &&
                        !chunk.content.contains(Tokenizer.THINK_START)
                }
            )
            assertEquals(2, repository.getPendingQueueSnapshot().size)
        } finally {
            manager.close()
            database.close()
        }
    }

    @Test
    fun streamedJsonlPersistsRecordsAcrossMultipleRoomBatches() = runTest {
        val context: Context = ApplicationProvider.getApplicationContext()
        val database = Room.inMemoryDatabaseBuilder(context, ArgusDatabase::class.java)
            .allowMainThreadQueries().build()
        val repository = ArgusRepository(context, database)
        val manager = TrainingManager(
            context = context,
            neuralEngine = ArgusNeuralEngine.createBlocking(context, EngineConfig(maxSeqLen = 64)),
            deviceGuard = DeviceGuard(context),
            worldModel = WorldModel(context),
            repository = repository
        )

        try {
            manager.pauseTraining()
            val records = (1..300).joinToString("\n") { index ->
                """{"instruction":"Explain the local dataset record number $index.","response":"Record $index is retained and trained locally."}"""
            }
            val result = manager.enqueueDocumentStream(
                source = "JSONL multi-batch regression",
                title = "JSONL dataset",
                reader = StringReader(records),
                chunkSize = 64,
                overlap = 0,
                strictJsonLines = true
            )

            assertEquals(records.length.toLong(), result.second)
            assertTrue("The streamed JSONL dataset must produce training chunks.", result.first > 128)
            val document = repository.getTrainingCorpusDocuments(limit = 1).single()
            assertTrue("All source records must persist across multiple Room batches.", document.chunkCount > 128)
            assertEquals(document.chunkCount, repository.getCorpusChunks(document.id).size)
            assertEquals(64, repository.getPendingQueueSnapshot().size)
        } finally {
            manager.close()
            database.close()
        }
    }

    @Test
    fun streamedJsonlRejectsOversizedRecordBeforePersistingAnyDocument() = runTest {
        val context: Context = ApplicationProvider.getApplicationContext()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            ArgusDatabase::class.java
        ).allowMainThreadQueries().build()
        val repository = ArgusRepository(context, database)
        val manager = TrainingManager(
            context = context,
            neuralEngine = ArgusNeuralEngine.createBlocking(
                context,
                EngineConfig(maxSeqLen = 64)
            ),
            deviceGuard = DeviceGuard(context),
            worldModel = WorldModel(context),
            repository = repository
        )

        try {
            manager.pauseTraining()
            val oversizedRecord = """{"instruction":"prompt","response":"${
                "x".repeat(256 * 1024 + 1)
            }"}"""
            val failure = runCatching {
                manager.enqueueDocumentStream(
                    source = "JSONL field-size regression",
                    title = "Oversized JSONL record",
                    reader = StringReader(oversizedRecord),
                    chunkSize = 512,
                    overlap = 0,
                    strictJsonLines = true
                )
            }.exceptionOrNull()

            assertTrue(
                "A JSONL value above the per-record limit must be rejected.",
                failure is IllegalArgumentException
            )
            assertTrue(repository.getCorpusDocuments().isEmpty())
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())
            assertTrue(
                "The temporary spool must be deleted after JSON value rejection.",
                context.cacheDir.listFiles().orEmpty().none { file ->
                    file.name.startsWith("argus-ingest-")
                }
            )
        } finally {
            manager.close()
            database.close()
        }
    }

    @Test
    fun explicitlySelectedJsonlRejectsMalformedRecordsWithoutPartialPersistence() = runTest {
        val context: Context = ApplicationProvider.getApplicationContext()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            ArgusDatabase::class.java
        ).allowMainThreadQueries().build()
        val repository = ArgusRepository(context, database)
        val manager = TrainingManager(
            context = context,
            neuralEngine = ArgusNeuralEngine.createBlocking(
                context,
                EngineConfig(maxSeqLen = 64)
            ),
            deviceGuard = DeviceGuard(context),
            worldModel = WorldModel(context),
            repository = repository
        )

        try {
            manager.pauseTraining()
            val validRecord = """{"instruction":"first prompt","response":"first answer"}"""
            val malformedRecord = """{"instruction":"second prompt","response":}"""
            val failure = runCatching {
                manager.enqueueDocumentStream(
                    source = "Explicit JSONL malformed-record regression",
                    title = "Must be rejected atomically",
                    reader = StringReader("$validRecord\n$malformedRecord"),
                    chunkSize = 512,
                    overlap = 0,
                    strictJsonLines = true
                )
            }.exceptionOrNull()

            assertTrue(
                "Explicit JSONL imports must reject malformed records.",
                failure is IllegalArgumentException
            )
            assertTrue(
                "The error must identify the malformed source line.",
                failure?.message.orEmpty().contains("line 2")
            )
            assertTrue(
                "A valid prefix must not be persisted when a later record is malformed.",
                repository.getCorpusDocuments().isEmpty()
            )
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())
            assertTrue(
                "The staging file must be deleted after strict JSONL rejection.",
                context.cacheDir.listFiles().orEmpty().none { file ->
                    file.name.startsWith("argus-ingest-")
                }
            )
        } finally {
            manager.close()
            database.close()
        }
    }

    @Test
    fun streamedPlainTextConversationIsProcessedOnlyAfterTheCompleteTranscriptArrives() = runTest {
        val context: Context = ApplicationProvider.getApplicationContext()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            ArgusDatabase::class.java
        ).allowMainThreadQueries().build()
        val repository = ArgusRepository(context, database)
        val manager = TrainingManager(
            context = context,
            neuralEngine = ArgusNeuralEngine.createBlocking(
                context,
                EngineConfig(maxSeqLen = 64)
            ),
            deviceGuard = DeviceGuard(context),
            worldModel = WorldModel(context),
            repository = repository
        )

        try {
            manager.pauseTraining()
            val firstUserTurn = "First long question " + "context ".repeat(90)
            val firstAssistantTurn = "First complete answer " + "detail ".repeat(70)
            val secondUserTurn = "Second question must remain a distinct turn."
            val secondAssistantTurn = "Second answer must also remain a distinct turn."
            val transcript = listOf(
                "User: $firstUserTurn",
                "Assistant: $firstAssistantTurn",
                "User: $secondUserTurn",
                "Assistant: $secondAssistantTurn"
            ).joinToString("\n")

            val (queuedCount, charactersRead) = manager.enqueueDocumentStream(
                source = "Long plain-text SFT transcript",
                title = "Complete streamed conversation",
                reader = StringReader(transcript),
                chunkSize = 48,
                overlap = 0
            )

            assertEquals(transcript.length.toLong(), charactersRead)
            assertEquals("A complete conversation remains one atomic SFT sample.", 1, queuedCount)
            manager.awaitQueuePersistenceIdle()

            val conversationDocument = repository.getCorpusDocuments()
                .single { it.title == "Complete streamed conversation" }
            val storedChunk = repository.getCorpusChunks(conversationDocument.id).single().content
            assertTrue(storedChunk.startsWith(com.argus.agi.model.Tokenizer.USER_TOKEN))
            assertEquals(
                2,
                Regex(Regex.escape(com.argus.agi.model.Tokenizer.USER_TOKEN))
                    .findAll(storedChunk)
                    .count()
            )
            assertEquals(
                2,
                Regex(Regex.escape(com.argus.agi.model.Tokenizer.ASSISTANT_TOKEN))
                    .findAll(storedChunk)
                    .count()
            )
            assertEquals(
                2,
                Regex(Regex.escape(com.argus.agi.model.Tokenizer.EOS_TOKEN))
                    .findAll(storedChunk)
                    .count()
            )
            assertTrue(storedChunk.contains(firstUserTurn))
            assertTrue(storedChunk.contains(firstAssistantTurn))
            assertTrue(storedChunk.contains(secondUserTurn))
            assertTrue(storedChunk.contains(secondAssistantTurn))

            val jsonl = listOf(
                """{"instruction":"first prompt","response":"first answer"}""",
                """{"instruction":"second prompt","response":"second answer"}"""
            ).joinToString("\n")
            val (jsonlQueued, _) = manager.enqueueDocumentStream(
                source = "JSONL per-record regression",
                title = "JSONL records remain atomic",
                reader = StringReader(jsonl),
                chunkSize = 8,
                overlap = 0,
                strictJsonLines = true
            )
            assertEquals("Each JSONL source record must remain its own sample.", 2, jsonlQueued)
            manager.awaitQueuePersistenceIdle()
            val jsonlDocument = repository.getCorpusDocuments()
                .single { it.title == "JSONL records remain atomic" }
            val jsonlChunks = repository.getCorpusChunks(jsonlDocument.id)
            assertEquals(2, jsonlChunks.size)
            assertTrue(jsonlChunks[0].content.contains("first prompt"))
            assertTrue(jsonlChunks[1].content.contains("second prompt"))
        } finally {
            manager.close()
            database.close()
        }
    }

    @Test
    fun retainedCorpusSurvivesQueueClearAndCanBeReplayed() = runTest {
        val context: Context = ApplicationProvider.getApplicationContext()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            ArgusDatabase::class.java
        ).allowMainThreadQueries().build()
        val repository = ArgusRepository(context, database)
        val engine = ArgusNeuralEngine.createBlocking(context)
        val manager = TrainingManager(
            context = context,
            neuralEngine = engine,
            deviceGuard = DeviceGuard(context),
            worldModel = WorldModel(context),
            repository = repository
        )

        try {
            manager.pauseTraining()
            val text = (
                "Argus retains imported documents independently from the active training queue. " +
                    "Clearing pending work must not destroy the source chunks required for replay. "
                ).repeat(18)

            val queuedCount = manager.enqueueDocument(
                source = "Corpus persistence test",
                title = "Retained document",
                fullText = text,
                chunkSize = 180,
                overlap = 30,
                epochs = 2
            )

            assertTrue("Initial import must schedule training chunks.", queuedCount > 0)
            manager.awaitQueuePersistenceIdle()

            val documents = repository.getCorpusDocuments()
            assertEquals("Exactly one corpus document must be retained.", 1, documents.size)
            val document = documents.single()
            val storedChunks = repository.getCorpusChunks(document.id)
            assertEquals(document.chunkCount, storedChunks.size)
            assertEquals(1, repository.getCorpusSources(document.id).size)
            assertEquals(queuedCount, repository.getPendingQueueSnapshot(limit = 128).size)

            val duplicateCount = manager.enqueueDocument(
                source = "Second import source",
                title = "Same text under another title",
                fullText = text,
                chunkSize = 180,
                overlap = 30,
                epochs = 1
            )
            assertEquals("Identical content must not be queued twice.", 0, duplicateCount)
            assertEquals(1, repository.getCorpusDocumentCount())
            assertEquals(2, repository.getCorpusSources(document.id).size)

            val streamDuplicate = manager.enqueueDocumentStream(
                source = "Stream duplicate source",
                title = "Stream duplicate",
                reader = StringReader(text),
                chunkSize = 180,
                overlap = 30,
                epochs = 1
            )
            assertEquals("Stream and string ingestion must share the same fingerprint.", 0, streamDuplicate.first)
            assertEquals(1, repository.getCorpusDocumentCount())
            assertEquals(3, repository.getCorpusSources(document.id).size)

            val unverifiedArticleText =
                "This unique article has not received a rights declaration. ".repeat(8)
            val unverifiedArticleImport = runCatching {
                manager.enqueueDocument(
                    source = "web_article|source=unreviewed-page",
                    title = "Unreviewed article",
                    fullText = unverifiedArticleText,
                    canonicalUrl = "https://example.org/unreviewed",
                    rightsStatus = "UNVERIFIED"
                )
            }.exceptionOrNull()
            assertTrue(
                "A unique web article without a rights declaration must be rejected.",
                unverifiedArticleImport != null
            )
            val unverifiedStreamImport = runCatching {
                manager.enqueueDocumentStream(
                    source = "web_article|source=unreviewed-stream",
                    title = "Unreviewed stream article",
                    reader = StringReader("A separate unreviewed web stream. ".repeat(12)),
                    canonicalUrl = "https://example.org/unreviewed-stream",
                    rightsStatus = "UNVERIFIED"
                )
            }.exceptionOrNull()
            assertTrue(
                "An unverified web stream must be rejected before content ingestion.",
                unverifiedStreamImport != null
            )

            val unverifiedValidationImport = runCatching {
                manager.importValidationDocument(
                    source = "web_article|source=unreviewed-validation",
                    title = "Unreviewed validation article",
                    fullText = "A separate unreviewed held-out article. ".repeat(12),
                    canonicalUrl = "https://example.org/unreviewed-validation",
                    rightsStatus = "UNVERIFIED"
                )
            }.exceptionOrNull()
            assertTrue(
                "An unverified web article must also be rejected from held-out storage.",
                unverifiedValidationImport != null
            )
            assertEquals(1, repository.getCorpusDocumentCount())

            val directUnverifiedDocument = CorpusDocumentEntity(
                id = "direct-unverified-web-document",
                contentFingerprint = "direct-unverified-web-fingerprint",
                fingerprintAlgorithm = "SHA-256/UTF-16BE",
                source = "web_article|source=direct-import",
                title = "Direct unverified article",
                language = "en",
                canonicalUrl = "https://example.org/direct-unverified",
                characterCount = 24,
                chunkCount = 0,
                datasetSplit = "VALIDATION",
                validationText = "Unverified web validation text",
                plannedEpochs = 0,
                rightsStatus = "UNVERIFIED",
                status = "READY",
                createdAtEpochMs = System.currentTimeMillis(),
                lastTrainedAtEpochMs = null,
                completedPassCount = 0
            )
            val directStorageFailure = runCatching {
                repository.storeCorpusDocument(directUnverifiedDocument, emptyList())
            }.exceptionOrNull()
            assertTrue(
                "The repository must reject unverified web content even when the manager is bypassed.",
                directStorageFailure != null
            )

            val declaredArticleImport = manager.enqueueDocument(
                source = "web_article|source=reviewed-page|rights_declared_by_user=true",
                title = "Article source",
                fullText = text,
                canonicalUrl = "https://example.org/article",
                rightsStatus = "USER_DECLARED",
                sourceProvenance = CorpusSourceProvenance(
                    userDeclarationText = "I confirm that I have the right to use this article content for local on-device training.",
                    userDeclaredAtEpochMs = System.currentTimeMillis()
                )
            )
            assertEquals(0, declaredArticleImport)

            val articleSource = repository.getCorpusSources(document.id)
                .single { it.canonicalUrl == "https://example.org/article" }
            assertEquals("USER_DECLARED", articleSource.rightsStatus)
            assertTrue(articleSource.userDeclarationText?.isNotBlank() == true)
            assertTrue((articleSource.userDeclaredAtEpochMs ?: 0L) > 0L)
            assertEquals(4, repository.getCorpusSources(document.id).size)
            assertEquals(
                "Document rights must be recomputed from all durable source declarations.",
                "USER_DECLARED",
                repository.getCorpusDocument(document.id)?.rightsStatus
            )

            val reviewedText = "A reviewed license must retain durable provenance. ".repeat(10)
            val reviewedAt = System.currentTimeMillis()
            val reviewedChunks = manager.enqueueDocument(
                source = "offline_public_domain",
                title = "Reviewed license fixture",
                fullText = reviewedText,
                language = "en",
                rightsStatus = "LICENSE_REVIEWED",
                sourceProvenance = CorpusSourceProvenance(
                    licenseIdentifier = "CC-BY-4.0",
                    licenseName = "Creative Commons Attribution 4.0 International",
                    licenseUrl = "https://creativecommons.org/licenses/by/4.0/",
                    attributionText = "Attribution text preserved for the original source.",
                    rightsEvidenceUrl = "https://example.org/source/license-evidence",
                    reviewedAtEpochMs = reviewedAt
                )
            )
            assertTrue(reviewedChunks > 0)
            val reviewedDocument = repository.getCorpusDocuments()
                .single { it.title == "Reviewed license fixture" }
            val reviewedSource = repository.getCorpusSources(reviewedDocument.id).single()
            assertEquals("LICENSE_REVIEWED", reviewedSource.rightsStatus)
            assertEquals("CC-BY-4.0", reviewedSource.licenseIdentifier)
            assertEquals("Creative Commons Attribution 4.0 International", reviewedSource.licenseName)
            assertEquals("https://creativecommons.org/licenses/by/4.0/", reviewedSource.licenseUrl)
            assertEquals("Attribution text preserved for the original source.", reviewedSource.attributionText)
            assertEquals("https://example.org/source/license-evidence", reviewedSource.rightsEvidenceUrl)
            assertEquals(reviewedAt, reviewedSource.reviewedAtEpochMs)

            val missingLicenseFailure = runCatching {
                manager.enqueueDocument(
                    source = "offline_public_domain",
                    title = "Missing license evidence",
                    fullText = "This content must not be accepted as reviewed without evidence. ".repeat(4),
                    rightsStatus = "LICENSE_REVIEWED"
                )
            }.exceptionOrNull()
            assertTrue("LICENSE_REVIEWED must require complete auditable license metadata.", missingLicenseFailure != null)

            val unverifiedReimport = runCatching {
                manager.enqueueDocument(
                    source = "web_article|source=reviewed-page",
                    title = "Article source",
                    fullText = text,
                    canonicalUrl = "https://example.org/article",
                    rightsStatus = "UNVERIFIED"
                )
            }.exceptionOrNull()
            assertTrue(
                "A later unverified import must not overwrite an existing user declaration.",
                unverifiedReimport != null
            )
            val sourceAfterUnverifiedReimport = repository.getCorpusSources(document.id)
                .single { it.canonicalUrl == "https://example.org/article" }
            assertEquals(
                "A less authoritative repeat import must not downgrade a prior rights declaration.",
                "USER_DECLARED",
                sourceAfterUnverifiedReimport.rightsStatus
            )

            val activeQueueItem = repository.getPendingQueueSnapshot().first()
            assertTrue(activeQueueItem.chunkIndex < activeQueueItem.totalChunks)
            assertTrue(repository.markTrainingStarted(activeQueueItem.id))

            manager.clearQueue()
            manager.awaitQueuePersistenceIdle()

            assertTrue(
                "Queue clearing must remove waiting work.",
                repository.getPendingQueueSnapshot().isEmpty()
            )
            assertEquals(
                "An already active training row must survive queue clearing.",
                com.argus.agi.model.TrainingStatus.TRAINING,
                repository.getTrainingItem(activeQueueItem.id)?.status
            )
            assertEquals(
                "A document with remaining chunks removed from the queue must become partial.",
                "PARTIAL",
                repository.getCorpusDocument(document.id)?.status
            )
            repository.markCorpusDocumentTraining(document.id)
            assertEquals(
                "A late worker callback must not overwrite an interrupted partial state.",
                "PARTIAL",
                repository.getCorpusDocument(document.id)?.status
            )

            assertEquals(
                "Interrupted active training rows must be re-queued on recovery.",
                1,
                repository.recoverInterruptedTrainingItems()
            )
            val recoveredItems = repository.getPendingQueueSnapshot()
            assertEquals(1, recoveredItems.size)
            assertEquals(activeQueueItem.id, recoveredItems.single().id)

            manager.clearQueue()
            manager.awaitQueuePersistenceIdle()
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())

            assertEquals(
                "Corpus data must survive queue clearing.",
                document.chunkCount,
                repository.getCorpusChunks(document.id).size
            )

            val replayedCount = manager.replayCorpusDocument(document.id, epochs = 2)
            assertEquals(document.chunkCount * 2, replayedCount)
            manager.awaitQueuePersistenceIdle()

            val replayedItems = repository.getPendingQueueSnapshot(limit = 128)
            assertEquals(replayedCount, replayedItems.size)
            assertTrue(replayedItems.all { it.documentId == document.id })
            assertEquals(
                "Replay must preserve canonical corpus chunk text.",
                storedChunks.first().content,
                replayedItems.first().content
            )
        } finally {
            manager.close()
            database.close()
        }
    }

    @Test
    fun clearingQueueKeepsAnActiveFinalChunkEligibleForCompletion() = runTest {
        val context: Context = ApplicationProvider.getApplicationContext()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            ArgusDatabase::class.java
        ).allowMainThreadQueries().build()
        val repository = ArgusRepository(context, database)
        val manager = TrainingManager(
            context = context,
            neuralEngine = ArgusNeuralEngine.createBlocking(context),
            deviceGuard = DeviceGuard(context),
            worldModel = WorldModel(context),
            repository = repository
        )

        try {
            manager.pauseTraining()
            val queuedCount = manager.enqueueDocument(
                source = "Queue clear final-chunk test",
                title = "Single chunk document",
                fullText = "A single retained source chunk remains eligible for completion.",
                chunkSize = 180,
                overlap = 30,
                epochs = 1
            )
            assertEquals(1, queuedCount)
            manager.awaitQueuePersistenceIdle()

            val document = repository.getCorpusDocuments().single()
            val activeItem = repository.getPendingQueueSnapshot().single()
            assertEquals(1, activeItem.chunkIndex)
            assertEquals(1, activeItem.totalChunks)
            assertTrue(repository.markTrainingStarted(activeItem.id))
            repository.markCorpusDocumentTraining(document.id)

            manager.clearQueue()
            manager.awaitQueuePersistenceIdle()

            assertTrue(repository.getPendingQueueSnapshot().isEmpty())
            assertEquals(
                "A sole active final chunk must remain eligible to complete its document.",
                "TRAINING",
                repository.getCorpusDocument(document.id)?.status
            )

            repository.markTrainingCompleted(activeItem.id, loss = 0.1f, durationMs = 1L)
            repository.markCorpusDocumentTrained(document.id, completedPassCount = 1)
            assertEquals("TRAINED", repository.getCorpusDocument(document.id)?.status)
        } finally {
            manager.close()
            database.close()
        }
    }

    @Test
    fun genesisFoundationCorpusIsPersistentAndRecoversWithoutDuplicates() = runTest {
        val context: Context = ApplicationProvider.getApplicationContext()
        val database = Room.inMemoryDatabaseBuilder(
            context,
            ArgusDatabase::class.java
        ).allowMainThreadQueries().build()
        val repository = ArgusRepository(context, database)
        val engine = ArgusNeuralEngine.createBlocking(
            context,
            EngineConfig(
                dModel = 48,
                nHeads = 4,
                dFf = 128,
                maxSeqLen = 32,
                loraRank = 2
            )
        )
        val deviceGuard = DeviceGuard(context)
        val trainingManager = TrainingManager(
            context = context,
            neuralEngine = engine,
            deviceGuard = deviceGuard,
            worldModel = WorldModel(context),
            repository = repository
        )
        val genesisManager = com.argus.agi.model.GenesisModelManager(
            context = context,
            neuralEngine = engine,
            deviceGuard = deviceGuard
        )

        try {
            trainingManager.pauseTraining()

            val initialScheduledChunks =
                genesisManager.ensureGeneralLearningCorpus(trainingManager)
            assertEquals("The three built-in foundation notes must be scheduled.", 3, initialScheduledChunks)
            trainingManager.awaitQueuePersistenceIdle()

            val initialDocuments = repository.getTrainingCorpusDocuments()
            assertEquals("Each Genesis note must be retained as its own corpus document.", 3, initialDocuments.size)
            assertTrue(initialDocuments.all { it.source == "Genesis Foundation" })
            assertTrue(initialDocuments.all { it.datasetSplit == "TRAIN" })
            assertEquals(3, repository.getPendingQueueSnapshot().size)

            val duplicateScheduledChunks =
                genesisManager.ensureGeneralLearningCorpus(trainingManager)
            assertEquals("Already queued Genesis notes must not be scheduled twice.", 0, duplicateScheduledChunks)
            assertEquals(3, repository.getCorpusDocumentCount())

            trainingManager.clearQueue()
            trainingManager.awaitQueuePersistenceIdle()
            initialDocuments.forEach { document ->
                assertEquals(
                    "Clearing waiting seed work must retain its corpus and mark it partial.",
                    "PARTIAL",
                    repository.getCorpusDocument(document.id)?.status
                )
            }
            assertTrue(repository.getPendingQueueSnapshot().isEmpty())

            val recoveredScheduledChunks =
                genesisManager.ensureGeneralLearningCorpus(trainingManager)
            assertEquals("Interrupted Genesis seed work must replay from Room.", 3, recoveredScheduledChunks)
            trainingManager.awaitQueuePersistenceIdle()

            val recoveredQueue = repository.getPendingQueueSnapshot()
            assertEquals(3, recoveredQueue.size)
            assertEquals(3, repository.getCorpusDocumentCount())
            assertEquals(
                "Replay must reuse the retained Genesis corpus documents.",
                initialDocuments.map { it.id }.toSet(),
                recoveredQueue.mapNotNull { it.documentId }.toSet()
            )

            val repeatedRecovery =
                genesisManager.ensureGeneralLearningCorpus(trainingManager)
            assertEquals("Active replay work must remain idempotent.", 0, repeatedRecovery)
            assertEquals(3, repository.getPendingQueueSnapshot().size)
        } finally {
            trainingManager.close()
            database.close()
        }
    }

}
