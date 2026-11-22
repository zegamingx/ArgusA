package com.argus.agi.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.defaultMinSize
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.argus.agi.model.ConversationValidationState
import com.argus.agi.model.ConversationValidationStatus
import com.argus.agi.model.DeviceModelCapacityPolicy
import com.argus.agi.model.TrainingData
import com.argus.agi.model.TrainingStatus
import com.argus.agi.sensor.DeviceSafetyState
import com.argus.agi.ui.ArgusViewModel
import com.argus.agi.ui.theme.ArgusCyanGlow
import com.argus.agi.ui.theme.ArgusCyanPrimary
import com.argus.agi.ui.theme.ArgusDarkBg
import com.argus.agi.ui.theme.ArgusError
import com.argus.agi.ui.theme.ArgusSuccess
import com.argus.agi.ui.theme.ArgusTextMuted
import com.argus.agi.ui.theme.ArgusTextPrimary
import com.argus.agi.ui.theme.ArgusTextSecondary
import com.argus.agi.ui.theme.ArgusWarning
import java.text.NumberFormat
import java.util.Locale

// =====================================================================
// NEURAL CORE SCREEN
// =====================================================================

@Composable
fun NeuralCoreScreen(viewModel: ArgusViewModel) {
    LaunchedEffect(viewModel) {
        viewModel.refreshDeviceMemorySnapshot()
    }

    val totalParams = viewModel.getTotalParameters()
    val adapterBytes = viewModel.getAdapterSizeBytes()
    val isPaused by viewModel.isTrainingPaused.collectAsState()
    val safetyState by viewModel.safetyState.collectAsState()
    val hardwareInfo by viewModel.hardwareInfo.collectAsState()
    val inferenceCapacity = remember(hardwareInfo.availRamGb) {
        DeviceModelCapacityPolicy.recommend(hardwareInfo.availRamGb.toDouble())
    }
    val currentItem by viewModel.currentTrainingItem.collectAsState()
    val nextQueue by viewModel.nextQueue.collectAsState()
    val completedQueue by viewModel.completedQueue.collectAsState()
    val lossHistory by viewModel.lossHistory.collectAsState()
    val totalSteps by viewModel.totalStepsExecuted.collectAsState()
    val trainingMetrics by viewModel.trainingMetrics.collectAsState()
    val activeDoc by viewModel.activeDocument.collectAsState()
    val tokenizerTraining by viewModel.tokenizerTrainingProgress.collectAsState()
    val conversationalValidation by viewModel.conversationalValidationState.collectAsState()
    val vocabSize = trainingMetrics.vocabularySize
    val worldEntities by viewModel.worldEntities.collectAsState()
    val modelConfig = viewModel.engineConfig
    val backendState = viewModel.activeComputeBackend
    val computeBackendStatus by viewModel.computeBackendStatus.collectAsState()
    val activeDocument = activeDoc
    val activeTrainingItem = currentItem

    val trainingEtaMs = remainingTrainingEtaMs(
        activeTrainingItem,
        nextQueue,
        completedQueue,
        safetyState,
        isPaused
    )
    val trainingEtaLabel = when {
        isPaused -> "Paused"
        !safetyState.isSafeToTrain -> "Blocked by device safety"
        trainingEtaMs == null -> "Calculating"
        trainingEtaMs == 0L -> "Queue empty"
        else -> "~${formatEta(trainingEtaMs)}"
    }
    val trainingStatusLabel = when {
        isPaused -> "TRAINING PAUSED"
        !safetyState.isSafeToTrain -> "TRAINING BLOCKED"
        activeTrainingItem != null -> "TRAINING IN PROGRESS"
        nextQueue.isNotEmpty() -> "TRAINING QUEUED"
        else -> "TRAINING IDLE"
    }
    val trainingIndicatorColor = when {
        isPaused || !safetyState.isSafeToTrain -> ArgusWarning
        activeTrainingItem != null -> ArgusSuccess
        else -> ArgusTextMuted
    }

    val formattedParams = NumberFormat.getNumberInstance(Locale.US).format(totalParams)
    val adapterSizeKb = adapterBytes / 1024.0
    val adapterSizeStr = if (adapterBytes > 0) {
        String.format(Locale.US, "%.2f KB", adapterSizeKb)
    } else {
        "Not available"
    }
    val lastLoss = lossHistory.lastOrNull()?.let { String.format(Locale.US, "%.4f", it) } ?: "--"
    val inputTokensLabel = NumberFormat.getNumberInstance(Locale.US).format(trainingMetrics.inputTokens)
    val predictionTargetsLabel = NumberFormat.getNumberInstance(Locale.US).format(trainingMetrics.predictionTargets)
    val contextWindowsLabel = NumberFormat.getNumberInstance(Locale.US).format(trainingMetrics.contextWindows)
    val optimizerUpdatesLabel = NumberFormat.getNumberInstance(Locale.US).format(trainingMetrics.optimizerUpdates)
    val reasoningExamplesLabel = NumberFormat.getNumberInstance(Locale.US).format(trainingMetrics.reasoningExamples)
    val reasoningTokensLabel = NumberFormat.getNumberInstance(Locale.US).format(trainingMetrics.reasoningTokens)
    val memoryRecordsLabel = NumberFormat.getNumberInstance(Locale.US).format(worldEntities.size)

    // Only completed chunks contribute to document progress.
    val docProgress = activeDocument?.let { document ->
        if (document.totalChunks > 0) {
            document.completedChunks.toFloat() / document.totalChunks.toFloat()
        } else {
            0f
        }
    } ?: 0f
    val docPct = (docProgress * 100).toInt()

    Box(
        modifier = Modifier
            .fillMaxSize()
            .testTag("neural_core_screen")
    ) {

        // Main screen content
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp),
            contentPadding = PaddingValues(vertical = 16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // HEADER
            item {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    ArgusEyeGlow()
                    Spacer(modifier = Modifier.width(12.dp))
                    Column {
                        Text(
                            text = "A.R.G.U.S",
                            style = MaterialTheme.typography.headlineMedium,
                            color = Color(0xFF8FE3FF),
                            fontWeight = FontWeight.ExtraBold,
                            lineHeight = 28.sp
                        )
                        Text(
                            text = "TRAINING",
                            style = MaterialTheme.typography.headlineMedium,
                            color = Color(0xFF8FE3FF),
                            fontWeight = FontWeight.ExtraBold,
                            lineHeight = 28.sp
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "AUTONOMOUS REASONING • GLOBAL UNDERSTANDING SYSTEM",
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusCyanGlow,
                            fontSize = 10.sp
                        )
                        Text(
                            text = "Local Model Training • Device-Adaptive Architecture",
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusTextSecondary,
                            fontSize = 10.sp
                        )
                    }
                }
            }

            // TRAINING STATUS
            item {
                Box(
                    Modifier
                        .fillMaxWidth()
                        .background(Color.White.copy(alpha = 0.04f), RoundedCornerShape(18.dp))
                        .padding(14.dp)
                ) {
                    Column {
                        Row(
                            Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = trainingStatusLabel,
                                style = MaterialTheme.typography.titleSmall,
                                color = ArgusCyanGlow,
                                fontWeight = FontWeight.Bold
                            )
                            Box(
                                Modifier
                                    .size(10.dp)
                                    .background(trainingIndicatorColor, CircleShape)
                            )
                        }
                        Spacer(modifier = Modifier.height(10.dp))
                        LinearProgressIndicator(
                            progress = { docProgress },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(6.dp)
                                .clip(RoundedCornerShape(3.dp)),
                            color = ArgusCyanPrimary,
                            trackColor = Color.White.copy(alpha = 0.08f)
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text(
                                text = activeDocument?.let { "Completed ${it.completedChunks} of ${it.totalChunks} chunks • $docPct%" }
                                    ?: if (activeTrainingItem != null) "Processing a single training item"
                                    else "No active training document",
                                style = MaterialTheme.typography.labelSmall,
                                color = ArgusTextPrimary,
                                fontFamily = FontFamily.Monospace
                            )
                            Text(
                                text = "$docPct%",
                                style = MaterialTheme.typography.labelSmall,
                                color = ArgusCyanGlow,
                                fontWeight = FontWeight.Bold,
                                fontFamily = FontFamily.Monospace
                            )
                        }
                    }
                }
            }

            // TRAINING METRICS
            item {
                Column {
                    Text(
                        text = "TRAINING METRICS",
                        style = MaterialTheme.typography.titleSmall,
                        color = ArgusTextPrimary,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(10.dp))
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        StatMetricBox(
                            label = "LAST LOSS",
                            value = lastLoss,
                            unit = "latest completed chunk",
                            valueColor = ArgusCyanPrimary,
                            modifier = Modifier.weight(1f)
                        )
                        StatMetricBox(
                            label = "VOCABULARY",
                            value = "$vocabSize / ${trainingMetrics.vocabularyCapacity}",
                            unit = "learned tokens / capacity",
                            valueColor = ArgusCyanGlow,
                            modifier = Modifier.weight(1f)
                        )
                        StatMetricBox(
                            label = "INPUT TOKENS",
                            value = inputTokensLabel,
                            unit = "session total",
                            valueColor = ArgusSuccess,
                            modifier = Modifier.weight(1f)
                        )
                    }
                    Spacer(modifier = Modifier.height(10.dp))
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        StatMetricBox(
                            label = "CONTEXT WINDOWS",
                            value = contextWindowsLabel,
                            unit = "processed windows",
                            valueColor = ArgusCyanPrimary,
                            modifier = Modifier.weight(1f)
                        )
                        StatMetricBox(
                            label = "OPTIMIZER UPDATES",
                            value = optimizerUpdatesLabel,
                            unit = "weight updates",
                            valueColor = ArgusCyanGlow,
                            modifier = Modifier.weight(1f)
                        )
                        StatMetricBox(
                            label = "MEMORY RECORDS",
                            value = memoryRecordsLabel,
                            unit = "World Model entries",
                            valueColor = ArgusSuccess,
                            modifier = Modifier.weight(1f)
                        )
                    }
                    Spacer(modifier = Modifier.height(10.dp))
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        StatMetricBox(
                            label = "REASONING EXAMPLES",
                            value = reasoningExamplesLabel,
                            unit = "structured training samples",
                            valueColor = ArgusCyanPrimary,
                            modifier = Modifier.weight(1f)
                        )
                        StatMetricBox(
                            label = "REASONING TOKENS",
                            value = reasoningTokensLabel,
                            unit = "tokens inside reasoning spans",
                            valueColor = ArgusCyanGlow,
                            modifier = Modifier.weight(1f)
                        )
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "$predictionTargetsLabel next-token targets scored in this session. " +
                            "${trainingMetrics.completedDocuments} documents completed across " +
                            "${trainingMetrics.completedChunks} training chunks.",
                        style = MaterialTheme.typography.labelSmall,
                        color = ArgusTextMuted
                    )
                }
            }

            // CONVERSATIONAL QUALITY GATE
            item {
                val validationNeedsRetry = conversationalValidation.status == ConversationValidationStatus.FAILED &&
                    nextQueue.isNotEmpty()
                ConversationValidationCard(
                    state = conversationalValidation,
                    trainingIsActive = currentItem != null || nextQueue.isNotEmpty(),
                    onAction = {
                        if (validationNeedsRetry) {
                            viewModel.retryConversationalValidation()
                        } else {
                            viewModel.runConversationalValidation()
                        }
                    }
                )
            }

            // INFERENCE CAPACITY — informational target only; not the active training model.
            item {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Color.White.copy(alpha = 0.04f), RoundedCornerShape(18.dp))
                        .padding(14.dp)
                ) {
                    Text(
                        text = "INFERENCE CAPACITY ESTIMATE",
                        style = MaterialTheme.typography.titleSmall,
                        color = ArgusTextPrimary,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Available system RAM: " +
                            String.format(Locale.US, "%.1f GB", hardwareInfo.availRamGb),
                        style = MaterialTheme.typography.labelSmall,
                        color = ArgusTextSecondary,
                        fontFamily = FontFamily.Monospace
                    )
                    Text(
                        text = inferenceCapacity.parameterRangeLabel,
                        style = MaterialTheme.typography.headlineSmall,
                        color = ArgusCyanGlow,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = "RAM-only Argus INT4 planning range. The active model remains the separate " +
                            "FP16/FP32 training engine. A model-specific memory and backend check is required " +
                            "before any quantized model can be loaded.",
                        style = MaterialTheme.typography.bodySmall,
                        color = ArgusTextMuted
                    )
                }
            }

            // TRAINING CONTROL
            item {
                Box(
                    Modifier
                        .fillMaxWidth()
                        .background(Color.White.copy(alpha = 0.04f), RoundedCornerShape(18.dp))
                        .padding(14.dp)
                ) {
                    Column {
                        Row(
                            Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(Modifier.weight(1f)) {
                                Text(
                                    text = "Background Training Worker",
                                    style = MaterialTheme.typography.titleMedium,
                                    color = ArgusTextPrimary,
                                    fontWeight = FontWeight.Bold
                                )
                                Text(
                                    text = "Completed items: $totalSteps • Estimated remaining: $trainingEtaLabel",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = if (isPaused || !safetyState.isSafeToTrain) ArgusWarning else ArgusCyanGlow,
                                    fontFamily = FontFamily.Monospace
                                )
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = safetyState.statusMessage,
                                    style = MaterialTheme.typography.labelSmall,
                                    color = ArgusTextSecondary,
                                    fontSize = 11.sp
                                )
                            }
                            Spacer(modifier = Modifier.width(10.dp))
                            Button(
                                onClick = { viewModel.toggleTrainingPause() },
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = if (isPaused) ArgusSuccess else ArgusWarning,
                                    contentColor = ArgusDarkBg
                                ),
                                shape = RoundedCornerShape(12.dp),
                                modifier = Modifier.testTag("toggle_training_pause_button")
                            ) {
                                Text(text = if (isPaused) "Resume" else "Pause", fontWeight = FontWeight.Bold, fontSize = 12.sp)
                            }
                        }
                        Spacer(modifier = Modifier.height(10.dp))
                        Text(
                            text = "Architecture: Argus Transformer · ${modelConfig.layerCount} layers · dModel=${modelConfig.dModel} · Heads=${modelConfig.nHeads} · FFN=${modelConfig.dFf} · Context=${modelConfig.maxSeqLen} · LoRA rank=${modelConfig.loraRank}",
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusTextMuted,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 10.sp
                        )
                        Text(
                            text = "Model profile: ${modelConfig.modelTier.displayName} • Tier ceiling: ${formatParameterScale(modelConfig.modelTier.maximumFamilyParameters)} • Active: $formattedParams • Batch: ${modelConfig.trainingBatchSize}/8 • LoRA: $adapterSizeStr",
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusTextMuted,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 10.sp
                        )
                        Text(
                            text = "Optimizer backend: $backendState",
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusTextMuted,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 10.sp
                        )
                        Text(
                            text = computeBackendStatus,
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusTextSecondary,
                            fontSize = 10.sp
                        )
                        Text(
                            text = "Vulkan is prepared after model restoration. Unsupported or unverified runtimes use Kotlin CPU.",
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusTextMuted,
                            fontSize = 10.sp
                        )
                    }
                }
            }

            // TRAINING ACTIVITY
            item {
                Box(
                    Modifier
                        .fillMaxWidth()
                        .background(Color.White.copy(alpha = 0.04f), RoundedCornerShape(18.dp))
                        .padding(14.dp)
                ) {
                    Column {
                        Text(text = "TRAINING ACTIVITY", style = MaterialTheme.typography.titleSmall, color = ArgusTextPrimary, fontWeight = FontWeight.Bold)
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "> Status: $trainingStatusLabel",
                            style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                            color = ArgusTextSecondary,
                            fontFamily = FontFamily.Monospace
                        )
                        Text(
                            text = "> Current item: ${activeTrainingItem?.let { displayTrainingItemTitle(it.title).take(60) } ?: "None"}",
                            style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                            color = ArgusTextSecondary,
                            fontFamily = FontFamily.Monospace
                        )
                        Text(
                            text = "> Source: ${activeTrainingItem?.source ?: "—"} • Loss: $lastLoss",
                            style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                            color = ArgusTextSecondary,
                            fontFamily = FontFamily.Monospace
                        )
                        activeDocument?.let { document ->
                            Text(
                                text = "> Active corpus: ${document.docTitle}",
                                style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                color = ArgusCyanGlow,
                                fontFamily = FontFamily.Monospace
                            )
                        }
                        Text(
                            text = "> Tokenizer: ${modelConfig.tokenizerConfig.algorithm.name} • ${vocabSize}/${trainingMetrics.vocabularyCapacity} tokens • +${trainingMetrics.vocabularyTokensAdded} this session • ${trainingMetrics.tokenizerMerges} BPE rules",
                            style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                            color = ArgusCyanGlow,
                            fontFamily = FontFamily.Monospace
                        )
                        tokenizerTraining?.let { status ->
                            val bpe = status.progress
                            val bpeColor = when (bpe.stage) {
                                com.argus.agi.model.TokenizerTrainingStage.COMPLETED -> ArgusSuccess
                                com.argus.agi.model.TokenizerTrainingStage.FAILED,
                                com.argus.agi.model.TokenizerTrainingStage.CANCELLED -> ArgusWarning
                                else -> ArgusCyanPrimary
                            }
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .background(Color.White.copy(alpha = 0.035f), RoundedCornerShape(10.dp))
                                    .padding(10.dp)
                            ) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(
                                        text = "BPE TOKENIZER • ${status.documentTitle}",
                                        style = MaterialTheme.typography.labelSmall,
                                        color = bpeColor,
                                        fontWeight = FontWeight.SemiBold,
                                        maxLines = 1
                                    )
                                    Text(
                                        text = "${(bpe.fraction * 100f).toInt()}%",
                                        style = MaterialTheme.typography.labelSmall,
                                        color = bpeColor,
                                        fontFamily = FontFamily.Monospace
                                    )
                                }
                                Text(
                                    text = bpe.stage.displayName,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = ArgusTextSecondary
                                )
                                Spacer(modifier = Modifier.height(6.dp))
                                LinearProgressIndicator(
                                    progress = { bpe.fraction.coerceIn(0f, 1f) },
                                    modifier = Modifier.fillMaxWidth().height(4.dp),
                                    color = bpeColor
                                )
                                Text(
                                    text = "${bpe.learnedMerges}/${bpe.targetMerges} new merge rules • Vocabulary ${bpe.vocabularySize}",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = ArgusTextMuted,
                                    fontFamily = FontFamily.Monospace
                                )
                                status.failureMessage?.let { message ->
                                    Text(
                                        text = message,
                                        style = MaterialTheme.typography.bodySmall,
                                        color = ArgusWarning
                                    )
                                }
                            }
                        }
                        Text(
                            text = "> Learning totals: $inputTokensLabel input tokens • $contextWindowsLabel context windows • $optimizerUpdatesLabel optimizer updates",
                            style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                            color = ArgusTextSecondary,
                            fontFamily = FontFamily.Monospace
                        )
                        Text(
                            text = "> Checkpoint cadence: every ${viewModel.checkpointInterval} optimizer updates",
                            style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                            color = ArgusSuccess,
                            fontFamily = FontFamily.Monospace
                        )
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(
                            text = "DEVICE SAFETY: ${if (safetyState.isSafeToTrain) "CLEAR" else "HOLD"} • $trainingStatusLabel",
                            style = MaterialTheme.typography.labelSmall,
                            color = if (safetyState.isSafeToTrain && !isPaused) ArgusSuccess else ArgusWarning,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }
            }

            // TRAINING QUEUE
            item {
                Text(text = "TRAINING QUEUE & RECENT ACTIVITY", style = MaterialTheme.typography.titleMedium, color = ArgusCyanPrimary, fontWeight = FontWeight.Bold)
            }

            item {
                QueueSectionHeader(title = "1. Currently Processing", count = if (activeTrainingItem != null) 1 else 0)
                Spacer(modifier = Modifier.height(8.dp))
                if (activeTrainingItem != null) ActiveTrainingCard(activeTrainingItem)
                else EmptyQueueCard("No training item is currently active.")
            }

            item {
                QueueSectionHeader(
                    title = "2. Next in Queue",
                    count = nextQueue.size,
                    clearTestTag = "clear_queue_button",
                    onClear = { viewModel.clearTrainingQueue() }
                )
                Spacer(modifier = Modifier.height(8.dp))
                if (nextQueue.isNotEmpty()) {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        nextQueue.take(5).forEach { QueuedItemCard(it) }
                        if (nextQueue.size > 5) Text(text = "+ ${nextQueue.size - 5} more queued items", style = MaterialTheme.typography.labelSmall, color = ArgusTextMuted)
                    }
                } else EmptyQueueCard("The queue is empty. Add text or import a document from the Terminal.")
            }

            item {
                QueueSectionHeader(
                    title = "3. Recently Completed",
                    count = completedQueue.size,
                    clearTestTag = "clear_completed_button",
                    onClear = { viewModel.clearCompletedTraining() }
                )
                Spacer(modifier = Modifier.height(8.dp))
                if (completedQueue.isNotEmpty()) {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        completedQueue.take(5).forEach { CompletedItemCard(it) }
                        if (completedQueue.size > 5) Text(text = "+ ${completedQueue.size - 5} more completed items", style = MaterialTheme.typography.labelSmall, color = ArgusTextMuted)
                    }
                } else EmptyQueueCard("No completed training items yet.")
            }
        }
    }
}

// =====================================================================
// ARGUS EYE INDICATOR
// =====================================================================
@Composable
fun ArgusEyeGlow() {
    Box(modifier = Modifier.size(88.dp), contentAlignment = Alignment.Center) {
        Box(
            Modifier
                .size(88.dp)
                .background(Brush.radialGradient(colors = listOf(Color(0xFF00BFFF).copy(alpha = 0.45f), Color.Transparent)), CircleShape)
        )
        Box(
            Modifier
                .size(62.dp)
                .background(Brush.radialGradient(colors = listOf(Color(0xFF00E5FF), Color(0xFF0057FF).copy(alpha = 0.4f))), CircleShape)
        )
        Box(
            Modifier
                .size(32.dp)
                .background(Color(0xFFEAFBFF), CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Box(Modifier.size(14.dp).background(Color(0xFF0033FF), CircleShape))
        }
    }
}

// =====================================================================
// METRIC AND QUEUE COMPONENTS
// =====================================================================
@Composable
fun StatMetricBox(label: String, value: String, unit: String, valueColor: Color, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(12.dp))
            .background(Color.White.copy(alpha = 0.05f))
            .padding(10.dp)
    ) {
        Text(text = label, style = MaterialTheme.typography.labelSmall, color = ArgusTextMuted, fontSize = 9.sp)
        Spacer(modifier = Modifier.height(4.dp))
        Text(text = value, style = MaterialTheme.typography.titleMedium, color = valueColor, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace, fontSize = 14.sp)
        Text(text = unit, style = MaterialTheme.typography.labelSmall, color = ArgusTextSecondary, fontSize = 9.sp)
    }
}

@Composable
private fun ConversationValidationCard(
    state: ConversationValidationState,
    trainingIsActive: Boolean,
    onAction: () -> Unit
) {
    val report = state.latestReport ?: state.baselineReport
    val busy = state.status == ConversationValidationStatus.PREPARING_BASELINE ||
        state.status == ConversationValidationStatus.TRAINING ||
        state.status == ConversationValidationStatus.EVALUATING
    val retrying = state.status == ConversationValidationStatus.FAILED && trainingIsActive
    val statusColor = when (state.status) {
        ConversationValidationStatus.PROMOTED -> ArgusSuccess
        ConversationValidationStatus.PROMOTED_NOT_READY,
        ConversationValidationStatus.ROLLED_BACK,
        ConversationValidationStatus.FAILED -> ArgusWarning
        else -> ArgusCyanGlow
    }
    val score = report?.overallScore?.coerceIn(0f, 1f) ?: 0f
    val scoreLabel = report?.let {
        String.format(Locale.US, "%.0f%%", it.overallScore * 100f)
    } ?: "Not measured"

    Box(
        Modifier
            .fillMaxWidth()
            .background(Color.White.copy(alpha = 0.04f), RoundedCornerShape(18.dp))
            .padding(14.dp)
            .testTag("conversation_validation_card")
    ) {
        Column {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "CONVERSATIONAL QUALITY GATE",
                    style = MaterialTheme.typography.titleSmall,
                    color = ArgusTextPrimary,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = state.status.name.replace('_', ' '),
                    style = MaterialTheme.typography.labelSmall,
                    color = statusColor,
                    fontWeight = FontWeight.Bold
                )
            }
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = state.message,
                style = MaterialTheme.typography.bodySmall,
                color = ArgusTextSecondary
            )

            if (report != null) {
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "EVALUATION V${report.suiteVersion} · ${report.caseCount} CASES · ${report.turnCount} TURNS",
                    style = MaterialTheme.typography.labelSmall,
                    color = ArgusTextMuted,
                    fontFamily = FontFamily.Monospace
                )
                Spacer(modifier = Modifier.height(10.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "OVERALL CONVERSATION SCORE",
                        style = MaterialTheme.typography.labelSmall,
                        color = ArgusTextMuted,
                        fontWeight = FontWeight.SemiBold
                    )
                    Text(
                        text = scoreLabel,
                        style = MaterialTheme.typography.titleMedium,
                        color = statusColor,
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.Monospace
                    )
                }
                LinearProgressIndicator(
                    progress = { score },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(6.dp)
                        .clip(RoundedCornerShape(3.dp)),
                    color = statusColor,
                    trackColor = Color.White.copy(alpha = 0.08f)
                )
                Spacer(modifier = Modifier.height(10.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    StatMetricBox(
                        label = "FACT COVERAGE",
                        value = String.format(Locale.US, "%.0f%%", report.factCoverage * 100f),
                        unit = "${report.matchedFactGroupCount}/${report.expectedFactGroupCount} fact groups",
                        valueColor = ArgusCyanPrimary,
                        modifier = Modifier.weight(1f)
                    )
                    StatMetricBox(
                        label = "INSTRUCTIONS",
                        value = String.format(Locale.US, "%.0f%%", report.instructionCompliance * 100f),
                        unit = "${report.passedInstructionCheckCount}/${report.instructionCheckCount} checks",
                        valueColor = ArgusCyanGlow,
                        modifier = Modifier.weight(1f)
                    )
                }
                Spacer(modifier = Modifier.height(8.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    StatMetricBox(
                        label = "MULTI-TURN",
                        value = String.format(Locale.US, "%.0f%%", report.multiTurnConsistency * 100f),
                        unit = "${report.matchedFollowUpFactGroupCount}/${report.followUpFactGroupCount} follow-up facts",
                        valueColor = ArgusSuccess,
                        modifier = Modifier.weight(1f)
                    )
                    StatMetricBox(
                        label = "CONTRADICTIONS",
                        value = String.format(Locale.US, "%.0f%%", report.contradictionRate * 100f),
                        unit = "${report.matchedForbiddenClaimCount}/${report.forbiddenClaimCount} forbidden claims",
                        valueColor = if (report.contradictionRate > 0f) ArgusWarning else ArgusSuccess,
                        modifier = Modifier.weight(1f)
                    )
                }
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = if (report.meetsThresholds) {
                        "All absolute conversational quality thresholds are met."
                    } else {
                        "Best-known checkpoint is not release-ready. Remaining checks: ${report.failedThresholds.joinToString()}."
                    },
                    style = MaterialTheme.typography.labelSmall,
                    color = if (report.meetsThresholds) ArgusSuccess else ArgusWarning
                )
            } else {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "A fixed, held-out conversation suite measures factual coverage, instruction following, multi-turn recall, and explicit contradictions.",
                    style = MaterialTheme.typography.bodySmall,
                    color = ArgusTextMuted
                )
            }

            Spacer(modifier = Modifier.height(10.dp))
            Button(
                onClick = onAction,
                enabled = !busy && (!trainingIsActive || retrying),
                colors = ButtonDefaults.buttonColors(
                    containerColor = ArgusCyanPrimary,
                    contentColor = ArgusDarkBg,
                    disabledContainerColor = Color.White.copy(alpha = 0.08f),
                    disabledContentColor = ArgusTextMuted
                ),
                shape = RoundedCornerShape(12.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("run_conversation_validation_button")
            ) {
                Text(
                    text = when {
                        busy -> "Validation in progress..."
                        retrying -> "Retry validation before training"
                        trainingIsActive -> "Available after the queue drains"
                        else -> "Evaluate current model"
                    },
                    fontWeight = FontWeight.Bold
                )
            }
        }
    }
}

@Composable
fun QueueSectionHeader(
    title: String,
    count: Int,
    clearTestTag: String? = null,
    onClear: (() -> Unit)? = null
) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
        Text(text = title, style = MaterialTheme.typography.titleMedium.copy(fontSize = 14.sp), color = ArgusCyanGlow, fontWeight = FontWeight.SemiBold)
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.clip(CircleShape).background(Color.White.copy(alpha = 0.07f)).padding(horizontal = 8.dp, vertical = 2.dp)) {
                Text(text = "$count ${if (count == 1) "item" else "items"}", style = MaterialTheme.typography.labelSmall, color = ArgusCyanPrimary)
            }
            if (onClear != null && count > 0) {
                Spacer(modifier = Modifier.width(6.dp))
                TextButton(
                    onClick = onClear,
                    modifier = Modifier
                        .defaultMinSize(minWidth = 48.dp, minHeight = 48.dp)
                        .let { base -> if (clearTestTag != null) base.testTag(clearTestTag) else base }
                ) {
                    Text(
                        text = "Clear",
                        style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp),
                        color = ArgusTextMuted
                    )
                }
            }
        }
    }
}


/**
 * Translates built-in Genesis seed titles for display only.
 * Corpus content, fingerprints, and persisted document metadata remain unchanged.
 */
private fun displayTrainingItemTitle(title: String): String = when (title) {
    "Prinsip Kemandirian AGI" -> "AGI Independence Principles"
    "Privasi dan Sensor" -> "Privacy and Sensors"
    "Arsitektur Transformer On-Device" -> "On-Device Transformer Architecture"
    else -> title
}

@Composable
fun ActiveTrainingCard(item: TrainingData) {
    Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(ArgusCyanPrimary.copy(alpha = 0.12f)).padding(12.dp)) {
        Column {
            Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(modifier = Modifier.size(16.dp), color = ArgusCyanPrimary, strokeWidth = 2.dp)
                Spacer(modifier = Modifier.width(8.dp))
                Text(text = "IN PROGRESS • ${item.source}", style = MaterialTheme.typography.labelSmall, color = ArgusCyanPrimary, fontWeight = FontWeight.Bold)
            }
            Spacer(modifier = Modifier.height(6.dp))
            Text(text = displayTrainingItemTitle(item.title), style = MaterialTheme.typography.titleMedium.copy(fontSize = 13.sp), color = ArgusTextPrimary, fontWeight = FontWeight.Medium)
            if (item.totalChunks > 1) {
                Spacer(modifier = Modifier.height(6.dp))
                val progress = (item.chunkIndex - 1).coerceAtLeast(0).toFloat() / item.totalChunks.toFloat()
                LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth().height(4.dp).clip(RoundedCornerShape(2.dp)), color = ArgusCyanPrimary, trackColor = Color.White.copy(alpha = 0.08f))
                Spacer(modifier = Modifier.height(4.dp))
                Text(text = "Working on chunk ${item.chunkIndex} of ${item.totalChunks}", style = MaterialTheme.typography.labelSmall, color = ArgusCyanGlow, fontFamily = FontFamily.Monospace, fontSize = 10.sp)
            }
            Spacer(modifier = Modifier.height(4.dp))
            Text(text = item.content.take(160) + if (item.content.length > 160) "..." else "", style = MaterialTheme.typography.bodyMedium.copy(fontSize = 11.sp), color = ArgusTextSecondary)
        }
    }
}

@Composable
fun QueuedItemCard(item: TrainingData) {
    Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(Color.White.copy(alpha = 0.04f)).padding(10.dp)) {
        Column {
            Text(text = displayTrainingItemTitle(item.title), style = MaterialTheme.typography.titleMedium.copy(fontSize = 12.sp), color = ArgusTextPrimary)
            val queueDescription = if (item.status == TrainingStatus.OPTIMIZER_COMMITTED) {
                "Optimizer committed • finalization pending"
            } else {
                "Source: ${item.source} (${item.content.length} characters)"
            }
            Text(text = queueDescription, style = MaterialTheme.typography.labelSmall, color = ArgusTextMuted)
        }
    }
}

@Composable
fun CompletedItemCard(item: TrainingData) {
    val isFailed = item.status == TrainingStatus.FAILED
    Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(Color.White.copy(alpha = 0.03f)).padding(10.dp)) {
        Column {
            Text(text = displayTrainingItemTitle(item.title), style = MaterialTheme.typography.titleMedium.copy(fontSize = 12.sp), color = ArgusTextPrimary)
            val lossStr = if (isFailed) "Status: Failed" else item.finalLoss?.let { String.format(Locale.US, "Loss: %.4f", it) } ?: "Loss: unavailable"
            Text(text = "$lossStr | ${item.stepDurationMs}ms | ${item.source}", style = MaterialTheme.typography.labelSmall, color = if (isFailed) ArgusError else ArgusTextSecondary, fontFamily = FontFamily.Monospace)
        }
    }
}

@Composable
fun EmptyQueueCard(text: String) {
    Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).background(Color.White.copy(alpha = 0.03f)).padding(12.dp), contentAlignment = Alignment.Center) {
        Text(text = text, style = MaterialTheme.typography.bodyMedium.copy(fontSize = 11.sp), color = ArgusTextMuted, fontFamily = FontFamily.Monospace)
    }
}

// =====================================================================
// ESTIMATED REMAINING TIME
// =====================================================================
private fun formatParameterScale(parameters: Long): String = when {
    parameters >= 1_000_000_000L ->
        String.format(Locale.US, "%.1fB", parameters / 1_000_000_000.0)
    parameters >= 1_000_000L ->
        String.format(Locale.US, "%.1fM", parameters / 1_000_000.0)
    parameters >= 1_000L ->
        String.format(Locale.US, "%.1fK", parameters / 1_000.0)
    else -> parameters.toString()
}

private fun avgStepMsFrom(completed: List<TrainingData>): Long {
    val valid = completed.filter { it.status == TrainingStatus.COMPLETED && it.stepDurationMs > 0 }.take(20)
    if (valid.isEmpty()) return 0L
    return valid.map { it.stepDurationMs }.average().toLong()
}

private fun throttleMsFor(tempC: Float): Long = when {
    tempC >= 45f -> 600L
    tempC >= 40f -> 300L
    else -> 100L
}

private fun remainingTrainingEtaMs(
    currentItem: TrainingData?,
    nextQueue: List<TrainingData>,
    completedQueue: List<TrainingData>,
    safetyState: DeviceSafetyState,
    isPaused: Boolean
): Long? {
    if (isPaused) return null
    if (!safetyState.isSafeToTrain) return null
    val avg = avgStepMsFrom(completedQueue)
    if (avg <= 0L) return null
    val remaining = nextQueue.size + (if (currentItem != null) 1 else 0)
    if (remaining == 0) return 0L
    return remaining * (avg + throttleMsFor(safetyState.temperatureCelsius))
}

private fun formatEta(ms: Long): String {
    if (ms <= 0L) return "0s"
    val s = ms / 1000
    return when {
        s < 60 -> "${s}s"
        s < 3600 -> "${s / 60}m ${s % 60}s"
        else -> "${s / 3600}h ${(s % 3600) / 60}m"
    }
}