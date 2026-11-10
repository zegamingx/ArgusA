package com.argus.agi.ui.screens

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
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
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
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
import kotlin.random.Random

// =====================================================================
// NEURAL CORE SCREEN
// =====================================================================

@Composable
fun NeuralCoreScreen(viewModel: ArgusViewModel) {
    val totalParams = viewModel.getTotalParameters()
    val adapterBytes = viewModel.getAdapterSizeBytes()
    val isPaused by viewModel.isTrainingPaused.collectAsState()
    val safetyState by viewModel.safetyState.collectAsState()
    val currentItem by viewModel.currentTrainingItem.collectAsState()
    val nextQueue by viewModel.nextQueue.collectAsState()
    val completedQueue by viewModel.completedQueue.collectAsState()
    val lossHistory by viewModel.lossHistory.collectAsState()
    val totalSteps by viewModel.totalStepsExecuted.collectAsState()
    val activeDoc by viewModel.activeDocument.collectAsState()
    val vocabSize = viewModel.getVocabSize()
    val worldEntities by viewModel.worldEntities.collectAsState()
    val modelConfig = viewModel.engineConfig
    val lastVulkanDispatch = viewModel.activeComputeBackend
    val isVulkanRuntimeReady = viewModel.isVulkanRuntimeReady
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
            .background(ArgusDarkBg)
            .testTag("neural_core_screen")
    ) {
        // Decorative neural background
        NeuronBackground()

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
                        StatMetricBox(label = "LAST LOSS", value = lastLoss, unit = "latest completed item", valueColor = ArgusCyanPrimary, modifier = Modifier.weight(1f))
                        StatMetricBox(label = "VOCABULARY", value = "$vocabSize", unit = "tokens", valueColor = ArgusCyanGlow, modifier = Modifier.weight(1f))
                        StatMetricBox(label = "WORLD MODEL", value = "${worldEntities.size}", unit = "records", valueColor = ArgusSuccess, modifier = Modifier.weight(1f))
                    }
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
                            text = "Architecture: Micro-Transformer · dModel=${modelConfig.dModel} · Heads=${modelConfig.nHeads} · FFN=${modelConfig.dFf} · Context=${modelConfig.maxSeqLen} · LoRA rank=${modelConfig.loraRank}",
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusTextMuted,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 10.sp
                        )
                        Text(
                            text = "Parameters: $formattedParams • LoRA adapter: $adapterSizeStr • Last successful dispatch: $lastVulkanDispatch • Vulkan runtime: ${if (isVulkanRuntimeReady) "SELF-CHECKED" else "UNAVAILABLE"}",
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusTextMuted,
                            fontFamily = FontFamily.Monospace,
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
                            text = "> Current item: ${activeTrainingItem?.title?.take(60) ?: "None"}",
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
                QueueSectionHeader(title = "2. Next in Queue", count = nextQueue.size, onClear = { viewModel.clearTrainingQueue() })
                Spacer(modifier = Modifier.height(8.dp))
                if (nextQueue.isNotEmpty()) {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        nextQueue.take(5).forEach { QueuedItemCard(it) }
                        if (nextQueue.size > 5) Text(text = "+ ${nextQueue.size - 5} more queued items", style = MaterialTheme.typography.labelSmall, color = ArgusTextMuted)
                    }
                } else EmptyQueueCard("The queue is empty. Add text or import a document from the Terminal.")
            }

            item {
                QueueSectionHeader(title = "3. Recently Completed", count = completedQueue.size, onClear = { viewModel.clearCompletedTraining() })
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
// DECORATIVE NEURAL BACKGROUND
// =====================================================================
@Composable
fun NeuronBackground() {
    val nodes = remember { List(45) { Offset(Random.nextFloat(), Random.nextFloat()) } }
    var pulse by remember { mutableStateOf(0f) }
    LaunchedEffect(Unit) {
        while (true) {
            pulse = if (pulse == 0f) 1f else 0f
            kotlinx.coroutines.delay(1200)
        }
    }
    Canvas(modifier = Modifier.fillMaxSize()) {
        val w = size.width
        val h = size.height
        for (i in nodes.indices) {
            for (j in i + 1 until nodes.size) {
                val a = Offset(nodes[i].x * w, nodes[i].y * h)
                val b = Offset(nodes[j].x * w, nodes[j].y * h)
                if ((a - b).getDistance() < 220f) {
                    drawLine(color = Color(0xFF00BFFF).copy(alpha = 0.16f), start = a, end = b, strokeWidth = 1f)
                }
            }
        }
        nodes.forEach { n ->
            val pos = Offset(n.x * w, n.y * h)
            drawCircle(color = Color(0xFF00BFFF).copy(alpha = 0.14f), radius = 18f + pulse * 8f, center = pos)
            drawCircle(color = Color(0xFF4FC3FF), radius = 4f, center = pos)
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
fun QueueSectionHeader(title: String, count: Int, onClear: (() -> Unit)? = null) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
        Text(text = title, style = MaterialTheme.typography.titleMedium.copy(fontSize = 14.sp), color = ArgusCyanGlow, fontWeight = FontWeight.SemiBold)
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.clip(CircleShape).background(Color.White.copy(alpha = 0.07f)).padding(horizontal = 8.dp, vertical = 2.dp)) {
                Text(text = "$count ${if (count == 1) "item" else "items"}", style = MaterialTheme.typography.labelSmall, color = ArgusCyanPrimary)
            }
            if (onClear != null && count > 0) {
                Spacer(modifier = Modifier.width(6.dp))
                Text(text = "Clear", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp), color = ArgusTextMuted, modifier = Modifier.clip(RoundedCornerShape(4.dp)).clickable { onClear() }.padding(horizontal = 4.dp, vertical = 2.dp))
            }
        }
    }
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
            Text(text = item.title, style = MaterialTheme.typography.titleMedium.copy(fontSize = 13.sp), color = ArgusTextPrimary, fontWeight = FontWeight.Medium)
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
            Text(text = item.title, style = MaterialTheme.typography.titleMedium.copy(fontSize = 12.sp), color = ArgusTextPrimary)
            Text(text = "Source: ${item.source} (${item.content.length} characters)", style = MaterialTheme.typography.labelSmall, color = ArgusTextMuted)
        }
    }
}

@Composable
fun CompletedItemCard(item: TrainingData) {
    val isFailed = item.status == TrainingStatus.FAILED
    Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(Color.White.copy(alpha = 0.03f)).padding(10.dp)) {
        Column {
            Text(text = item.title, style = MaterialTheme.typography.titleMedium.copy(fontSize = 12.sp), color = ArgusTextPrimary)
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