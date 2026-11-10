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
// MAIN SCREEN - VERSI BERSIH CYBERNETIC
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
    val hardwareInfo by viewModel.hardwareInfo.collectAsState()
    val totalSteps by viewModel.totalStepsExecuted.collectAsState()
    val activeDoc by viewModel.activeDocument.collectAsState()
    val vocabSize = viewModel.getVocabSize()
    val worldEntities by viewModel.worldEntities.collectAsState()

    val trainingEtaMs = remainingTrainingEtaMs(currentItem, nextQueue, completedQueue, safetyState, isPaused)
    val trainingEtaLabel = when {
        isPaused -> "Jeda"
        !safetyState.isSafeToTrain -> "Tertahan Guard"
        trainingEtaMs == null -> "--"
        trainingEtaMs == 0L -> "Tidak ada antrean"
        else -> "~${formatEta(trainingEtaMs)}"
    }

    val formattedParams = NumberFormat.getNumberInstance(Locale.US).format(totalParams)
    val adapterSizeKb = adapterBytes / 1024.0
    val adapterSizeStr = if (adapterBytes > 0) String.format(Locale.US, "%.2f KB", adapterSizeKb) else "belum ada"
    val lastLoss = lossHistory.lastOrNull()?.let { String.format(Locale.US, "%.4f", it) } ?: "--"

    // Progress dari dokumen aktif, bukan hardcode
    val docProgress = if (activeDoc != null && activeDoc!!.totalChunks > 0) {
        activeDoc!!.currentChunkIndex.toFloat() / activeDoc!!.totalChunks.toFloat()
    } else 0f
    val docPct = (docProgress * 100).toInt()

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(ArgusDarkBg)
            .testTag("neural_core_screen")
    ) {
        // Lapis 1: Neuron jadi background
        NeuronBackground()

        // Lapis 2: Konten, tanpa garis, tanpa border
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp),
            contentPadding = PaddingValues(vertical = 16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // HEADER + MATA GLOW
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
                            text = "On-Device Training • Neural Protocol v2.1 • NPU Accelerated",
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusTextSecondary,
                            fontSize = 10.sp
                        )
                    }
                }
            }

            // PROSES TRAINING
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
                                text = "PROSES TRAINING BERJALAN...",
                                style = MaterialTheme.typography.titleSmall,
                                color = ArgusCyanGlow,
                                fontWeight = FontWeight.Bold
                            )
                            Box(
                                Modifier
                                    .size(10.dp)
                                    .background(if (safetyState.isSafeToTrain && !isPaused) ArgusSuccess else ArgusWarning, CircleShape)
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
                                text = if (activeDoc != null) "Blok ${activeDoc!!.currentChunkIndex} / ${activeDoc!!.totalChunks} • $docPct% selesai" else "Idle • Menunggu stimulus",
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

            // METRICS - tanpa border
            item {
                Column {
                    Text(
                        text = "METRICS PELATIHAN",
                        style = MaterialTheme.typography.titleSmall,
                        color = ArgusTextPrimary,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(10.dp))
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                        StatMetricBox(label = "LOSS", value = lastLoss, unit = "$totalSteps langkah", valueColor = ArgusCyanPrimary, modifier = Modifier.weight(1f))
                        StatMetricBox(label = "KOSAKATA", value = "$vocabSize Token", unit = "Kamus aktif", valueColor = ArgusCyanGlow, modifier = Modifier.weight(1f))
                        StatMetricBox(label = "MEMORI", value = "${worldEntities.size} Entitas", unit = "World Model", valueColor = ArgusSuccess, modifier = Modifier.weight(1f))
                    }
                }
            }

            // TRAINING CONTROL - cuma Jeda / Lanjutkan
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
                                    text = "Autonomous Training Loop",
                                    style = MaterialTheme.typography.titleMedium,
                                    color = ArgusTextPrimary,
                                    fontWeight = FontWeight.Bold
                                )
                                Text(
                                    text = "Total Step: $totalSteps • ETA: $trainingEtaLabel",
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
                                Text(text = if (isPaused) "Lanjutkan" else "Jeda", fontWeight = FontWeight.Bold, fontSize = 12.sp)
                            }
                        }
                        Spacer(modifier = Modifier.height(10.dp))
                        Text(
                            text = "Arsitektur: Micro-Transformer SwiGLU | D_Model=128 | Heads=4 | D_FF=512 | LoRA Q/V Rank 4",
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusTextMuted,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 10.sp
                        )
                        Text(
                            text = "Params: $formattedParams • Adapter: $adapterSizeStr • Backend: ${hardwareInfo.primaryAccelerator}",
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusTextMuted,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 10.sp
                        )
                    }
                }
            }

            // REALTIME LOG SISTEM - dari data beneran
            item {
                Box(
                    Modifier
                        .fillMaxWidth()
                        .background(Color.White.copy(alpha = 0.04f), RoundedCornerShape(18.dp))
                        .padding(14.dp)
                ) {
                    Column {
                        Text(text = "REALTIME LOG SISTEM:", style = MaterialTheme.typography.titleSmall, color = ArgusTextPrimary, fontWeight = FontWeight.Bold)
                        Spacer(modifier = Modifier.height(8.dp))
                        if (currentItem != null) {
                            Text(text = "> Training: ${currentItem!!.title.take(60)}", style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp), color = ArgusTextSecondary, fontFamily = FontFamily.Monospace)
                            Text(text = "> Sumber: ${currentItem!!.source} | Loss: $lastLoss", style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp), color = ArgusTextSecondary, fontFamily = FontFamily.Monospace)
                        } else {
                            Text(text = "> Idle - menunggu data baru", style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp), color = ArgusTextMuted, fontFamily = FontFamily.Monospace)
                        }
                        if (activeDoc != null) {
                            Text(text = "> Corpus aktif: ${activeDoc!!.docTitle}", style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp), color = ArgusCyanGlow, fontFamily = FontFamily.Monospace)
                        }
                        Text(text = "> Checkpoint: auto-save tiap blok selesai", style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp), color = ArgusSuccess, fontFamily = FontFamily.Monospace)
                        Spacer(modifier = Modifier.height(8.dp))
                        Text(text = "STATUS MODEL: A.R.G.U.S v2.0 • Stabil • Safety Guardrails: AKTIF", style = MaterialTheme.typography.labelSmall, color = ArgusSuccess, fontWeight = FontWeight.Bold)
                    }
                }
            }

            // ANTREAN
            item {
                Text(text = "ANTREAN & HISTORI TRAINING", style = MaterialTheme.typography.titleMedium, color = ArgusCyanPrimary, fontWeight = FontWeight.Bold)
            }

            item {
                QueueSectionHeader(title = "1. Sedang Ditraining", count = if (currentItem != null) 1 else 0)
                Spacer(modifier = Modifier.height(8.dp))
                if (currentItem != null) ActiveTrainingCard(currentItem!!)
                else EmptyQueueCard("Tidak ada data yang sedang aktif (Idle)")
            }

            item {
                QueueSectionHeader(title = "2. Selanjutnya", count = nextQueue.size, onClear = { viewModel.clearTrainingQueue() })
                Spacer(modifier = Modifier.height(8.dp))
                if (nextQueue.isNotEmpty()) {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        nextQueue.take(5).forEach { QueuedItemCard(it) }
                        if (nextQueue.size > 5) Text(text = "+ ${nextQueue.size - 5} blok lagi di antrean", style = MaterialTheme.typography.labelSmall, color = ArgusTextMuted)
                    }
                } else EmptyQueueCard("Antrean kosong. Buka aplikasi lain / kirim dokumen di chat.")
            }

            item {
                QueueSectionHeader(title = "3. Sudah Ditraining", count = completedQueue.size, onClear = { viewModel.clearCompletedTraining() })
                Spacer(modifier = Modifier.height(8.dp))
                if (completedQueue.isNotEmpty()) {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        completedQueue.take(5).forEach { CompletedItemCard(it) }
                        if (completedQueue.size > 5) Text(text = "+ ${completedQueue.size - 5} blok lagi di histori", style = MaterialTheme.typography.labelSmall, color = ArgusTextMuted)
                    }
                } else EmptyQueueCard("Histori belum ada.")
            }
        }
    }
}

// =====================================================================
// NEURON BACKGROUND - 45 titik agar performa ringan & lancar
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
// MATA GLOW - 3 lapis efek cybernetic
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
// KOMPONEN CARD & METRICS
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
                Text(text = "$count item", style = MaterialTheme.typography.labelSmall, color = ArgusCyanPrimary)
            }
            if (onClear != null && count > 0) {
                Spacer(modifier = Modifier.width(6.dp))
                Text(text = "Bersihkan", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp), color = ArgusTextMuted, modifier = Modifier.clip(RoundedCornerShape(4.dp)).clickable { onClear() }.padding(horizontal = 4.dp, vertical = 2.dp))
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
                Text(text = "AKTIF: ${item.source}", style = MaterialTheme.typography.labelSmall, color = ArgusCyanPrimary, fontWeight = FontWeight.Bold)
            }
            Spacer(modifier = Modifier.height(6.dp))
            Text(text = item.title, style = MaterialTheme.typography.titleMedium.copy(fontSize = 13.sp), color = ArgusTextPrimary, fontWeight = FontWeight.Medium)
            if (item.totalChunks > 1) {
                Spacer(modifier = Modifier.height(6.dp))
                val progress = item.chunkIndex.toFloat() / item.totalChunks.toFloat()
                LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth().height(4.dp).clip(RoundedCornerShape(2.dp)), color = ArgusCyanPrimary, trackColor = Color.White.copy(alpha = 0.08f))
                Spacer(modifier = Modifier.height(4.dp))
                Text(text = "Blok ${item.chunkIndex} dari ${item.totalChunks}", style = MaterialTheme.typography.labelSmall, color = ArgusCyanGlow, fontFamily = FontFamily.Monospace, fontSize = 10.sp)
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
            Text(text = "Sumber: ${item.source} (${item.content.length} karakter)", style = MaterialTheme.typography.labelSmall, color = ArgusTextMuted)
        }
    }
}

@Composable
fun CompletedItemCard(item: TrainingData) {
    val isFailed = item.status == TrainingStatus.FAILED
    Box(Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(Color.White.copy(alpha = 0.03f)).padding(10.dp)) {
        Column {
            Text(text = item.title, style = MaterialTheme.typography.titleMedium.copy(fontSize = 12.sp), color = ArgusTextPrimary)
            val lossStr = if (isFailed) "Status: Gagal diproses" else item.finalLoss?.let { String.format(Locale.US, "Loss: %.4f", it) } ?: "Loss: belum ada"
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
// ESTIMASI WAKTU (ETA)
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
    if (ms <= 0L) return "0 dtk"
    val s = ms / 1000
    return when {
        s < 60 -> "$s dtk"
        s < 3600 -> "${s / 60} mnt ${s % 60} dtk"
        else -> "${s / 3600} jam ${(s % 3600) / 60} mnt"
    }
}