package com.argus.agi.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.ErrorOutline
import androidx.compose.material.icons.filled.HourglassTop
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Speed
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.argus.agi.model.TrainingData
import com.argus.agi.model.TrainingStatus
import com.argus.agi.sensor.DeviceSafetyState
import com.argus.agi.ui.ArgusViewModel
import com.argus.agi.ui.components.NeuronBrain3DCard
import com.argus.agi.ui.theme.ArgusBlueAccent
import com.argus.agi.ui.theme.ArgusCardBorder
import com.argus.agi.ui.theme.ArgusCyanGlow
import com.argus.agi.ui.theme.ArgusCyanPrimary
import com.argus.agi.ui.theme.ArgusDarkBg
import com.argus.agi.ui.theme.ArgusError
import com.argus.agi.ui.theme.ArgusSuccess
import com.argus.agi.ui.theme.ArgusSurface
import com.argus.agi.ui.theme.ArgusSurfaceVariant
import com.argus.agi.ui.theme.ArgusTextMuted
import com.argus.agi.ui.theme.ArgusTextPrimary
import com.argus.agi.ui.theme.ArgusTextSecondary
import com.argus.agi.ui.theme.ArgusWarning
import java.text.NumberFormat
import java.util.Locale
import kotlin.math.min

@Composable
fun NeuralCoreScreen(viewModel: ArgusViewModel) {
    val totalParams = viewModel.getTotalParameters()
    val adapterBytes = viewModel.getAdapterSizeBytes()
    val isPaused by viewModel.isTrainingPaused.collectAsState()
    val safetyState by viewModel.safetyState.collectAsState()
    val isForceActive by viewModel.isForceOverrideActive.collectAsState()
    val genesisState by viewModel.genesisState.collectAsState()
    val currentItem by viewModel.currentTrainingItem.collectAsState()
    val nextQueue by viewModel.nextQueue.collectAsState()
    val completedQueue by viewModel.completedQueue.collectAsState()
    val lossHistory by viewModel.lossHistory.collectAsState()
    val benchmark by viewModel.benchmarkState.collectAsState()
    val isBenchmarking by viewModel.isBenchmarking.collectAsState()
    val hardwareInfo by viewModel.hardwareInfo.collectAsState()
    val totalSteps by viewModel.totalStepsExecuted.collectAsState()
    val activeDoc by viewModel.activeDocument.collectAsState()

    // === ETA Training Selesai ===
    val trainingEtaMs = remainingTrainingEtaMs(
        currentItem, nextQueue, completedQueue, safetyState, isPaused
    )
    val trainingEtaLabel = when {
        isPaused -> "Jeda"
        !safetyState.isSafeToTrain -> "Tertahan Guard"
        trainingEtaMs == null -> "--"
        trainingEtaMs == 0L -> "Tidak ada antrean"
        else -> "~${formatEta(trainingEtaMs)}"
    }

    var showReplayBoot by remember { mutableStateOf(false) }
    var isSpecExpanded by remember { mutableStateOf(true) }
    val vocabSize = viewModel.getVocabSize()
    val worldEntities by viewModel.worldEntities.collectAsState()
    var isAgiExpanded by remember { mutableStateOf(true) }
    var isTrainingExpanded by remember { mutableStateOf(true) }
    var isBenchmarkExpanded by remember { mutableStateOf(true) }
    var isQueueExpanded by remember { mutableStateOf(true) }
    var showAllNextQueue by remember { mutableStateOf(false) }
    var showAllCompletedQueue by remember { mutableStateOf(false) }

    val formattedParams = NumberFormat.getNumberInstance(Locale.US).format(totalParams)
    val adapterSizeKb = adapterBytes / 1024.0
    val adapterSizeStr = if (adapterBytes > 0) String.format(Locale.US, "%.2f KB", adapterSizeKb) else "belum ada"

    if (showReplayBoot) {
        GenesisBootScreen(
            viewModel = viewModel,
            onFinish = { showReplayBoot = false }
        )
        return
    }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(ArgusDarkBg)
            .padding(horizontal = 16.dp),
        contentPadding = PaddingValues(vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        item {
            Column {
                Text(
                    text = "NEURAL CORE & TRAINING ENGINE",
                    style = MaterialTheme.typography.titleLarge,
                    color = ArgusCyanPrimary,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "100% On-Device Neural Backpropagation | Zero Hardcode",
                    style = MaterialTheme.typography.bodyMedium,
                    color = ArgusTextSecondary
                )
            }
        }

        item {
            NeuronBrain3DCard(
                hardwareInfo = hardwareInfo,
                identity = genesisState.identity,
                isTrainingActive = currentItem != null && !isPaused && safetyState.isSafeToTrain,
                currentLoss = lossHistory.lastOrNull(),
                onReplayBootSequence = { showReplayBoot = true }
            )
        }

        // 1. Model Architecture
        item {
            Surface(
                shape = RoundedCornerShape(16.dp),
                color = ArgusSurface,
                border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(14.dp)) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(8.dp))
                            .clickable { isSpecExpanded = !isSpecExpanded }
                            .padding(vertical = 2.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = Icons.Default.Memory,
                                contentDescription = "Core",
                                tint = ArgusCyanPrimary,
                                modifier = Modifier.size(24.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "Spesifikasi Model Aktif",
                                style = MaterialTheme.typography.titleMedium,
                                color = ArgusTextPrimary,
                                fontWeight = FontWeight.Bold
                            )
                        }

                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(ArgusCyanPrimary.copy(alpha = 0.15f))
                                    .padding(horizontal = 8.dp, vertical = 4.dp)
                            ) {
                                Text(
                                    text = "Micro-Transformer",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = ArgusCyanPrimary
                                )
                            }
                            Spacer(modifier = Modifier.width(8.dp))
                            Icon(
                                imageVector = if (isSpecExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                                contentDescription = "Toggle",
                                tint = ArgusCyanPrimary,
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }

                    AnimatedVisibility(
                        visible = isSpecExpanded,
                        enter = expandVertically() + fadeIn(),
                        exit = shrinkVertically() + fadeOut()
                    ) {
                        Column {
                            Spacer(modifier = Modifier.height(14.dp))
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                StatMetricBox(
                                    label = "JUMLAH PARAMETER",
                                    value = formattedParams,
                                    unit = "Weight Elements",
                                    valueColor = ArgusCyanPrimary,
                                    modifier = Modifier.weight(1f)
                                )
                                Spacer(modifier = Modifier.width(12.dp))
                                StatMetricBox(
                                    label = "UKURAN LORA ADAPTER",
                                    value = adapterSizeStr,
                                    unit = "Rank 4 (FP32)",
                                    valueColor = ArgusCyanGlow,
                                    modifier = Modifier.weight(1f)
                                )
                            }

                            Spacer(modifier = Modifier.height(12.dp))

                            Text(
                                text = "Arsitektur: Micro-Transformer SwiGLU | D_Model=128 | Heads=4 | D_FF=512 | KV-Cache | LoRA Q/V Rank 4",
                                style = MaterialTheme.typography.labelSmall,
                                color = ArgusTextMuted,
                                fontFamily = FontFamily.Monospace
                            )
                            Text(
                                text = "Backend Komputasi: ${hardwareInfo.primaryAccelerator}",
                                style = MaterialTheme.typography.labelSmall,
                                color = ArgusCyanGlow,
                                fontFamily = FontFamily.Monospace
                            )
                        }
                    }
                }
            }
        }

        // 1.5 Perkembangan AGI & Bahasa
        item {
            Surface(
                shape = RoundedCornerShape(16.dp),
                color = ArgusSurface,
                border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(14.dp)) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(8.dp))
                            .clickable { isAgiExpanded = !isAgiExpanded }
                            .padding(vertical = 2.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = Icons.Default.Psychology,
                                contentDescription = "AGI",
                                tint = ArgusCyanPrimary,
                                modifier = Modifier.size(24.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Column {
                                Text(
                                    text = "Perkembangan AGI & Bahasa",
                                    style = MaterialTheme.typography.titleMedium,
                                    color = ArgusTextPrimary,
                                    fontWeight = FontWeight.Bold
                                )
                                Text(
                                    text = "Evolusi Nyata On-Device | Zero Mock",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = ArgusTextSecondary
                                )
                            }
                        }

                        Icon(
                            imageVector = if (isAgiExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                            contentDescription = "Toggle",
                            tint = ArgusCyanPrimary,
                            modifier = Modifier.size(20.dp)
                        )
                    }

                    AnimatedVisibility(
                        visible = isAgiExpanded,
                        enter = expandVertically() + fadeIn(),
                        exit = shrinkVertically() + fadeOut()
                    ) {
                        Column {
                            Spacer(modifier = Modifier.height(14.dp))
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                StatMetricBox(
                                    label = "KOSAKATA BAHASA",
                                    value = "$vocabSize Token",
                                    unit = "Kamus Kosakata Aktif",
                                    valueColor = ArgusCyanPrimary,
                                    modifier = Modifier.weight(1f)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                StatMetricBox(
                                    label = "BLOK DIKUASAI",
                                    value = "${completedQueue.size} Blok",
                                    unit = "Total Korpus Terlatih",
                                    valueColor = ArgusSuccess,
                                    modifier = Modifier.weight(1f)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                StatMetricBox(
                                    label = "MEMORI KONTEKS",
                                    value = "${worldEntities.size} Entitas",
                                    unit = "World Model Persisten",
                                    valueColor = ArgusCyanGlow,
                                    modifier = Modifier.weight(1f)
                                )
                            }

                            Spacer(modifier = Modifier.height(12.dp))

                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clip(RoundedCornerShape(10.dp))
                                    .background(ArgusSurfaceVariant)
                                    .padding(10.dp)
                            ) {
                                Text(
                                    text = "Status Pembelajaran Bahasa & Multimodal On-Device:",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = ArgusTextPrimary,
                                    fontWeight = FontWeight.Bold
                                )
                                Spacer(modifier = Modifier.height(4.dp))
                                Text(
                                    text = "• Sintesis Suara (TTS): Adaptif (Bahasa Indonesia & English) sesuai korpus terlatih.",
                                    style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                    color = ArgusTextSecondary
                                )
                                Text(
                                    text = "• Pengenalan Suara (STT): 100% On-device dengan auto-fallback error guard.",
                                    style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                    color = ArgusTextSecondary
                                )
                                Text(
                                    text = "• Konvergensi Model: ${lossHistory.lastOrNull()?.let { String.format(Locale.US, "Loss %.4f (%d langkah)", it, totalSteps) } ?: "Belum ada langkah training"}.",
                                    style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp),
                                    color = if (lossHistory.isNotEmpty()) ArgusCyanGlow else ArgusTextMuted
                                )
                            }
                        }
                    }
                }
            }
        }

        // 2. Training Control + ETA
        item {
            Surface(
                shape = RoundedCornerShape(16.dp),
                color = ArgusSurface,
                border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(14.dp)) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(8.dp))
                            .clickable { isTrainingExpanded = !isTrainingExpanded }
                            .padding(vertical = 2.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column {
                            Text(
                                text = "Autonomous Training Loop",
                                style = MaterialTheme.typography.titleMedium,
                                color = ArgusTextPrimary,
                                fontWeight = FontWeight.Bold
                            )
                            Text(
                                text = "Total Step Selesai: $totalSteps",
                                style = MaterialTheme.typography.labelSmall,
                                color = ArgusTextSecondary
                            )
                            Text(
                                text = "ETA Training Selesai: $trainingEtaLabel",
                                style = MaterialTheme.typography.labelSmall,
                                color = when {
                                    isPaused || !safetyState.isSafeToTrain -> ArgusWarning
                                    trainingEtaMs == 0L -> ArgusTextMuted
                                    else -> ArgusCyanGlow
                                },
                                fontFamily = FontFamily.Monospace,
                                fontWeight = FontWeight.Bold
                            )
                        }

                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Button(
                                onClick = { viewModel.toggleTrainingPause() },
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = if (isPaused) ArgusSuccess else ArgusWarning,
                                    contentColor = ArgusDarkBg
                                ),
                                shape = RoundedCornerShape(12.dp),
                                modifier = Modifier.testTag("toggle_training_button")
                            ) {
                                Icon(
                                    imageVector = if (isPaused) Icons.Default.PlayArrow else Icons.Default.Pause,
                                    contentDescription = null,
                                    modifier = Modifier.size(16.dp)
                                )
                                Spacer(modifier = Modifier.width(4.dp))
                                Text(
                                    text = if (isPaused) "Lanjutkan" else "Jeda",
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 11.sp
                                )
                            }
                            Spacer(modifier = Modifier.width(6.dp))
                            Icon(
                                imageVector = if (isTrainingExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                                contentDescription = "Toggle",
                                tint = ArgusCyanPrimary,
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }

                    AnimatedVisibility(
                        visible = isTrainingExpanded,
                        enter = expandVertically() + fadeIn(),
                        exit = shrinkVertically() + fadeOut()
                    ) {
                        Column {
                            Spacer(modifier = Modifier.height(12.dp))
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clip(RoundedCornerShape(10.dp))
                                    .background(ArgusSurfaceVariant)
                                    .padding(12.dp)
                            ) {
                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Row(
                                        modifier = Modifier.weight(1f),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Box(
                                            modifier = Modifier
                                                .size(10.dp)
                                                .clip(CircleShape)
                                                .background(if (safetyState.isSafeToTrain) ArgusSuccess else ArgusError)
                                        )
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Column {
                                            Text(
                                                text = "Battery & Thermal Guard",
                                                style = MaterialTheme.typography.labelSmall,
                                                color = ArgusTextPrimary,
                                                fontWeight = FontWeight.Bold
                                            )
                                            Text(
                                                text = safetyState.statusMessage,
                                                style = MaterialTheme.typography.bodyMedium.copy(fontSize = 11.sp),
                                                color = if (safetyState.isSafeToTrain) ArgusTextSecondary else ArgusError
                                            )
                                        }
                                    }

                                    Switch(
                                        checked = isForceActive,
                                        onCheckedChange = { viewModel.setForceTrainingOverride(it) },
                                        colors = SwitchDefaults.colors(
                                            checkedThumbColor = ArgusCyanPrimary,
                                            checkedTrackColor = ArgusCyanPrimary.copy(alpha = 0.3f),
                                            uncheckedThumbColor = ArgusTextMuted,
                                            uncheckedTrackColor = ArgusSurface
                                        ),
                                        modifier = Modifier.testTag("force_training_switch")
                                    )
                                }

                                Spacer(modifier = Modifier.height(10.dp))

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    horizontalArrangement = Arrangement.SpaceBetween,
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    Text(
                                        text = if (isForceActive) "Mode Paksa: AKTIF (Guard Dilewati)" else "Mode Paksa: Standar (Auto Guard)",
                                        style = MaterialTheme.typography.labelSmall,
                                        color = if (isForceActive) ArgusCyanGlow else ArgusTextMuted,
                                        fontFamily = FontFamily.Monospace,
                                        fontSize = 10.sp
                                    )

                                    Button(
                                        onClick = { viewModel.triggerImmediateTrainingStep() },
                                        colors = ButtonDefaults.buttonColors(
                                            containerColor = ArgusCyanPrimary.copy(alpha = 0.2f),
                                            contentColor = ArgusCyanPrimary
                                        ),
                                        shape = RoundedCornerShape(8.dp),
                                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp),
                                        modifier = Modifier.testTag("trigger_immediate_step_button")
                                    ) {
                                        Text(
                                            text = "Latih Sekarang",
                                            fontSize = 11.sp,
                                            fontWeight = FontWeight.Bold
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3. Benchmark
        item {
            Surface(
                shape = RoundedCornerShape(16.dp),
                color = ArgusSurface,
                border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(14.dp)) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(RoundedCornerShape(8.dp))
                            .clickable { isBenchmarkExpanded = !isBenchmarkExpanded }
                            .padding(vertical = 2.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = Icons.Default.Speed,
                                contentDescription = "Benchmark",
                                tint = ArgusCyanPrimary,
                                modifier = Modifier.size(22.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "On-Device Benchmark",
                                style = MaterialTheme.typography.titleMedium,
                                color = ArgusTextPrimary,
                                fontWeight = FontWeight.Bold
                            )
                        }

                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Button(
                                onClick = { viewModel.runBenchmark() },
                                enabled = !isBenchmarking,
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = ArgusCyanPrimary,
                                    contentColor = ArgusDarkBg
                                ),
                                shape = RoundedCornerShape(10.dp),
                                modifier = Modifier.testTag("run_benchmark_button")
                            ) {
                                if (isBenchmarking) {
                                    CircularProgressIndicator(
                                        modifier = Modifier.size(12.dp),
                                        color = ArgusDarkBg,
                                        strokeWidth = 2.dp
                                    )
                                } else {
                                    Text(
                                        text = "Uji",
                                        fontWeight = FontWeight.Bold,
                                        fontSize = 11.sp
                                    )
                                }
                            }
                            Spacer(modifier = Modifier.width(6.dp))
                            Icon(
                                imageVector = if (isBenchmarkExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                                contentDescription = "Toggle",
                                tint = ArgusCyanPrimary,
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }

                    AnimatedVisibility(
                        visible = isBenchmarkExpanded,
                        enter = expandVertically() + fadeIn(),
                        exit = shrinkVertically() + fadeOut()
                    ) {
                        Column {
                            Spacer(modifier = Modifier.height(12.dp))
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                val tpsStr = benchmark?.let { String.format(Locale.US, "%.1f TPS", it.tokensPerSecond) } ?: "belum ada"
                                val latStr = benchmark?.let { String.format(Locale.US, "%.2f ms/tok", it.latencyMsPerToken) } ?: "belum ada"
                                val memStr = benchmark?.let { String.format(Locale.US, "%.1f MB", it.memoryUsageMb) } ?: "belum ada"

                                StatMetricBox(
                                    label = "KECEPATAN",
                                    value = tpsStr,
                                    unit = "Tokens / Detik",
                                    valueColor = ArgusCyanPrimary,
                                    modifier = Modifier.weight(1f)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                StatMetricBox(
                                    label = "LATENSI",
                                    value = latStr,
                                    unit = "Per Token",
                                    valueColor = ArgusCyanGlow,
                                    modifier = Modifier.weight(1f)
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                StatMetricBox(
                                    label = "MEMORI",
                                    value = memStr,
                                    unit = "Bobot Model (FP32)",
                                    valueColor = ArgusBlueAccent,
                                    modifier = Modifier.weight(1f)
                                )
                            }
                        }
                    }
                }
            }
        }

        // 5. Queue Header
        item {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .clickable { isQueueExpanded = !isQueueExpanded }
                    .padding(vertical = 4.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "ANTREAN & HISTORI TRAINING",
                    style = MaterialTheme.typography.titleMedium,
                    color = ArgusCyanPrimary,
                    fontWeight = FontWeight.Bold
                )
                Icon(
                    imageVector = if (isQueueExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                    contentDescription = "Toggle Antrean",
                    tint = ArgusCyanPrimary,
                    modifier = Modifier.size(20.dp)
                )
            }
        }

        if (isQueueExpanded) {
            if (activeDoc != null) {
                item {
                    val doc = activeDoc!!
                    val pct = if (doc.totalChunks > 0) (doc.currentChunkIndex.toFloat() / doc.totalChunks.toFloat() * 100).toInt() else 0
                    Surface(
                        shape = RoundedCornerShape(14.dp),
                        color = ArgusCyanPrimary.copy(alpha = 0.12f),
                        border = androidx.compose.foundation.BorderStroke(1.5.dp, ArgusCyanPrimary),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(14.dp)) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    CircularProgressIndicator(
                                        modifier = Modifier.size(16.dp),
                                        color = ArgusCyanPrimary,
                                        strokeWidth = 2.dp
                                    )
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Text(
                                        text = "CORPUS AKTIF: ${doc.docTitle}",
                                        style = MaterialTheme.typography.titleMedium.copy(fontSize = 13.sp),
                                        color = ArgusCyanPrimary,
                                        fontWeight = FontWeight.Bold
                                    )
                                }
                                Text(
                                    text = "$pct%",
                                    style = MaterialTheme.typography.titleMedium,
                                    color = ArgusCyanGlow,
                                    fontWeight = FontWeight.Bold,
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 13.sp
                                )
                            }

                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                text = "Total Karakter: ${doc.totalCharacters} char | Blok ${doc.currentChunkIndex} dari ${doc.totalChunks} (${doc.completedChunks} selesai)",
                                style = MaterialTheme.typography.labelSmall,
                                color = ArgusTextSecondary,
                                fontSize = 10.sp
                            )

                            if (doc.runningAverageLoss != null) {
                                Text(
                                    text = "Rata-rata Loss Dokumen: ${String.format(Locale.US, "%.4f", doc.runningAverageLoss)}",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = ArgusCyanGlow,
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 10.sp
                                )
                            }

                            Spacer(modifier = Modifier.height(8.dp))
                            androidx.compose.material3.LinearProgressIndicator(
                                progress = { if (doc.totalChunks > 0) doc.currentChunkIndex.toFloat() / doc.totalChunks.toFloat() else 0f },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .height(6.dp)
                                    .clip(RoundedCornerShape(3.dp)),
                                color = ArgusCyanPrimary,
                                trackColor = ArgusSurfaceVariant
                            )

                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                text = "Jaminan Pembelajaran Lengkap: Seluruh teks diproses tuntas blok demi blok melalui sliding window (zero truncation).",
                                style = MaterialTheme.typography.labelSmall,
                                color = ArgusTextMuted,
                                fontSize = 9.sp
                            )
                        }
                    }
                }
            }

            item {
                QueueSectionHeader(title = "1. Data Yang Sedang Ditraining", count = if (currentItem != null) 1 else 0)
                Spacer(modifier = Modifier.height(6.dp))

                if (currentItem != null) {
                    ActiveTrainingCard(currentItem!!)
                } else {
                    EmptyQueueCard("Tidak ada data yang sedang aktif ditraining (Idle / Menunggu Stimulus)")
                }
            }

            item {
                QueueSectionHeader(
                    title = "2. Data Selanjutnya Ditraining",
                    count = nextQueue.size,
                    onClear = { viewModel.clearTrainingQueue() }
                )
                Spacer(modifier = Modifier.height(6.dp))

                if (nextQueue.isNotEmpty()) {
                    val displayCount = if (showAllNextQueue) nextQueue.size else min(nextQueue.size, 10)
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        nextQueue.take(displayCount).forEach { item ->
                            QueuedItemCard(item)
                        }

                        if (nextQueue.size > 10) {
                            Button(
                                onClick = { showAllNextQueue = !showAllNextQueue },
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = ArgusSurfaceVariant,
                                    contentColor = ArgusCyanPrimary
                                ),
                                shape = RoundedCornerShape(8.dp),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text(
                                    text = if (showAllNextQueue) "Ciutkan Tampilan Antrean" else "Tampilkan Semua Antrean (${nextQueue.size} Blok)",
                                    fontSize = 11.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                    }
                } else {
                    EmptyQueueCard("Antrean kosong. Buka aplikasi lain untuk membaca layar atau kirim pesan/dokumen di chat.")
                }
            }

            item {
                QueueSectionHeader(
                    title = "3. Data Sudah Ditraining (Histori)",
                    count = completedQueue.size,
                    onClear = { viewModel.clearCompletedTraining() }
                )
                Spacer(modifier = Modifier.height(6.dp))

                if (completedQueue.isNotEmpty()) {
                    val displayCount = if (showAllCompletedQueue) completedQueue.size else min(completedQueue.size, 10)
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        completedQueue.take(displayCount).forEach { item ->
                            CompletedItemCard(item)
                        }

                        if (completedQueue.size > 10) {
                            Button(
                                onClick = { showAllCompletedQueue = !showAllCompletedQueue },
                                colors = ButtonDefaults.buttonColors(
                                    containerColor = ArgusSurfaceVariant,
                                    contentColor = ArgusCyanPrimary
                                ),
                                shape = RoundedCornerShape(8.dp),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Text(
                                    text = if (showAllCompletedQueue) "Ciutkan Tampilan Histori" else "Tampilkan Semua Histori (${completedQueue.size} Blok Selesai)",
                                    fontSize = 11.sp,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                        }
                    }
                } else {
                    EmptyQueueCard("Histori training belum ada.")
                }
            }
        }
    }
}

// =====================================================================
// ETA Helpers
// =====================================================================

private fun avgStepMsFrom(completed: List<TrainingData>): Long {
    val valid = completed
        .filter { it.status == TrainingStatus.COMPLETED && it.stepDurationMs > 0 }
        .take(20)
    if (valid.isEmpty()) return 0L
    return valid.map { it.stepDurationMs }.average().toLong()
}

private fun throttleMsFor(tempC: Float): Long = when {
    tempC >= 45f -> 600L
    tempC >= 40f -> 300L
    else -> 100L
}

/** Return null = data belum cukup untuk estimasi. */
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

// =====================================================================
// Composable Components
// =====================================================================



@Composable
fun StatMetricBox(
    label: String,
    value: String,
    unit: String,
    valueColor: Color,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(10.dp))
            .background(ArgusSurfaceVariant)
            .border(1.dp, ArgusCardBorder, RoundedCornerShape(10.dp))
            .padding(10.dp)
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = ArgusTextMuted,
            fontSize = 9.sp
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = value,
            style = MaterialTheme.typography.titleMedium,
            color = valueColor,
            fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace,
            fontSize = 15.sp
        )
        Text(
            text = unit,
            style = MaterialTheme.typography.labelSmall,
            color = ArgusTextSecondary,
            fontSize = 9.sp
        )
    }
}

@Composable
fun QueueSectionHeader(
    title: String,
    count: Int,
    onClear: (() -> Unit)? = null
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = title,
            style = MaterialTheme.typography.titleMedium.copy(fontSize = 14.sp),
            color = ArgusCyanGlow,
            fontWeight = FontWeight.SemiBold
        )
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .clip(CircleShape)
                    .background(ArgusSurfaceVariant)
                    .padding(horizontal = 8.dp, vertical = 2.dp)
            ) {
                Text(
                    text = "$count item",
                    style = MaterialTheme.typography.labelSmall,
                    color = ArgusCyanPrimary
                )
            }
            if (onClear != null && count > 0) {
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = "Bersihkan",
                    style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp),
                    color = ArgusTextMuted,
                    modifier = Modifier
                        .clip(RoundedCornerShape(4.dp))
                        .clickable { onClear() }
                        .padding(horizontal = 4.dp, vertical = 2.dp)
                )
            }
        }
    }
}

@Composable
fun ActiveTrainingCard(item: TrainingData) {
    Surface(
        shape = RoundedCornerShape(12.dp),
        color = ArgusCyanPrimary.copy(alpha = 0.12f),
        border = androidx.compose.foundation.BorderStroke(1.5.dp, ArgusCyanPrimary),
        modifier = Modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(16.dp),
                        color = ArgusCyanPrimary,
                        strokeWidth = 2.dp
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "AKTIF DILATIH: ${item.source}",
                        style = MaterialTheme.typography.labelSmall,
                        color = ArgusCyanPrimary,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
            Spacer(modifier = Modifier.height(6.dp))
            Text(
                text = item.title,
                style = MaterialTheme.typography.titleMedium.copy(fontSize = 13.sp),
                color = ArgusTextPrimary,
                fontWeight = FontWeight.Medium
            )

            if (item.totalChunks > 1) {
                Spacer(modifier = Modifier.height(6.dp))
                val progress = item.chunkIndex.toFloat() / item.totalChunks.toFloat()
                val pct = (progress * 100).toInt()
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "Progres Dokumen: Blok ${item.chunkIndex} dari ${item.totalChunks}",
                        style = MaterialTheme.typography.labelSmall,
                        color = ArgusCyanGlow,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 10.sp
                    )
                    Text(
                        text = "$pct%",
                        style = MaterialTheme.typography.labelSmall,
                        color = ArgusCyanPrimary,
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 10.sp
                    )
                }
                Spacer(modifier = Modifier.height(4.dp))
                androidx.compose.material3.LinearProgressIndicator(
                    progress = { progress },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(4.dp)
                        .clip(RoundedCornerShape(2.dp)),
                    color = ArgusCyanPrimary,
                    trackColor = ArgusSurfaceVariant
                )
            }

            Spacer(modifier = Modifier.height(4.dp))
            Text(
                text = item.content.take(160) + if (item.content.length > 160) "..." else "",
                style = MaterialTheme.typography.bodyMedium.copy(fontSize = 11.sp),
                color = ArgusTextSecondary
            )
        }
    }
}

@Composable
fun QueuedItemCard(item: TrainingData) {
    Surface(
        shape = RoundedCornerShape(10.dp),
        color = ArgusSurface,
        border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier.padding(10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                imageVector = Icons.Default.HourglassTop,
                contentDescription = "Queued",
                tint = ArgusWarning,
                modifier = Modifier.size(18.dp)
            )
            Spacer(modifier = Modifier.width(10.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = item.title,
                    style = MaterialTheme.typography.titleMedium.copy(fontSize = 12.sp),
                    color = ArgusTextPrimary
                )
                Text(
                    text = "Sumber: ${item.source} (${item.content.length} karakter)",
                    style = MaterialTheme.typography.labelSmall,
                    color = ArgusTextMuted
                )
            }
        }
    }
}

@Composable
fun CompletedItemCard(item: TrainingData) {
    Surface(
        shape = RoundedCornerShape(10.dp),
        color = ArgusSurfaceVariant.copy(alpha = 0.5f),
        border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder),
        modifier = Modifier.fillMaxWidth()
    ) {
        Row(
            modifier = Modifier.padding(10.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            val isFailed = item.status == TrainingStatus.FAILED
            Icon(
                imageVector = if (isFailed) Icons.Default.ErrorOutline else Icons.Default.CheckCircle,
                contentDescription = if (isFailed) "Gagal" else "Selesai",
                tint = if (isFailed) ArgusError else ArgusSuccess,
                modifier = Modifier.size(18.dp)
            )
            Spacer(modifier = Modifier.width(10.dp))
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = item.title,
                    style = MaterialTheme.typography.titleMedium.copy(fontSize = 12.sp),
                    color = ArgusTextPrimary
                )
                val lossStr = if (isFailed) {
                    "Status: Gagal diproses"
                } else {
                    item.finalLoss?.let { String.format(Locale.US, "Loss: %.4f", it) } ?: "Loss: belum ada"
                }
                Text(
                    text = "$lossStr | Waktu: ${item.stepDurationMs}ms | ${item.source}",
                    style = MaterialTheme.typography.labelSmall,
                    color = ArgusTextSecondary,
                    fontFamily = FontFamily.Monospace
                )
            }
        }
    }
}

@Composable
fun EmptyQueueCard(text: String) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(ArgusSurfaceVariant.copy(alpha = 0.4f))
            .padding(12.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = text,
            style = MaterialTheme.typography.bodyMedium.copy(fontSize = 11.sp),
            color = ArgusTextMuted,
            fontFamily = FontFamily.Monospace
        )
    }
}