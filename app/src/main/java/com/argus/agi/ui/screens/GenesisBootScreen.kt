package com.argus.agi.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
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
import androidx.compose.material.icons.filled.Error
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.argus.agi.model.GenesisLifecycleStage
import com.argus.agi.ui.ArgusViewModel
import com.argus.agi.ui.theme.ArgusCardBorder
import com.argus.agi.ui.theme.ArgusCyanGlow
import com.argus.agi.ui.theme.ArgusCyanPrimary
import com.argus.agi.ui.theme.ArgusDarkBg
import com.argus.agi.ui.theme.ArgusSuccess
import com.argus.agi.ui.theme.ArgusSurface
import com.argus.agi.ui.theme.ArgusSurfaceVariant
import com.argus.agi.ui.theme.ArgusTextMuted
import com.argus.agi.ui.theme.ArgusTextPrimary
import com.argus.agi.ui.theme.ArgusTextSecondary
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

@Composable
fun GenesisBootScreen(
    viewModel: ArgusViewModel,
    onFinish: () -> Unit
) {
    val genesisState by viewModel.genesisState.collectAsState()

    // Warna yang dipakai di dalam Canvas (non-@Composable context) harus di-extract disini
    val errorColor = MaterialTheme.colorScheme.error
    val errorContainerColor = MaterialTheme.colorScheme.errorContainer
    val onErrorContainerColor = MaterialTheme.colorScheme.onErrorContainer
    val onErrorColor = MaterialTheme.colorScheme.onError

    val isFailed = genesisState.stage == GenesisLifecycleStage.FAILED
    val isReady = !isFailed && (
        genesisState.stage == GenesisLifecycleStage.APPLICATION_READY ||
        genesisState.stage == GenesisLifecycleStage.GENERAL_LEARNING_SCHEDULED ||
        genesisState.isComplete
    )

    // Fallback trigger jika state masih IDLE (biasanya ViewModel.init sudah memicu)
    LaunchedEffect(genesisState.stage) {
        if (genesisState.stage == GenesisLifecycleStage.IDLE) {
            viewModel.retryGenesis()
        }
    }

    val infiniteTransition = rememberInfiniteTransition(label = "genesis_infinite")

    val scanY by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(2400, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "scan_line"
    )

    val orbPulse by infiniteTransition.animateFloat(
        initialValue = 0.85f,
        targetValue = 1.15f,
        animationSpec = infiniteRepeatable(
            animation = tween(1500, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "orb_pulse"
    )

    val animatedProgress by animateFloatAsState(
        targetValue = genesisState.progress,
        animationSpec = tween(250),
        label = "progress_smooth"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(ArgusDarkBg)
            .padding(16.dp),
        contentAlignment = Alignment.Center
    ) {
        // Background grid + scan line
        Canvas(modifier = Modifier.fillMaxSize()) {
            val w = size.width
            val h = size.height
            val step = 40.dp.toPx()
            var x = 0f
            while (x < w) {
                drawLine(
                    color = ArgusCyanPrimary.copy(alpha = 0.05f),
                    start = Offset(x, 0f),
                    end = Offset(x, h),
                    strokeWidth = 1f
                )
                x += step
            }
            var y = 0f
            while (y < h) {
                drawLine(
                    color = ArgusCyanPrimary.copy(alpha = 0.05f),
                    start = Offset(0f, y),
                    end = Offset(w, y),
                    strokeWidth = 1f
                )
                y += step
            }
            val currentScanY = scanY * h
            drawLine(
                brush = Brush.horizontalGradient(
                    colors = listOf(
                        Color.Transparent,
                        ArgusCyanPrimary.copy(alpha = 0.6f),
                        ArgusCyanGlow,
                        ArgusCyanPrimary.copy(alpha = 0.6f),
                        Color.Transparent
                    )
                ),
                start = Offset(0f, currentScanY),
                end = Offset(w, currentScanY),
                strokeWidth = 2.dp.toPx()
            )
        }

        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 4.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            // 1. Header
            item {
                Spacer(modifier = Modifier.height(10.dp))
                Text(
                    text = "A . R . G . U . S",
                    style = MaterialTheme.typography.headlineMedium,
                    color = if (isFailed) errorColor else ArgusCyanPrimary,
                    fontWeight = FontWeight.Bold,
                    fontFamily = FontFamily.Monospace,
                    letterSpacing = 4.sp
                )
                Text(
                    text = if (isFailed) "GENESIS INTERRUPTED"
                    else "ARGUS GENESIS — FIRST-LAUNCH LIFECYCLE",
                    style = MaterialTheme.typography.labelSmall,
                    color = if (isFailed) errorColor else ArgusCyanGlow,
                    letterSpacing = 1.5.sp,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "One Origin. Independent Evolution.",
                    style = MaterialTheme.typography.labelSmall,
                    color = ArgusTextMuted,
                    fontSize = 10.sp
                )
            }

            // 2. HUD Canvas
            item {
                Box(
                    modifier = Modifier
                        .size(190.dp)
                        .clip(RoundedCornerShape(20.dp))
                        .background(ArgusSurface.copy(alpha = 0.7f))
                        .border(
                            1.5.dp,
                            if (isFailed) errorColor else ArgusCardBorder,
                            RoundedCornerShape(20.dp)
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Canvas(modifier = Modifier.fillMaxSize()) {
                        val cx = size.width / 2f
                        val cy = size.height / 2f

                        if (!isReady && !isFailed) {
                            val numBranches = 14
                            val branchLength = size.width * 0.38f
                            for (i in 0 until numBranches) {
                                val angleRad = Math.toRadians((i * (360.0 / numBranches)) + 15.0)
                                val endX = cx + (branchLength * cos(angleRad)).toFloat()
                                val endY = cy + (branchLength * sin(angleRad)).toFloat()
                                drawLine(
                                    color = ArgusCyanPrimary.copy(alpha = 0.6f),
                                    start = Offset(cx, cy),
                                    end = Offset(endX, endY),
                                    strokeWidth = 2.dp.toPx(),
                                    cap = StrokeCap.Round
                                )
                                drawCircle(
                                    color = ArgusCyanGlow,
                                    radius = 4.dp.toPx(),
                                    center = Offset(endX, endY)
                                )
                            }
                            drawCircle(ArgusCyanPrimary, 10.dp.toPx(), Offset(cx, cy))
                            drawCircle(Color.White, 5.dp.toPx(), Offset(cx, cy))
                        } else if (isFailed) {
                            drawCircle(
                                color = errorColor.copy(alpha = 0.4f),
                                radius = 40.dp.toPx(),
                                center = Offset(cx, cy),
                                style = Stroke(width = 2.dp.toPx())
                            )
                            val r = 18.dp.toPx()
                            drawLine(
                                errorColor,
                                Offset(cx - r, cy - r), Offset(cx + r, cy + r),
                                strokeWidth = 3.dp.toPx()
                            )
                            drawLine(
                                errorColor,
                                Offset(cx + r, cy - r), Offset(cx - r, cy + r),
                                strokeWidth = 3.dp.toPx()
                            )
                        } else {
                            val baseR = 40.dp.toPx() * orbPulse
                            drawCircle(
                                ArgusCyanPrimary.copy(alpha = 0.2f),
                                baseR * 1.6f, Offset(cx, cy),
                                style = Stroke(width = 2.dp.toPx())
                            )
                            drawCircle(
                                ArgusCyanPrimary.copy(alpha = 0.4f),
                                baseR * 1.3f, Offset(cx, cy),
                                style = Stroke(width = 2.5.dp.toPx())
                            )
                            drawCircle(
                                brush = Brush.radialGradient(
                                    colors = listOf(
                                        Color.White, ArgusCyanGlow,
                                        ArgusCyanPrimary, Color.Transparent
                                    ),
                                    center = Offset(cx, cy),
                                    radius = baseR
                                ),
                                radius = baseR,
                                center = Offset(cx, cy)
                            )
                        }

                        // Corner brackets
                        val bLen = 14.dp.toPx()
                        val bPad = 10.dp.toPx()
                        val bColor = if (isFailed) errorColor.copy(alpha = 0.8f)
                        else ArgusCyanPrimary.copy(alpha = 0.8f)
                        val bW = 2.dp.toPx()
                        drawLine(bColor, Offset(bPad, bPad), Offset(bPad + bLen, bPad), bW)
                        drawLine(bColor, Offset(bPad, bPad), Offset(bPad, bPad + bLen), bW)
                        drawLine(bColor, Offset(size.width - bPad, bPad), Offset(size.width - bPad - bLen, bPad), bW)
                        drawLine(bColor, Offset(size.width - bPad, bPad), Offset(size.width - bPad, bPad + bLen), bW)
                        drawLine(bColor, Offset(bPad, size.height - bPad), Offset(bPad + bLen, size.height - bPad), bW)
                        drawLine(bColor, Offset(bPad, size.height - bPad), Offset(bPad, size.height - bPad - bLen), bW)
                        drawLine(bColor, Offset(size.width - bPad, size.height - bPad), Offset(size.width - bPad - bLen, size.height - bPad), bW)
                        drawLine(bColor, Offset(size.width - bPad, size.height - bPad), Offset(size.width - bPad, size.height - bPad - bLen), bW)
                    }

                    Box(
                        modifier = Modifier
                            .align(Alignment.BottomCenter)
                            .padding(bottom = 8.dp)
                            .clip(RoundedCornerShape(6.dp))
                            .background(ArgusDarkBg.copy(alpha = 0.85f))
                            .padding(horizontal = 8.dp, vertical = 2.dp)
                    ) {
                        Text(
                            text = when {
                                isFailed -> "GENESIS FAILED"
                                isReady -> "ORIGIN READY"
                                else -> "SYNTHESIZING CORE"
                            },
                            style = MaterialTheme.typography.labelSmall,
                            color = if (isFailed) errorColor else ArgusCyanGlow,
                            fontFamily = FontFamily.Monospace,
                            fontWeight = FontWeight.Bold,
                            fontSize = 10.sp
                        )
                    }
                }
            }

            // 3. Error Banner (hanya saat FAILED)
            if (isFailed) {
                item {
                    Surface(
                        shape = RoundedCornerShape(14.dp),
                        color = errorContainerColor,
                        border = androidx.compose.foundation.BorderStroke(1.dp, errorColor),
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("genesis_error_banner")
                    ) {
                        Column(modifier = Modifier.padding(14.dp)) {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    imageVector = Icons.Default.Error,
                                    contentDescription = null,
                                    tint = errorColor,
                                    modifier = Modifier.size(18.dp)
                                )
                                Spacer(Modifier.width(8.dp))
                                Text(
                                    text = "GENESIS GAGAL",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = errorColor,
                                    fontWeight = FontWeight.Bold,
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 11.sp
                                )
                            }
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = genesisState.errorMessage
                                    ?: "Penyebab tidak diketahui. Periksa storage, baterai, dan suhu perangkat.",
                                style = MaterialTheme.typography.bodySmall,
                                color = onErrorContainerColor,
                                fontSize = 12.sp
                            )
                        }
                    }
                }
            }

            // 4. Progress Card
            item {
                Surface(
                    shape = RoundedCornerShape(14.dp),
                    color = ArgusSurface,
                    border = androidx.compose.foundation.BorderStroke(
                        1.dp,
                        if (isFailed) errorColor else ArgusCardBorder
                    ),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(14.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Text(
                                text = genesisState.stage.displayName.uppercase(),
                                style = MaterialTheme.typography.labelSmall,
                                color = if (isFailed) errorColor else ArgusCyanPrimary,
                                fontWeight = FontWeight.Bold,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 11.sp
                            )
                            Text(
                                text = "${(animatedProgress * 100).toInt()}%",
                                style = MaterialTheme.typography.labelSmall,
                                color = if (isFailed) errorColor else ArgusCyanGlow,
                                fontWeight = FontWeight.Bold,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 11.sp
                            )
                        }
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = genesisState.stage.description,
                            style = MaterialTheme.typography.bodyMedium.copy(fontSize = 11.sp),
                            color = ArgusTextSecondary
                        )
                        Spacer(modifier = Modifier.height(10.dp))
                        LinearProgressIndicator(
                            progress = { animatedProgress },
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(6.dp)
                                .clip(RoundedCornerShape(3.dp)),
                            color = if (isFailed) errorColor else ArgusCyanPrimary,
                            trackColor = ArgusSurfaceVariant
                        )
                        if (genesisState.assessedDetails.isNotBlank()) {
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(
                                text = genesisState.assessedDetails,
                                style = MaterialTheme.typography.labelSmall,
                                color = ArgusCyanGlow,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 10.sp
                            )
                        }
                    }
                }
            }

            // 5. Lifecycle Checklist
            item {
                Surface(
                    shape = RoundedCornerShape(14.dp),
                    color = ArgusSurface,
                    border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder),
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Column(modifier = Modifier.padding(14.dp)) {
                        Text(
                            text = "TAHAPAN KELAHIRAN MODEL (LOCAL GENESIS)",
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusCyanPrimary,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 10.sp
                        )
                        Spacer(modifier = Modifier.height(10.dp))

                        val lifecycleStages = GenesisLifecycleStage.progressStages

                        val currentIdx = if (isFailed) {
                            (genesisState.progress * (lifecycleStages.size - 1))
                                .roundToInt()
                                .coerceIn(0, lifecycleStages.size - 1)
                        } else {
                            lifecycleStages.indexOf(genesisState.stage)
                        }

                        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            lifecycleStages.forEachIndexed { idx, stage ->
                                val isDone = !isFailed && (
                                    (currentIdx > idx) || (currentIdx == idx && isReady)
                                )
                                val isActive = !isFailed && currentIdx == idx && !isReady
                                val isFailurePoint = isFailed && idx == currentIdx

                                Row(
                                    modifier = Modifier.fillMaxWidth(),
                                    verticalAlignment = Alignment.CenterVertically
                                ) {
                                    when {
                                        isDone -> Icon(
                                            imageVector = Icons.Default.CheckCircle,
                                            contentDescription = "Done",
                                            tint = ArgusSuccess,
                                            modifier = Modifier.size(15.dp)
                                        )
                                        isFailurePoint -> Icon(
                                            imageVector = Icons.Default.Error,
                                            contentDescription = "Failed",
                                            tint = errorColor,
                                            modifier = Modifier.size(15.dp)
                                        )
                                        isActive -> CircularProgressIndicator(
                                            modifier = Modifier.size(14.dp),
                                            color = ArgusCyanPrimary,
                                            strokeWidth = 2.dp
                                        )
                                        else -> Box(
                                            modifier = Modifier
                                                .size(14.dp)
                                                .border(1.dp, ArgusTextMuted, CircleShape)
                                        )
                                    }
                                    Spacer(modifier = Modifier.width(8.dp))
                                    Column(modifier = Modifier.weight(1f)) {
                                        Text(
                                            text = stage.displayName,
                                            style = MaterialTheme.typography.titleMedium.copy(fontSize = 11.sp),
                                            color = when {
                                                isFailurePoint -> errorColor
                                                isDone || isActive -> ArgusTextPrimary
                                                else -> ArgusTextMuted
                                            },
                                            fontWeight = if (isActive || isFailurePoint) FontWeight.Bold else FontWeight.Normal
                                        )
                                        Text(
                                            text = stage.description,
                                            style = MaterialTheme.typography.labelSmall,
                                            color = when {
                                                isFailurePoint -> errorColor
                                                isActive -> ArgusCyanGlow
                                                else -> ArgusTextMuted
                                            },
                                            fontSize = 9.sp
                                        )
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // 6. Live Log
            if (genesisState.stageLog.isNotEmpty()) {
                item {
                    Surface(
                        shape = RoundedCornerShape(12.dp),
                        color = ArgusDarkBg,
                        border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Column(modifier = Modifier.padding(10.dp)) {
                            Text(
                                text = "LOG TELEMETRI GENESIS (REAL-TIME):",
                                style = MaterialTheme.typography.labelSmall,
                                color = ArgusCyanPrimary,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 9.sp,
                                fontWeight = FontWeight.Bold
                            )
                            Spacer(modifier = Modifier.height(4.dp))
                            genesisState.stageLog.takeLast(8).forEach { logLine ->
                                Text(
                                    text = "> $logLine",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = ArgusTextSecondary,
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 9.sp
                                )
                            }
                        }
                    }
                }
            }

            // 7. Action Button
            item {
                when {
                    isFailed -> Button(
                        onClick = { viewModel.retryGenesis() },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = errorColor,
                            contentColor = onErrorColor
                        ),
                        shape = RoundedCornerShape(12.dp),
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(50.dp)
                            .testTag("retry_genesis_button")
                    ) {
                        Icon(
                            imageVector = Icons.Default.Refresh,
                            contentDescription = null,
                            modifier = Modifier.size(18.dp)
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "COBA LAGI GENESIS",
                            fontWeight = FontWeight.Bold,
                            fontSize = 13.sp,
                            letterSpacing = 1.sp
                        )
                    }
                    isReady -> AnimatedVisibility(visible = true, enter = fadeIn()) {
                        Button(
                            onClick = {
                                viewModel.markGenesisBootSeen()
                                onFinish()
                            },
                            colors = ButtonDefaults.buttonColors(
                                containerColor = ArgusCyanPrimary,
                                contentColor = ArgusDarkBg
                            ),
                            shape = RoundedCornerShape(12.dp),
                            modifier = Modifier
                                .fillMaxWidth()
                                .height(50.dp)
                                .testTag("enter_terminal_button")
                        ) {
                            Icon(
                                imageVector = Icons.Default.CheckCircle,
                                contentDescription = null,
                                modifier = Modifier.size(18.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "MASUK TERMINAL AGI",
                                fontWeight = FontWeight.Bold,
                                fontSize = 13.sp,
                                letterSpacing = 1.sp
                            )
                        }
                    }
                    else -> {
                        // Genesis sedang berjalan — tidak ada tombol
                    }
                }
                Spacer(modifier = Modifier.height(16.dp))
            }
        }
    }
}