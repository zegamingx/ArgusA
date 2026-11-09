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
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
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
import kotlinx.coroutines.delay
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random
import androidx.compose.ui.res.painterResource
import com.argus.agi.R

@Composable
fun GenesisBootScreen(
    viewModel: ArgusViewModel,
    onFinish: () -> Unit
) {
    val genesisState by viewModel.genesisState.collectAsState()

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

    val eyeOpen = (genesisState.progress * 1.4f).coerceIn(0.08f, 1f)

    // Transisi mata ke neuron, 0 = mata full, 1 = neuron full
    val neuronAlpha by animateFloatAsState(
        targetValue = if (isReady) 1f else 0f,
        animationSpec = tween(800, easing = FastOutSlowInEasing),
        label = "neuron_alpha"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(ArgusDarkBg)
            .padding(16.dp),
        contentAlignment = Alignment.Center
    ) {
        BackgroundGrid()
        ScanLineOverlay(scanY = scanY)

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
                    else "ARGUS GENESIS - FIRST-LAUNCH LIFECYCLE",
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

            // 2. EYE HUD + TRANSISI KE NEURON
            item {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(210.dp)
                        .clip(RoundedCornerShape(20.dp))
                        .background(ArgusSurface.copy(alpha = 0.7f))
                        .border(
                            1.5.dp,
                            if (isFailed) errorColor else ArgusCardBorder,
                            RoundedCornerShape(20.dp)
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    // Lapisan mata, memudar pas neuron muncul
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .alpha(1f - neuronAlpha)
                    ) {
                        ArgusEyeCanvas(
                            eyeOpen = eyeOpen,
                            isReady = false,
                            isFailed = isFailed,
                            scanY = scanY,
                            orbPulse = orbPulse
                        )
                    }

                    // Lapisan neuron, muncul pas isReady
                    if (neuronAlpha > 0.02f) {
                        Box(
                            modifier = Modifier
                                .fillMaxSize()
                                .alpha(neuronAlpha)
                        ) {
                            ArgusNeuronNetwork()
                        }
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
                                isReady -> "ARGUS GENESIS MODEL - ORIGIN READY"
                                else -> "SYNTHESIZING CORE - ${(animatedProgress * 100).toInt()}%"
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
                                    painter = painterResource(R.drawable.ic_symbol_error),
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
                                            painter = painterResource(R.drawable.ic_symbol_check_circle),
                                            contentDescription = "Done",
                                            tint = ArgusSuccess,
                                            modifier = Modifier.size(15.dp)
                                        )
                                        isFailurePoint -> Icon(
                                            painter = painterResource(R.drawable.ic_symbol_error),
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
                            painter = painterResource(R.drawable.ic_symbol_refresh),
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
                                painter = painterResource(R.drawable.ic_symbol_check_circle),
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
                        // Genesis sedang berjalan, tidak ada tombol
                    }
                }
                Spacer(modifier = Modifier.height(16.dp))
            }
        }
    }
}

@Composable
private fun BackgroundGrid() {
    Canvas(modifier = Modifier.fillMaxSize()) {
        val w = size.width
        val h = size.height
        val step = 60.dp.toPx()
        var x = 0f
        while (x < w) {
            drawLine(
                color = ArgusCyanPrimary.copy(alpha = 0.03f),
                start = Offset(x, 0f),
                end = Offset(x, h),
                strokeWidth = 1f
            )
            x += step
        }
        var y = 0f
        while (y < h) {
            drawLine(
                color = ArgusCyanPrimary.copy(alpha = 0.03f),
                start = Offset(0f, y),
                end = Offset(w, y),
                strokeWidth = 1f
            )
            y += step
        }
    }
}

@Composable
private fun ScanLineOverlay(scanY: Float) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        val w = size.width
        val h = size.height
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
}

@Composable
private fun ArgusEyeCanvas(
    eyeOpen: Float,
    isReady: Boolean,
    isFailed: Boolean,
    scanY: Float,
    orbPulse: Float
) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        val w = size.width
        val h = size.height
        val cx = w / 2f
        val cy = h / 2f
        val eyeW = w * 0.78f
        val eyeH = (eyeW * 0.32f) * eyeOpen

        if (!isReady && !isFailed) {
            for (i in 0 until 10) {
                val angleRad = Math.toRadians((i * 36.0) + 10.0)
                val sx = cx + (eyeW * 0.58f * cos(angleRad)).toFloat()
                val sy = cy + (eyeH * 2.4f * sin(angleRad)).toFloat()
                val ex = cx + (eyeW * 0.68f * cos(angleRad)).toFloat()
                val ey = cy + (eyeH * 3.1f * sin(angleRad)).toFloat()
                drawLine(
                    ArgusCyanPrimary.copy(alpha = 0.5f),
                    Offset(sx, sy), Offset(ex, ey),
                    strokeWidth = 2f
                )
                drawCircle(ArgusCyanGlow, 3f, Offset(ex, ey))
            }
        }

        if (isFailed) {
            val r = 32.dp.toPx()
            drawCircle(
                color = Color.Red.copy(alpha = 0.3f),
                radius = 55.dp.toPx(),
                center = Offset(cx, cy),
                style = Stroke(width = 2.dp.toPx())
            )
            drawLine(Color.Red, Offset(cx - r, cy - r), Offset(cx + r, cy + r), strokeWidth = 4f)
            drawLine(Color.Red, Offset(cx + r, cy - r), Offset(cx - r, cy + r), strokeWidth = 4f)
            return@Canvas
        }

        val eyePath = Path().apply {
            moveTo(cx - eyeW / 2, cy)
            quadraticTo(cx, cy - eyeH * 2.2f, cx + eyeW / 2, cy)
            quadraticTo(cx, cy + eyeH * 2.2f, cx - eyeW / 2, cy)
            close()
        }
        drawPath(eyePath, Color(0xFF0B1A3A))
        drawPath(eyePath, ArgusCyanPrimary, style = Stroke(width = 3f))

        if (eyeOpen > 0.3f) {
            val irisR = eyeH * 1.1f * if (isReady) orbPulse else 1f
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(Color.White, ArgusCyanGlow, ArgusCyanPrimary, Color.Transparent),
                    center = Offset(cx, cy),
                    radius = irisR
                ),
                radius = irisR,
                center = Offset(cx, cy)
            )
            drawCircle(Color(0xFF020617), radius = irisR * 0.32f, center = Offset(cx, cy))
            drawCircle(Color.White, radius = irisR * 0.1f, center = Offset(cx, cy))
        }

        if (!isReady) {
            val scanInsideY = cy - eyeH + (eyeH * 2f) * scanY
            drawLine(
                ArgusCyanGlow,
                Offset(cx - eyeW / 2, scanInsideY),
                Offset(cx + eyeW / 2, scanInsideY),
                strokeWidth = 3f,
                cap = StrokeCap.Round
            )
        }
    }
}

// Jaring neuron versi lite, kayak akhir video lu
private data class NeuronDot(var x: Float, var y: Float, var vx: Float, var vy: Float)

@Composable
private fun ArgusNeuronNetwork() {
    var dots by remember {
        mutableStateOf(
            List(38) {
                NeuronDot(
                    x = Random.nextFloat(),
                    y = Random.nextFloat(),
                    vx = (Random.nextFloat() - 0.5f) * 0.008f,
                    vy = (Random.nextFloat() - 0.5f) * 0.008f
                )
            }
        )
    }

    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val w = size.width
            val h = size.height

            // Garis antar titik yang deket
            for (i in dots.indices) {
                for (j in i + 1 until dots.size) {
                    val dx = (dots[i].x - dots[j].x) * w
                    val dy = (dots[i].y - dots[j].y) * h
                    val dist = sqrt(dx * dx + dy * dy)
                    if (dist < 200f) {
                        drawLine(
                            ArgusCyanPrimary.copy(alpha = 0.25f),
                            Offset(dots[i].x * w, dots[i].y * h),
                            Offset(dots[j].x * w, dots[j].y * h),
                            strokeWidth = 1f
                        )
                    }
                }
            }
            // Titiknya
            dots.forEach {
                drawCircle(
                    ArgusCyanGlow,
                    radius = 5f,
                    center = Offset(it.x * w, it.y * h)
                )
            }
            // Pusat terang di tengah, bekas pupil mata
            drawCircle(
                brush = Brush.radialGradient(
                    colors = listOf(
                        Color.White.copy(alpha = 0.9f),
                        ArgusCyanGlow.copy(alpha = 0.5f),
                        Color.Transparent
                    ),
                    center = Offset(w / 2f, h / 2f),
                    radius = 90f
                ),
                radius = 90f,
                center = Offset(w / 2f, h / 2f)
            )
        }

        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                text = "Argus Genesis Model",
                color = Color.White,
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace
            )
            Text(
                text = "NEURAL NETWORK ONLINE",
                color = ArgusCyanGlow,
                fontSize = 9.sp,
                fontFamily = FontFamily.Monospace,
                letterSpacing = 1.5.sp
            )
        }
    }

    // Gerakin titik, 30fps aja biar enteng
    LaunchedEffect(Unit) {
        while (true) {
            delay(33)
            dots = dots.map {
                var nx = it.x + it.vx
                var ny = it.y + it.vy
                var nvx = it.vx
                var nvy = it.vy
                if (nx < 0f || nx > 1f) nvx = -nvx
                if (ny < 0f || ny > 1f) nvy = -nvy
                nx = nx.coerceIn(0f, 1f)
                ny = ny.coerceIn(0f, 1f)
                NeuronDot(nx, ny, nvx, nvy)
            }
        }
    }
}
