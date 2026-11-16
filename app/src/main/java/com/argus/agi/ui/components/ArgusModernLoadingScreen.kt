package com.argus.agi.ui.components

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
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
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.argus.agi.ui.theme.ArgusCyanGlow
import com.argus.agi.ui.theme.ArgusCyanPrimary
import com.argus.agi.ui.theme.ArgusDarkBg
import com.argus.agi.ui.theme.ArgusSurface
import com.argus.agi.ui.theme.ArgusTextMuted
import kotlin.math.cos
import kotlin.math.sin

@Composable
fun ArgusModernLoadingScreen(
    title: String = "A . R . G . U . S",
    subtitle: String = "INITIALIZING LOCAL NEURAL RUNTIME",
    progressPercent: Int? = null,
    supportingText: String? = null,
    driverInfo: String = "STANDALONE ON-DEVICE ARCHITECTURE",
    isError: Boolean = false,
    onRetry: (() -> Unit)? = null
) {
    val accentColor = if (isError) MaterialTheme.colorScheme.error else ArgusCyanPrimary
    val glowColor = if (isError) accentColor.copy(alpha = 0.82f) else ArgusCyanGlow
    val safeProgressPercent = progressPercent?.coerceIn(0, 100)
    val determinateProgress = safeProgressPercent?.div(100f)

    val animatedProgress by animateFloatAsState(
        targetValue = determinateProgress ?: 0f,
        animationSpec = tween(320, easing = FastOutSlowInEasing),
        label = "loading_progress"
    )

    val infiniteTransition = rememberInfiniteTransition(label = "loading_infinite")
    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(4_000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "reactor_rotation"
    )
    val pulse by infiniteTransition.animateFloat(
        initialValue = 0.85f,
        targetValue = 1.15f,
        animationSpec = infiniteRepeatable(
            animation = tween(1_400, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "reactor_pulse"
    )
    val scanLine by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(2_000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "progress_scan"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(ArgusDarkBg),
        contentAlignment = Alignment.Center
    ) {
        Canvas(modifier = Modifier.fillMaxSize()) {
            val step = 48.dp.toPx()
            var x = 0f
            while (x < size.width) {
                drawLine(
                    color = accentColor.copy(alpha = 0.035f),
                    start = Offset(x, 0f),
                    end = Offset(x, size.height),
                    strokeWidth = 1f
                )
                x += step
            }

            var y = 0f
            while (y < size.height) {
                drawLine(
                    color = accentColor.copy(alpha = 0.035f),
                    start = Offset(0f, y),
                    end = Offset(size.width, y),
                    strokeWidth = 1f
                )
                y += step
            }
        }

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.Center)
                .padding(horizontal = 24.dp, vertical = 28.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Box(
                modifier = Modifier.size(160.dp),
                contentAlignment = Alignment.Center
            ) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val centerX = size.width / 2f
                    val centerY = size.height / 2f
                    val baseRadius = size.width * 0.42f

                    for (index in 0 until 12) {
                        val angleDegrees = rotation + index * 30f
                        val angleRadians = Math.toRadians(angleDegrees.toDouble())
                        val innerRadius = baseRadius - 8.dp.toPx()
                        val start = Offset(
                            (centerX + innerRadius * cos(angleRadians)).toFloat(),
                            (centerY + innerRadius * sin(angleRadians)).toFloat()
                        )
                        val end = Offset(
                            (centerX + baseRadius * cos(angleRadians)).toFloat(),
                            (centerY + baseRadius * sin(angleRadians)).toFloat()
                        )
                        drawLine(
                            color = accentColor.copy(alpha = 0.45f),
                            start = start,
                            end = end,
                            strokeWidth = 2.dp.toPx(),
                            cap = StrokeCap.Round
                        )
                    }

                    drawArc(
                        brush = Brush.sweepGradient(
                            listOf(Color.Transparent, accentColor, glowColor, Color.Transparent)
                        ),
                        startAngle = rotation,
                        sweepAngle = 220f,
                        useCenter = false,
                        style = Stroke(width = 2.5.dp.toPx(), cap = StrokeCap.Round),
                        size = androidx.compose.ui.geometry.Size(
                            baseRadius * 2f,
                            baseRadius * 2f
                        ),
                        topLeft = Offset(centerX - baseRadius, centerY - baseRadius)
                    )

                    val innerRadius = baseRadius * 0.75f
                    drawArc(
                        brush = Brush.sweepGradient(
                            listOf(Color.Transparent, glowColor, Color.Transparent)
                        ),
                        startAngle = -rotation * 1.5f,
                        sweepAngle = 140f,
                        useCenter = false,
                        style = Stroke(width = 1.5.dp.toPx(), cap = StrokeCap.Round),
                        size = androidx.compose.ui.geometry.Size(
                            innerRadius * 2f,
                            innerRadius * 2f
                        ),
                        topLeft = Offset(centerX - innerRadius, centerY - innerRadius)
                    )

                    val coreRadius = 24.dp.toPx() * pulse
                    drawCircle(
                        brush = Brush.radialGradient(
                            colors = listOf(
                                Color.White,
                                glowColor,
                                accentColor.copy(alpha = 0.4f),
                                Color.Transparent
                            ),
                            center = Offset(centerX, centerY),
                            radius = coreRadius * 1.8f
                        ),
                        radius = coreRadius * 1.8f,
                        center = Offset(centerX, centerY)
                    )
                    drawCircle(
                        color = Color.White,
                        radius = 4.dp.toPx(),
                        center = Offset(centerX, centerY)
                    )
                }
            }

            Spacer(modifier = Modifier.height(32.dp))

            Text(
                text = title,
                style = MaterialTheme.typography.headlineSmall,
                color = accentColor,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace,
                letterSpacing = 6.sp,
                textAlign = TextAlign.Center,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = subtitle,
                style = MaterialTheme.typography.labelSmall,
                color = glowColor,
                fontFamily = FontFamily.Monospace,
                letterSpacing = 1.5.sp,
                fontSize = 10.sp,
                textAlign = TextAlign.Center,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )

            if (!supportingText.isNullOrBlank()) {
                Spacer(modifier = Modifier.height(10.dp))
                Text(
                    text = supportingText,
                    style = MaterialTheme.typography.bodySmall,
                    color = ArgusTextMuted,
                    fontSize = 11.sp,
                    textAlign = TextAlign.Center,
                    maxLines = 3,
                    overflow = TextOverflow.Ellipsis
                )
            }

            Spacer(modifier = Modifier.height(28.dp))

            Box(
                modifier = Modifier
                    .width(220.dp)
                    .height(3.dp)
                    .clip(RoundedCornerShape(1.5.dp))
                    .background(ArgusSurface)
            ) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val centerY = size.height / 2f
                    if (determinateProgress != null) {
                        drawLine(
                            color = accentColor,
                            start = Offset(0f, centerY),
                            end = Offset(size.width * animatedProgress.coerceIn(0f, 1f), centerY),
                            strokeWidth = size.height,
                            cap = StrokeCap.Round
                        )
                    } else {
                        val scanWidth = size.width * 0.35f
                        val startX = (size.width + scanWidth) * scanLine - scanWidth
                        val endX = startX + scanWidth
                        drawLine(
                            brush = Brush.horizontalGradient(
                                colors = listOf(
                                    Color.Transparent,
                                    accentColor,
                                    Color.White,
                                    accentColor,
                                    Color.Transparent
                                ),
                                startX = startX,
                                endX = endX
                            ),
                            start = Offset(startX, centerY),
                            end = Offset(endX, centerY),
                            strokeWidth = size.height,
                            cap = StrokeCap.Round
                        )
                    }
                }
            }

            if (safeProgressPercent != null) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = safeProgressPercent.toString() + "%",
                    style = MaterialTheme.typography.labelSmall,
                    color = glowColor,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 9.sp
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center
            ) {
                Box(
                    modifier = Modifier
                        .size(6.dp)
                        .clip(CircleShape)
                        .background(glowColor)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = driverInfo,
                    style = MaterialTheme.typography.labelSmall,
                    color = ArgusTextMuted,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 9.sp,
                    textAlign = TextAlign.Center,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )
            }

            if (isError && onRetry != null) {
                Spacer(modifier = Modifier.height(14.dp))
                TextButton(onClick = onRetry) {
                    Text(
                        text = "RETRY GENESIS",
                        color = accentColor,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold,
                        fontSize = 10.sp,
                        letterSpacing = 1.sp
                    )
                }
            }
        }
    }
}
