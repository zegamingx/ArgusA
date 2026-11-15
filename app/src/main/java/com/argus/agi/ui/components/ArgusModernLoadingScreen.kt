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
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.argus.agi.ui.theme.ArgusDarkBg
import com.argus.agi.ui.theme.ArgusTextMuted
import com.argus.agi.ui.theme.ArgusTextSecondary
import kotlin.math.ceil

@Composable
fun ArgusModernLoadingScreen(
    title: String = "ARGUS",
    subtitle: String = "Restoring the neural core",
    progressPercent: Int? = null,
    driverInfo: String = "Local neural runtime"
) {
    val determinateProgress = progressPercent?.coerceIn(0, 100)?.div(100f)
    val animatedProgress by animateFloatAsState(
        targetValue = determinateProgress ?: 0.22f,
        animationSpec = tween(320, easing = FastOutSlowInEasing),
        label = "startup_progress"
    )

    val transition = rememberInfiniteTransition(label = "startup_eye")
    val sweepRotation by transition.animateFloat(
        initialValue = -90f,
        targetValue = 270f,
        animationSpec = infiniteRepeatable(
            animation = tween(2200, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "startup_sweep"
    )
    val pulse by transition.animateFloat(
        initialValue = 0.94f,
        targetValue = 1.04f,
        animationSpec = infiniteRepeatable(
            animation = tween(1800, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "startup_pulse"
    )

    val progressColor = Color(0xFF8CA5CC)
    val ringTrack = Color(0xFF26344A)
    val activeDots = if (determinateProgress == null) {
        2 + ((sweepRotation + 90f) / 360f * 3f).toInt().coerceIn(0, 2)
    } else {
        ceil(animatedProgress * 5f).toInt().coerceIn(0, 5)
    }
    val percentLabel = determinateProgress?.let { "${(animatedProgress * 100f).toInt()}%" } ?: "..."

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(ArgusDarkBg)
            .padding(horizontal = 24.dp, vertical = 22.dp)
    ) {
        Column(modifier = Modifier.fillMaxSize()) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = title,
                    color = ArgusTextSecondary,
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.Medium,
                    fontSize = 12.sp,
                    letterSpacing = 1.6.sp
                )
                Text(
                    text = "STATUS: ${subtitle.uppercase()}${determinateProgress?.let { " · $percentLabel" } ?: ""}",
                    modifier = Modifier.width(230.dp),
                    color = ArgusTextMuted,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 9.sp,
                    letterSpacing = 0.3.sp,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    textAlign = androidx.compose.ui.text.style.TextAlign.End
                )
            }

            Spacer(modifier = Modifier.weight(1f))

            Column(
                modifier = Modifier.fillMaxWidth(),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(198.dp),
                    contentAlignment = Alignment.Center
                ) {
                    Canvas(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(190.dp)
                    ) {
                        val centerX = size.width / 2f
                        val centerY = size.height / 2f
                        val halfWidth = minOf(size.width * 0.43f, 190.dp.toPx())
                        val eyeHeight = 48.dp.toPx()
                        val left = centerX - halfWidth
                        val right = centerX + halfWidth

                        val eyeOutline = Path().apply {
                            moveTo(left, centerY)
                            cubicTo(
                                centerX - halfWidth * 0.52f,
                                centerY - eyeHeight,
                                centerX + halfWidth * 0.52f,
                                centerY - eyeHeight,
                                right,
                                centerY
                            )
                            cubicTo(
                                centerX + halfWidth * 0.52f,
                                centerY + eyeHeight,
                                centerX - halfWidth * 0.52f,
                                centerY + eyeHeight,
                                left,
                                centerY
                            )
                            close()
                        }
                        drawPath(
                            path = eyeOutline,
                            color = progressColor.copy(alpha = 0.58f),
                            style = Stroke(width = 1.7.dp.toPx())
                        )
                    }

                    Box(
                        modifier = Modifier.size(122.dp),
                        contentAlignment = Alignment.Center
                    ) {
                        Canvas(modifier = Modifier.fillMaxSize()) {
                            val center = Offset(size.width / 2f, size.height / 2f)
                            val radius = size.minDimension * 0.43f
                            drawCircle(
                                color = Color(0xFF19263A).copy(alpha = 0.96f),
                                radius = radius,
                                center = center
                            )
                            if (determinateProgress != null) {
                                drawArc(
                                    color = progressColor.copy(alpha = 0.22f),
                                    startAngle = -90f,
                                    sweepAngle = animatedProgress * 360f,
                                    useCenter = true,
                                    topLeft = Offset(center.x - radius, center.y - radius),
                                    size = androidx.compose.ui.geometry.Size(radius * 2f, radius * 2f)
                                )
                            }
                            drawCircle(
                                color = progressColor.copy(alpha = 0.72f),
                                radius = radius,
                                center = center,
                                style = Stroke(width = 2.dp.toPx())
                            )
                            drawCircle(
                                color = ringTrack,
                                radius = radius + 5.dp.toPx(),
                                center = center,
                                style = Stroke(width = 1.dp.toPx())
                            )
                            val arcSweep = if (determinateProgress != null) animatedProgress * 360f else 94f
                            val arcStart = if (determinateProgress != null) -90f else sweepRotation
                            drawArc(
                                color = progressColor.copy(alpha = pulse.coerceIn(0f, 1f)),
                                startAngle = arcStart,
                                sweepAngle = arcSweep.coerceAtLeast(2f),
                                useCenter = false,
                                topLeft = Offset(center.x - radius - 5.dp.toPx(), center.y - radius - 5.dp.toPx()),
                                size = androidx.compose.ui.geometry.Size(
                                    (radius + 5.dp.toPx()) * 2f,
                                    (radius + 5.dp.toPx()) * 2f
                                ),
                                style = Stroke(width = 2.dp.toPx(), cap = StrokeCap.Round)
                            )
                        }
                        Text(
                            text = percentLabel,
                            color = Color(0xFFD7E2F2),
                            fontSize = 24.sp,
                            fontWeight = FontWeight.Medium,
                            fontFamily = FontFamily.Monospace,
                            letterSpacing = 0.2.sp
                        )
                    }
                }

                Spacer(modifier = Modifier.height(22.dp))

                Text(
                    text = subtitle,
                    color = progressColor,
                    style = MaterialTheme.typography.titleMedium,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Medium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis
                )

                Spacer(modifier = Modifier.height(18.dp))

                Canvas(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 10.dp)
                        .height(3.dp)
                        .clip(RoundedCornerShape(2.dp))
                ) {
                    drawLine(
                        color = ringTrack,
                        start = Offset(0f, size.height / 2f),
                        end = Offset(size.width, size.height / 2f),
                        strokeWidth = size.height,
                        cap = StrokeCap.Round
                    )
                    val fillWidth = if (determinateProgress != null) {
                        size.width * animatedProgress
                    } else {
                        size.width * 0.28f
                    }
                    val startX = if (determinateProgress != null) {
                        0f
                    } else {
                        ((size.width + fillWidth) * (sweepRotation + 90f) / 360f) - fillWidth
                    }.coerceIn(0f, size.width)
                    val endX = if (determinateProgress != null) {
                        fillWidth
                    } else {
                        (startX + fillWidth).coerceAtMost(size.width)
                    }
                    drawLine(
                        color = progressColor,
                        start = Offset(startX, size.height / 2f),
                        end = Offset(endX, size.height / 2f),
                        strokeWidth = size.height,
                        cap = StrokeCap.Round
                    )
                }

                Spacer(modifier = Modifier.height(24.dp))

                Row(
                    horizontalArrangement = Arrangement.spacedBy(9.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    repeat(5) { index ->
                        Box(
                            modifier = Modifier
                                .size(7.dp)
                                .clip(CircleShape)
                                .background(if (index < activeDots) progressColor else ringTrack)
                        )
                    }
                }
            }

            Spacer(modifier = Modifier.weight(1f))

            Text(
                text = driverInfo,
                modifier = Modifier.fillMaxWidth(),
                color = ArgusTextMuted,
                fontFamily = FontFamily.Monospace,
                fontSize = 9.sp,
                letterSpacing = 0.3.sp,
                textAlign = androidx.compose.ui.text.style.TextAlign.Center,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
    }
}
