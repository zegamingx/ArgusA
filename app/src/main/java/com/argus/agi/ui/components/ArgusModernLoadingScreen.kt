package com.argus.agi.ui.components

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.argus.agi.ui.theme.ArgusCyanGlow
import com.argus.agi.ui.theme.ArgusCyanPrimary
import com.argus.agi.ui.theme.ArgusDarkBg
import com.argus.agi.ui.theme.ArgusSurface
import com.argus.agi.ui.theme.ArgusTextMuted
import com.argus.agi.ui.theme.ArgusTextSecondary
import kotlin.math.cos
import kotlin.math.sin

@Composable
fun ArgusModernLoadingScreen(
    title: String = "A . R . G . U . S",
    subtitle: String = "RESTORING THE NEURAL CORE & CHAT HISTORY..."
) {
    val infiniteTransition = rememberInfiniteTransition(label = "loading_infinite")

    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(4000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "rotation"
    )

    val pulse by infiniteTransition.animateFloat(
        initialValue = 0.85f,
        targetValue = 1.15f,
        animationSpec = infiniteRepeatable(
            animation = tween(1400, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse"
    )

    val scanLine by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(2000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "scanLine"
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(ArgusDarkBg),
        contentAlignment = Alignment.Center
    ) {
        // Background Cybernetic Grid
        Canvas(modifier = Modifier.fillMaxSize()) {
            val step = 48.dp.toPx()
            var x = 0f
            while (x < size.width) {
                drawLine(
                    color = ArgusCyanPrimary.copy(alpha = 0.035f),
                    start = Offset(x, 0f),
                    end = Offset(x, size.height),
                    strokeWidth = 1f
                )
                x += step
            }
            var y = 0f
            while (y < size.height) {
                drawLine(
                    color = ArgusCyanPrimary.copy(alpha = 0.035f),
                    start = Offset(0f, y),
                    end = Offset(size.width, y),
                    strokeWidth = 1f
                )
                y += step
            }
        }

        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
            modifier = Modifier.padding(24.dp)
        ) {
            // Cybernetic Reactor Graphic
            Box(
                modifier = Modifier.size(160.dp),
                contentAlignment = Alignment.Center
            ) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val cx = size.width / 2f
                    val cy = size.height / 2f
                    val baseR = size.width * 0.42f

                    // Outer Orbit with ticks
                    for (i in 0 until 12) {
                        val angleDeg = rotation + (i * 30f)
                        val angleRad = Math.toRadians(angleDeg.toDouble())
                        val p1 = Offset(
                            (cx + (baseR - 8.dp.toPx()) * cos(angleRad)).toFloat(),
                            (cy + (baseR - 8.dp.toPx()) * sin(angleRad)).toFloat()
                        )
                        val p2 = Offset(
                            (cx + baseR * cos(angleRad)).toFloat(),
                            (cy + baseR * sin(angleRad)).toFloat()
                        )
                        drawLine(
                            color = ArgusCyanPrimary.copy(alpha = 0.45f),
                            start = p1,
                            end = p2,
                            strokeWidth = 2.dp.toPx(),
                            cap = StrokeCap.Round
                        )
                    }

                    // Dual Rotating Ring Arcs
                    drawArc(
                        brush = Brush.sweepGradient(
                            listOf(Color.Transparent, ArgusCyanPrimary, ArgusCyanGlow, Color.Transparent)
                        ),
                        startAngle = rotation,
                        sweepAngle = 220f,
                        useCenter = false,
                        style = Stroke(width = 2.5.dp.toPx(), cap = StrokeCap.Round),
                        size = androidx.compose.ui.geometry.Size(baseR * 2, baseR * 2),
                        topLeft = Offset(cx - baseR, cy - baseR)
                    )

                    drawArc(
                        brush = Brush.sweepGradient(
                            listOf(Color.Transparent, ArgusCyanGlow, Color.Transparent)
                        ),
                        startAngle = -rotation * 1.5f,
                        sweepAngle = 140f,
                        useCenter = false,
                        style = Stroke(width = 1.5.dp.toPx(), cap = StrokeCap.Round),
                        size = androidx.compose.ui.geometry.Size(baseR * 1.5f, baseR * 1.5f),
                        topLeft = Offset(cx - baseR * 0.75f, cy - baseR * 0.75f)
                    )

                    // Core Reactor Glow
                    val coreR = 24.dp.toPx() * pulse
                    drawCircle(
                        brush = Brush.radialGradient(
                            colors = listOf(
                                Color.White,
                                ArgusCyanGlow,
                                ArgusCyanPrimary.copy(alpha = 0.4f),
                                Color.Transparent
                            ),
                            center = Offset(cx, cy),
                            radius = coreR * 1.8f
                        ),
                        radius = coreR * 1.8f,
                        center = Offset(cx, cy)
                    )

                    // Inner Dot
                    drawCircle(
                        color = Color.White,
                        radius = 4.dp.toPx(),
                        center = Offset(cx, cy)
                    )
                }
            }

            Spacer(modifier = Modifier.height(32.dp))

            Text(
                text = title,
                style = MaterialTheme.typography.headlineSmall,
                color = ArgusCyanPrimary,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace,
                letterSpacing = 6.sp
            )

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = subtitle,
                style = MaterialTheme.typography.labelSmall,
                color = ArgusCyanGlow,
                fontFamily = FontFamily.Monospace,
                letterSpacing = 2.sp,
                fontSize = 10.sp
            )

            Spacer(modifier = Modifier.height(28.dp))

            // Sleek Cyber Progress Indicator
            Box(
                modifier = Modifier
                    .width(220.dp)
                    .height(3.dp)
                    .clip(RoundedCornerShape(1.5.dp))
                    .background(ArgusSurface)
            ) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val scanW = size.width * 0.35f
                    val startX = (size.width + scanW) * scanLine - scanW
                    drawLine(
                        brush = Brush.horizontalGradient(
                            listOf(
                                Color.Transparent,
                                ArgusCyanPrimary,
                                Color.White,
                                ArgusCyanPrimary,
                                Color.Transparent
                            ),
                            startX = startX,
                            endX = startX + scanW
                        ),
                        start = Offset(startX, size.height / 2f),
                        end = Offset(startX + scanW, size.height / 2f),
                        strokeWidth = size.height
                    )
                }
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
                        .background(ArgusCyanGlow)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "STANDALONE ON-DEVICE ARCHITECTURE",
                    style = MaterialTheme.typography.labelSmall,
                    color = ArgusTextMuted,
                    fontFamily = FontFamily.Monospace,
                    fontSize = 9.sp
                )
            }
        }
    }
}
