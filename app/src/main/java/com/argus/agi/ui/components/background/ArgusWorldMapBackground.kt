package com.argus.agi.ui.components.background

import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithCache
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.Stroke
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.unit.dp

/**
 * Static, device-independent world map background shared by the main Argus screens.
 *
 * Geometry is normalized to the available canvas size and drawing paths are cached
 * until the layout size changes. No animation, coroutine, timer, or per-frame state
 * is used by this renderer.
 */
@Composable
fun ArgusWorldMapBackground(
    modifier: Modifier = Modifier
) {
    val continents = remember { worldContinents() }
    val dots = remember(continents) { generateWorldMapDots(continents) }
    val hubs = remember { worldHubs() }
    val arcs = remember { worldArcs() }

    Box(
        modifier = modifier.drawWithCache {
            val continentPaths = continents.map { continent ->
                Path().apply {
                    continent.forEachIndexed { index, point ->
                        val x = point.x * size.width
                        val y = point.y * size.height
                        if (index == 0) {
                            moveTo(x, y)
                        } else {
                            lineTo(x, y)
                        }
                    }
                    close()
                }
            }

            val arcPaths = arcs.map { arc ->
                val start = Offset(
                    arc.from.x * size.width,
                    arc.from.y * size.height
                )
                val end = Offset(
                    arc.to.x * size.width,
                    arc.to.y * size.height
                )
                val control = Offset(
                    (start.x + end.x) / 2f,
                    minOf(start.y, end.y) - size.height * 0.07f
                )
                Path().apply {
                    moveTo(start.x, start.y)
                    quadraticBezierTo(
                        control.x,
                        control.y,
                        end.x,
                        end.y
                    )
                }
            }

            val backgroundBrush = Brush.verticalGradient(
                colors = listOf(
                    Color(0xFF04081A),
                    Color(0xFF071428),
                    Color(0xFF04081A)
                )
            )
            val bottomFadeBrush = Brush.verticalGradient(
                colors = listOf(
                    Color.Transparent,
                    Color(0xFF050B18).copy(alpha = 0.55f),
                    Color(0xFF050B18).copy(alpha = 0.85f)
                ),
                startY = size.height * 0.55f,
                endY = size.height
            )

            val continentOutlineWidth = 0.7.dp.toPx()
            val gridStrokeWidth = 0.55.dp.toPx()
            val arcStrokeWidth = 0.8.dp.toPx()
            val dotRadius = 1.05.dp.toPx()
            val highlightDotRadius = 1.55.dp.toPx()
            val hubOuterRadius = 5.5.dp.toPx()
            val hubMiddleRadius = 3.dp.toPx()
            val hubInnerRadius = 1.8.dp.toPx()
            val hubCoreRadius = 0.8.dp.toPx()

            onDrawBehind {
                drawRect(brush = backgroundBrush)

                continentPaths.forEach { path ->
                    drawPath(
                        path = path,
                        color = Color(0xFF00CFFF).copy(alpha = 0.07f)
                    )
                    drawPath(
                        path = path,
                        color = Color(0xFF00CFFF).copy(alpha = 0.18f),
                        style = Stroke(width = continentOutlineWidth)
                    )
                }

                val gridColor = Color(0xFF00CFFF).copy(alpha = 0.05f)
                for (index in 1..5) {
                    val x = size.width * index / 6f
                    drawLine(
                        color = gridColor,
                        start = Offset(x, 0f),
                        end = Offset(x, size.height),
                        strokeWidth = gridStrokeWidth
                    )
                }
                for (index in 1..3) {
                    val y = size.height * index / 4f
                    drawLine(
                        color = gridColor,
                        start = Offset(0f, y),
                        end = Offset(size.width, y),
                        strokeWidth = gridStrokeWidth
                    )
                }

                drawLine(
                    color = Color(0xFF00CFFF).copy(alpha = 0.09f),
                    start = Offset(0f, size.height * 0.52f),
                    end = Offset(size.width, size.height * 0.52f),
                    strokeWidth = gridStrokeWidth
                )

                dots.forEachIndexed { index, point ->
                    val hash = (
                        (point.x * 1_000).toInt() +
                            (point.y * 1_000).toInt() +
                            index
                        ) % 10
                    val alpha = 0.18f + (hash % 4) * 0.07f
                    val center = Offset(
                        point.x * size.width,
                        point.y * size.height
                    )

                    drawCircle(
                        color = Color(0xFF00D4FF).copy(alpha = alpha),
                        radius = dotRadius,
                        center = center
                    )
                    if (hash == 3) {
                        drawCircle(
                            color = Color(0xFF8FE3FF).copy(alpha = 0.32f),
                            radius = highlightDotRadius,
                            center = center
                        )
                    }
                }

                arcPaths.forEach { path ->
                    drawPath(
                        path = path,
                        color = Color(0xFF00CFFF).copy(alpha = 0.20f),
                        style = Stroke(
                            width = arcStrokeWidth,
                            cap = StrokeCap.Round
                        )
                    )
                }

                hubs.forEach { hub ->
                    val center = Offset(
                        hub.x * size.width,
                        hub.y * size.height
                    )
                    drawCircle(
                        color = Color(0xFF00CFFF).copy(alpha = 0.12f),
                        radius = hubOuterRadius,
                        center = center
                    )
                    drawCircle(
                        color = Color(0xFF00CFFF).copy(alpha = 0.25f),
                        radius = hubMiddleRadius,
                        center = center
                    )
                    drawCircle(
                        color = Color(0xFF8FE3FF),
                        radius = hubInnerRadius,
                        center = center
                    )
                    drawCircle(
                        color = Color.White,
                        radius = hubCoreRadius,
                        center = center
                    )
                }

                drawRect(brush = bottomFadeBrush)
            }
        }
    )
}

private data class WorldMapPoint(
    val x: Float,
    val y: Float
)

private data class WorldMapArc(
    val from: WorldMapPoint,
    val to: WorldMapPoint
)

private fun worldContinents(): List<List<WorldMapPoint>> = listOf(
    // North America
    listOf(
        WorldMapPoint(0.05f, 0.30f),
        WorldMapPoint(0.08f, 0.22f),
        WorldMapPoint(0.13f, 0.15f),
        WorldMapPoint(0.20f, 0.13f),
        WorldMapPoint(0.27f, 0.16f),
        WorldMapPoint(0.29f, 0.22f),
        WorldMapPoint(0.26f, 0.30f),
        WorldMapPoint(0.21f, 0.36f),
        WorldMapPoint(0.14f, 0.39f),
        WorldMapPoint(0.07f, 0.36f)
    ),
    // South America
    listOf(
        WorldMapPoint(0.29f, 0.50f),
        WorldMapPoint(0.35f, 0.46f),
        WorldMapPoint(0.39f, 0.54f),
        WorldMapPoint(0.37f, 0.64f),
        WorldMapPoint(0.33f, 0.73f),
        WorldMapPoint(0.29f, 0.68f),
        WorldMapPoint(0.27f, 0.56f)
    ),
    // Europe
    listOf(
        WorldMapPoint(0.46f, 0.18f),
        WorldMapPoint(0.52f, 0.14f),
        WorldMapPoint(0.58f, 0.18f),
        WorldMapPoint(0.59f, 0.25f),
        WorldMapPoint(0.54f, 0.30f),
        WorldMapPoint(0.47f, 0.28f)
    ),
    // Africa
    listOf(
        WorldMapPoint(0.47f, 0.34f),
        WorldMapPoint(0.56f, 0.31f),
        WorldMapPoint(0.62f, 0.39f),
        WorldMapPoint(0.62f, 0.48f),
        WorldMapPoint(0.58f, 0.59f),
        WorldMapPoint(0.52f, 0.66f),
        WorldMapPoint(0.48f, 0.54f)
    ),
    // Asia
    listOf(
        WorldMapPoint(0.60f, 0.16f),
        WorldMapPoint(0.74f, 0.12f),
        WorldMapPoint(0.87f, 0.17f),
        WorldMapPoint(0.92f, 0.25f),
        WorldMapPoint(0.90f, 0.33f),
        WorldMapPoint(0.82f, 0.39f),
        WorldMapPoint(0.70f, 0.39f),
        WorldMapPoint(0.60f, 0.30f)
    ),
    // Southeast Asia and Indonesia
    listOf(
        WorldMapPoint(0.73f, 0.53f),
        WorldMapPoint(0.82f, 0.52f),
        WorldMapPoint(0.87f, 0.58f),
        WorldMapPoint(0.84f, 0.64f),
        WorldMapPoint(0.74f, 0.63f)
    ),
    // Australia
    listOf(
        WorldMapPoint(0.80f, 0.61f),
        WorldMapPoint(0.89f, 0.59f),
        WorldMapPoint(0.93f, 0.68f),
        WorldMapPoint(0.87f, 0.76f),
        WorldMapPoint(0.79f, 0.72f)
    )
)

private fun worldHubs(): List<WorldMapPoint> = listOf(
    WorldMapPoint(0.80f, 0.58f), // Jakarta
    WorldMapPoint(0.88f, 0.29f), // Tokyo
    WorldMapPoint(0.50f, 0.20f), // London
    WorldMapPoint(0.20f, 0.27f), // New York
    WorldMapPoint(0.33f, 0.64f), // Sao Paulo
    WorldMapPoint(0.55f, 0.42f)  // Cairo
)

private fun worldArcs(): List<WorldMapArc> {
    val jakarta = WorldMapPoint(0.80f, 0.58f)
    val tokyo = WorldMapPoint(0.88f, 0.29f)
    val london = WorldMapPoint(0.50f, 0.20f)
    val newYork = WorldMapPoint(0.20f, 0.27f)
    val saoPaulo = WorldMapPoint(0.33f, 0.64f)

    return listOf(
        WorldMapArc(jakarta, tokyo),
        WorldMapArc(jakarta, london),
        WorldMapArc(newYork, london),
        WorldMapArc(london, saoPaulo),
        WorldMapArc(jakarta, saoPaulo)
    )
}

private fun generateWorldMapDots(
    continents: List<List<WorldMapPoint>>
): List<WorldMapPoint> {
    val points = ArrayList<WorldMapPoint>()
    var y = 0.12f

    while (y <= 0.78f) {
        var x = 0.03f
        while (x <= 0.96f) {
            if (continents.any { continent -> isPointInPolygon(x, y, continent) }) {
                points += WorldMapPoint(x, y)
            }
            x += 0.017f
        }
        y += 0.023f
    }

    return points
}

private fun isPointInPolygon(
    x: Float,
    y: Float,
    polygon: List<WorldMapPoint>
): Boolean {
    var inside = false
    var previousIndex = polygon.lastIndex

    for (index in polygon.indices) {
        val current = polygon[index]
        val previous = polygon[previousIndex]

        if ((current.y > y) != (previous.y > y)) {
            val intersectionX =
                (previous.x - current.x) * (y - current.y) /
                    (previous.y - current.y) + current.x
            if (x < intersectionX) {
                inside = !inside
            }
        }

        previousIndex = index
    }

    return inside
}
