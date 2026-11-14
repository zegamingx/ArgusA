package com.argus.agi.ui.components

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
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
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.argus.agi.model.GenesisIdentity
import com.argus.agi.sensor.SystemHardwareInfo
import com.argus.agi.ui.theme.ArgusBlueAccent
import com.argus.agi.ui.theme.ArgusCardBorder
import com.argus.agi.ui.theme.ArgusCyanGlow
import com.argus.agi.ui.theme.ArgusCyanPrimary
import com.argus.agi.ui.theme.ArgusDarkBg
import com.argus.agi.ui.theme.ArgusSuccess
import com.argus.agi.ui.theme.ArgusSurface
import com.argus.agi.ui.theme.ArgusTextMuted
import com.argus.agi.ui.theme.ArgusTextPrimary
import com.argus.agi.ui.theme.ArgusTextSecondary
import com.argus.agi.ui.theme.ArgusWarning
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt
import androidx.compose.ui.res.painterResource
import com.argus.agi.R

/**
 * Representasi simpul neuron dalam koordinat 3D bola kortikal otak.
 */
private data class NeuronNode3D(
    val id: Int,
    val x: Float, // -1f .. +1f (kiri -> kanan)
    val y: Float, // -1f .. +1f (atas -> bawah)
    val z: Float, // -1f .. +1f (belakang -> depan)
    val region: BrainRegion
)

private enum class BrainRegion(val label: String, val baseColor: Color) {
    LEFT_CORTEX("Left Cortex", Color(0xFF00E5FF)),
    RIGHT_CORTEX("Right Cortex", Color(0xFF00B0FF)),
    FRONTAL_LOBE("Frontal Lobe", Color(0xFF18FFFF)),
    OCCIPITAL_LOBE("Occipital Lobe", Color(0xFF7C4DFF)),
    TEMPORAL_LOBE("Temporal Lobe", Color(0xFF00BFA5)),
    CORPUS_CALLOSUM("Corpus Callosum", Color(0xFFFFD54F)),
    CEREBELLUM("Cerebellum", Color(0xFF64FFDA))
}

private data class SynapseEdge(
    val fromIdx: Int,
    val toIdx: Int,
    val baseWeight: Float
)

/**
 * Satu-satunya animasi visual di Neural Core:
 * Animasi 3D Neuron Otak (3D Interactive Brain Neuron Network)
 * - Proyeksi 3D Euler nyata dengan depth-scaling & z-sorting
 * - Interaktif: dapat disentuh dan diputar bebas (drag gesture)
 * - Impuls sinaptik elektrik berjalan aktif mengikuti state training & nilai loss nyata
 * - Menggantikan HUD matrix naik-turun yang berantakan dengan pusat neuron holistik
 */
@Composable
fun NeuronBrain3DCard(
    hardwareInfo: SystemHardwareInfo,
    identity: GenesisIdentity?,
    isTrainingActive: Boolean,
    currentLoss: Float?,
    onReplayBootSequence: (() -> Unit)? = null,
    modifier: Modifier = Modifier
) {
    // Bangun topologi 3D neuron otak (hanya sekali per lifecycle)
    val brainNodes = remember { generateBrainTopology() }
    val synapses = remember { generateSynapseNetwork(brainNodes) }

    // Sudut rotasi bebas (drag gesture)
    var dragYaw by remember { mutableFloatStateOf(0f) }
    var dragPitch by remember { mutableFloatStateOf(0f) }

    // Animasi rotasi otomatis dan impuls sinapsis
    val infiniteTransition = rememberInfiniteTransition(label = "brain_engine_rot")

    // Rotasi yaw otomatis halus
    val autoYaw by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(18000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "auto_yaw"
    )

    // Osilasi pitch biologis halus
    val autoPitch by infiniteTransition.animateFloat(
        initialValue = -12f,
        targetValue = 12f,
        animationSpec = infiniteRepeatable(
            animation = tween(9000, easing = LinearEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "auto_pitch"
    )

    // Impuls sinaptik elektrik (kecepatan meningkat saat aktif training)
    val impulseDuration = if (isTrainingActive) 1600 else 3400
    val impulseProgress by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(impulseDuration, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "synaptic_impulse"
    )

    // Warna impuls aktif menyesuaikan real loss
    val impulseColor = when {
        currentLoss == null -> ArgusCyanPrimary
        currentLoss < 1.5f -> ArgusCyanGlow
        currentLoss < 3.0f -> Color(0xFF64FFDA)
        else -> Color(0xFFFFB74D) // Amber/Gold saat loss awal adaptasi
    }

    Surface(
        shape = RoundedCornerShape(16.dp),
        color = ArgusSurface,
        border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder),
        modifier = modifier.fillMaxWidth()
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            // Header: Judul Cortex & Status Real Hardware
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .clip(CircleShape)
                                .background(if (isTrainingActive) ArgusCyanGlow else ArgusSuccess)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "3D NEURAL NETWORK VISUALIZATION",
                            style = MaterialTheme.typography.titleMedium,
                            color = ArgusCyanPrimary,
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 1.sp
                        )
                    }
                    Text(
                        text = if (isTrainingActive) "Training active • visualization follows training state" else "Idle visualization",
                        style = MaterialTheme.typography.labelSmall,
                        color = if (isTrainingActive) ArgusCyanGlow else ArgusTextSecondary
                    )
                }

                Box(
                    modifier = Modifier
                        .border(1.dp, ArgusCyanPrimary.copy(alpha = 0.5f), RoundedCornerShape(6.dp))
                        .background(ArgusCyanPrimary.copy(alpha = 0.12f))
                        .padding(horizontal = 8.dp, vertical = 3.dp)
                ) {
                    Text(
                        text = identity?.originId ?: "CORTEX ACTIVE",
                        style = MaterialTheme.typography.labelSmall,
                        color = ArgusCyanGlow,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold,
                        fontSize = 10.sp
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Canvas 3D Neuron Otak
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(240.dp)
                    .clip(RoundedCornerShape(14.dp))
                    .background(
                        Brush.radialGradient(
                            colors = listOf(
                                Color(0xFF0D1B2A),
                                ArgusDarkBg
                            )
                        )
                    )
                    .border(1.dp, ArgusCardBorder, RoundedCornerShape(14.dp))
                    .pointerInput(Unit) {
                        detectDragGestures { change, dragAmount ->
                            change.consume()
                            dragYaw += dragAmount.x * 0.45f
                            dragPitch = (dragPitch - dragAmount.y * 0.45f).coerceIn(-60f, 60f)
                        }
                    }
                    .testTag("neuron_brain_3d_canvas"),
                contentAlignment = Alignment.Center
            ) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val cx = size.width / 2f
                    val cy = size.height / 2f
                    val brainRadius = minOf(size.width, size.height) * 0.42f
                    val focalLength = 2.4f

                    // Hitung total rotasi Euler (Auto + Gesture Drag)
                    val totalYawRad = Math.toRadians((autoYaw + dragYaw).toDouble()).toFloat()
                    val totalPitchRad = Math.toRadians((autoPitch + dragPitch).toDouble()).toFloat()

                    val cosY = cos(totalYawRad)
                    val sinY = sin(totalYawRad)
                    val cosP = cos(totalPitchRad)
                    val sinP = sin(totalPitchRad)

                    // 1. Proyeksikan seluruh simpul 3D ke layar 2D dengan z-sorting
                    data class ProjectedNode(
                        val node: NeuronNode3D,
                        val sx: Float,
                        val sy: Float,
                        val sz: Float,
                        val scale: Float,
                        val alpha: Float
                    )

                    val projected = brainNodes.map { node ->
                        // Rotasi Yaw (mengitari sumbu Y)
                        val x1 = node.x * cosY + node.z * sinY
                        val z1 = -node.x * sinY + node.z * cosY

                        // Rotasi Pitch (mengitari sumbu X)
                        val y2 = node.y * cosP - z1 * sinP
                        val z2 = node.y * sinP + z1 * cosP

                        val scale = focalLength / (focalLength + z2)
                        val screenX = cx + x1 * scale * brainRadius
                        val screenY = cy + y2 * scale * brainRadius
                        val depthAlpha = ((z2 + 1f) / 2f).coerceIn(0.2f, 1f)

                        ProjectedNode(node, screenX, screenY, z2, scale, depthAlpha)
                    }

                    // 2. Gambar akson sinapsis antar simpul (diurutkan berdasarkan kedalaman rata-rata)
                    synapses.forEach { synapse ->
                        val p1 = projected[synapse.fromIdx]
                        val p2 = projected[synapse.toIdx]
                        val avgZ = (p1.sz + p2.sz) / 2f
                        val edgeAlpha = (((avgZ + 1f) / 2f).coerceIn(0.12f, 0.75f) * synapse.baseWeight)

                        // Gambar garis akson sinapsis
                        val isCrossHemisphere = p1.node.region == BrainRegion.CORPUS_CALLOSUM ||
                                p2.node.region == BrainRegion.CORPUS_CALLOSUM ||
                                (p1.node.x * p2.node.x < 0)

                        val axonColor = if (isCrossHemisphere) {
                            Color(0xFFFFD54F).copy(alpha = edgeAlpha * 0.8f)
                        } else {
                            ArgusCyanPrimary.copy(alpha = edgeAlpha * 0.6f)
                        }

                        drawLine(
                            color = axonColor,
                            start = Offset(p1.sx, p1.sy),
                            end = Offset(p2.sx, p2.sy),
                            strokeWidth = (1.2.dp.toPx() * ((avgZ + 1.2f) / 2f)).coerceAtLeast(0.6f),
                            cap = StrokeCap.Round
                        )

                        // 3. Gambar potensial aksi listrik (Impuls sinaptik berjalan)
                        val t = (impulseProgress + (synapse.fromIdx * 0.13f)) % 1f
                        val impulseX = p1.sx + (p2.sx - p1.sx) * t
                        val impulseY = p1.sy + (p2.sy - p1.sy) * t
                        val impulseRadius = (2.2.dp.toPx() * p1.scale).coerceIn(1.2f, 4.5f)

                        drawCircle(
                            color = impulseColor.copy(alpha = (edgeAlpha * 1.3f).coerceIn(0.3f, 1f)),
                            radius = impulseRadius,
                            center = Offset(impulseX, impulseY)
                        )
                    }

                    // 4. Gambar simpul neuron (urutan dari belakang ke depan z-order)
                    val sortedNodes = projected.sortedBy { it.sz }
                    for (pn in sortedNodes) {
                        val baseColor = pn.node.region.baseColor
                        val radius = (3.6.dp.toPx() * pn.scale).coerceIn(2.0f, 7.5f)

                        // Aura pijar simpul (Glow)
                        drawCircle(
                            color = baseColor.copy(alpha = pn.alpha * 0.35f),
                            radius = radius * 2.2f,
                            center = Offset(pn.sx, pn.sy)
                        )

                        // Inti simpul neuron
                        drawCircle(
                            color = if (pn.sz > 0.4f) Color.White else baseColor.copy(alpha = pn.alpha),
                            radius = radius,
                            center = Offset(pn.sx, pn.sy)
                        )
                    }

                    // 5. Border bracket estetika taktis sudut
                    val bLen = 14.dp.toPx()
                    val pad = 10.dp.toPx()
                    val bCol = ArgusCyanPrimary.copy(alpha = 0.5f)
                    val bW = 1.5.dp.toPx()

                    drawLine(bCol, Offset(pad, pad), Offset(pad + bLen, pad), bW)
                    drawLine(bCol, Offset(pad, pad), Offset(pad, pad + bLen), bW)
                    drawLine(bCol, Offset(size.width - pad, pad), Offset(size.width - pad - bLen, pad), bW)
                    drawLine(bCol, Offset(size.width - pad, pad), Offset(size.width - pad, pad + bLen), bW)
                    drawLine(bCol, Offset(pad, size.height - pad), Offset(pad + bLen, size.height - pad), bW)
                    drawLine(bCol, Offset(pad, size.height - pad), Offset(pad, size.height - pad - bLen), bW)
                    drawLine(bCol, Offset(size.width - pad, size.height - pad), Offset(size.width - pad - bLen, size.height - pad), bW)
                    drawLine(bCol, Offset(size.width - pad, size.height - pad), Offset(size.width - pad, size.height - pad - bLen), bW)
                }

                // Overlay label sentuh & navigasi 3D
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(12.dp),
                    verticalArrangement = Arrangement.SpaceBetween
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = "VISUALIZATION • 56 NODES • 108 CONNECTIONS",
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusCyanPrimary.copy(alpha = 0.8f),
                            fontFamily = FontFamily.Monospace,
                            fontSize = 9.sp
                        )
                        Text(
                            text = "ROT: 3D DUAL-HEMISPHERE",
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusTextMuted,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 9.sp
                        )
                    }

                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.Bottom
                    ) {
                        Text(
                            text = "Touch and drag to rotate the 3D visualization",
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusTextMuted.copy(alpha = 0.7f),
                            fontSize = 9.sp
                        )
                        if (currentLoss != null) {
                            Text(
                                text = "Loss: ${String.format(java.util.Locale.US, "%.4f", currentLoss)}",
                                style = MaterialTheme.typography.labelSmall,
                                color = impulseColor,
                                fontWeight = FontWeight.Bold,
                                fontFamily = FontFamily.Monospace,
                                fontSize = 10.sp
                            )
                        }
                    }
                }
            }

            Spacer(modifier = Modifier.height(14.dp))

            // Info Hardware Nyata (Bebas dari progress bar naik turun palsu)
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .background(ArgusDarkBg.copy(alpha = 0.7f), RoundedCornerShape(10.dp))
                    .border(1.dp, ArgusCardBorder, RoundedCornerShape(10.dp))
                    .padding(12.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            text = "DEVICE RAM: ${hardwareInfo.totalRamGb} GB (${String.format(java.util.Locale.US, "%.1f", hardwareInfo.availRamGb)} GB available)",
                            style = MaterialTheme.typography.bodyMedium,
                            color = ArgusTextPrimary,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 11.sp
                        )
                        Text(
                            text = "Compute capability: ${hardwareInfo.primaryAccelerator} | App heap: ${hardwareInfo.maxHeapMb} MB",
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusCyanGlow,
                            fontSize = 10.sp
                        )
                    }

                    Text(
                        text = "${hardwareInfo.socManufacturer} ${hardwareInfo.socModel}",
                        style = MaterialTheme.typography.labelSmall,
                        color = ArgusTextSecondary,
                        fontSize = 10.sp,
                        fontFamily = FontFamily.Monospace
                    )
                }

                if (onReplayBootSequence != null) {
                    Spacer(modifier = Modifier.height(10.dp))
                    OutlinedButton(
                        onClick = onReplayBootSequence,
                        modifier = Modifier.fillMaxWidth(),
                        shape = RoundedCornerShape(8.dp),
                        colors = androidx.compose.material3.ButtonDefaults.outlinedButtonColors(
                            contentColor = ArgusCyanPrimary
                        ),
                        border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCyanPrimary.copy(alpha = 0.5f))
                    ) {
                        Icon(
                            painter = painterResource(R.drawable.ic_symbol_restart_alt),
                            contentDescription = null,
                            modifier = Modifier.size(14.dp)
                        )
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(
                            text = "REPLAY GENESIS BOOT SEQUENCE",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 0.5.sp
                        )
                    }
                }
            }
        }
    }
}

/**
 * Membentuk simpul 3D anatomi otak (Hemisfer Kiri & Kanan terpisah oleh fisura longitudinal,
 * ditambah Lobus Frontal, Oksipital, Temporal, Korpus Kalosum, dan Serebelum).
 */
private fun generateBrainTopology(): List<NeuronNode3D> {
    val nodes = mutableListOf<NeuronNode3D>()
    var nextId = 0

    // Hemisfer Kiri (x < -0.06f)
    val leftCoords = listOf(
        Triple(-0.35f, -0.65f, 0.25f) to BrainRegion.FRONTAL_LOBE,
        Triple(-0.55f, -0.45f, 0.45f) to BrainRegion.FRONTAL_LOBE,
        Triple(-0.25f, -0.55f, 0.55f) to BrainRegion.FRONTAL_LOBE,
        Triple(-0.70f, -0.20f, 0.20f) to BrainRegion.LEFT_CORTEX,
        Triple(-0.65f, 0.15f, 0.10f) to BrainRegion.LEFT_CORTEX,
        Triple(-0.75f, 0.00f, -0.20f) to BrainRegion.TEMPORAL_LOBE,
        Triple(-0.50f, 0.35f, -0.15f) to BrainRegion.TEMPORAL_LOBE,
        Triple(-0.35f, -0.20f, -0.55f) to BrainRegion.OCCIPITAL_LOBE,
        Triple(-0.55f, 0.10f, -0.50f) to BrainRegion.OCCIPITAL_LOBE,
        Triple(-0.20f, 0.20f, -0.65f) to BrainRegion.OCCIPITAL_LOBE,
        Triple(-0.40f, -0.70f, -0.10f) to BrainRegion.LEFT_CORTEX,
        Triple(-0.30f, -0.40f, 0.00f) to BrainRegion.LEFT_CORTEX,
        Triple(-0.60f, -0.30f, -0.25f) to BrainRegion.LEFT_CORTEX,
        Triple(-0.45f, -0.05f, 0.40f) to BrainRegion.FRONTAL_LOBE,
        Triple(-0.20f, 0.55f, -0.40f) to BrainRegion.CEREBELLUM,
        Triple(-0.35f, 0.65f, -0.25f) to BrainRegion.CEREBELLUM
    )

    for ((c, region) in leftCoords) {
        nodes.add(NeuronNode3D(nextId++, c.first, c.second, c.third, region))
    }

    // Hemisfer Kanan (x > +0.06f simetris dengan variasi mikro)
    val rightCoords = listOf(
        Triple(0.35f, -0.65f, 0.25f) to BrainRegion.FRONTAL_LOBE,
        Triple(0.55f, -0.45f, 0.45f) to BrainRegion.FRONTAL_LOBE,
        Triple(0.25f, -0.55f, 0.55f) to BrainRegion.FRONTAL_LOBE,
        Triple(0.70f, -0.20f, 0.20f) to BrainRegion.RIGHT_CORTEX,
        Triple(0.65f, 0.15f, 0.10f) to BrainRegion.RIGHT_CORTEX,
        Triple(0.75f, 0.00f, -0.20f) to BrainRegion.TEMPORAL_LOBE,
        Triple(0.50f, 0.35f, -0.15f) to BrainRegion.TEMPORAL_LOBE,
        Triple(0.35f, -0.20f, -0.55f) to BrainRegion.OCCIPITAL_LOBE,
        Triple(0.55f, 0.10f, -0.50f) to BrainRegion.OCCIPITAL_LOBE,
        Triple(0.20f, 0.20f, -0.65f) to BrainRegion.OCCIPITAL_LOBE,
        Triple(0.40f, -0.70f, -0.10f) to BrainRegion.RIGHT_CORTEX,
        Triple(0.30f, -0.40f, 0.00f) to BrainRegion.RIGHT_CORTEX,
        Triple(0.60f, -0.30f, -0.25f) to BrainRegion.RIGHT_CORTEX,
        Triple(0.45f, -0.05f, 0.40f) to BrainRegion.FRONTAL_LOBE,
        Triple(0.20f, 0.55f, -0.40f) to BrainRegion.CEREBELLUM,
        Triple(0.35f, 0.65f, -0.25f) to BrainRegion.CEREBELLUM
    )

    for ((c, region) in rightCoords) {
        nodes.add(NeuronNode3D(nextId++, c.first, c.second, c.third, region))
    }

    // Jembatan Korpus Kalosum & Batang Otak Sentral (x mendekati 0)
    val centralCoords = listOf(
        Triple(0.00f, -0.35f, 0.20f) to BrainRegion.CORPUS_CALLOSUM,
        Triple(0.00f, -0.25f, -0.10f) to BrainRegion.CORPUS_CALLOSUM,
        Triple(0.00f, -0.10f, -0.30f) to BrainRegion.CORPUS_CALLOSUM,
        Triple(0.00f, 0.20f, -0.15f) to BrainRegion.CORPUS_CALLOSUM,
        Triple(0.00f, 0.50f, -0.30f) to BrainRegion.CEREBELLUM
    )

    for ((c, region) in centralCoords) {
        nodes.add(NeuronNode3D(nextId++, c.first, c.second, c.third, region))
    }

    return nodes
}

/**
 * Membentuk akson sinapsis antar simpul berdasarkan kedekatan jarak spasial 3D Euclidean.
 */
private fun generateSynapseNetwork(nodes: List<NeuronNode3D>): List<SynapseEdge> {
    val edges = mutableListOf<SynapseEdge>()
    val maxDistSq = 0.52f * 0.52f // ambang batas konektivitas sinapsis

    for (i in nodes.indices) {
        val ni = nodes[i]
        for (j in (i + 1) until nodes.size) {
            val nj = nodes[j]
            val dx = ni.x - nj.x
            val dy = ni.y - nj.y
            val dz = ni.z - nj.z
            val distSq = dx * dx + dy * dy + dz * dz

            if (distSq <= maxDistSq) {
                val dist = sqrt(distSq)
                val weight = (1f - (dist / 0.52f)).coerceIn(0.2f, 1f)
                edges.add(SynapseEdge(i, j, weight))
            }
        }
    }
    return edges
}
