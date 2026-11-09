package com.argus.agi.ui.screens

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
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
import androidx.compose.material.icons.automirrored.filled.OpenInNew
import androidx.compose.material.icons.automirrored.filled.ScreenShare
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.DeveloperBoard
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.VolumeUp
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.argus.agi.service.ArgusAccessibilityService
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
import com.argus.agi.ui.theme.ArgusWarning
import java.util.Locale

@Composable
fun PerceptionScreen(viewModel: ArgusViewModel) {
    val context = LocalContext.current
    val hardware by viewModel.hardwareInfo.collectAsState()
    val contextSummary by viewModel.currentContextSummary.collectAsState()
    val cameraLuminance by viewModel.cameraLuminance.collectAsState()
    val isAccessibilityActive = ArgusAccessibilityService.instance != null
    val ambientDb by viewModel.ambientDecibel.collectAsState()
    val isAcousticActive by viewModel.isAcousticMonitoring.collectAsState()
    val isSpeaking by viewModel.isSpeaking.collectAsState()
    val isListening by viewModel.isListeningVoice.collectAsState()

    var isCameraActive by rememberSaveable { mutableStateOf(false) }
    var isCameraExpanded by remember { mutableStateOf(true) }
    var isScreenExpanded by remember { mutableStateOf(true) }
    var isHardwareExpanded by remember { mutableStateOf(true) }
    var isAudioExpanded by remember { mutableStateOf(true) }

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(ArgusDarkBg)
            .padding(horizontal = 16.dp),
        contentPadding = PaddingValues(vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Title Header
        item {
            Column {
                Text(
                    text = "PERSEPSI LINGKUNGAN & SENSOR",
                    style = MaterialTheme.typography.titleLarge,
                    color = ArgusCyanPrimary,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Sensor Visual Kamera Adaptif, Pemindaian Layar, dan Akselerasi Hardware",
                    style = MaterialTheme.typography.bodyMedium,
                    color = ArgusTextSecondary
                )
            }
        }

        // 1. Camera Vision Card (Collapsible & Adaptive)
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
                            .clickable { isCameraExpanded = !isCameraExpanded }
                            .padding(vertical = 2.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = Icons.Default.CameraAlt,
                                contentDescription = "Camera",
                                tint = if (isCameraActive) ArgusCyanPrimary else ArgusTextMuted,
                                modifier = Modifier.size(22.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "Camera Vision (Adaptif)",
                                style = MaterialTheme.typography.titleMedium,
                                color = ArgusTextPrimary,
                                fontWeight = FontWeight.Bold
                            )
                        }

                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                text = if (isCameraActive) String.format(Locale.US, "%.1f lx", cameraLuminance) else "Siaga",
                                style = MaterialTheme.typography.labelSmall,
                                color = if (isCameraActive) ArgusCyanGlow else ArgusTextMuted,
                                fontFamily = FontFamily.Monospace
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Switch(
                                checked = isCameraActive,
                                onCheckedChange = { isCameraActive = it },
                                colors = SwitchDefaults.colors(
                                    checkedThumbColor = ArgusCyanPrimary,
                                    checkedTrackColor = ArgusCyanPrimary.copy(alpha = 0.3f),
                                    uncheckedThumbColor = ArgusTextMuted,
                                    uncheckedTrackColor = ArgusSurfaceVariant
                                ),
                                modifier = Modifier.testTag("camera_active_switch")
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Icon(
                                imageVector = if (isCameraExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                                contentDescription = "Toggle",
                                tint = ArgusCyanPrimary,
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }

                    AnimatedVisibility(
                        visible = isCameraExpanded,
                        enter = expandVertically() + fadeIn(),
                        exit = shrinkVertically() + fadeOut()
                    ) {
                        Column {
                            Spacer(modifier = Modifier.height(12.dp))

                            if (isCameraActive) {
                                // Real Camera Preview Viewport with Adaptive Throttled Analyzer
                                Box(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .height(200.dp)
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(ArgusDarkBg)
                                        .border(1.dp, ArgusCyanPrimary.copy(alpha = 0.5f), RoundedCornerShape(12.dp))
                                ) {
                                    CameraViewfinder(onLuminanceUpdate = { lum ->
                                        viewModel.updateCameraFrame(lum)
                                    })
                                }

                                Spacer(modifier = Modifier.height(8.dp))
                                Text(
                                    text = "Mode Adaptif Aktif: Kamera mensampling frame setiap 2.5 detik untuk hemat baterai dan menjaga suhu perangkat.",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = ArgusCyanGlow
                                )
                            } else {
                                // Power-saving Standby Placeholder
                                Box(
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .clip(RoundedCornerShape(12.dp))
                                        .background(ArgusSurfaceVariant.copy(alpha = 0.5f))
                                        .border(1.dp, ArgusCardBorder, RoundedCornerShape(12.dp))
                                        .padding(16.dp),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                        Icon(
                                            imageVector = Icons.Default.CameraAlt,
                                            contentDescription = null,
                                            tint = ArgusTextMuted,
                                            modifier = Modifier.size(28.dp)
                                        )
                                        Spacer(modifier = Modifier.height(6.dp))
                                        Text(
                                            text = "Sensor Kamera Dalam Mode Siaga",
                                            style = MaterialTheme.typography.titleSmall,
                                            color = ArgusTextPrimary,
                                            fontWeight = FontWeight.Bold
                                        )
                                        Spacer(modifier = Modifier.height(4.dp))
                                        Text(
                                            text = "Kamera tidak menyala terus-menerus demi menghemat baterai & mencegah HP panas. Aktifkan sakelar saat ingin mendeteksi visual.",
                                            style = MaterialTheme.typography.bodyMedium.copy(fontSize = 11.sp),
                                            color = ArgusTextSecondary,
                                            textAlign = TextAlign.Center
                                        )
                                        Spacer(modifier = Modifier.height(10.dp))
                                        Button(
                                            onClick = { isCameraActive = true },
                                            colors = ButtonDefaults.buttonColors(
                                                containerColor = ArgusCyanPrimary.copy(alpha = 0.2f),
                                                contentColor = ArgusCyanPrimary
                                            ),
                                            shape = RoundedCornerShape(8.dp),
                                            modifier = Modifier.testTag("activate_camera_button")
                                        ) {
                                            Text("Aktifkan Sensor Visual", fontWeight = FontWeight.Bold, fontSize = 11.sp)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Screen Perception & Accessibility Service Card (Collapsible)
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
                            .clickable { isScreenExpanded = !isScreenExpanded }
                            .padding(vertical = 2.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.ScreenShare,
                                contentDescription = "Screen",
                                tint = ArgusCyanPrimary,
                                modifier = Modifier.size(22.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "Persepsi Layar HP",
                                style = MaterialTheme.typography.titleMedium,
                                color = ArgusTextPrimary,
                                fontWeight = FontWeight.Bold
                            )
                        }

                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(if (isAccessibilityActive) ArgusSuccess.copy(alpha = 0.15f) else ArgusWarning.copy(alpha = 0.15f))
                                    .padding(horizontal = 8.dp, vertical = 4.dp)
                            ) {
                                Text(
                                    text = if (isAccessibilityActive) "AKTIF" else "BELUM AKTIF",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = if (isAccessibilityActive) ArgusSuccess else ArgusWarning,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                            Spacer(modifier = Modifier.width(6.dp))
                            Icon(
                                imageVector = if (isScreenExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                                contentDescription = "Toggle",
                                tint = ArgusCyanPrimary,
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }

                    AnimatedVisibility(
                        visible = isScreenExpanded,
                        enter = expandVertically() + fadeIn(),
                        exit = shrinkVertically() + fadeOut()
                    ) {
                        Column {
                            Spacer(modifier = Modifier.height(10.dp))

                            Text(
                                text = "Status: $contextSummary",
                                style = MaterialTheme.typography.bodyMedium.copy(fontSize = 12.sp),
                                color = ArgusTextPrimary,
                                fontFamily = FontFamily.Monospace
                            )

                            Spacer(modifier = Modifier.height(12.dp))

                            if (!isAccessibilityActive) {
                                Button(
                                    onClick = {
                                        val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
                                        context.startActivity(intent)
                                    },
                                    colors = ButtonDefaults.buttonColors(
                                        containerColor = ArgusCyanPrimary,
                                        contentColor = ArgusDarkBg
                                    ),
                                    shape = RoundedCornerShape(10.dp),
                                    modifier = Modifier
                                        .fillMaxWidth()
                                        .testTag("enable_accessibility_button")
                                ) {
                                    Icon(
                                        imageVector = Icons.AutoMirrored.Filled.OpenInNew,
                                        contentDescription = null,
                                        modifier = Modifier.size(16.dp)
                                    )
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text(
                                        text = "Aktifkan Izin Baca Layar (Accessibility)",
                                        fontWeight = FontWeight.Bold,
                                        fontSize = 12.sp
                                    )
                                }
                            } else {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(
                                        imageVector = Icons.Default.CheckCircle,
                                        contentDescription = null,
                                        tint = ArgusSuccess,
                                        modifier = Modifier.size(16.dp)
                                    )
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text(
                                        text = "Layar dipantau secara otomatis saat berpindah aplikasi.",
                                        style = MaterialTheme.typography.labelSmall,
                                        color = ArgusSuccess
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3. Hardware & SoC Acceleration Card (Collapsible)
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
                            .clickable { isHardwareExpanded = !isHardwareExpanded }
                            .padding(vertical = 2.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = Icons.Default.DeveloperBoard,
                                contentDescription = "SoC",
                                tint = ArgusCyanPrimary,
                                modifier = Modifier.size(22.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "Hardware Telemetry & Accelerator",
                                style = MaterialTheme.typography.titleMedium,
                                color = ArgusTextPrimary,
                                fontWeight = FontWeight.Bold
                            )
                        }

                        Icon(
                            imageVector = if (isHardwareExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                            contentDescription = "Toggle",
                            tint = ArgusCyanPrimary,
                            modifier = Modifier.size(20.dp)
                        )
                    }

                    AnimatedVisibility(
                        visible = isHardwareExpanded,
                        enter = expandVertically() + fadeIn(),
                        exit = shrinkVertically() + fadeOut()
                    ) {
                        Column {
                            Spacer(modifier = Modifier.height(14.dp))

                            HardwareSpecRow(label = "Chipset / SoC", value = "${hardware.socManufacturer} (${hardware.socModel})")
                            HardwareSpecRow(label = "Akselerator Primer", value = hardware.primaryAccelerator)
                            HardwareSpecRow(label = "NPU / APU", value = hardware.npuApuDetected)
                            HardwareSpecRow(label = "Vulkan Support", value = if (hardware.vulkanSupported) "Didukung (Level ${hardware.vulkanVersionCode})" else "Tidak Didukung")
                            HardwareSpecRow(label = "CPU Cores", value = "${hardware.cpuCores} Cores SIMD")
                            HardwareSpecRow(label = "Large Heap Limit", value = "${hardware.maxHeapMb} MB")
                        }
                    }
                }
            }
        }

        // 4. Audio & Voice Perception Card
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
                            .clickable { isAudioExpanded = !isAudioExpanded }
                            .padding(vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Box(
                            modifier = Modifier
                                .size(36.dp)
                                .clip(CircleShape)
                                .background(ArgusCyanPrimary.copy(alpha = 0.15f)),
                            contentAlignment = Alignment.Center
                        ) {
                            Icon(
                                imageVector = Icons.Default.Mic,
                                contentDescription = "Audio Voice",
                                tint = if (isAcousticActive || isListening) ArgusCyanPrimary else ArgusTextMuted,
                                modifier = Modifier.size(20.dp)
                            )
                        }

                        Spacer(modifier = Modifier.width(12.dp))

                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = "Persepsi Suara & Vokal (Offline)",
                                style = MaterialTheme.typography.titleMedium,
                                color = ArgusTextPrimary,
                                fontWeight = FontWeight.Bold
                            )
                            Text(
                                text = if (isAcousticActive) String.format(Locale.US, "Kebisingan: %.1f dB", ambientDb) else "Sensor Akustik: Standby",
                                style = MaterialTheme.typography.bodySmall,
                                color = if (isAcousticActive) ArgusCyanGlow else ArgusTextMuted
                            )
                        }

                        Switch(
                            checked = isAcousticActive,
                            onCheckedChange = { viewModel.toggleAcousticMonitoring() },
                            colors = SwitchDefaults.colors(
                                checkedThumbColor = ArgusDarkBg,
                                checkedTrackColor = ArgusCyanPrimary,
                                uncheckedThumbColor = ArgusTextMuted,
                                uncheckedTrackColor = ArgusSurfaceVariant
                            )
                        )

                        Spacer(modifier = Modifier.width(6.dp))

                        Icon(
                            imageVector = if (isAudioExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                            contentDescription = "Expand",
                            tint = ArgusTextMuted
                        )
                    }

                    AnimatedVisibility(
                        visible = isAudioExpanded,
                        enter = expandVertically() + fadeIn(),
                        exit = shrinkVertically() + fadeOut()
                    ) {
                        Column {
                            Spacer(modifier = Modifier.height(14.dp))

                            HardwareSpecRow(label = "Sintetis Vokal (TTS)", value = if (isSpeaking) "Sedang Bicara (Aktif)" else "Siap (100% Offline)")
                            HardwareSpecRow(label = "Pengenal Suara (STT)", value = if (isListening) "Mendengarkan..." else "Siap (100% Offline)")
                            HardwareSpecRow(label = "Tingkat Kebisingan Lingkungan", value = if (isAcousticActive) String.format(Locale.US, "%.1f dB SPL", ambientDb) else "Sensor Nonaktif")
                            HardwareSpecRow(label = "Koneksi Audio", value = "Lokal Tanpa Cloud")
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun HardwareSpecRow(label: String, value: String) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 5.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = ArgusTextMuted
        )
        Text(
            text = value,
            style = MaterialTheme.typography.labelSmall,
            color = ArgusTextPrimary,
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
fun CameraViewfinder(onLuminanceUpdate: (Float) -> Unit) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current

    var hasCameraPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }

    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasCameraPermission = granted
    }

    if (!hasCameraPermission) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(
                    imageVector = Icons.Default.CameraAlt,
                    contentDescription = null,
                    tint = ArgusCyanPrimary,
                    modifier = Modifier.size(32.dp)
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Sensor visual membutuhkan izin kamera fisik",
                    style = MaterialTheme.typography.bodyMedium.copy(fontSize = 12.sp),
                    color = ArgusTextSecondary
                )
                Spacer(modifier = Modifier.height(10.dp))
                Button(
                    onClick = { launcher.launch(Manifest.permission.CAMERA) },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = ArgusCyanPrimary,
                        contentColor = ArgusDarkBg
                    ),
                    shape = RoundedCornerShape(10.dp)
                ) {
                    Text("Beri Izin Kamera", fontWeight = FontWeight.Bold, fontSize = 11.sp)
                }
            }
        }
        return
    }

    var cameraProviderRef by remember { mutableStateOf<ProcessCameraProvider?>(null) }

    DisposableEffect(lifecycleOwner) {
        onDispose {
            try {
                cameraProviderRef?.unbindAll()
            } catch (_: Exception) {}
        }
    }

    var lastAnalysisNs by remember { mutableStateOf(0L) }
    val throttleIntervalNs = 2_500_000_000L // 2.5 seconds adaptive sampling rate

    AndroidView(
        factory = { ctx ->
            val previewView = PreviewView(ctx)
            val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)

            cameraProviderFuture.addListener({
                try {
                    val cameraProvider = cameraProviderFuture.get()
                    cameraProviderRef = cameraProvider

                    val preview = Preview.Builder().build().also {
                        it.surfaceProvider = previewView.surfaceProvider
                    }

                    val imageAnalysis = ImageAnalysis.Builder()
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build()

                    imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(ctx)) { imageProxy ->
                        val now = System.nanoTime()
                        // Adaptive sampling: only analyze 1 frame every 2.5 seconds to protect battery & thermals
                        if (now - lastAnalysisNs < throttleIntervalNs) {
                            imageProxy.close()
                            return@setAnalyzer
                        }
                        lastAnalysisNs = now

                        try {
                            val buffer = imageProxy.planes[0].buffer
                            val data = ByteArray(buffer.remaining())
                            buffer.get(data)

                            // Compute average luminance across sampled pixels
                            var sum = 0L
                            val step = (data.size / 500).coerceAtLeast(1)
                            var count = 0
                            for (i in 0 until data.size step step) {
                                sum += (data[i].toInt() and 0xFF)
                                count++
                            }
                            val avgLum = if (count > 0) sum.toFloat() / count else 0f
                            onLuminanceUpdate(avgLum)
                        } finally {
                            imageProxy.close()
                        }
                    }

                    val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector, preview, imageAnalysis)
                } catch (_: Exception) {}
            }, ContextCompat.getMainExecutor(ctx))

            previewView
        },
        modifier = Modifier.fillMaxSize()
    )
}
