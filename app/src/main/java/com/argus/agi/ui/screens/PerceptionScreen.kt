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
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
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
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.sqrt

@Composable
fun PerceptionScreen(viewModel: ArgusViewModel) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val hardware by viewModel.hardwareInfo.collectAsState()
    val contextSummary by viewModel.currentContextSummary.collectAsState()
    val cameraLuminance by viewModel.cameraLuminance.collectAsState()
    var isAccessibilityActive by remember {
        mutableStateOf(ArgusAccessibilityService.instance != null)
    }
    DisposableEffect(lifecycleOwner) {
        val observer = LifecycleEventObserver { _, event ->
            if (event == Lifecycle.Event.ON_RESUME) {
                isAccessibilityActive = ArgusAccessibilityService.instance != null
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    val ambientDb by viewModel.ambientDecibel.collectAsState()
    val isAcousticActive by viewModel.isAcousticMonitoring.collectAsState()
    val isSpeaking by viewModel.isSpeaking.collectAsState()
    val isListening by viewModel.isListeningVoice.collectAsState()

    var isCameraActive by remember { mutableStateOf(false) }
    var isCameraStreaming by remember { mutableStateOf(false) }
    var isCameraExpanded by remember { mutableStateOf(true) }
    var isScreenExpanded by remember { mutableStateOf(false) }
    var isHardwareExpanded by remember { mutableStateOf(false) }
    var isAudioExpanded by remember { mutableStateOf(false) }

    val isSensing = isCameraStreaming || isAcousticActive || isAccessibilityActive

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(ArgusDarkBg)
            .padding(horizontal = 12.dp),
        contentPadding = PaddingValues(vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        // HEADER COMPACT + LIVE STATUS
        item {
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(
                        text = "PERCEPTION",
                        style = MaterialTheme.typography.titleLarge.copy(fontSize = 20.sp),
                        color = ArgusTextPrimary,
                        fontWeight = FontWeight.Bold,
                        letterSpacing = 1.sp,
                        modifier = Modifier.weight(1f)
                    )
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.clip(RoundedCornerShape(20.dp)).background(if (isSensing) ArgusSuccess.copy(alpha = 0.15f) else ArgusSurfaceVariant).padding(horizontal = 9.dp, vertical = 4.dp)
                    ) {
                        Box(modifier = Modifier.size(7.dp).clip(CircleShape).background(if (isSensing) ArgusSuccess else ArgusTextMuted))
                        Spacer(modifier = Modifier.width(5.dp))
                        Text(text = if (isSensing) "SENSING" else "STANDBY", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp), color = if (isSensing) ArgusSuccess else ArgusTextMuted, fontWeight = FontWeight.Bold)
                    }
                }
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = "Camera, audio, and screen context. Sensor processing runs on-device with bounded sampling.",
                    style = MaterialTheme.typography.bodySmall.copy(fontSize = 12.sp),
                    color = ArgusTextMuted,
                    lineHeight = 17.sp
                )
                Spacer(modifier = Modifier.height(10.dp))
                // MINI STATUS ROW
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    MiniStatus(
                        label = "LUMA",
                        value = if (isCameraStreaming) String.format(Locale.US, "%.0f/255", cameraLuminance.coerceIn(0f, 255f)) else "Off",
                        active = isCameraStreaming,
                        modifier = Modifier.weight(1f)
                    )
                    MiniStatus(
                        label = "AUDIO",
                        value = if (isAcousticActive) String.format(Locale.US, "%.1f dB rel", ambientDb) else "Off",
                        active = isAcousticActive,
                        modifier = Modifier.weight(1f)
                    )
                    MiniStatus(
                        label = "SCREEN",
                        value = if (isAccessibilityActive) "Active" else "Off",
                        active = isAccessibilityActive,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }

        // 1. CAMERA HERO
        item {
            Surface(
                shape = RoundedCornerShape(18.dp),
                color = ArgusSurface,
                border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder),
                modifier = Modifier.fillMaxWidth()
            ) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).clickable { isCameraExpanded = !isCameraExpanded }.padding(vertical = 2.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(modifier = Modifier.size(34.dp).clip(CircleShape).background(ArgusCyanPrimary.copy(alpha = 0.15f)), contentAlignment = Alignment.Center) {
                                Icon(painter = painterResource(R.drawable.ic_symbol_photo_camera), contentDescription = "Camera", tint = if (isCameraActive) ArgusCyanPrimary else ArgusTextMuted, modifier = Modifier.size(18.dp))
                            }
                            Spacer(modifier = Modifier.width(9.dp))
                            Column {
                                Text(text = "Camera Vision", style = MaterialTheme.typography.titleMedium.copy(fontSize = 15.sp), color = ArgusTextPrimary, fontWeight = FontWeight.Bold)
                                Text(text = when {
                                    isCameraStreaming -> "Streaming • 1 sample / 2.5s"
                                    isCameraActive -> "Waiting for camera permission or provider"
                                    else -> "Off • battery friendly"
                                }, style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp), color = ArgusTextMuted)
                            }
                        }
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Switch(
                                checked = isCameraActive,
                                onCheckedChange = { enabled ->
                                    isCameraActive = enabled
                                    if (!enabled) isCameraStreaming = false
                                },
                                colors = SwitchDefaults.colors(checkedThumbColor = ArgusDarkBg, checkedTrackColor = ArgusCyanPrimary, uncheckedThumbColor = ArgusTextMuted, uncheckedTrackColor = ArgusSurfaceVariant),
                                modifier = Modifier.testTag("camera_active_switch")
                            )
                            Icon(painter = if (isCameraExpanded) painterResource(R.drawable.ic_symbol_keyboard_arrow_up) else painterResource(R.drawable.ic_symbol_keyboard_arrow_down), contentDescription = "Toggle", tint = ArgusCyanPrimary, modifier = Modifier.size(20.dp))
                        }
                    }

                    AnimatedVisibility(visible = isCameraExpanded, enter = expandVertically() + fadeIn(), exit = shrinkVertically() + fadeOut()) {
                        Column {
                            Spacer(modifier = Modifier.height(10.dp))
                            if (isCameraActive) {
                                Box(
                                    modifier = Modifier.fillMaxWidth().height(220.dp).clip(RoundedCornerShape(14.dp)).background(ArgusDarkBg).border(1.dp, ArgusCyanPrimary.copy(alpha = 0.4f), RoundedCornerShape(14.dp))
                                ) {
                                    CameraViewfinder(
                                        onLuminanceUpdate = { luma -> viewModel.updateCameraFrame(luma) },
                                        onStreamingChanged = { isCameraStreaming = it },
                                    )
                                    // Scan frame overlay, lite, no particle
                                    Canvas(modifier = Modifier.fillMaxSize()) {
                                        val stroke = 2.dp.toPx()
                                        val corner = 22.dp.toPx()
                                        val c = ArgusCyanPrimary.copy(alpha = 0.9f)
                                        // top left
                                        drawLine(c, Offset(12f, 12f + corner), Offset(12f, 12f), strokeWidth = stroke)
                                        drawLine(c, Offset(12f, 12f), Offset(12f + corner, 12f), strokeWidth = stroke)
                                        // top right
                                        drawLine(c, Offset(size.width - 12f - corner, 12f), Offset(size.width - 12f, 12f), strokeWidth = stroke)
                                        drawLine(c, Offset(size.width - 12f, 12f), Offset(size.width - 12f, 12f + corner), strokeWidth = stroke)
                                        // bottom left
                                        drawLine(c, Offset(12f, size.height - 12f - corner), Offset(12f, size.height - 12f), strokeWidth = stroke)
                                        drawLine(c, Offset(12f, size.height - 12f), Offset(12f + corner, size.height - 12f), strokeWidth = stroke)
                                        // bottom right
                                        drawLine(c, Offset(size.width - 12f - corner, size.height - 12f), Offset(size.width - 12f, size.height - 12f), strokeWidth = stroke)
                                        drawLine(c, Offset(size.width - 12f, size.height - 12f), Offset(size.width - 12f, size.height - 12f - corner), strokeWidth = stroke)
                                        // scan line
                                        val y = size.height * 0.5f
                                        drawLine(Color(0xFF38BDF8).copy(alpha = 0.35f), Offset(20f, y), Offset(size.width - 20f, y), strokeWidth = 1.5f, pathEffect = PathEffect.dashPathEffect(floatArrayOf(10f, 8f), 0f))
                                    }
                                    Box(modifier = Modifier.align(Alignment.TopStart).padding(10.dp).clip(RoundedCornerShape(20.dp)).background(Color.Black.copy(alpha = 0.6f)).padding(horizontal = 8.dp, vertical = 4.dp)) {
                                        Text(text = String.format(Locale.US, "LUMA %.0f/255 • LIVE", cameraLuminance.coerceIn(0f, 255f)), style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp), color = Color.White, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold)
                                    }
                                }
                                Spacer(modifier = Modifier.height(8.dp))
                                Text(text = "Frame analysis is throttled to one sample every 2.5 seconds to limit CPU and battery use.", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp), color = ArgusCyanGlow)
                            } else {
                                Box(
                                    modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(ArgusSurfaceVariant.copy(alpha = 0.5f)).border(1.dp, ArgusCardBorder, RoundedCornerShape(14.dp)).padding(16.dp),
                                    contentAlignment = Alignment.Center
                                ) {
                                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                        Icon(painter = painterResource(R.drawable.ic_symbol_photo_camera), contentDescription = null, tint = ArgusTextMuted, modifier = Modifier.size(28.dp))
                                        Spacer(modifier = Modifier.height(6.dp))
                                        Text(text = "Camera standby", style = MaterialTheme.typography.titleSmall, color = ArgusTextPrimary, fontWeight = FontWeight.Bold)
                                        Spacer(modifier = Modifier.height(4.dp))
                                        Text(text = "The camera remains off until you explicitly enable visual input.", style = MaterialTheme.typography.bodyMedium.copy(fontSize = 11.sp), color = ArgusTextSecondary, textAlign = TextAlign.Center)
                                        Spacer(modifier = Modifier.height(10.dp))
                                        Button(
                                            onClick = { isCameraActive = true },
                                            colors = ButtonDefaults.buttonColors(containerColor = ArgusCyanPrimary, contentColor = ArgusDarkBg),
                                            shape = RoundedCornerShape(20.dp),
                                            modifier = Modifier.testTag("activate_camera_button")
                                        ) {
                                            Text("Enable Camera", fontWeight = FontWeight.Bold, fontSize = 11.sp)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. SCREEN PERCEPTION - Collapsed by default
        item {
            Surface(shape = RoundedCornerShape(18.dp), color = ArgusSurface, border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder), modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).clickable { isScreenExpanded = !isScreenExpanded }.padding(vertical = 2.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.weight(1f)) {
                            Box(modifier = Modifier.size(34.dp).clip(CircleShape).background(ArgusCyanPrimary.copy(alpha = 0.15f)), contentAlignment = Alignment.Center) {
                                Icon(painter = painterResource(R.drawable.ic_symbol_screen_share), contentDescription = "Screen", tint = ArgusCyanPrimary, modifier = Modifier.size(18.dp))
                            }
                            Spacer(modifier = Modifier.width(9.dp))
                            Column {
                                Text(text = "Screen Text", style = MaterialTheme.typography.titleMedium.copy(fontSize = 15.sp), color = ArgusTextPrimary, fontWeight = FontWeight.Bold)
                                Text(text = contextSummary.take(42) + if (contextSummary.length > 42) "..." else "", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp), color = ArgusTextMuted)
                            }
                        }
                        Spacer(modifier = Modifier.width(8.dp))
                        Box(modifier = Modifier.clip(RoundedCornerShape(20.dp)).background(if (isAccessibilityActive) ArgusSuccess.copy(alpha = 0.15f) else ArgusWarning.copy(alpha = 0.15f)).padding(horizontal = 8.dp, vertical = 4.dp)) {
                            Text(text = if (isAccessibilityActive) "ACTIVE" else "OFF", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp), color = if (isAccessibilityActive) ArgusSuccess else ArgusWarning, fontWeight = FontWeight.Bold)
                        }
                        Icon(painter = if (isScreenExpanded) painterResource(R.drawable.ic_symbol_keyboard_arrow_up) else painterResource(R.drawable.ic_symbol_keyboard_arrow_down), contentDescription = "Toggle", tint = ArgusCyanPrimary, modifier = Modifier.size(20.dp))
                    }

                    AnimatedVisibility(visible = isScreenExpanded, enter = expandVertically() + fadeIn(), exit = shrinkVertically() + fadeOut()) {
                        Column {
                            Spacer(modifier = Modifier.height(10.dp))
                            Text(text = "Status: $contextSummary", style = MaterialTheme.typography.bodyMedium.copy(fontSize = 12.sp), color = ArgusTextPrimary, fontFamily = FontFamily.Monospace)
                            Spacer(modifier = Modifier.height(12.dp))
                            if (!isAccessibilityActive) {
                                Button(
                                    onClick = { val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS); context.startActivity(intent) },
                                    colors = ButtonDefaults.buttonColors(containerColor = ArgusCyanPrimary, contentColor = ArgusDarkBg),
                                    shape = RoundedCornerShape(20.dp),
                                    modifier = Modifier.fillMaxWidth().testTag("enable_accessibility_button")
                                ) {
                                    Icon(painter = painterResource(R.drawable.ic_symbol_open_in_new), contentDescription = null, modifier = Modifier.size(16.dp))
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text(text = "Enable Accessibility Access", fontWeight = FontWeight.Bold, fontSize = 12.sp)
                                }
                            } else {
                                Row(verticalAlignment = Alignment.CenterVertically) {
                                    Icon(painter = painterResource(R.drawable.ic_symbol_check_circle), contentDescription = null, tint = ArgusSuccess, modifier = Modifier.size(16.dp))
                                    Spacer(modifier = Modifier.width(6.dp))
                                    Text(text = "Screen text is observed when the active window changes; excluded apps are skipped.", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp), color = ArgusSuccess)
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3. HARDWARE - Collapsed by default, fixed row layout
        item {
            Surface(shape = RoundedCornerShape(18.dp), color = ArgusSurface, border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder), modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).clickable { isHardwareExpanded = !isHardwareExpanded }.padding(vertical = 2.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(modifier = Modifier.size(34.dp).clip(CircleShape).background(ArgusCyanPrimary.copy(alpha = 0.15f)), contentAlignment = Alignment.Center) {
                                Icon(painter = painterResource(R.drawable.ic_symbol_developer_board), contentDescription = "SoC", tint = ArgusCyanPrimary, modifier = Modifier.size(18.dp))
                            }
                            Spacer(modifier = Modifier.width(9.dp))
                            Column {
                                Text(text = "Hardware & Compute", style = MaterialTheme.typography.titleMedium.copy(fontSize = 15.sp), color = ArgusTextPrimary, fontWeight = FontWeight.Bold)
                                Text(text = "${hardware.socManufacturer} • ${hardware.cpuCores} cores • ${hardware.maxHeapMb} MB heap", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp), color = ArgusTextMuted)
                            }
                        }
                        Icon(painter = if (isHardwareExpanded) painterResource(R.drawable.ic_symbol_keyboard_arrow_up) else painterResource(R.drawable.ic_symbol_keyboard_arrow_down), contentDescription = "Toggle", tint = ArgusCyanPrimary, modifier = Modifier.size(20.dp))
                    }

                    AnimatedVisibility(visible = isHardwareExpanded, enter = expandVertically() + fadeIn(), exit = shrinkVertically() + fadeOut()) {
                        Column {
                            Spacer(modifier = Modifier.height(12.dp))
                            HardwareSpecRow(label = "Chipset / SoC", value = "${hardware.socManufacturer} (${hardware.socModel})")
                            HardwareSpecRow(label = "Vulkan GPU", value = hardware.primaryAccelerator)
                            HardwareSpecRow(label = "NPU / APU", value = if (hardware.npuApuDetected == "Not detected") "Not detected" else "${hardware.npuApuDetected} • no vendor runtime")
                            HardwareSpecRow(label = "Vulkan Support", value = if (hardware.vulkanSupported) "Yes • v${hardware.vulkanVersionCode}" else "Not advertised")
                            HardwareSpecRow(label = "CPU Cores", value = "${hardware.cpuCores} logical cores")
                            HardwareSpecRow(label = "App Heap Limit", value = "${hardware.maxHeapMb} MB")
                        }
                    }
                }
            }
        }

        // 4. AUDIO
        item {
            Surface(shape = RoundedCornerShape(18.dp), color = ArgusSurface, border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder), modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).clickable { isAudioExpanded = !isAudioExpanded }.padding(vertical = 2.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Box(modifier = Modifier.size(34.dp).clip(CircleShape).background(ArgusCyanPrimary.copy(alpha = 0.15f)), contentAlignment = Alignment.Center) {
                            Icon(painter = painterResource(R.drawable.ic_symbol_mic), contentDescription = "Audio", tint = if (isAcousticActive || isListening) ArgusCyanPrimary else ArgusTextMuted, modifier = Modifier.size(18.dp))
                        }
                        Spacer(modifier = Modifier.width(9.dp))
                        Column(modifier = Modifier.weight(1f)) {
                            Text(text = "Voice & Acoustic", style = MaterialTheme.typography.titleMedium.copy(fontSize = 15.sp), color = ArgusTextPrimary, fontWeight = FontWeight.Bold)
                            Text(text = if (isAcousticActive) String.format(Locale.US, "Ambient: %.1f dB", ambientDb) else "Standby", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp), color = if (isAcousticActive) ArgusCyanGlow else ArgusTextMuted)
                        }
                        Switch(
                            checked = isAcousticActive,
                            onCheckedChange = { viewModel.toggleAcousticMonitoring() },
                            colors = SwitchDefaults.colors(checkedThumbColor = ArgusDarkBg, checkedTrackColor = ArgusCyanPrimary, uncheckedThumbColor = ArgusTextMuted, uncheckedTrackColor = ArgusSurfaceVariant)
                        )
                        Icon(painter = if (isAudioExpanded) painterResource(R.drawable.ic_symbol_keyboard_arrow_up) else painterResource(R.drawable.ic_symbol_keyboard_arrow_down), contentDescription = "Expand", tint = ArgusTextMuted, modifier = Modifier.size(20.dp))
                    }

                    AnimatedVisibility(visible = isAudioExpanded, enter = expandVertically() + fadeIn(), exit = shrinkVertically() + fadeOut()) {
                        Column {
                            Spacer(modifier = Modifier.height(12.dp))
                            HardwareSpecRow(label = "Text-to-Speech", value = if (isSpeaking) "Speaking" else "Ready • device TTS")
                            HardwareSpecRow(label = "Speech Recognition", value = if (isListening) "Listening..." else "Ready • offline preferred")
                            HardwareSpecRow(label = "Relative Audio Level", value = if (isAcousticActive) String.format(Locale.US, "%.1f dB relative (uncalibrated)", ambientDb) else "Sensor Off")
                            HardwareSpecRow(label = "Speech Service", value = "Device speech service; network fallback may be used when offline recognition is unavailable")
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun MiniStatus(label: String, value: String, active: Boolean, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier.clip(RoundedCornerShape(12.dp)).background(if (active) ArgusCyanPrimary.copy(alpha = 0.12f) else ArgusSurfaceVariant).border(1.dp, if (active) ArgusCyanPrimary.copy(alpha = 0.4f) else ArgusCardBorder, RoundedCornerShape(12.dp)).padding(horizontal = 10.dp, vertical = 8.dp)
    ) {
        Text(text = label, style = MaterialTheme.typography.labelSmall.copy(fontSize = 9.sp), color = ArgusTextMuted, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(2.dp))
        Text(text = value, style = MaterialTheme.typography.labelSmall.copy(fontSize = 12.sp), color = if (active) ArgusCyanPrimary else ArgusTextPrimary, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace)
    }
}

@Composable
fun HardwareSpecRow(label: String, value: String) {
    Column(modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp)) {
        Text(text = label, style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp), color = ArgusTextMuted)
        Spacer(modifier = Modifier.height(2.dp))
        Text(text = value, style = MaterialTheme.typography.bodySmall.copy(fontSize = 13.sp), color = ArgusTextPrimary, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Medium)
    }
}

@Composable
fun CameraViewfinder(
    onLuminanceUpdate: (Float) -> Unit,
    onStreamingChanged: (Boolean) -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val currentOnLuminanceUpdate by rememberUpdatedState(onLuminanceUpdate)
    val currentOnStreamingChanged by rememberUpdatedState(onStreamingChanged)
    val mainExecutor = remember(context) { ContextCompat.getMainExecutor(context) }

    var hasCameraPermission by remember(context) {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.CAMERA,
            ) == PackageManager.PERMISSION_GRANTED,
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission(),
    ) { granted ->
        hasCameraPermission = granted
    }

    if (!hasCameraPermission) {
        Box(
            modifier = Modifier.fillMaxSize().padding(16.dp),
            contentAlignment = Alignment.Center,
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(
                    painter = painterResource(R.drawable.ic_symbol_photo_camera),
                    contentDescription = null,
                    tint = ArgusCyanPrimary,
                    modifier = Modifier.size(32.dp),
                )
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "Camera permission is required for visual input.",
                    style = MaterialTheme.typography.bodyMedium.copy(fontSize = 12.sp),
                    color = ArgusTextSecondary,
                    textAlign = TextAlign.Center,
                )
                Spacer(modifier = Modifier.height(10.dp))
                Button(
                    onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) },
                    colors = ButtonDefaults.buttonColors(
                        containerColor = ArgusCyanPrimary,
                        contentColor = ArgusDarkBg,
                    ),
                    shape = RoundedCornerShape(20.dp),
                ) {
                    Text("Grant camera permission", fontWeight = FontWeight.Bold, fontSize = 11.sp)
                }
            }
        }
        return
    }

    var cameraError by remember(lifecycleOwner) { mutableStateOf<String?>(null) }
    val analysisExecutor = remember(lifecycleOwner) { Executors.newSingleThreadExecutor() }
    val cameraProviderRef = remember(lifecycleOwner) {
        AtomicReference<ProcessCameraProvider?>(null)
    }
    val isDisposed = remember(lifecycleOwner) { AtomicBoolean(false) }

    DisposableEffect(lifecycleOwner, analysisExecutor) {
        onDispose {
            isDisposed.set(true)
            currentOnStreamingChanged(false)
            cameraProviderRef.getAndSet(null)?.unbindAll()
            analysisExecutor.shutdownNow()
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = { viewContext ->
                val previewView = PreviewView(viewContext).apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                }
                val cameraProviderFuture = ProcessCameraProvider.getInstance(viewContext)
                cameraProviderFuture.addListener({
                    if (isDisposed.get()) {
                        return@addListener
                    }

                    try {
                        val cameraProvider = cameraProviderFuture.get()
                        cameraProviderRef.set(cameraProvider)

                        val preview = Preview.Builder().build().also {
                            it.surfaceProvider = previewView.surfaceProvider
                        }
                        val imageAnalysis = ImageAnalysis.Builder()
                            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                            .build()

                        var lastAnalysisNs = 0L
                        imageAnalysis.setAnalyzer(analysisExecutor) { imageProxy ->
                            try {
                                val now = System.nanoTime()
                                if (now - lastAnalysisNs >= 2_500_000_000L) {
                                    lastAnalysisNs = now
                                    val yPlane = imageProxy.planes.firstOrNull()
                                    if (yPlane != null) {
                                        val buffer = yPlane.buffer.duplicate()
                                        val basePosition = buffer.position()
                                        val width = imageProxy.width
                                        val height = imageProxy.height
                                        val sampleStep = sqrt(
                                            width.toDouble() * height.toDouble() / 500.0,
                                        ).toInt().coerceAtLeast(1)
                                        var luminanceSum = 0L
                                        var sampleCount = 0

                                        for (row in 0 until height step sampleStep) {
                                            val rowOffset = basePosition + row * yPlane.rowStride
                                            for (column in 0 until width step sampleStep) {
                                                val index = rowOffset + column * yPlane.pixelStride
                                                if (index < buffer.limit()) {
                                                    luminanceSum += (buffer.get(index).toInt() and 0xFF).toLong()
                                                    sampleCount += 1
                                                }
                                            }
                                        }

                                        if (sampleCount > 0) {
                                            val averageLuma = luminanceSum.toFloat() / sampleCount
                                            mainExecutor.execute {
                                                if (!isDisposed.get()) {
                                                    cameraError = null
                                                    currentOnLuminanceUpdate(averageLuma)
                                                }
                                            }
                                        }
                                    }
                                }
                            } catch (error: Exception) {
                                mainExecutor.execute {
                                    if (!isDisposed.get() && cameraError == null) {
                                        cameraError = "Frame analysis failed: ${error.localizedMessage ?: "unknown error"}"
                                    }
                                }
                            } finally {
                                imageProxy.close()
                            }
                        }

                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_BACK_CAMERA,
                            preview,
                            imageAnalysis,
                        )
                        cameraError = null
                        currentOnStreamingChanged(true)
                    } catch (error: Exception) {
                        currentOnStreamingChanged(false)
                        cameraError = "Camera could not start: ${error.localizedMessage ?: "check camera availability and permissions"}"
                    }
                }, mainExecutor)
                previewView
            },
            modifier = Modifier.fillMaxSize(),
        )

        if (cameraError != null) {
            Box(
                modifier = Modifier.fillMaxSize()
                    .background(ArgusDarkBg.copy(alpha = 0.88f))
                    .padding(20.dp),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text = cameraError.orEmpty(),
                    style = MaterialTheme.typography.bodySmall,
                    color = ArgusWarning,
                    textAlign = TextAlign.Center,
                )
            }
        }
    }
}
