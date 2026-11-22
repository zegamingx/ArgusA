package com.argus.agi.ui.screens

import android.content.ClipData
import android.net.Uri
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.expandVertically
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.shrinkVertically
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.asPaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.ime
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.ClipEntry
import androidx.compose.ui.platform.LocalClipboard
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import coil3.compose.rememberAsyncImagePainter
import com.argus.agi.R
import com.argus.agi.ui.ArgusViewModel
import com.argus.agi.ui.ChatMessage
import com.argus.agi.model.StreamedImportStage
import com.argus.agi.ui.theme.ArgusBlueAccent
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
import java.text.NumberFormat
import java.util.Locale
import kotlinx.coroutines.launch

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(viewModel: ArgusViewModel) {
    val context = LocalContext.current
    val clipboard = LocalClipboard.current
    val clipboardScope = rememberCoroutineScope()

    val messages by viewModel.messages.collectAsStateWithLifecycle()
    val isGenerating by viewModel.isGenerating.collectAsStateWithLifecycle()
    val activeDoc by viewModel.activeDocument.collectAsStateWithLifecycle()
    val isGrpoTraining by viewModel.isGrpoTraining.collectAsStateWithLifecycle()
    val grpoProgress by viewModel.grpoProgress.collectAsStateWithLifecycle()
    val streamedImportProgress by viewModel.streamedImportProgress.collectAsStateWithLifecycle()
    val isListening by viewModel.isListeningVoice.collectAsStateWithLifecycle()
    val isSpeaking by viewModel.isSpeaking.collectAsStateWithLifecycle()
    val isAutoSpeak by viewModel.isAutoSpeakEnabled.collectAsStateWithLifecycle()
    val partialSpeech by viewModel.partialTranscription.collectAsStateWithLifecycle()
    val totalParams = viewModel.getTotalParameters()
    var inputText by remember { mutableStateOf("") }
    val listState = rememberLazyListState()

    var isHeaderExpanded by remember { mutableStateOf(false) }
    var showAttachDialog by remember { mutableStateOf(false) }

    val imeBottom = WindowInsets.ime.asPaddingValues().calculateBottomPadding()
    val isKeyboardOpen = imeBottom > 0.dp

    LaunchedEffect(messages.size, isKeyboardOpen, isGenerating) {
        val lastVisibleItemIndex = messages.lastIndex + if (isGenerating) 1 else 0
        if (lastVisibleItemIndex >= 0) {
            listState.animateScrollToItem(lastVisibleItemIndex)
        }
    }

    val filePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { viewModel.importFileContent(it) }
    }

    val grpoFilePickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { viewModel.importGrpoFileContent(it) }
    }

    val mediaPickerLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.PickVisualMedia()
    ) { uri: Uri? ->
        uri?.let {
            val mime = context.contentResolver.getType(it)
            if (mime?.startsWith("video/") == true) {
                viewModel.importVideoContent(it)
            } else {
                viewModel.importImageContent(it)
            }
        }
    }

    val audioPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            viewModel.startVoiceInput(
                onResult = { recognized ->
                    if (recognized.isNotBlank()) {
                        viewModel.sendMessage(recognized)
                    }
                },
                onError = { err ->
                    Toast.makeText(context, err, Toast.LENGTH_SHORT).show()
                }
            )
        } else {
            Toast.makeText(context, "Microphone permission is required for voice input.", Toast.LENGTH_SHORT).show()
        }
    }

    if (showAttachDialog) {
        AlertDialog(
            onDismissRequest = { showAttachDialog = false },
            containerColor = ArgusSurface,
            title = { Text("Attach content", color = ArgusTextPrimary, fontWeight = FontWeight.Bold) },
            text = {
                Column {
                    Text("Choose the type of content to send to ARGUS.", color = ArgusTextSecondary, fontSize = 13.sp)
                    Spacer(modifier = Modifier.height(12.dp))
                    Text(
                        text = "Document / ZIP",
                        color = ArgusCyanPrimary,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(ArgusSurfaceVariant).clickable {
                            showAttachDialog = false
                            filePickerLauncher.launch("*/*")
                        }.padding(12.dp)
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "GRPO JSONL — policy optimization",
                        color = ArgusCyanPrimary,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(ArgusSurfaceVariant).clickable {
                            showAttachDialog = false
                            grpoFilePickerLauncher.launch("*/*")
                        }.padding(12.dp)
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "Uses prompt/answer pairs to sample response groups and optimize the local policy. Requires an idle SFT queue.",
                        color = ArgusTextMuted,
                        fontSize = 11.sp,
                        lineHeight = 15.sp
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Image / Video",
                        color = ArgusCyanPrimary,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(ArgusSurfaceVariant).clickable {
                            showAttachDialog = false
                            mediaPickerLauncher.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageAndVideo))
                        }.padding(12.dp)
                    )
                }
            },
            confirmButton = {},
            dismissButton = {
                TextButton(onClick = { showAttachDialog = false }) {
                    Text("Cancel", color = ArgusTextMuted)
                }
            }
        )
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .imePadding()
    ) {
        // HEADER - Compact
        Surface(
            modifier = Modifier.fillMaxWidth(),
            color = ArgusSurface,
            tonalElevation = 2.dp
        ) {
            Column {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable { isHeaderExpanded = !isHeaderExpanded }
                        .padding(horizontal = 14.dp, vertical = 10.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Image(
                        painter = painterResource(id = R.drawable.ic_argus_logo),
                        contentDescription = "ARGUS Core",
                        modifier = Modifier
                            .size(36.dp)
                            .clip(CircleShape)
                            .border(1.dp, ArgusCyanPrimary, CircleShape)
                    )
                    Spacer(modifier = Modifier.width(10.dp))
                    Column(modifier = Modifier.weight(1f)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(
                                text = "A.R.G.U.S",
                                style = MaterialTheme.typography.titleMedium.copy(fontSize = 15.sp),
                                color = ArgusTextPrimary,
                                fontWeight = FontWeight.Bold,
                                letterSpacing = 1.sp
                            )
                            Spacer(modifier = Modifier.width(6.dp))
                            Box(
                                modifier = Modifier.clip(RoundedCornerShape(10.dp)).background(ArgusCyanPrimary.copy(alpha = 0.15f)).padding(horizontal = 7.dp, vertical = 2.dp)
                            ) {
                                Text(text = "ON-DEVICE", style = MaterialTheme.typography.labelSmall.copy(fontSize = 9.sp), color = ArgusCyanPrimary, fontWeight = FontWeight.Bold)
                            }
                        }
                        Text(
                            text = "$totalParams parameters • Local model",
                            style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp),
                            color = ArgusTextMuted
                        )
                    }
                    Icon(
                        painter = if (isHeaderExpanded) painterResource(R.drawable.ic_symbol_keyboard_arrow_up) else painterResource(R.drawable.ic_symbol_keyboard_arrow_down),
                        contentDescription = null,
                        tint = ArgusCyanPrimary,
                        modifier = Modifier.size(20.dp)
                    )
                }

                AnimatedVisibility(
                    visible = isHeaderExpanded,
                    enter = expandVertically() + fadeIn(),
                    exit = shrinkVertically() + fadeOut()
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth().background(ArgusDarkBg.copy(alpha = 0.5f)).padding(horizontal = 14.dp, vertical = 8.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(
                            modifier = Modifier.clip(RoundedCornerShape(20.dp)).clickable { viewModel.toggleAutoSpeak() }.background(ArgusSurfaceVariant).padding(horizontal = 10.dp, vertical = 5.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(
                                painter = if (isAutoSpeak) painterResource(R.drawable.ic_symbol_volume_up) else painterResource(R.drawable.ic_symbol_volume_off),
                                contentDescription = "Auto Speak",
                                tint = if (isAutoSpeak) ArgusCyanPrimary else ArgusTextMuted,
                                modifier = Modifier.size(14.dp)
                            )
                            Spacer(modifier = Modifier.width(5.dp))
                            Text(text = if (isAutoSpeak) "Voice on" else "Voice off", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp), color = if (isAutoSpeak) ArgusCyanPrimary else ArgusTextMuted)
                        }
                        Text(
                            text = "Clear chat",
                            style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp),
                            color = ArgusCyanPrimary,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.clip(RoundedCornerShape(20.dp)).clickable {
                                viewModel.clearChat()
                                Toast.makeText(context, "Chat cleared", Toast.LENGTH_SHORT).show()
                            }.background(ArgusSurfaceVariant).padding(horizontal = 10.dp, vertical = 5.dp)
                        )
                    }
                }
            }
        }

        // STREAMED DOCUMENT IMPORT PROGRESS
        streamedImportProgress?.let { import ->
            val progressFraction = import.totalChunks?.takeIf { total -> total > 0 }?.let { total ->
                (import.chunksProcessed.toFloat() / total.toFloat()).coerceIn(0f, 1f)
            } ?: if (import.stage == StreamedImportStage.COMPLETED) {
                1f
            } else {
                null
            }
            val statusColor = when (import.stage) {
                StreamedImportStage.COMPLETED -> ArgusSuccess
                StreamedImportStage.FAILED,
                StreamedImportStage.CANCELLED -> ArgusWarning
                else -> ArgusCyanPrimary
            }

            Surface(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 5.dp),
                shape = RoundedCornerShape(14.dp),
                color = ArgusSurfaceVariant
            ) {
                Column(modifier = Modifier.fillMaxWidth().padding(12.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = import.stage.displayName,
                                style = MaterialTheme.typography.labelLarge,
                                color = statusColor,
                                fontWeight = FontWeight.SemiBold
                            )
                            Text(
                                text = import.documentTitle,
                                style = MaterialTheme.typography.bodySmall,
                                color = ArgusTextPrimary,
                                maxLines = 1
                            )
                        }
                        if (!import.stage.isTerminal) {
                            CircularProgressIndicator(
                                modifier = Modifier.size(18.dp),
                                color = ArgusCyanPrimary,
                                strokeWidth = 2.dp
                            )
                        }
                    }
                    Spacer(modifier = Modifier.height(6.dp))
                    Text(
                        text = "${NumberFormat.getIntegerInstance(Locale.US).format(import.charactersRead)} characters read • " +
                            if (import.totalChunks != null) {
                                "${import.chunksProcessed}/${import.totalChunks} chunks"
                            } else {
                                "${import.chunksProcessed} chunks staged"
                            },
                        style = MaterialTheme.typography.labelSmall,
                        color = ArgusTextSecondary,
                        fontFamily = FontFamily.Monospace
                    )
                    if (progressFraction != null) {
                        Spacer(modifier = Modifier.height(6.dp))
                        LinearProgressIndicator(
                            progress = { progressFraction },
                            modifier = Modifier.fillMaxWidth().height(4.dp),
                            color = statusColor,
                            trackColor = ArgusSurface
                        )
                    }
                    import.detail?.let { detail ->
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = detail,
                            style = MaterialTheme.typography.bodySmall,
                            color = if (import.stage == StreamedImportStage.FAILED) {
                                ArgusWarning
                            } else {
                                ArgusTextSecondary
                            }
                        )
                    }
                    if (import.stage.isTerminal) {
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
                            TextButton(onClick = viewModel::dismissStreamedImportProgress) {
                                Text("Dismiss", color = ArgusCyanPrimary)
                            }
                        }
                    }
                }
            }
        }

        // MESSAGES
        if (messages.isEmpty() && !isGenerating) {
            // EMPTY STATE
            Column(
                modifier = Modifier.weight(1f).fillMaxWidth().padding(horizontal = 24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                Image(
                    painter = painterResource(id = R.drawable.ic_argus_logo),
                    contentDescription = null,
                    modifier = Modifier.size(84.dp).clip(CircleShape).border(1.dp, ArgusCyanPrimary, CircleShape)
                )
                Spacer(modifier = Modifier.height(16.dp))
                Text("Hello, I am ARGUS", color = ArgusTextPrimary, fontWeight = FontWeight.Bold, fontSize = 18.sp, textAlign = TextAlign.Center)
                Text(
                    "Ask a question, import a document for training, or use voice input. The local model runs on this device.",
                    color = ArgusTextMuted,
                    fontSize = 13.sp,
                    textAlign = TextAlign.Center,
                    lineHeight = 19.sp
                )
                Spacer(modifier = Modifier.height(18.dp))
                LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    val chips = listOf("Calculate 125 * 34", "What is the A.R.G.U.S. architecture?", "How does LoRA work?")
                    items(chips) { chip ->
                        Surface(
                            shape = RoundedCornerShape(20.dp),
                            color = ArgusSurfaceVariant,
                            border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder),
                            modifier = Modifier.clickable { viewModel.sendMessage(chip) }
                        ) {
                            Text(text = chip, style = MaterialTheme.typography.labelSmall, color = ArgusTextPrimary, modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp))
                        }
                    }
                }
            }
        } else {
            LazyColumn(
                state = listState,
                modifier = Modifier.weight(1f).fillMaxWidth().padding(horizontal = 12.dp),
                contentPadding = PaddingValues(vertical = 12.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                itemsIndexed(
                    items = messages,
                    key = { _, message -> message.id }
                ) { index, msg ->
                    val previousUserMessage = if (msg.isUser) {
                        null
                    } else {
                        messages.subList(0, index)
                            .lastOrNull { it.isUser }
                            ?.takeIf { it.attachedImageUri == null }
                            ?.text
                    }
                    ChatMessageItem(
                        msg = msg,
                        userPromptForTraining = previousUserMessage,
                        onSpeak = { text -> viewModel.speakMessage(text) },
                        onTrain = { prompt, answer, onComplete ->
                            viewModel.enqueueApprovedConversation(prompt, answer, onComplete)
                        }
                    )
                }

                if (isGenerating) {
                    item {
                        Row(
                            modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(ArgusSurfaceVariant.copy(alpha = 0.6f)).padding(12.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            CircularProgressIndicator(modifier = Modifier.size(16.dp), color = ArgusCyanPrimary, strokeWidth = 2.dp)
                            Spacer(modifier = Modifier.width(10.dp))
                            Text(text = "ARGUS is thinking...", style = MaterialTheme.typography.bodyMedium.copy(fontSize = 13.sp), color = ArgusCyanGlow, fontFamily = FontFamily.Monospace)
                        }
                    }
                }
            }
        }

        // GRPO POLICY TRAINING BANNER
        AnimatedVisibility(visible = isGrpoTraining) {
            val progress = grpoProgress
            val completed = progress?.exampleIndex ?: 0
            val total = progress?.totalExamples ?: 0
            val fraction = if (total > 0) completed.toFloat() / total.toFloat() else 0f
            Surface(
                modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 4.dp),
                shape = RoundedCornerShape(12.dp),
                color = ArgusSurfaceVariant,
                border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder)
            ) {
                Column(modifier = Modifier.padding(10.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "GRPO policy training",
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusCyanPrimary,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = if (total > 0) "$completed/$total prompts" else "Validating dataset",
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusTextMuted,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                    Spacer(modifier = Modifier.height(6.dp))
                    androidx.compose.material3.LinearProgressIndicator(
                        progress = { fraction.coerceIn(0f, 1f) },
                        modifier = Modifier.fillMaxWidth().height(4.dp).clip(RoundedCornerShape(2.dp)),
                        color = ArgusCyanPrimary,
                        trackColor = ArgusSurface
                    )
                    if (progress != null && progress.generatedResponses > 0) {
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = "Groups updated: ${progress.trainedGroups} • Mean reward: " +
                                if (progress.meanReward.isFinite()) String.format(Locale.US, "%.3f", progress.meanReward) else "n/a",
                            style = MaterialTheme.typography.labelSmall,
                            color = ArgusTextSecondary
                        )
                    }
                }
            }
        }

        // TRAINING BANNER
        AnimatedVisibility(
            visible = activeDoc != null && (activeDoc?.completedChunks ?: 0) < (activeDoc?.totalChunks ?: 1)
        ) {
            activeDoc?.let { doc ->
                val pct = if (doc.totalChunks > 0) (doc.completedChunks.toFloat() / doc.totalChunks.toFloat() * 100).toInt() else 0
                Surface(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 4.dp),
                    shape = RoundedCornerShape(12.dp),
                    color = ArgusCyanPrimary.copy(alpha = 0.12f),
                    border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCyanPrimary)
                ) {
                    Column(modifier = Modifier.padding(10.dp)) {
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                            Text(text = "Training: ${doc.docTitle}", style = MaterialTheme.typography.labelSmall, color = ArgusCyanPrimary, fontWeight = FontWeight.Bold, fontSize = 11.sp)
                            Text(text = "${doc.completedChunks}/${doc.totalChunks} ($pct%)", style = MaterialTheme.typography.labelSmall, color = ArgusCyanGlow, fontFamily = FontFamily.Monospace, fontSize = 10.sp)
                        }
                        Spacer(modifier = Modifier.height(6.dp))
                        androidx.compose.material3.LinearProgressIndicator(
                            progress = { if (doc.totalChunks > 0) doc.completedChunks.toFloat() / doc.totalChunks.toFloat() else 0f },
                            modifier = Modifier.fillMaxWidth().height(4.dp).clip(RoundedCornerShape(2.dp)),
                            color = ArgusCyanPrimary,
                            trackColor = ArgusSurfaceVariant
                        )
                    }
                }
            }
        }

        // SUGGESTION CHIPS - hide when keyboard open
        if (messages.isNotEmpty()) {
            AnimatedVisibility(visible = !isKeyboardOpen) {
                LazyRow(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 6.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    val chips = listOf("Calculate 125 * 34", "What is the A.R.G.U.S architecture?", "Explore recent AGI developments", "How does LoRA work?")
                    items(chips) { chip ->
                        Surface(
                            shape = RoundedCornerShape(20.dp),
                            color = ArgusSurfaceVariant,
                            border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder),
                            modifier = Modifier.clickable { viewModel.sendMessage(chip) }
                        ) {
                            Text(text = chip, style = MaterialTheme.typography.labelSmall, color = ArgusTextPrimary, modifier = Modifier.padding(horizontal = 12.dp, vertical = 7.dp))
                        }
                    }
                }
            }
        }

        // INPUT BAR
        Surface(modifier = Modifier.fillMaxWidth(), color = ArgusSurface, tonalElevation = 4.dp) {
            Column(modifier = Modifier.fillMaxWidth()) {
                AnimatedVisibility(visible = isListening) {
                    Row(
                        modifier = Modifier.fillMaxWidth().background(ArgusWarning.copy(alpha = 0.18f)).padding(horizontal = 12.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(painter = painterResource(R.drawable.ic_symbol_mic), contentDescription = null, tint = ArgusWarning, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(text = if (partialSpeech.isNotBlank()) "Heard: \"$partialSpeech\"" else "Listening...", style = MaterialTheme.typography.bodySmall, color = ArgusWarning, modifier = Modifier.weight(1f))
                        Text(text = "Done", style = MaterialTheme.typography.labelSmall, color = ArgusWarning, fontWeight = FontWeight.Bold, modifier = Modifier.clip(RoundedCornerShape(4.dp)).clickable { viewModel.stopVoiceInput() }.padding(horizontal = 6.dp, vertical = 2.dp))
                    }
                }

                AnimatedVisibility(visible = isSpeaking) {
                    Row(
                        modifier = Modifier.fillMaxWidth().background(ArgusCyanPrimary.copy(alpha = 0.15f)).padding(horizontal = 12.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(painter = painterResource(R.drawable.ic_symbol_volume_up), contentDescription = null, tint = ArgusCyanPrimary, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(text = "Argus is speaking...", style = MaterialTheme.typography.bodySmall, color = ArgusCyanPrimary, modifier = Modifier.weight(1f))
                        IconButton(onClick = { viewModel.stopSpeaking() }, modifier = Modifier.size(26.dp)) {
                            Icon(painter = painterResource(R.drawable.ic_symbol_stop), contentDescription = "Stop", tint = ArgusCyanPrimary, modifier = Modifier.size(16.dp))
                        }
                    }
                }

                Row(
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    IconButton(onClick = { showAttachDialog = true }, modifier = Modifier.size(40.dp)) {
                        Icon(painter = painterResource(R.drawable.ic_symbol_attach_file), contentDescription = "Attach", tint = ArgusCyanPrimary)
                    }

                    OutlinedTextField(
                        value = inputText,
                        onValueChange = { inputText = it },
                        placeholder = { Text("Ask a question, run an action, or train...", color = ArgusTextMuted, fontSize = 13.sp) },
                        trailingIcon = {
                            if (inputText.isEmpty()) {
                                IconButton(
                                    onClick = {
                                        clipboardScope.launch {
                                            val clipText = clipboard.getClipEntry()?.clipData?.takeIf { it.itemCount > 0 }?.getItemAt(0)?.coerceToText(context)?.toString()
                                            if (!clipText.isNullOrBlank()) {
                                                inputText = clipText
                                                Toast.makeText(context, "Text pasted from clipboard", Toast.LENGTH_SHORT).show()
                                            } else {
                                                Toast.makeText(context, "Clipboard is empty", Toast.LENGTH_SHORT).show()
                                            }
                                        }
                                    },
                                    modifier = Modifier.size(30.dp)
                                ) {
                                    Icon(painter = painterResource(R.drawable.ic_symbol_content_paste), contentDescription = "Paste", tint = ArgusTextMuted, modifier = Modifier.size(16.dp))
                                }
                            }
                        },
                        modifier = Modifier.weight(1f).padding(horizontal = 4.dp).testTag("chat_input_field"),
                        shape = RoundedCornerShape(22.dp),
                        colors = OutlinedTextFieldDefaults.colors(
                            focusedBorderColor = ArgusCyanPrimary,
                            unfocusedBorderColor = ArgusCardBorder,
                            focusedTextColor = ArgusTextPrimary,
                            unfocusedTextColor = ArgusTextPrimary,
                            focusedContainerColor = ArgusSurfaceVariant,
                            unfocusedContainerColor = ArgusSurfaceVariant
                        ),
                        maxLines = 5
                    )

                    IconButton(
                        onClick = {
                            if (isListening) {
                                viewModel.stopVoiceInput()
                            } else {
                                val permCheck = androidx.core.content.ContextCompat.checkSelfPermission(context, android.Manifest.permission.RECORD_AUDIO)
                                if (permCheck == android.content.pm.PackageManager.PERMISSION_GRANTED) {
                                    viewModel.startVoiceInput(
                                        onResult = { recognized -> if (recognized.isNotBlank()) viewModel.sendMessage(recognized) },
                                        onError = { err -> Toast.makeText(context, err, Toast.LENGTH_SHORT).show() }
                                    )
                                } else {
                                    audioPermissionLauncher.launch(android.Manifest.permission.RECORD_AUDIO)
                                }
                            }
                        },
                        modifier = Modifier.size(40.dp).clip(CircleShape).background(if (isListening) ArgusWarning else ArgusSurfaceVariant)
                    ) {
                        Icon(painter = if (isListening) painterResource(R.drawable.ic_symbol_mic_off) else painterResource(R.drawable.ic_symbol_mic), contentDescription = "Voice", tint = if (isListening) ArgusDarkBg else ArgusCyanPrimary, modifier = Modifier.size(18.dp))
                    }

                    Spacer(modifier = Modifier.width(6.dp))

                    IconButton(
                        onClick = {
                            val text = inputText.trim()
                            if (text.isNotBlank()) {
                                viewModel.sendMessage(text)
                                inputText = ""
                            }
                        },
                        modifier = Modifier.size(44.dp).clip(CircleShape).background(ArgusCyanPrimary).testTag("send_button")
                    ) {
                        Icon(painter = painterResource(R.drawable.ic_symbol_send), contentDescription = "Send", tint = ArgusDarkBg, modifier = Modifier.size(20.dp))
                    }
                }
            }
        }
    }
}

@Composable
fun ChatMessageItem(
    msg: ChatMessage,
    userPromptForTraining: String? = null,
    onSpeak: ((String) -> Unit)? = null,
    onTrain: ((String, String, (Boolean) -> Unit) -> Unit)? = null
) {
    val context = LocalContext.current
    val clipboard = LocalClipboard.current
    val clipboardScope = rememberCoroutineScope()

    var isThinkingExpanded by remember { mutableStateOf(false) }
    var isLongTextExpanded by remember { mutableStateOf(false) }
    var isCopied by remember { mutableStateOf(false) }
    var isTrainingRequestPending by remember(msg.id) { mutableStateOf(false) }
    var isTrainingSubmitted by remember(msg.id) { mutableStateOf(false) }

    val isLongMessage = msg.text.length > 450
    val displayText = if (isLongMessage && !isLongTextExpanded) msg.text.take(512) + "..." else msg.text

    Column(modifier = Modifier.fillMaxWidth(), horizontalAlignment = if (msg.isUser) Alignment.End else Alignment.Start) {
        Text(
            text = if (msg.isUser) "You" else "A.R.G.U.S",
            style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp),
            color = if (msg.isUser) ArgusBlueAccent else ArgusCyanPrimary,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(horizontal = 6.dp)
        )
        Spacer(modifier = Modifier.height(4.dp))

        Surface(
            shape = RoundedCornerShape(topStart = 18.dp, topEnd = 18.dp, bottomStart = if (msg.isUser) 18.dp else 4.dp, bottomEnd = if (msg.isUser) 4.dp else 18.dp),
            color = if (msg.isUser) ArgusBlueAccent else ArgusSurfaceVariant,
            border = androidx.compose.foundation.BorderStroke(1.dp, if (msg.isUser) ArgusBlueAccent else ArgusCardBorder),
            modifier = Modifier.padding(start = if (msg.isUser) 48.dp else 0.dp, end = if (msg.isUser) 0.dp else 48.dp)
        ) {
            Column(modifier = Modifier.padding(12.dp)) {
                if (msg.attachedImageUri != null) {
                    Image(
                        painter = rememberAsyncImagePainter(model = msg.attachedImageUri),
                        contentDescription = "Attached image",
                        modifier = Modifier.fillMaxWidth().height(160.dp).clip(RoundedCornerShape(10.dp)),
                        contentScale = ContentScale.Crop
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                }

                if (msg.attachedFileName != null) {
                    Row(
                        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).background(ArgusDarkBg.copy(alpha = 0.5f)).padding(8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(painter = painterResource(R.drawable.ic_symbol_attach_file), contentDescription = "File", tint = ArgusCyanPrimary, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(text = msg.attachedFileName, style = MaterialTheme.typography.labelSmall, color = ArgusTextPrimary)
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                }

                if (!msg.isUser && !msg.thinkReasoning.isNullOrBlank()) {
                    Row(
                        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).background(ArgusDarkBg.copy(alpha = 0.6f)).clickable { isThinkingExpanded = !isThinkingExpanded }.padding(horizontal = 8.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(painter = painterResource(R.drawable.ic_symbol_psychology), contentDescription = "Reasoning", tint = ArgusCyanGlow, modifier = Modifier.size(16.dp))
                        Spacer(modifier = Modifier.width(6.dp))
                        Text(text = "Reasoning", style = MaterialTheme.typography.labelSmall, color = ArgusCyanGlow, modifier = Modifier.weight(1f))
                        Icon(painter = if (isThinkingExpanded) painterResource(R.drawable.ic_symbol_keyboard_arrow_up) else painterResource(R.drawable.ic_symbol_keyboard_arrow_down), contentDescription = "Toggle", tint = ArgusCyanGlow, modifier = Modifier.size(16.dp))
                    }
                    AnimatedVisibility(visible = isThinkingExpanded, enter = expandVertically() + fadeIn(), exit = shrinkVertically() + fadeOut()) {
                        Text(text = msg.thinkReasoning, style = MaterialTheme.typography.bodyMedium.copy(fontSize = 11.sp, fontFamily = FontFamily.Monospace), color = ArgusTextSecondary, modifier = Modifier.fillMaxWidth().padding(top = 6.dp, start = 4.dp, end = 4.dp))
                    }
                    Spacer(modifier = Modifier.height(8.dp))
                }

                SelectionContainer {
                    Text(text = displayText, style = MaterialTheme.typography.bodyLarge.copy(fontSize = 15.sp, lineHeight = 22.sp), color = if (msg.isUser) androidx.compose.ui.graphics.Color.White else ArgusTextPrimary)
                }

                if (isLongMessage) {
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = if (isLongTextExpanded) "Collapse" else "Read more",
                        style = MaterialTheme.typography.labelSmall,
                        color = if (msg.isUser) androidx.compose.ui.graphics.Color.White else ArgusCyanPrimary,
                        fontWeight = FontWeight.Bold,
                        modifier = Modifier.clip(RoundedCornerShape(4.dp)).clickable { isLongTextExpanded = !isLongTextExpanded }.padding(vertical = 2.dp)
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End, verticalAlignment = Alignment.CenterVertically) {
                    if (!msg.isUser) {
                        val trainCallback = onTrain
                        if (!userPromptForTraining.isNullOrBlank() && trainCallback != null) {
                            TextButton(
                                enabled = !isTrainingRequestPending && !isTrainingSubmitted,
                                onClick = {
                                    val prompt = userPromptForTraining
                                    if (!prompt.isNullOrBlank()) {
                                        isTrainingRequestPending = true
                                        trainCallback.invoke(prompt, msg.text) { accepted ->
                                            isTrainingRequestPending = false
                                            isTrainingSubmitted = accepted
                                            val feedback = if (accepted) {
                                                "Exchange added to the local training corpus"
                                            } else {
                                                "Exchange was already stored or could not be queued"
                                            }
                                            Toast.makeText(context, feedback, Toast.LENGTH_SHORT).show()
                                        }
                                    }
                                }
                            ) {
                                val label = when {
                                    isTrainingRequestPending -> "Adding…"
                                    isTrainingSubmitted -> "Added"
                                    else -> "Train exchange"
                                }
                                Text(
                                    text = label,
                                    color = if (isTrainingSubmitted) ArgusSuccess else ArgusCyanPrimary,
                                    fontSize = 10.sp
                                )
                            }
                            Spacer(modifier = Modifier.width(4.dp))
                        }
                        Row(
                            modifier = Modifier.clip(RoundedCornerShape(20.dp)).background(ArgusDarkBg.copy(alpha = 0.45f)).clickable { onSpeak?.invoke(msg.text) }.padding(horizontal = 8.dp, vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Icon(painter = painterResource(R.drawable.ic_symbol_volume_up), contentDescription = "Speak", tint = ArgusCyanPrimary, modifier = Modifier.size(13.dp))
                            Spacer(modifier = Modifier.width(4.dp))
                            Text(text = "Speak", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp), color = ArgusCyanPrimary)
                        }
                        Spacer(modifier = Modifier.width(6.dp))
                    }
                    Row(
                        modifier = Modifier.clip(RoundedCornerShape(20.dp)).background(ArgusDarkBg.copy(alpha = 0.4f)).clickable {
                            clipboardScope.launch {
                                clipboard.setClipEntry(ClipEntry(ClipData.newPlainText("Argus message", msg.text)))
                            }
                            isCopied = true
                            Toast.makeText(context, "Message copied", Toast.LENGTH_SHORT).show()
                        }.padding(horizontal = 8.dp, vertical = 4.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Icon(painter = if (isCopied) painterResource(R.drawable.ic_symbol_check) else painterResource(R.drawable.ic_symbol_content_copy), contentDescription = "Copy", tint = if (isCopied) ArgusSuccess else ArgusTextMuted, modifier = Modifier.size(13.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(text = if (isCopied) "Copied" else "Copy", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp), color = if (isCopied) ArgusSuccess else ArgusTextMuted)
                    }
                }
            }
        }
    }
}
