package com.argus.agi.ui.screens

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.PowerManager
import android.provider.Settings
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
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.CheckboxDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
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
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.argus.agi.ui.ArgusViewModel
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
import androidx.compose.ui.res.painterResource
import com.argus.agi.R

@Composable
fun SkillsAndSettingsScreen(viewModel: ArgusViewModel) {
    val context = LocalContext.current
    val installedApps by viewModel.installedApps.collectAsState()
    val worldEntities by viewModel.worldEntities.collectAsState()
    val searchResults by viewModel.searchResults.collectAsState()
    val articlePreview by viewModel.articlePreview.collectAsState()
    val isExtractingArticle by viewModel.isExtractingArticle.collectAsState()
    val articleExtractionError by viewModel.articleExtractionError.collectAsState()
    val isQueueingArticle by viewModel.isQueueingArticle.collectAsState()
    val articleQueueStatus by viewModel.articleQueueStatus.collectAsState()

    var isBatteryExpanded by remember { mutableStateOf(true) }
    var isSkillsExpanded by remember { mutableStateOf(true) }
    var isExcludeAppsExpanded by remember { mutableStateOf(true) }
    var isWorldModelExpanded by remember { mutableStateOf(true) }

    var searchQuery by remember { mutableStateOf("") }
    var allowArticleTraining by remember { mutableStateOf(false) }
    LaunchedEffect(articlePreview?.canonicalUrl) {
        allowArticleTraining = false
    }
    var actionClickQuery by remember { mutableStateOf("") }
    var actionClickResult by remember { mutableStateOf<String?>(null) }
    var appFilterText by remember { mutableStateOf("") }
    var appFilterTab by remember { mutableStateOf(0) }

    val pm = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
    val isIgnoringOpt = pm?.isIgnoringBatteryOptimizations(context.packageName) ?: false

    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .background(ArgusDarkBg)
            .imePadding()
            .padding(horizontal = 16.dp),
        contentPadding = PaddingValues(vertical = 16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Header
        item {
            Column {
                Text(
                    text = "AGENT CAPABILITIES & PRIVACY",
                    style = MaterialTheme.typography.titleLarge,
                    color = ArgusCyanPrimary,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Configure app exclusions, background power settings, and accessibility actions",
                    style = MaterialTheme.typography.bodyMedium,
                    color = ArgusTextSecondary
                )
            }
        }

        // 1. Battery Optimization Card (Collapsible)
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
                            .clickable { isBatteryExpanded = !isBatteryExpanded }
                            .padding(vertical = 2.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                painter = if (isIgnoringOpt) painterResource(R.drawable.ic_symbol_battery_charging_full) else painterResource(R.drawable.ic_symbol_battery_alert),
                                contentDescription = "Battery",
                                tint = if (isIgnoringOpt) ArgusSuccess else ArgusWarning,
                                modifier = Modifier.size(22.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "Battery & Background Optimization",
                                style = MaterialTheme.typography.titleMedium,
                                color = ArgusTextPrimary,
                                fontWeight = FontWeight.Bold
                            )
                        }

                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(if (isIgnoringOpt) ArgusSuccess.copy(alpha = 0.15f) else ArgusWarning.copy(alpha = 0.15f))
                                    .padding(horizontal = 8.dp, vertical = 4.dp)
                            ) {
                                Text(
                                    text = if (isIgnoringOpt) "EXEMPT" else "RESTRICTED",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = if (isIgnoringOpt) ArgusSuccess else ArgusWarning,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                            Spacer(modifier = Modifier.width(6.dp))
                            Icon(
                                painter = if (isBatteryExpanded) painterResource(R.drawable.ic_symbol_keyboard_arrow_up) else painterResource(R.drawable.ic_symbol_keyboard_arrow_down),
                                contentDescription = "Toggle",
                                tint = ArgusCyanPrimary,
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }

                    AnimatedVisibility(
                        visible = isBatteryExpanded,
                        enter = expandVertically() + fadeIn(),
                        exit = shrinkVertically() + fadeOut()
                    ) {
                        Column {
                            Spacer(modifier = Modifier.height(10.dp))

                            Text(
                                text = "To help background training and perception continue, exempt Argus from battery optimization. Android may still apply system background limits.",
                                style = MaterialTheme.typography.labelSmall,
                                color = ArgusTextSecondary
                            )

                            if (!isIgnoringOpt) {
                                Spacer(modifier = Modifier.height(12.dp))
                                Button(
                                    onClick = {
                                        try {
                                            val intent = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply {
                                                data = Uri.parse("package:${context.packageName}")
                                            }
                                            context.startActivity(intent)
                                        } catch (_: Exception) {
                                            val fallbackIntent = Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS)
                                            context.startActivity(fallbackIntent)
                                        }
                                    },
                                    colors = ButtonDefaults.buttonColors(
                                        containerColor = ArgusCyanPrimary,
                                        contentColor = ArgusDarkBg
                                    ),
                                    shape = RoundedCornerShape(10.dp),
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Text(
                                        text = "Open battery optimization settings",
                                        fontWeight = FontWeight.Bold,
                                        fontSize = 12.sp
                                    )
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. Skills Catalog & Tester (Collapsible)
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
                            .clickable { isSkillsExpanded = !isSkillsExpanded }
                            .padding(vertical = 2.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                painter = painterResource(R.drawable.ic_symbol_extension),
                                contentDescription = "Skills",
                                tint = ArgusCyanPrimary,
                                modifier = Modifier.size(22.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "A.R.G.U.S Core Capabilities",
                                style = MaterialTheme.typography.titleMedium,
                                color = ArgusTextPrimary,
                                fontWeight = FontWeight.Bold
                            )
                        }

                        Icon(
                            painter = if (isSkillsExpanded) painterResource(R.drawable.ic_symbol_keyboard_arrow_up) else painterResource(R.drawable.ic_symbol_keyboard_arrow_down),
                            contentDescription = "Toggle",
                            tint = ArgusCyanPrimary,
                            modifier = Modifier.size(20.dp)
                        )
                    }

                    AnimatedVisibility(
                        visible = isSkillsExpanded,
                        enter = expandVertically() + fadeIn(),
                        exit = shrinkVertically() + fadeOut()
                    ) {
                        Column {
                            Spacer(modifier = Modifier.height(12.dp))

                            SkillBadge(title = "1. General Web Search", desc = "Uses DuckDuckGo general-web search; Wikipedia supplements short result sets.")
                            SkillBadge(title = "2. Article Extraction", desc = "Fetches a selected web page and extracts readable article text for review.")
                            SkillBadge(title = "3. Screen Text Perception", desc = "Reads accessible text from active app windows while Accessibility is enabled; excluded apps are skipped.")
                            SkillBadge(title = "4. Accessibility Actions", desc = "Attempts to activate a matching visible text node through Android Accessibility.")
                            SkillBadge(title = "5. Deterministic Arithmetic", desc = "Evaluates supported arithmetic expressions deterministically.")
                            SkillBadge(title = "6. World Model Memory", desc = "Stores local key-value facts, source labels, and application context records.")
                            SkillBadge(title = "7. Local Model Training", desc = "Updates model weights and LoRA adapters locally through on-device training.")

                            Spacer(modifier = Modifier.height(14.dp))

                            // Direct Web Search Tester
                            Text(
                                text = "Test Web Search:",
                                style = MaterialTheme.typography.labelSmall,
                                color = ArgusCyanGlow,
                                fontWeight = FontWeight.Bold
                            )
                            Spacer(modifier = Modifier.height(6.dp))
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                OutlinedTextField(
                                    value = searchQuery,
                                    onValueChange = { searchQuery = it },
                                    placeholder = { Text("Search query (e.g. on-device AI)", color = ArgusTextMuted, fontSize = 12.sp) },
                                    modifier = Modifier.weight(1f),
                                    shape = RoundedCornerShape(10.dp),
                                    colors = OutlinedTextFieldDefaults.colors(
                                        focusedBorderColor = ArgusCyanPrimary,
                                        unfocusedBorderColor = ArgusCardBorder,
                                        focusedTextColor = ArgusTextPrimary,
                                        unfocusedTextColor = ArgusTextPrimary,
                                        focusedContainerColor = ArgusSurfaceVariant,
                                        unfocusedContainerColor = ArgusSurfaceVariant
                                    ),
                                    singleLine = true
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Button(
                                    onClick = {
                                        if (searchQuery.isNotBlank()) {
                                            viewModel.searchWebDirectly(searchQuery)
                                        }
                                    },
                                    colors = ButtonDefaults.buttonColors(
                                        containerColor = ArgusCyanPrimary,
                                        contentColor = ArgusDarkBg
                                    ),
                                    shape = RoundedCornerShape(10.dp)
                                ) {
                                    Text("Search", fontWeight = FontWeight.Bold, fontSize = 12.sp)
                                }
                            }

                            if (searchResults.isNotEmpty()) {
                                Spacer(modifier = Modifier.height(8.dp))
                                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                    searchResults.take(5).forEach { item ->
                                        Surface(
                                            shape = RoundedCornerShape(10.dp),
                                            color = ArgusDarkBg,
                                            modifier = Modifier.fillMaxWidth()
                                        ) {
                                            Column(modifier = Modifier.padding(10.dp)) {
                                                Text(
                                                    text = item.title,
                                                    style = MaterialTheme.typography.labelLarge,
                                                    color = ArgusCyanPrimary,
                                                    fontWeight = FontWeight.Bold
                                                )
                                                Spacer(modifier = Modifier.height(4.dp))
                                                Text(
                                                    text = item.snippet,
                                                    style = MaterialTheme.typography.bodySmall,
                                                    color = ArgusTextSecondary
                                                )
                                                Spacer(modifier = Modifier.height(4.dp))
                                                Text(
                                                    text = item.sourceUrl,
                                                    style = MaterialTheme.typography.labelSmall,
                                                    color = ArgusTextMuted
                                                )
                                                Text(
                                                    text = "Provider: ${item.provider}",
                                                    style = MaterialTheme.typography.labelSmall,
                                                    color = ArgusTextMuted
                                                )
                                                Spacer(modifier = Modifier.height(6.dp))
                                                Button(
                                                    onClick = { viewModel.extractWebArticle(item) },
                                                    enabled = !isExtractingArticle,
                                                    colors = ButtonDefaults.buttonColors(
                                                        containerColor = ArgusCyanPrimary,
                                                        contentColor = ArgusDarkBg
                                                    ),
                                                    shape = RoundedCornerShape(8.dp)
                                                ) {
                                                    Text("Extract article", fontWeight = FontWeight.SemiBold, fontSize = 11.sp)
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            if (isExtractingArticle) {
                                Text(
                                    text = "Fetching and extracting page content…",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = ArgusCyanGlow
                                )
                            }

                            articleExtractionError?.let { error ->
                                Text(
                                    text = "Extraction failed: $error",
                                    style = MaterialTheme.typography.bodySmall,
                                    color = ArgusWarning
                                )
                            }

                            articlePreview?.let { article ->
                                Spacer(modifier = Modifier.height(8.dp))
                                Surface(
                                    shape = RoundedCornerShape(12.dp),
                                    color = ArgusSurfaceVariant,
                                    border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder),
                                    modifier = Modifier.fillMaxWidth()
                                ) {
                                    Column(
                                        modifier = Modifier.padding(12.dp),
                                        verticalArrangement = Arrangement.spacedBy(6.dp)
                                    ) {
                                        Text(
                                            text = "ARTICLE PREVIEW · NOT QUEUED",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = ArgusCyanGlow,
                                            fontWeight = FontWeight.Bold
                                        )
                                        Text(
                                            text = article.title,
                                            style = MaterialTheme.typography.titleSmall,
                                            color = ArgusTextPrimary,
                                            fontWeight = FontWeight.Bold
                                        )
                                        Text(
                                            text = "${article.siteName} · ${article.domain} · ${article.characterCount} karakter",
                                            style = MaterialTheme.typography.labelSmall,
                                            color = ArgusTextMuted
                                        )
                                        article.publishedAt?.let { publishedAt ->
                                            Text(
                                                text = "Source date: $publishedAt",
                                                style = MaterialTheme.typography.labelSmall,
                                                color = ArgusTextMuted
                                            )
                                        }
                                        Text(
                                            text = article.canonicalUrl,
                                            style = MaterialTheme.typography.labelSmall,
                                            color = ArgusTextMuted
                                        )
                                        Text(
                                            text = article.text.take(2_800),
                                            style = MaterialTheme.typography.bodySmall,
                                            color = ArgusTextSecondary
                                        )
                                        Text(
                                            text = "Review the content, date, publisher, and source URL before adding it to training.",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = ArgusTextPrimary
                                        )
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            Checkbox(
                                                checked = allowArticleTraining,
                                                onCheckedChange = { allowArticleTraining = it },
                                                colors = CheckboxDefaults.colors(
                                                    checkedColor = ArgusCyanPrimary,
                                                    uncheckedColor = ArgusTextMuted,
                                                    checkmarkColor = ArgusDarkBg
                                                )
                                            )
                                            Text(
                                                text = "I confirm that I have the rights or permission to use this article as training data.",
                                                style = MaterialTheme.typography.bodySmall,
                                                color = ArgusTextSecondary
                                            )
                                        }
                                        Row(
                                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Button(
                                                onClick = { viewModel.enqueueReviewedWebArticle() },
                                                enabled = allowArticleTraining && !isQueueingArticle,
                                                colors = ButtonDefaults.buttonColors(
                                                    containerColor = ArgusSuccess,
                                                    contentColor = ArgusDarkBg,
                                                    disabledContainerColor = ArgusSurface,
                                                    disabledContentColor = ArgusTextMuted
                                                ),
                                                shape = RoundedCornerShape(8.dp)
                                            ) {
                                                Text(
                                                    if (isQueueingArticle) "Memasukkan…" else "Masukkan ke training",
                                                    fontSize = 11.sp
                                                )
                                            }
                                            Button(
                                                onClick = { viewModel.dismissWebArticlePreview() },
                                                enabled = !isQueueingArticle,
                                                colors = ButtonDefaults.buttonColors(
                                                    containerColor = ArgusSurface,
                                                    contentColor = ArgusTextPrimary
                                                ),
                                                shape = RoundedCornerShape(8.dp)
                                            ) {
                                                Text("Close", fontSize = 11.sp)
                                            }
                                        }
                                    }
                                }
                            }

                            articleQueueStatus?.let { status ->
                                Text(
                                    text = status,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = ArgusTextPrimary
                                )
                            }

                            Spacer(modifier = Modifier.height(14.dp))

                            // Action Agent Click Tester
                            Text(
                                text = "Test an Accessibility Action:",
                                style = MaterialTheme.typography.labelSmall,
                                color = ArgusCyanGlow,
                                fontWeight = FontWeight.Bold
                            )
                            Spacer(modifier = Modifier.height(6.dp))
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                OutlinedTextField(
                                    value = actionClickQuery,
                                    onValueChange = { actionClickQuery = it },
                                    placeholder = { Text("Visible button text", color = ArgusTextMuted, fontSize = 12.sp) },
                                    modifier = Modifier.weight(1f),
                                    shape = RoundedCornerShape(10.dp),
                                    colors = OutlinedTextFieldDefaults.colors(
                                        focusedBorderColor = ArgusCyanPrimary,
                                        unfocusedBorderColor = ArgusCardBorder,
                                        focusedTextColor = ArgusTextPrimary,
                                        unfocusedTextColor = ArgusTextPrimary,
                                        focusedContainerColor = ArgusSurfaceVariant,
                                        unfocusedContainerColor = ArgusSurfaceVariant
                                    ),
                                    singleLine = true
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Button(
                                    onClick = {
                                        if (actionClickQuery.isNotBlank()) {
                                            val success = viewModel.executeActionClick(actionClickQuery)
                                            actionClickResult = if (success) "Success: click action sent to '${actionClickQuery}'." else "Failed: '${actionClickQuery}' was not found in the active window."
                                        }
                                    },
                                    colors = ButtonDefaults.buttonColors(
                                        containerColor = ArgusBlueAccent,
                                        contentColor = ArgusDarkBg
                                    ),
                                    shape = RoundedCornerShape(10.dp)
                                ) {
                                    Text("Tap", fontWeight = FontWeight.Bold, fontSize = 12.sp)
                                }
                            }

                            if (actionClickResult != null) {
                                Spacer(modifier = Modifier.height(6.dp))
                                Text(
                                    text = actionClickResult!!,
                                    style = MaterialTheme.typography.labelSmall,
                                    color = if (actionClickResult!!.startsWith("Success")) ArgusSuccess else ArgusWarning,
                                    fontFamily = FontFamily.Monospace
                                )
                            }
                        }
                    }
                }
            }
        }

        // 3. Exclude App Management Section (Collapsible)
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
                            .clickable { isExcludeAppsExpanded = !isExcludeAppsExpanded }
                            .padding(vertical = 2.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                painter = painterResource(R.drawable.ic_symbol_visibility_off),
                                contentDescription = "Exclude",
                                tint = ArgusCyanPrimary,
                                modifier = Modifier.size(22.dp)
                            )
                            Spacer(modifier = Modifier.width(8.dp))
                            Text(
                                text = "Excluded Apps",
                                style = MaterialTheme.typography.titleMedium,
                                color = ArgusTextPrimary,
                                fontWeight = FontWeight.Bold
                            )
                        }

                        Row(verticalAlignment = Alignment.CenterVertically) {
                            val excludedCount = installedApps.count { it.isExcluded }
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(ArgusCyanPrimary.copy(alpha = 0.15f))
                                    .padding(horizontal = 8.dp, vertical = 4.dp)
                            ) {
                                Text(
                                    text = "$excludedCount excluded",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = ArgusCyanPrimary
                                )
                            }
                            Spacer(modifier = Modifier.width(6.dp))
                            Icon(
                                painter = if (isExcludeAppsExpanded) painterResource(R.drawable.ic_symbol_keyboard_arrow_up) else painterResource(R.drawable.ic_symbol_keyboard_arrow_down),
                                contentDescription = "Toggle",
                                tint = ArgusCyanPrimary,
                                modifier = Modifier.size(20.dp)
                            )
                        }
                    }

                    AnimatedVisibility(
                        visible = isExcludeAppsExpanded,
                        enter = expandVertically() + fadeIn(),
                        exit = shrinkVertically() + fadeOut()
                    ) {
                        Column {
                            Spacer(modifier = Modifier.height(8.dp))

                            Text(
                                text = "Text from selected apps is excluded from Accessibility-based screen capture and processing. This reduces exposure but does not replace Android permission controls.",
                                style = MaterialTheme.typography.labelSmall,
                                color = ArgusTextSecondary
                            )

                            Spacer(modifier = Modifier.height(10.dp))

                            OutlinedTextField(
                                value = appFilterText,
                                onValueChange = { appFilterText = it },
                                placeholder = { Text("Search installed apps...", color = ArgusTextMuted, fontSize = 12.sp) },
                                modifier = Modifier.fillMaxWidth(),
                                shape = RoundedCornerShape(10.dp),
                                colors = OutlinedTextFieldDefaults.colors(
                                    focusedBorderColor = ArgusCyanPrimary,
                                    unfocusedBorderColor = ArgusCardBorder,
                                    focusedTextColor = ArgusTextPrimary,
                                    unfocusedTextColor = ArgusTextPrimary,
                                    focusedContainerColor = ArgusSurfaceVariant,
                                    unfocusedContainerColor = ArgusSurfaceVariant
                                ),
                                singleLine = true
                            )

                            Spacer(modifier = Modifier.height(8.dp))

                            // Filter Category Chips
                            val sensitiveKeywords = remember {
                                listOf("bank", "bca", "bri", "mandiri", "bni", "cimb", "dana", "ovo", "gopay", "shopeepay", "wallet", "crypto", "binance", "indodax", "auth", "authenticator", "password", "bitwarden", "1password", "keepass", "whatsapp", "telegram", "signal")
                            }

                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                val tabs = listOf(
                                    "All (${installedApps.size})",
                                    "Excluded (${installedApps.count { it.isExcluded }})",
                                    "Privacy suggestions (${installedApps.count { item -> sensitiveKeywords.any { item.packageName.lowercase().contains(it) } }})"
                                )

                                tabs.forEachIndexed { idx, label ->
                                    val isSelected = appFilterTab == idx
                                    Box(
                                        modifier = Modifier
                                            .clip(RoundedCornerShape(8.dp))
                                            .background(if (isSelected) ArgusCyanPrimary else ArgusSurfaceVariant)
                                            .clickable { appFilterTab = idx }
                                            .padding(horizontal = 8.dp, vertical = 5.dp)
                                    ) {
                                        Text(
                                            text = label,
                                            style = MaterialTheme.typography.labelSmall,
                                            color = if (isSelected) ArgusDarkBg else ArgusTextSecondary,
                                            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal,
                                            fontSize = 10.sp
                                        )
                                    }
                                }
                            }

                            Spacer(modifier = Modifier.height(10.dp))

                            val filteredApps = installedApps.filter { app ->
                                val matchesText = appFilterText.isBlank() ||
                                        app.appName.contains(appFilterText, ignoreCase = true) ||
                                        app.packageName.contains(appFilterText, ignoreCase = true)

                                val matchesTab = when (appFilterTab) {
                                    1 -> app.isExcluded
                                    2 -> sensitiveKeywords.any { app.packageName.lowercase().contains(it) }
                                    else -> true
                                }
                                matchesText && matchesTab
                            }

                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Text(
                                    text = "Showing ${filteredApps.size} apps (scroll to browse)",
                                    style = MaterialTheme.typography.labelSmall,
                                    color = ArgusCyanGlow,
                                    fontFamily = FontFamily.Monospace,
                                    fontSize = 10.sp
                                )
                            }

                            Spacer(modifier = Modifier.height(6.dp))

                            // Smoothly scrollable container allowing access to ALL installed apps
                            Column(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .heightIn(max = 380.dp)
                                    .verticalScroll(rememberScrollState()),
                                verticalArrangement = Arrangement.spacedBy(6.dp)
                            ) {
                                if (filteredApps.isEmpty()) {
                                    Box(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .padding(20.dp),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        Text(
                                            text = "No apps match this filter.",
                                            style = MaterialTheme.typography.bodyMedium.copy(fontSize = 11.sp),
                                            color = ArgusTextMuted
                                        )
                                    }
                                } else {
                                    filteredApps.forEach { app ->
                                        Row(
                                            modifier = Modifier
                                                .fillMaxWidth()
                                                .clip(RoundedCornerShape(8.dp))
                                                .background(ArgusSurfaceVariant)
                                                .clickable { viewModel.toggleAppExclusion(app.packageName) }
                                                .padding(horizontal = 10.dp, vertical = 6.dp),
                                            horizontalArrangement = Arrangement.SpaceBetween,
                                            verticalAlignment = Alignment.CenterVertically
                                        ) {
                                            Column(modifier = Modifier.weight(1f)) {
                                                Text(
                                                    text = app.appName,
                                                    style = MaterialTheme.typography.titleMedium.copy(fontSize = 12.sp),
                                                    color = ArgusTextPrimary,
                                                    fontWeight = if (app.isExcluded) FontWeight.Bold else FontWeight.Normal
                                                )
                                                Text(
                                                    text = app.packageName,
                                                    style = MaterialTheme.typography.labelSmall,
                                                    color = if (app.isExcluded) ArgusCyanGlow else ArgusTextMuted,
                                                    fontSize = 10.sp
                                                )
                                            }

                                            Checkbox(
                                                checked = app.isExcluded,
                                                onCheckedChange = { viewModel.toggleAppExclusion(app.packageName) },
                                                colors = CheckboxDefaults.colors(
                                                    checkedColor = ArgusCyanPrimary,
                                                    uncheckedColor = ArgusCardBorder,
                                                    checkmarkColor = ArgusDarkBg
                                                )
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 4. World Model Episodic Memory Card (Collapsible)
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
                            .clickable { isWorldModelExpanded = !isWorldModelExpanded }
                            .padding(vertical = 2.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "World Model Memory",
                            style = MaterialTheme.typography.titleMedium,
                            color = ArgusTextPrimary,
                            fontWeight = FontWeight.Bold
                        )
                        Icon(
                            painter = if (isWorldModelExpanded) painterResource(R.drawable.ic_symbol_keyboard_arrow_up) else painterResource(R.drawable.ic_symbol_keyboard_arrow_down),
                            contentDescription = "Toggle",
                            tint = ArgusCyanPrimary,
                            modifier = Modifier.size(20.dp)
                        )
                    }

                    AnimatedVisibility(
                        visible = isWorldModelExpanded,
                        enter = expandVertically() + fadeIn(),
                        exit = shrinkVertically() + fadeOut()
                    ) {
                        Column {
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(
                                text = "Locally stored key-value facts, source labels, and context records.",
                                style = MaterialTheme.typography.labelSmall,
                                color = ArgusTextSecondary
                            )

                            Spacer(modifier = Modifier.height(10.dp))

                            Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                worldEntities.take(6).forEach { entity ->
                                    Box(
                                        modifier = Modifier
                                            .fillMaxWidth()
                                            .clip(RoundedCornerShape(8.dp))
                                            .background(ArgusSurfaceVariant)
                                            .padding(8.dp)
                                    ) {
                                        Column {
                                            Row(
                                                modifier = Modifier.fillMaxWidth(),
                                                horizontalArrangement = Arrangement.SpaceBetween
                                            ) {
                                                Text(
                                                    text = entity.key,
                                                    style = MaterialTheme.typography.labelSmall,
                                                    color = ArgusCyanPrimary,
                                                    fontWeight = FontWeight.Bold,
                                                    fontFamily = FontFamily.Monospace
                                                )
                                                Text(
                                                    text = entity.source,
                                                    style = MaterialTheme.typography.labelSmall,
                                                    color = ArgusTextMuted
                                                )
                                            }
                                            Spacer(modifier = Modifier.height(2.dp))
                                            Text(
                                                text = entity.value,
                                                style = MaterialTheme.typography.bodyMedium.copy(fontSize = 11.sp),
                                                color = ArgusTextPrimary
                                            )
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
fun SkillBadge(title: String, desc: String) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp)
    ) {
        Text(
            text = title,
            style = MaterialTheme.typography.labelSmall,
            color = ArgusTextPrimary,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = desc,
            style = MaterialTheme.typography.bodyMedium.copy(fontSize = 11.sp),
            color = ArgusTextMuted
        )
    }
}
