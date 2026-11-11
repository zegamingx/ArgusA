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
    var isSkillsExpanded by remember { mutableStateOf(false) }
    var isTesterExpanded by remember { mutableStateOf(false) }
    var isExcludeAppsExpanded by remember { mutableStateOf(false) }
    var isWorldModelExpanded by remember { mutableStateOf(false) }

    var searchQuery by remember { mutableStateOf("") }
    var allowArticleTraining by remember { mutableStateOf(false) }
    LaunchedEffect(articlePreview?.canonicalUrl) { allowArticleTraining = false }
    var actionClickQuery by remember { mutableStateOf("") }
    var actionClickResult by remember { mutableStateOf<String?>(null) }
    var appFilterText by remember { mutableStateOf("") }
    var appFilterTab by remember { mutableStateOf(0) }

    val pm = context.getSystemService(Context.POWER_SERVICE) as? PowerManager
    val isIgnoringOpt = pm?.isIgnoringBatteryOptimizations(context.packageName) ?: false
    val excludedCount = installedApps.count { it.isExcluded }

    val sensitiveKeywords = remember {
        listOf("bank", "bca", "bri", "【entity-mandiri¦canonical_name=mandiri】", "bni", "【entity-cimb¦canonical_name=cimb】", "dana", "ovo", "gopay", "shopeepay", "wallet", "crypto", "【entity-binance¦canonical_name=binance】", "【entity-indodax¦canonical_name=indodax】", "auth", "authenticator", "password", "bitwarden", "1password", "keepass", "whatsapp", "telegram", "signal")
    }
    val filteredApps = installedApps.filter { app ->
        val matchesText = appFilterText.isBlank() || app.appName.contains(appFilterText, ignoreCase = true) || app.packageName.contains(appFilterText, ignoreCase = true)
        val matchesTab = when (appFilterTab) {
            1 -> app.isExcluded
            2 -> sensitiveKeywords.any { app.packageName.lowercase().contains(it) }
            else -> true
        }
        matchesText && matchesTab
    }

    LazyColumn(
        modifier = Modifier.fillMaxSize().background(ArgusDarkBg).imePadding().padding(horizontal = 12.dp),
        contentPadding = PaddingValues(vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        // HEADER COMPACT + SUMMARY
        item {
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(text = "SKILLS & PRIVACY", style = MaterialTheme.typography.titleLarge.copy(fontSize = 20.sp), color = ArgusTextPrimary, fontWeight = FontWeight.Bold, letterSpacing = 1.sp, modifier = Modifier.weight(1f))
                    Box(modifier = Modifier.clip(RoundedCornerShape(20.dp)).background(if (isIgnoringOpt) ArgusSuccess.copy(alpha = 0.15f) else ArgusWarning.copy(alpha = 0.15f)).padding(horizontal = 9.dp, vertical = 4.dp)) {
                        Text(text = if (isIgnoringOpt) "BATTERY OK" else "BATTERY LIMITED", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp), color = if (isIgnoringOpt) ArgusSuccess else ArgusWarning, fontWeight = FontWeight.Bold)
                    }
                }
                Spacer(modifier = Modifier.height(4.dp))
                Text(text = "Atur apa yang boleh ARGUS baca, klik, dan latih. Semua diproses lokal, kecuali web search.", style = MaterialTheme.typography.bodySmall.copy(fontSize = 12.sp), color = ArgusTextMuted, lineHeight = 17.sp)
                Spacer(modifier = Modifier.height(10.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    MiniStat(label = "SKILLS", value = "7 aktif", modifier = Modifier.weight(1f))
                    MiniStat(label = "EXCLUDED", value = "$excludedCount app", modifier = Modifier.weight(1f))
                    MiniStat(label = "MEMORY", value = "${worldEntities.size} fakta", modifier = Modifier.weight(1f))
                }
            }
        }

        // 1. BATTERY
        item {
            Surface(shape = RoundedCornerShape(18.dp), color = ArgusSurface, border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder), modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Row(modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).clickable { isBatteryExpanded = !isBatteryExpanded }.padding(vertical = 2.dp), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(modifier = Modifier.size(34.dp).clip(CircleShape).background(if (isIgnoringOpt) ArgusSuccess.copy(alpha = 0.15f) else ArgusWarning.copy(alpha = 0.15f)), contentAlignment = Alignment.Center) {
                                Icon(painter = if (isIgnoringOpt) painterResource(R.drawable.ic_symbol_battery_charging_full) else painterResource(R.drawable.ic_symbol_battery_alert), contentDescription = "Battery", tint = if (isIgnoringOpt) ArgusSuccess else ArgusWarning, modifier = Modifier.size(18.dp))
                            }
                            Spacer(modifier = Modifier.width(9.dp))
                            Column {
                                Text(text = "Battery & Background", style = MaterialTheme.typography.titleMedium.copy(fontSize = 15.sp), color = ArgusTextPrimary, fontWeight = FontWeight.Bold)
                                Text(text = if (isIgnoringOpt) "Background diizinkan" else "Background dibatasi sistem", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp), color = ArgusTextMuted)
                            }
                        }
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(modifier = Modifier.clip(RoundedCornerShape(20.dp)).background(if (isIgnoringOpt) ArgusSuccess.copy(alpha = 0.15f) else ArgusWarning.copy(alpha = 0.15f)).padding(horizontal = 8.dp, vertical = 4.dp)) {
                                Text(text = if (isIgnoringOpt) "EXEMPT" else "RESTRICTED", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp), color = if (isIgnoringOpt) ArgusSuccess else ArgusWarning, fontWeight = FontWeight.Bold)
                            }
                            Icon(painter = if (isBatteryExpanded) painterResource(R.drawable.ic_symbol_keyboard_arrow_up) else painterResource(R.drawable.ic_symbol_keyboard_arrow_down), contentDescription = "Toggle", tint = ArgusCyanPrimary, modifier = Modifier.size(20.dp))
                        }
                    }
                    AnimatedVisibility(visible = isBatteryExpanded, enter = expandVertically() + fadeIn(), exit = shrinkVertically() + fadeOut()) {
                        Column {
                            Spacer(modifier = Modifier.height(10.dp))
                            Text(text = "Exempt ARGUS dari optimasi baterai biar training dan persepsi background tetap jalan. Android tetap bisa ngebatesin di kondisi tertentu.", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp), color = ArgusTextSecondary, lineHeight = 16.sp)
                            if (!isIgnoringOpt) {
                                Spacer(modifier = Modifier.height(12.dp))
                                Button(onClick = {
                                    try {
                                        val intent = Intent(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS).apply { data = Uri.parse("package:${context.packageName}") }
                                        context.startActivity(intent)
                                    } catch (_: Exception) {
                                        context.startActivity(Intent(Settings.ACTION_IGNORE_BATTERY_OPTIMIZATION_SETTINGS))
                                    }
                                }, colors = ButtonDefaults.buttonColors(containerColor = ArgusCyanPrimary, contentColor = ArgusDarkBg), shape = RoundedCornerShape(20.dp), modifier = Modifier.fillMaxWidth()) {
                                    Text(text = "Buka Pengaturan Baterai", fontWeight = FontWeight.Bold, fontSize = 12.sp)
                                }
                            }
                        }
                    }
                }
            }
        }

        // 2. SKILLS
        item {
            Surface(shape = RoundedCornerShape(18.dp), color = ArgusSurface, border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder), modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Row(modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).clickable { isSkillsExpanded = !isSkillsExpanded }.padding(vertical = 2.dp), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(modifier = Modifier.size(34.dp).clip(CircleShape).background(ArgusCyanPrimary.copy(alpha = 0.15f)), contentAlignment = Alignment.Center) {
                                Icon(painter = painterResource(R.drawable.ic_symbol_extension), contentDescription = "Skills", tint = ArgusCyanPrimary, modifier = Modifier.size(18.dp))
                            }
                            Spacer(modifier = Modifier.width(9.dp))
                            Column {
                                Text(text = "Core Capabilities", style = MaterialTheme.typography.titleMedium.copy(fontSize = 15.sp), color = ArgusTextPrimary, fontWeight = FontWeight.Bold)
                                Text(text = "7 skill • semua lokal kecuali web", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp), color = ArgusTextMuted)
                            }
                        }
                        Icon(painter = if (isSkillsExpanded) painterResource(R.drawable.ic_symbol_keyboard_arrow_up) else painterResource(R.drawable.ic_symbol_keyboard_arrow_down), contentDescription = "Toggle", tint = ArgusCyanPrimary, modifier = Modifier.size(20.dp))
                    }

                    AnimatedVisibility(visible = isSkillsExpanded, enter = expandVertically() + fadeIn(), exit = shrinkVertically() + fadeOut()) {
                        Column {
                            Spacer(modifier = Modifier.height(8.dp))
                            SkillRow(number = "01", title = "General Web Search", desc = "DuckDuckGo, 【entity-Wikipedia¦canonical_name=Wikipedia】 buat nambal hasil pendek.")
                            SkillRow(number = "02", title = "Article Extraction", desc = "Ambil teks artikel yang lu pilih, buat direview dulu.")
                            SkillRow(number = "03", title = "Screen Text Perception", desc = "Baca teks app aktif, app yang di-exclude di-skip.")
                            SkillRow(number = "04", title = "Accessibility Actions", desc = "Coba klik node teks yang keliatan di layar.")
                            SkillRow(number = "05", title = "Deterministic Arithmetic", desc = "Hitung ekspresi matematika secara deterministik.")
                            SkillRow(number = "06", title = "World Model Memory", desc = "Simpan fakta key-value + sumber + konteks app.")
                            SkillRow(number = "07", title = "Local Model Training", desc = "Update weights dan LoRA langsung di HP.")

                            Spacer(modifier = Modifier.height(10.dp))
                            // TESTER - pisah, default ketutup biar gak berat
                            Row(modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(ArgusSurfaceVariant).clickable { isTesterExpanded = !isTesterExpanded }.padding(horizontal = 10.dp, vertical = 9.dp), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                                Text(text = "Test Skills • Web & Tap", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp), color = ArgusCyanGlow, fontWeight = FontWeight.Bold)
                                Icon(painter = if (isTesterExpanded) painterResource(R.drawable.ic_symbol_keyboard_arrow_up) else painterResource(R.drawable.ic_symbol_keyboard_arrow_down), contentDescription = null, tint = ArgusCyanPrimary, modifier = Modifier.size(18.dp))
                            }
                            AnimatedVisibility(visible = isTesterExpanded, enter = expandVertically() + fadeIn(), exit = shrinkVertically() + fadeOut()) {
                                Column {
                                    Spacer(modifier = Modifier.height(12.dp))
                                    Text(text = "Test Web Search", style = MaterialTheme.typography.labelSmall, color = ArgusCyanGlow, fontWeight = FontWeight.Bold)
                                    Spacer(modifier = Modifier.height(6.dp))
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        OutlinedTextField(value = searchQuery, onValueChange = { searchQuery = it }, placeholder = { Text("Search query (e.g. on-device AI)", color = ArgusTextMuted, fontSize = 12.sp) }, modifier = Modifier.weight(1f), shape = RoundedCornerShape(12.dp), colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = ArgusCyanPrimary, unfocusedBorderColor = ArgusCardBorder, focusedTextColor = ArgusTextPrimary, unfocusedTextColor = ArgusTextPrimary, focusedContainerColor = ArgusSurfaceVariant, unfocusedContainerColor = ArgusSurfaceVariant), singleLine = true)
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Button(onClick = { if (searchQuery.isNotBlank()) viewModel.searchWebDirectly(searchQuery) }, colors = ButtonDefaults.buttonColors(containerColor = ArgusCyanPrimary, contentColor = ArgusDarkBg), shape = RoundedCornerShape(12.dp)) {
                                            Text("Search", fontWeight = FontWeight.Bold, fontSize = 12.sp)
                                        }
                                    }
                                    if (searchResults.isNotEmpty()) {
                                        Spacer(modifier = Modifier.height(8.dp))
                                        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                                            searchResults.take(5).forEach { item ->
                                                Surface(shape = RoundedCornerShape(12.dp), color = ArgusDarkBg, modifier = Modifier.fillMaxWidth()) {
                                                    Column(modifier = Modifier.padding(10.dp)) {
                                                        Text(text = item.title, style = MaterialTheme.typography.labelLarge, color = ArgusCyanPrimary, fontWeight = FontWeight.Bold)
                                                        Spacer(modifier = Modifier.height(4.dp))
                                                        Text(text = item.snippet, style = MaterialTheme.typography.bodySmall, color = ArgusTextSecondary)
                                                        Spacer(modifier = Modifier.height(4.dp))
                                                        Text(text = item.sourceUrl, style = MaterialTheme.typography.labelSmall, color = ArgusTextMuted)
                                                        Text(text = "Provider: ${item.provider}", style = MaterialTheme.typography.labelSmall, color = ArgusTextMuted)
                                                        Spacer(modifier = Modifier.height(6.dp))
                                                        Button(onClick = { viewModel.extractWebArticle(item) }, enabled = !isExtractingArticle, colors = ButtonDefaults.buttonColors(containerColor = ArgusCyanPrimary, contentColor = ArgusDarkBg), shape = RoundedCornerShape(10.dp)) {
                                                            Text("Extract article", fontWeight = FontWeight.SemiBold, fontSize = 11.sp)
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    if (isExtractingArticle) Text(text = "Fetching and extracting page content…", style = MaterialTheme.typography.bodySmall, color = ArgusCyanGlow)
                                    articleExtractionError?.let { Text(text = "Extraction failed: $it", style = MaterialTheme.typography.bodySmall, color = ArgusWarning) }
                                    articlePreview?.let { article ->
                                        Spacer(modifier = Modifier.height(8.dp))
                                        Surface(shape = RoundedCornerShape(12.dp), color = ArgusSurfaceVariant, border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder), modifier = Modifier.fillMaxWidth()) {
                                            Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                                Text(text = "ARTICLE PREVIEW • NOT QUEUED", style = MaterialTheme.typography.labelSmall, color = ArgusCyanGlow, fontWeight = FontWeight.Bold)
                                                Text(text = article.title, style = MaterialTheme.typography.titleSmall, color = ArgusTextPrimary, fontWeight = FontWeight.Bold)
                                                Text(text = "${article.siteName} • ${article.domain} • ${article.characterCount} chars", style = MaterialTheme.typography.labelSmall, color = ArgusTextMuted)
                                                article.publishedAt?.let { Text(text = "Source date: $it", style = MaterialTheme.typography.labelSmall, color = ArgusTextMuted) }
                                                Text(text = article.canonicalUrl, style = MaterialTheme.typography.labelSmall, color = ArgusTextMuted)
                                                Text(text = article.text.take(2800), style = MaterialTheme.typography.bodySmall, color = ArgusTextSecondary)
                                                Text(text = "Review konten, tanggal, publisher, dan URL sebelum masuk training.", style = MaterialTheme.typography.bodySmall, color = ArgusTextPrimary)
                                                Row(verticalAlignment = Alignment.CenterVertically) {
                                                    Checkbox(checked = allowArticleTraining, onCheckedChange = { allowArticleTraining = it }, colors = CheckboxDefaults.colors(checkedColor = ArgusCyanPrimary, uncheckedColor = ArgusTextMuted, checkmarkColor = ArgusDarkBg))
                                                    Text(text = "Gue confirm punya hak / izin buat pake artikel ini sebagai data training.", style = MaterialTheme.typography.bodySmall, color = ArgusTextSecondary)
                                                }
                                                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                                    Button(onClick = { viewModel.enqueueReviewedWebArticle() }, enabled = allowArticleTraining && !isQueueingArticle, colors = ButtonDefaults.buttonColors(containerColor = ArgusSuccess, contentColor = ArgusDarkBg, disabledContainerColor = ArgusSurface, disabledContentColor = ArgusTextMuted), shape = RoundedCornerShape(10.dp)) {
                                                        Text(if (isQueueingArticle) "Adding…" else "Add to training", fontSize = 11.sp)
                                                    }
                                                    Button(onClick = { viewModel.dismissWebArticlePreview() }, enabled = !isQueueingArticle, colors = ButtonDefaults.buttonColors(containerColor = ArgusSurface, contentColor = ArgusTextPrimary), shape = RoundedCornerShape(10.dp)) {
                                                        Text("Close", fontSize = 11.sp)
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    articleQueueStatus?.let { Text(text = it, style = MaterialTheme.typography.bodySmall, color = ArgusTextPrimary) }
                                    Spacer(modifier = Modifier.height(14.dp))
                                    Text(text = "Test Accessibility Action", style = MaterialTheme.typography.labelSmall, color = ArgusCyanGlow, fontWeight = FontWeight.Bold)
                                    Spacer(modifier = Modifier.height(6.dp))
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        OutlinedTextField(value = actionClickQuery, onValueChange = { actionClickQuery = it }, placeholder = { Text("Visible button text", color = ArgusTextMuted, fontSize = 12.sp) }, modifier = Modifier.weight(1f), shape = RoundedCornerShape(12.dp), colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = ArgusCyanPrimary, unfocusedBorderColor = ArgusCardBorder, focusedTextColor = ArgusTextPrimary, unfocusedTextColor = ArgusTextPrimary, focusedContainerColor = ArgusSurfaceVariant, unfocusedContainerColor = ArgusSurfaceVariant), singleLine = true)
                                        Spacer(modifier = Modifier.width(8.dp))
                                        Button(onClick = {
                                            if (actionClickQuery.isNotBlank()) {
                                                val success = viewModel.executeActionClick(actionClickQuery)
                                                actionClickResult = if (success) "Success: click sent to '$actionClickQuery'." else "Failed: '$actionClickQuery' gak ketemu di jendela aktif."
                                            }
                                        }, colors = ButtonDefaults.buttonColors(containerColor = ArgusBlueAccent, contentColor = ArgusDarkBg), shape = RoundedCornerShape(12.dp)) {
                                            Text("Tap", fontWeight = FontWeight.Bold, fontSize = 12.sp)
                                        }
                                    }
                                    actionClickResult?.let { result ->
                                        Spacer(modifier = Modifier.height(6.dp))
                                        Text(text = result, style = MaterialTheme.typography.labelSmall, color = if (result.startsWith("Success")) ArgusSuccess else ArgusWarning, fontFamily = FontFamily.Monospace)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 3. EXCLUDED APPS - FIX TABS + ROW
        item {
            Surface(shape = RoundedCornerShape(18.dp), color = ArgusSurface, border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder), modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Row(modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).clickable { isExcludeAppsExpanded = !isExcludeAppsExpanded }.padding(vertical = 2.dp), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(modifier = Modifier.size(34.dp).clip(CircleShape).background(ArgusCyanPrimary.copy(alpha = 0.15f)), contentAlignment = Alignment.Center) {
                                Icon(painter = painterResource(R.drawable.ic_symbol_visibility_off), contentDescription = "Exclude", tint = ArgusCyanPrimary, modifier = Modifier.size(18.dp))
                            }
                            Spacer(modifier = Modifier.width(9.dp))
                            Column {
                                Text(text = "Excluded Apps", style = MaterialTheme.typography.titleMedium.copy(fontSize = 15.sp), color = ArgusTextPrimary, fontWeight = FontWeight.Bold)
                                Text(text = "$excludedCount app di-skip dari screen reading", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp), color = ArgusTextMuted)
                            }
                        }
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(modifier = Modifier.clip(RoundedCornerShape(20.dp)).background(ArgusCyanPrimary.copy(alpha = 0.15f)).padding(horizontal = 8.dp, vertical = 4.dp)) {
                                Text(text = "$excludedCount excluded", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp), color = ArgusCyanPrimary, fontWeight = FontWeight.Bold)
                            }
                            Icon(painter = if (isExcludeAppsExpanded) painterResource(R.drawable.ic_symbol_keyboard_arrow_up) else painterResource(R.drawable.ic_symbol_keyboard_arrow_down), contentDescription = "Toggle", tint = ArgusCyanPrimary, modifier = Modifier.size(20.dp))
                        }
                    }
                    AnimatedVisibility(visible = isExcludeAppsExpanded, enter = expandVertically() + fadeIn(), exit = shrinkVertically() + fadeOut()) {
                        Column {
                            Spacer(modifier = Modifier.height(8.dp))
                            Text(text = "Teks dari app yang lu pilih gak bakal dibaca sama Accessibility. Ini ngurangin exposure, tapi bukan pengganti permission Android.", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp), color = ArgusTextSecondary, lineHeight = 16.sp)
                            Spacer(modifier = Modifier.height(10.dp))
                            OutlinedTextField(value = appFilterText, onValueChange = { appFilterText = it }, placeholder = { Text("Cari app...", color = ArgusTextMuted, fontSize = 12.sp) }, modifier = Modifier.fillMaxWidth(), shape = RoundedCornerShape(12.dp), colors = OutlinedTextFieldDefaults.colors(focusedBorderColor = ArgusCyanPrimary, unfocusedBorderColor = ArgusCardBorder, focusedTextColor = ArgusTextPrimary, unfocusedTextColor = ArgusTextPrimary, focusedContainerColor = ArgusSurfaceVariant, unfocusedContainerColor = ArgusSurfaceVariant), singleLine = true)
                            Spacer(modifier = Modifier.height(8.dp))
                            // FIX: LazyRow biar tab gak kepotong
                            LazyRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                                val tabs = listOf("All (${installedApps.size})", "Excluded ($excludedCount)", "Privacy suggestions (${installedApps.count { item -> sensitiveKeywords.any { item.packageName.lowercase().contains(it) } }})")
                                items(tabs.indices.toList()) { idx ->
                                    val label = tabs[idx]
                                    val isSelected = appFilterTab == idx
                                    Box(modifier = Modifier.clip(RoundedCornerShape(20.dp)).background(if (isSelected) ArgusCyanPrimary else ArgusSurfaceVariant).clickable { appFilterTab = idx }.padding(horizontal = 10.dp, vertical = 6.dp)) {
                                        Text(text = label, style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp), color = if (isSelected) ArgusDarkBg else ArgusTextSecondary, fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal)
                                    }
                                }
                            }
                            Spacer(modifier = Modifier.height(10.dp))
                            Text(text = "Showing ${filteredApps.size} apps • scroll di dalam kotak", style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp), color = ArgusCyanGlow, fontFamily = FontFamily.Monospace)
                            Spacer(modifier = Modifier.height(6.dp))
                            Column(modifier = Modifier.fillMaxWidth().heightIn(max = 340.dp).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                if (filteredApps.isEmpty()) {
                                    Box(modifier = Modifier.fillMaxWidth().padding(20.dp), contentAlignment = Alignment.Center) {
                                        Text(text = "Gak ada app yang cocok.", style = MaterialTheme.typography.bodyMedium.copy(fontSize = 11.sp), color = ArgusTextMuted)
                                    }
                                } else {
                                    filteredApps.forEach { app ->
                                        Row(modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(ArgusSurfaceVariant).clickable { viewModel.toggleAppExclusion(app.packageName) }.padding(horizontal = 10.dp, vertical = 7.dp), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                                            Column(modifier = Modifier.weight(1f).padding(end = 8.dp)) {
                                                Text(text = app.appName, style = MaterialTheme.typography.titleMedium.copy(fontSize = 12.sp), color = ArgusTextPrimary, fontWeight = if (app.isExcluded) FontWeight.Bold else FontWeight.Normal)
                                                Text(text = app.packageName, style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp), color = if (app.isExcluded) ArgusCyanGlow else ArgusTextMuted)
                                            }
                                            Checkbox(checked = app.isExcluded, onCheckedChange = { viewModel.toggleAppExclusion(app.packageName) }, colors = CheckboxDefaults.colors(checkedColor = ArgusCyanPrimary, uncheckedColor = ArgusCardBorder, checkmarkColor = ArgusDarkBg))
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 4. WORLD MODEL
        item {
            Surface(shape = RoundedCornerShape(18.dp), color = ArgusSurface, border = androidx.compose.foundation.BorderStroke(1.dp, ArgusCardBorder), modifier = Modifier.fillMaxWidth()) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Row(modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(8.dp)).clickable { isWorldModelExpanded = !isWorldModelExpanded }.padding(vertical = 2.dp), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Box(modifier = Modifier.size(34.dp).clip(CircleShape).background(ArgusCyanPrimary.copy(alpha = 0.15f)), contentAlignment = Alignment.Center) {
                                Icon(painter = painterResource(R.drawable.ic_symbol_extension), contentDescription = null, tint = ArgusCyanPrimary, modifier = Modifier.size(18.dp))
                            }
                            Spacer(modifier = Modifier.width(9.dp))
                            Column {
                                Text(text = "World Model Memory", style = MaterialTheme.typography.titleMedium.copy(fontSize = 15.sp), color = ArgusTextPrimary, fontWeight = FontWeight.Bold)
                                Text(text = "${worldEntities.size} fakta tersimpan lokal", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp), color = ArgusTextMuted)
                            }
                        }
                        Icon(painter = if (isWorldModelExpanded) painterResource(R.drawable.ic_symbol_keyboard_arrow_up) else painterResource(R.drawable.ic_symbol_keyboard_arrow_down), contentDescription = "Toggle", tint = ArgusCyanPrimary, modifier = Modifier.size(20.dp))
                    }
                    AnimatedVisibility(visible = isWorldModelExpanded, enter = expandVertically() + fadeIn(), exit = shrinkVertically() + fadeOut()) {
                        Column {
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(text = "Fakta key-value, label sumber, dan konteks app, semua lokal.", style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp), color = ArgusTextSecondary)
                            Spacer(modifier = Modifier.height(10.dp))
                            if (worldEntities.isEmpty()) {
                                Box(modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(ArgusSurfaceVariant).padding(14.dp), contentAlignment = Alignment.Center) {
                                    Text(text = "Belum ada fakta tersimpan.", style = MaterialTheme.typography.bodySmall.copy(fontSize = 11.sp), color = ArgusTextMuted)
                                }
                            } else {
                                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                    worldEntities.take(6).forEach { entity ->
                                        Box(modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(10.dp)).background(ArgusSurfaceVariant).padding(10.dp)) {
                                            Column {
                                                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                                    Text(text = entity.key, style = MaterialTheme.typography.labelSmall, color = ArgusCyanPrimary, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace)
                                                    Text(text = entity.source, style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp), color = ArgusTextMuted)
                                                }
                                                Spacer(modifier = Modifier.height(2.dp))
                                                Text(text = entity.value, style = MaterialTheme.typography.bodyMedium.copy(fontSize = 11.sp), color = ArgusTextPrimary)
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
}

@Composable
fun MiniStat(label: String, value: String, modifier: Modifier = Modifier) {
    Column(modifier = modifier.clip(RoundedCornerShape(12.dp)).background(ArgusSurfaceVariant).border(1.dp, ArgusCardBorder, RoundedCornerShape(12.dp)).padding(horizontal = 10.dp, vertical = 8.dp)) {
        Text(text = label, style = MaterialTheme.typography.labelSmall.copy(fontSize = 9.sp), color = ArgusTextMuted, fontWeight = FontWeight.Bold)
        Spacer(modifier = Modifier.height(2.dp))
        Text(text = value, style = MaterialTheme.typography.labelSmall.copy(fontSize = 12.sp), color = ArgusTextPrimary, fontWeight = FontWeight.Bold, fontFamily = FontFamily.Monospace)
    }
}

@Composable
fun SkillRow(number: String, title: String, desc: String) {
    Row(modifier = Modifier.fillMaxWidth().padding(vertical = 5.dp), verticalAlignment = Alignment.Top) {
        Text(text = number, style = MaterialTheme.typography.labelSmall.copy(fontSize = 11.sp), color = ArgusCyanPrimary, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Bold, modifier = Modifier.width(28.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(text = title, style = MaterialTheme.typography.labelSmall.copy(fontSize = 13.sp), color = ArgusTextPrimary, fontWeight = FontWeight.Bold)
            Text(text = desc, style = MaterialTheme.typography.bodyMedium.copy(fontSize = 11.sp), color = ArgusTextMuted, lineHeight = 15.sp)
        }
    }
}
