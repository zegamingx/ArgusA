package com.argus.agi

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.ime
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.systemBars
import androidx.compose.material3.Icon
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.LifecycleEventEffect
import com.argus.agi.ui.ArgusViewModel
import com.argus.agi.ui.screens.ChatScreen
import com.argus.agi.ui.screens.GenesisBootScreen
import com.argus.agi.ui.screens.NeuralCoreScreen
import com.argus.agi.ui.screens.PerceptionScreen
import com.argus.agi.ui.screens.SkillsAndSettingsScreen
import com.argus.agi.ui.theme.ArgusCyanPrimary
import com.argus.agi.ui.theme.ArgusDarkBg
import com.argus.agi.ui.theme.ArgusSurface
import com.argus.agi.ui.theme.ArgusTextMuted
import com.argus.agi.ui.theme.ArgusTheme
import androidx.compose.ui.res.painterResource

enum class ArgusScreen {
    CHAT,
    NEURAL_CORE,
    PERCEPTION,
    SETTINGS
}

val ArgusScreen.label: String
    get() = when (this) {
        ArgusScreen.CHAT -> "Terminal"
        ArgusScreen.NEURAL_CORE -> "Neural Core"
        ArgusScreen.PERCEPTION -> "Persepsi"
        ArgusScreen.SETTINGS -> "Skill & Privasi"
    }

val ArgusScreen.icon: ImageVector
    get() = when (this) {
        ArgusScreen.CHAT -> painterResource(R.drawable.ic_symbol_chat)
        ArgusScreen.NEURAL_CORE -> painterResource(R.drawable.ic_symbol_memory)
        ArgusScreen.PERCEPTION -> painterResource(R.drawable.ic_symbol_visibility)
        ArgusScreen.SETTINGS -> painterResource(R.drawable.ic_symbol_tune)
    }

class MainActivity : ComponentActivity() {

    private val viewModel: ArgusViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            ArgusTheme {
                MainScreen(viewModel = viewModel)
            }
        }
    }
}

@Composable
fun MainScreen(viewModel: ArgusViewModel) {
    val context = LocalContext.current
    var showGenesisBoot by rememberSaveable { mutableStateOf(!viewModel.hasSeenGenesisBoot()) }
    var currentScreen by rememberSaveable { mutableStateOf(ArgusScreen.CHAT) }
    var hasRequestedNotificationPermission by rememberSaveable { mutableStateOf(false) }

    if (showGenesisBoot) {
        GenesisBootScreen(
            viewModel = viewModel,
            onFinish = { showGenesisBoot = false }
        )
        return
    }

    // Resilient notification permission launcher with result piped into ViewModel
    val notificationLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        viewModel.onNotificationPermissionResult(isGranted)
    }

    // Contextual one-time notification permission request for background training service
    LaunchedEffect(Unit) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU && !hasRequestedNotificationPermission) {
            val status = ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS)
            if (status == PackageManager.PERMISSION_GRANTED) {
                viewModel.onNotificationPermissionResult(true)
            } else {
                hasRequestedNotificationPermission = true
                notificationLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
            }
        }
    }

    // Lifecycle-aware permission re-evaluation on resume (e.g. after user changes settings)
    LifecycleEventEffect(Lifecycle.Event.ON_RESUME) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val status = ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS)
            viewModel.onNotificationPermissionResult(status == PackageManager.PERMISSION_GRANTED)
        }
    }

    val imeInsets = WindowInsets.ime
    val density = LocalDensity.current
    val isKeyboardOpen by remember(imeInsets, density) {
        derivedStateOf { imeInsets.getBottom(density) > 0 }
    }

    BackHandler(enabled = currentScreen != ArgusScreen.CHAT) {
        currentScreen = ArgusScreen.CHAT
    }

    Scaffold(
        modifier = Modifier
            .fillMaxSize()
            .testTag("main_screen_scaffold"),
        containerColor = ArgusDarkBg,
        contentWindowInsets = WindowInsets.systemBars,
        bottomBar = {
            // Smoothly slide NavigationBar away when keyboard is open to maximize chat/terminal viewport
            AnimatedVisibility(
                visible = !isKeyboardOpen,
                enter = slideInVertically(initialOffsetY = { it }),
                exit = slideOutVertically(targetOffsetY = { it })
            ) {
                NavigationBar(
                    containerColor = ArgusSurface,
                    tonalElevation = 6.dp,
                    windowInsets = WindowInsets(0, 0, 0, 0)
                ) {
                    ArgusScreen.entries.forEach { screen ->
                        val isSelected = currentScreen == screen
                        NavigationBarItem(
                            selected = isSelected,
                            onClick = { currentScreen = screen },
                            icon = {
                                Icon(
                                    painter = painterResource(screen.iconRes),
                                    contentDescription = screen.label
                                )
                            },
                            label = {
                                Text(
                                    text = screen.label,
                                    fontSize = 10.sp,
                                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal
                                )
                            },
                            colors = NavigationBarItemDefaults.colors(
                                selectedIconColor = ArgusDarkBg,
                                selectedTextColor = ArgusCyanPrimary,
                                indicatorColor = ArgusCyanPrimary,
                                unselectedIconColor = ArgusTextMuted,
                                unselectedTextColor = ArgusTextMuted
                            ),
                            modifier = Modifier.testTag("nav_tab_${screen.name.lowercase()}")
                        )
                    }
                }
            }
        }
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .testTag("main_content_container")
        ) {
            when (currentScreen) {
                ArgusScreen.CHAT -> ChatScreen(viewModel = viewModel)
                ArgusScreen.NEURAL_CORE -> NeuralCoreScreen(viewModel = viewModel)
                ArgusScreen.PERCEPTION -> PerceptionScreen(viewModel = viewModel)
                ArgusScreen.SETTINGS -> SkillsAndSettingsScreen(viewModel = viewModel)
            }
        }
    }
}
