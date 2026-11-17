package com.argus.agi.ui.screens

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import com.argus.agi.model.GenesisLifecycleStage
import com.argus.agi.ui.ArgusViewModel
import com.argus.agi.ui.components.ArgusModernLoadingScreen
import kotlinx.coroutines.delay
import kotlin.math.roundToInt

@Composable
fun GenesisBootScreen(
    viewModel: ArgusViewModel,
    onFinish: () -> Unit
) {
    val genesisState by viewModel.genesisState.collectAsState()

    val isFailed = genesisState.stage == GenesisLifecycleStage.FAILED
    val isReady = !isFailed && (
        genesisState.stage == GenesisLifecycleStage.APPLICATION_READY ||
            genesisState.stage == GenesisLifecycleStage.GENERAL_LEARNING_SCHEDULED ||
            genesisState.isComplete
        )

    LaunchedEffect(genesisState.stage) {
        if (genesisState.stage == GenesisLifecycleStage.IDLE) {
            viewModel.retryGenesis()
        }
    }

    LaunchedEffect(isReady) {
        if (isReady) {
            delay(1_200)
            viewModel.markGenesisBootSeen()
            onFinish()
        }
    }

    val measuredProgress = if (genesisState.progress.isFinite()) {
        genesisState.progress.coerceIn(0f, 1f)
    } else {
        0f
    }
    val progressPercent = if (isReady) 100 else (measuredProgress * 100f).roundToInt()
    val subtitle = when {
        isFailed -> "GENESIS INTERRUPTED"
        isReady -> "ARGUS GENESIS READY"
        else -> genesisState.stage.displayName.uppercase() + " · " + progressPercent + "%"
    }
    val supportingText = if (isFailed) {
        genesisState.errorMessage
            ?: "Genesis could not complete. Check available memory, storage, and device safety state."
    } else {
        listOf(genesisState.stage.description, genesisState.assessedDetails)
            .filter(String::isNotBlank)
            .joinToString(" · ")
    }

    ArgusModernLoadingScreen(
        title = "A . R . G . U . S",
        subtitle = subtitle,
        progressPercent = progressPercent,
        supportingText = supportingText,
        driverInfo = "STANDALONE ON-DEVICE ARCHITECTURE",
        isError = isFailed,
        onRetry = if (isFailed) viewModel::retryGenesis else null
    )
}
