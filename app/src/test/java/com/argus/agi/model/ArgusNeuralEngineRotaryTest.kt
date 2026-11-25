package com.argus.agi.model

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import java.io.File
import kotlin.math.abs
import kotlin.math.max
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ArgusNeuralEngineRotaryTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun engineArchitectureContractReportsRotaryAndNoPositionTable() {
        val checkpoint = File(context.filesDir, "argus_model_weights.bin")
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()
        try {
            checkpoint.delete()
            val engine = ArgusNeuralEngine.createBlocking(
                context = context,
                config = EngineConfig(
                    dModel = 16,
                    nHeads = 4,
                    dFf = 32,
                    maxSeqLen = 32,
                    loraRank = 2
                )
            )

            val architecture = engine.architectureConfig
            assertEquals("argus.transformer.swiglu.rope.v2", architecture.architectureId)
            assertEquals(ArgusPositionEncodingKind.ROTARY, architecture.positionEncodingKind)
            assertEquals(0L, engine.parameterBreakdown.fixedPositionBufferElements)
            assertEquals(
                engine.countTrainableParameters(),
                engine.countTotalParameters()
            )
            assertEquals(
                ArgusPositionEncodingKind.ROTARY,
                engine.config.positionEncoding
            )
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    @Test
    fun engineConfigRejectsTheRetiredSinusoidalEncoding() {
        val failure = runCatching {
            EngineConfig(
                dModel = 16,
                nHeads = 4,
                dFf = 32,
                maxSeqLen = 32,
                loraRank = 2,
                positionEncoding = ArgusPositionEncodingKind.FIXED_SINUSOIDAL_TABLE
            )
        }.exceptionOrNull()

        assertTrue(failure is IllegalArgumentException)
    }

    @Test
    fun queryAndKeyGradientsMatchFiniteDifferencesThroughTheRotation() {
        val checkpoint = File(context.filesDir, "argus_model_weights.bin")
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()
        try {
            // Fresh random weights, mirroring ArgusNeuralEngineGradientTest, so
            // the probe measures the rotated attention backward of this build.
            checkpoint.delete()
            val engine = ArgusNeuralEngine.createBlocking(
                context = context,
                config = EngineConfig(
                    dModel = 16,
                    nHeads = 4,
                    dFf = 32,
                    maxSeqLen = 64,
                    loraRank = 2,
                    isFullTrainingEnabled = true
                )
            )

            // The text spans many positions so the Q/K rotations differ per
            // token; a missing or wrong adjoint rotation in the backward pass
            // would push the analytic gradients away from the finite
            // differences for exactly these two matrices.
            for (parameterName in listOf("wQ", "wK")) {
                val result = engine.checkGradient(
                    text = "argus rotary position embedding gradient",
                    parameterName = parameterName,
                    parameterIndex = -1,
                    epsilon = 1e-2f
                )
                val gradientMagnitude = max(
                    abs(result.analyticalGradient),
                    abs(result.numericalGradient)
                )
                val withinTolerance = if (gradientMagnitude < 1e-3f) {
                    result.absoluteError <= 1e-4f
                } else {
                    result.relativeError <= 0.15f
                }
                if (!withinTolerance) {
                    fail(
                        "Rotated-attention gradient mismatch for $parameterName at index " +
                            "${result.parameterIndex}: analytic=${result.analyticalGradient}, " +
                            "numerical=${result.numericalGradient}, " +
                            "absoluteError=${result.absoluteError}, " +
                            "relativeError=${result.relativeError}"
                    )
                }
            }
        } finally {
            restoreCheckpoint(checkpoint, previousCheckpoint)
        }
    }

    private fun restoreCheckpoint(file: File, previous: ByteArray?) {
        if (previous == null) {
            file.delete()
        } else {
            file.writeBytes(previous)
        }
    }
}
