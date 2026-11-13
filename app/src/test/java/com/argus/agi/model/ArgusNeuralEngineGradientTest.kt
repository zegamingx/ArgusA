package com.argus.agi.model

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import java.io.File
import kotlin.math.abs
import kotlin.math.max
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ArgusNeuralEngineGradientTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun analyticGradientsMatchCentralFiniteDifferencesAcrossTrainableParameters() {
        val checkpoint = File(context.filesDir, "argus_model_weights.bin")
        val previousCheckpoint = checkpoint.takeIf(File::isFile)?.readBytes()

        try {
            // Start from fresh random weights so a stale checkpoint cannot change the
            // architecture or make this numerical test depend on another test.
            checkpoint.delete()

            val engine = ArgusNeuralEngine.createBlocking(
                context = context,
                config = EngineConfig(
                    dModel = 16,
                    nHeads = 4,
                    dFf = 32,
                    maxSeqLen = 32,
                    loraRank = 2,
                    isFullTrainingEnabled = true
                )
            )
            val parameters = listOf(
                "wHead",
                "wEmbed",
                "wDown",
                "wUp",
                "wGate",
                "wO",
                "wQ",
                "wK",
                "wV",
                "loraQa",
                "loraQb",
                "loraVa",
                "loraVb"
            )

            for (parameterName in parameters) {
                val result = engine.checkGradient(
                    text = "argus model training",
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
                val diagnostic = buildString {
                    append("Gradient mismatch for ")
                    append(parameterName)
                    append(" at index ")
                    append(result.parameterIndex)
                    append(": analytic=")
                    append(result.analyticalGradient)
                    append(", numerical=")
                    append(result.numericalGradient)
                    append(", absoluteError=")
                    append(result.absoluteError)
                    append(", relativeError=")
                    append(result.relativeError)
                }

                assertTrue(diagnostic, withinTolerance)
            }
        } finally {
            if (previousCheckpoint == null) {
                checkpoint.delete()
            } else {
                checkpoint.writeBytes(previousCheckpoint)
            }
        }
    }
}
