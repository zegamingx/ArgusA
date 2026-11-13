package com.argus.agi.model

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class WorldModelSensorStateTest {

    private val context: Context
        get() = ApplicationProvider.getApplicationContext()

    @Test
    fun cameraLumaIsClampedAndPublishedAsRelativeBrightness() {
        val worldModel = WorldModel(context)

        worldModel.updateCameraContext(
            lumaLevel = 300f,
            visualSummary = "Camera sample",
        )

        val cameraLuma = worldModel.entitiesFlow.value.firstOrNull {
            it.key == "camera_luma"
        }

        assertNotNull(cameraLuma)
        assertEquals("255.0/255 relative luminance", cameraLuma?.value)
        assertEquals("Camera Y-plane Sensor", cameraLuma?.source)
    }

    @Test
    fun transientSensorUpdatesReplaceThePreviousSample() {
        val worldModel = WorldModel(context)

        worldModel.updateTransientEntities(
            WorldEntity(
                key = "ambient_acoustic",
                value = "32.0 dB relative (uncalibrated)",
                source = "Microphone Acoustic Sensor",
            ),
        )
        worldModel.updateTransientEntities(
            WorldEntity(
                key = "ambient_acoustic",
                value = "28.0 dB relative (uncalibrated)",
                source = "Microphone Acoustic Sensor",
            ),
        )

        val acousticSamples = worldModel.entitiesFlow.value.filter {
            it.key == "ambient_acoustic"
        }

        assertEquals(1, acousticSamples.size)
        assertEquals("28.0 dB relative (uncalibrated)", acousticSamples.single().value)
        assertTrue(acousticSamples.single().timestamp > 0L)
    }
}
