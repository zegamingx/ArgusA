package com.argus.agi

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.ResourcesMode

/**
 * Exercises Robolectric's native Android resource-table path on API 37.
 *
 * The regular test suite intentionally retains its API 34 coverage. This
 * focused smoke test verifies the API 36+ native resource implementation
 * introduced in Robolectric 4.18 without changing the resource mode globally.
 */
@RunWith(RobolectricTestRunner::class)
@Config(sdk = [37])
@ResourcesMode(ResourcesMode.Mode.NATIVE)
class RobolectricNativeResourcesApi37Test {

    @Test
    fun applicationAndPlatformResourcesLoadInNativeMode() {
        val context = ApplicationProvider.getApplicationContext<Context>()

        assertEquals("A.R.G.U.S", context.getString(R.string.app_name))
        assertNotNull(context.resources.getText(android.R.string.ok))
    }
}
