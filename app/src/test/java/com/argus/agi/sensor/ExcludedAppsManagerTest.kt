package com.argus.agi.sensor

import android.content.Context
import androidx.test.core.app.ApplicationProvider
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [34])
class ExcludedAppsManagerTest {

    private lateinit var context: Context
    private lateinit var manager: ExcludedAppsManager

    @Before
    fun setUp() {
        context = ApplicationProvider.getApplicationContext()
        context.getSharedPreferences("argus_excluded_apps", Context.MODE_PRIVATE)
            .edit()
            .clear()
            .commit()
        manager = ExcludedAppsManager(context)
    }

    @Test
    fun screenPerceptionAndTrainingAreSeparateDefaultOffConsents() {
        assertFalse(manager.isScreenPerceptionEnabled())
        assertFalse(manager.isScreenTrainingEnabled())

        manager.setScreenTrainingEnabled(true)
        assertFalse(manager.isScreenTrainingEnabled())

        manager.setScreenPerceptionEnabled(true)
        assertTrue(manager.isScreenPerceptionEnabled())
        assertFalse(manager.isScreenTrainingEnabled())

        manager.setScreenTrainingEnabled(true)
        assertTrue(manager.isScreenTrainingEnabled())
        assertFalse(manager.isAppTrainingAllowed("com.example.notes"))

        manager.toggleAppAccess("com.example.notes")
        assertFalse(manager.isAppTrainingAllowed("com.example.notes"))

        manager.toggleAppTrainingAccess("com.example.notes")
        assertTrue(manager.isAppTrainingAllowed("com.example.notes"))

        manager.setScreenPerceptionEnabled(false)
        assertFalse(manager.isScreenPerceptionEnabled())
        assertFalse(manager.isScreenTrainingEnabled())
        assertFalse(manager.isAppTrainingAllowed("com.example.notes"))
    }

    @Test
    fun accessibilityActionsRequireTheirOwnConsent() {
        assertFalse(manager.isAccessibilityActionsEnabled())

        manager.setScreenPerceptionEnabled(true)
        assertFalse(manager.isAccessibilityActionsEnabled())

        manager.setAccessibilityActionsEnabled(true)
        assertTrue(manager.isAccessibilityActionsEnabled())
    }

    @Test
    fun accessibilityActionsFailClosedForUnknownOrUnapprovedPackages() {
        assertFalse(manager.isAccessibilityActionAllowed(null))
        assertFalse(manager.isAccessibilityActionAllowed(""))
        assertFalse(manager.isAccessibilityActionAllowed("   "))

        manager.setAccessibilityActionsEnabled(true)
        assertFalse(manager.isAccessibilityActionAllowed("com.example.reader"))

        manager.toggleAppAccess("com.example.reader")
        assertTrue(manager.isAccessibilityActionAllowed("com.example.reader"))

        assertFalse(manager.isAccessibilityActionAllowed("com.whatsapp"))

        manager.setAccessibilityActionsEnabled(false)
        assertFalse(manager.isAccessibilityActionAllowed("com.example.reader"))
    }

    @Test
    fun appAccessIsDefaultDenyAndSensitiveAppsStayBlocked() {
        assertTrue(manager.isAppExcluded("com.example.notes"))

        manager.toggleAppAccess("com.example.notes")
        assertFalse(manager.isAppExcluded("com.example.notes"))

        manager.toggleAppAccess("com.example.notes")
        assertTrue(manager.isAppExcluded("com.example.notes"))

        manager.toggleAppAccess("com.whatsapp")
        assertTrue(manager.isAppExcluded("com.whatsapp"))

        context.getSharedPreferences("argus_excluded_apps", Context.MODE_PRIVATE)
            .edit()
            .putStringSet("allowed_screen_packages", setOf("com.whatsapp"))
            .commit()
        assertTrue(ExcludedAppsManager(context).isAppExcluded("com.whatsapp"))
    }

    @Test
    fun clearingScreenAppAllowlistRevokesReadAndTrainingForEveryApp() {
        manager.setScreenPerceptionEnabled(true)
        manager.setScreenTrainingEnabled(true)
        manager.toggleAppAccess("com.example.reader")
        manager.toggleAppTrainingAccess("com.example.reader")
        assertFalse(manager.isAppExcluded("com.example.reader"))
        assertTrue(manager.isAppTrainingAllowed("com.example.reader"))

        manager.clearScreenAppAllowlist()

        assertTrue(manager.isAppExcluded("com.example.reader"))
        assertFalse(manager.isAppTrainingAllowed("com.example.reader"))
        assertTrue(
            context.getSharedPreferences("argus_excluded_apps", Context.MODE_PRIVATE)
                .getStringSet("allowed_screen_packages", emptySet())
                .orEmpty()
                .isEmpty()
        )
        assertTrue(
            context.getSharedPreferences("argus_excluded_apps", Context.MODE_PRIVATE)
                .getStringSet("allowed_screen_training_packages", emptySet())
                .orEmpty()
                .isEmpty()
        )
    }

    @Test
    fun allowlistAndConsentsPersistAcrossManagerInstances() {
        manager.setScreenPerceptionEnabled(true)
        manager.setScreenTrainingEnabled(true)
        manager.setAccessibilityActionsEnabled(true)
        manager.toggleAppAccess("com.example.notes")
        manager.toggleAppTrainingAccess("com.example.notes")

        val restored = ExcludedAppsManager(context)
        assertTrue(restored.isScreenPerceptionEnabled())
        assertTrue(restored.isScreenTrainingEnabled())
        assertTrue(restored.isAccessibilityActionsEnabled())
        assertFalse(restored.isAppExcluded("com.example.notes"))
        assertTrue(restored.isAppTrainingAllowed("com.example.notes"))
    }
}
