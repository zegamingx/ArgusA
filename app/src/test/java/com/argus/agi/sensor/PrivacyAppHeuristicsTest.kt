package com.argus.agi.sensor

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PrivacyAppHeuristicsTest {

    @Test
    fun detectsSensitiveAppsFromPackageNames() {
        assertTrue(
            PrivacyAppHeuristics.isPrivacySensitiveApp(
                appName = "Livin' by Mandiri",
                packageName = "com.bankmandiri",
            ),
        )
        assertTrue(
            PrivacyAppHeuristics.isPrivacySensitiveApp(
                appName = "Authenticator",
                packageName = "com.example.secure",
            ),
        )
        assertTrue(
            PrivacyAppHeuristics.isPrivacySensitiveApp(
                appName = "WhatsApp",
                packageName = "com.whatsapp",
            ),
        )
    }

    @Test
    fun detectsSensitiveAppsFromDisplayNames() {
        assertTrue(
            PrivacyAppHeuristics.isPrivacySensitiveApp(
                appName = "DANA",
                packageName = "com.example.payment",
            ),
        )
        assertTrue(
            PrivacyAppHeuristics.isPrivacySensitiveApp(
                appName = "Secure Wallet",
                packageName = "com.example.finance",
            ),
        )
    }

    @Test
    fun doesNotFlagUnrelatedApps() {
        assertFalse(
            PrivacyAppHeuristics.isPrivacySensitiveApp(
                appName = "Calculator",
                packageName = "com.example.calculator",
            ),
        )
    }
}
