package com.argus.agi.sensor

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class PrivacyAppHeuristicsTest {

    @Test
    fun detectsFinancialCredentialAndPrivateCommunicationApps() {
        val protectedApps = listOf(
            "Livin' by Mandiri" to "com.bankmandiri",
            "Authenticator" to "com.example.secure",
            "WhatsApp" to "com.whatsapp",
            "Gmail" to "com.google.android.gm",
            "Private Messenger" to "com.example.messenger"
        )

        protectedApps.forEach { (appName, packageName) ->
            assertTrue(
                "$appName should be protected",
                PrivacyAppHeuristics.isPrivacySensitiveApp(appName, packageName)
            )
        }
    }

    @Test
    fun detectsSensitiveAppsFromDisplayNames() {
        assertTrue(
            PrivacyAppHeuristics.isPrivacySensitiveApp(
                appName = "DANA",
                packageName = "com.example.payment"
            )
        )
        assertTrue(
            PrivacyAppHeuristics.isPrivacySensitiveApp(
                appName = "Secure Wallet",
                packageName = "com.example.finance"
            )
        )
        assertTrue(
            PrivacyAppHeuristics.isPrivacySensitiveApp(
                appName = "Personal Health",
                packageName = "com.example.records"
            )
        )
    }

    @Test
    fun doesNotFlagUnrelatedApps() {
        assertFalse(
            PrivacyAppHeuristics.isPrivacySensitiveApp(
                appName = "Calculator",
                packageName = "com.example.calculator"
            )
        )
        assertFalse(
            PrivacyAppHeuristics.isPrivacySensitiveApp(
                appName = "Offline Notes",
                packageName = "com.example.notes"
            )
        )
    }
}
