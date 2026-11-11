package com.argus.agi.sensor

import java.util.Locale

/**
 * Identifies applications that deserve an explicit privacy review.
 *
 * This classifier does not grant or revoke accessibility access and does not
 * replace the user's persisted excluded-app selection.
 */
object PrivacyAppHeuristics {
    private val sensitiveTerms = setOf(
        "bank",
        "bca",
        "brimo",
        "mandiri",
        "bni",
        "cimb",
        "dana",
        "ovo",
        "gopay",
        "shopeepay",
        "wallet",
        "crypto",
        "binance",
        "indodax",
        "authenticator",
        "password",
        "bitwarden",
        "1password",
        "keepass",
        "autofill",
        "keychain",
        "whatsapp",
        "telegram",
        "signal"
    )

    fun isPrivacySensitiveApp(appName: String, packageName: String): Boolean {
        val identity = "$appName $packageName".lowercase(Locale.ROOT)
        return sensitiveTerms.any { term -> identity.contains(term) }
    }
}
