package com.argus.agi.sensor

import java.util.Locale

/**
 * Conservatively identifies applications whose visible content commonly contains
 * financial, credential, private-communication, health, or personal-media data.
 *
 * Matching is a defense-in-depth block, not a replacement for the default-deny
 * app allowlist or the user's explicit consent.
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
        "tokocrypto",
        "authenticator",
        "password",
        "bitwarden",
        "1password",
        "keepass",
        "autofill",
        "keychain",
        "whatsapp",
        "telegram",
        "signal",
        "messenger",
        "instagram",
        "facebook",
        "discord",
        "snapchat",
        "gmail",
        "outlook",
        "mail",
        "messages",
        "sms",
        "health",
        "medical",
        "photos",
        "gallery"
    )

    fun isPrivacySensitiveApp(appName: String, packageName: String): Boolean {
        val identity = "$appName $packageName".lowercase(Locale.ROOT)
        return sensitiveTerms.any(identity::contains)
    }
}
