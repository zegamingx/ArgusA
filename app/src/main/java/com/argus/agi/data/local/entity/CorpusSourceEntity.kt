package com.argus.agi.data.local.entity

import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.PrimaryKey
import java.net.URI

/**
 * Provenance record for one origin of a deduplicated corpus document.
 * A single content fingerprint may therefore retain multiple source references.
 */
@Entity(
    tableName = "corpus_sources",
    foreignKeys = [
        ForeignKey(
            entity = CorpusDocumentEntity::class,
            parentColumns = ["id"],
            childColumns = ["documentId"],
            onDelete = ForeignKey.CASCADE
        )
    ],
    indices = [Index(value = ["documentId"])]
)
data class CorpusSourceEntity(
    @PrimaryKey
    val id: String,
    val documentId: String,
    val source: String,
    val canonicalUrl: String?,
    val language: String?,
    val rightsStatus: String,
    val importedAtEpochMs: Long,
    val licenseIdentifier: String? = null,
    val licenseName: String? = null,
    val licenseUrl: String? = null,
    val attributionText: String? = null,
    val rightsEvidenceUrl: String? = null,
    val userDeclarationText: String? = null,
    val userDeclaredAtEpochMs: Long? = null,
    val reviewedAtEpochMs: Long? = null
)

/**
 * Auditable rights evidence associated with one corpus source.
 *
 * USER_DECLARED records a deliberate user approval. LICENSE_REVIEWED records
 * a reviewed license, its attribution, and the evidence used for the review.
 */
data class CorpusSourceProvenance(
    val licenseIdentifier: String? = null,
    val licenseName: String? = null,
    val licenseUrl: String? = null,
    val attributionText: String? = null,
    val rightsEvidenceUrl: String? = null,
    val userDeclarationText: String? = null,
    val userDeclaredAtEpochMs: Long? = null,
    val reviewedAtEpochMs: Long? = null
) {

    fun validateFor(rightsStatus: String) {
        require(rightsStatus in SUPPORTED_RIGHTS_STATUSES) {
            "Unsupported corpus rights status: $rightsStatus."
        }

        when (rightsStatus) {
            USER_DECLARED -> {
                require(!userDeclarationText.isNullOrBlank()) {
                    "User-declared corpus content requires a recorded declaration."
                }
                require(userDeclaredAtEpochMs != null && userDeclaredAtEpochMs > 0L) {
                    "User-declared corpus content requires a positive declaration timestamp."
                }
            }

            LICENSE_REVIEWED -> {
                require(!licenseIdentifier.isNullOrBlank()) {
                    "License-reviewed corpus content requires a license identifier."
                }
                require(!licenseName.isNullOrBlank()) {
                    "License-reviewed corpus content requires a license name."
                }
                require(isHttpsUrl(licenseUrl)) {
                    "License-reviewed corpus content requires an HTTPS license URL."
                }
                require(!attributionText.isNullOrBlank()) {
                    "License-reviewed corpus content requires attribution text."
                }
                require(isHttpsUrl(rightsEvidenceUrl)) {
                    "License-reviewed corpus content requires an HTTPS rights evidence URL."
                }
                require(reviewedAtEpochMs != null && reviewedAtEpochMs > 0L) {
                    "License-reviewed corpus content requires a positive review timestamp."
                }
            }
        }
    }

    private fun isHttpsUrl(value: String?): Boolean {
        if (value.isNullOrBlank()) return false
        return runCatching {
            val uri = URI(value)
            uri.scheme.equals("https", ignoreCase = true) && !uri.host.isNullOrBlank()
        }.getOrDefault(false)
    }

    companion object {
        const val UNVERIFIED = "UNVERIFIED"
        const val USER_DECLARED = "USER_DECLARED"
        const val LICENSE_REVIEWED = "LICENSE_REVIEWED"

        val SUPPORTED_RIGHTS_STATUSES: Set<String> = setOf(
            UNVERIFIED,
            USER_DECLARED,
            LICENSE_REVIEWED
        )
    }
}
