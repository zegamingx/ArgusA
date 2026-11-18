package com.argus.agi.model

import java.nio.charset.StandardCharsets
import java.security.MessageDigest

/**
 * Immutable tokenizer state participating in the active model identity.
 *
 * The vocabulary and merge fingerprint identifies the exact token-ID mapping;
 * the tokenizer config identifies its decoding and capacity contract.
 */
internal data class TokenizerModelState(
    val schemaVersion: Int,
    val algorithm: ArgusTokenizerAlgorithm,
    val maximumVocabularySize: Int,
    val maxTokenLength: Int,
    val vocabularySize: Int,
    val snapshotFingerprint: String
) {
    init {
        require(schemaVersion > 0)
        require(maximumVocabularySize >= TokenizerConfig.MINIMUM_VOCAB_SIZE)
        require(vocabularySize in TokenizerConfig.MINIMUM_VOCAB_SIZE..maximumVocabularySize)
        require(maxTokenLength in TokenizerConfig.MINIMUM_TOKEN_LENGTH..TokenizerConfig.MAX_SUPPORTED_TOKEN_LENGTH)
        require(
            snapshotFingerprint.length == SHA256_HEX_LENGTH &&
                snapshotFingerprint.all { character ->
                    character in '0'..'9' || character in 'a'..'f'
                }
        ) {
            "Tokenizer snapshot fingerprint must be a lowercase SHA-256 hex digest."
        }
    }

    private companion object {
        const val SHA256_HEX_LENGTH = 64
    }
}

/**
 * Backend-independent model identity. Runtime batch size, workgroup geometry,
 * thermal state and memory residency are deliberately excluded.
 */
internal data class ModelStateIdentity(
    val architecture: ArgusArchitectureConfig,
    val tokenizer: TokenizerModelState
) {
    init {
        require(architecture.vocabularySize >= tokenizer.vocabularySize) {
            "Architecture vocabulary capacity cannot be smaller than the active tokenizer vocabulary."
        }
    }

    /** Stable identity for compatibility checks and diagnostic correlation. */
    val fingerprint: String by lazy(LazyThreadSafetyMode.PUBLICATION) {
        val canonical = buildString {
            append("ARGUS-MODEL-STATE-IDENTITY-V1\n")
            append(architecture.schemaVersion).append('\n')
            append(architecture.architectureId.length).append(':')
                .append(architecture.architectureId).append('\n')
            append(architecture.layerCount).append('|')
            append(architecture.hiddenSize).append('|')
            append(architecture.attentionHeadCount).append('|')
            append(architecture.feedForwardSize).append('|')
            append(architecture.vocabularySize).append('|')
            append(architecture.contextLength).append('|')
            append(architecture.adapterRank).append('|')
            append(architecture.attentionKind.name).append('|')
            append(architecture.feedForwardKind.name).append('|')
            append(architecture.positionEncodingKind.name).append('|')
            append(architecture.normalizationKind.name).append('|')
            append(architecture.tokenEmbeddingsAreTied).append('\n')
            append(tokenizer.schemaVersion).append('|')
            append(tokenizer.algorithm.checkpointId).append('|')
            append(tokenizer.maximumVocabularySize).append('|')
            append(tokenizer.maxTokenLength).append('|')
            append(tokenizer.vocabularySize).append('\n')
            append(tokenizer.snapshotFingerprint.length).append(':')
                .append(tokenizer.snapshotFingerprint)
        }
        MessageDigest.getInstance("SHA-256")
            .digest(canonical.toByteArray(StandardCharsets.UTF_8))
            .joinToString(separator = "") { byte ->
                "%02x".format(java.util.Locale.ROOT, byte.toInt() and 0xFF)
            }
    }
}

internal enum class ModelCheckpointFormat(
    val magic: String,
    val version: Int
) {
    ARG5("ARG5", 5),
    ARG6("ARG6", 6)
}

/**
 * Immutable view of the committed model state.
 *
 * [identity] describes model and tokenizer semantics; optimizerStep describes
 * learning progress; checkpointVerified only reports local persistence integrity.
 */
internal data class ModelState(
    val identity: ModelStateIdentity,
    val optimizerStep: Int,
    val checkpointFormat: ModelCheckpointFormat,
    val checkpointVerified: Boolean,
    val optimizerConfiguration: OptimizerConfigIdentity = OptimizerConfig.DEFAULT.identity,
    val checkpointOptimizerConfigurationFingerprint: String? = null
) {
    init {
        require(optimizerStep >= 0) {
            "Optimizer step cannot be negative."
        }
        val persistedFingerprint = checkpointOptimizerConfigurationFingerprint
        require(persistedFingerprint == null || (
            persistedFingerprint.length == 64 && persistedFingerprint.all { character ->
                character in '0'..'9' || character in 'a'..'f'
            }
        )) {
            "Persisted optimizer configuration identity must be a lowercase SHA-256 fingerprint."
        }
        if (checkpointFormat == ModelCheckpointFormat.ARG5) {
            require(persistedFingerprint == null) {
                "ARG5 checkpoints cannot claim a persisted optimizer configuration fingerprint."
            }
        }
        if (checkpointVerified && checkpointFormat == ModelCheckpointFormat.ARG6) {
            require(persistedFingerprint != null) {
                "A verified ARG6 checkpoint must expose its persisted optimizer configuration fingerprint."
            }
        }
    }
}
