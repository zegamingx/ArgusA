package com.argus.agi.model

/**
 * Bounded automatic-recovery policy for a Vulkan runtime that was verified on this
 * device and then latched off by a runtime dispatch failure.
 *
 * AGENTS.md §5.2 makes Vulkan the primary training backend and permits CPU only as
 * a compatibility, initialization, or capability fallback, or as a controlled
 * recovery mechanism: "CPU must not silently become the permanent primary backend
 * on a compatible device." A single transient dispatch failure (for example a
 * fence timeout or a device-lost report on a contended GPU) therefore opens a
 * recovery episode instead of ending GPU execution for the rest of the session.
 *
 * The policy is a pure, clock-injected state machine. Callers pass the current
 * time and the observed backend state; the policy decides whether a recovery
 * attempt may start and accounts for its outcome. It performs no recovery itself
 * and has no Android or native dependencies, so the complete attempt accounting
 * is unit-testable.
 *
 * One failure episode allows up to [maxAttemptsPerEpisode] attempts with
 * exponential backoff ([initialBackoffMillis], doubling up to
 * [maxBackoffMillis]). When an episode is exhausted the policy enters a
 * cooldown of [episodeCooldownMillis]; a failure still latched after the
 * cooldown opens a fresh episode, so a GPU that recovers later in a long
 * training session is retried while a persistently dead GPU is never hammered.
 */
internal class VulkanRuntimeRecoveryPolicy(
    val maxAttemptsPerEpisode: Int = DEFAULT_MAX_ATTEMPTS_PER_EPISODE,
    val initialBackoffMillis: Long = DEFAULT_INITIAL_BACKOFF_MILLIS,
    val maxBackoffMillis: Long = DEFAULT_MAX_BACKOFF_MILLIS,
    val episodeCooldownMillis: Long = DEFAULT_EPISODE_COOLDOWN_MILLIS
) {

    enum class Decision {
        /** No recovery is required, or recovery must never be attempted. */
        NOT_NEEDED,

        /** A failure is latched but the next attempt is not due yet. */
        WAIT,

        /** A recovery attempt may start now; its outcome must be recorded. */
        ATTEMPT
    }

    init {
        require(maxAttemptsPerEpisode > 0) {
            "maxAttemptsPerEpisode must be positive."
        }
        require(initialBackoffMillis > 0L) {
            "initialBackoffMillis must be positive."
        }
        require(maxBackoffMillis >= initialBackoffMillis) {
            "maxBackoffMillis cannot be below initialBackoffMillis."
        }
        require(episodeCooldownMillis > 0L) {
            "episodeCooldownMillis must be positive."
        }
    }

    /** Attempts already consumed by the current failure episode. */
    var attemptsUsedInEpisode: Int = 0
        private set

    private var episodeActive = false
    private var attemptInFlight = false
    private var nextAttemptAtMillis = Long.MAX_VALUE
    private var cooldownUntilMillis = Long.MIN_VALUE

    /** True after an episode exhausted its attempts and before its cooldown ends. */
    val isExhausted: Boolean
        get() = !episodeActive && cooldownUntilMillis != Long.MIN_VALUE

    /** 1-based number of the attempt currently in flight, for diagnostics. */
    fun currentAttemptNumber(): Int = attemptsUsedInEpisode + 1

    /**
     * Evaluates whether a recovery attempt may start at [nowMillis].
     *
     * [failureLatched] reports that a dispatch failure has latched the runtime
     * (or its FP16 path) off. [runtimeVerified] reports that Vulkan completed
     * initialization and kernel verification earlier in this process; without it
     * the CPU path is the device-capability fallback and is never auto-retried.
     */
    fun evaluate(
        nowMillis: Long,
        failureLatched: Boolean,
        runtimeVerified: Boolean
    ): Decision {
        if (!failureLatched) {
            // A healthy observation (initial state, or a recovery performed by
            // any path) closes the episode bookkeeping.
            resetEpisode()
            return Decision.NOT_NEEDED
        }
        if (!runtimeVerified) return Decision.NOT_NEEDED
        if (attemptInFlight) return Decision.WAIT
        if (!episodeActive) {
            if (nowMillis < cooldownUntilMillis) return Decision.WAIT
            // Open a new episode. The first attempt waits one initial backoff so
            // a GPU that just hung is not reinitialized in the same instant.
            episodeActive = true
            attemptsUsedInEpisode = 0
            nextAttemptAtMillis = saturatedAdd(nowMillis, initialBackoffMillis)
            return Decision.WAIT
        }
        if (attemptsUsedInEpisode >= maxAttemptsPerEpisode) {
            // Defensive totality: exhaustion is normally recorded with the final
            // attempt's outcome, but never let a stale episode attempt again.
            closeEpisodeWithCooldown(nowMillis)
            return Decision.WAIT
        }
        if (nowMillis < nextAttemptAtMillis) return Decision.WAIT
        attemptInFlight = true
        return Decision.ATTEMPT
    }

    /**
     * Records the outcome of the attempt that [evaluate] released. A success
     * closes the episode completely; a failure consumes one attempt and either
     * schedules the next one with exponential backoff or, when the episode is
     * exhausted, starts the cooldown.
     */
    fun recordAttemptOutcome(nowMillis: Long, succeeded: Boolean) {
        check(attemptInFlight) {
            "A recovery outcome was recorded without an attempt in flight."
        }
        attemptInFlight = false
        if (succeeded) {
            resetEpisode()
            return
        }
        attemptsUsedInEpisode += 1
        if (attemptsUsedInEpisode >= maxAttemptsPerEpisode) {
            closeEpisodeWithCooldown(nowMillis)
        } else {
            nextAttemptAtMillis =
                saturatedAdd(nowMillis, backoffAfterFailedAttempts(attemptsUsedInEpisode))
        }
    }

    private fun backoffAfterFailedAttempts(failedAttempts: Int): Long {
        var backoff = initialBackoffMillis
        repeat(failedAttempts) {
            backoff = if (backoff > maxBackoffMillis / 2L) {
                maxBackoffMillis
            } else {
                (backoff * 2L).coerceAtMost(maxBackoffMillis)
            }
        }
        return backoff
    }

    private fun closeEpisodeWithCooldown(nowMillis: Long) {
        episodeActive = false
        attemptsUsedInEpisode = 0
        nextAttemptAtMillis = Long.MAX_VALUE
        cooldownUntilMillis = saturatedAdd(nowMillis, episodeCooldownMillis)
    }

    private fun resetEpisode() {
        episodeActive = false
        attemptInFlight = false
        attemptsUsedInEpisode = 0
        nextAttemptAtMillis = Long.MAX_VALUE
        cooldownUntilMillis = Long.MIN_VALUE
    }

    private fun saturatedAdd(value: Long, delta: Long): Long =
        if (value > Long.MAX_VALUE - delta) Long.MAX_VALUE else value + delta

    companion object {
        const val DEFAULT_MAX_ATTEMPTS_PER_EPISODE = 4
        const val DEFAULT_INITIAL_BACKOFF_MILLIS = 5_000L
        const val DEFAULT_MAX_BACKOFF_MILLIS = 60_000L
        const val DEFAULT_EPISODE_COOLDOWN_MILLIS = 300_000L
    }
}
