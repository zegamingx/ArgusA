package com.argus.agi.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Attempt accounting for the §5.2 automatic Vulkan recovery policy. All times
 * are injected, so the backoff schedule is verified exactly without sleeping.
 */
class VulkanRuntimeRecoveryPolicyTest {

    private fun policy(
        maxAttempts: Int = 3,
        initialBackoffMillis: Long = 1_000L,
        maxBackoffMillis: Long = 4_000L,
        episodeCooldownMillis: Long = 10_000L
    ) = VulkanRuntimeRecoveryPolicy(
        maxAttemptsPerEpisode = maxAttempts,
        initialBackoffMillis = initialBackoffMillis,
        maxBackoffMillis = maxBackoffMillis,
        episodeCooldownMillis = episodeCooldownMillis
    )

    @Test
    fun productionDefaultsMatchTheDocumentedRecoveryBudget() {
        val defaults = VulkanRuntimeRecoveryPolicy()
        assertEquals(4, defaults.maxAttemptsPerEpisode)
        assertEquals(5_000L, defaults.initialBackoffMillis)
        assertEquals(60_000L, defaults.maxBackoffMillis)
        assertEquals(300_000L, defaults.episodeCooldownMillis)
    }

    @Test
    fun healthyRuntimeNeverNeedsRecovery() {
        val recovery = policy()
        repeat(3) {
            assertEquals(
                VulkanRuntimeRecoveryPolicy.Decision.NOT_NEEDED,
                recovery.evaluate(nowMillis = 1_000L, failureLatched = false, runtimeVerified = true)
            )
        }
        assertEquals(0, recovery.attemptsUsedInEpisode)
        assertFalse(recovery.isExhausted)
    }

    @Test
    fun runtimeThatNeverVerifiedIsACapabilityFallbackAndIsNeverRetried() {
        val recovery = policy()
        // A device whose Vulkan runtime never completed verification keeps CPU
        // as its capability fallback; no episode may open and no attempt may
        // ever be released, however much time passes.
        for (now in longArrayOf(0L, 1_000L, 60_000L, 3_600_000L)) {
            assertEquals(
                VulkanRuntimeRecoveryPolicy.Decision.NOT_NEEDED,
                recovery.evaluate(nowMillis = now, failureLatched = true, runtimeVerified = false)
            )
        }
        assertEquals(0, recovery.attemptsUsedInEpisode)
        assertFalse(recovery.isExhausted)
    }

    @Test
    fun firstAttemptWaitsForTheInitialBackoff() {
        val recovery = policy()
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.WAIT,
            recovery.evaluate(nowMillis = 10_000L, failureLatched = true, runtimeVerified = true)
        )
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.WAIT,
            recovery.evaluate(nowMillis = 10_999L, failureLatched = true, runtimeVerified = true)
        )
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.ATTEMPT,
            recovery.evaluate(nowMillis = 11_000L, failureLatched = true, runtimeVerified = true)
        )
    }

    @Test
    fun attemptInFlightBlocksFurtherAttemptsUntilItsOutcomeIsRecorded() {
        val recovery = policy()
        recovery.evaluate(nowMillis = 0L, failureLatched = true, runtimeVerified = true)
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.ATTEMPT,
            recovery.evaluate(nowMillis = 1_000L, failureLatched = true, runtimeVerified = true)
        )
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.WAIT,
            recovery.evaluate(nowMillis = 5_000L, failureLatched = true, runtimeVerified = true)
        )
        recovery.recordAttemptOutcome(nowMillis = 5_000L, succeeded = false)
        assertEquals(1, recovery.attemptsUsedInEpisode)
    }

    @Test
    fun recordingAnOutcomeWithoutAnAttemptInFlightThrows() {
        val recovery = policy()
        assertThrows(IllegalStateException::class.java) {
            recovery.recordAttemptOutcome(nowMillis = 0L, succeeded = true)
        }
    }

    @Test
    fun failedAttemptsBackOffExponentiallyAndExhaustTheEpisode() {
        val recovery = policy()
        // Episode opens at t=0; attempt 1 is due after the 1s initial backoff.
        recovery.evaluate(nowMillis = 0L, failureLatched = true, runtimeVerified = true)
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.ATTEMPT,
            recovery.evaluate(nowMillis = 1_000L, failureLatched = true, runtimeVerified = true)
        )
        assertEquals(1, recovery.currentAttemptNumber())
        recovery.recordAttemptOutcome(nowMillis = 1_000L, succeeded = false)

        // Attempt 2 after a doubled 2s backoff.
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.WAIT,
            recovery.evaluate(nowMillis = 2_999L, failureLatched = true, runtimeVerified = true)
        )
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.ATTEMPT,
            recovery.evaluate(nowMillis = 3_000L, failureLatched = true, runtimeVerified = true)
        )
        assertEquals(2, recovery.currentAttemptNumber())
        recovery.recordAttemptOutcome(nowMillis = 3_000L, succeeded = false)

        // Attempt 3 after the 4s maximum backoff.
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.ATTEMPT,
            recovery.evaluate(nowMillis = 7_000L, failureLatched = true, runtimeVerified = true)
        )
        assertEquals(3, recovery.currentAttemptNumber())
        recovery.recordAttemptOutcome(nowMillis = 7_000L, succeeded = false)

        // The episode is exhausted: no further attempts during the cooldown.
        assertTrue(recovery.isExhausted)
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.WAIT,
            recovery.evaluate(nowMillis = 16_999L, failureLatched = true, runtimeVerified = true)
        )
    }

    @Test
    fun exhaustedEpisodeOpensAFreshEpisodeAfterTheCooldown() {
        val recovery = policy(maxAttempts = 1)
        recovery.evaluate(nowMillis = 0L, failureLatched = true, runtimeVerified = true)
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.ATTEMPT,
            recovery.evaluate(nowMillis = 1_000L, failureLatched = true, runtimeVerified = true)
        )
        recovery.recordAttemptOutcome(nowMillis = 1_000L, succeeded = false)
        assertTrue(recovery.isExhausted)

        // Cooldown ends at t=11_000; the still-latched failure opens a fresh
        // episode whose first attempt is due one initial backoff later.
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.WAIT,
            recovery.evaluate(nowMillis = 11_000L, failureLatched = true, runtimeVerified = true)
        )
        assertFalse(recovery.isExhausted)
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.ATTEMPT,
            recovery.evaluate(nowMillis = 12_000L, failureLatched = true, runtimeVerified = true)
        )
        assertEquals(1, recovery.currentAttemptNumber())
    }

    @Test
    fun successfulRecoveryResetsTheEpisodeCompletely() {
        val recovery = policy()
        recovery.evaluate(nowMillis = 0L, failureLatched = true, runtimeVerified = true)
        recovery.evaluate(nowMillis = 1_000L, failureLatched = true, runtimeVerified = true)
        recovery.recordAttemptOutcome(nowMillis = 1_000L, succeeded = false)
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.ATTEMPT,
            recovery.evaluate(nowMillis = 3_000L, failureLatched = true, runtimeVerified = true)
        )
        recovery.recordAttemptOutcome(nowMillis = 3_000L, succeeded = true)
        assertEquals(0, recovery.attemptsUsedInEpisode)
        assertFalse(recovery.isExhausted)

        // A later, unrelated failure starts a brand-new episode with the full
        // attempt budget and the initial backoff, not the escalated one.
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.WAIT,
            recovery.evaluate(nowMillis = 100_000L, failureLatched = true, runtimeVerified = true)
        )
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.ATTEMPT,
            recovery.evaluate(nowMillis = 101_000L, failureLatched = true, runtimeVerified = true)
        )
        assertEquals(1, recovery.currentAttemptNumber())
    }

    @Test
    fun healthyObservationMidEpisodeClearsTheEpisode() {
        val recovery = policy()
        recovery.evaluate(nowMillis = 0L, failureLatched = true, runtimeVerified = true)
        recovery.evaluate(nowMillis = 1_000L, failureLatched = true, runtimeVerified = true)
        recovery.recordAttemptOutcome(nowMillis = 1_000L, succeeded = false)
        assertEquals(1, recovery.attemptsUsedInEpisode)

        // Another path (for example the explicit core-initialization retry)
        // recovered the runtime: the failure is gone, so the episode closes.
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.NOT_NEEDED,
            recovery.evaluate(nowMillis = 2_000L, failureLatched = false, runtimeVerified = true)
        )
        assertEquals(0, recovery.attemptsUsedInEpisode)
        assertFalse(recovery.isExhausted)
    }

    @Test
    fun backoffIsCappedAtTheConfiguredMaximum() {
        val recovery = policy(
            maxAttempts = 4,
            initialBackoffMillis = 3_000L,
            maxBackoffMillis = 5_000L,
            episodeCooldownMillis = 60_000L
        )
        recovery.evaluate(nowMillis = 0L, failureLatched = true, runtimeVerified = true)
        // Attempt 1 at +3s; doubling would give 6s, capped to 5s.
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.ATTEMPT,
            recovery.evaluate(nowMillis = 3_000L, failureLatched = true, runtimeVerified = true)
        )
        recovery.recordAttemptOutcome(nowMillis = 3_000L, succeeded = false)
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.WAIT,
            recovery.evaluate(nowMillis = 7_999L, failureLatched = true, runtimeVerified = true)
        )
        assertEquals(
            VulkanRuntimeRecoveryPolicy.Decision.ATTEMPT,
            recovery.evaluate(nowMillis = 8_000L, failureLatched = true, runtimeVerified = true)
        )
    }

    @Test
    fun invalidPolicyConfigurationIsRejected() {
        assertThrows(IllegalArgumentException::class.java) {
            VulkanRuntimeRecoveryPolicy(maxAttemptsPerEpisode = 0)
        }
        assertThrows(IllegalArgumentException::class.java) {
            VulkanRuntimeRecoveryPolicy(initialBackoffMillis = 0L)
        }
        assertThrows(IllegalArgumentException::class.java) {
            VulkanRuntimeRecoveryPolicy(
                initialBackoffMillis = 10_000L,
                maxBackoffMillis = 5_000L
            )
        }
        assertThrows(IllegalArgumentException::class.java) {
            VulkanRuntimeRecoveryPolicy(episodeCooldownMillis = 0L)
        }
    }
}
