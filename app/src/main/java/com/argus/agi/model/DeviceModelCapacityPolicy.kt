package com.argus.agi.model

import java.util.Locale
import kotlin.math.roundToInt

/**
 * RAM-only model-size planning policy.
 *
 * The anchors intentionally preserve the device-capacity ranges supplied for Argus.
 * They are not a promise that a particular GGUF file can be loaded: model metadata,
 * quantization, KV cache, runtime workspace, process limits, and backend support must
 * still be checked by the inference loader.
 *
 * availableRamGb is the conservative free-memory estimate produced by HardwareProfile
 * (availableMem minus Android's low-memory threshold), not total installed RAM.
 */
data class DeviceModelCapacityRecommendation(
    val availableRamGb: Double,
    val minimumParametersBillions: Double?,
    val maximumParametersBillions: Double?,
    val isExtrapolated: Boolean
) {
    val hasRecommendation: Boolean
        get() = minimumParametersBillions != null && maximumParametersBillions != null

    val parameterRangeLabel: String
        get() {
            val minimum = minimumParametersBillions ?: return "No target below 1 GB available RAM"
            val maximum = maximumParametersBillions ?: return "No target below 1 GB available RAM"
            return "${formatBillions(minimum)}B – ${formatBillions(maximum)}B"
        }

    private fun formatBillions(value: Double): String {
        val rounded = (value * 10.0).roundToInt() / 10.0
        return if (rounded % 1.0 == 0.0) {
            rounded.toLong().toString()
        } else {
            String.format(Locale.US, "%.1f", rounded)
        }
    }
}

object DeviceModelCapacityPolicy {

    private data class Anchor(
        val availableRamGb: Double,
        val minimumParametersBillions: Double,
        val maximumParametersBillions: Double
    )

    private val anchors = listOf(
        Anchor(1.0, 0.5, 1.0),
        Anchor(2.0, 1.5, 2.5),
        Anchor(3.0, 2.5, 3.5),
        Anchor(4.0, 3.0, 5.0),
        Anchor(5.0, 4.0, 6.0),
        Anchor(6.0, 5.0, 7.0),
        Anchor(7.0, 6.0, 8.0),
        Anchor(8.0, 7.0, 9.0),
        Anchor(10.0, 9.0, 12.0),
        Anchor(12.0, 12.0, 14.0),
        Anchor(16.0, 18.0, 21.0),
        Anchor(24.0, 28.0, 32.0),
        Anchor(32.0, 38.0, 42.0)
    )

    /**
     * Returns null bounds below the first calibrated point. Values between anchors are
     * linearly interpolated. Above 32 GB, the requested 1.3B–1.5B per-GB growth is used.
     */
    fun recommend(availableRamGb: Double): DeviceModelCapacityRecommendation {
        if (!availableRamGb.isFinite() || availableRamGb < anchors.first().availableRamGb) {
            return DeviceModelCapacityRecommendation(
                availableRamGb = availableRamGb.takeIf { it.isFinite() }?.coerceAtLeast(0.0) ?: 0.0,
                minimumParametersBillions = null,
                maximumParametersBillions = null,
                isExtrapolated = false
            )
        }

        val exact = anchors.firstOrNull { it.availableRamGb == availableRamGb }
        if (exact != null) {
            return exact.toRecommendation(availableRamGb, isExtrapolated = false)
        }

        val last = anchors.last()
        if (availableRamGb > last.availableRamGb) {
            val additionalGb = availableRamGb - last.availableRamGb
            return DeviceModelCapacityRecommendation(
                availableRamGb = availableRamGb,
                minimumParametersBillions = last.minimumParametersBillions + (additionalGb * 1.3),
                maximumParametersBillions = last.maximumParametersBillions + (additionalGb * 1.5),
                isExtrapolated = true
            )
        }

        val upperIndex = anchors.indexOfFirst { it.availableRamGb > availableRamGb }
        val lower = anchors[upperIndex - 1]
        val upper = anchors[upperIndex]
        val fraction =
            (availableRamGb - lower.availableRamGb) /
                (upper.availableRamGb - lower.availableRamGb)

        val minimum = interpolate(
            lower.minimumParametersBillions,
            upper.minimumParametersBillions,
            fraction
        )
        val maximum = interpolate(
            lower.maximumParametersBillions,
            upper.maximumParametersBillions,
            fraction
        )

        return DeviceModelCapacityRecommendation(
            availableRamGb = availableRamGb,
            minimumParametersBillions = minimum,
            maximumParametersBillions = maximum,
            isExtrapolated = false
        )
    }

    private fun interpolate(start: Double, end: Double, fraction: Double): Double =
        start + ((end - start) * fraction)

    private fun Anchor.toRecommendation(
        availableRamGb: Double,
        isExtrapolated: Boolean
    ): DeviceModelCapacityRecommendation = DeviceModelCapacityRecommendation(
        availableRamGb = availableRamGb,
        minimumParametersBillions = minimumParametersBillions,
        maximumParametersBillions = maximumParametersBillions,
        isExtrapolated = isExtrapolated
    )
}
