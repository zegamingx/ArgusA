package com.argus.agi.model

import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

/**
 * Rotary position embedding (RoPE) rotation applied to attention queries and
 * keys, as described by Su et al. ("RoFormer", 2021).
 *
 * Each head vector is split into two-dimensional pairs (2i, 2i + 1). The pair
 * for index i of a token at position p is rotated by the angle
 * p * BASE^(-2i / headDimension). Because the rotation is orthogonal, the
 * attention score between a query at position m and a key at position n
 * depends only on the relative offset m - n, and the exact inverse rotation
 * (negated angle) is the correct adjoint for the backward pass.
 *
 * Rotation is applied to the host Q/K rows before they are consumed by either
 * the Kotlin attention loop or the resident Vulkan attention graph, which
 * packs the same host rows into its tensor arena. Both execution paths
 * therefore compute genuine rotary scores from identical rotated inputs.
 *
 * If a head dimension is odd, the final unpaired component is left unrotated
 * (partial rotary application); forward and inverse stay exact adjoints
 * because they share the same pairing.
 */
internal object RotaryPositionEncoding {

    /** Frequency base from the RoFormer formulation. */
    const val BASE: Double = 10_000.0

    /**
     * Rotates one head vector of [headDimension] components starting at
     * [offset] inside [vector] for the token at [position]. With
     * [inverse] = true the adjoint (transpose) rotation is applied instead,
     * which is what the backward pass needs for query/key gradients.
     */
    fun rotateVectorInPlace(
        vector: FloatArray,
        offset: Int,
        headDimension: Int,
        position: Int,
        inverse: Boolean = false
    ) {
        require(headDimension > 0) { "Head dimension must be positive." }
        require(position >= 0) { "Position must not be negative." }
        require(offset >= 0 && offset + headDimension <= vector.size) {
            "Head vector range exceeds the row bounds."
        }
        if (position == 0) return
        val pairCount = headDimension / 2
        val direction = if (inverse) -1.0 else 1.0
        for (pair in 0 until pairCount) {
            val angle = position * BASE.pow(-2.0 * pair / headDimension) * direction
            val cosAngle = cos(angle).toFloat()
            val sinAngle = sin(angle).toFloat()
            val firstIndex = offset + pair * 2
            val first = vector[firstIndex]
            val second = vector[firstIndex + 1]
            vector[firstIndex] = first * cosAngle - second * sinAngle
            vector[firstIndex + 1] = first * sinAngle + second * cosAngle
        }
    }

    /**
     * Rotates every head of a single token row at [position].
     */
    fun rotateRowInPlace(
        row: FloatArray,
        headCount: Int,
        headDimension: Int,
        position: Int,
        inverse: Boolean = false
    ) {
        for (head in 0 until headCount) {
            rotateVectorInPlace(
                vector = row,
                offset = head * headDimension,
                headDimension = headDimension,
                position = position,
                inverse = inverse
            )
        }
    }

    /**
     * Rotates every head of every row; the row index is the token position.
     * Used for the training forward pass (forward rotation) and for the
     * assembled query/key gradients in the backward pass ([inverse] = true).
     */
    fun rotateRowsInPlace(
        rows: Array<FloatArray>,
        headCount: Int,
        headDimension: Int,
        inverse: Boolean = false
    ) {
        for (position in rows.indices) {
            rotateRowInPlace(
                row = rows[position],
                headCount = headCount,
                headDimension = headDimension,
                position = position,
                inverse = inverse
            )
        }
    }
}
