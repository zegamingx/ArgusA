package com.argus.agi.model

import kotlin.math.sqrt
import kotlin.random.Random
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class RotaryPositionEncodingTest {

    @Test
    fun attentionScoreDependsOnlyOnRelativePosition() {
        val headDimension = 32
        val random = Random(20261125)
        val query = FloatArray(headDimension) { random.nextFloat() * 2f - 1f }
        val key = FloatArray(headDimension) { random.nextFloat() * 2f - 1f }

        // For every absolute placement with the same offset, the rotated dot
        // product must be identical: this is the defining RoPE property.
        val offset = 5
        var referenceScore: Float? = null
        for (keyPosition in 0..24) {
            val queryPosition = keyPosition + offset
            val rotatedQuery = query.copyOf()
            val rotatedKey = key.copyOf()
            RotaryPositionEncoding.rotateVectorInPlace(
                rotatedQuery, 0, headDimension, queryPosition
            )
            RotaryPositionEncoding.rotateVectorInPlace(
                rotatedKey, 0, headDimension, keyPosition
            )
            val score = dot(rotatedQuery, rotatedKey)
            val reference = referenceScore
            if (reference == null) {
                referenceScore = score
            } else {
                assertEquals(
                    "Score drifted between absolute placements at key position $keyPosition",
                    reference.toDouble(),
                    score.toDouble(),
                    1e-3
                )
            }
        }

        // And the shared score equals rotating only the query by the offset
        // against the unrotated key.
        val offsetRotatedQuery = query.copyOf()
        RotaryPositionEncoding.rotateVectorInPlace(
            offsetRotatedQuery, 0, headDimension, offset
        )
        assertEquals(
            referenceScore!!.toDouble(),
            dot(offsetRotatedQuery, key).toDouble(),
            1e-3
        )
    }

    @Test
    fun inverseRotationRestoresTheOriginalVector() {
        val headDimension = 24
        val random = Random(7)
        val original = FloatArray(headDimension) { random.nextFloat() * 2f - 1f }
        val rotated = original.copyOf()

        RotaryPositionEncoding.rotateVectorInPlace(rotated, 0, headDimension, 37)
        RotaryPositionEncoding.rotateVectorInPlace(
            rotated, 0, headDimension, 37, inverse = true
        )

        for (index in original.indices) {
            assertEquals(original[index].toDouble(), rotated[index].toDouble(), 1e-5)
        }
    }

    @Test
    fun rotationPreservesVectorNormAndLeavesPositionZeroUntouched() {
        val headDimension = 32
        val random = Random(99)
        val vector = FloatArray(headDimension) { random.nextFloat() * 2f - 1f }
        val normBefore = sqrt(dot(vector, vector).toDouble())

        val rotated = vector.copyOf()
        RotaryPositionEncoding.rotateVectorInPlace(rotated, 0, headDimension, 123)
        val normAfter = sqrt(dot(rotated, rotated).toDouble())
        assertEquals(normBefore, normAfter, 1e-4)

        val positionZero = vector.copyOf()
        RotaryPositionEncoding.rotateVectorInPlace(positionZero, 0, headDimension, 0)
        for (index in vector.indices) {
            assertEquals(vector[index].toDouble(), positionZero[index].toDouble(), 0.0)
        }
    }

    @Test
    fun rowRotationAppliesEachHeadsOwnOffset() {
        val headCount = 4
        val headDimension = 8
        val row = FloatArray(headCount * headDimension) { index -> index.toFloat() }
        val rotated = row.copyOf()

        RotaryPositionEncoding.rotateRowInPlace(rotated, headCount, headDimension, 3)

        for (head in 0 until headCount) {
            val expected = row.copyOf()
            RotaryPositionEncoding.rotateVectorInPlace(
                expected, head * headDimension, headDimension, 3
            )
            for (dimension in 0 until headDimension) {
                assertEquals(
                    expected[head * headDimension + dimension].toDouble(),
                    rotated[head * headDimension + dimension].toDouble(),
                    1e-6
                )
            }
        }
        assertTrue("Rotation must actually change a non-zero position row",
            rotated.indices.any { rotated[it] != row[it] })
    }

    @Test
    fun rowsRotationUsesTheRowIndexAsPosition() {
        val headCount = 2
        val headDimension = 16
        val random = Random(4242)
        val rows = Array(9) {
            FloatArray(headCount * headDimension) { random.nextFloat() * 2f - 1f }
        }
        val rotated = Array(rows.size) { rows[it].copyOf() }

        RotaryPositionEncoding.rotateRowsInPlace(rotated, headCount, headDimension)

        for (position in rows.indices) {
            val expected = rows[position].copyOf()
            RotaryPositionEncoding.rotateRowInPlace(
                expected, headCount, headDimension, position
            )
            for (index in expected.indices) {
                assertEquals(
                    expected[index].toDouble(),
                    rotated[position][index].toDouble(),
                    1e-6
                )
            }
        }
    }

    private fun dot(left: FloatArray, right: FloatArray): Float {
        var total = 0f
        for (index in left.indices) total += left[index] * right[index]
        return total
    }
}
