package com.argus.agi.model

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class TokenizerBitsPerByteTest {

    @Test
    fun computesBitsPerByteFromTotalNegativeLogLikelihood() {
        val oneBit = TokenizerEvaluationMetrics.bitsPerByte(
            totalNegativeLogLikelihood = kotlin.math.ln(2.0),
            utf8ByteCount = 1L
        )
        assertEquals(1.0f, oneBit, 1e-6f)

        val twoBitsPerByte = TokenizerEvaluationMetrics.bitsPerByte(
            totalNegativeLogLikelihood = kotlin.math.ln(2.0) * 8.0,
            utf8ByteCount = 4L
        )
        assertEquals(2.0f, twoBitsPerByte, 1e-6f)
    }

    @Test
    fun invalidMetricInputsReturnNaN() {
        assertTrue(
            TokenizerEvaluationMetrics.bitsPerByte(Double.NaN, 10L).isNaN()
        )
        assertTrue(
            TokenizerEvaluationMetrics.bitsPerByte(-1.0, 10L).isNaN()
        )
        assertTrue(
            TokenizerEvaluationMetrics.bitsPerByte(1.0, 0L).isNaN()
        )
    }

    @Test
    fun byteDenominatorUsesTheSameReplacementPolicyAsEncoding() {
        val tokenizer = Tokenizer()
        assertEquals(1L, tokenizer.utf8ByteCount("A"))
        assertEquals(2L, tokenizer.utf8ByteCount("é"))
        assertEquals(4L, tokenizer.utf8ByteCount("😀"))
        assertEquals(3L, tokenizer.utf8ByteCount(0xD800.toChar().toString()))
        assertEquals(
            "A😀".toByteArray(Charsets.UTF_8).size.toLong(),
            tokenizer.utf8ByteCount("A😀")
        )
    }
}
