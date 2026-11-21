package com.argus.agi.ui

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ZipImportBudgetTest {

    private val reserveBytes = 256L * 1024L * 1024L
    private val mebibyte = 1024L * 1024L

    @Test
    fun returnsZeroWhenAvailableStorageDoesNotExceedTheReserve() {
        assertEquals(
            0L,
            calculateZipImportBudgetBytes(
                availableBytes = reserveBytes - 1L,
                reservedBytes = reserveBytes
            )
        )
        assertEquals(
            0L,
            calculateZipImportBudgetBytes(
                availableBytes = reserveBytes,
                reservedBytes = reserveBytes
            )
        )
    }

    @Test
    fun assignsOnlyOneQuarterOfStorageAboveTheReserve() {
        assertEquals(
            1L * mebibyte,
            calculateZipImportBudgetBytes(
                availableBytes = reserveBytes + 4L * mebibyte,
                reservedBytes = reserveBytes
            )
        )
        assertEquals(
            2L * mebibyte,
            calculateZipImportBudgetBytes(
                availableBytes = reserveBytes + 8L * mebibyte,
                reservedBytes = reserveBytes
            )
        )
    }

    @Test
    fun neverReturnsMoreThanOneQuarterOfAvailableStorage() {
        val availableBytes = 2L * 1024L * 1024L * 1024L
        val budget = calculateZipImportBudgetBytes(
            availableBytes = availableBytes,
            reservedBytes = reserveBytes
        )

        assertEquals((availableBytes - reserveBytes) / 4L, budget)
        assertTrue(budget <= availableBytes / 4L)
    }

    @Test
    fun rejectsNegativeStorageValues() {
        assertTrue(
            runCatching {
                calculateZipImportBudgetBytes(
                    availableBytes = -1L,
                    reservedBytes = reserveBytes
                )
            }.exceptionOrNull() is IllegalArgumentException
        )
        assertTrue(
            runCatching {
                calculateZipImportBudgetBytes(
                    availableBytes = reserveBytes,
                    reservedBytes = -1L
                )
            }.exceptionOrNull() is IllegalArgumentException
        )
    }
}
