package com.argus.agi.runtime

import org.junit.Assert.assertEquals
import org.junit.Test

class AdaptiveComputePerformanceMonitorTest {

    @Test
    fun recordsCpuFallbackAndVulkanSamplesAsPassiveTelemetry() {
        AdaptiveComputePerformanceMonitor.resetForTests()

        AdaptiveComputePerformanceMonitor.recordCpu(
            AdaptiveComputeOperation.LINEAR,
            operationCount = 300_000L,
            elapsedNanoseconds = 3_000_000L
        )
        AdaptiveComputePerformanceMonitor.recordGpu(
            AdaptiveComputeOperation.LINEAR,
            operationCount = 300_000L,
            elapsedNanoseconds = 2_000_000L
        )

        val snapshot = AdaptiveComputePerformanceMonitor.snapshot(
            AdaptiveComputeOperation.LINEAR,
            operationCount = 300_000L
        )
        assertEquals(1, snapshot.cpuSamples)
        assertEquals(1, snapshot.gpuSamples)
        val cpuNanosecondsPerOperation = requireNotNull(snapshot.cpuNanosecondsPerOperation)
        val gpuNanosecondsPerOperation = requireNotNull(snapshot.gpuNanosecondsPerOperation)
        assertEquals(10.0, cpuNanosecondsPerOperation, 0.0001)
        assertEquals(2_000_000.0 / 300_000.0, gpuNanosecondsPerOperation, 0.0001)
    }

    @Test
    fun profilesAreSeparatedByOperationAndWorkloadSizeBucket() {
        AdaptiveComputePerformanceMonitor.resetForTests()
        AdaptiveComputePerformanceMonitor.recordGpu(
            AdaptiveComputeOperation.LINEAR,
            operationCount = 300_000L,
            elapsedNanoseconds = 2_000_000L
        )
        AdaptiveComputePerformanceMonitor.recordGpu(
            AdaptiveComputeOperation.ADAMW,
            operationCount = 300_000L,
            elapsedNanoseconds = 1_000_000L
        )
        AdaptiveComputePerformanceMonitor.recordGpu(
            AdaptiveComputeOperation.LINEAR,
            operationCount = 3_000_000L,
            elapsedNanoseconds = 8_000_000L
        )

        assertEquals(
            1,
            AdaptiveComputePerformanceMonitor.snapshot(
                AdaptiveComputeOperation.LINEAR,
                operationCount = 300_000L
            ).gpuSamples
        )
        assertEquals(
            1,
            AdaptiveComputePerformanceMonitor.snapshot(
                AdaptiveComputeOperation.ADAMW,
                operationCount = 300_000L
            ).gpuSamples
        )
        assertEquals(
            1,
            AdaptiveComputePerformanceMonitor.snapshot(
                AdaptiveComputeOperation.LINEAR,
                operationCount = 3_000_000L
            ).gpuSamples
        )
    }

    @Test
    fun timingSamplesUseExponentialMovingAverage() {
        AdaptiveComputePerformanceMonitor.resetForTests()
        AdaptiveComputePerformanceMonitor.recordCpu(
            AdaptiveComputeOperation.ADAMW,
            operationCount = 100L,
            elapsedNanoseconds = 1_000L
        )
        AdaptiveComputePerformanceMonitor.recordCpu(
            AdaptiveComputeOperation.ADAMW,
            operationCount = 100L,
            elapsedNanoseconds = 2_000L
        )

        val snapshot = AdaptiveComputePerformanceMonitor.snapshot(
            AdaptiveComputeOperation.ADAMW,
            operationCount = 100L
        )
        assertEquals(2, snapshot.cpuSamples)
        val cpuNanosecondsPerOperation = requireNotNull(snapshot.cpuNanosecondsPerOperation)
        assertEquals(12.5, cpuNanosecondsPerOperation, 0.0001)
    }

    @Test
    fun ignoresInvalidTimingSamples() {
        AdaptiveComputePerformanceMonitor.resetForTests()
        AdaptiveComputePerformanceMonitor.recordCpu(
            AdaptiveComputeOperation.LINEAR,
            operationCount = 100L,
            elapsedNanoseconds = 0L
        )
        AdaptiveComputePerformanceMonitor.recordGpu(
            AdaptiveComputeOperation.LINEAR,
            operationCount = -1L,
            elapsedNanoseconds = 1_000L
        )

        val snapshot = AdaptiveComputePerformanceMonitor.snapshot(
            AdaptiveComputeOperation.LINEAR,
            operationCount = 100L
        )
        assertEquals(0, snapshot.cpuSamples)
        assertEquals(0, snapshot.gpuSamples)
    }
}
