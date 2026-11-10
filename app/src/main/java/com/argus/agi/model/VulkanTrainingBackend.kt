package com.argus.agi.model

import android.util.Log

/**
 * Optional Vulkan Compute backend for batched dense operations and AdamW updates.
 *
 * Dense forward operations and selected output-head backward operations use Vulkan
 * when the runtime passes its numerical self-check and the workload is large enough.
 * All callers retain a numerically equivalent Kotlin fallback.
 */
internal object VulkanTrainingBackend {

    private const val TAG = "VulkanTrainingBackend"
    private const val MIN_GPU_UPDATE_ELEMENTS = 49_152L
    private const val MIN_LINEAR_OPERATION_COUNT = 262_144L

    @Volatile
    private var disabledAfterRuntimeFailure = false

    @Volatile
    private var _currentBackend = "Kotlin CPU"

    val currentBackend: String
        get() = _currentBackend

    private val nativeReady: Boolean = try {
        System.loadLibrary("argus_training")
        val initialized = nativeInitialize()
        val verified = initialized && runCatching {
            verifyNativeKernels()
        }.getOrDefault(false)

        if (initialized && !verified) {
            runCatching { nativeDestroy() }
                .onFailure { error ->
                    Log.w(TAG, "Failed to release Vulkan after its startup self-check failed.", error)
                }
            Log.e(TAG, "Vulkan numerical self-check failed; keeping the Kotlin CPU backend.")
        } else if (verified) {
            Log.i(TAG, "Vulkan Compute initialized and passed linear/AdamW numerical self-checks.")
        } else {
            Log.i(TAG, "Vulkan Compute is unavailable; Kotlin CPU backend will be used.")
        }
        verified
    } catch (error: UnsatisfiedLinkError) {
        Log.i(TAG, "Native Vulkan library or Vulkan loader is unavailable.", error)
        false
    } catch (error: SecurityException) {
        Log.w(TAG, "Native Vulkan library loading was blocked.", error)
        false
    } catch (error: RuntimeException) {
        Log.w(TAG, "Native Vulkan initialization failed.", error)
        false
    }

    private fun verifyNativeKernels(): Boolean {
        val input = floatArrayOf(
            1f, 2f, 3f,
            4f, 5f, 6f
        )
        val normalWeights = floatArrayOf(
            1f, 2f,
            3f, 4f,
            5f, 6f
        )
        val normalOutput = FloatArray(4)
        if (!nativeApplyLinear(
                input,
                normalWeights,
                normalOutput,
                2,
                3,
                2,
                false
            )) {
            return false
        }

        val expectedOutput = floatArrayOf(22f, 28f, 49f, 64f)
        if (!approximatelyEqual(normalOutput, expectedOutput, 1e-4f)) {
            return false
        }

        val transposedWeights = floatArrayOf(
            1f, 3f, 5f,
            2f, 4f, 6f
        )
        val transposedOutput = FloatArray(4)
        if (!nativeApplyLinear(
                input,
                transposedWeights,
                transposedOutput,
                2,
                3,
                2,
                true
            ) ||
            !approximatelyEqual(transposedOutput, expectedOutput, 1e-4f)) {
            return false
        }

        val parameters = floatArrayOf(1f, 2f)
        val gradients = floatArrayOf(0.1f, -0.2f)
        val firstMoments = FloatArray(2)
        val secondMoments = FloatArray(2)
        val adamwUpdated = nativeApplyAdamWUpdate(
            parameterGroups = arrayOf(parameters),
            gradientGroups = arrayOf(gradients),
            firstMomentGroups = arrayOf(firstMoments),
            secondMomentGroups = arrayOf(secondMoments),
            weightDecayByGroup = floatArrayOf(0f),
            learningRate = 0.01f,
            beta1 = 0.9f,
            beta2 = 0.999f,
            biasCorrection1 = 0.1f,
            biasCorrection2 = 0.001f,
            epsilon = 1e-8f,
            gradientClipFactor = 1f
        )

        return adamwUpdated &&
            approximatelyEqual(parameters, floatArrayOf(0.99f, 2.01f), 1e-4f) &&
            approximatelyEqual(firstMoments, floatArrayOf(0.01f, -0.02f), 1e-5f) &&
            approximatelyEqual(secondMoments, floatArrayOf(0.00001f, 0.00004f), 1e-7f)
    }

    private fun approximatelyEqual(
        actual: FloatArray,
        expected: FloatArray,
        tolerance: Float
    ): Boolean {
        if (actual.size != expected.size) return false
        for (index in actual.indices) {
            if (!actual[index].isFinite() ||
                kotlin.math.abs(actual[index] - expected[index]) > tolerance) {
                return false
            }
        }
        return true
    }

    @Synchronized
    fun tryApplyLinear(
        inputRows: Array<FloatArray>,
        weights: FloatArray,
        outputRows: Array<FloatArray>,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean = false
    ): Boolean {
        require(inputRows.isNotEmpty()) { "Linear input must contain at least one row." }
        require(inputColumns > 0 && outputColumns > 0) {
            "Linear dimensions must be positive."
        }
        require(outputRows.size == inputRows.size) {
            "Linear input and output row counts must match."
        }
        val requiredWeightElements = inputColumns.toLong() * outputColumns.toLong()
        require(requiredWeightElements <= Int.MAX_VALUE.toLong() &&
            weights.size >= requiredWeightElements.toInt()) {
            "Linear weight tensor is smaller than its dimensions."
        }
        // A tensor may reserve extra trailing rows (e.g. wHead's vocab capacity).
        // Only the active K×N prefix is copied to the native dispatch buffer.
        for (index in inputRows.indices) {
            require(inputRows[index].size == inputColumns) {
                "Linear input row $index has an unexpected width."
            }
            require(outputRows[index].size == outputColumns) {
                "Linear output row $index has an unexpected width."
            }
        }

        val operationCount = inputRows.size.toLong() *
            inputColumns.toLong() *
            outputColumns.toLong()
        if (!nativeReady ||
            disabledAfterRuntimeFailure ||
            operationCount < MIN_LINEAR_OPERATION_COUNT) {
            return false
        }

        val inputElementCountLong = inputRows.size.toLong() * inputColumns.toLong()
        val outputElementCountLong = outputRows.size.toLong() * outputColumns.toLong()
        if (inputElementCountLong > Int.MAX_VALUE.toLong() ||
            outputElementCountLong > Int.MAX_VALUE.toLong()) {
            return false
        }

        val flattenedInput: FloatArray
        val flattenedWeights: FloatArray
        val flattenedOutput: FloatArray
        try {
            flattenedInput = FloatArray(inputElementCountLong.toInt())
            flattenedWeights = if (weights.size == requiredWeightElements.toInt()) {
                weights
            } else {
                weights.copyOf(requiredWeightElements.toInt())
            }
            flattenedOutput = FloatArray(outputElementCountLong.toInt())
        } catch (error: OutOfMemoryError) {
            Log.w(TAG, "Not enough heap memory to stage a Vulkan linear operation.", error)
            return false
        }

        for (row in inputRows.indices) {
            System.arraycopy(
                inputRows[row],
                0,
                flattenedInput,
                row * inputColumns,
                inputColumns
            )
        }

        val updated = try {
            nativeApplyLinear(
                input = flattenedInput,
                weights = flattenedWeights,
                output = flattenedOutput,
                rows = inputRows.size,
                inputColumns = inputColumns,
                outputColumns = outputColumns,
                transposeWeights = transposeWeights
            )
        } catch (error: UnsatisfiedLinkError) {
            Log.e(TAG, "Vulkan linear entry point is unavailable; disabling the backend.", error)
            false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Vulkan linear dispatch failed; disabling the backend.", error)
            false
        }

        if (!updated) {
            disabledAfterRuntimeFailure = true
            _currentBackend = "Kotlin CPU"
            runCatching { nativeDestroy() }
                .onFailure { error ->
                    Log.w(TAG, "Failed to release Vulkan after a linear dispatch failure.", error)
                }
            Log.w(TAG, "Vulkan linear dispatch was not completed; using the Kotlin CPU path.")
            return false
        }

        for (row in outputRows.indices) {
            System.arraycopy(
                flattenedOutput,
                row * outputColumns,
                outputRows[row],
                0,
                outputColumns
            )
        }
        _currentBackend = "Vulkan Compute (Linear)"
        return true
    }

    @Synchronized
    fun tryApplyAdamWUpdate(
        parameters: Array<FloatArray>,
        gradients: Array<FloatArray>,
        firstMoments: Array<FloatArray>,
        secondMoments: Array<FloatArray>,
        weightDecayByGroup: FloatArray,
        learningRate: Float,
        beta1: Float,
        beta2: Float,
        biasCorrection1: Float,
        biasCorrection2: Float,
        epsilon: Float,
        gradientClipFactor: Float
    ): Boolean {
        require(parameters.isNotEmpty()) { "AdamW requires at least one parameter group." }
        require(parameters.size == gradients.size) {
            "Parameter and gradient group counts must match."
        }
        require(parameters.size == firstMoments.size) {
            "Parameter and first-moment group counts must match."
        }
        require(parameters.size == secondMoments.size) {
            "Parameter and second-moment group counts must match."
        }
        require(parameters.size == weightDecayByGroup.size) {
            "Weight-decay configuration must match the parameter groups."
        }

        var totalElements = 0L
        for (index in parameters.indices) {
            val groupSize = minOf(
                parameters[index].size,
                gradients[index].size,
                firstMoments[index].size,
                secondMoments[index].size
            )
            totalElements += groupSize.toLong()
        }

        if (!nativeReady ||
            disabledAfterRuntimeFailure ||
            totalElements < MIN_GPU_UPDATE_ELEMENTS) {
            _currentBackend = "Kotlin CPU"
            return false
        }

        val updated = try {
            nativeApplyAdamWUpdate(
                parameterGroups = parameters,
                gradientGroups = gradients,
                firstMomentGroups = firstMoments,
                secondMomentGroups = secondMoments,
                weightDecayByGroup = weightDecayByGroup,
                learningRate = learningRate,
                beta1 = beta1,
                beta2 = beta2,
                biasCorrection1 = biasCorrection1,
                biasCorrection2 = biasCorrection2,
                epsilon = epsilon,
                gradientClipFactor = gradientClipFactor
            )
        } catch (error: UnsatisfiedLinkError) {
            Log.e(TAG, "Vulkan optimizer entry point is unavailable; disabling the backend.", error)
            false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Vulkan optimizer dispatch failed; disabling the backend.", error)
            false
        }

        if (updated) {
            _currentBackend = "Vulkan Compute (AdamW)"
            return true
        }

        disabledAfterRuntimeFailure = true
        _currentBackend = "Kotlin CPU"
        runCatching { nativeDestroy() }
            .onFailure { error ->
                Log.w(TAG, "Failed to release the Vulkan runtime after a dispatch failure.", error)
            }
        Log.w(TAG, "Vulkan AdamW dispatch was not completed; using the Kotlin CPU optimizer.")
        return false
    }

    private external fun nativeInitialize(): Boolean

    private external fun nativeApplyLinear(
        input: FloatArray,
        weights: FloatArray,
        output: FloatArray,
        rows: Int,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean
    ): Boolean

    private external fun nativeApplyAdamWUpdate(
        parameterGroups: Array<FloatArray>,
        gradientGroups: Array<FloatArray>,
        firstMomentGroups: Array<FloatArray>,
        secondMomentGroups: Array<FloatArray>,
        weightDecayByGroup: FloatArray,
        learningRate: Float,
        beta1: Float,
        beta2: Float,
        biasCorrection1: Float,
        biasCorrection2: Float,
        epsilon: Float,
        gradientClipFactor: Float
    ): Boolean

    private external fun nativeDestroy()
}
