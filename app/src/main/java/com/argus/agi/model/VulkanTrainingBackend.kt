package com.argus.agi.model

import android.util.Log

/**
 * Optional Vulkan Compute backend for the fused AdamW parameter update.
 *
 * Forward propagation and backpropagation remain in the Kotlin engine. If the
 * Vulkan loader, GPU queue, shader pipeline, memory allocation, or dispatch is
 * unavailable, the caller must execute the numerically equivalent Kotlin path.
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
        nativeInitialize().also { initialized ->
            if (initialized) {
                Log.i(TAG, "Vulkan Compute optimizer initialized.")
            } else {
                Log.i(TAG, "Vulkan Compute is unavailable; Kotlin CPU optimizer will be used.")
            }
        }
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
        require(weights.size == inputColumns * outputColumns) {
            "Linear weight size does not match its dimensions."
        }
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

        val inputElementCount = inputRows.size * inputColumns
        val outputElementCount = outputRows.size * outputColumns
        val flattenedInput = FloatArray(inputElementCount)
        val flattenedOutput = FloatArray(outputElementCount)

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
                weights = weights,
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
