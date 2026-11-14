package com.argus.agi.model

import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Optional Vulkan Compute backend for batched dense operations and AdamW updates.
 *
 * Dense forward and backward operations use Vulkan when the runtime passes its
 * numerical self-check and the workload is large enough.
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

    @Volatile
    private var fp16RuntimeReady = false

    @Volatile
    private var fp16DisabledAfterRuntimeFailure = false

    @Volatile
    private var nativeReady = false

    @Volatile
    private var initializationAttempted = false

    private val _runtimeStatus = MutableStateFlow("Kotlin CPU — Vulkan disabled by default")
    val runtimeStatus: StateFlow<String> = _runtimeStatus.asStateFlow()

    private val _isVulkanEnabledForCurrentSession = MutableStateFlow(false)
    val isVulkanEnabledForCurrentSession: StateFlow<Boolean> =
        _isVulkanEnabledForCurrentSession.asStateFlow()

    private val _isInitializingRuntime = MutableStateFlow(false)
    val isInitializingRuntime: StateFlow<Boolean> = _isInitializingRuntime.asStateFlow()

    private val _canInitializeForCurrentSession = MutableStateFlow(true)
    val canInitializeForCurrentSession: StateFlow<Boolean> =
        _canInitializeForCurrentSession.asStateFlow()

    internal val hasAttemptedInitialization: Boolean
        get() = initializationAttempted

    val currentBackend: String
        get() = _currentBackend

    /**
     * Status inspection is deliberately side-effect free. Vulkan must never be loaded
     * as a side effect of rendering a screen or building the foreground notification.
     */
    val isRuntimeAvailable: Boolean
        get() = nativeReady &&
            _isVulkanEnabledForCurrentSession.value &&
            !disabledAfterRuntimeFailure

    val isFp16ComputeAvailable: Boolean
        get() = isRuntimeAvailable && fp16RuntimeReady && !fp16DisabledAfterRuntimeFailure

    /**
     * Vulkan is opt-in for the current process only. The default path remains Kotlin CPU,
     * so a device-specific Vulkan driver failure cannot prevent the application from opening.
     */
    @Synchronized
    fun enableForCurrentSession(): Boolean {
        if (isRuntimeAvailable) return true
        if (initializationAttempted || !_canInitializeForCurrentSession.value) return false

        initializationAttempted = true
        _canInitializeForCurrentSession.value = false
        _isInitializingRuntime.value = true
        _runtimeStatus.value = "Checking the optional Vulkan runtime…"

        return try {
            System.loadLibrary("argus_training")
            val initialized = nativeInitialize()
            val verified = initialized && runCatching {
                verifyNativeKernels()
            }.getOrDefault(false)

            if (verified && runCatching { nativeSupportsFp16() }.getOrDefault(false)) {
                fp16RuntimeReady = runCatching {
                    verifyNativeFp16Kernel()
                }.getOrDefault(false)
                if (!fp16RuntimeReady) {
                    Log.w(
                        TAG,
                        "Vulkan FP16 kernel self-check failed; the verified FP32 Vulkan path remains available."
                    )
                }
            }

            if (initialized && !verified) {
                runCatching { nativeDestroy() }
                    .onFailure { error ->
                        Log.w(TAG, "Failed to release Vulkan after its startup self-check failed.", error)
                    }
            }

            nativeReady = verified
            _isVulkanEnabledForCurrentSession.value = verified
            if (verified) {
                _currentBackend = "Kotlin CPU"
                _runtimeStatus.value = "Vulkan verified — enabled for this session"
                Log.i(
                    TAG,
                    "Vulkan Compute initialized and passed linear, batched-linear, and AdamW numerical self-checks."
                )
            } else {
                disabledAfterRuntimeFailure = true
                _currentBackend = "Kotlin CPU"
                _runtimeStatus.value = "Kotlin CPU fallback — Vulkan self-check was unsuccessful"
                Log.w(TAG, "Vulkan is unavailable or failed verification; Kotlin CPU remains active.")
            }
            verified
        } catch (error: UnsatisfiedLinkError) {
            nativeReady = false
            disabledAfterRuntimeFailure = true
            _isVulkanEnabledForCurrentSession.value = false
            _currentBackend = "Kotlin CPU"
            _runtimeStatus.value = "Kotlin CPU fallback — native Vulkan library unavailable"
            Log.i(TAG, "Native Vulkan library or Vulkan loader is unavailable.", error)
            false
        } catch (error: SecurityException) {
            nativeReady = false
            disabledAfterRuntimeFailure = true
            _isVulkanEnabledForCurrentSession.value = false
            _currentBackend = "Kotlin CPU"
            _runtimeStatus.value = "Kotlin CPU fallback — native Vulkan access was blocked"
            Log.w(TAG, "Native Vulkan library loading was blocked.", error)
            false
        } catch (error: RuntimeException) {
            nativeReady = false
            disabledAfterRuntimeFailure = true
            _isVulkanEnabledForCurrentSession.value = false
            _currentBackend = "Kotlin CPU"
            _runtimeStatus.value = "Kotlin CPU fallback — Vulkan initialization failed"
            Log.w(TAG, "Native Vulkan initialization failed.", error)
            false
        } finally {
            _isInitializingRuntime.value = false
        }
    }

    @Synchronized
    fun disableForCurrentSession() {
        val wasInitialized = nativeReady
        _isVulkanEnabledForCurrentSession.value = false

        if (wasInitialized) {
            runCatching { nativeDestroy() }
                .onFailure { error ->
                    Log.w(TAG, "Failed to release the Vulkan runtime after it was disabled.", error)
                }
            nativeReady = false
        }

        fp16RuntimeReady = false
        fp16DisabledAfterRuntimeFailure = false
        _currentBackend = "Kotlin CPU"
        _runtimeStatus.value = "Kotlin CPU — Vulkan disabled for this session"

        if (wasInitialized && !disabledAfterRuntimeFailure) {
            initializationAttempted = false
            _canInitializeForCurrentSession.value = true
        }
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
                1,
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
                1,
                2,
                3,
                2,
                true
            ) ||
            !approximatelyEqual(transposedOutput, expectedOutput, 1e-4f)) {
            return false
        }

        if (!verifyBatchedLinearKernel()) {
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

    /**
     * Compares one non-square, multi-batch Vulkan dispatch with a Kotlin CPU reference.
     * 72 outputs deliberately cross the 64-invocation workgroup boundary.
     */
    private fun verifyBatchedLinearKernel(): Boolean {
        val batchCount = 3
        val rowsPerBatch = 3
        val inputColumns = 5
        val outputColumns = 8
        val rowCount = batchCount * rowsPerBatch
        val input = FloatArray(rowCount * inputColumns) { index ->
            ((index % 17) - 8) * 0.125f
        }
        val weightsPerBatch = inputColumns * outputColumns
        val weights = FloatArray(batchCount * weightsPerBatch) { index ->
            ((index % 13) - 6) * 0.0625f
        }
        val expected = FloatArray(rowCount * outputColumns)

        for (batchIndex in 0 until batchCount) {
            val batchInputOffset = batchIndex * rowsPerBatch * inputColumns
            val batchWeightOffset = batchIndex * weightsPerBatch
            for (row in 0 until rowsPerBatch) {
                val inputRowOffset = batchInputOffset + row * inputColumns
                val outputRowOffset = (batchIndex * rowsPerBatch + row) * outputColumns
                for (column in 0 until outputColumns) {
                    var sum = 0f
                    for (inner in 0 until inputColumns) {
                        val inputValue = input[inputRowOffset + inner]
                        val weightValue = weights[
                            batchWeightOffset + inner * outputColumns + column
                        ]
                        sum += inputValue * weightValue
                    }
                    expected[outputRowOffset + column] = sum
                }
            }
        }

        val actual = FloatArray(expected.size)
        return nativeApplyLinear(
            input,
            weights,
            actual,
            rows = rowCount,
            batchCount = batchCount,
            rowsPerBatch = rowsPerBatch,
            inputColumns = inputColumns,
            outputColumns = outputColumns,
            transposeWeights = false
        ) && approximatelyEqual(actual, expected, 1e-4f)
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

    private fun verifyNativeFp16Kernel(): Boolean {
        val input = floatArrayOf(
            0.5f, 1.0f, 1.5f,
            2.0f, 2.5f, 3.0f
        )
        val weights = Float16ParameterArray.fromFloatArray(
            floatArrayOf(
                0.5f, 1.0f,
                1.5f, 2.0f,
                2.5f, 3.0f
            )
        ).toShortArray()
        val output = FloatArray(4)
        val succeeded = nativeApplyFp16Linear(
            input = input,
            weights = weights,
            output = output,
            rows = 2,
            batchCount = 1,
            rowsPerBatch = 2,
            inputColumns = 3,
            outputColumns = 2,
            transposeWeights = false
        )
        return succeeded &&
            approximatelyEqual(output, floatArrayOf(5.5f, 7.0f, 12.25f, 16.0f), 1e-3f)
    }

    @Synchronized
    fun tryApplyFp16Linear(
        inputRows: Array<FloatArray>,
        weights: Float16ParameterArray,
        outputRows: Array<FloatArray>,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean = false
    ): Boolean = tryApplyFp16BatchedLinear(
        inputBatches = arrayOf(inputRows),
        weightBatches = arrayOf(weights.toShortArray()),
        outputBatches = arrayOf(outputRows),
        inputColumns = inputColumns,
        outputColumns = outputColumns,
        transposeWeights = transposeWeights
    )

    @Synchronized
    fun tryApplyFp16BatchedLinear(
        inputBatches: Array<Array<FloatArray>>,
        weightBatches: Array<ShortArray>,
        outputBatches: Array<Array<FloatArray>>,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean = false
    ): Boolean {
        require(inputBatches.isNotEmpty()) { "FP16 batched linear input must contain at least one batch." }
        require(inputColumns > 0 && outputColumns > 0) {
            "FP16 batched linear dimensions must be positive."
        }
        val batchCount = inputBatches.size
        require(weightBatches.size == batchCount && outputBatches.size == batchCount) {
            "FP16 batched linear inputs, weights, and outputs must have the same batch count."
        }
        val rowsPerBatch = inputBatches[0].size
        require(rowsPerBatch > 0) { "Every FP16 batched linear input must contain at least one row." }
        val requiredWeightElements = Math.multiplyExact(inputColumns, outputColumns)

        for (batchIndex in 0 until batchCount) {
            require(inputBatches[batchIndex].size == rowsPerBatch) {
                "FP16 batched linear inputs must have an identical row count."
            }
            require(outputBatches[batchIndex].size == rowsPerBatch) {
                "FP16 batched linear outputs must match the input row count."
            }
            // Some model tensors are allocated to their maximum capacity while a
            // training pass only uses an active prefix (for example, wHead when the
            // tokenizer vocabulary has not filled its configured capacity).
            // The staging copy below intentionally copies only this active matrix.
            if (weightBatches[batchIndex].size < requiredWeightElements) {
                return false
            }
            for (row in 0 until rowsPerBatch) {
                require(inputBatches[batchIndex][row].size == inputColumns) {
                    "FP16 input row $row in batch $batchIndex has an unexpected width."
                }
                require(outputBatches[batchIndex][row].size == outputColumns) {
                    "FP16 output row $row in batch $batchIndex has an unexpected width."
                }
            }
        }

        val totalRowsLong = batchCount.toLong() * rowsPerBatch.toLong()
        val inputElementsLong = totalRowsLong * inputColumns.toLong()
        val weightElementsLong = batchCount.toLong() * requiredWeightElements.toLong()
        val outputElementsLong = totalRowsLong * outputColumns.toLong()
        if (totalRowsLong > Int.MAX_VALUE.toLong() ||
            inputElementsLong > Int.MAX_VALUE.toLong() ||
            weightElementsLong > Int.MAX_VALUE.toLong() ||
            outputElementsLong > Int.MAX_VALUE.toLong()) {
            return false
        }

        val operationCount = totalRowsLong *
            inputColumns.toLong() *
            outputColumns.toLong()
        if (!isFp16ComputeAvailable ||
            operationCount < MIN_LINEAR_OPERATION_COUNT) {
            return false
        }

        val flattenedInput: FloatArray
        val flattenedWeights: ShortArray
        val flattenedOutput: FloatArray
        try {
            flattenedInput = FloatArray(inputElementsLong.toInt())
            flattenedWeights = ShortArray(weightElementsLong.toInt())
            flattenedOutput = FloatArray(outputElementsLong.toInt())
        } catch (error: OutOfMemoryError) {
            Log.w(TAG, "Not enough heap memory to stage a Vulkan FP16 linear operation.", error)
            return false
        }

        val inputBatchSize = rowsPerBatch * inputColumns
        val weightBatchSize = requiredWeightElements
        for (batchIndex in 0 until batchCount) {
            val inputOffset = batchIndex * inputBatchSize
            for (row in 0 until rowsPerBatch) {
                System.arraycopy(
                    inputBatches[batchIndex][row],
                    0,
                    flattenedInput,
                    inputOffset + row * inputColumns,
                    inputColumns
                )
            }
            System.arraycopy(
                weightBatches[batchIndex],
                0,
                flattenedWeights,
                batchIndex * weightBatchSize,
                weightBatchSize
            )
        }

        val updated = try {
            nativeApplyFp16Linear(
                input = flattenedInput,
                weights = flattenedWeights,
                output = flattenedOutput,
                rows = totalRowsLong.toInt(),
                batchCount = batchCount,
                rowsPerBatch = rowsPerBatch,
                inputColumns = inputColumns,
                outputColumns = outputColumns,
                transposeWeights = transposeWeights
            )
        } catch (error: UnsatisfiedLinkError) {
            Log.e(TAG, "Vulkan FP16 linear entry point is unavailable; disabling FP16 dispatch.", error)
            false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Vulkan FP16 linear dispatch failed; disabling FP16 dispatch.", error)
            false
        }

        if (!updated) {
            fp16DisabledAfterRuntimeFailure = true
            fp16RuntimeReady = false
            Log.w(TAG, "Vulkan FP16 dispatch was not completed; the verified FP32 Vulkan/CPU path remains available.")
            return false
        }

        val outputBatchSize = rowsPerBatch * outputColumns
        for (batchIndex in 0 until batchCount) {
            val outputBatchOffset = batchIndex * outputBatchSize
            for (row in 0 until rowsPerBatch) {
                System.arraycopy(
                    flattenedOutput,
                    outputBatchOffset + row * outputColumns,
                    outputBatches[batchIndex][row],
                    0,
                    outputColumns
                )
            }
        }
        _currentBackend = "Vulkan Compute (FP16 multiply / FP32 accumulation)"
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
    ): Boolean = tryApplyBatchedLinear(
        inputBatches = arrayOf(inputRows),
        weightBatches = arrayOf(weights),
        outputBatches = arrayOf(outputRows),
        inputColumns = inputColumns,
        outputColumns = outputColumns,
        transposeWeights = transposeWeights
    )

    /**
     * Runs independent per-batch dense products in one Vulkan dispatch. Each batch
     * has its own contiguous input matrix and weight tensor (one batch per attention head).
     */
    @Synchronized
    fun tryApplyBatchedLinear(
        inputBatches: Array<Array<FloatArray>>,
        weightBatches: Array<FloatArray>,
        outputBatches: Array<Array<FloatArray>>,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean = false
    ): Boolean {
        require(inputBatches.isNotEmpty()) { "Batched linear input must contain at least one batch." }
        require(inputColumns > 0 && outputColumns > 0) {
            "Batched linear dimensions must be positive."
        }
        val batchCount = inputBatches.size
        require(weightBatches.size == batchCount && outputBatches.size == batchCount) {
            "Batched linear inputs, weights, and outputs must have the same batch count."
        }
        val rowsPerBatch = inputBatches[0].size
        require(rowsPerBatch > 0) { "Every batched linear input must contain at least one row." }
        val requiredWeightElements = inputColumns.toLong() * outputColumns.toLong()
        require(requiredWeightElements <= Int.MAX_VALUE.toLong()) {
            "Batched linear weight tensor exceeds the supported array size."
        }

        for (batchIndex in 0 until batchCount) {
            require(inputBatches[batchIndex].size == rowsPerBatch) {
                "Batched linear inputs must have an identical row count."
            }
            require(outputBatches[batchIndex].size == rowsPerBatch) {
                "Batched linear outputs must match the input row count."
            }
            require(weightBatches[batchIndex].size >= requiredWeightElements.toInt()) {
                "Batched linear weight tensor $batchIndex is smaller than its dimensions."
            }
            for (row in 0 until rowsPerBatch) {
                require(inputBatches[batchIndex][row].size == inputColumns) {
                    "Batched linear input row $row in batch $batchIndex has an unexpected width."
                }
                require(outputBatches[batchIndex][row].size == outputColumns) {
                    "Batched linear output row $row in batch $batchIndex has an unexpected width."
                }
            }
        }

        val totalRowsLong = batchCount.toLong() * rowsPerBatch.toLong()
        val inputElementCountLong = totalRowsLong * inputColumns.toLong()
        val weightElementCountLong = batchCount.toLong() * requiredWeightElements
        val outputElementCountLong = totalRowsLong * outputColumns.toLong()
        if (totalRowsLong > Int.MAX_VALUE.toLong() ||
            inputElementCountLong > Int.MAX_VALUE.toLong() ||
            weightElementCountLong > Int.MAX_VALUE.toLong() ||
            outputElementCountLong > Int.MAX_VALUE.toLong()) {
            return false
        }

        val operationCount = totalRowsLong *
            inputColumns.toLong() *
            outputColumns.toLong()
        if (!isRuntimeAvailable ||
            disabledAfterRuntimeFailure ||
            operationCount < MIN_LINEAR_OPERATION_COUNT) {
            return false
        }

        val flattenedInput: FloatArray
        val flattenedWeights: FloatArray
        val flattenedOutput: FloatArray
        try {
            flattenedInput = FloatArray(inputElementCountLong.toInt())
            flattenedWeights = FloatArray(weightElementCountLong.toInt())
            flattenedOutput = FloatArray(outputElementCountLong.toInt())
        } catch (error: OutOfMemoryError) {
            Log.w(TAG, "Not enough heap memory to stage a Vulkan batched-linear operation.", error)
            return false
        }

        val weightBatchSize = requiredWeightElements.toInt()
        for (batchIndex in 0 until batchCount) {
            val inputBatchOffset = batchIndex * rowsPerBatch * inputColumns
            for (row in 0 until rowsPerBatch) {
                System.arraycopy(
                    inputBatches[batchIndex][row],
                    0,
                    flattenedInput,
                    inputBatchOffset + row * inputColumns,
                    inputColumns
                )
            }
            System.arraycopy(
                weightBatches[batchIndex],
                0,
                flattenedWeights,
                batchIndex * weightBatchSize,
                weightBatchSize
            )
        }

        val updated = try {
            nativeApplyLinear(
                input = flattenedInput,
                weights = flattenedWeights,
                output = flattenedOutput,
                rows = totalRowsLong.toInt(),
                batchCount = batchCount,
                rowsPerBatch = rowsPerBatch,
                inputColumns = inputColumns,
                outputColumns = outputColumns,
                transposeWeights = transposeWeights
            )
        } catch (error: UnsatisfiedLinkError) {
            Log.e(TAG, "Vulkan batched-linear entry point is unavailable; disabling the backend.", error)
            false
        } catch (error: RuntimeException) {
            Log.e(TAG, "Vulkan batched-linear dispatch failed; disabling the backend.", error)
            false
        }

        if (!updated) {
            disabledAfterRuntimeFailure = true
            _isVulkanEnabledForCurrentSession.value = false
            _runtimeStatus.value = "Kotlin CPU fallback — Vulkan linear dispatch failed"
            _currentBackend = "Kotlin CPU"
            runCatching { nativeDestroy() }
                .onFailure { error ->
                    Log.w(TAG, "Failed to release Vulkan after a batched-linear dispatch failure.", error)
                }
            Log.w(TAG, "Vulkan batched-linear dispatch was not completed; using the Kotlin CPU path.")
            return false
        }

        val outputBatchSize = rowsPerBatch * outputColumns
        for (batchIndex in 0 until batchCount) {
            val outputBatchOffset = batchIndex * outputBatchSize
            for (row in 0 until rowsPerBatch) {
                System.arraycopy(
                    flattenedOutput,
                    outputBatchOffset + row * outputColumns,
                    outputBatches[batchIndex][row],
                    0,
                    outputColumns
                )
            }
        }
        _currentBackend = "Vulkan Compute (Batched Linear)"
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

        if (!isRuntimeAvailable ||
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
        _isVulkanEnabledForCurrentSession.value = false
        _runtimeStatus.value = "Kotlin CPU fallback — Vulkan optimizer dispatch failed"
        _currentBackend = "Kotlin CPU"
        runCatching { nativeDestroy() }
            .onFailure { error ->
                Log.w(TAG, "Failed to release the Vulkan runtime after a dispatch failure.", error)
            }
        Log.w(TAG, "Vulkan AdamW dispatch was not completed; using the Kotlin CPU optimizer.")
        return false
    }

    private external fun nativeInitialize(): Boolean

    private external fun nativeSupportsFp16(): Boolean

    private external fun nativeApplyFp16Linear(
        input: FloatArray,
        weights: ShortArray,
        output: FloatArray,
        rows: Int,
        batchCount: Int,
        rowsPerBatch: Int,
        inputColumns: Int,
        outputColumns: Int,
        transposeWeights: Boolean
    ): Boolean

    private external fun nativeApplyLinear(
        input: FloatArray,
        weights: FloatArray,
        output: FloatArray,
        rows: Int,
        batchCount: Int,
        rowsPerBatch: Int,
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
