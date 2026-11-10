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
