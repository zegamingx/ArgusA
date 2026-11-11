package com.argus.agi.sensor

import android.app.ActivityManager
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import java.io.File

data class SystemHardwareInfo(
    val socManufacturer: String,
    val socModel: String,
    val hardwareBoard: String,
    val cpuCores: Int,
    val totalRamGb: Int,
    val availRamGb: Float,
    val maxHeapMb: Long,
    val vulkanSupported: Boolean,
    val vulkanVersionCode: Int,
    val vulkanVersionName: String,
    val npuApuDetected: String,
    val primaryAccelerator: String,
    val isSnapdragon: Boolean,
    val isMediaTek: Boolean,
    val isExynos: Boolean,
    val isTensor: Boolean
)

class HardwareProfile(private val context: Context) {

    fun getHardwareInfo(): SystemHardwareInfo {
        val pm = context.packageManager

        // ---------- 1. SoC Detection (Multi-Layer Fallback) ----------
        val socManufacturer = resolveSocManufacturer()
        val socModel = resolveSocModel()
        val board = Build.HARDWARE ?: "Generic"

        val isSnapdragon = socManufacturer.contains("Qualcomm", true) ||
                board.contains("qcom", true) ||
                board.contains("snapdragon", true) ||
                socModel.contains("sm", true) // SM8550, SM8650, dll.

        val isMediaTek = socManufacturer.contains("MediaTek", true) ||
                board.contains("mt6", true) ||
                board.contains("mt8", true) ||
                (board.startsWith("mt", true) && board.length > 2)

        val isExynos = socManufacturer.contains("Samsung", true) ||
                socModel.contains("exynos", true) ||
                board.contains("exynos", true)

        val isTensor = socManufacturer.contains("Google", true) ||
                socModel.contains("tensor", true) ||
                board.contains("gs", true) // gs101, gs201, dll.

        // ---------- 2. Vulkan Detection (Version-Aware) ----------
        val vulkanInfo = resolveVulkanInfo(pm)

        // ---------- 3. NPU / APU Detection (Conservative) ----------
        val npuApuDetected = resolveNpuInfo(
            isSnapdragon, isMediaTek, isExynos, isTensor, socModel
        )

        // ---------- 4. Primary Accelerator Decision ----------
        val primaryAccelerator = resolvePrimaryAccelerator(vulkanInfo)

        // ---------- 5. Memory Detection (Accurate) ----------
        val memInfo = ActivityManager.MemoryInfo()
        val actManager = context.getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
        actManager?.getMemoryInfo(memInfo)

        val totalRamGb = if (memInfo.totalMem > 0) {
            (memInfo.totalMem / (1024.0 * 1024.0 * 1024.0)).toInt().coerceAtLeast(1)
        } else {
            8 // Fallback aman
        }

        // availMem adalah memori sistem yang bebas; kurangi threshold agar lebih realistis untuk aplikasi
        val availableForApp = (memInfo.availMem - memInfo.threshold).coerceAtLeast(0L)
        val availRamGb = availableForApp / (1024.0f * 1024.0f * 1024.0f)

        val maxHeapMb = Runtime.getRuntime().maxMemory() / (1024 * 1024)
        val cpuCores = Runtime.getRuntime().availableProcessors()

        return SystemHardwareInfo(
            socManufacturer = socManufacturer.ifBlank { "ARM Architecture" },
            socModel = socModel.ifBlank { board },
            hardwareBoard = board,
            cpuCores = cpuCores,
            totalRamGb = totalRamGb,
            availRamGb = availRamGb,
            maxHeapMb = maxHeapMb,
            vulkanSupported = vulkanInfo.supported,
            vulkanVersionCode = vulkanInfo.versionCode,
            vulkanVersionName = vulkanInfo.versionName,
            npuApuDetected = npuApuDetected,
            primaryAccelerator = primaryAccelerator,
            isSnapdragon = isSnapdragon,
            isMediaTek = isMediaTek,
            isExynos = isExynos,
            isTensor = isTensor
        )
    }

    // =====================================================================
    // 1. SoC Detection
    // =====================================================================

    private fun resolveSocManufacturer(): String {
        // Prioritas 1: API 31+
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val m = Build.SOC_MANUFACTURER
            if (!m.isNullOrBlank() && m != "unknown") return m
        }

        // Prioritas 2: ro.soc.manufacturer (beberapa vendor menyediakannya)
        val propMfr = getSystemProperty("ro.soc.manufacturer")
        if (propMfr.isNotBlank() && propMfr != "unknown") return propMfr

        // Prioritas 3: /proc/cpuinfo (GKI 2.0 ke bawah)
        val cpuInfoMfr = detectManufacturerFromCpuInfo()
        if (cpuInfoMfr.isNotBlank()) return cpuInfoMfr

        // Prioritas 4: Build.MANUFACTURER sebagai fallback terakhir
        return Build.MANUFACTURER ?: "ARM Architecture"
    }

    private fun resolveSocModel(): String {
        // Prioritas 1: API 31+
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val model = Build.SOC_MODEL
            if (!model.isNullOrBlank() && model != "unknown") return model
        }

        // Prioritas 2: ro.soc.model (wajib untuk GKI 2.0+)
        val propModel = getSystemProperty("ro.soc.model")
        if (propModel.isNotBlank() && propModel != "unknown") return propModel

        // Prioritas 3: /proc/cpuinfo Hardware line
        val cpuInfoModel = detectModelFromCpuInfo()
        if (cpuInfoModel.isNotBlank()) return cpuInfoModel

        // Prioritas 4: Build.HARDWARE / Build.BOARD
        return (Build.HARDWARE ?: Build.BOARD) ?: "Generic"
    }

    private fun getSystemProperty(key: String): String {
        return try {
            val clazz = Class.forName("android.os.SystemProperties")
            val get = clazz.getMethod("get", String::class.java, String::class.java)
            (get.invoke(null, key, "") as? String) ?: ""
        } catch (_: Exception) {
            ""
        }
    }

    private fun detectManufacturerFromCpuInfo(): String {
        return try {
            val file = File("/proc/cpuinfo")
            if (!file.exists()) return ""

            val hardwareValue = file.readLines()
                .firstOrNull { it.startsWith("Hardware", true) }
                ?.substringAfter(":")?.trim()
                ?: return ""

            when {
                hardwareValue.contains("qcom", true) ||
                        hardwareValue.contains("Qualcomm", true) -> "Qualcomm"
                hardwareValue.contains("MT", true) ||
                        hardwareValue.contains("MediaTek", true) -> "MediaTek"
                hardwareValue.contains("Exynos", true) ||
                        hardwareValue.contains("Samsung", true) -> "Samsung"
                hardwareValue.contains("Tensor", true) ||
                        hardwareValue.contains("Google", true) -> "Google"
                else -> hardwareValue
            }
        } catch (_: Exception) {
            ""
        }
    }

    private fun detectModelFromCpuInfo(): String {
        return try {
            val file = File("/proc/cpuinfo")
            if (!file.exists()) return ""

            file.readLines()
                .firstOrNull { it.startsWith("Hardware", true) }
                ?.substringAfter(":")?.trim()
                ?: ""
        } catch (_: Exception) {
            ""
        }
    }

    // =====================================================================
    // 2. Vulkan Detection
    // =====================================================================

    private data class VulkanInfo(
        val supported: Boolean,
        val versionCode: Int,
        val versionName: String
    )

    private fun resolveVulkanInfo(pm: PackageManager): VulkanInfo {
        if (!pm.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_VERSION)) {
            return VulkanInfo(false, 0, "None")
        }

        // Cek versi tertinggi terlebih dahulu menggunakan hasSystemFeature(name, version)
        // Referensi encoding: major (31-22), minor (21-12), patch (11-0)
        val versionName = when {
            pm.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_VERSION, 0x404000) -> "1.4"
            pm.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_VERSION, 0x403000) -> "1.3"
            pm.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_VERSION, 0x402000) -> "1.2"
            pm.hasSystemFeature(PackageManager.FEATURE_VULKAN_HARDWARE_VERSION, 0x401000) -> "1.1"
            else -> "1.0"
        }

        // Ambil kode versi aktual dari systemAvailableFeatures
        val versionCode = pm.systemAvailableFeatures
            .firstOrNull { it.name == PackageManager.FEATURE_VULKAN_HARDWARE_VERSION }
            ?.version
            ?: 0

        return VulkanInfo(true, versionCode, versionName)
    }

    // =====================================================================
    // 3. NPU / APU Detection
    // =====================================================================

    private fun resolveNpuInfo(
        isSnapdragon: Boolean,
        isMediaTek: Boolean,
        isExynos: Boolean,
        isTensor: Boolean,
        socModel: String
    ): String {
        // Tidak semua chipset dari vendor yang sama memiliki NPU yang dapat diakses aplikasi.
        // Gunakan lookup berbasis model untuk akurasi lebih baik.

        return when {
            // Google Tensor: TPU (sejak Tensor G1)
            isTensor && (socModel.contains("tensor", true) ||
                    socModel.contains("gs101", true) ||
                    socModel.contains("gs201", true) ||
                    socModel.contains("gs301", true)) -> "Google Tensor TPU"

            // Samsung Exynos: NPU (banyak seri 9000/8000/7000 modern)
            isExynos && (socModel.contains("exynos", true) ||
                    socModel.contains("e9", true) ||
                    socModel.contains("e8", true)) -> "Samsung Exynos NPU"

            // MediaTek: APU (Dimensity 700 ke atas umumnya memiliki APU)
            isMediaTek && (socModel.contains("dimensity", true) ||
                    socModel.contains("mt6", true) ||
                    socModel.contains("mt8", true)) -> "MediaTek NeuroPilot APU"

            // Qualcomm Snapdragon: Hexagon NPU (seri 700/800/8xx modern)
            isSnapdragon && (socModel.contains("sm7", true) ||
                    socModel.contains("sm8", true) ||
                    socModel.contains("snapdragon", true) ||
                    socModel.contains("sdm", true)) -> "Qualcomm Hexagon NPU"

            // Deteksi generik: jika tidak yakin, jangan klaim NPU secara spesifik
            else -> "Not detected"
        }
    }

    // =====================================================================
    // 4. Primary Accelerator
    // =====================================================================

    private fun resolvePrimaryAccelerator(vulkan: VulkanInfo): String =
        if (vulkan.supported) {
            "Vulkan ${vulkan.versionName} GPU feature advertised (runtime verification required)"
        } else {
            "Vulkan GPU feature not advertised; Kotlin CPU fallback"
        }
}