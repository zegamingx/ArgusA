import java.io.FileInputStream
import java.io.FileOutputStream
import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.google.devtools.ksp)
    alias(libs.plugins.secrets)
}

android {
    namespace = "com.argus.agi"
    compileSdk { version = release(36) { minorApiLevel = 1 } }

    val versionPropsFile = rootProject.file("version.properties")
    val versionProps = Properties()

    if (versionPropsFile.exists()) {
        FileInputStream(versionPropsFile).use { versionProps.load(it) }
    } else {
        versionProps["VERSION_MAJOR"] = "1"
        versionProps["VERSION_MINOR"] = "0"
        versionProps["VERSION_PATCH"] = "5"
        versionProps["VERSION_BUILD"] = "10005"
        FileOutputStream(versionPropsFile).use { versionProps.store(it, "Initial Version Properties") }
    }

    val currentMajor = versionProps.getProperty("VERSION_MAJOR", "1").toIntOrNull() ?: 1
    val currentMinor = versionProps.getProperty("VERSION_MINOR", "0").toIntOrNull() ?: 0
    val currentPatch = versionProps.getProperty("VERSION_PATCH", "4").toIntOrNull() ?: 4
    val currentBuild = versionProps.getProperty("VERSION_BUILD", "10004").toIntOrNull() ?: 10004

    val currentVersionCode = currentBuild
    val currentVersionName = "$currentMajor.$currentMinor.$currentPatch"

    defaultConfig {
        applicationId = "com.argus.agi"
        minSdk = 33
        targetSdk = 36
        versionCode = currentVersionCode
        versionName = currentVersionName

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    // Do not remove or modify these signingConfigs. They are necessary for building, installing,
    // and updating Android apps in AI Studio.
    signingConfigs {
        create("release") {
            val keystorePath = System.getenv("KEYSTORE_PATH") ?: "${rootDir}/my-upload-key.jks"
            storeFile = file(keystorePath)
            storePassword = System.getenv("STORE_PASSWORD")
            keyAlias = "upload"
            keyPassword = System.getenv("KEY_PASSWORD")
        }
        create("debugConfig") {
            storeFile = file("${rootDir}/debug.keystore")
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
        }
    }

    buildTypes {
        release {
            isCrunchPngs = false
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            // Do not remove or modify this signingConfig assignment. It is necessary for Android apps in
            // AI Studio.
            signingConfig = signingConfigs.getByName("release")
        }
        debug {
            // Do not remove or modify this signingConfig assignment. It is necessary for Android apps in
            // AI Studio.
            signingConfig = signingConfigs.getByName("debugConfig")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    buildFeatures {
        compose = true
        buildConfig = true
    }

    testOptions { unitTests { isIncludeAndroidResources = true } }

    dependenciesInfo {
        includeInApk = false
        includeInBundle = true
    }
}

// Configure the Secrets Gradle Plugin to use .env and .env.example files
// to match the convention used in Web projects.
secrets {
    propertiesFileName = ".env"
    defaultPropertiesFileName = ".env.example"
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))

    implementation(libs.accompanist.permissions)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)
    implementation(libs.androidx.compose.material.icons.core)
    implementation(libs.androidx.compose.material.icons.extended)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.navigation.compose)
    implementation(libs.androidx.room.ktx)
    implementation(libs.androidx.room.runtime)
    implementation(libs.coil.compose)
    implementation(libs.converter.moshi)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.logging.interceptor)
    implementation(libs.moshi.kotlin)
    implementation(libs.okhttp)
    implementation(libs.retrofit)

    testImplementation(libs.androidx.compose.ui.test.junit4)
    testImplementation(libs.androidx.core)
    testImplementation(libs.androidx.junit)
    testImplementation(libs.junit)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.robolectric)

    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.runner)

    debugImplementation(libs.androidx.compose.ui.test.manifest)
    debugImplementation(libs.androidx.compose.ui.tooling)

    "ksp"(libs.androidx.room.compiler)
    "ksp"(libs.moshi.kotlin.codegen)
}

// =====================================================================
// Copy APK ke root project supaya bisa diunduh dari AI Studio.
// Hanya 1 APK di root: ARGUS.apk. APK lain (hasil build lama) dihapus.
//
// PERBAIKAN:
// - Sebelumnya hanya menyalin release APK, dan hanya terpicu oleh
//   assembleRelease / assemble. Karena build system AI Studio memakai
//   `assembleDebug`, task ini TIDAK PERNAH jalan → ARGUS.apk tidak muncul.
// - Sekarang: terima release ATAU debug APK (release diprioritaskan),
//   dan terpicu oleh assembleRelease, assemble, dan assembleDebug.
// - Bump versi tetap HANYA di release (tidak spam versionCode saat debug).
// =====================================================================

abstract class CopyApkTask : DefaultTask() {

    @get:InputFiles
    abstract val apk: ConfigurableFileCollection

    @get:OutputFile
    abstract val target: RegularFileProperty

    @TaskAction
    fun copy() {
        // Prioritas: release dulu, lalu debug.
        val src = apk.files.firstOrNull { it.isFile }
        val dst = target.get().asFile
        val rootDir = dst.parentFile ?: run {
            logger.warn("Tidak bisa resolve root dir dari target: ${dst.absolutePath}")
            return
        }

        rootDir.mkdirs()

        // 1) SELALU bersihkan APK stale, terlepas dari src ada atau tidak.
        // Ini mencegah APK lama nyangkut saat path output berubah
        // (mis. AGP versi baru pakai nama file berbeda).
        var removedCount = 0
        var failedCount = 0
        rootDir.listFiles { f -> f.isFile && f.extension.equals("apk", true) }
            ?.filter { it.name != dst.name }
            ?.forEach { stale ->
                if (stale.delete()) {
                    removedCount++
                    logger.lifecycle("Removed stale APK: ${stale.name}")
                } else {
                    // Fallback: file mungkin sedang di-lock (di-download,
                    // di-scan antivirus, dibuka app lain, dll).
                    stale.deleteOnExit()
                    failedCount++
                    logger.warn(
                        "Gagal hapus ${stale.name} (mungkin sedang di-lock). " +
                            "Ditandai deleteOnExit — restart Gradle daemon bila tetap ada."
                    )
                }
            }
        if (removedCount > 0 || failedCount > 0) {
            logger.lifecycle(
                "Cleanup root APK selesai: $removedCount dihapus, $failedCount gagal."
            )
        }

        // 2) Copy APK baru.
        if (src == null) {
            logger.warn(
                "APK tidak ditemukan di: ${apk.files.joinToString { it.absolutePath }}. " +
                    "Skip copy — kemungkinan nama output berubah (cek build/outputs/apk/)."
            )
            return
        }

        try {
            src.copyTo(dst, overwrite = true)
            logger.lifecycle(
                "APK copied to: ${dst.absolutePath} " +
                    "(${src.length() / 1024} KB, sumber: ${src.name})"
            )
        } catch (e: Exception) {
            logger.warn("Gagal menyalin APK ke root (build tetap lanjut): $e")
        }
    }
}

val copyArgusApkToRoot = tasks.register<CopyApkTask>("copyArgusApkToRoot") {
    // Urutan penting: release dulu supaya firstOrNull() memilih release bila ada.
    apk.from(layout.buildDirectory.file("outputs/apk/release/app-release.apk"))
    apk.from(layout.buildDirectory.file("outputs/apk/debug/app-debug.apk"))
    target.set(layout.projectDirectory.file("../ARGUS.apk"))
    // Tidak pakai dependsOn("packageRelease") — biar tidak maksa build release
    // saat yang jalan cuma assembleDebug.
    outputs.upToDateWhen { false }
}

abstract class AutoUpdateVersionTask : DefaultTask() {

    @get:OutputFile
    abstract val versionFile: RegularFileProperty

    @TaskAction
    fun updateVersion() {
        val file = versionFile.get().asFile
        val props = Properties()
        if (file.exists()) {
            FileInputStream(file).use { props.load(it) }
        }
        val maj = props.getProperty("VERSION_MAJOR", "1").toIntOrNull() ?: 1
        val min = props.getProperty("VERSION_MINOR", "0").toIntOrNull() ?: 0
        var pat = props.getProperty("VERSION_PATCH", "4").toIntOrNull() ?: 4
        var bld = props.getProperty("VERSION_BUILD", "10004").toIntOrNull() ?: 10004

        pat += 1
        bld += 1

        props["VERSION_MAJOR"] = maj.toString()
        props["VERSION_MINOR"] = min.toString()
        props["VERSION_PATCH"] = pat.toString()
        props["VERSION_BUILD"] = bld.toString()
        FileOutputStream(file).use { props.store(it, "Auto-updated by ARGUS Build System") }
        logger.lifecycle("Version auto-updated for next build: $maj.$min.$pat (Build: $bld)")
    }
}

abstract class BumpMinorTask : DefaultTask() {

    @get:OutputFile
    abstract val versionFile: RegularFileProperty

    @TaskAction
    fun bump() {
        val file = versionFile.get().asFile
        val props = Properties()
        if (file.exists()) {
            FileInputStream(file).use { props.load(it) }
        }
        val maj = props.getProperty("VERSION_MAJOR", "1").toIntOrNull() ?: 1
        var min = props.getProperty("VERSION_MINOR", "0").toIntOrNull() ?: 0
        var bld = props.getProperty("VERSION_BUILD", "10004").toIntOrNull() ?: 10004

        min += 1
        bld += 1

        props["VERSION_MAJOR"] = maj.toString()
        props["VERSION_MINOR"] = min.toString()
        props["VERSION_PATCH"] = "0"
        props["VERSION_BUILD"] = bld.toString()
        FileOutputStream(file).use { props.store(it, "Auto-updated by ARGUS Build System") }
        logger.lifecycle("Version bumped to: $maj.$min.0 (Build: $bld)")
    }
}

abstract class BumpMajorTask : DefaultTask() {

    @get:OutputFile
    abstract val versionFile: RegularFileProperty

    @TaskAction
    fun bump() {
        val file = versionFile.get().asFile
        val props = Properties()
        if (file.exists()) {
            FileInputStream(file).use { props.load(it) }
        }
        var maj = props.getProperty("VERSION_MAJOR", "1").toIntOrNull() ?: 1
        var bld = props.getProperty("VERSION_BUILD", "10004").toIntOrNull() ?: 10004

        maj += 1
        bld += 1

        props["VERSION_MAJOR"] = maj.toString()
        props["VERSION_MINOR"] = "0"
        props["VERSION_PATCH"] = "0"
        props["VERSION_BUILD"] = bld.toString()
        FileOutputStream(file).use { props.store(it, "Auto-updated by ARGUS Build System") }
        logger.lifecycle("Version bumped to: $maj.0.0 (Build: $bld)")
    }
}

val autoUpdateVersionTask = tasks.register<AutoUpdateVersionTask>("autoUpdateVersion") {
    versionFile.set(layout.projectDirectory.file("../version.properties"))
    outputs.upToDateWhen { false }
}

tasks.register<BumpMinorTask>("bumpMinor") {
    versionFile.set(layout.projectDirectory.file("../version.properties"))
    outputs.upToDateWhen { false }
}

tasks.register<BumpMajorTask>("bumpMajor") {
    versionFile.set(layout.projectDirectory.file("../version.properties"))
    outputs.upToDateWhen { false }
}

// ---------------------------------------------------------------------
// Trigger copy APK di SEMUA varian yang mungkin dijalankan build system.
// `assembleDebug` ditambahkan supaya ARGUS.apk tetap muncul walau AI Studio
// hanya menjalankan debug build (kasus yang sebelumnya bikin APK "hilang").
// ---------------------------------------------------------------------
tasks.matching {
    it.name in setOf("assembleRelease", "assemble", "assembleDebug")
}.configureEach {
    finalizedBy(copyArgusApkToRoot)
}

// ---------------------------------------------------------------------
// Bump versi HANYA di release, supaya versionCode tidak meledak gara-gara
// build debug berulang-ulang.
// ---------------------------------------------------------------------
tasks.matching {
    it.name in setOf("assembleRelease", "assemble")
}.configureEach {
    finalizedBy(autoUpdateVersionTask)
}