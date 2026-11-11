import java.io.FileInputStream
import java.io.FileOutputStream
import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.google.devtools.ksp)
}

android {
    namespace = "com.argus.agi"
    compileSdk { version = release(37) }
    ndkVersion = "30.0.16248370"

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    val versionFile = rootProject.file("version.properties")
    check(versionFile.isFile) {
        "Missing version.properties. Restore the application version metadata before building."
    }

    val versionProperties = Properties().apply {
        versionFile.inputStream().use { stream -> load(stream) }
    }

    fun readVersion(key: String): Int =
        versionProperties.getProperty(key)?.toIntOrNull()
            ?: error("Missing or invalid $key in version.properties.")

    defaultConfig {
        applicationId = "com.argus.agi"
        minSdk = 33
        targetSdk = 36
        versionCode = readVersion("VERSION_BUILD")
        versionName = "${readVersion("VERSION_MAJOR")}.${readVersion("VERSION_MINOR")}.${readVersion("VERSION_PATCH")}"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }

        externalNativeBuild {
            cmake {
                cppFlags += listOf("-std=c++20")
            }
        }
    }

    val jksFile = rootProject.file("my-upload-key.jks")
    val p12File = rootProject.file("signing/argus-release.p12")
    // A release must never silently fall back to the Android debug identity.
    // Keep the expected upload-key path so release packaging fails closed when
    // the official signing material has not been provisioned.
    val defaultKeystore = when {
        jksFile.isFile -> jksFile
        p12File.isFile -> p12File
        else -> jksFile
    }

    val releaseKeystorePath = providers.environmentVariable("KEYSTORE_PATH")
        .orElse(defaultKeystore.absolutePath)
        .get()

    val defaultPass = "arguspassword"
    val defaultAlias = "upload"

    val releaseStorePassword = providers.environmentVariable("STORE_PASSWORD")
        .orElse(defaultPass)
        .get()
    val releaseKeyPassword = providers.environmentVariable("KEY_PASSWORD")
        .orElse(defaultPass)
        .get()
    val releaseKeyAlias = providers.environmentVariable("KEY_ALIAS")
        .orElse(defaultAlias)
        .get()

    signingConfigs {
        create("release") {
            storeFile = file(releaseKeystorePath)
            storePassword = releaseStorePassword
            keyAlias = releaseKeyAlias
            keyPassword = releaseKeyPassword
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
            signingConfig = signingConfigs.getByName("release")
        }
        debug {
            // Debug builds use AGP's machine-local debug signer.
        }
    }

    compileOptions {
            sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    buildFeatures {
        compose = true
    }

    testOptions {
        unitTests {
            isIncludeAndroidResources = true
        }
    }

    lint {
        checkTestSources = false
        ignoreTestSources = true
        abortOnError = true
    }

    dependenciesInfo {
        includeInApk = false
        includeInBundle = true
    }
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))

    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)
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
    implementation(libs.moshi)
    implementation(libs.okhttp)
    implementation(libs.jsoup)
    implementation(libs.pdfbox.android)
    implementation(libs.retrofit)

    testImplementation(platform(libs.androidx.compose.bom))
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

    ksp(libs.androidx.room.compiler)
}

// =====================================================================
// Copy APK ke root project supaya bisa diunduh dari AI Studio.
// HANYA 1 FILE APK DI ROOT: ARGUS.apk.
// File APK lama atau varian lain otomatis dibersihkan.
// =====================================================================
abstract class CopyApkTask : DefaultTask() {

    @get:InputFiles
    abstract val apk: ConfigurableFileCollection

    @get:OutputFile
    abstract val target: RegularFileProperty

    @TaskAction
    fun copy() {
        val dst = target.get().asFile
        val rootDir = dst.parentFile ?: return
        rootDir.mkdirs()

        // 1) Hapus semua APK lain di root project (mis. ARGUS-release.apk, ARGUS-debug.apk)
        //    sehingga HANYA ADA 1 FILE APK di root: ARGUS.apk
        rootDir.listFiles { f -> f.isFile && f.extension.equals("apk", true) }
            ?.filter { it.name != dst.name }
            ?.forEach { stale ->
                if (stale.delete()) {
                    logger.lifecycle("Dihapus APK lama: ${stale.name}")
                }
            }

        // 2) Salin APK release (atau fallback debug jika hanya debug yang dibuild)
        val src = apk.files.firstOrNull { it.isFile }
        if (src == null) {
            logger.warn("APK sumber belum tersedia untuk disalin.")
            return
        }

        try {
            src.copyTo(dst, overwrite = true)
            logger.lifecycle("Tersimpan 1 APK utama di root: ${dst.absolutePath} (${src.length() / 1024} KB, dari ${src.name})")
        } catch (e: Exception) {
            logger.warn("Gagal menyalin APK ke root: $e")
        }
    }
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
        var pat = props.getProperty("VERSION_PATCH", "7").toIntOrNull() ?: 7
        var bld = props.getProperty("VERSION_BUILD", "10007").toIntOrNull() ?: 10007

        pat += 1
        bld += 1

        props["VERSION_MAJOR"] = maj.toString()
        props["VERSION_MINOR"] = min.toString()
        props["VERSION_PATCH"] = pat.toString()
        props["VERSION_BUILD"] = bld.toString()
        FileOutputStream(file).use { props.store(it, "Auto-updated by ARGUS Build System") }
        logger.lifecycle("Version auto-updated untuk build berikutnya: $maj.$min.$pat (Build: $bld)")
    }
}

val copyArgusApkToRoot = tasks.register<CopyApkTask>("copyArgusApkToRoot") {
    // Prioritaskan release, fallback ke debug bila hanya debug yang dibuild
    apk.from(layout.buildDirectory.file("outputs/apk/release/app-release.apk"))
    apk.from(layout.buildDirectory.file("outputs/apk/debug/app-debug.apk"))
    target.set(layout.projectDirectory.file("../ARGUS.apk"))
    outputs.upToDateWhen { false }
}

val autoUpdateVersionTask = tasks.register<AutoUpdateVersionTask>("autoUpdateVersion") {
    versionFile.set(layout.projectDirectory.file("../version.properties"))
    outputs.upToDateWhen { false }
}

tasks.matching {
    it.name in setOf("assembleRelease", "assemble", "assembleDebug")
}.configureEach {
    finalizedBy(copyArgusApkToRoot)
}

tasks.matching {
    it.name in setOf("assembleRelease", "assemble")
}.configureEach {
    finalizedBy(autoUpdateVersionTask)
}
