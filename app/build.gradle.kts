import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.google.devtools.ksp)
}

android {
    namespace = "com.argus.agi"
    compileSdk { version = release(37) }

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
    }

    val releaseKeystorePath = providers.environmentVariable("KEYSTORE_PATH")
        .orElse(rootProject.file("signing/argus-release.p12").absolutePath)
        .get()
    val releaseStorePassword = providers.environmentVariable("STORE_PASSWORD")
        .orElse("")
        .get()
    val releaseKeyPassword = providers.environmentVariable("KEY_PASSWORD")
        .orElse("")
        .get()

    signingConfigs {
        create("release") {
            storeFile = file(releaseKeystorePath)
            storeType = "PKCS12"
            storePassword = releaseStorePassword
            keyAlias = "upload"
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
    implementation(libs.moshi.kotlin)
    implementation(libs.okhttp)
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
    ksp(libs.moshi.kotlin.codegen)
}
