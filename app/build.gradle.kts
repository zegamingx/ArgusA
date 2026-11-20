import java.io.FileInputStream
import java.io.FileOutputStream
import java.util.Properties
import org.gradle.api.tasks.testing.Test
import org.gradle.api.tasks.testing.logging.TestExceptionFormat
import org.gradle.api.tasks.testing.logging.TestLogEvent

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.google.devtools.ksp)
}

// Release variants are assembled unsigned during validation. The protected
// main-branch distribution workflow applies the stable Argus 2026 signer only
// after 16 KiB ZIP alignment and certificate verification.

android {
    namespace = "com.argus.agi"
    compileSdk { version = release(37) }
    buildToolsVersion = "37.0.0"
    ndkVersion = "30.0.16248370"

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "4.5.0"
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
        minSdk = 34
        targetSdk = 37
        versionCode = readVersion("VERSION_BUILD")
        versionName = "${readVersion("VERSION_MAJOR")}.${readVersion("VERSION_MINOR")}.${readVersion("VERSION_PATCH")}"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
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
        }
        debug {
            // Debug builds use AGP's machine-local debug signer.
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_21
        targetCompatibility = JavaVersion.VERSION_21
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

// Android Lint reads machine-local files such as local.properties, but AGP does
// not consistently model those files as lint task inputs. Reusing a cached
// PropertyEscape result can therefore analyse a different runner SDK path.
// Re-run only Lint tasks on every validation; keep compile/native/test tasks cacheable.
tasks.configureEach {
    if (name.startsWith("lint")) {
        outputs.cacheIf("Lint consumes machine-local Android SDK metadata.") { false }
        outputs.upToDateWhen { false }
    }
}

// Android local unit-test tasks run on the JUnit Platform. Vintage keeps the
// existing JUnit 4 Robolectric runners operational while Jupiter 6 is available
// for new tests and gradual migration.
tasks.withType<Test>().configureEach {
    useJUnitPlatform()
    // Robolectric downloads instrumented Android SDK jars outside Gradle's
    // repository resolver. Use Google's Maven Central mirror to avoid intermittent
    // missing checksum/POM responses from repo1.maven.org on CI.
    systemProperty(
        "robolectric.dependency.repo.id",
        "google-maven-central-mirror"
    )
    systemProperty(
        "robolectric.dependency.repo.url",
        "https://maven-central.storage-download.googleapis.com/maven2"
    )

    // JEP 472 (JDK 24+): explicitly opt in for Conscrypt native calls in JVM tests.
    if (Runtime.version().feature() >= 24) {
        jvmArgs("--enable-native-access=ALL-UNNAMED")
    }
    testLogging {
        events(
            TestLogEvent.STARTED,
            TestLogEvent.PASSED,
            TestLogEvent.SKIPPED,
            TestLogEvent.FAILED
        )
        exceptionFormat = TestExceptionFormat.FULL
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
    implementation(libs.androidx.room.ktx)
    implementation(libs.androidx.room.runtime)
    implementation(libs.coil.compose)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.okhttp)
    implementation(libs.jsoup)
    implementation(libs.pdfbox.android)

    testImplementation(platform(libs.androidx.compose.bom))
    testImplementation(platform(libs.junit.bom))
    testImplementation(libs.androidx.compose.ui.test.junit4)
    testImplementation(libs.androidx.core)
    testImplementation(libs.androidx.junit)
    testImplementation(libs.junit4)
    testImplementation(libs.org.json)
    testImplementation(libs.junit.jupiter)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.robolectric)
    testRuntimeOnly(libs.junit.platform.launcher)
    testRuntimeOnly(libs.junit.vintage.engine)

    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.core)
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.runner)

    debugImplementation(libs.androidx.compose.ui.test.manifest)
    debugImplementation(libs.androidx.compose.ui.tooling)

    ksp(libs.androidx.room.compiler)
}

// Artifact export and version changes are explicit operations. A normal assemble
// must not mutate project files or delete APKs outside the build directory.
abstract class CopyApkTask : DefaultTask() {

    @get:InputFiles
    abstract val apk: ConfigurableFileCollection

    @get:OutputFile
    abstract val target: RegularFileProperty

    @TaskAction
    fun copy() {
        val src = apk.files.firstOrNull { it.isFile }
            ?: throw IllegalStateException("The requested APK artifact was not produced.")
        val dst = target.get().asFile
        val parent = dst.parentFile
            ?: throw IllegalStateException("APK export target has no parent directory.")

        check(parent.isDirectory || parent.mkdirs()) {
            "Could not create APK export directory: ${parent.absolutePath}"
        }
        src.copyTo(dst, overwrite = true)
        logger.lifecycle(
            "Exported ${src.name} to ${dst.absolutePath} (${src.length() / 1024} KB)."
        )
    }
}

abstract class BumpVersionTask : DefaultTask() {

    @get:OutputFile
    abstract val versionFile: RegularFileProperty

    @TaskAction
    fun bumpVersion() {
        val file = versionFile.get().asFile
        check(file.isFile) { "Missing version.properties: ${file.absolutePath}" }

        val props = Properties().apply {
            FileInputStream(file).use { stream -> load(stream) }
        }

        fun requiredVersionPart(key: String): Int =
            props.getProperty(key)?.toIntOrNull()
                ?: throw IllegalStateException("Missing or invalid $key in version.properties.")

        val major = requiredVersionPart("VERSION_MAJOR")
        val minor = requiredVersionPart("VERSION_MINOR")
        val patch = requiredVersionPart("VERSION_PATCH")
        val build = requiredVersionPart("VERSION_BUILD")
        val nextPatch = Math.addExact(patch, 1)
        val nextBuild = Math.addExact(build, 1)

        props["VERSION_PATCH"] = nextPatch.toString()
        props["VERSION_BUILD"] = nextBuild.toString()
        FileOutputStream(file).use {
            props.store(it, "Updated by the explicit bumpVersion task")
        }
        logger.lifecycle(
            "Version bumped for the next build: $major.$minor.$nextPatch (Build: $nextBuild)."
        )
    }
}

val exportReleaseApk = tasks.register<CopyApkTask>("exportReleaseApk") {
    group = "distribution"
    description = "Builds and exports an unsigned release APK for inspection; it is not installable."
    dependsOn("assembleRelease")
    apk.from(layout.buildDirectory.file("outputs/apk/release/app-release-unsigned.apk"))
    target.set(layout.projectDirectory.file("../ARGUS-unsigned.apk"))
    outputs.upToDateWhen { false }
}

val exportDebugApk = tasks.register<CopyApkTask>("exportDebugApk") {
    group = "distribution"
    description = "Builds and exports the automatically debug-signed APK as ARGUS.apk."
    dependsOn("assembleDebug")
    apk.from(layout.buildDirectory.file("outputs/apk/debug/app-debug.apk"))
    target.set(layout.projectDirectory.file("../ARGUS.apk"))
    outputs.upToDateWhen { false }
}

tasks.register<BumpVersionTask>("bumpVersion") {
    group = "versioning"
    description = "Explicitly increments VERSION_PATCH and VERSION_BUILD for the next build."
    versionFile.set(layout.projectDirectory.file("../version.properties"))
    outputs.upToDateWhen { false }
}

