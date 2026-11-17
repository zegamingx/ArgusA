// AGP 9 provides built-in Kotlin. Pin the newer Kotlin Gradle Plugin and KSP
// versions explicitly so their runtimes stay aligned with the version catalog.
buildscript {
  repositories {
    google()
    mavenCentral()
  }
  dependencies {
    classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.4.21")
    classpath("com.google.devtools.ksp:symbol-processing-gradle-plugin:2.3.14")
  }
}

// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
  alias(libs.plugins.android.application) apply false
  alias(libs.plugins.kotlin.compose) apply false
  alias(libs.plugins.google.devtools.ksp) apply false
}

