pluginManagement {
  repositories {
    google {
      content {
        includeGroupByRegex("com\\.android.*")
        includeGroupByRegex("com\\.google.*")
        includeGroupByRegex("androidx.*")
      }
    }
    mavenCentral()
    gradlePluginPortal()
  }
}

dependencyResolutionManagement {
  repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
  repositories {
    google()
    // Android instrumentation pulls Error Prone annotations from Maven Central.
    // Route only this group through Google's hosted Central mirror, retaining the
    // primary and secondary Maven Central endpoints as fallbacks.
    maven {
      name = "GoogleMavenCentralErrorProneMirror"
      url = uri("https://maven-central.storage-download.googleapis.com/maven2/")
      content {
        includeGroup("com.google.errorprone")
      }
    }
    mavenCentral()
    maven {
      url = uri("https://repo1.maven.org/maven2/")
    }
  }
}

rootProject.name = "A.R.G.U.S"

include(":app")
