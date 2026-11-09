# Argus

Argus is a native Android application built with Kotlin and Jetpack Compose.

## Requirements

- JDK 17
- Android SDK Platform 37
- Android SDK Build Tools 36.0.0
- Gradle Wrapper (repository-pinned)

## Local builds

Build a debug APK:

```bash
./gradlew :app:assembleDebug
```

Run release checks and build:

```bash
./gradlew :app:lintRelease :app:testReleaseUnitTest :app:assembleRelease
```

A release build requires the Argus release signing key at `signing/argus-release.p12`, or a different path supplied through `KEYSTORE_PATH`. The alias is `upload`; the keystore format is PKCS#12.

## GitHub releases

Publishing a versioned GitHub Release (for example, `v1.0.7`) runs lint, unit tests, a signed release build, certificate verification, and APK publication.

The private repository keeps the CI signing-key backup in the private GitHub Release named `argus-signing-key`. Repository access therefore also grants access to the app-signing identity. Keep this repository private and restrict collaborator access. Never redistribute the signing-key asset.

## Compatibility note

The standalone signing identity is separate from the legacy AI Studio upload key. APKs signed with the standalone key cannot update installations signed by a different certificate. Keep the existing installation until its data has been safely exported or backed up before switching.
