# CMake 4.5 launcher compatibility

## Why this exists

CMake 4.5 deprecates the legacy command-line source-directory option `-H` in favor of `-S`. Argus uses Android Gradle Plugin external native builds, and the AGP CMake command-line builder still constructs its internal CMakeLists source argument as `-H<path>`. This warning therefore comes from AGP's launcher argument, not from Argus' CMakeLists file, compiler flags, shader code, or Vulkan implementation.

The compatibility launcher is a narrow boundary adapter. It rewrites only `-H` / `-H<path>` to `-S` / `-S<path>`, then forwards every argument to the pinned official CMake 4.5.0 executable. It does not modify CMake source files, compiler options, the CMake distribution, or generated native sources.

## CI integration

All Android build workflows install the launcher after verifying the official Kitware CMake archive SHA-256. The installer:

- Keeps the downloaded CMake 4.5.0 executable unchanged.
- Places the launcher and the matching Ninja executable in a temporary CMake root.
- Writes a machine-local `cmake.dir` entry to the ignored `local.properties` file so AGP selects the launcher.
- Runs argument-rewrite tests before Gradle starts.

## Local Linux/macOS setup

Run this from the repository root after installing or extracting the official CMake 4.5.0 distribution:

```bash
tools/cmake-compat/install-wrapper.sh "$ANDROID_HOME/cmake/4.5.0/bin/cmake"
```

Use the actual location of the CMake 4.5.0 executable when it is installed elsewhere. The command creates an ignored runtime directory and updates only the `cmake.dir` entry in the ignored `local.properties` file. It does not download or replace CMake.

To return to the unwrapped launcher, remove the `cmake.dir` line from `local.properties` and remove `tools/cmake-compat/runtime/`.

The checked-in launcher and installer use Bash and are intended for Linux/macOS hosts and the Ubuntu CI runners. Windows Android Studio builds still require a Windows-native launcher or an AGP update that replaces the internal `-H` argument.
