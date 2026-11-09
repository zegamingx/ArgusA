# Material Symbols assets

This directory contains Android Vector Drawable XML assets from Google's official
Material Symbols collection.

- Source: https://github.com/google/material-design-icons
- Style: Material Symbols Rounded
- License: Apache License 2.0
- Copyright: Google LLC and the Material Symbols contributors

The original vector paths are retained. The Android theme tint attribute is
removed so Jetpack Compose Material 3 can apply each icon's tint through
`Icon(painter = painterResource(...))`.

These resources replace the legacy `androidx.compose.material.icons` code
generation dependency. They are static Vector Drawables and do not require the
Material Icons Compose artifact.
