# Security Policy

## Reporting a vulnerability

Do not report security vulnerabilities in public issues or pull requests.

Use GitHub's private vulnerability reporting feature for this repository when it
is enabled. If private reporting is unavailable, contact the repository
maintainer privately through the maintainer's GitHub profile and include only
the minimum information needed to reproduce the issue.

Do not include personal data, production user data, access tokens, passwords,
private signing material, or unredacted device logs in a report.

## Signing material

The previously distributed v1.0.13 signing identity appeared in an unencrypted
GitHub Release asset and must be treated as compromised, even after asset
deletion. Do not use that identity for new installations or releases.

A separate offline Argus 2026 signing identity is used for clean installs and
future releases signed with the same key:

- Alias: `argus-release-2026`
- Certificate SHA-256: `40969E172BC0F8CB18ED2DDF274CAF3D3DB1456EDAEAF03939C0C0511797E64E`
- The new key is not a signing-certificate lineage successor to the exposed
  v1.0.13 identity. Existing installations signed by the old identity cannot
  be updated with this key; they require a separate migration or package plan.
- Keep the PKCS#12 keystore and its passwords offline. Never commit the
  keystore, passwords, or private key to Git, releases, Actions artifacts,
  caches, or public issues.

Use `tools/security/sign-release.ps1` with the official Android SDK
`zipalign` and `apksigner` tools to align and sign release APKs locally. The
helper requires the expected certificate fingerprint and verifies the final
APK before writing its output. It enables v1, v2, and v3 signing and aligns
uncompressed native libraries for 16 KiB pages.

A protected GitHub Environment is an additional access-control layer, not a
remedy for a compromised signing identity. Do not enable automated CI signing
until the offline key backup and repository secret-handling process have been
reviewed.

## Screen privacy

Screen perception, screen-derived training, and accessibility actions are
separate opt-ins. They are disabled by default. App access is default-deny, and
sensitive app categories remain blocked. Report any path that bypasses these
checks as a security issue.
