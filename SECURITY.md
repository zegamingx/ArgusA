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
deletion. Never use that identity for new installations or releases.

The replacement offline Argus 2026 identity is pinned for clean installations
and future updates signed with the same key:

- Alias: `argus-release-2026`
- Certificate SHA-256: `40969E172BC0F8CB18ED2DDF274CAF3D3DB1456EDAEAF03939C0C0511797E64E`
- It is not a signing-certificate lineage successor to the exposed v1.0.13
  identity. Existing installations signed by the old certificate cannot be
  updated with this signer. Back up local app data before changing identity.

Keep the active Argus 2026 private keystore and both passwords offline. For
automated release builds, store them only as protected GitHub Environment
secrets in the `argus-release` Environment. The release workflow detects JKS
or PKCS#12 from the keystore bytes, verifies both passwords, the private-key
alias, and the pinned certificate before Gradle starts:

- `ARGUS_RELEASE_KEYSTORE_BASE64`
- `ARGUS_RELEASE_STORE_PASSWORD`
- `ARGUS_RELEASE_KEY_PASSWORD`

Restrict the Environment deployment rule to the approved `Release` tag and
apply required reviewers where appropriate. The private key must never enter
Git, pull-request jobs, Actions artifacts, caches, logs, or public issues.
The production workflow checks out the `Release` tag and verifies that its
commit exactly matches trusted `main`; PR validation has no access to release
signing secrets.

The `tools/security/verify-release-keystore.sh` preflight validates the
current keystore format, both passwords, private-key alias, and pinned
certificate before Gradle starts. It unwraps at most one extra Base64 layer,
and only if the result has a recognized JKS, JCEKS or PKCS#12 header. The `tools/security/sign-release.sh`
helper then decodes the same protected secret into a short-lived
permission-restricted temporary directory, detects JKS, JCEKS or PKCS#12, aligns
native libraries for 16 KiB pages, signs with v1/v2/v3 schemes, verifies the
APK with `apksigner`, checks the pinned certificate fingerprint, and confirms
the expected package ID before writing the final APK. Temporary key material is
removed on exit.

For local Windows signing, use `tools/security/sign-release.ps1` with the
official Android SDK `zipalign` and `apksigner` tools. It detects JKS or
PKCS#12, checks the expected certificate fingerprint, and verifies the final
APK before publishing it. Never store the private keystore or passwords in
the repository.

## Screen privacy

Screen perception, screen-derived training, and accessibility actions are
separate opt-ins. They are disabled by default. App access is default-deny, and
sensitive app categories remain blocked. Report any path that bypasses these
checks as a security issue.
