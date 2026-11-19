# ArgusA repository security and public-release checklist

Keep repository visibility private until historical signing material has been removed from all refs and the remaining controls below have been verified.

## Signing incident and current release state

- The supplied Argus v1.0.13 APK uses the standalone release certificate.
- That standalone PKCS#12 identity was previously distributed as an unencrypted GitHub Release asset. Removing the asset does not undo prior exposure. Treat the private key as compromised.
- A separate historical JKS-named PKCS#12 file was committed to Git history. The README identifies this legacy upload identity as separate from the standalone identity. Treat both as exposed until fingerprints and actual APK distribution channels are mapped.
- The signed-release workflow has been removed from this branch. Gradle release variants are unsigned. Do not sign new APKs with either exposed identity.

## Signing migration

The local helper `tools/security/create-signing-lineage.ps1` checks that the old keystore matches the supplied v1.0.13 APK, checks that the replacement keystore has a different certificate, creates a lineage with `apksigner rotate`, signs an unsigned APK with the old/new signer lineage, and verifies the output.

Run it only on an encrypted, trusted workstation. The old key is used only for this controlled migration and must not be restored to CI. Keep passwords separate from keystores. The script never uploads either keystore or the lineage file.

After producing the migration APK:

- Install it over an existing v1.0.13 installation using `adb install -r`.
- Verify package identity, launch, local database, model/checkpoint files, and user data.
- Install a second APK signed with the replacement identity and the same lineage to verify subsequent updates.
- Test on a physical Android 14+ device and emulator/device API boundaries.
- If the old signer key is unavailable, do not create an unrelated key and call it an update. Preserve the existing installation and plan a separate package/application migration.

Only after migration succeeds should a protected signing workflow be designed around the replacement signer and its verified SHA-256 fingerprint. A GitHub Environment is an access-control layer, not a remedy for a compromised key.

## Historical Git cleanup

The repository previously contained a JKS-named PKCS#12 file in Git history and a hard-coded password assignment in a historical Gradle file. Removing files from the current tree does not remove historical Git objects.

The PowerShell helper `tools/security/rewrite-sensitive-history.ps1`:

- mirror-clones the repository into an existing encrypted working folder;
- removes `my-upload-key.jks` from all refs;
- redacts historical Gradle lines containing the legacy password variable identifier;
- checks common signing-material filenames and the Gradle identifier are absent;
- does not push without a separate explicit switch and confirmation.

Before rewriting remote refs, finish or close active pull requests, freeze pushes, tell collaborators not to push old clones, inspect `.git/filter-repo/changed-refs`, and review the rewritten refs. GitHub-managed `refs/pull/*` refs are not force-pushable. After rewriting branches and tags, contact GitHub Support to request removal of cached views and affected PR references. Re-clone the sanitized repository; do not reuse an old clone.

## Repository controls required before public visibility

- Protect `main`: require pull requests, passing validation checks, code-owner review, conversation resolution, stale approval dismissal, and prohibit force-push and deletion.
- Enable secret scanning and push protection. Run a full-history credential scan after the rewrite.
- Review every branch, tag, release, Actions log/artifact, issue, pull request, wiki page, and discussion for private data.
- Keep Dependabot enabled for Gradle and GitHub Actions.
- Decide whether Argus is intended to be open source. No license has been selected. Public visibility alone does not grant reuse rights.
- Verify README feature claims, metadata, permissions, and release instructions.

## Application privacy boundary

Screen perception, screen-derived training, and accessibility clicks are separate opt-ins and disabled by default in this branch. App access is default-deny; sensitive app categories remain protected. The broad `QUERY_ALL_PACKAGES` permission has been removed; the app list uses a launcher-intent query.

The normal app flow displays a disclosure. Settings explains what is read and stored locally and exposes separate controls. This implementation still needs a complete Android build, lint, unit tests, and emulator validation. Re-review it against current Google Play User Data, Accessibility API, and package-visibility policies before Play distribution.

## Go/no-go

Keep repository visibility private until:

- each signing identity has been mapped to APKs users actually installed;
- the v1.0.13 signer is rotated through a tested lineage, or a separate package migration is chosen;
- legacy signing material and passwords have been removed from all Git refs and GitHub cached views have been addressed;
- a replacement-key release workflow has passed validation;
- branch protection, secret scanning, and push protection are enabled;
- a license decision is recorded;
- Android build, lint, unit, and emulator validation pass;
- screen privacy controls pass code review and runtime tests.
