# ArgusA repository security and public-release checklist

This checklist separates GitHub repository publication from distributing the Android app. Do not make the repository public while historical signing credentials remain in Git history.

## Current protected-release design

The release workflow in this branch is manual-only and limited to `main`. It uses a dedicated GitHub Environment named `release-signing` and expects three environment secrets:

- `ARGUS_SIGNING_KEY_BASE64`: base64 of an encrypted PKCS#12 file
- `ARGUS_SIGNING_STORE_PASSWORD`: PKCS#12 store password
- `ARGUS_SIGNING_KEY_PASSWORD`: private-key password

Set required reviewers on the Environment. Do not put the private key in a repository secret if a protected environment secret can be used instead. Do not upload keystores as release assets, workflow artifacts, caches, or logs.

The workflow pins the certificate SHA-256 taken from the supplied v1.0.13 APK. It intentionally fails when the uploaded signing key does not match. Do not change this fingerprint as a workaround.

## Signing migration prerequisites

1. Keep the old signing material on encrypted local storage only while establishing the update path.
2. Compare the signer SHA-256 fingerprints of every APK distribution channel and the historical `my-upload-key.jks` identity.
3. Determine whether the old JKS-named file and the standalone PKCS#12 key are the same certificate. Do not assume that they are.
4. Before rotating any signer, use Android signing lineage and test an in-place update from each supported installed APK. Confirm app data survives.
5. Encrypt the retained signing identity in PKCS#12 with a strong, unique password. Store the password separately from the file.
6. Configure the three `release-signing` environment secrets and required reviewers. Run the manual release workflow only after the exact certificate pin passes.

If an identity cannot be retrieved or its private key is unavailable, do not generate an unrelated key and publish it as an update. Plan a new package/application migration instead.

## Historical Git cleanup

The repository previously contained a JKS-named PKCS#12 file in Git history and a hard-coded password assignment in a historical Gradle file. Removing a file from the current tree is not enough.

The PowerShell helper `tools/security/rewrite-sensitive-history.ps1`:

- mirror-clones the repository into an existing encrypted working folder;
- removes `my-upload-key.jks` from all refs;
- redacts historical lines containing the legacy password variable identifier;
- checks that common signing-material filenames and the Gradle identifier are absent;
- does not push unless a separate explicit switch and confirmation are supplied.

Before the remote rewrite:

- merge or close all open pull requests;
- stop all pushes and ask collaborators not to push old clones;
- verify the rewritten mirror and check `.git/filter-repo/changed-refs`;
- record the first changed commits and the count/list of affected pull requests;
- temporarily disable branch protection only if it blocks the mirror push, then restore it immediately afterward.

After the remote rewrite, contact GitHub Support to request removal of cached views and affected pull-request references. GitHub-managed `refs/pull/*` refs cannot be force-pushed by the mirror operation.

Do not reuse an old clone after this operation. Re-clone the sanitized repository.

## Repository controls required before changing visibility

- Protect `main`: require pull requests, required successful validation checks, code-owner review, conversation resolution, stale approval dismissal, and no force-push or deletion.
- Keep signing secrets only in the protected release Environment.
- Enable secret scanning and push protection; run a full-history credential scan after the rewrite.
- Keep Dependabot enabled for Gradle and GitHub Actions.
- Review GitHub Actions logs and artifacts, all issues and pull requests, tags, releases, wiki content, and repository discussions for sensitive data.
- Choose and add an explicit license before describing Argus as open source. No license is selected by this checklist.
- Confirm that the root README accurately describes behavior and capabilities.

## Application privacy boundary

Screen text perception, screen-derived training, and accessibility clicks are separate opt-ins and default to off in this branch. Application access is default-deny. Sensitive app categories remain protected. The broad `QUERY_ALL_PACKAGES` permission was removed; the app list is limited to launchable applications.

Before app-store distribution, review the final UX and runtime behavior against current Google Play User Data, Accessibility API, and package-visibility policies. The in-app disclosure must be prominent during normal app use and must immediately precede the corresponding consent action. The tests in this branch cover default-off consent and allowlist policy, but a full Android build and emulator validation remain required.

## Go/no-go

Repository visibility must remain private until all of the following are true:

- signing identity and update lineage are verified;
- the protected Environment secrets are configured and manual release validation succeeds;
- the complete Git history is rewritten and verified;
- GitHub Support has been contacted about cached and PR refs;
- all branch protections, secret scanning, and push protection are enabled;
- license decision is recorded;
- the privacy implementation passes Gradle, unit, lint, and emulator validation.
