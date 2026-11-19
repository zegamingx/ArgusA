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

Release signing is suspended while the previously distributed signing identity
is migrated. The v1.0.13 signer appeared in an unencrypted GitHub Release asset
and must be treated as compromised, even after asset deletion.

Do not store signing identities in Git history, repository files, release
assets, workflow artifacts, or caches. A new signing workflow must use a new
offline-generated key and a verified Android signing-certificate lineage. A
protected GitHub Environment is an additional access control, not a remedy for
a compromised signing identity.

## Screen privacy

Screen perception, screen-derived training, and accessibility actions are
separate opt-ins. They are disabled by default. App access is default-deny, and
sensitive app categories remain blocked. Report any path that bypasses these
checks as a security issue.
