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

Release signing material must remain outside Git history, repository files,
release assets, workflow artifacts, and caches. CI signing is restricted to
the manually dispatched release workflow and the protected
`release-signing` environment. Treat any signing key that has appeared in Git
history or a downloadable repository asset as compromised until its
replacement or migration has been verified.
