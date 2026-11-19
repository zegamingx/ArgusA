#Requires -Version 7.0
<#
.SYNOPSIS
Prepare and optionally push a verified rewrite of sensitive ArgusA Git history.

.DESCRIPTION
The script:
- mirror-clones the repository to an existing encrypted local working folder;
- removes keystore, signing, environment and signing-sidecar paths from every ref;
- redacts historical hard-coded Gradle signing-password assignments;
- verifies rewritten refs contain no known signing-material paths or credential assignments;
- stops without pushing unless explicitly confirmed.

Run only after mapping the exposed signing identities and completing any required
APK signing-key migration. Freeze collaborator pushes and resolve/close active PRs
before a remote rewrite. GitHub-managed refs/pull/* are not pushed by this script.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$WorkingRoot,

    [switch]$PushCleanedHistory,

    [string]$Confirmation
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryUrl = "https://github.com/zegamingx/ArgusA.git"
$mirrorPath = Join-Path $WorkingRoot "ArgusA-clean-history.git"
$rulesPath = Join-Path $WorkingRoot "argusa-history-redactions.txt"

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Git command failed with exit code $LASTEXITCODE: git $($Arguments -join ' ')"
    }
}

function Get-RefMap {
    param([Parameter(Mandatory = $true)][string[]]$Lines)

    $map = @{}
    foreach ($line in $Lines) {
        if ($line -match '^([0-9a-fA-F]+)\s+(.+)$') {
            $map[$Matches[2]] = $Matches[1].ToLowerInvariant()
        }
    }
    return $map
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required. Install Git for Windows first."
}
if (-not (Test-Path -LiteralPath $WorkingRoot -PathType Container)) {
    throw "WorkingRoot must be an existing folder on encrypted storage."
}
if (Test-Path -LiteralPath $mirrorPath) {
    throw "The destination already exists. Move it aside or securely remove it before retrying."
}
if (Test-Path -LiteralPath $rulesPath) {
    throw "A previous redaction rules file exists at the selected working root. Inspect and remove it before retrying."
}

# git-filter-repo --version prints a script hash, not a reliable semantic version.
# Feature-probe the minimum options required by this sensitive-data rewrite.
$filterRepoHelp = (& git filter-repo -h 2>&1 | Out-String)
$filterRepoExitCode = $LASTEXITCODE
if ($filterRepoExitCode -ne 0 -or
    -not $filterRepoHelp.Contains("--sensitive-data-removal") -or
    -not $filterRepoHelp.Contains("--path-glob")) {
    throw "git-filter-repo with --sensitive-data-removal and --path-glob is required (v2.47.0 or newer)."
}

# Split the known legacy identifier so the scrubber source itself does not
# contain a complete copy of the historical variable name.
$legacyPasswordIdentifier = "default" + "Pass"
$redactionRules = @(
    ('regex:(?im)^.*' + [regex]::Escape($legacyPasswordIdentifier) + '.*$==>REDACTED_BY_HISTORY_CLEANUP'),
    'regex:(?im)^.*(?:storePassword|keyPassword)\s*=\s*["''][^"'']+["''].*$==>REDACTED_BY_HISTORY_CLEANUP'
)
[System.IO.File]::WriteAllLines(
    $rulesPath,
    $redactionRules,
    [System.Text.UTF8Encoding]::new($false)
)

try {
    Write-Host "Cloning all advertised refs into encrypted working storage..."
    Invoke-Git -Arguments @("clone", "--mirror", $repositoryUrl, $mirrorPath)

    Push-Location $mirrorPath
    try {
        # GitHub pull-request refs may not be copied by every mirror transport.
        # Fetch them explicitly so the sanitized mirror can report all affected PR refs.
        Invoke-Git -Arguments @(
            "fetch",
            "--force",
            "origin",
            "+refs/pull/*:refs/pull/*"
        )

        Write-Host "Rewriting signing files and credential assignments..."
        Invoke-Git -Arguments @(
            "filter-repo",
            "--sensitive-data-removal",
            "--force",
            "--invert-paths",
            "--path-glob", "*.jks",
            "--path-glob", "*.keystore",
            "--path-glob", "*.jceks",
            "--path-glob", "*.bks",
            "--path-glob", "*.p12",
            "--path-glob", "*.pfx",
            "--path-glob", "*.pem",
            "--path-glob", "*.pk8",
            "--path-glob", "*.key",
            "--path-glob", "*.cer",
            "--path-glob", "*.crt",
            "--path-glob", "*.der",
            "--path-glob", "*.mobileprovision",
            "--path-glob", "*.idsig",
            "--path-glob", "*.lineage",
            "--path-glob", ".env",
            "--path-glob", ".env.*",
            "--path-glob", "signing/**",
            "--replace-text", $rulesPath
        )

        $objectPaths = & git rev-list --objects --all
        if ($LASTEXITCODE -ne 0) {
            throw "Could not enumerate rewritten Git objects."
        }

        $sensitivePathPattern = '(?i)(^|/)([^/]+\.(jks|keystore|jceks|bks|p12|pfx|pem|pk8|key|cer|crt|der|mobileprovision|idsig|lineage)|\.env(?:\.[^/]*)?)$|(^|/)signing/'
        $sensitivePaths = @(
            $objectPaths | Where-Object { $_ -match $sensitivePathPattern }
        )
        if ($sensitivePaths.Count -gt 0) {
            throw "Signing material paths remain in rewritten history. Do not push."
        }

        $credentialPattern = [regex]::Escape($legacyPasswordIdentifier) +
            '|(storePassword|keyPassword)[[:space:]]*=[[:space:]]*[''"][^''"]+[''"]'
        $credentialMatches = & git log --all --format="%H" -G $credentialPattern --
        if ($LASTEXITCODE -ne 0) {
            throw "Could not scan rewritten commit contents for credential assignments."
        }
        if ($credentialMatches) {
            throw "A legacy credential assignment remains in rewritten history. Do not push."
        }

        $changedRefsPath = (& git rev-parse --git-path "filter-repo/changed-refs").Trim()
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $changedRefsPath -PathType Leaf)) {
            throw "git-filter-repo did not produce changed-refs metadata. Do not push."
        }

        $changedRefs = @(Get-Content -LiteralPath $changedRefsPath)
        $changedPullRefs = @(
            $changedRefs | Where-Object { $_ -match '^refs/pull/\d+/(head|merge)$' }
        )

        Write-Host ""
        Write-Host "History rewrite verification passed."
        Write-Host "Changed refs recorded: $($changedRefs.Count)"
        Write-Host "GitHub-managed pull-request refs listed by filter-repo: $($changedPullRefs.Count)"
        Write-Host "Review .git/filter-repo/changed-refs and first-changed-commits before any push."

        if (-not $PushCleanedHistory) {
            Write-Host "No remote refs were changed. Re-run with -PushCleanedHistory and explicit confirmation only after manual review."
            return
        }

        if ($Confirmation -cne "PUSH CLEANED HISTORY") {
            throw 'Remote rewrite blocked. Supply -Confirmation "PUSH CLEANED HISTORY" only after reviewing the sanitized mirror and freezing repository writes.'
        }

        # git-filter-repo removes the origin remote by design; add it back explicitly.
        & git remote add origin $repositoryUrl
        if ($LASTEXITCODE -ne 0) {
            throw "Could not restore the origin remote after history rewrite."
        }

        Write-Host "Pushing only branches and tags. GitHub-managed refs/pull/* are intentionally excluded."
        & git push --force --prune origin "+refs/heads/*:refs/heads/*" "+refs/tags/*:refs/tags/*"
        if ($LASTEXITCODE -ne 0) {
            throw "Branch/tag push failed. Inspect all remote refs before any retry."
        }

        Write-Host "Verifying remote branch and tag refs..."
        $localRefLines = & git for-each-ref --format="%(objectname) %(refname)" refs/heads refs/tags
        if ($LASTEXITCODE -ne 0) {
            throw "Could not enumerate local rewritten branches and tags."
        }
        $remoteRefLines = & git ls-remote --heads --tags origin
        if ($LASTEXITCODE -ne 0) {
            throw "Could not enumerate remote branches and tags after push."
        }

        $localRefs = Get-RefMap -Lines @($localRefLines)
        $remoteRefs = Get-RefMap -Lines @(
            $remoteRefLines | Where-Object { $_ -notmatch '\^\{\}$' }
        )

        $refMismatches = @()
        foreach ($refName in $localRefs.Keys) {
            if (-not $remoteRefs.ContainsKey($refName) -or
                $remoteRefs[$refName] -ne $localRefs[$refName]) {
                $refMismatches += $refName
            }
        }
        foreach ($refName in $remoteRefs.Keys) {
            if (-not $localRefs.ContainsKey($refName)) {
                $refMismatches += $refName
            }
        }
        if ($refMismatches.Count -gt 0) {
            throw "Remote branch/tag refs do not exactly match the sanitized mirror: $($refMismatches -join ', ')"
        }

        Write-Host "All remote branches and tags match the sanitized mirror."
        Write-Host "Contact GitHub Support to request removal of cached sensitive-data views and affected PR refs."
        Write-Host "Re-clone the repository after the rewrite. Do not reuse old working clones."
    }
    finally {
        Pop-Location
    }
}
finally {
    Remove-Item -LiteralPath $rulesPath -Force -ErrorAction SilentlyContinue
}
