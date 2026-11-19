#Requires -Version 7.0
<#
.SYNOPSIS
Rewrites the ArgusA Git history to remove the legacy signing keystore and
redact the historical hard-coded signing-password assignment.

.DESCRIPTION
The script creates a fresh mirror clone, runs git-filter-repo, verifies that
known signing paths and the legacy password assignment are absent, and stops.
It does not push unless -PushCleanedHistory is explicitly supplied together
with -Confirmation "PUSH CLEANED HISTORY".

Keep the working directory on encrypted storage. Do not create an unencrypted
backup of the tainted repository. Rotate or migrate signing material first.
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
        throw "Git command failed: git $($Arguments -join ' ')"
    }
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required. Install Git for Windows first."
}
if (-not (Get-Command git-filter-repo -ErrorAction SilentlyContinue)) {
    throw "git-filter-repo >= 2.47 is required. Install it before running this script."
}
if (-not (Test-Path -LiteralPath $WorkingRoot -PathType Container)) {
    throw "WorkingRoot must be an existing folder on encrypted storage."
}
if (Test-Path -LiteralPath $mirrorPath) {
    throw "The destination already exists. Move or securely remove it before retrying."
}

# Build the sensitive variable name from fragments so this scrubber does not
# preserve the exact legacy identifier in the new repository history.
$legacyPasswordIdentifier = "default" + "Pass"
$redactionRule = ('regex:(?im)^.*' + [regex]::Escape($legacyPasswordIdentifier) + '.*$==>REDACTED_BY_HISTORY_CLEANUP')
Set-Content -LiteralPath $rulesPath -Value $redactionRule -Encoding utf8

try {
    Write-Host "Cloning the complete repository history to encrypted working storage..."
    Invoke-Git -Arguments @("clone", "--mirror", $repositoryUrl, $mirrorPath)

    Push-Location $mirrorPath
    try {
        Write-Host "Removing the legacy keystore path and redacting the historical password line..."
        Invoke-Git -Arguments @(
            "filter-repo",
            "--sensitive-data-removal",
            "--force",
            "--invert-paths",
            "--path", "my-upload-key.jks",
            "--replace-text", $rulesPath
        )

        $objectPaths = & git rev-list --objects --all
        if ($LASTEXITCODE -ne 0) {
            throw "Could not enumerate rewritten Git objects."
        }

        $sensitivePaths = @(
            $objectPaths | Where-Object {
                $_ -match '(?i)(^|/)(my-upload-key\.jks|[^/]+\.(jks|keystore|p12|pfx|pem|key))$'
            }
        )
        if ($sensitivePaths.Count -gt 0) {
            throw "Signing material paths remain in rewritten history. Do not push."
        }

        $passwordMatches = & git log --all --format="%H" -G $legacyPasswordIdentifier -- "*.gradle" "*.gradle.kts"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not verify historical Gradle signing configuration."
        }
        if ($passwordMatches) {
            throw "The legacy signing-password identifier remains in Gradle history. Do not push."
        }

        $changedRefsPath = (& git rev-parse --git-path "filter-repo/changed-refs").Trim()
        $changedPullRefs = @()
        if (Test-Path -LiteralPath $changedRefsPath) {
            $changedPullRefs = @(
                Get-Content -LiteralPath $changedRefsPath |
                    Where-Object { $_ -match '^refs/pull/\d+/(head|merge)$' }
            )
        }

        Write-Host ""
        Write-Host "History rewrite verification passed."
        Write-Host "Changed pull-request refs: $($changedPullRefs.Count)"
        Write-Host "First changed commits and ref details are recorded under the mirror's filter-repo/ metadata directory."
        Write-Host "Review those files before any remote push."

        if ($PushCleanedHistory) {
            if ($Confirmation -cne "PUSH CLEANED HISTORY") {
                throw 'Remote rewrite blocked. Supply -Confirmation "PUSH CLEANED HISTORY" only after reviewing the rewritten mirror.'
            }

            $origin = & git remote get-url origin 2>$null
            if ($LASTEXITCODE -ne 0 -or -not $origin) {
                Invoke-Git -Arguments @("remote", "add", "origin", $repositoryUrl)
            }

            Write-Host "Force-pushing rewritten branches and tags. GitHub-managed refs/pull refs are read-only."
            $pushOutput = & git push --force --mirror origin 2>&1
            $pushExitCode = $LASTEXITCODE
            $pushOutput | ForEach-Object { Write-Host $_ }

            $pushLines = @($pushOutput | ForEach-Object { [string]$_ })
            $pullRefRejections = @(
                $pushLines | Where-Object {
                    $_ -match '(?i)(rejected|failed)' -and $_ -match 'refs/pull/'
                }
            )
            $unexpectedPushErrors = @(
                $pushLines | Where-Object {
                    $_ -match '(?i)(fatal:|error:|remote error:|unable to|could not|timed out|hang up|rejected|failed)' -and
                    $_ -notmatch 'refs/pull/'
                }
            )

            if ($pushExitCode -ne 0) {
                if ($pullRefRejections.Count -gt 0 -and $unexpectedPushErrors.Count -eq 0) {
                    Write-Warning "GitHub rejected only GitHub-managed refs/pull entries. Branch/tag updates still require independent verification."
                }
                else {
                    throw "Mirror push returned exit code $pushExitCode. Do not proceed; inspect the complete output and verify every remote ref."
                }
            }

            if ($unexpectedPushErrors.Count -gt 0) {
                throw "One or more non-PR refs failed to update. Do not proceed."
            }

            Write-Host "Mirror push command completed. Verify every remote branch/tag and contact GitHub Support with changed-ref details."
        }
        else {
            Write-Host "No remote refs were changed. Run again with -PushCleanedHistory and the explicit confirmation only after review."
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    Remove-Item -LiteralPath $rulesPath -Force -ErrorAction SilentlyContinue
}
