#Requires -Version 7.0
<#
.SYNOPSIS
Create an Android APK signing lineage and migration APK locally.

.DESCRIPTION
Checks that the old keystore matches the supplied Argus v1.0.13 APK, verifies
that the replacement keystore has a different certificate, creates the
certificate lineage with apksigner rotate, signs an unsigned APK with the
old/new signer lineage, and verifies the migration APK.

Run only on encrypted local storage. Keys, passwords, and lineage are never
uploaded. The old key must be the standalone signer used by v1.0.13, not the
separate historical upload-key identity.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$ApksignerPath,

    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$ExistingV113Apk,

    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$OldKeystorePath,

    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$NewKeystorePath,

    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$UnsignedReleaseApk,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$OutputDirectory,

    [string]$Alias = "upload",

    [switch]$InstallOnConnectedDevice
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function ConvertTo-PlainText {
    param([Parameter(Mandatory = $true)][Security.SecureString]$Value)

    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Value)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

function Get-KeystoreCertificateSha256 {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Password
    )

    $certificates = [System.Security.Cryptography.X509Certificates.X509Certificate2Collection]::new()
    $certificates.Import(
        (Resolve-Path -LiteralPath $Path).Path,
        $Password,
        [System.Security.Cryptography.X509Certificates.X509KeyStorageFlags]::EphemeralKeySet
    )

    try {
        $certificate = $certificates |
            Where-Object { $_.HasPrivateKey } |
            Select-Object -First 1

        if ($null -eq $certificate) {
            throw "No private-key certificate was found in keystore: $Path"
        }

        return $certificate.GetCertHashString(
            [System.Security.Cryptography.HashAlgorithmName]::SHA256
        ).Replace(":", "").ToLowerInvariant()
    }
    finally {
        foreach ($certificate in $certificates) {
            $certificate.Dispose()
        }
    }
}

function Invoke-ApkSigner {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    & $script:ResolvedApksigner @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "apksigner failed with exit code $LASTEXITCODE."
    }
}

function Get-ApkSignerReport {
    param([Parameter(Mandatory = $true)][string]$ApkPath)

    $report = & $script:ResolvedApksigner verify --verbose --print-certs --min-sdk-version 34 $ApkPath 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "APK signature verification failed: $ApkPath"
    }

    return ($report | Out-String)
}

$script:ResolvedApksigner = (Resolve-Path -LiteralPath $ApksignerPath).Path
$resolvedExistingApk = (Resolve-Path -LiteralPath $ExistingV113Apk).Path
$resolvedOldKeystore = (Resolve-Path -LiteralPath $OldKeystorePath).Path
$resolvedNewKeystore = (Resolve-Path -LiteralPath $NewKeystorePath).Path
$resolvedUnsignedApk = (Resolve-Path -LiteralPath $UnsignedReleaseApk).Path

if ($resolvedOldKeystore -eq $resolvedNewKeystore) {
    throw "Old and new keystores must be different files."
}
if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) {
    $null = New-Item -ItemType Directory -Path $OutputDirectory -Force
}
$resolvedOutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
$lineagePath = Join-Path $resolvedOutputDirectory "argus-signing-lineage.bin"
$migrationApkPath = Join-Path $resolvedOutputDirectory "ARGUS-signing-migration.apk"
$tempUnsignedCopy = Join-Path $resolvedOutputDirectory "argus-unsigned-input.apk"

$oldStorePassword = ConvertTo-PlainText (Read-Host "Old standalone keystore password" -AsSecureString)
$oldKeyPassword = ConvertTo-PlainText (Read-Host "Old standalone private-key password" -AsSecureString)
$newStorePassword = ConvertTo-PlainText (Read-Host "Replacement keystore password" -AsSecureString)
$newKeyPassword = ConvertTo-PlainText (Read-Host "Replacement private-key password" -AsSecureString)

try {
    $env:ARGUS_OLD_STORE_PASSWORD = $oldStorePassword
    $env:ARGUS_OLD_KEY_PASSWORD = $oldKeyPassword
    $env:ARGUS_NEW_STORE_PASSWORD = $newStorePassword
    $env:ARGUS_NEW_KEY_PASSWORD = $newKeyPassword

    $existingApkReport = Get-ApkSignerReport -ApkPath $resolvedExistingApk
    $existingMatches = [regex]::Matches(
        $existingApkReport,
        '(?i)certificate SHA-256 digest:\s*([0-9a-f:]{64,95})'
    )
    if ($existingMatches.Count -lt 1) {
        throw "Could not extract the v1.0.13 APK signer SHA-256."
    }
    $existingFingerprint = $existingMatches[0].Groups[1].Value.Replace(":", "").ToLowerInvariant()

    $oldFingerprint = Get-KeystoreCertificateSha256 -Path $resolvedOldKeystore -Password $oldStorePassword
    $newFingerprint = Get-KeystoreCertificateSha256 -Path $resolvedNewKeystore -Password $newStorePassword

    if ($oldFingerprint -ne $existingFingerprint) {
        throw "The old keystore does not match the v1.0.13 APK signer. Do not rotate this identity."
    }
    if ($newFingerprint -eq $oldFingerprint) {
        throw "The replacement key has the same certificate as the old signer."
    }

    if (Test-Path -LiteralPath $lineagePath) {
        throw "Lineage output already exists. Choose a new, empty output directory."
    }
    if (Test-Path -LiteralPath $migrationApkPath) {
        throw "Migration APK output already exists. Choose a new, empty output directory."
    }

    Write-Host "Verified old signer: $oldFingerprint"
    Write-Host "Verified replacement signer: $newFingerprint"
    Write-Host "Creating Android signing-certificate lineage..."

    Invoke-ApkSigner -Arguments @(
        "rotate",
        "--out", $lineagePath,
        "--old-signer",
        "--ks", $resolvedOldKeystore,
        "--ks-type", "PKCS12",
        "--ks-key-alias", $Alias,
        "--ks-pass", "env:ARGUS_OLD_STORE_PASSWORD",
        "--key-pass", "env:ARGUS_OLD_KEY_PASSWORD",
        "--new-signer",
        "--ks", $resolvedNewKeystore,
        "--ks-type", "PKCS12",
        "--ks-key-alias", $Alias,
        "--ks-pass", "env:ARGUS_NEW_STORE_PASSWORD",
        "--key-pass", "env:ARGUS_NEW_KEY_PASSWORD"
    )

    if (-not (Test-Path -LiteralPath $lineagePath -PathType Leaf)) {
        throw "apksigner did not produce the expected signing lineage."
    }

    Copy-Item -LiteralPath $resolvedUnsignedApk -Destination $tempUnsignedCopy
    Invoke-ApkSigner -Arguments @(
        "sign",
        "--ks", $resolvedOldKeystore,
        "--ks-type", "PKCS12",
        "--ks-key-alias", $Alias,
        "--ks-pass", "env:ARGUS_OLD_STORE_PASSWORD",
        "--key-pass", "env:ARGUS_OLD_KEY_PASSWORD",
        "--next-signer",
        "--ks", $resolvedNewKeystore,
        "--ks-type", "PKCS12",
        "--ks-key-alias", $Alias,
        "--ks-pass", "env:ARGUS_NEW_STORE_PASSWORD",
        "--key-pass", "env:ARGUS_NEW_KEY_PASSWORD",
        "--lineage", $lineagePath,
        "--rotation-min-sdk-version", "33",
        "--out", $migrationApkPath,
        $tempUnsignedCopy
    )

    $migrationReport = Get-ApkSignerReport -ApkPath $migrationApkPath
    if (-not $migrationReport.ToLowerInvariant().Contains($newFingerprint)) {
        throw "The verified migration APK does not report the replacement signing certificate."
    }

    $lineageHash = (Get-FileHash -LiteralPath $lineagePath -Algorithm SHA256).Hash.ToLowerInvariant()
    $apkHash = (Get-FileHash -LiteralPath $migrationApkPath -Algorithm SHA256).Hash.ToLowerInvariant()

    Write-Host ""
    Write-Host "Signing lineage and migration APK created and cryptographically verified."
    Write-Host "Old signer SHA-256: $oldFingerprint"
    Write-Host "New signer SHA-256: $newFingerprint"
    Write-Host "Lineage SHA-256: $lineageHash"
    Write-Host "Migration APK SHA-256: $apkHash"
    Write-Host "Lineage file: $lineagePath"
    Write-Host "Migration APK: $migrationApkPath"
    Write-Host ""
    Write-Host "Install this APK over an existing v1.0.13 installation with adb install -r."
    Write-Host "Then inspect app launch, local database, model checkpoints, and user data."
    Write-Host "Do not upload either keystore, passwords, or lineage to Git."

    if ($InstallOnConnectedDevice) {
        if (-not (Get-Command adb -ErrorAction SilentlyContinue)) {
            throw "adb is unavailable. Install Android SDK Platform-Tools."
        }
        $deviceSdk = (& adb shell getprop ro.build.version.sdk 2>$null | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or $deviceSdk -notmatch '^\d+$' -or [int]$deviceSdk -lt 34) {
            throw "A connected Android 14+ device is required for this migration test."
        }

        & adb install -r $migrationApkPath
        if ($LASTEXITCODE -ne 0) {
            throw "In-place migration install failed. The installed application was not replaced."
        }
        Write-Host "Migration APK installed. Verify the existing app data manually before publishing."
    }
}
finally {
    $env:ARGUS_OLD_STORE_PASSWORD = $null
    $env:ARGUS_OLD_KEY_PASSWORD = $null
    $env:ARGUS_NEW_STORE_PASSWORD = $null
    $env:ARGUS_NEW_KEY_PASSWORD = $null
    $oldStorePassword = $null
    $oldKeyPassword = $null
    $newStorePassword = $null
    $newKeyPassword = $null
    Remove-Item -LiteralPath $tempUnsignedCopy -Force -ErrorAction SilentlyContinue
}
