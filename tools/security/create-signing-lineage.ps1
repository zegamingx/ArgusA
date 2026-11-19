#Requires -Version 7.0
<#
.SYNOPSIS
Prepare and verify the Android signing-certificate rotation for Argus v1.0.13.

.DESCRIPTION
Verifies that the old standalone PKCS#12 certificate matches the supplied
v1.0.13 APK, verifies a distinct replacement certificate, generates an Android
SigningCertificateLineage, signs an unsigned migration APK with old/new keys,
and verifies both the APK and lineage.

Use only on encrypted local storage. The old and replacement private keys,
passwords, and lineage must never be committed, uploaded, or placed in CI until
migration is verified. This does not migrate the separate historical upload
key identity.
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

    [string]$OldAlias = "upload",

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$NewAlias,

    [string]$PackageName = "com.argus.agi",

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
        [Parameter(Mandatory = $true)][string]$Password,
        [Parameter(Mandatory = $true)][string]$Alias
    )

    $certificates = [System.Security.Cryptography.X509Certificates.X509Certificate2Collection]::new()
    $certificates.Import(
        (Resolve-Path -LiteralPath $Path).Path,
        $Password,
        [System.Security.Cryptography.X509Certificates.X509KeyStorageFlags]::EphemeralKeySet
    )

    try {
        $privateCertificates = @($certificates | Where-Object { $_.HasPrivateKey })
        if ($privateCertificates.Count -eq 0) {
            throw "No private-key certificate was found in keystore: $Path"
        }

        if ($privateCertificates.Count -gt 1) {
            $aliasMatches = @($privateCertificates | Where-Object { $_.FriendlyName -eq $Alias })
            if ($aliasMatches.Count -ne 1) {
                throw "Could not unambiguously select alias '$Alias' from keystore: $Path"
            }
            $certificate = $aliasMatches[0]
        }
        else {
            $certificate = $privateCertificates[0]
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

function Get-CertificateFingerprints {
    param([Parameter(Mandatory = $true)][string]$Report)

    $matches = [regex]::Matches(
        $Report,
        '(?i)certificate SHA-256 digest:\s*([0-9a-f:]{64,95})'
    )
    $fingerprints = @()
    foreach ($match in $matches) {
        $fingerprints += $match.Groups[1].Value.Replace(":", "").ToLowerInvariant()
    }
    return $fingerprints
}

function Get-LineageReport {
    param([Parameter(Mandatory = $true)][string]$LineagePath)

    $report = & $script:ResolvedApksigner lineage --in $LineagePath --print-certs 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "apksigner could not read and verify the generated signing lineage."
    }
    return ($report | Out-String)
}

$script:ResolvedApksigner = (Resolve-Path -LiteralPath $ApksignerPath).Path
$resolvedExistingApk = (Resolve-Path -LiteralPath $ExistingV113Apk).Path
$resolvedOldKeystore = (Resolve-Path -LiteralPath $OldKeystorePath).Path
$resolvedNewKeystore = (Resolve-Path -LiteralPath $NewKeystorePath).Path
$resolvedUnsignedApk = (Resolve-Path -LiteralPath $UnsignedReleaseApk).Path

if ($resolvedOldKeystore -eq $resolvedNewKeystore) {
    throw "Old and replacement keystores must be different files."
}
if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) {
    $null = New-Item -ItemType Directory -Path $OutputDirectory -Force
}
$resolvedOutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
$lineagePath = Join-Path $resolvedOutputDirectory "argus-signing-lineage.bin"
$migrationApkPath = Join-Path $resolvedOutputDirectory "ARGUS-signing-migration.apk"
$tempUnsignedCopy = Join-Path $resolvedOutputDirectory "argus-unsigned-input.apk"

if ((Test-Path -LiteralPath $lineagePath) -or
    (Test-Path -LiteralPath $migrationApkPath) -or
    (Test-Path -LiteralPath $tempUnsignedCopy)) {
    throw "One or more output files already exist. Use a new, empty output directory."
}

$oldStorePassword = ConvertTo-PlainText (Read-Host "Old standalone keystore password" -AsSecureString)
$oldKeyPassword = ConvertTo-PlainText (Read-Host "Old standalone private-key password" -AsSecureString)
$newStorePassword = ConvertTo-PlainText (Read-Host "Replacement keystore password" -AsSecureString)
$newKeyPassword = ConvertTo-PlainText (Read-Host "Replacement private-key password" -AsSecureString)

try {
    $env:ARGUS_OLD_STORE_PASSWORD = $oldStorePassword
    $env:ARGUS_OLD_KEY_PASSWORD = $oldKeyPassword
    $env:ARGUS_NEW_STORE_PASSWORD = $newStorePassword
    $env:ARGUS_NEW_KEY_PASSWORD = $newKeyPassword

    $existingReport = Get-ApkSignerReport -ApkPath $resolvedExistingApk
    $existingFingerprints = Get-CertificateFingerprints -Report $existingReport
    if ($existingFingerprints.Count -ne 1) {
        throw "Expected exactly one current signer certificate in the v1.0.13 APK."
    }
    $existingFingerprint = $existingFingerprints[0]

    $oldFingerprint = Get-KeystoreCertificateSha256 -Path $resolvedOldKeystore -Password $oldStorePassword -Alias $OldAlias
    $newFingerprint = Get-KeystoreCertificateSha256 -Path $resolvedNewKeystore -Password $newStorePassword -Alias $NewAlias

    if ($oldFingerprint -ne $existingFingerprint) {
        throw "Old keystore alias '$OldAlias' does not match the v1.0.13 APK signer. Do not rotate."
    }
    if ($newFingerprint -eq $oldFingerprint) {
        throw "Replacement key has the same certificate as the exposed signer."
    }

    Write-Host "Verified v1.0.13 signer SHA-256: $oldFingerprint"
    Write-Host "Verified replacement signer SHA-256: $newFingerprint"
    Write-Host "Creating the signing-certificate lineage..."

    Invoke-ApkSigner -Arguments @(
        "rotate",
        "--out", $lineagePath,
        "--old-signer",
        "--ks", $resolvedOldKeystore,
        "--ks-type", "PKCS12",
        "--ks-key-alias", $OldAlias,
        "--ks-pass", "env:ARGUS_OLD_STORE_PASSWORD",
        "--key-pass", "env:ARGUS_OLD_KEY_PASSWORD",
        "--set-installed-data", "true",
        "--set-shared-uid", "false",
        "--set-permission", "false",
        "--set-rollback", "false",
        "--set-auth", "false",
        "--new-signer",
        "--ks", $resolvedNewKeystore,
        "--ks-type", "PKCS12",
        "--ks-key-alias", $NewAlias,
        "--ks-pass", "env:ARGUS_NEW_STORE_PASSWORD",
        "--key-pass", "env:ARGUS_NEW_KEY_PASSWORD"
    )

    if (-not (Test-Path -LiteralPath $lineagePath -PathType Leaf)) {
        throw "apksigner did not create a lineage file."
    }

    $lineageReport = Get-LineageReport -LineagePath $lineagePath
    $lineageMatches = [regex]::Matches(
        $lineageReport,
        '(?i)SHA-256 digest:\\s*([0-9a-f:]{64,95})'
    )
    $lineageFingerprints = @(
        foreach ($match in $lineageMatches) {
            $match.Groups[1].Value.Replace(":", "").ToLowerInvariant()
        }
    )
    if ($oldFingerprint -notin $lineageFingerprints -or
        $newFingerprint -notin $lineageFingerprints) {
        throw "The lineage does not contain both the old and replacement certificates."
    }

    Copy-Item -LiteralPath $resolvedUnsignedApk -Destination $tempUnsignedCopy
    Invoke-ApkSigner -Arguments @(
        "sign",
        "--ks", $resolvedOldKeystore,
        "--ks-type", "PKCS12",
        "--ks-key-alias", $OldAlias,
        "--ks-pass", "env:ARGUS_OLD_STORE_PASSWORD",
        "--key-pass", "env:ARGUS_OLD_KEY_PASSWORD",
        "--next-signer",
        "--ks", $resolvedNewKeystore,
        "--ks-type", "PKCS12",
        "--ks-key-alias", $NewAlias,
        "--ks-pass", "env:ARGUS_NEW_STORE_PASSWORD",
        "--key-pass", "env:ARGUS_NEW_KEY_PASSWORD",
        "--lineage", $lineagePath,
        "--rotation-min-sdk-version", "33",
        "--out", $migrationApkPath,
        $tempUnsignedCopy
    )

    $migrationReport = Get-ApkSignerReport -ApkPath $migrationApkPath
    $migrationFingerprints = Get-CertificateFingerprints -Report $migrationReport
    if ($migrationFingerprints.Count -ne 1 -or
        $migrationFingerprints[0] -ne $newFingerprint) {
        throw "Migration APK signer does not match the replacement certificate."
    }

    $lineageHash = (Get-FileHash -LiteralPath $lineagePath -Algorithm SHA256).Hash.ToLowerInvariant()
    $apkHash = (Get-FileHash -LiteralPath $migrationApkPath -Algorithm SHA256).Hash.ToLowerInvariant()

    Write-Host ""
    Write-Host "Lineage and migration APK cryptographic checks passed."
    Write-Host "Old signer SHA-256: $oldFingerprint"
    Write-Host "Replacement signer SHA-256: $newFingerprint"
    Write-Host "Lineage SHA-256: $lineageHash"
    Write-Host "Migration APK SHA-256: $apkHash"
    Write-Host "Lineage file: $lineagePath"
    Write-Host "Migration APK: $migrationApkPath"
    Write-Host ""
    Write-Host "Next: install over an already-installed v1.0.13 package using adb install -r."
    Write-Host "Check app launch, Room database, model checkpoints, and user data before any release."
    Write-Host "Keep the replacement key and lineage offline; do not upload either key to Git."

    if ($InstallOnConnectedDevice) {
        if (-not (Get-Command adb -ErrorAction SilentlyContinue)) {
            throw "adb is unavailable. Install Android SDK Platform-Tools."
        }
        $deviceSdk = (& adb shell getprop ro.build.version.sdk 2>$null | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or $deviceSdk -notmatch '^\d+$' -or [int]$deviceSdk -lt 34) {
            throw "A connected Android 14+ device is required for this migration test."
        }

        $installedPackage = (& adb shell pm path $PackageName 2>$null | Out-String).Trim()
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($installedPackage)) {
            throw "Package '$PackageName' is not installed. Refusing to perform a fresh install instead of an update."
        }

        & adb install -r $migrationApkPath
        if ($LASTEXITCODE -ne 0) {
            throw "In-place migration install failed. Preserve the existing v1.0.13 installation."
        }
        Write-Host "Migration APK installed as an update. Verify local data manually before publishing."
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
