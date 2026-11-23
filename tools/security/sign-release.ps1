#Requires -Version 7.0
<#
.SYNOPSIS
Align and sign an Argus release APK with the offline Argus 2026 signing identity.

.DESCRIPTION
Uses the official Android SDK zipalign and apksigner tools. The keystore must be
kept offline and must never be committed to the repository or uploaded as a
public build artifact. The script verifies the final APK signature and its
certificate fingerprint before publishing the output path.

The new signer is for clean installs and future updates signed with this same
key. It is not a signing lineage for the previously exposed v1.0.13 identity.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$ZipalignPath,

    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$ApksignerPath,

    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$UnsignedApkPath,

    [Parameter(Mandatory = $true)]
    [ValidateScript({ Test-Path -LiteralPath $_ -PathType Leaf })]
    [string]$KeystorePath,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$OutputApkPath,

    [string]$KeyAlias = "argus-release-2026",

    [string]$ExpectedCertificateSha256 =
        "D4D24FEDCDC3395ADA6B89ACDE861CE3523560D35EDEB7DA48709C8CE4F63CAA",

    [ValidateRange(1, 100)]
    [int]$MinimumAndroidApi = 34
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function ConvertTo-PlainText {
    param(
        [Parameter(Mandatory = $true)]
        [Security.SecureString]$Value
    )

    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Value)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

function Invoke-NativeTool {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,

        [Parameter(Mandatory = $true)]
        [string]$Operation
    )

    & $Executable @Arguments
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$Operation failed with exit code $exitCode."
    }
}

$resolvedZipalign = (Resolve-Path -LiteralPath $ZipalignPath).Path
$resolvedApksigner = (Resolve-Path -LiteralPath $ApksignerPath).Path
$resolvedInput = (Resolve-Path -LiteralPath $UnsignedApkPath).Path
$resolvedKeystore = (Resolve-Path -LiteralPath $KeystorePath).Path

$outputParent = Split-Path -Parent $OutputApkPath
if ([string]::IsNullOrWhiteSpace($outputParent)) {
    $outputParent = (Get-Location).Path
}
if (-not (Test-Path -LiteralPath $outputParent -PathType Container)) {
    $null = New-Item -ItemType Directory -Path $outputParent -Force
}
$resolvedOutputParent = (Resolve-Path -LiteralPath $outputParent).Path
$outputName = Split-Path -Leaf $OutputApkPath
if ([string]::IsNullOrWhiteSpace($outputName)) {
    throw "OutputApkPath must include an APK filename."
}

$resolvedOutput = Join-Path $resolvedOutputParent $outputName
if ([string]::Equals(
    $resolvedInput,
    $resolvedOutput,
    [StringComparison]::OrdinalIgnoreCase
)) {
    throw "Input and output APK paths must be different."
}
if (Test-Path -LiteralPath $resolvedOutput) {
    throw "Output already exists. Choose a new path; the script will not overwrite an APK."
}

$alignedApk = Join-Path $resolvedOutputParent ".$outputName.aligned.tmp.apk"
$signedApk = Join-Path $resolvedOutputParent ".$outputName.signed.tmp.apk"
if ((Test-Path -LiteralPath $alignedApk) -or (Test-Path -LiteralPath $signedApk)) {
    throw "A temporary signing file already exists in the output directory."
}

$storePassword = $null
$keyPassword = $null

try {
    $storePassword = ConvertTo-PlainText (
        Read-Host "Argus release keystore password" -AsSecureString
    )
    $keyPassword = ConvertTo-PlainText (
        Read-Host "Argus release private-key password" -AsSecureString
    )

    $env:ARGUS_RELEASE_STORE_PASSWORD = $storePassword
    $env:ARGUS_RELEASE_KEY_PASSWORD = $keyPassword

    $keystoreBytes = [System.IO.File]::ReadAllBytes($resolvedKeystore)
    if ($keystoreBytes.Length -lt 4) {
        throw "The configured release keystore is truncated."
    }
    if ($keystoreBytes[0] -eq 0xFE -and $keystoreBytes[1] -eq 0xED -and $keystoreBytes[2] -eq 0xFE -and $keystoreBytes[3] -eq 0xED) {
        $keystoreType = "JKS"
    }
    elseif ($keystoreBytes[0] -eq 0xCE -and $keystoreBytes[1] -eq 0xCE -and $keystoreBytes[2] -eq 0xCE -and $keystoreBytes[3] -eq 0xCE) {
        $keystoreType = "JCEKS"
    }
    elseif ($keystoreBytes[0] -eq 0x30) {
        $keystoreType = "PKCS12"
    }
    else {
        throw "Unsupported release keystore format. Expected JKS, JCEKS or PKCS12."
    }
    $keystoreBytes = $null

    Write-Host "Aligning APK for 16 KiB native library pages..."
    Invoke-NativeTool -Executable $resolvedZipalign -Operation "zipalign" -Arguments @(
        "-f",
        "-P", "16",
        "4",
        $resolvedInput,
        $alignedApk
    )

    Invoke-NativeTool -Executable $resolvedZipalign -Operation "zipalign verification" -Arguments @(
        "-c",
        "-P", "16",
        "4",
        $alignedApk
    )

    Write-Host "Signing APK with v1, v2, and v3 schemes..."
    Invoke-NativeTool -Executable $resolvedApksigner -Operation "apksigner sign" -Arguments @(
        "sign",
        "--ks", $resolvedKeystore,
        "--ks-type", $keystoreType,
        "--ks-key-alias", $KeyAlias,
        "--ks-pass", "env:ARGUS_RELEASE_STORE_PASSWORD",
        "--key-pass", "env:ARGUS_RELEASE_KEY_PASSWORD",
        "--min-sdk-version", $MinimumAndroidApi.ToString(),
        "--v1-signing-enabled", "true",
        "--v2-signing-enabled", "true",
        "--v3-signing-enabled", "true",
        "--out", $signedApk,
        $alignedApk
    )

    Write-Host "Verifying APK signature and certificate fingerprint..."
    $verifyArguments = @(
        "verify",
        "--verbose",
        "--print-certs",
        "--min-sdk-version", $MinimumAndroidApi.ToString(),
        $signedApk
    )
    $verificationOutput = & $resolvedApksigner @verifyArguments 2>&1
    $verifyExitCode = $LASTEXITCODE
    $verificationText = $verificationOutput | Out-String

    if ($verifyExitCode -ne 0) {
        throw "apksigner verify failed with exit code $verifyExitCode.`n$verificationText"
    }

    $fingerprintMatch = [regex]::Match(
        $verificationText,
        '(?im)certificate SHA-256 digest:[\s]*([0-9a-f:]{64,95})'
    )
    if (-not $fingerprintMatch.Success) {
        throw "The verified APK did not report a signer SHA-256 fingerprint."
    }

    $actualFingerprint = $fingerprintMatch.Groups[1].Value.Replace(":", "").ToUpperInvariant()
    $expectedFingerprint = $ExpectedCertificateSha256.Replace(":", "").Replace(" ", "").ToUpperInvariant()
    if ($actualFingerprint -ne $expectedFingerprint) {
        throw "APK signer fingerprint mismatch. Expected $expectedFingerprint, got $actualFingerprint."
    }

    Move-Item -LiteralPath $signedApk -Destination $resolvedOutput
    $apkHash = (Get-FileHash -LiteralPath $resolvedOutput -Algorithm SHA256).Hash.ToLowerInvariant()

    Write-Host ""
    Write-Host "Signed APK created and verified."
    Write-Host "Output: $resolvedOutput"
    Write-Host "Certificate SHA-256: $actualFingerprint"
    Write-Host "APK SHA-256: $apkHash"
    Write-Host ""
    Write-Host "Keep the verified Argus release keystore and both passwords offline for all future updates."
}
finally {
    $env:ARGUS_RELEASE_STORE_PASSWORD = $null
    $env:ARGUS_RELEASE_KEY_PASSWORD = $null
    $storePassword = $null
    $keyPassword = $null
    Remove-Item -LiteralPath $alignedApk -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $signedApk -Force -ErrorAction SilentlyContinue
}
