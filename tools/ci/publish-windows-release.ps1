[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Invoke-NativeCaptured {
    param([string]$FilePath, [string[]]$NativeArguments)
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $output = (& $FilePath @NativeArguments 2>&1 | Out-String)
    $nativeExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    if ($nativeExitCode -ne 0) {
        throw ("Native command failed with exit code " + $nativeExitCode + ": " + $FilePath + " " + ($NativeArguments -join ' ') + [Environment]::NewLine + $output)
    }
    return $output
}

function Invoke-NativeStreaming {
    param([string]$FilePath, [string[]]$NativeArguments)
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $FilePath @NativeArguments
    $nativeExitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousPreference
    if ($nativeExitCode -ne 0) { throw ("Native command failed with exit code " + $nativeExitCode + ": " + $FilePath) }
}

function Require-Env {
    param([string]$Name)
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ([string]::IsNullOrWhiteSpace($value)) { throw "Missing environment variable: $Name" }
    return $value
}

$workspace = Require-Env 'GITHUB_WORKSPACE'
$repository = Require-Env 'GITHUB_REPOSITORY'
$commitSha = Require-Env 'GITHUB_SHA'
$token = Require-Env 'ARGUS_GH_TOKEN'
$sdk = Require-Env 'ANDROID_SDK_ROOT'
$ndk = Require-Env 'ANDROID_NDK_ROOT'
$javaHome = Require-Env 'JAVA_HOME'
$cmakeHome = Require-Env 'CMAKE_HOME'

$java = Join-Path $javaHome 'bin\java.exe'
$keytool = Join-Path $javaHome 'bin\keytool.exe'
$cmake = Join-Path $cmakeHome 'bin\cmake.exe'
$ninja = 'C:\Tools\Ninja\ninja.exe'
$apksigner = Join-Path $sdk 'build-tools\37.0.0\apksigner.bat'
$expectedNdk = Join-Path $sdk 'ndk\30.0.16248370'
$glslc = Join-Path $expectedNdk 'shader-tools\windows-x86_64\glslc.exe'

$requiredFiles = @(
    $java, $keytool, $cmake, $ninja,
    (Join-Path $sdk 'platforms\android-37.0\android.jar'),
    $apksigner,
    (Join-Path $expectedNdk 'ndk-build.cmd'),
    $glslc
)
foreach ($requiredFile in $requiredFiles) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) { throw "Build tool inaccessible: $requiredFile" }
}
if ([IO.Path]::GetFullPath($ndk).TrimEnd('\') -ine [IO.Path]::GetFullPath($expectedNdk).TrimEnd('\')) {
    throw "ANDROID_NDK_ROOT must point to $expectedNdk"
}

$javaVersion = Invoke-NativeCaptured -FilePath $java -NativeArguments @('-version')
if ($javaVersion -notmatch 'version "25\.') { throw "JDK 25 required. Detected: $javaVersion" }
if ((& $cmake --version | Select-Object -First 1).Trim() -ne 'cmake version 4.5.0') { throw 'CMake 4.5.0 required.' }
if ((& $ninja --version | Out-String).Trim() -ne '1.13.2') { throw 'Ninja 1.13.2 required.' }

$env:CMAKE_BUILD_PARALLEL_LEVEL = '2'
$installer = Join-Path $workspace 'tools\cmake-compat\install-wrapper.ps1'
$wrapperRoot = Join-Path $env:RUNNER_TEMP 'argus-cmake-compat'
& $installer -RealCMake $cmake -WrapperRoot $wrapperRoot

$versionFile = Join-Path $workspace 'version.properties'
if (-not (Test-Path -LiteralPath $versionFile -PathType Leaf)) { throw 'version.properties is missing.' }
$versionProperties = @{}
foreach ($line in Get-Content -LiteralPath $versionFile) {
    if ($line -match '^\s*([A-Z_]+)\s*=\s*(.*?)\s*$') { $versionProperties[$Matches[1]] = $Matches[2] }
}
foreach ($key in @('VERSION_MAJOR', 'VERSION_MINOR', 'VERSION_PATCH', 'VERSION_BUILD')) {
    if (-not $versionProperties.ContainsKey($key) -or $versionProperties[$key] -notmatch '^\d+$') { throw "Missing or invalid $key." }
}
$versionName = "$($versionProperties.VERSION_MAJOR).$($versionProperties.VERSION_MINOR).$($versionProperties.VERSION_PATCH)"
$versionBuild = [int]$versionProperties.VERSION_BUILD
if ($versionBuild -le 0) { throw 'VERSION_BUILD must be positive.' }
Write-Host "Building Argus $versionName, build $versionBuild, source $commitSha."

$apiRoot = "https://api.github.com/repos/$repository"
$headers = @{
    Authorization = "Bearer $token"
    Accept = 'application/vnd.github+json'
    'X-GitHub-Api-Version' = '2022-11-28'
    'User-Agent' = 'Argus-Windows-SelfHosted-Runner'
}
$signingDirectory = Join-Path $workspace 'signing'
$keystore = Join-Path $signingDirectory 'argus-release.p12'
New-Item -ItemType Directory -Path $signingDirectory -Force | Out-Null

try {
    $keyRelease = Invoke-RestMethod -Method Get -Uri "$apiRoot/releases/tags/argus-signing-key" -Headers $headers
    $asset = @($keyRelease.assets | Where-Object { $_.name -eq 'ARGUS-RELEASE-KEY.p12' } | Select-Object -First 1)
    if ($asset.Count -eq 0) { throw 'ARGUS-RELEASE-KEY.p12 missing from argus-signing-key release.' }

    $binaryHeaders = @{
        Authorization = "Bearer $token"
        Accept = 'application/octet-stream'
        'X-GitHub-Api-Version' = '2022-11-28'
        'User-Agent' = 'Argus-Windows-SelfHosted-Runner'
    }
    Invoke-WebRequest -UseBasicParsing -Method Get -Uri "$apiRoot/releases/assets/$($asset[0].id)" -Headers $binaryHeaders -OutFile $keystore
    if (-not (Test-Path -LiteralPath $keystore -PathType Leaf) -or (Get-Item -LiteralPath $keystore).Length -le 0) { throw 'Downloaded signing key is empty.' }

    $null = Invoke-NativeCaptured -FilePath $keytool -NativeArguments @('-list','-storetype','PKCS12','-keystore',$keystore,'-storepass','','-alias','upload')
    Write-Host 'Persistent Argus signing identity restored and verified.'

    $env:KEYSTORE_PATH = $keystore
    $env:STORE_PASSWORD = ''
    $env:KEY_PASSWORD = ''

    Push-Location $workspace
    try {
        Invoke-NativeStreaming -FilePath (Join-Path $workspace 'gradlew.bat') -NativeArguments @(
            '--no-daemon','--no-configuration-cache','--max-workers=2','--stacktrace',
            'clean',':app:lintRelease',':app:testReleaseUnitTest',':app:assembleRelease'
        )
    }
    finally { Pop-Location }

    $apkPath = Join-Path $workspace 'app\build\outputs\apk\release\app-release.apk'
    if (-not (Test-Path -LiteralPath $apkPath -PathType Leaf) -or (Get-Item -LiteralPath $apkPath).Length -le 0) { throw 'Release APK was not produced.' }

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($apkPath)
    try {
        $entries = @($zip.Entries | ForEach-Object { $_.FullName })
        foreach ($abi in @('arm64-v8a','armeabi-v7a','x86_64')) {
            $requiredEntry = "lib/$abi/libargus_training.so"
            if ($entries -notcontains $requiredEntry) { throw "APK missing native library: $requiredEntry" }
        }
    }
    finally { $zip.Dispose() }

    $verify = Invoke-NativeCaptured -FilePath $apksigner -NativeArguments @('verify','--verbose','--print-certs',$apkPath)
    $match = [regex]::Match($verify,'(?im)certificate SHA-256 digest:\s*([0-9a-f:]+)')
    if (-not $match.Success) { throw "Could not parse APK signer digest: $verify" }
    $actualFingerprint = $match.Groups[1].Value.Replace(':','').ToLowerInvariant()

    $certificatePath = Join-Path $env:RUNNER_TEMP 'argus-upload-certificate.der'
    $null = Invoke-NativeCaptured -FilePath $keytool -NativeArguments @('-exportcert','-storetype','PKCS12','-keystore',$keystore,'-storepass','','-alias','upload','-file',$certificatePath)
    $expectedFingerprint = (Get-FileHash -LiteralPath $certificatePath -Algorithm SHA256).Hash.ToLowerInvariant()
    Remove-Item -LiteralPath $certificatePath -Force
    if ($actualFingerprint -ne $expectedFingerprint) { throw 'APK signer does not match persistent Argus upload identity.' }
    Write-Host "APK signer fingerprint verified: $actualFingerprint"

    $dist = Join-Path $workspace 'dist'
    New-Item -ItemType Directory -Path $dist -Force | Out-Null
    $versionedName = "ARGUS-v$versionName.apk"
    Copy-Item -LiteralPath $apkPath -Destination (Join-Path $dist $versionedName) -Force
    Copy-Item -LiteralPath $apkPath -Destination (Join-Path $dist 'ARGUS.apk') -Force
    foreach ($name in @($versionedName,'ARGUS.apk')) {
        $path = Join-Path $dist $name
        $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        [IO.File]::WriteAllText("$path.sha256","$hash  $name" + [Environment]::NewLine,[Text.Encoding]::ASCII)
    }
}
finally {
    if (Test-Path -LiteralPath $signingDirectory) { Remove-Item -LiteralPath $signingDirectory -Recurse -Force }
}

$notes = "Latest successfully verified Argus main build." + [Environment]::NewLine +
    "Version: v$versionName" + [Environment]::NewLine +
    "Build: $versionBuild" + [Environment]::NewLine +
    "Source commit: $commitSha" + [Environment]::NewLine + [Environment]::NewLine +
    "Published after lint, unit tests, packaging and signing verification on the self-hosted Windows runner."

$release = $null
try {
    $release = Invoke-RestMethod -Method Get -Uri "$apiRoot/releases/tags/Release" -Headers $headers
}
catch {
    $statusCode = 0
    if ($null -ne $_.Exception.Response) { $statusCode = [int]$_.Exception.Response.StatusCode }
    if ($statusCode -ne 404) { throw }
    $createBody = @{
        tag_name='Release'; target_commitish=$commitSha; name='Argus Latest Release'; body=$notes
        draft=$false; prerelease=$false; make_latest='true'
    } | ConvertTo-Json -Depth 5
    $release = Invoke-RestMethod -Method Post -Uri "$apiRoot/releases" -Headers $headers -ContentType 'application/json' -Body $createBody
}

$updateBody = @{ name='Argus Latest Release'; body=$notes; make_latest='true' } | ConvertTo-Json -Depth 5
$release = Invoke-RestMethod -Method Patch -Uri "$apiRoot/releases/$($release.id)" -Headers $headers -ContentType 'application/json' -Body $updateBody
$release = Invoke-RestMethod -Method Get -Uri "$apiRoot/releases/$($release.id)" -Headers $headers

$uploadNames = @("ARGUS-v$versionName.apk","ARGUS-v$versionName.apk.sha256",'ARGUS.apk','ARGUS.apk.sha256')
foreach ($asset in @($release.assets)) {
    $oldVersioned = $asset.name -match '^ARGUS-v.*\.apk(\.sha256)?$'
    if ($oldVersioned -or $uploadNames -contains $asset.name) {
        Invoke-RestMethod -Method Delete -Uri "$apiRoot/releases/assets/$($asset.id)" -Headers $headers | Out-Null
        Write-Host "Removed previous asset: $($asset.name)"
    }
}
foreach ($name in $uploadNames) {
    $assetPath = Join-Path $workspace "dist\$name"
    if (-not (Test-Path -LiteralPath $assetPath -PathType Leaf)) { throw "Release asset missing: $assetPath" }
    $uploadUri = "https://uploads.github.com/repos/$repository/releases/$($release.id)/assets?name=$([Uri]::EscapeDataString($name))"
    Invoke-WebRequest -UseBasicParsing -Method Post -Uri $uploadUri -Headers $headers -ContentType 'application/octet-stream' -InFile $assetPath | Out-Null
    Write-Host "Uploaded release asset: $name"
}
Write-Host ("Published Argus v" + $versionName + " to https://github.com/" + $repository + "/releases/tag/Release")
