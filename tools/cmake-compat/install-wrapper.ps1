[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$RealCMake,
    [Parameter(Mandatory = $true)][string]$WrapperRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if ($PSVersionTable.PSVersion.Major -ge 7) { throw 'Use Windows PowerShell 5.1.' }
if ([string]::IsNullOrWhiteSpace($env:ANDROID_SDK_ROOT)) { throw 'ANDROID_SDK_ROOT is missing.' }
if ([string]::IsNullOrWhiteSpace($env:GITHUB_WORKSPACE)) { throw 'GITHUB_WORKSPACE is missing.' }

$RealCMake = (Resolve-Path -LiteralPath $RealCMake).Path
$version = (& $RealCMake --version | Select-Object -First 1).Trim()
if ($version -ne 'cmake version 4.5.0') { throw "Expected CMake 4.5.0, received: $version" }

$ninjaPath = 'C:\Tools\Ninja\ninja.exe'
if (-not (Test-Path -LiteralPath $ninjaPath -PathType Leaf)) { throw "Ninja not found: $ninjaPath" }
$ninjaVersion = (& $ninjaPath --version | Out-String).Trim()
if ($ninjaVersion -ne '1.13.2') { throw "Expected Ninja 1.13.2, received: $ninjaVersion" }

$WrapperRoot = [IO.Path]::GetFullPath($WrapperRoot)
if ([string]::IsNullOrWhiteSpace($WrapperRoot) -or $WrapperRoot -eq [IO.Path]::GetPathRoot($WrapperRoot)) {
    throw "Unsafe CMake wrapper path: $WrapperRoot"
}
if (Test-Path -LiteralPath $WrapperRoot) { Remove-Item -LiteralPath $WrapperRoot -Recurse -Force }

$binRoot = Join-Path $WrapperRoot 'bin'
New-Item -ItemType Directory -Path $binRoot -Force | Out-Null
[IO.File]::WriteAllText((Join-Path $WrapperRoot 'real-cmake.path'), $RealCMake, [Text.UTF8Encoding]::new($false))
Copy-Item -LiteralPath $ninjaPath -Destination (Join-Path $binRoot 'ninja.exe') -Force

$sourceCode = @'
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;

public static class ArgusCMakeCompatLauncher
{
    public static int Main(string[] args)
    {
        try
        {
            string binDirectory = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            string wrapperRoot = Directory.GetParent(binDirectory).FullName;
            string realCMake = File.ReadAllText(Path.Combine(wrapperRoot, "real-cmake.path")).Trim();
            if (!File.Exists(realCMake))
            {
                Console.Error.WriteLine("Real CMake executable not found: " + realCMake);
                return 127;
            }
            List<string> rewritten = new List<string>();
            for (int i = 0; i < args.Length; i++)
            {
                string argument = args[i];
                if (String.Equals(argument, "-H", StringComparison.Ordinal))
                {
                    if (i + 1 >= args.Length) return 2;
                    rewritten.Add("-S");
                    rewritten.Add(args[++i]);
                }
                else if (argument.StartsWith("-H", StringComparison.Ordinal) && argument.Length > 2)
                {
                    rewritten.Add("-S" + argument.Substring(2));
                }
                else
                {
                    rewritten.Add(argument);
                }
            }
            ProcessStartInfo start = new ProcessStartInfo();
            start.FileName = realCMake;
            start.Arguments = String.Join(" ", rewritten.ConvertAll(QuoteArgument));
            start.WorkingDirectory = Environment.CurrentDirectory;
            start.UseShellExecute = false;
            using (Process child = Process.Start(start))
            {
                if (child == null) return 127;
                child.WaitForExit();
                return child.ExitCode;
            }
        }
        catch (Exception exception)
        {
            Console.Error.WriteLine("CMake launcher failed: " + exception.Message);
            return 127;
        }
    }

    private static string QuoteArgument(string value)
    {
        if (value.Length > 0 && value.IndexOfAny(new char[] { ' ', '\t', '\r', '\n', '"' }) < 0) return value;
        StringBuilder result = new StringBuilder();
        result.Append('"');
        int backslashes = 0;
        for (int i = 0; i < value.Length; i++)
        {
            char current = value[i];
            if (current == '\\') { backslashes++; continue; }
            if (current == '"')
            {
                result.Append('\\', backslashes * 2 + 1);
                result.Append('"');
                backslashes = 0;
                continue;
            }
            if (backslashes > 0)
            {
                result.Append('\\', backslashes);
                backslashes = 0;
            }
            result.Append(current);
        }
        if (backslashes > 0) result.Append('\\', backslashes * 2);
        result.Append('"');
        return result.ToString();
    }
}
'@

$launcherPath = Join-Path $binRoot 'cmake.exe'
Add-Type -TypeDefinition $sourceCode -Language CSharp -OutputAssembly $launcherPath -OutputType ConsoleApplication
$env:PATH = "$binRoot;$env:PATH"
if (-not [string]::IsNullOrWhiteSpace($env:GITHUB_PATH)) {
    [IO.File]::AppendAllText($env:GITHUB_PATH, $binRoot + [Environment]::NewLine, [Text.Encoding]::ASCII)
}

$propertiesPath = Join-Path $env:GITHUB_WORKSPACE 'local.properties'
$preserved = @()
if (Test-Path -LiteralPath $propertiesPath) {
    $preserved = @(Get-Content -LiteralPath $propertiesPath | Where-Object { $_ -notmatch '^\s*(sdk\.dir|cmake\.dir)\s*=' })
}
$properties = @($preserved) + @(
    "sdk.dir=$($env:ANDROID_SDK_ROOT.Replace('\', '/'))",
    "cmake.dir=$($WrapperRoot.Replace('\', '/'))"
)
[IO.File]::WriteAllText($propertiesPath, (($properties -join [Environment]::NewLine) + [Environment]::NewLine), [Text.UTF8Encoding]::new($false))

$probeRoot = Join-Path $env:RUNNER_TEMP ('argus-cmake-probe-' + [Guid]::NewGuid().ToString('N'))
$probeSource = Join-Path $probeRoot 'source'
$probeBuild = Join-Path $probeRoot 'build'
try {
    New-Item -ItemType Directory -Path $probeSource -Force | Out-Null
    $probe = 'cmake_minimum_required(VERSION 3.28)' + [Environment]::NewLine + 'project(ArgusCMakeCompatProbe NONE)' + [Environment]::NewLine
    [IO.File]::WriteAllText((Join-Path $probeSource 'CMakeLists.txt'), $probe, [Text.UTF8Encoding]::new($false))
    & $launcherPath "-H$probeSource" "-B$probeBuild" '-GNinja'
    if ($LASTEXITCODE -ne 0) { throw "CMake compatibility probe failed: $LASTEXITCODE" }
    if (-not (Test-Path -LiteralPath (Join-Path $probeBuild 'build.ninja') -PathType Leaf)) {
        throw 'CMake compatibility probe did not create build.ninja.'
    }
}
finally {
    if (Test-Path -LiteralPath $probeRoot) { Remove-Item -LiteralPath $probeRoot -Recurse -Force }
}

$launcherVersion = (& $launcherPath --version | Select-Object -First 1).Trim()
if ($launcherVersion -ne 'cmake version 4.5.0') { throw "Launcher version mismatch: $launcherVersion" }
Write-Host "CMake 4.5.0 launcher installed. Ninja $ninjaVersion."
