$ErrorActionPreference = "Stop"

$nativeDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildDir = Join-Path $nativeDir "build-win64"
$ncnnRoot = Join-Path $nativeDir "ncnn-20260113-windows-vs2022-shared\\x64"

cmake -S $nativeDir -B $buildDir -G "Visual Studio 17 2022" -A x64 -DNCNN_ROOT="$ncnnRoot"
cmake --build $buildDir --config Release

$dllSrc = Join-Path $buildDir "Release\\realesrgan_unity.dll"
$ncnnDll = Join-Path $ncnnRoot "bin\\ncnn.dll"

function Resolve-UnityProjectRoot([string]$startDir) {
    $d = (Resolve-Path $startDir).Path
    for ($i = 0; $i -lt 8; $i++) {
        if (Test-Path (Join-Path $d "Assets")) { return $d }
        $p = Split-Path -Parent $d
        if ($p -eq $d) { break }
        $d = $p
    }
    throw "Unity project root not found from: $startDir"
}

$projectRoot = Resolve-UnityProjectRoot $nativeDir
$pluginsDir = Join-Path $projectRoot "Assets\\Plugins"
$dstDir = Join-Path $pluginsDir "x86_64"
New-Item -ItemType Directory -Force -Path $dstDir | Out-Null

try {
    Copy-Item -Force $dllSrc (Join-Path $dstDir "realesrgan_unity.dll")
} catch {
    Copy-Item -Force $dllSrc (Join-Path $dstDir "realesrgan_unity.dll.new")
}

try {
    Copy-Item -Force $ncnnDll (Join-Path $dstDir "ncnn.dll")
} catch {
    Copy-Item -Force $ncnnDll (Join-Path $dstDir "ncnn.dll.new")
}
