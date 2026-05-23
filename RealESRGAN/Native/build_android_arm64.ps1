$ErrorActionPreference = "Stop"

$nativeDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildDir = Join-Path $nativeDir "build-android-arm64"
$ncnnRoot = Join-Path $nativeDir "ncnn-20260113-android-vulkan-shared\\arm64-v8a"

$ndk = $env:ANDROID_NDK_ROOT
if ([string]::IsNullOrWhiteSpace($ndk)) { throw "ANDROID_NDK_ROOT not set" }

$toolchain = Join-Path $ndk "build\\cmake\\android.toolchain.cmake"

cmake -S $nativeDir -B $buildDir -G Ninja `
  -DCMAKE_TOOLCHAIN_FILE="$toolchain" `
  -DANDROID_ABI=arm64-v8a `
  -DANDROID_PLATFORM=android-24 `
  -DNCNN_ROOT="$ncnnRoot"

cmake --build $buildDir --config Release

$soSrc = Join-Path $buildDir "librealesrgan_unity.so"
$ncnnSo = Join-Path $ncnnRoot "lib\\libncnn.so"

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
$dstDir = Join-Path $pluginsDir "Android\\arm64-v8a"
New-Item -ItemType Directory -Force -Path $dstDir | Out-Null

Copy-Item -Force $soSrc (Join-Path $dstDir "librealesrgan_unity.so")
Copy-Item -Force $ncnnSo (Join-Path $dstDir "libncnn.so")
