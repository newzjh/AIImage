param(
    [string]$UnityExe = 'C:\Program Files\Unity 6000.2.7f2\Editor\Unity.exe',
    [string]$ProjectPath = 'E:\Projects\AIImage',
    [string]$ModelDir = '',
    [string]$ImagePath = '',
    [string]$Prompt = '',
    [ValidateRange(1, 512)]
    [int]$MaxNewTokens = 160,
    [switch]$RequireOcrMarkers,
    [string]$UnityReport = '',
    [string]$UnityLog = '',
    [string]$ValidationReport = ''
)

$ErrorActionPreference = 'Stop'

function Get-Sha256WithRetry([string]$Path, [int]$Attempts = 80, [int]$DelayMilliseconds = 250) {
    $lastError = $null
    for ($attempt = 0; $attempt -lt $Attempts; $attempt++) {
        try {
            return (Get-FileHash -LiteralPath $Path -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()
        } catch {
            $lastError = $_.Exception
            Start-Sleep -Milliseconds $DelayMilliseconds
        }
    }
    throw "Failed to hash file after $Attempts attempts: $Path`n$lastError"
}

$toolDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($ModelDir)) {
    $ModelDir = Join-Path $toolDir '_models\qwen3.5_0.8b'
}
if ([string]::IsNullOrWhiteSpace($ImagePath)) {
    $ImagePath = Join-Path $ProjectPath 'ref\ncnn_llm-main\test.jpg'
}
if ([string]::IsNullOrWhiteSpace($UnityReport)) {
    $UnityReport = Join-Path $toolDir 'reports\unity_multimodal_ocr_e2e.json'
}
if ([string]::IsNullOrWhiteSpace($UnityLog)) {
    $UnityLog = Join-Path $ProjectPath 'Logs\qwen35_multimodal_ocr_e2e.log'
}
if ([string]::IsNullOrWhiteSpace($ValidationReport)) {
    $ValidationReport = Join-Path $toolDir 'reports\unity_multimodal_ocr_batch_validation.json'
}

foreach ($path in @($UnityExe, $ProjectPath, $ModelDir, $ImagePath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required Qwen3.5 path is missing: $path"
    }
}
foreach ($path in @($UnityReport, $UnityLog, $ValidationReport)) {
    $parent = Split-Path -Parent ([IO.Path]::GetFullPath($path))
    [IO.Directory]::CreateDirectory($parent) | Out-Null
}

$stdoutPath = [IO.Path]::ChangeExtension($ValidationReport, '.stdout.txt')
$stderrPath = [IO.Path]::ChangeExtension($ValidationReport, '.stderr.txt')
$env:AIIMAGE_QWEN35_MODEL_DIR = [IO.Path]::GetFullPath($ModelDir)
$env:AIIMAGE_QWEN35_IMAGE = [IO.Path]::GetFullPath($ImagePath)
$env:AIIMAGE_QWEN35_MAX_NEW_TOKENS = $MaxNewTokens.ToString([Globalization.CultureInfo]::InvariantCulture)
$env:AIIMAGE_QWEN35_REQUIRE_OCR_MARKERS = if ($RequireOcrMarkers) { '1' } else { '0' }
$env:AIIMAGE_QWEN35_MULTIMODAL_REPORT = [IO.Path]::GetFullPath($UnityReport)
$env:AIIMAGE_QWEN35_UNITY_LOG = [IO.Path]::GetFullPath($UnityLog)
if ([string]::IsNullOrWhiteSpace($Prompt)) {
    Remove-Item Env:AIIMAGE_QWEN35_IMAGE_PROMPT -ErrorAction SilentlyContinue
} else {
    $env:AIIMAGE_QWEN35_IMAGE_PROMPT = $Prompt
}

$unityArgs = @(
    '-batchmode',
    '-quit',
    '-projectPath', [IO.Path]::GetFullPath($ProjectPath),
    '-executeMethod', 'NcnnDebugRunner.RunQwen35MultimodalGenerationBatch',
    '-logFile', [IO.Path]::GetFullPath($UnityLog)
)
$stopwatch = [Diagnostics.Stopwatch]::StartNew()
$peakWorkingSet = 0L
$peakPrivate = 0L
$process = Start-Process `
    -FilePath $UnityExe `
    -ArgumentList $unityArgs `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -PassThru `
    -WindowStyle Hidden
# Retain the OS handle before Unity exits; otherwise Windows PowerShell can
# lose ExitCode when the Process handle is opened lazily after termination.
$processHandle = $process.Handle

while (-not $process.HasExited) {
    try {
        $process.Refresh()
        $peakWorkingSet = [Math]::Max($peakWorkingSet, $process.WorkingSet64)
        $peakPrivate = [Math]::Max($peakPrivate, $process.PrivateMemorySize64)
    } catch { }
    Start-Sleep -Milliseconds 250
}
$process.WaitForExit()
$process.Refresh()
$exitCode = $process.ExitCode
if ($null -eq $exitCode) {
    throw "Unity exited without an observable process exit code (pid=$($process.Id), handle=$processHandle)."
}
$exitCode = [int]$exitCode
$stopwatch.Stop()

$unityReportObject = $null
$unityReportError = ''
try {
    $unityReportObject = Get-Content -LiteralPath $UnityReport -Raw -Encoding UTF8 | ConvertFrom-Json
} catch {
    $unityReportError = $_.Exception.ToString()
}
$unityValid = $null -ne $unityReportObject -and [bool]$unityReportObject.valid
$validation = [ordered]@{
    schema = 'qwen35.unity.batch-validation/v1'
    valid = ($exitCode -eq 0 -and $unityValid)
    unity_executable = [IO.Path]::GetFullPath($UnityExe)
    project_path = [IO.Path]::GetFullPath($ProjectPath)
    execute_method = 'NcnnDebugRunner.RunQwen35MultimodalGenerationBatch'
    command = @($UnityExe) + $unityArgs
    process_id = $process.Id
    exit_code = $exitCode
    elapsed_ms = $stopwatch.ElapsedMilliseconds
    peak_working_set_bytes = $peakWorkingSet
    peak_private_bytes = $peakPrivate
    stdout_path = [IO.Path]::GetFullPath($stdoutPath)
    stderr_path = [IO.Path]::GetFullPath($stderrPath)
    stdout = if (Test-Path -LiteralPath $stdoutPath) { [IO.File]::ReadAllText([IO.Path]::GetFullPath($stdoutPath)) } else { '' }
    stderr = if (Test-Path -LiteralPath $stderrPath) { [IO.File]::ReadAllText([IO.Path]::GetFullPath($stderrPath)) } else { '' }
    unity_log = [IO.Path]::GetFullPath($UnityLog)
    unity_log_sha256 = if (Test-Path -LiteralPath $UnityLog) { Get-Sha256WithRetry ([IO.Path]::GetFullPath($UnityLog)) } else { '' }
    unity_report = [IO.Path]::GetFullPath($UnityReport)
    unity_report_sha256 = if (Test-Path -LiteralPath $UnityReport) { Get-Sha256WithRetry ([IO.Path]::GetFullPath($UnityReport)) } else { '' }
    unity_report_valid = $unityValid
    unity_report_error = $unityReportError
    strict_texture_execution = if ($null -ne $unityReportObject) { [bool]$unityReportObject.strict_texture_execution } else { $false }
    compute_buffer_fallback = if ($null -ne $unityReportObject) { [bool]$unityReportObject.compute_buffer_fallback } else { $true }
    marker_hit_count = if ($null -ne $unityReportObject) { [int]$unityReportObject.marker_hit_count } else { 0 }
    marker_group_count = if ($null -ne $unityReportObject) { [int]$unityReportObject.marker_group_count } else { 0 }
}

$validation | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $ValidationReport -Encoding UTF8
$validation | ConvertTo-Json -Depth 4
if (-not $validation.valid) {
    exit 2
}
exit 0
