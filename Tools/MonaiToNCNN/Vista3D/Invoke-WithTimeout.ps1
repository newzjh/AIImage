param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath,

    [string[]]$ArgumentList = @(),

    [string]$ArgumentLine = "",

    [int]$TimeoutSeconds = 600,

    [string]$WorkingDirectory = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($WorkingDirectory)) {
    $WorkingDirectory = (Get-Location).Path
}

if (-not (Test-Path -LiteralPath $FilePath)) {
    throw "File not found: $FilePath"
}

$resolvedArgumentList = if (-not [string]::IsNullOrWhiteSpace($ArgumentLine)) {
    $ArgumentLine
}
else {
    $ArgumentList
}

$process = Start-Process `
    -FilePath $FilePath `
    -ArgumentList $resolvedArgumentList `
    -WorkingDirectory $WorkingDirectory `
    -WindowStyle Hidden `
    -PassThru

try {
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        try {
            $process.Kill($true)
        }
        catch {
        }
        Write-Error "Process timed out after $TimeoutSeconds seconds: $FilePath"
        exit 124
    }

    exit $process.ExitCode
}
finally {
    try {
        if (-not $process.HasExited) {
            $process.Kill($true)
        }
    }
    catch {
    }
}
