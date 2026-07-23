param(
    [string]$Manifest = "",
    [string]$ObservationRoot = "",
    [string]$OutputDir = "",
    [string]$InjectPerturbation = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$repoRoot = Split-Path -Parent $root
$tool = Join-Path $PSScriptRoot "golden_regression.py"

if ([string]::IsNullOrWhiteSpace($Manifest)) {
    $Manifest = Join-Path $PSScriptRoot "manifests"
}
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repoRoot "output\golden-regression"
}

$arguments = @($tool, "--manifest", $Manifest, "--output-dir", $OutputDir)
if (-not [string]::IsNullOrWhiteSpace($ObservationRoot)) {
    $arguments += @("--observation-root", $ObservationRoot)
}
if (-not [string]::IsNullOrWhiteSpace($InjectPerturbation)) {
    $arguments += @("--inject-perturbation", $InjectPerturbation)
}

& python @arguments
exit $LASTEXITCODE
