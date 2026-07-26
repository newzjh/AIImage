param(
    [string]$ReleaseTag = 'model',
    [string]$AssetDirectory = 'Builds\ReducedModelRelease'
)

$ErrorActionPreference = 'Stop'
$repository = 'newzjh/AIImage'

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw 'GitHub CLI (gh) is required to upload the generated model release assets.'
}

$root = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $root
$directory = [IO.Path]::GetFullPath((Join-Path $root $AssetDirectory))
if (-not (Test-Path -LiteralPath $directory)) {
    throw "Model release asset directory does not exist: $directory"
}

$assets = Get-ChildItem -LiteralPath $directory -File -Filter 'AIImageModels.*.zip' | ForEach-Object FullName
$manifest = Join-Path $directory 'AIImageModelReleaseManifest.json'
if (-not (Test-Path -LiteralPath $manifest)) {
    throw "Model release manifest does not exist: $manifest"
}
if ($assets.Count -eq 0) {
    throw "No AIImage model archives were found in: $directory"
}

gh release view $ReleaseTag --repo $repository 2>$null
if ($LASTEXITCODE -ne 0) {
    gh release create $ReleaseTag --repo $repository --title $ReleaseTag --notes 'AIImage on-demand model archives.'
}

gh release upload $ReleaseTag @assets $manifest --clobber --repo $repository
