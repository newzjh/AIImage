param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '..\Documents\AssetStore\MarketingImages')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if ([System.IO.Path]::IsPathRooted($OutputDirectory)) {
    $output = [System.IO.Path]::GetFullPath($OutputDirectory)
}
else {
    $output = [System.IO.Path]::GetFullPath((Join-Path $root $OutputDirectory))
}
[System.IO.Directory]::CreateDirectory($output) | Out-Null

$colors = @{
    Canvas = [System.Drawing.Color]::FromArgb(255, 15, 19, 23)
    Panel = [System.Drawing.Color]::FromArgb(255, 25, 31, 37)
    Border = [System.Drawing.Color]::FromArgb(255, 62, 72, 82)
    Text = [System.Drawing.Color]::FromArgb(255, 244, 247, 249)
    Muted = [System.Drawing.Color]::FromArgb(255, 169, 181, 192)
    Teal = [System.Drawing.Color]::FromArgb(255, 55, 207, 189)
    Coral = [System.Drawing.Color]::FromArgb(255, 244, 133, 99)
    Blue = [System.Drawing.Color]::FromArgb(255, 87, 153, 255)
}

function Get-Image([string]$Path) {
    return [System.Drawing.Image]::FromFile((Join-Path $root $Path))
}

function New-Canvas([int]$Width, [int]$Height) {
    $bitmap = [System.Drawing.Bitmap]::new($Width, $Height, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    return @($bitmap, $graphics)
}

function Get-Font([float]$Size, [bool]$Bold = $false) {
    $style = if ($Bold) { [System.Drawing.FontStyle]::Bold } else { [System.Drawing.FontStyle]::Regular }
    return [System.Drawing.Font]::new('Segoe UI', $Size, $style, [System.Drawing.GraphicsUnit]::Pixel)
}

function Draw-Grid([System.Drawing.Graphics]$Graphics, [int]$Width, [int]$Height) {
    $Graphics.Clear($colors.Canvas)
    $gridPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(48, 120, 138, 150), 1)
    for ($x = 0; $x -le $Width; $x += 64) { $Graphics.DrawLine($gridPen, $x, 0, $x, $Height) }
    for ($y = 0; $y -le $Height; $y += 64) { $Graphics.DrawLine($gridPen, 0, $y, $Width, $y) }
    $gridPen.Dispose()

    $tealPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(180, $colors.Teal), 5)
    $coralPen = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(160, $colors.Coral), 5)
    $Graphics.DrawLine($tealPen, 0, 8, [Math]::Min($Width, 550), 8)
    $Graphics.DrawLine($coralPen, [Math]::Max(0, $Width - 320), $Height - 8, $Width, $Height - 8)
    $tealPen.Dispose(); $coralPen.Dispose()
}

function Draw-Text([System.Drawing.Graphics]$Graphics, [string]$Text, [float]$Size, [System.Drawing.Color]$Color, [float]$X, [float]$Y, [bool]$Bold = $false) {
    $font = Get-Font $Size $Bold
    $brush = [System.Drawing.SolidBrush]::new($Color)
    $Graphics.DrawString($Text, $font, $brush, $X, $Y)
    $brush.Dispose(); $font.Dispose()
}

function Draw-ImageContained([System.Drawing.Graphics]$Graphics, [System.Drawing.Image]$Image, [int]$X, [int]$Y, [int]$Width, [int]$Height) {
    $scale = [Math]::Min($Width / $Image.Width, $Height / $Image.Height)
    $drawWidth = [int][Math]::Round($Image.Width * $scale)
    $drawHeight = [int][Math]::Round($Image.Height * $scale)
    $drawX = $X + [int](($Width - $drawWidth) / 2)
    $drawY = $Y + [int](($Height - $drawHeight) / 2)
    $Graphics.DrawImage($Image, $drawX, $drawY, $drawWidth, $drawHeight)
}

function Draw-Panel([System.Drawing.Graphics]$Graphics, [int]$X, [int]$Y, [int]$Width, [int]$Height) {
    $brush = [System.Drawing.SolidBrush]::new($colors.Panel)
    $pen = [System.Drawing.Pen]::new($colors.Border, 2)
    $rect = [System.Drawing.Rectangle]::new($X, $Y, $Width, $Height)
    $Graphics.FillRectangle($brush, $rect)
    $Graphics.DrawRectangle($pen, $rect)
    $brush.Dispose(); $pen.Dispose()
}

function Save-Png([System.Drawing.Bitmap]$Bitmap, [string]$Name) {
    $Bitmap.Save((Join-Path $output $Name), [System.Drawing.Imaging.ImageFormat]::Png)
    $Bitmap.Dispose()
}

function New-Showcase([string]$FileName, [string]$Title, [string]$Subtitle, [string]$ImagePath, [string]$Evidence, [string]$Caution = '') {
    $parts = New-Canvas 1920 1080
    $bitmap = $parts[0]; $graphics = $parts[1]
    Draw-Grid $graphics 1920 1080
    Draw-Text $graphics 'AEXIS  /  RUNNER SHOWCASE' 24 $colors.Teal 72 52 $true
    Draw-Text $graphics $Title 52 $colors.Text 72 96 $true
    Draw-Text $graphics $Subtitle 24 $colors.Muted 72 165
    Draw-Panel $graphics 72 235 1776 675
    $image = Get-Image $ImagePath
    Draw-ImageContained $graphics $image 100 263 1720 619
    $image.Dispose()
    Draw-Text $graphics $Evidence 20 $colors.Muted 72 952
    if ($Caution.Length -gt 0) { Draw-Text $graphics $Caution 19 $colors.Coral 72 990 $true }
    Draw-Text $graphics 'Repository runner artifact. Environment-specific evidence; not a universal performance or quality claim.' 16 $colors.Muted 72 1036
    $graphics.Dispose()
    Save-Png $bitmap $FileName
}

function New-ResultShowcase([string]$FileName, [string]$Title, [string]$Subtitle, [string]$ImagePath, [string[]]$Rows, [string]$Evidence) {
    $parts = New-Canvas 1920 1080
    $bitmap = $parts[0]; $graphics = $parts[1]
    Draw-Grid $graphics 1920 1080
    Draw-Text $graphics 'AEXIS  /  RUNNER SHOWCASE' 24 $colors.Teal 72 52 $true
    Draw-Text $graphics $Title 52 $colors.Text 72 96 $true
    Draw-Text $graphics $Subtitle 24 $colors.Muted 72 165
    Draw-Panel $graphics 72 235 820 620
    Draw-Panel $graphics 940 235 908 620
    Draw-Text $graphics 'INPUT' 22 $colors.Muted 108 263 $true
    Draw-Text $graphics 'OUTPUT / REPORT' 22 $colors.Muted 976 263 $true
    $image = Get-Image $ImagePath
    Draw-ImageContained $graphics $image 108 310 748 500
    $image.Dispose()
    $rowY = 350
    foreach ($row in $Rows) {
        Draw-Text $graphics $row 30 $colors.Text 990 $rowY $true
        $rowY += 78
    }
    Draw-Text $graphics $Evidence 20 $colors.Muted 72 928
    Draw-Text $graphics 'Recorded runner evidence. Exact model, device, and profile determine results.' 16 $colors.Muted 72 988
    $graphics.Dispose()
    Save-Png $bitmap $FileName
}

function New-ProcessShowcase([string]$FileName, [string]$Title, [string]$Subtitle, [string]$LeftPath, [string]$LeftLabel, [string]$RightPath, [string]$RightLabel, [string]$Evidence, [string]$Caution = '') {
    $parts = New-Canvas 1920 1080
    $bitmap = $parts[0]; $graphics = $parts[1]
    Draw-Grid $graphics 1920 1080
    Draw-Text $graphics 'AEXIS  /  RUNNER SHOWCASE' 24 $colors.Teal 72 52 $true
    Draw-Text $graphics $Title 52 $colors.Text 72 96 $true
    Draw-Text $graphics $Subtitle 24 $colors.Muted 72 165
    Draw-Panel $graphics 72 235 850 620
    Draw-Panel $graphics 998 235 850 620
    Draw-Text $graphics $LeftLabel 22 $colors.Muted 108 263 $true
    Draw-Text $graphics $RightLabel 22 $colors.Muted 1034 263 $true
    $left = Get-Image $LeftPath; $right = Get-Image $RightPath
    Draw-ImageContained $graphics $left 108 310 778 500
    Draw-ImageContained $graphics $right 1034 310 778 500
    $left.Dispose(); $right.Dispose()
    Draw-Text $graphics $Evidence 20 $colors.Muted 72 928
    if ($Caution.Length -gt 0) { Draw-Text $graphics $Caution 19 $colors.Coral 72 970 $true }
    Draw-Text $graphics 'Repository runner artifacts. Retain visible residual artifacts and model-specific limitations.' 16 $colors.Muted 72 1024
    $graphics.Dispose()
    Save-Png $bitmap $FileName
}

function New-Cover([int]$Width, [int]$Height, [string]$Name, [bool]$IncludeUi) {
    $parts = New-Canvas $Width $Height
    $bitmap = $parts[0]; $graphics = $parts[1]
    Draw-Grid $graphics $Width $Height
    $chipSize = [int]($Height * 0.18); $chipX = [int]($Width * 0.08); $chipY = [int]($Height * 0.17)
    $chipBrush = [System.Drawing.SolidBrush]::new($colors.Panel)
    $chipPen = [System.Drawing.Pen]::new($colors.Teal, [Math]::Max(2, [int]($Height / 250)))
    $graphics.FillRectangle($chipBrush, $chipX, $chipY, $chipSize, $chipSize)
    $graphics.DrawRectangle($chipPen, $chipX, $chipY, $chipSize, $chipSize)
    $chipBrush.Dispose(); $chipPen.Dispose()
    $tile = [int]($chipSize * 0.32)
    for ($i = 0; $i -lt 4; $i++) {
        if ($i % 2 -eq 0) {
            $tileColor = $colors.Teal
        }
        else {
            $tileColor = $colors.Coral
        }
        $tileBrush = [System.Drawing.SolidBrush]::new($tileColor)
        $tileX = $chipX + [int]($chipSize * 0.18) + (($i % 2) * [int]($chipSize * 0.34))
        $tileY = $chipY + [int]($chipSize * 0.18) + ([int]($i / 2) * [int]($chipSize * 0.34))
        $graphics.FillRectangle($tileBrush, $tileX, $tileY, $tile, $tile)
        $tileBrush.Dispose()
    }
    Draw-Text $graphics 'AEXIS' ([int]($Height * 0.115)) $colors.Text ([int]($Width * 0.08)) ([int]($Height * 0.41)) $true
    Draw-Text $graphics 'ON-DEVICE INFERENCE FOR UNITY' ([int]($Height * 0.026)) $colors.Teal ([int]($Width * 0.083)) ([int]($Height * 0.54)) $true
    Draw-Text $graphics 'ONNX AND NCNN  |  TEXTURE-NATIVE GPU PATH  |  UPM PACKAGE' ([int]($Height * 0.018)) $colors.Muted ([int]($Width * 0.083)) ([int]($Height * 0.61))
    if ($IncludeUi) {
        Draw-Panel $graphics ([int]($Width * 0.59)) ([int]($Height * 0.14)) ([int]($Width * 0.32)) ([int]($Height * 0.72))
        $ui = Get-Image 'tmp/aiimage-main2-fence.png'
        Draw-ImageContained $graphics $ui ([int]($Width * 0.61)) ([int]($Height * 0.17)) ([int]($Width * 0.28)) ([int]($Height * 0.66))
        $ui.Dispose()
    }
    Draw-Text $graphics 'ENGINEERED FOR REAL-TIME UNITY APPLICATIONS' ([int]($Height * 0.018)) $colors.Muted ([int]($Width * 0.083)) ([int]($Height * 0.87))
    $graphics.Dispose()
    Save-Png $bitmap $Name
}

function New-Icon() {
    $parts = New-Canvas 160 160
    $bitmap = $parts[0]; $graphics = $parts[1]
    $graphics.Clear($colors.Canvas)
    $brush = [System.Drawing.SolidBrush]::new($colors.Panel)
    $pen = [System.Drawing.Pen]::new($colors.Teal, 3)
    $graphics.FillRectangle($brush, 18, 18, 124, 124); $graphics.DrawRectangle($pen, 18, 18, 124, 124)
    $brush.Dispose(); $pen.Dispose()
    $tiles = @(@(42,42,$colors.Teal), @(82,42,$colors.Coral), @(42,82,$colors.Coral), @(82,82,$colors.Teal))
    foreach ($entry in $tiles) { $tileBrush = [System.Drawing.SolidBrush]::new($entry[2]); $graphics.FillRectangle($tileBrush, $entry[0], $entry[1], 34, 34); $tileBrush.Dispose() }
    $graphics.Dispose()
    Save-Png $bitmap 'aexis-icon.png'
}

function New-Card() {
    $parts = New-Canvas 420 280
    $bitmap = $parts[0]; $graphics = $parts[1]
    Draw-Grid $graphics 420 280
    $chipBrush = [System.Drawing.SolidBrush]::new($colors.Panel)
    $chipPen = [System.Drawing.Pen]::new($colors.Teal, 3)
    $graphics.FillRectangle($chipBrush, 40, 58, 92, 92)
    $graphics.DrawRectangle($chipPen, 40, 58, 92, 92)
    $chipBrush.Dispose(); $chipPen.Dispose()
    $tiles = @(@(54,72,$colors.Teal), @(87,72,$colors.Coral), @(54,105,$colors.Coral), @(87,105,$colors.Teal))
    foreach ($entry in $tiles) { $tileBrush = [System.Drawing.SolidBrush]::new($entry[2]); $graphics.FillRectangle($tileBrush, $entry[0], $entry[1], 27, 27); $tileBrush.Dispose() }
    Draw-Text $graphics 'AEXIS' 48 $colors.Text 155 70 $true
    Draw-Text $graphics 'ON-DEVICE INFERENCE' 15 $colors.Teal 158 126 $true
    Draw-Text $graphics 'ONNX  |  NCNN  |  GPU' 15 $colors.Muted 40 194 $true
    Draw-Text $graphics 'FOR UNITY' 15 $colors.Muted 40 221 $true
    $graphics.Dispose()
    Save-Png $bitmap 'aexis-card.png'
}

New-Cover 1950 1300 'aexis-cover.png' $true
New-Cover 1200 630 'aexis-social.png' $true
New-Card
New-Icon

New-Showcase 'showcase-codeformer-face-restoration.png' 'CodeFormer Face Restoration' 'Before and after from the documented runner artifact.' 'Packages/com.aexis/Documentation~/images/codeformer-03-before-after.png' 'Windows / Vulkan / Unity 6000.2.7f2 / 2026-07-29 / 17,958 ms on ref/03.jpg'
New-Showcase 'showcase-realesrgan-x4-upscaling.png' 'Real-ESRGAN AnimeVideo v3 x4' 'Before and after from the documented Pack4-only validation.' 'Packages/com.aexis/Documentation~/images/realesrgan-03-before-after.png' 'Windows / Vulkan / Unity 6000.2.7f2 / 2026-07-29 / 1,057 ms on ref/03.jpg'
New-ProcessShowcase 'showcase-foreground-matting.png' 'Foreground Matting' 'Input processing produces an alpha matte and a composited foreground result.' 'Packages/com.aexis/Documentation~/images/matting-matte.png' 'ALPHA MATTE OUTPUT' 'Packages/com.aexis/Documentation~/images/matting-composite.png' 'COMPOSITED OUTPUT' 'Strict texture plan and CommandBuffer path / 360 x 202 / 1,103 ms.'
New-ProcessShowcase 'showcase-yolo-person-segmentation.png' 'YOLO Person Segmentation' 'Person-mask overlay followed by the documented masked-image output.' 'Packages/com.aexis/Documentation~/images/yolo-deepfill-overlay.png' 'PERSON MASK OVERLAY' 'Packages/com.aexis/Documentation~/images/yolo-deepfill-output.png' 'MASKED IMAGE OUTPUT' 'YOLO segmentation output used by the documented DeepFillV2 workflow.'
New-Showcase 'showcase-yolo-deepfillv2-inpainting.png' 'YOLO + DeepFillV2 Inpainting' 'Before and after on the documented beach input.' 'Packages/com.aexis/Documentation~/images/yolo-deepfillv2-3-before-after.png' 'Windows / Vulkan / Unity 6000.2.7f2 / 2026-07-29 / 2,196 ms total / seven persons detected.' 'Raw runner output; residual removal artifacts are intentionally retained.'
New-Showcase 'showcase-yolo-sd-inpainting.png' 'YOLO + Stable Diffusion Inpainting' 'Before and after on the documented beach input.' 'Packages/com.aexis/Documentation~/images/yolo-sd-inpainting-3-before-after.png' 'Windows / Vulkan / Unity 6000.2.7f2 / 2026-07-29 / 12 steps / 630,213 ms.' 'External model weights. Raw runner output; residual artifacts are intentionally retained.'
New-ResultShowcase 'showcase-clip-mobileclip-s0.png' 'CLIP MobileCLIP S0' 'Image input and the latest recorded ranked-label result.' 'Packages/com.aexis/Documentation~/images/qwen-and-clip-input.jpg' @('Photo    0.332389', 'Portrait  0.265945', 'Embedding and ranked labels', 'Strict texture runner evidence') 'Recorded successful score artifact: 2026-07-23. Current strict CommandBuffer profile rejects an undeclared temporary RT at transpose_121.'
New-ResultShowcase 'showcase-qwen35-multimodal.png' 'Qwen3.5 Mobile Q4 / Q8' 'Multimodal image input and recorded generation report.' 'Packages/com.aexis/Documentation~/images/qwen-and-clip-input.jpg' @('Q4 text smoke: passed', '65.533 s / 48 cache textures', 'Q8 multimodal: passed', '71.618 s / 6 generated tokens', 'Android Q4: 397 visible chars') 'Qwen variants are external model archives. Q4/Q8 labels describe model variants, not generic engine-wide quantization support.'
New-Showcase 'showcase-gfpgan-face-restoration.png' 'GFPGAN Face Restoration' 'Before and after from the documented runner artifact.' 'Packages/com.aexis/Documentation~/images/gfpgan-03-before-after.png' 'Windows / Vulkan / Unity 6000.2.7f2 / 2026-07-29 / 6,053 ms on ref/03.jpg' 'Raw execution evidence only: this documented input visibly distorts, so it is not a quality claim.'

Write-Host "Generated Asset Store images in $output"
