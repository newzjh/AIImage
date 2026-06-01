# Matting NCNN Repro Handoff

Updated: 2026-06-01

## Goal

Reproduce the matting pipeline from:

- `ref/ncnn_matting-main`
- `ref/MODNet-master`

inside Unity with a separable compute path, while:

- not touching `NcnnRepro` / `NcnnRepro2`
- using new `MatterNcnnReproRunner` / `NcnnRepro3`
- preferring `pack4 RenderTexture` where correctness allows

## Current Key Files

Runtime:

- `Assets/Scripts/MatterNcnnReproRunner.cs`
- `Assets/Scripts/NcnnCompute/NcnnRepro3.cs`
- `Assets/Scripts/NcnnCompute/NcnnOps.cs`
- `Assets/Resources/NcnnCompute.compute`

Debug entry:

- `Assets/Editor/NcnnDebugRunner.cs`
  - `RunMattingDebugMenu`
  - `RunMattingDebugBatch`

Models and samples:

- `Assets/StreamingAssets/Matting/matting.param`
- `Assets/StreamingAssets/Matting/matting.bin`
- `ref/ncnn_matting-main/test_img.jpg`
- `ref/ncnn_matting-main/test_result.jpg`
- `ref/ncnn_matting-main/windows/matting.dll`
- `ref/ncnn_matting-main/windows/seg_out.jpg`
- `ref/CodeFormer-ncnn-main/data/03.jpg`

## Official Windows DLL Notes

`ref/ncnn_matting-main/windows/matting.dll` exports:

- `Init`
- `MakeUpFile`

It can be called silently with Python `ctypes`:

```python
import os, ctypes
root = r"E:\Projects\AIImage\ref\ncnn_matting-main\windows"
os.chdir(root)
os.add_dll_directory(root)
dll = ctypes.WinDLL(os.path.join(root, "matting.dll"))
dll.Init()
dll.MakeUpFile(r"E:\Projects\AIImage\ref\ncnn_matting-main\test_img.jpg".encode("utf-8"))
```

Output is written to:

- `ref/ncnn_matting-main/windows/seg_out.jpg`

Important: the Windows reference path is CPU `float32` style, not a Vulkan/half path.

## What Has Been Implemented

### New isolated graph runner

`NcnnRepro3` covers the layer set used by this matting model:

- `Input`
- `Split`
- `Convolution`
- `BinaryOp`
- `ReLU`
- `Pooling`
- `MaxPoolingInd`
- `MaxUnPooling`
- `Interp`
- `Concat`

### Added compute capabilities

Incremental additions in `NcnnOps.cs` / `NcnnCompute.compute`:

- convolution builtin `sigmoid`
- generic pack4 `Interp`
- `MaxPoolingIndPack4`
- `MaxPoolingIndicesFromValuePack4`
- `MaxUnPoolingPack4`

### Runner defaults for best correctness

Current default values in `MatterNcnnReproRunner`:

- `preserveAspectRatioInput = false`
- `useArgbFloatTensor = true`
- `forceBufferConvolution = false`
- `useTextureMaxPoolingInd = false`
- `enableWinograd23 = false`
- foreground cleanup enabled

Foreground cleanup currently does:

1. largest foreground connected component from alpha threshold
2. small binary close on that support
3. gray alpha close inside the retained support

## Current Best Correctness Path

The current best stable path is:

- `MaxPooling`: pack4 texture
- `MaxPoolingInd`: CPU/buffer reference implementation
- `MaxUnPooling`: CPU/buffer reference implementation
- `Convolution`: pack4 texture
- `Winograd23`: off
- `TensorTextureFormat`: `ARGBFloat`

This is intentional. It is the most correct path found so far.

## Verified Results

### `test_img.jpg`

Best recent stable result against `ref/ncnn_matting-main/test_result.jpg`:

- `mean_abs_rgb ~= 3.11`
- `max_abs_rgb ~= 215`

This is the current baseline to preserve.

Background dirty blobs are largely gone.

### `03.jpg`

Official reference for comparison is the DLL-generated:

- `ref/ncnn_matting-main/windows/seg_out.jpg`

Current Unity result for `03.jpg` is still much worse than official.
Observed error was around:

- `mean_abs ~= 27.8` with fixed-square input path
- `mean_abs ~= 23.5` when temporarily testing aspect-preserving input

So `03.jpg` is still not aligned.

## Important Conclusions Already Proven

### 1. `Winograd23` hurts this model

On this matting graph, enabling Winograd degrades results badly.

Conclusion:

- keep `enableWinograd23 = false`

### 2. `ARGBFloat` is currently required for alignment

Half precision previously caused saturation and background corruption.

Conclusion:

- do not switch back to `ARGBHalf` yet
- only revisit after the remaining graph mismatches are fixed

### 3. Pack4 convolution path is basically aligned on the good path

Using `conv_compare`, many convolution layers match the float buffer reference at roughly `1e-6` or better when the stable path is used.

Conclusion:

- the main remaining problem is no longer the bulk convolution trunk

### 4. CPU `MaxPoolingInd` is correct

`maxpool_compare` has shown:

- `MaxPool_19`
- `MaxPool_41`

can match reference exactly under the CPU/buffer implementation.

### 5. Texture `MaxPoolingInd` is not yet ready to replace CPU

Tried variants:

1. single-pass texture pooled value + indices
2. two-pass:
   - `PoolingPack4`
   - `MaxPoolingIndicesFromValuePack4`
3. split SRV/UAV style index lookup pass

These variants still regressed strongly in full-run validation.
The full image error would jump back toward `58.x`.

Conclusion:

- keep default `useTextureMaxPoolingInd = false`
- texture `MaxPoolingInd` is still an active debug branch, not production-ready

## Debug Methods

### Unity headless batch

```powershell
$env:AIIMAGE_DEBUG_INPUT='E:\Projects\AIImage\ref\ncnn_matting-main\test_img.jpg'
& 'C:\Program Files\Unity 6000.2.7f2\Editor\Unity.exe' `
  -batchmode `
  -projectPath 'E:\Projects\AIImage' `
  -executeMethod NcnnDebugRunner.RunMattingDebugBatch `
  -logFile 'E:\Projects\AIImage\Logs\matting-test.log'
```

For `03.jpg`:

```powershell
$env:AIIMAGE_DEBUG_INPUT='E:\Projects\AIImage\ref\CodeFormer-ncnn-main\data\03.jpg'
& 'C:\Program Files\Unity 6000.2.7f2\Editor\Unity.exe' `
  -batchmode `
  -projectPath 'E:\Projects\AIImage' `
  -executeMethod NcnnDebugRunner.RunMattingDebugBatch `
  -logFile 'E:\Projects\AIImage\Logs\matting-03.log'
```

### Output locations

Result images:

- `%LOCALAPPDATA%\Temp\YanQi\AIImage\AIImage_MattingRepro_*`

Intermediate dumps:

- `%LOCALAPPDATA%\Temp\YanQi\AIImage\AIImage_MattingBlobDump_*`

Useful files:

- `17_composite.png`
- `18_matte.png`
- `blob_stats.txt`
- `conv_compare.txt`

### Existing compare tools in runner

`MatterNcnnReproRunner` already supports:

- blob dumps
- texture-conv compare
- maxpool compare

These are useful when temporarily enabled from `NcnnDebugRunner`.

## Current Open Problems

### Problem A: `03.jpg` still differs strongly from official

The current cleanup that helps `test_img.jpg` does not solve `03.jpg`.
This means the remaining issue is not just simple blob cleanup.

Strong suspicion:

- there is still an algorithmic mismatch versus the official graph behavior on more complex portraits
- likely around later decoder behavior / unpool semantics / input policy interactions

### Problem B: texture `MaxPoolingInd` still not aligned

Even after multiple attempts, the texture path still causes catastrophic regression in end-to-end runs.

This is the main unresolved pack4 item.

## Recommended Next Steps

### 1. Focus on `03.jpg` with full graph compare

Run the good path on `03.jpg` with:

- `enableTextureConvCompare = true`
- `enableMaxPoolingCompare = true`

and determine whether intermediate layers still match reference.

If intermediate graph still matches well, the remaining gap is likely final alpha shaping / unpool semantics.

### 2. Continue rebuilding texture `MaxPoolingInd`

Recommended order:

1. add standalone compare for `PoolingPack4` itself
2. verify whether pooled values are correct independently of indices
3. then compare only the texture index lookup pass versus CPU indices
4. only switch default to texture once `maxpool_compare mean_abs = 0`
   and end-to-end error stays around the current good baseline

### 3. Only after graph alignment, revisit `ARGBHalf`

Suggested rollback test:

1. keep `Winograd23 = false`
2. keep the same graph path otherwise
3. switch `TensorTextureFormat` from `ARGBFloat` to `ARGBHalf`
4. re-run both:
   - `test_img.jpg`
   - `03.jpg`
5. reject the change if dirty blobs / local miscuts return

## Critical Facts For Next Agent / IDE

1. The project does have a good stable path already. Do not assume it is still stuck at `58.x`.
2. `enableWinograd23` should stay `false`.
3. `ARGBFloat` is currently deliberate, not accidental.
4. `MaxPooling` is already pack4 texture and correct.
5. `MaxPoolingInd` texture path is still not validated and must not replace CPU by default yet.
6. `03.jpg` is the current best discriminator for remaining algorithm mismatch.
