# Aexis

Aexis is a self-contained, on-device inference engine for Unity. It imports and executes selected ONNX and NCNN model graphs on a texture-native GPU path, with compute shaders and Pack4 RenderTexture storage designed for real-time applications on edge devices.

`com.aexis` is the engine package. AIImage is the complete application example built on top of it; it is not a runtime namespace or package dependency. The runtime does not wrap or require Unity Sentis, Tencent ncnn, ONNX Runtime, MNN, UniTask, or a native inference plug-in.

- **Current package:** `com.aexis` `0.1.0-pre.1`
- **Unity range:** 2022.3 through 6000.3
- **Primary production path:** Pack4 RenderTexture plus CommandBuffer-compatible compute dispatch

## What Aexis Provides

- A self-owned ONNX protobuf reader, graph lowering, shape/index planner, and ONNX execution path.
- A self-owned NCNN `.param` and `.bin` reader, graph loader, and texture-native Pack4 backend.
- Package-owned compute shaders loaded from `Runtime/Resources/Aexis`.
- Import support for `.onnx`, NCNN `.param`/`.bin`, and versioned `.aexis` (`AEXM`) model archives.
- Public runtime assemblies: `Aexis`, `Aexis.Async`, `Aexis.Onnx`, `Aexis.Ncnn`, and `Aexis.Execution`.
- Importable samples, including the full **AIImage Main2 Application Example** and reusable runner components.

The engine keeps logical tensor shapes separate from physical storage shapes. In the NCNN production path, activations remain texture-backed. A strict plan rejects unsupported or unplanned command-buffer allocations instead of silently falling back to a transient ComputeBuffer activation.

## Examples

The images below are documentation copies of actual runner artifacts. Each timing names its test input and capture date. They are evidence for the listed input and environment, not universal performance claims.

### Qwen3.5 0.8B Mobile Q4 and Q8

![Qwen and CLIP multimodal input](Packages/com.aexis/Documentation~/images/qwen-and-clip-input.jpg)

| Variant | Runner evidence | Strict texture execution | Result |
| --- | --- | --- | --- |
| Q4 | Windows text smoke: 65.533 s and 48 cache textures; Android MuMu Vulkan Main2 pass: 326.821 s and 103 decoder steps | Yes; no ComputeBuffer fallback | Both runs executed successfully. The Windows smoke generated only an end-of-turn token; the MuMu pass emitted 397 visible characters and detected four people. |
| Q8 | Multimodal smoke with the image above, 71.618 s, 6 generated tokens | Yes; texture-backed activations and no ComputeBuffer fallback | Valid execution; generated text begins with a Chinese-language phrase. |

The Q4 and Q8 labels describe mobile model archive formats. They do not imply that every engine operator supports generic INT4 or INT8 execution.

### CLIP MobileCLIP S0

CLIP produces an embedding and ranked labels rather than a transformed image. The latest successful score artifact ranked `Photo` at `0.332389` and `Portrait` at `0.265945` (2026-07-23). A new strict CommandBuffer run on 2026-07-28 deliberately failed at `transpose_121`: its loaded profile did not declare a required temporary `512x64 RHalf` RT. The rejection confirms the strict no-fallback guard and is tracked as a current runner limitation.

### CodeFormer Face Restoration

![CodeFormer before and after on 03.jpg](Packages/com.aexis/Documentation~/images/codeformer-03-before-after.png)

The face-restoration runner completed in **17,958 ms** on `ref/03.jpg`, detected one face, and composited the restoration shown above (Windows/Vulkan, 2026-07-29).

### Foreground Matting

![Matting composite result](Packages/com.aexis/Documentation~/images/matting-composite.png)

![Matting alpha result](Packages/com.aexis/Documentation~/images/matting-matte.png)

The matting runner completed at **360x202** in **1,103 ms** with the strict texture plan and CommandBuffer path.

### GFPGAN Face Restoration

![GFPGAN before and after on 03.jpg](Packages/com.aexis/Documentation~/images/gfpgan-03-before-after.png)

The GFPGAN runner completed in **6,053 ms** on `ref/03.jpg` with the strict Pack4 guard (Windows/Vulkan, 2026-07-29). This is the raw runner output: this input visibly produces facial distortion, so it is shown as execution evidence rather than a quality claim. Its weights are excluded from the package payload and are delivered separately.

### YOLO Person Segmentation plus DeepFillV2

![YOLO plus DeepFillV2 before and after on 3.png](Packages/com.aexis/Documentation~/images/yolo-deepfillv2-3-before-after.png)

On `ref/deepfillv2/DeepFillv2-main/test_data/3.png`, YOLO found seven people (mask coverage `0.042730`). The NCNN DeepFillV2 pass completed in **2,196 ms** (**1,995 ms** inference; Windows/Vulkan, 2026-07-29). The comparison intentionally retains the residual removal artifacts in the documented output.

### Real-ESRGAN AnimeVideo v3 x4

![Real-ESRGAN before and after on 03.jpg](Packages/com.aexis/Documentation~/images/realesrgan-03-before-after.png)

The Real-ESRGAN AnimeVideo v3 x4 CommandBuffer Pack4-only validation completed in **1,057 ms** for `ref/03.jpg` (Windows/Vulkan, 2026-07-29).

### YOLO plus Stable Diffusion Inpainting

![YOLO plus Stable Diffusion inpainting before and after on 3.png](Packages/com.aexis/Documentation~/images/yolo-sd-inpainting-3-before-after.png)

On `ref/deepfillv2/DeepFillv2-main/test_data/3.png`, YOLO found seven people (mask coverage `0.042730`) and the 12-step Stable Diffusion inpainting run completed in **630,213 ms** (Windows/Vulkan, 2026-07-29). It removes people more completely than the DeepFillV2 run above, but the documented raw output still contains residual objects and artifacts. Stable Diffusion inpainting weights are external and are not included in the package sample.

### MONAI and VISTA

> **Screenshot slot reserved for validated medical input.**
>
> No medical screenshot is published here. MONAI/VISTA weights and data are external to the package, and a visual sample will be added only after a reproducible run with distributable input and the applicable data/model permissions.

## Model Releases

The application resolves model assets from [newzjh/AIImage releases](https://github.com/newzjh/AIImage/releases). The release pages below are the corresponding download locations; select the current asset listed on the page or use the application's model-download UI. Generated archive names are taken from `AIImageModelReleaseManifest.json` produced by `Aexis/Release/Build Reduced/Prepare Model Release Assets` rather than guessed in documentation.

| Runner or model group | GitHub Release download page | Package payload |
| --- | --- | --- |
| Qwen3.5 mobile Q4 | [`qwen3.5_0.8b_mobile_q4`](https://github.com/newzjh/AIImage/releases/tag/qwen3.5_0.8b_mobile_q4) | External download. |
| Qwen3.5 mobile Q8 | [`qwen3.5_0.8b_mobile_q8`](https://github.com/newzjh/AIImage/releases/tag/qwen3.5_0.8b_mobile_q8) | External download. |
| CLIP, CodeFormer, Matting, and YOLO configuration | [`model`](https://github.com/newzjh/AIImage/releases/tag/model) | Group-dependent package/sample payload. |
| Real-ESRGAN | [`realesr`](https://github.com/newzjh/AIImage/releases/tag/realesr) | Default Real-ESRGAN NCNN pair is allowed only after provenance review. |
| GFPGAN | [`gfpgan`](https://github.com/newzjh/AIImage/releases/tag/gfpgan) | External download; not in the package sample. |
| DeepFillV2 | [`DeepFileV2`](https://github.com/newzjh/AIImage/releases/tag/DeepFileV2) | Default NCNN case is sample-dependent; verify its release record. |
| Stable Diffusion inpainting | [`sdinpainting`](https://github.com/newzjh/AIImage/releases/tag/sdinpainting) | External download; not in the package sample. |
| MONAI WholeBrain | [`MONAI_WholeBrain`](https://github.com/newzjh/AIImage/releases/tag/MONAI_WholeBrain) | External model; medical input and data remain excluded. |
| VISTA3D skull | [`vista3d_skull`](https://github.com/newzjh/AIImage/releases/tag/vista3d_skull) | External model; medical input and data remain excluded. |
| VISTA3D spine | [`vista3d_spine`](https://github.com/newzjh/AIImage/releases/tag/vista3d_spine) | External model; medical input and data remain excluded. |

Release tags are part of the application delivery configuration. A successful download does not grant a model license; see [model distribution](Packages/com.aexis/Documentation~/model-distribution.md) before redistributing a model artifact.

## Operator and Precision Coverage

The generated [operator capability snapshot](output/operator-capabilities/operator-capabilities.json) is the source of the following counts. It is an implementation-status snapshot, not a promise that every imported graph will run.

| Import family | Import entries | RenderTexture + CommandBuffer entries | FP32 flagged entries | Interpretation |
| --- | ---: | ---: | ---: | --- |
| NCNN | 96 | 56 | 90 | Self-owned NCNN import and Pack4 execution coverage. Strict planning can still reject an operator/profile. |
| Sentis/ONNX | 19 | 13 | 13 | ONNX/Sentis-dialect import and lowering coverage. Aexis does not use Sentis or ONNX Runtime as a runtime backend. |

The snapshot contains 56 `partial`, 29 `debug-only`, 5 `alias-only`, and 6 `unsupported` operator entries. A `partial` entry may have a texture branch for selected shapes but has not passed full Pack4 CommandBuffer model validation. Treat the strict preflight result for the concrete model as authoritative.

`ModelManifest` supports FP32, FP16, BF16 storage, INT8/INT4 weight-only metadata, calibrated W8A8 plans, per-layer mixed precision, and precision gates. Actual checked-in manifests include model-specific INT8/INT4 profiles for MobileCLIP/Matting and FP16/FP32 profiles for selected runners. The capability snapshot reports zero universal per-operator FP16/INT8 flags, so this repository does **not** claim blanket FP16 or INT8 coverage. BF16 storage uses FP32 RenderTexture storage with deterministic Pack4 cast rounding because Unity has no portable BF16 RenderTexture format.

## Tested Environment

Compatible Unity targets and completed validation are different claims. The table records Windows, macOS Editor, macOS Player, and iPadOS Simulator evidence separately from the still-pending physical-iOS check.

| Platform | Device / GPU | Unity | Graphics API | Current runner evidence | Status |
| --- | --- | --- | --- | --- | --- |
| Windows 11 Pro 64-bit | Intel Arc Graphics | 6000.2.7f2 | Vulkan | 2026-07-29: CodeFormer 17,958 ms; GFPGAN 6,053 ms; DeepFillV2 2,196 ms; Real-ESRGAN 1,057 ms; SD inpainting 630,213 ms | Passed for the documented runs |
| macOS Editor | Mac16,10; Apple M4; 16 GB | 6000.2.7f2 | Metal | 2026-07-30, 600 x 337 input: CLIP 11,755 ms; CodeFormer 15,332 ms; GFPGAN 4,696 ms; Real-ESRGAN 1,295 ms; Matting 2,182 ms; YOLO 986 ms (two people, 78.69% mask); YOLO + DeepFillV2 13,811 ms | Completed: seven runners passed; Qwen3.5 Q4 and SD inpainting skipped because their model groups were absent. Editor evidence only. |
| macOS Player | Mac16,10; Apple M4; 16 GB | 6000.2.7f2 | Metal | 2026-07-30, 600 x 337 input: CLIP 97 ms; CodeFormer 2,515 ms; GFPGAN 1,136 ms; Real-ESRGAN 707 ms; Matting 421 ms; YOLO 294 ms (two people, 78.69% mask); YOLO + DeepFillV2 3,528 ms | Completed: seven runners passed; Qwen3.5 Q4 and SD inpainting skipped because their model groups were absent. |
| Android (MuMu emulator) | MuMu ADB `127.0.0.1:16384`; runtime report: vivo V2241A, Adreno (TM) 650, 8961 MiB | 6000.2.7f2 | Vulkan, GameActivity | 2026-07-26 full default Main2 pass: CodeFormer 6,946 ms; Real-ESRGAN 519 ms; GFPGAN 3,160 ms; YOLO 657 ms (four persons); DeepFillV2 3,442 ms; Matting 1,014 ms; CLIP 2,208 ms; Qwen Q4 326,821 ms (103 steps, 397 visible characters) | Passed; conservative Android fallback |
| iPadOS Simulator (`iPad8,6` profile) | Apple M4 host GPU; 16 GB host memory | 6000.2.7f2 | Metal | 2026-07-30, 600 x 337 input: CLIP 111 ms; CodeFormer 2,974 ms; GFPGAN 1,336 ms; Real-ESRGAN 903 ms; Matting 486 ms; YOLO 475 ms (two people, 78.69% mask); YOLO + DeepFillV2 4,731 ms | Completed: seven runners passed; Qwen3.5 Q4 and SD inpainting skipped because their model groups were absent. Simulator evidence only. |
| iPhone / iPad physical device | **Pending command-run report: device, SoC, GPU** | **Pending command-run report: Unity version** | Metal required | Run the [iOS Metal device procedure](Packages/com.aexis/Documentation~/apple-runtime-smoke.md#ios-metal-device) on the target Mac and return its build JSON plus `runner-report=` console log. | Ready for target-machine device validation |

Use a real graphics device for all Aexis validation. `-nographics` is not a valid package, shader, or runner test mode.

The MuMu result is a conservative Android fallback measurement, not a physical-device benchmark. The iPadOS report is a simulator measurement, not a physical-device benchmark. A Snapdragon 888 handset has been reported as materially faster; record the handset, build, thermal state, graphics API, runner configuration, and timing before adding an exact physical-device value.

### Development runner report

In the Editor and Development Players only, MainView2 adds a **Test** button immediately before its Chinese and English language buttons. It uses the current history image and runs CLIP, CodeFormer, GFPGAN, Real-ESRGAN, Matting, Qwen3.5, YOLO segmentation, YOLO + DeepFillV2, and YOLO + SD inpainting. Each runner gets a 600-second cancellation budget. Missing local or bundled model payloads are recorded as skipped; this workflow never opens the model-download dialog.

The test updates one JSON file named `AexisDevelopmentRunnerTest_*.json` below `Application.persistentDataPath` after every runner. It records platform/device data, source dimensions, model-group status, elapsed time, output dimensions, person count, mask coverage, and failure or timeout detail. Windows uses the native Shell to open the report's folder and select the report, macOS reveals it in Finder, Android requests a compatible JSON viewer, and iOS opens the native document preview.

## How to Install

### UPM

Use an embedded package reference during development:

```json
{
  "dependencies": {
    "com.aexis": "file:../AIImage/Packages/com.aexis"
  }
}
```

For a published registry or Git package, install `com.aexis` through Unity Package Manager. Do not copy `Runtime` into `Assets`; the package's assembly definitions and package-owned resources are part of the runtime contract.

### `.unitypackage`

From an Aexis project, run `Aexis/Release/Export Complete UnityPackage`, then import the resulting archive into the target Unity project. The editor bootstrap restores `Packages/com.aexis` and registers it with Package Manager.

To import the full application example, select **AIImage Main2 Application Example** in Package Manager, then run:

```text
Aexis/Examples/Install Main2 Application StreamingAssets
```

Open `Scenes/Main2.unity`, or choose `Aexis/Examples/Open Main2 Application Scene`. The installer copies the sample payload to `Assets/StreamingAssets`, which is the Player-included location. Model paths are relative to `Application.streamingAssetsPath`.

## License

The package manifest declares **MIT** for Aexis source. Model weights, medical data, and third-party sample artifacts are not covered by that source license and retain their own terms.

This pre-release repository still contains a release-audit gate in [Packages/com.aexis/LICENSE.md](Packages/com.aexis/LICENSE.md). Complete the source, shader, sample, and model provenance audit before publishing or representing a package archive as an MIT release. See [Third Party Notices](Packages/com.aexis/Third%20Party%20Notices.md) and [model distribution](Packages/com.aexis/Documentation~/model-distribution.md).

## Documentation

- [Package README](Packages/com.aexis/README.md)
- [Integration manual](Documents/Aexis-Engine-Integration-Manual.md)
- [Runner sample guide](Packages/com.aexis/Documentation~/runner-samples.md)
- [Model distribution and exclusions](Packages/com.aexis/Documentation~/model-distribution.md)
