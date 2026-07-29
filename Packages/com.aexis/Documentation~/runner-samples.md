# AIImage Main2 Application Example

## Import workflow

1. Install `com.aexis`.
2. In Package Manager, import **AIImage Main2 Application Example**.
3. Run `Aexis/Examples/Install Main2 Application StreamingAssets`.
4. Open `Scenes/Main2.unity`, or choose `Aexis/Examples/Open Main2 Application Scene`.
5. Add a runner component to a suitable GameObject, then use paths relative to `Application.streamingAssetsPath`.

This is the only application/runner sample. Its installer copies the sample's complete `StreamingAssets` tree to `Assets/StreamingAssets`. This keeps UPM package contents immutable and makes player-included assets explicit. Every Player build also stages any missing sample payload files before Unity collects Player data, so package-default `.param`, `.bin`, ONNX, tokenizer, and inference-manifest files are included even when the installer was not run manually. Existing project files are preserved.

## Documented runner results

The package README and repository README carry the current evidence-backed runner gallery. These images are copied artifacts, not promotional mockups.

![CodeFormer before and after on 03.jpg](images/codeformer-03-before-after.png)

![Matting composite](images/matting-composite.png)

![GFPGAN before and after on 03.jpg](images/gfpgan-03-before-after.png)

![YOLO and DeepFillV2 before and after on 3.png](images/yolo-deepfillv2-3-before-after.png)

![Real-ESRGAN before and after on 03.jpg](images/realesrgan-03-before-after.png)

![YOLO and Stable Diffusion inpainting before and after on 3.png](images/yolo-sd-inpainting-3-before-after.png)

The Windows 11 / Intel Arc / Vulkan / Unity 6000.2.7f2 batch record from 2026-07-29 is: CodeFormer 17,958 ms with one detected face on `ref/03.jpg`; GFPGAN 6,053 ms on the same image; Real-ESRGAN 1,057 ms on the same image; and the specified `ref/deepfillv2/DeepFillv2-main/test_data/3.png` run found seven people with mask coverage `0.042730`, then completed DeepFillV2 in 2,196 ms (1,995 ms inference) and 12-step Stable Diffusion inpainting in 630,213 ms. The GFPGAN result is visibly distorted on this input, and both inpainting outputs retain residual artifacts; these are raw runner outputs, not quality claims. Qwen Q4 and Q8 passed strict texture execution smoke reports; the MuMu Vulkan Main2 pass also ran Q4 in 326,821 ms for 103 decoder steps and emitted 397 visible characters. The successful CLIP score artifact is from 2026-07-23; the 2026-07-28 strict CommandBuffer run was rejected because `transpose_121` requested an undeclared temporary RT.

Qwen, GFPGAN, Stable Diffusion inpainting, MONAI, and VISTA weights are external. The model-delivery table in the [package README](../README.md#release-downloads) links their configured GitHub Release pages. MONAI/VISTA retains a screenshot slot until a reproducible run has distributable medical input and verified data/model permissions.

The Android evidence is a 2026-07-26 full Main2 pass through MuMu ADB `127.0.0.1:16384` (runtime report: ARM64, vivo V2241A, Adreno (TM) 650, Unity 6000.2.7f2, Vulkan GameActivity): CodeFormer 6,946 ms; Real-ESRGAN 519 ms; GFPGAN 3,160 ms; YOLO 657 ms for four people; DeepFillV2 3,442 ms; Matting 1,014 ms; CLIP 2,208 ms; and Qwen Q4 326,821 ms across 103 decoder steps. MuMu is a conservative fallback measurement, not a physical-device benchmark. A Snapdragon 888 handset has been reported as materially faster; its exact result remains TBD until device, build, thermal state, graphics API, runner configuration, and timing are recorded.

## Included reusable runners

| Component | Function | Default model family |
| --- | --- | --- |
| `AexisNcnnModelLoadRunner` | Cross-platform StreamingAssets loading, session construction, cancellation, load progress, cleanup | Any NCNN `.param` + `.bin` pair |
| `AexisOnnxInspectionRunner` | Cross-platform ONNX byte loading and graph summary | Any `.onnx` model |
| `Aexis.Samples.Runners.ClipNcnnReproRunner` | Image embedding and label-cache classification | Clip MobileCLIP S0 |
| `Aexis.Samples.Runners.CodeFormerNcnnReproRunner2` | Face detection, alignment, restoration, and compositing | CodeFormer encoder/generator/detector |
| `Aexis.Samples.Runners.DeepFillV2Runner` | Masked-image inpainting using the available NCNN pair or source ONNX representation | DeepFillV2 |
| `Aexis.Samples.Runners.MatterNcnnReproRunner` | Foreground matting and alpha composite generation | Matting |
| `Aexis.Samples.Runners.RealEsrganNcnnReproRunner` | Tiled texture-native upscaling | RealESRGAN AnimeVideo v3 x4 |
| `Aexis.Samples.Runners.YoloSegNcnnReproRunner` | Segmentation detection, masks, transparent output, and overlay | YOLOv8n/YOLO11n segmentation |

The catalog exposes default paths under `Clip`, `CodeFormer`, `DeepFileV2`, `Matting`, `RealESRGAN`, and `Yolo`. Application-specific preprocessing/postprocessing remains example code: applications own input normalization, prompt/token construction, detection decoding, alpha composition, UI, and result presentation.

## Complete application scope

The example contains the `Main2` scene, MainView2, DesignView, LibraryView, every original AIImage runner, QWEN sessions, MONAI/VISTA code, UI Toolkit assets, shaded SharpZip source, shaded async source, and copied Editor tests/debug tools. GFPGAN, Stable Diffusion, SD Inpainting, MONAI/VISTA, and QWEN runner configuration is present, but their model weights are not included.

The sample uses `Aexis.Samples.Async` and `Aexis.Samples.SharpZipLib` for its isolated source dependencies. It neither imports nor declares `Cysharp.Threading.Tasks` or `ICSharpCode.SharpZipLib`, so the sample does not collide with those libraries in a consuming Unity project.

The fourteen runner and editor-support files that need nested token traversal and diagnostic JSON reports use the sample-only `Aexis.Samples.Json` source copy. These dynamic JSON workflows cannot be represented by `JsonUtility`. The copy is Json.NET 13.0.2 under its MIT license, shaded away from `Newtonsoft.Json` so it cannot collide with a consuming project's own Json.NET package or DLL. Its provenance and modification record are in `ThirdParty/AexisSampleJson/UPSTREAM.md`; no JSON DLL is copied into `Assets` or declared as a UPM dependency.

Editor NUnit sources remain in the example but are excluded from a default UnityPackage import. To compile and run them, install Unity Test Framework and add `AEXIS_INCLUDE_EDITOR_TESTS` to the Editor scripting define symbols. This keeps a new project's first import compilable without discarding validation coverage.
