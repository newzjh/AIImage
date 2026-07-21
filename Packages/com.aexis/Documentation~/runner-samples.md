# AIImage Main2 Application Example

## Import workflow

1. Install `com.aexis`.
2. In Package Manager, import **AIImage Main2 Application Example**.
3. Run `Aexis/Examples/Install Main2 Application StreamingAssets`.
4. Open `Scenes/Main2.unity`, or choose `Aexis/Examples/Open Main2 Application Scene`.
5. Add a runner component to a suitable GameObject, then use paths relative to `Application.streamingAssetsPath`.

This is the only application/runner sample. Its installer copies the sample's complete `StreamingAssets` tree to `Assets/StreamingAssets`. This keeps UPM package contents immutable and makes player-included assets explicit.

## Included reusable runners

| Component | Function | Default model family |
| --- | --- | --- |
| `AexisNcnnModelLoadRunner` | Cross-platform StreamingAssets loading, session construction, cancellation, load progress, cleanup | Any NCNN `.param` + `.bin` pair |
| `AexisOnnxInspectionRunner` | Cross-platform ONNX byte loading and graph summary | Any `.onnx` model |
| `Aexis.Samples.Runners.ClipNcnnReproRunner` | Image embedding and label-cache classification | Clip MobileCLIP S0 |
| `Aexis.Samples.Runners.CodeFormerNcnnReproRunner2` | Face detection, alignment, restoration, and compositing | CodeFormer encoder/generator/detector |
| `Aexis.Samples.Runners.DeepFillV2Runner` | Masked-image inpainting through ONNX-direct or NCNN loading | DeepFillV2 |
| `Aexis.Samples.Runners.MatterNcnnReproRunner` | Foreground matting and alpha composite generation | Matting |
| `Aexis.Samples.Runners.RealEsrganNcnnReproRunner` | Tiled texture-native upscaling | RealESRGAN |
| `Aexis.Samples.Runners.YoloSegNcnnReproRunner` | Segmentation detection, masks, transparent output, and overlay | YOLOv8n/YOLO11n segmentation |

The catalog exposes default paths under `Clip`, `CodeFormer`, `DeepFileV2`, `Matting`, `RealESRGAN`, and `Yolo`. Application-specific preprocessing/postprocessing remains example code: applications own input normalization, prompt/token construction, detection decoding, alpha composition, UI, and result presentation.

## Complete application scope

The example contains the `Main2` scene, MainView2, DesignView, LibraryView, every original AIImage runner, QWEN sessions, MONAI/VISTA code, UI Toolkit assets, shaded SharpZip source, shaded async source, and copied Editor tests/debug tools. GFPGAN, Stable Diffusion, SD Inpainting, MONAI/VISTA, and QWEN runner configuration is present, but their model weights are not included.

The sample uses `Aexis.Samples.Async` and `Aexis.Samples.SharpZipLib` for its isolated source dependencies. It neither imports nor declares `Cysharp.Threading.Tasks` or `ICSharpCode.SharpZipLib`, so the sample does not collide with those libraries in a consuming Unity project.

The fourteen runner and editor-support files that need nested token traversal and diagnostic JSON reports use the sample-only `Aexis.Samples.Json` source copy. These dynamic JSON workflows cannot be represented by `JsonUtility`. The copy is Json.NET 13.0.2 under its MIT license, shaded away from `Newtonsoft.Json` so it cannot collide with a consuming project's own Json.NET package or DLL. Its provenance and modification record are in `ThirdParty/AexisSampleJson/UPSTREAM.md`; no JSON DLL is copied into `Assets` or declared as a UPM dependency.

Editor NUnit sources remain in the example but are excluded from a default UnityPackage import. To compile and run them, install Unity Test Framework and add `AEXIS_INCLUDE_EDITOR_TESTS` to the Editor scripting define symbols. This keeps a new project's first import compilable without discarding validation coverage.
