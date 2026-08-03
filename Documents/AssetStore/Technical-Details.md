# Technical Details

The text below is ready to paste into a Unity Asset Store technical-details field. It describes the package as it exists in this repository and avoids unsupported compatibility or performance claims.

## Product Summary

Aexis is a self-contained, on-device inference engine for Unity. It imports selected ONNX and NCNN model graphs and executes planned GPU workloads through compute shaders and texture-native Pack4 RenderTexture storage. The package includes public runtime assemblies, model import tooling, reusable runner components, and the AIImage Main2 application example.

## Key Features

- Self-owned ONNX protobuf reader, graph lowering, shape/index planner, and execution-planning path.
- Self-owned NCNN `.param` and `.bin` loader with texture-native Pack4 execution.
- Package-owned compute shaders loaded from `Runtime/Resources/Aexis`.
- Unity import support for `.onnx`, NCNN `.param` plus `.bin`, and versioned `.aexis` (`AEXM`) model archives.
- Strict preflight and allocation planning. Unsupported texture profiles fail clearly instead of silently materializing normal activations through a transient ComputeBuffer.
- Public runtime assemblies: `Aexis`, `Aexis.Async`, `Aexis.Onnx`, `Aexis.Ncnn`, and `Aexis.Execution`.
- Importable samples: Core Contracts, NCNN Runtime Integration, and AIImage Main2 Application Example.
- Reusable sample runners for model loading, ONNX inspection, CLIP classification, CodeFormer restoration, DeepFillV2 inpainting, foreground matting, Real-ESRGAN upscaling, and YOLO segmentation.

## Unity and Platform Requirements

| Item | Requirement or validated evidence |
| --- | --- |
| Unity | Unity 2022.3 LTS through Unity 6000.3. Package manifest minimum: Unity 2022.3. |
| Graphics hardware | A compute-shader-capable graphics API with RenderTexture support. A real graphics device is required for package, shader, and runner validation. |
| Validated graphics APIs | Vulkan on Windows and Android; Metal on macOS Editor, macOS Player, and iPadOS Simulator. |
| Platforms with current runner evidence | Windows 11, macOS, Android through MuMu Vulkan emulator, and iPadOS Simulator. |
| Physical iPhone/iPad | Not yet validated. Do not advertise physical iOS/iPadOS support until the device test report is recorded. |
| Render pipeline | The inference path is compute-shader based and has no URP or HDRP package dependency. Validate the host project's render-pipeline configuration and target GPU before shipping. |
| Scripting | C#; `Aexis.Async` is based on BCL `Task` and does not require UniTask. |

## Package Contents

| Location | Content |
| --- | --- |
| `Runtime/Core` | Backend-neutral contracts, tensor descriptors, and precision/quantization manifests. |
| `Runtime/Onnx` | ONNX reader and planning adapters. |
| `Runtime/Ncnn` | NCNN readers, graph loader, Pack4 texture execution, and operators. |
| `Runtime/Execution` | ONNX shape and index GPU operators. |
| `Runtime/Resources/Aexis` | Package-owned compute shaders. |
| `Editor` | Editor-only tooling. |
| `Samples` | Importable examples and permitted default model payloads. |
| `Documentation~` | Package Manager documentation. |
| `Tests/Editor` | Package-boundary and planning tests. |

## Model Formats and Delivery

Supported import formats are `.onnx`, NCNN `.param` plus `.bin`, and `.aexis` archives. An optional model manifest can declare precision, quantization, and planning constraints. The bundled sample model set is intentionally limited: default files may be included only for Clip, CodeFormer, DeepFillV2, Matting, RealESRGAN, and YOLO after provenance review.

GFPGAN, Stable Diffusion, SD Inpainting, MONAI/VISTA, and Qwen weights are external downloads. Every model has its own license and redistribution conditions. A download location is not a license grant; review `Packages/com.aexis/Documentation~/model-distribution.md` before placing any weight in an Asset Store upload.

## Dependencies and Runtime Boundaries

The runtime does not wrap or require Unity Sentis, Tencent ncnn, ONNX Runtime, MNN, UniTask, or a native inference plug-in. It uses Unity modules for image conversion, UI, UIElements, and UnityWebRequest. The optional application sample contains namespace-isolated source copies for dynamic JSON and compressed-image workflows; these source copies do not add a package-level Json.NET or SharpZipLib DLL dependency.

## Precision and Operator Scope

`ModelManifest` supports FP32, FP16, BF16 storage, INT8/INT4 weight-only metadata, calibrated W8A8 plans, and per-layer mixed precision plans. This is a model-contract capability, not a generic claim that every imported graph or every operator runs at every precision. The checked-in operator snapshot reports 96 NCNN import entries and 19 ONNX/Sentis-dialect import entries; strict preflight for the concrete graph remains authoritative.

The production NCNN path keeps activations texture-backed using Pack4 RenderTextures and CommandBuffer-compatible dispatch. Compute buffers are limited to immutable uploads and explicit diagnostic paths, not a normal-inference activation fallback.

## Included Example

Import **AIImage Main2 Application Example** from Package Manager, then run `Aexis/Examples/Install Main2 Application StreamingAssets` and open `Scenes/Main2.unity`. The example demonstrates the full application UI and runner configuration. It includes English and Chinese UI controls, model download configuration, development-player reports, and reusable runner components.

## Important Limitations

- A successful import does not guarantee that every ONNX or NCNN graph is supported. Run strict preflight for the exact model and input profile.
- Performance varies with device, graphics API, thermal state, model, precision plan, image size, and runner configuration. Published timings are dated observations for their named environment only.
- The Android evidence is from a MuMu Vulkan emulator and is not a physical-device benchmark.
- The iPadOS evidence is from a simulator and is not physical-device evidence.
- Model weights, medical data, private golden outputs, and third-party sample artifacts are not covered by the Aexis source-license target.

## License

The package manifest declares MIT for Aexis source. The current repository retains a release-audit gate in `Packages/com.aexis/LICENSE.md`; complete the required provenance audit before publishing or describing the package archive as a completed MIT release.
