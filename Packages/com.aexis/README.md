# Aexis

`com.aexis` is the single Unity Package Manager (UPM) package for the Aexis on-device inference engine. AIImage is an application built on Aexis; it is not the engine namespace, a package dependency, or a runtime requirement.

## Status and compatibility

| Item | Supported release baseline |
| --- | --- |
| Unity | 2022.3 LTS through Unity 6000.3 |
| Script runtime | Unity-supported .NET profile for the installed editor |
| GPU path | Compute-shader-capable graphics API with RenderTexture support |
| Package dependencies | Unity modules only; dynamic JSON is source-vendored in the optional application sample |
| Runtime namespaces | `Aexis`, `Aexis.Async`, `Aexis.Onnx`, `Aexis.Ncnn`, `Aexis.Execution` |

`Aexis.Async` exposes BCL `Task` and does not ship, vendor, or reference UniTask. This keeps the engine compatible with Unity 2022.3 and prevents package-level conflicts when a host project uses its own UniTask version or distribution.

## Runner evidence

AIImage Main2 is the complete application example built on Aexis. It contains runner configuration for Qwen3.5 Q4/Q8, CLIP, CodeFormer, Matting, Real-ESRGAN, GFPGAN, YOLO plus DeepFillV2, YOLO plus Stable Diffusion inpainting, and MONAI/VISTA. The images below are actual runner artifacts copied into package documentation; the complete evidence and release table are in the [repository README](../../README.md).

| Runner | Evidence | Result |
| --- | --- | --- |
| Qwen3.5 mobile Q4 | Strict texture text smoke, 65.533 s, 48 cache textures | Valid execution; only one end-of-turn token, so it is not presented as a quality demo. |
| Qwen3.5 mobile Q8 | Strict texture multimodal smoke, 71.618 s, 6 generated tokens | Valid execution; generated text begins with a Chinese-language phrase. |
| CLIP MobileCLIP S0 | Successful 2026-07-23 score artifact | `Photo` 0.332389; `Portrait` 0.265945. The current strict profile rejects an undeclared temporary RT at `transpose_121`. |
| CodeFormer | Windows/Vulkan batch, 2026-07-28 | 16,075 ms |
| Matting | Windows/Vulkan batch, 360x202 | 1,103 ms |
| GFPGAN | Windows/Vulkan batch | 4,786 ms |
| YOLO + DeepFillV2 | Windows/Vulkan batch | 1,529 ms YOLO; 1,686 ms DeepFillV2 |
| Real-ESRGAN AnimeVideo v3 x4 | Windows/Vulkan Pack4 CommandBuffer validation | 661 ms for `ref/03.jpg`; paired output parity mean/max RGB error was 0. |
| YOLO + SD inpainting | Archived validation artifact, 2026-07-16 | Visual sample only; not a current benchmark. |

![Qwen multimodal input](Documentation~/images/qwen-and-clip-input.jpg)

![CodeFormer result](Documentation~/images/codeformer-face-restoration.png)

![Matting composite](Documentation~/images/matting-composite.png)

![GFPGAN result](Documentation~/images/gfpgan-face-restoration.png)

![YOLO and DeepFillV2 result](Documentation~/images/yolo-deepfill-output.png)

![Real-ESRGAN x4 result](Documentation~/images/realesrgan-x4.png)

![YOLO and SD inpainting result](Documentation~/images/yolo-sd-inpainting.png)

> **MONAI/VISTA screenshot slot:** Reserved for a reproducible run with distributable medical input and confirmed data/model permissions. No medical image is published by this package.

## Release downloads

The Main2 model-delivery configuration resolves its assets from [newzjh/AIImage releases](https://github.com/newzjh/AIImage/releases). Download the current generated asset named in `AIImageModelReleaseManifest.json`, or use the application's model-download UI. Do not infer an archive name from a runner name.

| Model group | Release download page |
| --- | --- |
| Qwen3.5 Q4, CLIP, CodeFormer, Matting, YOLO, SD inpainting configuration | [`model`](https://github.com/newzjh/AIImage/releases/tag/model) |
| Qwen3.5 Q8 | [`qwen3.5_0.8b_mobile_q8`](https://github.com/newzjh/AIImage/releases/tag/qwen3.5_0.8b_mobile_q8) |
| Real-ESRGAN | [`realesr`](https://github.com/newzjh/AIImage/releases/tag/realesr) |
| GFPGAN | [`gfpgan`](https://github.com/newzjh/AIImage/releases/tag/gfpgan) |
| DeepFillV2 | [`DeepFileV2`](https://github.com/newzjh/AIImage/releases/tag/DeepFileV2) |
| MONAI / VISTA | External model and data acquisition only |

Model weights have their own licenses and redistribution conditions. The listed release pages are download locations, not license grants.

## Install

For an embedded package, add the following entry to the consuming project's `Packages/manifest.json`:

```json
{
  "dependencies": {
    "com.aexis": "file:../path-to-aexis/Packages/com.aexis"
  }
}
```

For a registry or Git release, install the published `com.aexis` package through Unity Package Manager. Do not copy `Runtime` files into `Assets`; package-local resources and assembly definitions are part of the runtime contract.

For a standalone `.unitypackage`, run `Aexis/Release/Export Complete UnityPackage` from an Aexis project, then import the exported archive. Its editor bootstrap restores the complete package to `Packages/com.aexis` and registers it with Package Manager. The Main2 application is managed directly at `Packages/com.aexis/Samples/AIImageApplicationExample`; open its scene or run its installer from there.

## Package layout

| Path | Purpose |
| --- | --- |
| `Runtime/Core` | Backend-neutral contracts, tensor descriptors, precision and quantization manifests |
| `Runtime/Onnx` | Self-owned ONNX protobuf reader and execution-planning adapters |
| `Runtime/Ncnn` | Self-owned NCNN param/bin readers, graph loader, Pack4 texture execution, operators |
| `Runtime/Execution` | ONNX shape/index texture backend |
| `Runtime/Resources/Aexis` | Compute shaders loaded through `Resources`, owned by the package |
| `Editor` | Package editor tooling |
| `Samples` | Unity-managed application examples and their model payloads |
| `Documentation~` | Package documentation rendered by Package Manager |
| `Tests/Editor` | Package boundary and planning tests |

The runtime is partitioned into multiple asmdefs inside one UPM package. Unity supports this arrangement; consumers import only `com.aexis` and get the assemblies transitively. `Aexis.Ncnn` production inference stays on Pack4 RenderTextures and CommandBuffer-compatible texture flows. Compute buffers are limited to immutable uploads and explicit diagnostic paths.

`ForwardPack4WithFixedInputs(...)` records the same strict Pack4 CommandBuffer path for mixed texture/fixed-buffer model inputs. A fixed buffer is dispatched into a GPU texture before the first layer (including exact RFloat token-id upload for `Embed`); it cannot become an activation or trigger a CPU/ComputeBuffer fallback.

## Model import and extension contracts

Unity imports `.onnx`, NCNN `.param`, and versioned `.aexis` archive files as `AexisModelAsset`. The importer preserves source bytes, emits a versioned binary graph, attaches a sibling NCNN `.bin` when present, and records lowering/preflight diagnostics on the asset. `AexisModelPackager` exposes the same offline prepack path for build tooling.

`AexisNcnnBinaryParam` is the stable binary `.param` representation and `AexisModelArchive` (`AEXM` v1) packages the binary graph, weights, source, manifest, and diagnostics. Use `AexisCustomLayerRegistry` to register a public layer factory with a versioned parameter/arity schema. Model archives may declare their required custom layer type and shader kernel id; unresolved declarations fail before execution.

P1 visual operators (`GridSample`, deformable/ROI/detection families, `Fold`, `Flip`, `GLU`, `Einsum`, `Diag`, and `SPP`) ship with built-in Pack4 RenderTexture and CommandBuffer profiles. Strict preflight reuses the same parameter/shape proof as dispatch; an unsupported profile is rejected before execution rather than materialized through a ComputeBuffer. BF16 uses FP32 texture storage because Unity has no portable BF16 render-texture format; Pack4 `Cast` provides deterministic BF16 rounding. Per-layer mixed plans select FP16/FP32/BF16 physical storage as appropriate, while calibrated signed or unsigned Pack4 INT8 plans feed Conv/DWConv/Gemm/InnerProduct dispatch directly.

## Operator and quantization status

`output/operator-capabilities/operator-capabilities.json` in the repository is the implementation snapshot used for release documentation. It lists 96 NCNN import entries (56 with both RenderTexture and CommandBuffer flags) and 19 Sentis/ONNX import entries (13 with both flags). The snapshot contains 56 `partial`, 29 `debug-only`, 5 `alias-only`, and 6 `unsupported` entries, so a flag is not a blanket model compatibility promise. Aexis imports ONNX/Sentis-dialect graphs but does not depend on Sentis or ONNX Runtime at runtime.

`ModelManifest` can represent FP32, FP16, BF16, INT8/INT4 weight-only, calibrated W8A8, and per-layer mixed plans. Checked-in model manifests include model-specific INT8/INT4 profiles for MobileCLIP and Matting. The capability snapshot has zero universal per-operator FP16/INT8 flags; use strict preflight and the model's precision gate for a concrete graph. Qwen mobile Q4/Q8 are model archive variants, not a claim of generic engine-wide INT4/INT8 support.

## Tested environments

| Platform | Hardware and Unity | Current evidence | Status |
| --- | --- | --- | --- |
| Windows 11 Pro 64-bit | Intel Arc Graphics, Unity 6000.2.7f2, Vulkan | Documented 2026-07-28 runner results above | Passed for those runs |
| macOS | **TBD: machine/GPU/Unity** | `Tools/AIImage_MACOS.build-failure.txt` records a failure, not a pass | Blocked / not validated |
| Android phone | **TBD: device/SoC/GPU/Unity** | **TBD: runner and timing** | Not yet validated |
| iPhone / iPad | **TBD: device/SoC/GPU/Unity** | **TBD: runner and timing** | Not yet validated |

Validation must use a real graphics device. `-nographics` is not valid for Aexis shader, package, or runner validation.

## Quick start

```csharp
using Aexis.Ncnn;

var ops = new NcnnOps();
var session = NcnnInferenceSessionFactory.Create(ops);
using var model = new NcnnBinReader(binStream);
await session.LoadModelAsync(paramText, model);
// Prepare the Pack4 input expected by the model, then execute ForwardPack4.
session.Release();
```

See [API manual](Documentation~/api-manual.md) for input/output ownership and [runner samples](Documentation~/runner-samples.md) for an importable component.

## Application example and model files

Import **AIImage Main2 Application Example** from Package Manager, then run `Aexis/Examples/Install Main2 Application StreamingAssets`. This is the single full example: it combines `Main2`, MainView2, DesignView, LibraryView, UI/application code, all runners including GFPGAN, Stable Diffusion, SD Inpainting, MONAI/VISTA, and QWEN, plus editor tests and debug tools.

Its installer copies the complete sample payload to `Assets/StreamingAssets`, the Unity Player-included location. The reusable runner catalog uses `Clip/...`, `CodeFormer/...`, `DeepFileV2/...`, `Matting/...`, `RealESRGAN/...`, and `Yolo/...` paths below that root. Default model files are included only for those six model families. GFPGAN, Stable Diffusion, SD Inpainting, MONAI/VISTA, and QWEN weights are deliberately excluded because of package-size and redistribution constraints. See [model distribution](Documentation~/model-distribution.md) before redistributing any sample model.

Reduced Main2 Player builds do not stage, move, or modify `Assets/StreamingAssets` or package sample `StreamingAssets`. `Aexis/Release/Build Reduced` selects the Main2 default model files after Unity has generated the Player output (or during Android Gradle asset generation), so the source tree remains unchanged. Non-default model groups are downloaded into persistent data from the AIImage GitHub Release and are not copied into a Player by default.

The complete sample retains its Editor NUnit sources. A default sample import excludes them through `AEXIS_INCLUDE_EDITOR_TESTS`; install Unity Test Framework and add that define symbol when running the included test suite.

## Scope and licensing

`package.json` declares MIT as the Aexis source license target. The current pre-release [LICENSE.md](LICENSE.md) retains a release-audit gate; complete that audit before publishing or representing an archive as an MIT release. Aexis does not include Unity Sentis, Tencent ncnn, ONNX Runtime, MNN, MONAI, or VISTA source/binaries as runtime dependencies. Compatibility targets do not imply affiliation or use of upstream code.

The complete application sample uses its namespace-isolated `Aexis.Samples.Json` source copy in fourteen files for dynamic JSON documents, token traversal, and editor diagnostics. These uses are not DTO-only configuration payloads, so Unity `JsonUtility` is not a compatible replacement. The source copy is MIT-licensed Json.NET 13.0.2, with its immutable revision, checksum, license, and shading record under `Samples/AIImageApplicationExample/ThirdParty/AexisSampleJson`; it does not install a Newtonsoft package or copy a duplicate DLL into `Assets`.

The sample's shaded `Aexis.Samples.SharpZipLib` source is required by `StandardImageIO` and the MONAI runner for compressed input. It is intentionally namespaced away from `ICSharpCode.SharpZipLib` and must not be removed as unused code.

Models are separate artifacts with their own licenses and redistribution conditions. A release manager must complete the provenance table in `Documentation~/model-distribution.md` before publishing a package archive containing sample models. Do not place medical data, private goldens, checkpoints, or application-specific tooling in this package.

## Documentation

- [Installation and architecture](Documentation~/architecture.md)
- [Public API manual](Documentation~/api-manual.md)
- [Runner sample guide](Documentation~/runner-samples.md)
- [Model distribution and exclusions](Documentation~/model-distribution.md)
- [Changelog](CHANGELOG.md)
- [Third-party audit](Third%20Party%20Notices.md)
