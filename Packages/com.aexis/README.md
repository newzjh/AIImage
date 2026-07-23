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

Every Player build automatically stages any missing Main2 sample payload files from the package into `Assets/StreamingAssets` before Unity collects Player data. This includes `.param`, `.bin`, ONNX, tokenizer, and manifest files, preserves user-supplied files at the same paths, and applies to all build targets. The `AEXIS_SKIP_SAMPLE_STREAMING_ASSETS_STAGING=1` process setting is reserved for specialized builds that deliberately replace the whole StreamingAssets tree.

The complete sample retains its Editor NUnit sources. A default sample import excludes them through `AEXIS_INCLUDE_EDITOR_TESTS`; install Unity Test Framework and add that define symbol when running the included test suite.

## Scope and licensing

The source implementation is released under the MIT license in [LICENSE.md](LICENSE.md). Aexis does not include Unity Sentis, Tencent ncnn, ONNX Runtime, MNN, MONAI, or VISTA source/binaries as runtime dependencies. Compatibility targets do not imply affiliation or use of upstream code.

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
