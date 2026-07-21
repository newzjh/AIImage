# Aexis Architecture

## Single package, multiple assemblies

`com.aexis` is one UPM distribution unit. Its Runtime assembly definitions are intentionally separate so parsing, GPU execution, and optional consumers do not create one monolithic compile unit:

| Assembly | References | Responsibility |
| --- | --- | --- |
| `Aexis` | none | Contracts shared by every backend |
| `Aexis.Async` | UnityEngine | BCL `Task` frame scheduling, compatible with Unity 2022.3 |
| `Aexis.Onnx` | `Aexis` | ONNX parsing and execution planning |
| `Aexis.Ncnn` | `Aexis`, `Aexis.Async`, `Aexis.Onnx` | NCNN loading and texture-native execution |
| `Aexis.Execution` | `Aexis`, `Aexis.Onnx` | Shape/index GPU operators |
| `Aexis.Editor` | Runtime assemblies | Editor-only tooling |

No Runtime assembly may reference `AIImage`, UniTask, Unity Sentis, ONNX Runtime, Tencent ncnn, MNN, or a native plugin. `com.unity.modules.imageconversion` and `com.unity.modules.unitywebrequest` are Unity built-in dependencies used only by sample PNG diagnostics and cross-platform StreamingAssets loading. The namespace-isolated `Aexis.Samples.Json` source copy is confined to optional sample/editor tooling; it is needed for dynamic JSON documents and must not enter Runtime.

## Model formats

- `.onnx`: Read by `Aexis.Onnx.OnnxModelReader` for protobuf graph inspection and lowering/import planning. The parser fails on malformed or unsupported protobuf data rather than inferring missing semantics.
- `.param` + `.bin`: Read by `Aexis.Ncnn.NcnnParamParser`, `NcnnBinReader`, and `NcnnGraphSession`.
- model manifest JSON: Optional precision and quantization contract, resolved through `NcnnModelManifestLoader`.

## Resource ownership

Compute shaders live in `Runtime/Resources/Aexis/*` and are located with package-owned resource names. Runtime code must not use `AssetDatabase`, absolute project paths, or application `Assets` paths. Model files belong in an application's `StreamingAssets`, persistent data location, or a controlled downloader; the package owns no application model directory.

## Execution invariant

The NCNN execution path uses Pack4 RenderTextures and texture-backed tensors. Preserve logical and storage shapes independently. Do not add a normal-inference ComputeBuffer fallback, texture-to-buffer materialization, or buffer-to-texture recovery path. A missing texture implementation must fail clearly in strict production mode; diagnostic-only readback is the exception.

## Extension policy

Add a format/backend under `Runtime/<Backend>` with its own asmdef only when it has a meaningful dependency boundary. MNN support will follow this pattern as `Aexis.Mnn` without changing the package name or the core namespace. Do not put application UI, model checkpoints, sample datasets, or private test output under `Runtime`.
