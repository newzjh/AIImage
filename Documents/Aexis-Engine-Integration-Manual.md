# Aexis Engine Integration Manual

## 1. Purpose and scope

Aexis is the Unity on-device inference engine delivered as the single `com.aexis` UPM package. AIImage is the full application example that uses Aexis; it is not a Runtime dependency, Runtime namespace, or required host-project folder.

The package owns its ONNX reader/lowering path, NCNN `.param`/`.bin` reader, model graph loaders, Pack4 texture-native GPU execution, compute shaders, and ONNX shape/index execution. Runtime does not reference Unity Sentis, Tencent ncnn, ONNX Runtime, MNN, UniTask, or a native inference plug-in.

Use this manual together with the [root README](../README.md), [package README](../Packages/com.aexis/README.md), [runner guide](../Packages/com.aexis/Documentation~/runner-samples.md), and [model-distribution policy](../Packages/com.aexis/Documentation~/model-distribution.md).

## 2. Install Aexis

### 2.1 Embedded UPM package

Add a `file:` reference to the host project's `Packages/manifest.json`:

```json
{
  "dependencies": {
    "com.aexis": "file:../AIImage/Packages/com.aexis"
  }
}
```

For a registry or Git distribution, install `com.aexis` through Unity Package Manager instead. Do not copy `Runtime` into `Assets`; package-local resources and assembly definitions are required at runtime.

### 2.2 Complete `.unitypackage`

From an Aexis project, choose:

```text
Aexis/Release/Export Complete UnityPackage
```

Import the exported archive into the target Unity project. Its editor bootstrap restores `Packages/com.aexis` and registers the package with Package Manager.

### 2.3 AIImage Main2 Application Example

After importing **AIImage Main2 Application Example** in Package Manager, run:

```text
Aexis/Examples/Install Main2 Application StreamingAssets
```

The installer copies the sample's permitted `StreamingAssets` payload into `Assets/StreamingAssets`, the location included by a Unity Player. Open `Scenes/Main2.unity`, or choose `Aexis/Examples/Open Main2 Application Scene`.

## 3. Package architecture

| Package path | Assembly or responsibility |
| --- | --- |
| `Runtime/Core` | `Aexis`: contracts, tensor descriptors, precision and quantization manifests |
| `Runtime/Async` | `Aexis.Async`: BCL `Task` frame-yield helpers |
| `Runtime/Onnx` | `Aexis.Onnx`: ONNX protobuf reading and execution planning |
| `Runtime/Ncnn` | `Aexis.Ncnn`: NCNN graph/weight readers and Pack4 operator implementation |
| `Runtime/Execution` | `Aexis.Execution`: ONNX shape/index texture execution |
| `Runtime/Resources/Aexis` | Package-owned compute shaders loaded through `Resources` |
| `Editor` | Editor import, packaging, and validation tools |
| `Samples` | Importable samples and the AIImage application example |
| `Documentation~` | Package Manager documentation and documented runner artifacts |
| `Tests/Editor` | Package-boundary and execution-planning tests |

The multiple Runtime asmdefs are compilation boundaries inside one UPM package, not separate packages. `Aexis.Async` exposes BCL `Task`; it does not ship or reference UniTask. Sample-only code may use `Aexis.Samples.Async`, which is namespace-isolated from the engine.

### 3.1 Production GPU contract

The NCNN production path uses Pack4 RenderTextures and CommandBuffer-compatible texture flows. Logical and physical storage shapes are recorded independently. `ForwardPack4WithFixedInputs(...)` uploads a fixed input into a GPU texture before the first layer; it cannot become a persistent ComputeBuffer activation.

Do not add a generic ComputeBuffer fallback to make an unsupported layer appear to work. Strict preflight must reject an unsupported profile with a diagnostic that identifies the layer, allocation, or shape that needs a texture-native implementation or plan declaration.

## 4. Model import, operators, and precision

Unity imports `.onnx`, NCNN `.param`, and versioned `.aexis` archives as `AexisModelAsset`. An import preserves source bytes, produces the versioned graph representation, associates an NCNN `.bin` when present, and records lower/preflight diagnostics. `AexisModelPackager`, `AexisNcnnBinaryParam`, and `AexisModelArchive` are the corresponding offline/prepack forms.

The generated capability snapshot at `output/operator-capabilities/operator-capabilities.json` currently records:

| Import family | Import entries | Entries flagged for RenderTexture + CommandBuffer | FP32-flagged entries |
| --- | ---: | ---: | ---: |
| NCNN | 96 | 56 | 90 |
| Sentis/ONNX dialect | 19 | 13 | 13 |

The snapshot status breakdown is 56 `partial`, 29 `debug-only`, 5 `alias-only`, and 6 `unsupported`. A flag shows an implementation branch, not guaranteed compatibility for arbitrary models. The concrete import plus strict preflight result is authoritative. The Sentis/ONNX count means Aexis recognizes and lowers selected source dialects; it does not introduce a Sentis or ONNX Runtime backend dependency.

`ModelManifest` supports FP32, FP16, BF16 storage, INT8/INT4 weight-only metadata, calibrated W8A8 plans, per-layer mixed precision, and precision gates. Existing manifests provide model-specific MobileCLIP/Matting INT8 and INT4 profiles. The same snapshot has zero universal per-operator FP16/INT8 flags, so do not advertise generic INT8 or FP16 operator coverage. Qwen Q4/Q8 are mobile model variants, not an engine-wide quantization guarantee. BF16 is stored in FP32 textures because Unity has no portable BF16 RenderTexture format; Pack4 cast supplies deterministic rounding.

## 5. Basic integration

The following is the low-level NCNN pattern. An application must provide the model's expected Pack4 input and should own preprocessing, postprocessing, UI, and output lifetime.

```csharp
using System.IO;
using Aexis.Ncnn;

var ops = new NcnnOps();
var session = NcnnInferenceSessionFactory.Create(ops);
using var weights = new NcnnBinReader(new MemoryStream(binBytes, writable: false));

await session.LoadModelAsync(paramText, weights, progress =>
    Debug.Log($"{progress.stage} {progress.progress01:P0}"));

// Build the model-specific Pack4 RenderTexture input, then execute it.
var output = session.ForwardPack4(input, inputPacks, "data");

// Release GPU resources when the component is disabled or changes model.
session.Release();
```

For ONNX inspection and lowering, use `OnnxModelReader.Read(...)`, `OnnxD3Importer.Import(...)`, `OnnxExecutionAdapter.TryAdapt(...)`, and `OnnxExecutionShapePlanner.Validate(...)`. Reading an ONNX graph does not mean every ONNX operator can execute.

## 6. Runner catalog and model delivery

Importable components in the Main2 sample include `AexisNcnnModelLoadRunner`, `AexisOnnxInspectionRunner`, `ClipNcnnReproRunner`, `CodeFormerNcnnReproRunner2`, `DeepFillV2Runner`, `MatterNcnnReproRunner`, `RealEsrganNcnnReproRunner`, and `YoloSegNcnnReproRunner`.

| Runner / model | Delivery status | GitHub Release download page |
| --- | --- | --- |
| Qwen3.5 0.8B mobile Q4 | External model group | [`qwen3.5_0.8b_mobile_q4`](https://github.com/newzjh/AIImage/releases/tag/qwen3.5_0.8b_mobile_q4) |
| Qwen3.5 0.8B mobile Q8 | External model group | [`qwen3.5_0.8b_mobile_q8`](https://github.com/newzjh/AIImage/releases/tag/qwen3.5_0.8b_mobile_q8) |
| CLIP MobileCLIP S0 | Default sample model family | [`model`](https://github.com/newzjh/AIImage/releases/tag/model) |
| CodeFormer | Default sample model family | [`model`](https://github.com/newzjh/AIImage/releases/tag/model) |
| Matting | Default sample model family | [`model`](https://github.com/newzjh/AIImage/releases/tag/model) |
| Real-ESRGAN | Default sample model family, after provenance review | [`realesr`](https://github.com/newzjh/AIImage/releases/tag/realesr) |
| GFPGAN | External model group | [`gfpgan`](https://github.com/newzjh/AIImage/releases/tag/gfpgan) |
| YOLO + DeepFillV2 | Default YOLO family; DeepFillV2 group is configuration-dependent | [`DeepFileV2`](https://github.com/newzjh/AIImage/releases/tag/DeepFileV2) |
| YOLO + SD inpainting | External weights | [`sdinpainting`](https://github.com/newzjh/AIImage/releases/tag/sdinpainting) |
| MONAI WholeBrain | External model; medical data is excluded | [`MONAI_WholeBrain`](https://github.com/newzjh/AIImage/releases/tag/MONAI_WholeBrain) |
| VISTA3D skull | External model; medical data is excluded | [`vista3d_skull`](https://github.com/newzjh/AIImage/releases/tag/vista3d_skull) |
| VISTA3D spine | External model; medical data is excluded | [`vista3d_spine`](https://github.com/newzjh/AIImage/releases/tag/vista3d_spine) |

Use the current `AIImageModelReleaseManifest.json` written by `Aexis/Release/Build Reduced/Prepare Model Release Assets` to identify an exact archive asset. The runtime/model-download UI understands both generated archives and configured flat release files. Do not invent an archive name from a display name.

Default sample model paths are below `Application.streamingAssetsPath` under `Clip`, `CodeFormer`, `DeepFileV2`, `Matting`, `RealESRGAN`, and `Yolo`. GFPGAN, Stable Diffusion, SD inpainting, MONAI/VISTA, and Qwen weights are excluded from the normal package sample because of size and redistribution constraints.

## 7. Example evidence

All following images are actual documented runner outputs. Complete runner timings and the test input notes are maintained in the [root README](../README.md#examples).

### Face restoration

![CodeFormer before and after on 03.jpg](../Packages/com.aexis/Documentation~/images/codeformer-03-before-after.png)

![GFPGAN before and after on 03.jpg](../Packages/com.aexis/Documentation~/images/gfpgan-03-before-after.png)

Windows/Vulkan 2026-07-29 evidence, both using `ref/03.jpg`: CodeFormer 17,958 ms with one detected face; GFPGAN 6,053 ms with the strict Pack4 guard. The GFPGAN image is the raw runner output and visibly distorts this input, so it is execution evidence rather than a quality claim.

### Matting, inpainting, and upscaling

![Matting composite](../Packages/com.aexis/Documentation~/images/matting-composite.png)

![YOLO plus DeepFillV2 before and after on 3.png](../Packages/com.aexis/Documentation~/images/yolo-deepfillv2-3-before-after.png)

![Real-ESRGAN before and after on 03.jpg](../Packages/com.aexis/Documentation~/images/realesrgan-03-before-after.png)

![YOLO plus Stable Diffusion inpainting before and after on 3.png](../Packages/com.aexis/Documentation~/images/yolo-sd-inpainting-3-before-after.png)

Windows/Vulkan 2026-07-29 evidence: Real-ESRGAN AnimeVideo v3 x4 completed in 1,057 ms for `ref/03.jpg`. For `ref/deepfillv2/DeepFillv2-main/test_data/3.png`, YOLO found seven people with mask coverage `0.042730`; DeepFillV2 completed in 2,196 ms (1,995 ms inference), and 12-step Stable Diffusion inpainting completed in 630,213 ms. The before/after pairs intentionally retain residual artifacts from the actual outputs.

### Qwen, CLIP, MONAI, and VISTA

![Qwen multimodal input](../Packages/com.aexis/Documentation~/images/qwen-and-clip-input.jpg)

Qwen Q4 passed a Windows strict-texture smoke without a visible response, then completed the 2026-07-26 MuMu Vulkan Main2 pass in 326,821 ms across 103 decoder steps, emitting 397 visible characters and detecting four people. Q8 passed a multimodal strict-texture smoke with six generated tokens. CLIP's successful score evidence is from 2026-07-23 (`Photo` 0.332389 and `Portrait` 0.265945). The 2026-07-28 strict CLIP run was rejected because `transpose_121` requested an undeclared temporary RT; this is a documented current limitation.

> **MONAI/VISTA screenshot slot:** Reserved for validated, distributable medical input. No medical image, private golden, checkpoint, or data result is included in `com.aexis`.

## 8. Tested environments

| Platform | Device / Unity / graphics API | Current result |
| --- | --- | --- |
| Windows 11 Pro 64-bit | Intel Arc Graphics, Unity 6000.2.7f2, Vulkan | Passed for the documented 2026-07-29 runner results |
| macOS | **TBD: machine, GPU, Unity, graphics API** | Existing `Tools/AIImage_MACOS.build-failure.txt`; no pass claimed |
| Android (MuMu emulator) | MuMu ADB `127.0.0.1:16384`; runtime report: vivo V2241A, ARM64, Adreno (TM) 650, 8961 MiB, Unity 6000.2.7f2, Vulkan GameActivity | Passed 2026-07-26 full default Main2: CodeFormer 6,946 ms; Real-ESRGAN 519 ms; GFPGAN 3,160 ms; YOLO 657 ms (four persons); DeepFillV2 3,442 ms; Matting 1,014 ms; CLIP 2,208 ms; Qwen Q4 326,821 ms (103 steps, 397 visible characters). |
| iOS | **TBD: device, SoC/GPU, Unity, graphics API** | **TBD: record build, runner, timing, and thermal condition** |

Unity 2022.3 through 6000.3 is the package compatibility target, not a record that every editor/platform combination has passed this application workload. Always validate with a real graphics device; `-nographics` is forbidden for Aexis package, shader, or runner validation.

The MuMu result is a conservative Android fallback measurement, not a physical-device benchmark. A Snapdragon 888 handset has been reported as materially faster; its exact measurement remains a placeholder until the device, build, thermal state, graphics API, runner configuration, and timing are recorded.

## 9. Release validation

1. Run `dotnet build AIImage.sln -v minimal -m:1`.
2. Compile and run the default release smoke with Unity 6000.2.7f2 on a graphics-capable host.
3. For full-range claims, create separate empty projects outside this repository for Unity 2022.3.9f1, 2023.2.20f1c1, and 6000.3.2f1. Never upgrade the current project to perform those checks.
4. In each isolated project, validate both a `file:` package reference and the exported `.unitypackage`; compile first with only the package, then with the Main2 sample installed.
5. Review logs for C# errors, shader errors, import failures, package-lock conflicts, and strict texture-plan rejections.
6. Do not publish GFPGAN, Stable Diffusion, SD inpainting, MONAI/VISTA, Qwen, private golden data, or any other model artifact until its source URL, immutable revision/checksum, license, copyright, conversion history, and redistribution approval are recorded.

## 10. License

`com.aexis/package.json` declares MIT as the Aexis source-license target. The pre-release `Packages/com.aexis/LICENSE.md` retains a release-audit gate, so the audit must be completed before any archive is represented as an MIT release.

MIT source licensing does not grant rights to weights, checkpoints, medical data, or third-party sample artifacts. Follow [Third Party Notices](../Packages/com.aexis/Third%20Party%20Notices.md) and the model-distribution policy for every release.
