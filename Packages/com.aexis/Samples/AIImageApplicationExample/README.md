# AIImage Main2 Application Example

This is the single complete application example for `com.aexis`. It combines the former reusable runner sample with the full AIImage Main2 application: `Main2`, MainView2, DesignView, LibraryView, UI Toolkit assets, all runners, QWEN sessions, MONAI/VISTA code, editor tests, debug tools, batch helpers, and sample-local dependency isolation.

## Import

1. Install `com.aexis`.
2. In Package Manager, import **AIImage Main2 Application Example**.
3. Run `Aexis/Examples/Install Main2 Application StreamingAssets`.
4. Open `Scenes/Main2.unity`, or choose `Aexis/Examples/Open Main2 Application Scene`.

The installer copies the sample `StreamingAssets` tree to `Assets/StreamingAssets`, the location Unity includes in player builds. All model paths are relative to `Application.streamingAssetsPath`; the runner catalog defaults to `Clip`, `CodeFormer`, `DeepFileV2`, `Matting`, `RealESRGAN`, and `Yolo` under that directory.

## Included model payload

The sample carries the permitted default model files for Clip, CodeFormer, DeepFillV2, Matting, RealESRGAN, and YOLO, together with their tokenizer/configuration assets. Before publishing an archive containing those files, complete the provenance and redistribution review in `../../Documentation~/model-distribution.md`.

GFPGAN, Stable Diffusion, SD Inpainting, MONAI/VISTA, and QWEN runners and configuration are included, but their model weights are deliberately omitted because of package-size and redistribution constraints. Add licensed weights to the paths expected by the corresponding runner before running inference.

## External model releases

Use the current asset list on the relevant GitHub Release page or the model-download UI; do not infer archive names from a runner name.

| Model group | Release download page |
| --- | --- |
| Qwen3.5 mobile Q4 | [`qwen3.5_0.8b_mobile_q4`](https://github.com/newzjh/AIImage/releases/tag/qwen3.5_0.8b_mobile_q4) |
| Qwen3.5 mobile Q8 | [`qwen3.5_0.8b_mobile_q8`](https://github.com/newzjh/AIImage/releases/tag/qwen3.5_0.8b_mobile_q8) |
| GFPGAN | [`gfpgan`](https://github.com/newzjh/AIImage/releases/tag/gfpgan) |
| Stable Diffusion inpainting | [`sdinpainting`](https://github.com/newzjh/AIImage/releases/tag/sdinpainting) |
| MONAI WholeBrain | [`MONAI_WholeBrain`](https://github.com/newzjh/AIImage/releases/tag/MONAI_WholeBrain) |
| VISTA3D skull | [`vista3d_skull`](https://github.com/newzjh/AIImage/releases/tag/vista3d_skull) |
| VISTA3D spine | [`vista3d_spine`](https://github.com/newzjh/AIImage/releases/tag/vista3d_spine) |

These are external model downloads, not package payloads or license grants. Medical input, data, and checkpoints remain excluded from `com.aexis`; obtain and use them only under their applicable terms.

## Reduced player release and model delivery

Use `Aexis/Release/Build Reduced` for Main2 player builds. The release builder never moves,
deletes, or stages either `Assets/StreamingAssets` or package sample `StreamingAssets`. It
rewrites only the generated player output and includes the Main2 default payloads: MobileCLIP
S0, CodeFormer, Matting, Real-ESRGAN x4plus anime, GFPGAN, YOLOv8 person segmentation, DeepFillV2 case1
ONNX, and Qwen3.5 mobile Q8. MONAI/VISTA, Stable Diffusion, HiFill, alternate YOLO/ESRGAN
variants, full-precision Qwen, and other non-default assets are excluded.

`Aexis/Release/Build Reduced/Prepare Model Release Assets` creates named ZIP artifacts and a
release manifest for `newzjh/AIImage`. The downloader also maps the repository's existing
per-release flat assets to their required model paths. Use `Tools/AIImage/Download Models...` to
download optional models into the editor persistent model directory. Main2 presents a UI Toolkit confirmation dialog with progress and cancellation
when a runtime action needs a missing downloadable group. The reduced `.unitypackage` excludes
GFPGAN and Qwen weights in accordance with package redistribution policy; those assets are
always delivered through the model release archives.

## Dependency isolation

`Aexis` Runtime does not reference UniTask or SharpZipLib. The application example shades the source it needs into `Aexis.Samples.Async` and `Aexis.Samples.SharpZipLib`, so importing the sample does not declare, preload, or conflict with the host project's `Cysharp.Threading.Tasks` or `ICSharpCode.SharpZipLib` assemblies.

## Test scope

The copied Editor directory contains the existing AIImage tests and batch-debug tooling. Checks needing an omitted model, external executable, medical input, private golden result, or platform-native plugin remain unavailable until that artifact is installed. MONAI/VISTA model execution is intentionally not a post-import smoke test.

For target-machine macOS and iOS timing evidence, use the [Apple runtime smoke guide](../../Documentation~/apple-runtime-smoke.md). It builds a Metal test Player, runs the default runner set, and writes a JSON report without adding the test inputs to package or project source assets.
