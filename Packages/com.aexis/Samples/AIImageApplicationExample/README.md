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

## Dependency isolation

`Aexis` Runtime does not reference UniTask or SharpZipLib. The application example shades the source it needs into `Aexis.Samples.Async` and `Aexis.Samples.SharpZipLib`, so importing the sample does not declare, preload, or conflict with the host project's `Cysharp.Threading.Tasks` or `ICSharpCode.SharpZipLib` assemblies.

## Test scope

The copied Editor directory contains the existing AIImage tests and batch-debug tooling. Checks needing an omitted model, external executable, medical input, private golden result, or platform-native plugin remain unavailable until that artifact is installed. MONAI/VISTA model execution is intentionally not a post-import smoke test.
