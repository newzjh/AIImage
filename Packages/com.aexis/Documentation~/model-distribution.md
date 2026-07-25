# Sample Model Distribution

## Included payload paths

After installing the sample payload, paths are relative to `Application.streamingAssetsPath`.

| Family | Default path prefix | Files |
| --- | --- | --- |
| Clip | `Clip/mobileclip_s0_export` | NCNN image encoder, tokenizer data, label embeddings |
| CodeFormer | `CodeFormer/models` | Encoder, generator, face detector NCNN pairs |
| DeepFillV2 | `DeepFileV2` | ONNX source and NCNN pairs |
| Matting | `Matting` | NCNN pair |
| RealESRGAN | `RealESRGAN/models` | `realesrgan-x4plus-anime` default NCNN pair; other variants on demand |
| YOLO | `Yolo` | YOLOv8n-seg and YOLO11n-seg NCNN pairs |

## Excluded payloads

Do not put model files for GFPGAN, Stable Diffusion, SD Inpainting, MONAI/VISTA, or QWEN in this package or its samples. Their Runner configuration may name an external model location, but consumers must obtain the model separately under its own license.

## Release gate

Before publishing a package archive that contains any model artifact, complete an auditable record for every file: upstream project and URL, immutable revision or checksum, upstream license text, copyright notice, modification/export steps, and redistribution approval. Keep this record with the release tag. The MIT license for Aexis source does not grant rights to model weights.

The current repository contains `RealESRGAN/LICENSE`; copy that notice with the RealESRGAN sample payload. Do not infer licenses for other files from their model family name. If provenance is incomplete, omit that model from the public archive while retaining the runner configuration.

## Reduced Main2 delivery

The reduced Main2 release policy is defined by `AIImageModelDelivery`. Its bundled Player set is
limited to MobileCLIP S0, CodeFormer, Matting, Real-ESRGAN x4plus anime, GFPGAN, YOLOv8 person
segmentation, DeepFillV2 case1 ONNX, and Qwen3.5 mobile Q8. The Q8 archive is used instead of the FP32 Qwen
weights. MONAI and VISTA are never added to the reduced release catalog.

`Aexis/Release/Build Reduced/Prepare Model Release Assets` writes one ZIP per delivery group and
`AIImageModelReleaseManifest.json`. Release managers must review the provenance gate above before
uploading those artifacts to `newzjh/AIImage` GitHub Releases. Runtime and editor downloads
support both those ZIP archives and the repository's existing per-release flat assets. Each model
group maps its release tag and asset name to the required persistent target path, so a flat asset
such as a Qwen weight can be installed below `QWEN35/.../weights` without modifying any project
`StreamingAssets` directory.

On Android, each bundled model group is streamed from APK assets into that persistent model root
when first used, because the runners require ordinary file paths. If preparation cannot complete,
the runtime download dialog falls back to the matching GitHub Release archive.
