# Sample Model Distribution

## Included payload paths

After installing the sample payload, paths are relative to `Application.streamingAssetsPath`.

| Family | Default path prefix | Files |
| --- | --- | --- |
| Clip | `Clip/mobileclip_s0_export` | NCNN image encoder, tokenizer data, label embeddings |
| CodeFormer | `CodeFormer/models` | Encoder, generator, face detector NCNN pairs |
| DeepFillV2 | `DeepFileV2` | ONNX source and NCNN pairs |
| Matting | `Matting` | NCNN pair |
| RealESRGAN | `RealESRGAN/models` | Default and anime NCNN pairs |
| YOLO | `Yolo` | YOLOv8n-seg and YOLO11n-seg NCNN pairs |

## Excluded payloads

Do not put model files for GFPGAN, Stable Diffusion, SD Inpainting, MONAI/VISTA, or QWEN in this package or its samples. Their Runner configuration may name an external model location, but consumers must obtain the model separately under its own license.

## Release gate

Before publishing a package archive that contains any model artifact, complete an auditable record for every file: upstream project and URL, immutable revision or checksum, upstream license text, copyright notice, modification/export steps, and redistribution approval. Keep this record with the release tag. The MIT license for Aexis source does not grant rights to model weights.

The current repository contains `RealESRGAN/LICENSE`; copy that notice with the RealESRGAN sample payload. Do not infer licenses for other files from their model family name. If provenance is incomplete, omit that model from the public archive while retaining the runner configuration.
