# Screenshots and Videos

## Screenshot Gallery

Use the PNGs in `MarketingImages/` in the order below. Each showcase is rendered from a repository runner artifact and has English labels. The left panel is the input or intermediate state; the right panel is the corresponding output or measured inference result.

| Order | Upload file | Runner | Evidence shown | Listing caption |
| ---: | --- | --- | --- | --- |
| 1 | `showcase-codeformer-face-restoration.png` | CodeFormer | Before/after restoration on `ref/03.jpg` | Face restoration runner: input and raw composited output. |
| 2 | `showcase-realesrgan-x4-upscaling.png` | Real-ESRGAN AnimeVideo v3 x4 | Before/after output on `ref/03.jpg` | Texture-native x4 upscaling runner example. |
| 3 | `showcase-foreground-matting.png` | Matting | Alpha matte and green-background composite | Foreground matting: alpha output and composited result. |
| 4 | `showcase-yolo-person-segmentation.png` | YOLO segmentation | Person mask overlay and masked output | Person segmentation runner: mask overlay and output artifact. |
| 5 | `showcase-yolo-deepfillv2-inpainting.png` | YOLO + DeepFillV2 | Before/after on the documented beach input | Person segmentation followed by DeepFillV2 inpainting. Raw runner output; residual artifacts are retained. |
| 6 | `showcase-yolo-sd-inpainting.png` | YOLO + Stable Diffusion Inpainting | Before/after on the documented beach input | Person segmentation followed by Stable Diffusion inpainting. External weights; raw runner output. |
| 7 | `showcase-clip-mobileclip-s0.png` | CLIP MobileCLIP S0 | Image input and recorded ranked-label result | Image embedding and label ranking example. |
| 8 | `showcase-qwen35-multimodal.png` | Qwen3.5 Mobile Q4/Q8 | Image input and recorded generation report | Multimodal runner evidence for the external Qwen model variants. |
| 9 | `showcase-gfpgan-face-restoration.png` | GFPGAN | Before/after on `ref/03.jpg` | Raw execution evidence only. This documented input visibly distorts, so it is not a quality claim. |

Do not replace a runner artifact with a generated image and describe it as an inference result. Generated or procedural visual material may be used only for the Cover, Card, Icon, and Social image background layers.

## Product Video Script

Create one 75 to 90 second English video at 1920x1080, 30 fps. Capture the actual Main2 sample on a graphics-capable machine. Keep the final video free of private test data, medical data, credentials, download tokens, or unlicensed model files.

| Time | Scene | On-screen English copy | Capture notes |
| --- | --- | --- | --- |
| 0:00-0:05 | Cover image and Main2 scene | `Aexis | On-device inference for Unity` | Use the generated Cover only as an introduction, then cut to the real sample. |
| 0:05-0:14 | Package Manager | `Install com.aexis and import the Main2 Application Example` | Show the three shipped samples. Do not show local paths or credentials. |
| 0:14-0:23 | Installer and scene open | `Install StreamingAssets, then open Main2` | Show `Aexis/Examples/Install Main2 Application StreamingAssets`. |
| 0:23-0:38 | CodeFormer and Real-ESRGAN | `Input -> runner output` | Use a licensed demo image. Show the original and the actual final output side by side. |
| 0:38-0:52 | Matting and YOLO | `Texture-native GPU runner examples` | Show alpha/mask output followed by the actual composite. |
| 0:52-1:06 | DeepFillV2 and CLIP | `Image transformation and embedding workflows` | Retain visible artifacts; do not imply perfect removal or universal quality. |
| 1:06-1:17 | Development report and platform evidence | `Runner reports record environment, timing, and diagnostics` | Show a redacted report summary, not an unsupported benchmark chart. |
| 1:17-1:25 | Final product frame | `ONNX and NCNN import | Compute-shader GPU path | Unity 2022.3+` | End on the real Main2 UI or product Cover. |

## Video Metadata

**Title:** Aexis - On-Device Inference Engine for Unity

**Description:**

Explore the Aexis Unity package and its AIImage Main2 example. The video shows the import flow, texture-native GPU runner examples, actual output comparisons, and development-player reporting. Model availability, compatibility, and performance depend on the selected graph, device, graphics API, and licensing terms. See the package documentation for model distribution and validated environments.

## Capture Checklist

- Record in English and use a clean Unity layout at 1920x1080 or higher.
- Use a real graphics device; do not present `-nographics` as runner validation.
- Show both the input and output for image-transforming runners.
- State the model and device only when the recording contains the matching runner report.
- Keep model-download URLs and external-weight notices visible where relevant.
- Remove API keys, local usernames, project paths, browser tabs, notifications, and development console errors.
- Use only source images whose license permits product marketing. Preserve the raw-output disclaimer for GFPGAN and the inpainting examples.
- Do not show medical data, MONAI/VISTA examples, or third-party weights unless their permissions and release records are complete.
