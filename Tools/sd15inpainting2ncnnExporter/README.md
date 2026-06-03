# sd15inpainting2ncnnExporter

Exports `sd-v1-5-inpainting.ckpt` or `.safetensors` into NCNN `.param` + `.bin` files.

## What it does

- Verifies the checkpoint is an SD1.5 inpainting model by reading the raw state dict.
- Checks the first UNet conv and fails unless `in_channels == 9`.
- Converts the checkpoint into a Diffusers `StableDiffusionInpaintPipeline`.
- Exports `UNet`, `VAE decoder`, `VAE encoder`, and `Text Encoder` to ONNX.
- Runs `onnxsim` on each ONNX graph.
- Converts ONNX to NCNN with `onnx2ncnn` when available.
- Falls back to `pnnx` if `onnx2ncnn` is missing or fails.
- Runs `ncnnoptimize` on the generated NCNN files when available.

## Outputs

The required outputs are written under `output_root/ncnn/`:

- `unet.param`
- `unet.bin`
- `vae.param`
- `vae.bin`
- `text_encoder.param`
- `text_encoder.bin`

The tool also exports:

- `vae_encoder.param`
- `vae_encoder.bin`

That extra encoder pair is useful for `img2img` and `inpainting` runtimes that need a VAE encoder graph in addition to the VAE decoder graph.

## CLI

```powershell
python export_sd15_inpaint_to_ncnn.py `
    --ckpt E:\Projects\AIImage\ref\sd1.5inpainting\sd-v1-5-inpainting.ckpt `
    --output .\output `
    --fp16
```

Optional arguments:

- `--width 512 --height 512`
- `--opset 17`
- `--device cpu|cuda`
- `--onnx2ncnn <path>`
- `--ncnnoptimize <path>`
- `--pnnx <path>`
- `--no-pnnx-fallback`
- `--local-files-only`

## Windows launcher

Use the bundled launcher to create a Python 3.10 virtual environment and install dependencies:

```powershell
.\sd15inpainting2ncnnExporter.bat --ckpt ..\..\ref\sd1.5inpainting\sd-v1-5-inpainting.ckpt --output .\output --fp16
```

## Notes

- `--fp16` keeps export stable on CPU by exporting ONNX in fp32 and using NCNN optimization for fp16 storage.
- The first Diffusers conversion may still need Hugging Face config assets unless they are already cached locally.
- The script writes `manifest.json` and `export.log` to the output root for debugging.
