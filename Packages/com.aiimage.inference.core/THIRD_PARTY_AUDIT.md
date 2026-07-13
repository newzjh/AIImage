# Third-Party License Audit

The table is an inventory, not a declaration that any source or model is redistributable. Each entry must be reviewed before the package is published under MIT, Apache-2.0, or another project license.

| Source | Current use or planned adapter | Reported upstream license | Required audit action |
| --- | --- | --- | --- |
| Tencent [ncnn](https://github.com/Tencent/ncnn) | `.param/.bin` semantics, layer and Vulkan behavior used as a compatibility target | BSD 3-Clause | Confirm whether any source was copied, retain notices, and review model-file licenses separately. |
| Unity [Sentis](https://docs.unity3d.com/Packages/com.unity.sentis@latest/) / ONNX adapters | Operator naming and future importer compatibility | Unity package license and ONNX licenses vary by version | Pin source versions and include the applicable Unity/ONNX notices before copying code or schemas. |
| Alibaba [MNN](https://github.com/alibaba/MNN) | Future offline MNN-to-IR importer and oracle only | Apache-2.0 | Verify converter, FlatBuffer schema, and bundled tool licenses; do not include native runtime by default. |
| [MONAI](https://github.com/Project-MONAI/MONAI) and VISTA/MONAI Model Zoo | Private medical export/oracle workflow; no code, data, or weights are in these packages | MONAI Apache-2.0; model and data terms vary | Review each bundle, model card, prompt asset, dataset, and clinical-data restriction before any distribution. |
| AIImage Pack4 HLSL (`NcnnCompute.compute` and includes) | Migrated Unity shader assets | AIImage provenance under review; operator behavior may derive from public references | Trace every borrowed snippet/reference, document authorship, and preserve required notices before publishing. |
| Real-ESRGAN ncnn Vulkan assets | Existing private StreamingAssets reference application | MIT stated by `Assets/StreamingAssets/RealESRGAN/README.md`; weights may differ | Keep out of packages; verify each `.bin/.param` model's license and attribution. |
| CLIP, YOLO, GFPGAN, CodeFormer, Stable Diffusion, matting, and VISTA weights | Existing private application runners and StreamingAssets/tools | Varies by model and checkpoint | Keep out of packages; record exact checkpoint URL, version, license, attribution, and redistribution terms. |

The release gate is: all copied source identified, all notices retained, package license compatibility approved, and every sample verified to contain only synthetic/openly redistributable assets.
