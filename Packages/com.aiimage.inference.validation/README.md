# AIImage Inference Validation

`com.aiimage.inference.validation` contains editor-only package-boundary checks and future oracle, golden, and regression tooling. It may depend on `core`, `unitygpu`, and `kernels`; no production package may reference it.

Public boundary: validation entry points and test assets. Private boundary: application-specific runner tests, patient data, model checkpoints, and proprietary golden fixtures, which remain in AIImage.
