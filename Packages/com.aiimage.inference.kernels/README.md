# AIImage Inference Kernels

`com.aiimage.inference.kernels` contains the `NcnnCompute` compute shader and its Pack4 HLSL include groups. It depends on `unitygpu` to make the dependency direction explicit, but the backend resolves the shader through Unity Resources to avoid a reverse C# assembly reference.

Public boundary: packaged kernel asset names and metadata. Private boundary: experimental kernels, model-specific shader branches, and any HLSL whose origin has not passed the audit. Production code must never depend on `validation`.
