# Aexis

`com.aexis` is the single Unity Package Manager entry point for the Aexis on-device inference engine.

The package owns its ONNX and NCNN format readers, graph contracts, texture-native GPU execution, compute shader assets, editor tooling, and tests. It does not reference Unity Sentis, Tencent ncnn, ONNX Runtime, or MNN at runtime.

Runtime API assemblies are `Aexis`, `Aexis.Onnx`, `Aexis.Ncnn`, `Aexis.Execution`, and `Aexis.Async`. They are all installed by importing this one package. `Aexis.Ncnn` keeps production inference on Pack4 RenderTextures and CommandBuffer-compatible texture flows; compute buffers are limited to immutable uploads and explicit debug/inspection paths.

The package currently depends on UniTask 2.5.4 through `Aexis.Async`. Its registry resolution is validated separately before public distribution; the fallback is an audited, namespaced UniTask source inclusion.

AIImage is an example application and is not part of the Aexis runtime API.
