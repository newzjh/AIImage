# AIImage Inference Core

`com.aiimage.inference.core` defines the stable contracts shared by AIImage inference backends. It is deliberately free of `UnityEngine`, scenes, `MonoBehaviour`, RenderTexture, CommandBuffer, model parsers, and domain logic.

Public API: `TensorDescriptor`, tensor layout/dtype enums, `IInferenceTensor`, `IInferenceSession`, and `InferenceContractException` in the `AIImage.Inference.Core` namespace.

Private implementation belongs in backend packages. `unitygpu` owns Pack4 texture resources and command recording; `kernels` owns HLSL assets; importers and AIImage application runners remain outside this package. Validation code must depend on production packages, never the reverse.

For a file dependency in another Unity project, add the following to that project's `Packages/manifest.json`:

```json
"com.aiimage.inference.core": "file:../AIImage/Packages/com.aiimage.inference.core"
```

The package is pre-release until the third-party license audit in `THIRD_PARTY_AUDIT.md` is closed.
