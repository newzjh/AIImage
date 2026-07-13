# AIImage Inference Unity GPU

`com.aiimage.inference.unitygpu` is the Unity-specific backend for Pack4 RenderTexture/TextureArray and CommandBuffer execution. Its existing compatibility namespace is `NcnnCompute`; application runners construct sessions through `NcnnInferenceSessionFactory` so the execution core remains package-owned.

Public boundary: backend session creation and texture-native execution types needed by application integration. Private boundary: model paths, `MonoBehaviour` runners, UI, private model manifests, data, and debug/oracle workflows. The package depends only on `com.aiimage.inference.core` plus the host project's UniTask assembly. It locates kernel assets by Unity Resources name, so it must be installed with `com.aiimage.inference.kernels`.

This package must not reference `com.aiimage.inference.validation`.
