---
name: aiimage-ncnn-pack4-runtime
description: AIImage NCNN inference/runtime guardrails for avoiding compute-buffer execution paths except explicit debugging. Use when Codex edits or reviews NCNN layers, NcnnRepro, NcnnOps, runner output readback, tensor layout/storage code, pack4 RenderTexture paths, CommandBuffer paths, RepoVkMat-style storage, fallback/materialization behavior, or any AIImage inference path that might create temporary ComputeBuffer data during model execution.
---

# AIImage NCNN Pack4 Runtime

## Core Rule

Keep AIImage NCNN inference on texture-backed/runtime-native storage. Outside explicit debug-only code, do not add or preserve model-execution paths that fall back to temporary compute buffers.

Preferred runtime storage:

- Use pack4 RenderTexture / texture-array / texture-backed `ComputeTexture` paths for Unity inference.
- Use CommandBuffer-compatible texture flows for asynchronous execution.
- Use RepoVkMat-style runtime-native tensor storage when working on exporter/runtime concepts outside Unity texture code.

## Buffer Policy

Treat buffer execution as a debug or fixed-input exception, not the normal inference path.

- Allow fixed input buffers and immutable constants when they are only upload sources for runtime tensors.
- Allow explicit debug, inspection, baseline dumping, or legacy comparison code to read/write buffers when the caller is clearly debug-only.
- Do not use `GetBufferData`, `Pack4ToBufferCHW`, `Pack4ToBufferCDHW`, texture-to-buffer materialization, or buffer fallback to make normal pack4 RT / CommandBuffer inference pass.
- Do not publish a temporary compute-buffer result and then re-materialize it into texture output as a substitute for implementing the texture path.
- If a layer is missing pack4 RT / CommandBuffer support, implement that path or fail loudly in strict inference mode instead of silently falling back to buffer.

## CommandBuffer Constraint

Unity `CommandBuffer` cannot create arbitrary temporary compute buffers in the same way it can allocate temporary render textures. For CommandBuffer inference:

- Allocate temporary intermediates as pack4 RT / texture-array `ComputeTexture` objects.
- Preserve logical tensor shape separately from physical storage shape.
- Reuse aliasing when reshape/view semantics allow it instead of repacking data.
- Keep fixed buffer inputs as upload sources only; convert them to texture/native runtime storage before layer execution when possible.

## Implementation Checklist

Before changing NCNN inference code, check whether the code runs during model execution or only during debug/readback. If it runs during model execution:

- Prefer `ExecuteRenderTexturePath` or `ExecuteCommandBuffer` over `ExecuteComputeBufferPath`.
- Preserve pack count, logical shape, and storage shape metadata when publishing outputs.
- Avoid hidden fallback helpers such as `GetOrConvertToBuffer` when the caller is in pack4 RT / CommandBuffer mode.
- Route output readback through texture-aware APIs when the result is texture-backed.
- Keep errors actionable: name the layer, blob, path mode, logical shape, storage shape, and fallback that was rejected.

## Validation

After changing inference/runtime behavior:

- Run `dotnet build AIImage.sln -v minimal` or a stricter equivalent.
- Prefer a targeted silent Unity validation for the affected runner.
- For MONAI or VISTA work, also use `$aiimage-unity-batch` and keep its memory-safety defaults.
