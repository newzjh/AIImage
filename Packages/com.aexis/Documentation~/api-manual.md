# Aexis Public API Manual

## Core contracts

Namespace: `Aexis`

| API | Use |
| --- | --- |
| `IInferenceSession` | Common session identity, state, backend, and disposal contract |
| `IInferenceTensor` | Backend-owned tensor descriptor and resource identity contract |
| `TensorDescriptor` | Immutable logical/storage shape, layout, element type, and debug name |
| `ModelManifest` | Serializable FP16/BF16/FP32 precision, mixed-precision, quantization, and precision-gate contract |
| `AexisImportCalibration` | Import-time activation-range calibration helper |
| `AexisPrecisionGateEvaluator` | Measure FP32-versus-low-precision output error before accepting a model variant |
| `AexisDetectionPostprocessing` | Proposal, DetectionOutput, and YOLO class-aware NMS decoding helper |
| `InferenceContractException` | Invalid model or execution-contract configuration |

Create `TensorDescriptor` with positive logical and storage dimensions. A descriptor describes a backend tensor; it does not transfer ownership of its native texture.

## Scheduling

Namespace: `Aexis.Async`

`AexisAsync.YieldFrame()` returns `Task` and resumes after the next Unity synchronization-context continuation. It is the only package async scheduling abstraction. Callers that use UniTask may await Aexis calls from their own adapters, but Aexis never exposes UniTask in a public signature.

## ONNX inspection and planning

Namespace: `Aexis.Onnx`

| API | Use |
| --- | --- |
| `OnnxModelReader.Read(string)` | Parse an `.onnx` file |
| `OnnxModelReader.Read(byte[])` | Parse downloaded or encrypted ONNX bytes |
| `OnnxD3Importer.Import(...)` | Build ONNX D3 import metadata |
| `OnnxExecutionAdapter.TryAdapt(...)` | Convert a supported shape/index node to an execution contract |
| `OnnxExecutionShapePlanner.Validate(...)` | Validate GPU shape-capacity and dynamic-shape policy |

`OnnxModel` exposes graph nodes, graph inputs/outputs, value information, and initializers. The reader also accepts a `SparseTensorProto` initializer only when its rank is 1-4, its dense capacity is at most 16,777,216 elements, its INT64 coordinates are in range and unique, and its values can be represented by the immutable texture upload. It is densified once during import into the immutable weight source; it never creates a sparse activation or a CPU execution fallback. Use the reader for validation and lowering. It does not promise arbitrary ONNX operator execution by itself; route a supported model to the appropriate execution lowering.

The verified default-domain ONNX import range is opset 7 through 20. ONNX `Gelu` is admitted only at its introduction opset 20 or later and preserves `approximate=none` as the erf semantic and `approximate=tanh` as the fast semantic in one Pack4 CommandBuffer dispatch. Other values are terminal import diagnostics.

`AexisOnnxGraphLoweringOptions.enableBoundedControlFlowLowering` is an opt-in static expansion profile, not a runtime control-flow backend. It accepts only a constant-selected `If`, a fixed-trip `Loop` with a constant-true body condition, and a forward axis-0 `Scan` with declared static capacity. The importer emits ordinary `Slice`, `Concat`, `Identity`, and compute layers into the one Pack4 CommandBuffer render-texture plan. Bounded `SequenceConstruct`/`SplitToSequence`, `SequenceAt`, `SequenceLength`, `SequenceInsert`, `ConcatFromSequence`, and concrete `Optional` values use the same representation. Data-dependent branches, loop termination, unbounded sequence values, empty optionals, and sparse activation values are rejected before dispatch.

`AexisOnnxGraphLoweringOptions.enableBoundedHighRankArenaLowering` is an opt-in static rank-5..8 activation profile. It requires a leading static batch extent of `1`, positive `Int32`-sized extents, and no initializer-backed high-rank value. Aexis preserves the full ONNX logical shape for diagnostics and graph contracts, then removes batch and folds only leading non-spatial axes into physical Pack4 `C`, retaining trailing `D,H,W`; for example `[1,2,2,2,2,2]` maps to physical `CDHW [4,2,2,2]`. Only `Identity`, `Relu`, `Sigmoid`, `Tanh`, and exact-logical-shape `Add`/`Sub`/`Mul`/`Div` may consume this profile. Axis-aware, reshaping, reduction, convolution, high-rank immutable-weight, and data-dependent operations are rejected before dispatch. The external texture descriptor must use the mapped physical `CDHW` shape; no activation buffer conversion or readback is available.

Unary lowering uses the stable ncnn `UnaryOp::OperationType` ABI and records a single Pack4 CommandBuffer node. In addition to the standard ONNX unary operators, reference-compatible `Square`, `Rsqrt`, `Log10`, `Trunc`, `Expm1`, and `Log1p` lower to operation codes `4`, `6`, `17`, `19`, `21`, and `27`; Sentis-compatible `Relu6` lowers directly to the same single-dispatch `Clip[0,6]` profile. These profiles preserve the logical tensor shape, mask Pack4 tail lanes, and accept only statically proven rank-1 through rank-4 texture storage; they never materialize an activation through a `ComputeBuffer` or CPU readback.

## NCNN load and session lifecycle

Namespace: `Aexis.Ncnn`

```csharp
var ops = new NcnnOps();
var session = NcnnInferenceSessionFactory.Create(ops);
using var weights = new NcnnBinReader(binStream);
await session.LoadModelAsync(paramText, weights, OnLoadProgress, cancellationToken);

// Build a Pack4 RenderTexture input matching the model input contract.
RenderTexture output = session.ForwardPack4(input, inputPacks, "data");
session.Release();
```

| API | Use |
| --- | --- |
| `NcnnOps()` | Create the package-owned compute operator facade and load package shaders |
| `NcnnInferenceSessionFactory.Create(...)` | Create a graph session and apply an optional model manifest |
| `NcnnParamParser.Parse(string)` | Parse NCNN `.param` text for inspection or merge workflows |
| `NcnnParamParser.Parse(byte[])` / `WriteBinary(...)` | Read/write versioned Aexis binary NCNN graph parameters |
| `AexisModelArchive` | Read/write the versioned `.aexis` model archive |
| `NcnnBinReader(Stream)` | Read NCNN `.bin` weights; dispose it after model load |
| `NcnnGraphSession.LoadModel(...)` | Synchronous model load |
| `NcnnGraphSession.LoadModelAsync(...)` | Cooperative frame-yielding load; returns `Task` |
| `NcnnGraphSession.ForwardPack4(...)` | Execute texture-native Pack4 inference |
| `NcnnGraphSession.ForwardPack4WithFixedInputs(...)` | Record strict Pack4 CommandBuffer inference; fixed buffers are GPU upload sources and become textures before layer execution |
| `NcnnGraphSession.Release()` | Release session-owned textures and fixed uploads; idempotent cleanup boundary |

`LoadProgress` reports model/layer load stages. Pass `CancellationToken` to cancel loading safely. Always call `Release` when a component is destroyed or a model is replaced. Do not retain an output after releasing its session.

For asynchronous command recording, use `ForwardPack4WithFixedInputs` when token ids or other immutable GPU buffers are required. The method dispatches each fixed input into an exact Pack4/LinearMat texture at the graph boundary and rejects profiles that would retain a ComputeBuffer activation or materialize an intermediate through the CPU.

## Precision and manifests

Use `NcnnPrecisionMode.Auto` unless a tested manifest selects a supported precision. `ModelManifest` validates FP16/BF16/FP32 activation contracts, INT8/INT4 weight-only contracts, calibrated Pack4 INT8 activation plans, per-layer mixed-precision plans, and output precision gates. Quantized model metadata must provide calibration provenance and explicit node plans where activation quantization is used. Activation plans drive signed or unsigned Pack4 INT8 quantize/dequantize arithmetic in Conv/DWConv/Gemm/InnerProduct on both immediate and CommandBuffer paths.

## Unity model assets and custom layers

`AexisModelAsset` is produced by the package `ScriptedImporter` implementations for `.onnx`, `.param`, and `.aexis`. It contains source bytes, a binary graph, optional weights, and diagnostic JSON. `AexisModelPackager` exposes the offline equivalents for build tools.

Use `AexisCustomLayerRegistry.Register(...)` with `AexisCustomLayerDefinition` to publish a layer factory, schema, and shader kernel identifier. The schema is validated before the factory runs. Put matching `AexisModelExtensionDeclaration` entries in a model graph/archive; unresolved declarations and missing Pack4 kernel profiles are terminal import or execution errors, never Buffer fallbacks.

The built-in P1 visual set is verified by the same strict profile used for dispatch: GridSample, DeformableConv2D, Fold, Flip, GLU, Einsum, Diag, SPP, ROIAlign, ROIPooling, PSROIPooling, Proposal, DetectionOutput, and YOLO output variants. ONNX `RoiAlign` is accepted only for the verified FP32 profile: static batch-one feature input, static `rois[num_rois,4]`, immutable all-zero INT32/INT64 `batch_indices`, `mode=avg`, and `coordinate_transformation_mode=half_pixel`. Each ROI is a Pack4 Texture2DArray depth slice; no result count is read back. BF16 is a logical precision: Aexis keeps its texture storage in FP32, and Pack4 `Cast` applies deterministic BF16 rounding.

## Error handling

Treat parse errors, invalid manifests, unsupported texture paths, and cancellation as actionable failures. Surface the model path, input/output blob name, logical shape, storage shape, and precision mode in application logs. Do not silently reroute a failed texture operation through a generic ComputeBuffer path.
