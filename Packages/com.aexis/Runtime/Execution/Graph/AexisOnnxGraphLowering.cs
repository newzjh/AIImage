using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Aexis.Onnx;

namespace Aexis.Execution
{
    [Serializable]
    public sealed class AexisOnnxTensorDescriptor
    {
        public string name = string.Empty;
        public TensorDataType dataType = TensorDataType.Unknown;
        // BOOL and INT64 are narrowed to the logical Int32 texture contract.
        // Keep the ONNX elem_type so Cast/CastLike retain exact conversion semantics.
        public int onnxDataType;
        // Source ONNX shape and the texture-runtime shape are both retained so
        // diagnostics never hide the singleton batch dimension removed by ncnn.
        public long[] shape = Array.Empty<long>();
        public long[] runtimeShape = Array.Empty<long>();
        public int batchAxis = -1;
        public bool isInitializer;
    }

    [Serializable]
    public sealed class AexisOnnxLoweringDiagnostic
    {
        public int nodeIndex;
        public string node = string.Empty;
        public string opType = string.Empty;
        public string code = string.Empty;
        public string message = string.Empty;
        public string recommendedAction = string.Empty;
        public bool blocking;
    }

    [Serializable]
    public sealed class AexisOnnxGraphLoweringResult
    {
        public AexisGraphModel graph = new AexisGraphModel();
        public AexisOnnxTensorDescriptor[] tensors = Array.Empty<AexisOnnxTensorDescriptor>();
        // Initializers deliberately remain immutable import artifacts. A caller must upload
        // them into a texture-native model weight store before it dispatches this graph.
        public OnnxTensor[] initializers = Array.Empty<OnnxTensor>();
        public AexisOnnxLoweringDiagnostic[] diagnostics = Array.Empty<AexisOnnxLoweringDiagnostic>();
        public bool IsEligible => Array.TrueForAll(diagnostics, diagnostic => diagnostic != null && !diagnostic.blocking);
    }

    [Serializable]
    public sealed class AexisOnnxGraphLoweringOptions
    {
        public bool rejectDynamicShapes = true;
        public bool rejectUnboundInitializers = true;
        public bool requireStaticBatchOne = true;
        // NonZero/Compress use Aexis' bounded LinearMat profile rather than the
        // variable-sized standard ONNX result. Keep it opt-in so a stock ONNX
        // graph is never silently rewritten to consume a second count output.
        public bool enableBoundedDataIndexLowering;
        public Dictionary<string, int> outputCapacities = new Dictionary<string, int>(StringComparer.Ordinal);
        // Scatter duplicate writes are intentionally rejected by the texture
        // kernel. The exporter/import caller supplies proof by node name.
        public string[] verifiedUniqueScatterNodes = Array.Empty<string>();
        // GPU kernels cannot synchronously throw for an out-of-range dynamic index.
        // Strict import therefore requires exporter proof for index-bearing nodes.
        public string[] verifiedInRangeIndexNodes = Array.Empty<string>();
        // File import enables this automatically. Programmatically assembled graphs may
        // leave it disabled while testing an isolated lowering rule.
        public bool requireDeclaredGraphOutputs;
    }

    // The general ONNX front-end owns parsing/type/shape lowering, while AexisGraphSession
    // remains the runtime. This avoids introducing Unity, Sentis, ORT, or native ncnn
    // dependencies into the ONNX assembly and makes unsupported nodes fail before dispatch.
    public static class AexisOnnxGraphLowering
    {
        private const int MinimumSupportedOpset = 7;
        private const int MaximumSupportedOpset = 19;

        private readonly struct OperatorSchema
        {
            public readonly int minInputs;
            public readonly int maxInputs;
            public readonly int minOutputs;
            public readonly int maxOutputs;
            public readonly int minimumOpset;

            public OperatorSchema(int minInputs, int maxInputs, int minOutputs, int maxOutputs, int minimumOpset = 1)
            {
                this.minInputs = minInputs;
                this.maxInputs = maxInputs;
                this.minOutputs = minOutputs;
                this.maxOutputs = maxOutputs;
                this.minimumOpset = minimumOpset;
            }
        }

        private static readonly Dictionary<string, OperatorSchema> OperatorSchemas = CreateOperatorSchemas();
        private static readonly Dictionary<string, string> OperatorMap = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            { "Identity", "Noop" }, { "Dropout", "Dropout" }, { "Relu", "ReLU" }, { "Sigmoid", "Sigmoid" },
            { "Tanh", "TanH" }, { "Abs", "AbsVal" }, { "Exp", "Exp" }, { "Log", "Log" },
            { "Celu", "CELU" }, { "Elu", "ELU" }, { "Erf", "Erf" }, { "Gelu", "GELU" },
            { "HardSigmoid", "HardSigmoid" }, { "HardSwish", "HardSwish" }, { "LeakyRelu", "ReLU" },
            { "PRelu", "PReLU" }, { "Selu", "SELU" }, { "Softplus", "Softplus" },
            { "Softsign", "Softsign" }, { "Shrink", "Shrink" }, { "IsInf", "IsInf" }, { "IsNaN", "IsNaN" },
            { "Swish", "Swish" }, { "ThresholdedRelu", "ThresholdedRelu" }, { "Clip", "Clip" },
            { "LRN", "LRN" }, { "Pad", "Padding" }, { "Resize", "Interp" }, { "Upsample", "Interp" },
            { "ExtractImagePatches", "ExtractPatches" },
            { "DepthToSpace", "PixelShuffle" }, { "SpaceToDepth", "Reorg" },
            { "Conv", "Convolution" }, { "ConvTranspose", "Deconvolution" }, { "BatchNormalization", "BatchNorm" },
            { "InstanceNormalization", "InstanceNorm" }, { "LayerNormalization", "LayerNorm" },
            { "MaxPool", "Pooling" }, { "AveragePool", "Pooling" }, { "GlobalAveragePool", "Pooling" }, { "GlobalMaxPool", "Pooling" },
            { "Gemm", "Gemm" }, { "MatMul", "MatMul" }, { "Softmax", "Softmax" }, { "LogSoftmax", "Softmax" }, { "Hardmax", "Softmax" }, { "Concat", "Concat" },
            { "Trilu", "Trilu" },
            { "Transpose", "Permute" }, { "Reshape", "Reshape" }, { "Flatten", "Flatten" }, { "Squeeze", "Squeeze" }, { "Split", "Slice" },
            { "Unsqueeze", "ExpandDims" }, { "Slice", "Crop" }, { "Tile", "Tile" }, { "Cast", "Cast" }, { "CastLike", "Cast" },
            { "ReduceSum", "Reduction" }, { "ReduceSumSquare", "Reduction" }, { "ReduceMean", "Reduction" },
            { "ReduceMax", "Reduction" }, { "ReduceMin", "Reduction" }, { "ReduceProd", "Reduction" },
            { "ReduceL1", "Reduction" }, { "ReduceL2", "Reduction" }, { "ReduceLogSum", "Reduction" }, { "ReduceLogSumExp", "Reduction" },
            { "Shape", "Shape" }, { "Size", "Size" }, { "Range", "Range" }, { "ConstantOfShape", "ConstantOfShape" },
            { "Expand", "Expand" }, { "ArgMax", "ArgMax" }, { "ArgMin", "ArgMin" }, { "Where", "Where" },
            { "TopK", "TopK" }, { "NonZero", "NonZero" }, { "OneHot", "OneHot" }, { "CumSum", "CumSum" },
            { "Compress", "Compress" }, { "Gather", "Gather" }, { "GatherElements", "GatherElements" },
            { "GatherND", "GatherND" }, { "Scatter", "Scatter" }, { "ScatterElements", "ScatterElements" }, { "ScatterND", "ScatterND" },
            { "Einsum", "MatMul" },
            { "Mod", "BinaryOp" }, { "Equal", "BinaryOp" }, { "Greater", "BinaryOp" }, { "GreaterOrEqual", "BinaryOp" },
            { "Less", "BinaryOp" }, { "LessOrEqual", "BinaryOp" }, { "And", "BinaryOp" }, { "Or", "BinaryOp" },
            { "Xor", "BinaryOp" }, { "Not", "UnaryOp" }, { "Sum", "Eltwise" }, { "Mean", "Eltwise" }
        };

        private static readonly Dictionary<string, int> BinaryOps = new Dictionary<string, int>(StringComparer.Ordinal)
        {
            { "Add", 0 }, { "Sub", 1 }, { "Mul", 2 }, { "Div", 3 }, { "Max", 4 }, { "Min", 5 }, { "Pow", 6 }, { "Mod", 19 },
            { "Equal", 20 }, { "Greater", 21 }, { "GreaterOrEqual", 22 }, { "Less", 23 }, { "LessOrEqual", 24 },
            { "And", 25 }, { "Or", 26 }, { "Xor", 27 }
        };

        // These values are the stable ncnn UnaryOp::OperationType ABI. Keeping
        // the mapping centralized prevents named ONNX aliases from drifting
        // away from native .param UnaryOp layers.
        private static readonly Dictionary<string, int> UnaryOps = new Dictionary<string, int>(StringComparer.Ordinal)
        {
            { "Neg", 1 }, { "Floor", 2 }, { "Ceil", 3 }, { "Sqrt", 5 },
            { "Reciprocal", 15 }, { "Sin", 9 }, { "Cos", 10 }, { "Tan", 11 },
            { "Asin", 12 }, { "Acos", 13 }, { "Atan", 14 }, { "Round", 18 },
            { "Sign", 20 }, { "Sinh", 22 }, { "Asinh", 23 }, { "Cosh", 24 },
            { "Acosh", 25 }, { "Atanh", 26 }, { "Not", 28 }
        };

        public static AexisOnnxGraphLoweringResult Import(string path, AexisOnnxGraphLoweringOptions options = null)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("An ONNX path is required.", nameof(path));
            return Lower(OnnxModelReader.Read(path), RequireDeclaredOutputs(options));
        }

        public static AexisOnnxGraphLoweringResult Import(byte[] modelBytes, AexisOnnxGraphLoweringOptions options = null)
        {
            return Lower(OnnxModelReader.Read(modelBytes), RequireDeclaredOutputs(options));
        }

        public static AexisOnnxGraphLoweringResult Lower(OnnxModel model, AexisOnnxGraphLoweringOptions options = null)
        {
            if (model?.graph == null)
                throw new ArgumentException("An ONNX graph is required.", nameof(model));
            options ??= new AexisOnnxGraphLoweringOptions();

            var result = new AexisOnnxGraphLoweringResult();
            var descriptors = new Dictionary<string, AexisOnnxTensorDescriptor>(StringComparer.Ordinal);
            var diagnostics = new List<AexisOnnxLoweringDiagnostic>();
            ValidateGraphContract(model, options, diagnostics);
            var immutableTensors = new Dictionary<string, OnnxTensor>(model.graph.initializers, StringComparer.Ordinal);
            foreach (var input in model.graph.inputs)
                Register(descriptors, input.name, input.dataType, input.dims, false, input.onnxDataType);
            foreach (var output in model.graph.outputs)
                Register(descriptors, output.name, output.dataType, output.dims, false, output.onnxDataType);
            foreach (var value in model.graph.valueInfos)
                Register(descriptors, value.name, value.dataType, value.dims, false, value.onnxDataType);
            foreach (var pair in model.graph.initializers)
                Register(descriptors, pair.Key, pair.Value.dataType, pair.Value.dims, true, pair.Value.onnxDataType);

            foreach (var input in model.graph.inputs)
            {
                if (model.graph.initializers.ContainsKey(input.name)
                    || !descriptors.TryGetValue(input.name, out var descriptor))
                    continue;
                if (input.onnxDataType == 7)
                {
                    diagnostics.Add(Diagnostic(-1, input.name, "Input", "int64-runtime-input-requires-narrowing",
                        "Aexis runtime index textures use logical Int32 values and cannot accept a dynamic ONNX INT64 input without a range-checked conversion.",
                        "Export this input as INT32 or insert a range-proven Cast to INT32 before strict import.", true));
                }
                if (descriptor.shape != null && (descriptor.shape.Length == 3 || descriptor.shape.Length == 4))
                {
                    if (descriptor.shape[0] == 1)
                    {
                        SetBatchContract(descriptor, 0);
                    }
                    else if (options.requireStaticBatchOne)
                    {
                        diagnostics.Add(Diagnostic(-1, input.name, "Input", "unsupported-batch-size",
                            "Aexis Pack4 ONNX execution requires static batch=1, but input " + input.name + " has batch=" + descriptor.shape[0].ToString(CultureInfo.InvariantCulture) + ".",
                            "Export a static batch=1 model or implement an explicit batched texture contract.", true));
                    }
                }
                if (descriptor.shape == null || descriptor.shape.Length > 4 || options.rejectDynamicShapes && HasDynamic(descriptor.shape))
                {
                    diagnostics.Add(Diagnostic(-1, input.name, "Input", "unsupported-input-shape",
                        "Aexis P0 requires a static input rank from 0 through 4.",
                        "Export static value_info with rank <= 4.", true));
                }
                else if (HasZeroExtent(descriptor.shape))
                {
                    diagnostics.Add(Diagnostic(-1, input.name, "Input", "empty-input-tensor",
                        "Aexis texture storage cannot represent an input with a zero extent.",
                        "Remove empty tensors before import.", true));
                }
            }

            // ONNX initializers are not graph inputs. The result exposes the exact source
            // tensors so the package importer can create immutable Texture2DArray/RT stores.
            // We intentionally do not synthesize a ComputeBuffer-backed Constant layer.
            if (options.rejectUnboundInitializers)
            {
                foreach (var pair in model.graph.initializers)
                    diagnostics.Add(Diagnostic(-1, pair.Key, "Initializer", "initializer-upload-required",
                        "Initializer " + pair.Key + " requires a texture-native immutable upload before graph dispatch.",
                        "Bind this tensor through the package weight importer; do not materialize it into a normal inference ComputeBuffer.", false));
            }

            foreach (var input in model.graph.inputs)
            {
                if (model.graph.initializers.ContainsKey(input.name))
                    continue;
                var inputLayer = CreateLayer("Input", "input_" + input.name, Array.Empty<string>(), new[] { input.name });
                if (descriptors.TryGetValue(input.name, out var descriptor))
                {
                    SetNcnnShapeParams(inputLayer, descriptor.runtimeShape);
                    inputLayer.stringParams["onnx.shape"] = Join(descriptor.shape);
                    inputLayer.stringParams["runtime.shape"] = Join(descriptor.runtimeShape);
                    inputLayer.stringParams["batch_axis"] = descriptor.batchAxis.ToString(CultureInfo.InvariantCulture);
                }
                result.graph.layers.Add(inputLayer);
            }

            for (var index = 0; index < model.graph.nodes.Count; index++)
            {
                var node = model.graph.nodes[index];
                if (node == null || string.IsNullOrWhiteSpace(node.opType))
                {
                    diagnostics.Add(Diagnostic(index, string.Empty, string.Empty, "invalid-node", "ONNX graph contains a node without an operator type.", "Re-export the ONNX graph.", true));
                    continue;
                }
                if (!string.IsNullOrEmpty(node.domain) && !string.Equals(node.domain, "ai.onnx", StringComparison.Ordinal))
                {
                    diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "unsupported-domain", "Custom ONNX domain " + node.domain + " is not registered.", "Lower the custom operator to the canonical ONNX set before import.", true));
                    continue;
                }
                if (string.Equals(node.opType, "Constant", StringComparison.Ordinal))
                {
                    LowerConstant(index, node, descriptors, immutableTensors, diagnostics);
                    continue;
                }
                if ((string.Equals(node.opType, "Shape", StringComparison.Ordinal)
                     || string.Equals(node.opType, "Size", StringComparison.Ordinal))
                    && TryLowerStaticDimensionNode(index, node, descriptors, immutableTensors, diagnostics))
                {
                    continue;
                }
                if (TryFoldStaticShapeNode(index, node, descriptors, immutableTensors, diagnostics))
                    continue;
                if (TryRejectKnownUnsupportedOperator(index, node, diagnostics))
                    continue;
                if (string.Equals(node.opType, "Einsum", StringComparison.Ordinal)
                    && !TryValidateDirectMatMulEinsum(node, descriptors, out var einsumReason))
                {
                    diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "unsupported-einsum-equation",
                        einsumReason,
                        "Rewrite this Einsum to Transpose/Reshape/MatMul, or use a two-input direct batched-matmul equation.", true));
                    continue;
                }

                var canonical = ResolveOperator(node.opType);
                if (string.IsNullOrEmpty(canonical) || !AexisLayerFactory.IsRegistered(canonical))
                {
                    diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "unsupported-operator",
                        "ONNX operator " + node.opType + " has no registered Aexis canonical mapping.",
                        "Implement a texture-native Aexis layer and capability profile, or lower this node before import.", true));
                    continue;
                }

                var nodeName = NodeName(node, index);
                var inputDescriptors = ResolveInputs(node, descriptors, index, diagnostics);
                ValidateInputBatchContracts(node, inputDescriptors, index, diagnostics);
                if (string.Equals(node.opType, "ExtractImagePatches", StringComparison.Ordinal)
                    && TryLowerExtractImagePatches(node, nodeName, inputDescriptors, index, diagnostics, out var patchLayers))
                {
                    InferOutputs(node, inputDescriptors, descriptors, immutableTensors, options, index, diagnostics);
                    result.graph.layers.AddRange(patchLayers);
                    continue;
                }
                if (string.Equals(node.opType, "MatMul", StringComparison.Ordinal)
                    && TryLowerConstantBMatMul(node, nodeName, inputDescriptors, immutableTensors, index, diagnostics, out var projectionLayers))
                {
                    InferOutputs(node, inputDescriptors, descriptors, immutableTensors, options, index, diagnostics);
                    result.graph.layers.AddRange(projectionLayers);
                    continue;
                }
                var layer = CreateLayer(canonical, nodeName, node.inputs.ToArray(), node.outputs.ToArray());
                CopyAttributes(node, layer);
                if (string.Equals(node.opType, "Einsum", StringComparison.Ordinal))
                    layer.stringParams["onnx.equation"] = GetString(node, "equation");
                if (BinaryOps.TryGetValue(node.opType, out var binaryOp))
                    layer.intParams[0] = binaryOp.ToString(CultureInfo.InvariantCulture);
                if (UnaryOps.TryGetValue(node.opType, out var unaryOp))
                    layer.intParams[0] = unaryOp.ToString(CultureInfo.InvariantCulture);
                if (string.Equals(node.opType, "MaxPool", StringComparison.Ordinal)) layer.intParams[0] = "0";
                if (string.Equals(node.opType, "AveragePool", StringComparison.Ordinal) || string.Equals(node.opType, "GlobalAveragePool", StringComparison.Ordinal)) layer.intParams[0] = "1";
                if (string.Equals(node.opType, "GlobalAveragePool", StringComparison.Ordinal) || string.Equals(node.opType, "GlobalMaxPool", StringComparison.Ordinal)) layer.intParams[4] = "1";
                if (string.Equals(node.opType, "ReduceSum", StringComparison.Ordinal)) layer.intParams[0] = "0";
                if (string.Equals(node.opType, "ReduceSumSquare", StringComparison.Ordinal)) layer.intParams[0] = "2";
                if (string.Equals(node.opType, "ReduceMean", StringComparison.Ordinal)) layer.intParams[0] = "3";
                if (string.Equals(node.opType, "ReduceMax", StringComparison.Ordinal)) layer.intParams[0] = "4";
                if (string.Equals(node.opType, "ReduceMin", StringComparison.Ordinal)) layer.intParams[0] = "5";
                if (string.Equals(node.opType, "ReduceProd", StringComparison.Ordinal)) layer.intParams[0] = "6";
                if (string.Equals(node.opType, "ReduceL1", StringComparison.Ordinal)) layer.intParams[0] = "7";
                if (string.Equals(node.opType, "ReduceL2", StringComparison.Ordinal)) layer.intParams[0] = "8";
                if (string.Equals(node.opType, "ReduceLogSum", StringComparison.Ordinal)) layer.intParams[0] = "9";
                if (string.Equals(node.opType, "ReduceLogSumExp", StringComparison.Ordinal)) layer.intParams[0] = "10";

                ConfigureCoreNcnnContract(node, layer, inputDescriptors, immutableTensors, diagnostics, index);
                ConfigureCanonicalContract(node, layer, inputDescriptors, immutableTensors, diagnostics, index, model.opset);
                if (!ConfigureBoundedDataIndexNode(node, layer, inputDescriptors, model.graph.nodes, options, index, diagnostics))
                    continue;
                InferOutputs(node, inputDescriptors, descriptors, immutableTensors, options, index, diagnostics);
                if (IsLegacyAxisActivation(node.opType, model.opset))
                {
                    if (TryRewriteLegacyAxisActivation(node, layer, inputDescriptors, index, diagnostics, out var rewrittenLayers))
                    {
                        for (var rewrittenIndex = 0; rewrittenIndex < rewrittenLayers.Length; rewrittenIndex++)
                            result.graph.layers.Add(rewrittenLayers[rewrittenIndex]);
                    }
                    continue;
                }
                result.graph.layers.Add(layer);
            }

            InsertRuntimeInitializerLayers(result.graph, immutableTensors, diagnostics);

            result.tensors = ToArray(descriptors);
            var initializerList = new List<OnnxTensor>();
            foreach (var pair in immutableTensors) initializerList.Add(pair.Value);
            result.initializers = initializerList.ToArray();
            result.diagnostics = diagnostics.ToArray();
            return result;
        }

        private static bool TryLowerExtractImagePatches(
            OnnxNode node,
            string nodeName,
            List<AexisOnnxTensorDescriptor> inputs,
            int index,
            List<AexisOnnxLoweringDiagnostic> diagnostics,
            out AexisGraphModel.Layer[] layers)
        {
            layers = Array.Empty<AexisGraphModel.Layer>();
            if (!string.Equals(node?.opType, "ExtractImagePatches", StringComparison.Ordinal))
                return false;
            if (node.inputs.Count != 1 || node.outputs.Count != 1 || string.IsNullOrWhiteSpace(node.outputs[0])
                || inputs.Count != 1 || inputs[0] == null)
            {
                diagnostics.Add(Diagnostic(index, nodeName, node.opType, "invalid-extract-image-patches-arity",
                    "ExtractImagePatches requires exactly one activation input and one named output.",
                    "Export the TensorFlow-style one-input ExtractImagePatches schema.", true));
                return true;
            }
            string reason = null;
            if (inputs[0].onnxDataType != 1 || !TryResolveExtractImagePatchesSpec(
                    node, inputs[0].shape, out var kernelH, out var kernelW,
                    out var strideH, out var strideW, out var rateH, out var rateW,
                    out var padTop, out var padBottom, out var padLeft, out var padRight,
                    out _, out reason))
            {
                diagnostics.Add(Diagnostic(index, nodeName, node.opType, "unsupported-extract-image-patches-profile",
                    reason ?? "ExtractImagePatches requires a static FP32 NHWC input.",
                    "Use static FP32 [1,H,W,C], positive NHWC ksizes/strides/rates, and SAME or VALID padding.", true));
                return true;
            }

            var prefix = "__aexis_extract_patches_" + index.ToString(CultureInfo.InvariantCulture) + "_" + SanitizeName(nodeName);
            var nativeInput = prefix + "_native_input";
            var nativeOutput = prefix + "_native_output";

            // Generic ONNX tensors use reversed NCNN axes. Convert [C,W,H] to
            // native Pack4 image [W,H,C], then restore [K,outW,outH].
            var toNative = CreateLayer("Permute", nodeName + "_nhwc_to_pack4", new[] { node.inputs[0] }, new[] { nativeInput });
            toNative.intParams[0] = "4"; // output axes [h,c,w]
            toNative.stringParams["aexis.synthetic"] = "extract_image_patches_nhwc_to_pack4";

            var extract = CreateLayer("ExtractPatches", nodeName, new[] { nativeInput }, new[] { nativeOutput });
            extract.intParams[1] = kernelW.ToString(CultureInfo.InvariantCulture);
            extract.intParams[11] = kernelH.ToString(CultureInfo.InvariantCulture);
            extract.intParams[2] = rateW.ToString(CultureInfo.InvariantCulture);
            extract.intParams[12] = rateH.ToString(CultureInfo.InvariantCulture);
            extract.intParams[3] = strideW.ToString(CultureInfo.InvariantCulture);
            extract.intParams[13] = strideH.ToString(CultureInfo.InvariantCulture);
            extract.intParams[4] = padLeft.ToString(CultureInfo.InvariantCulture);
            extract.intParams[14] = padTop.ToString(CultureInfo.InvariantCulture);
            extract.intParams[15] = padRight.ToString(CultureInfo.InvariantCulture);
            extract.intParams[16] = padBottom.ToString(CultureInfo.InvariantCulture);
            extract.intParams[18] = "0";
            extract.stringParams["onnx.original_op"] = "ExtractImagePatches";

            var fromNative = CreateLayer("Permute", nodeName + "_pack4_to_nhwc", new[] { nativeOutput }, new[] { node.outputs[0] });
            fromNative.intParams[0] = "3"; // output axes [c,w,h]
            fromNative.stringParams["aexis.synthetic"] = "extract_image_patches_pack4_to_nhwc";
            layers = new[] { toNative, extract, fromNative };
            return true;
        }

        private static bool TryLowerConstantBMatMul(
            OnnxNode node,
            string nodeName,
            List<AexisOnnxTensorDescriptor> inputs,
            Dictionary<string, OnnxTensor> initializers,
            int index,
            List<AexisOnnxLoweringDiagnostic> diagnostics,
            out AexisGraphModel.Layer[] layers)
        {
            layers = Array.Empty<AexisGraphModel.Layer>();
            if (node.inputs.Count != 2 || node.outputs.Count != 1 || string.IsNullOrEmpty(node.outputs[0])
                || !initializers.TryGetValue(node.inputs[1], out var weight))
                return false;

            if (inputs.Count < 2 || inputs[0] == null || inputs[0].isInitializer
                || inputs[0].shape == null || inputs[0].runtimeShape == null
                || HasDynamic(inputs[0].runtimeShape) || inputs[0].runtimeShape.Length < 1 || inputs[0].runtimeShape.Length > 4
                || weight == null || weight.dataType != TensorDataType.Float32 || weight.dims == null || weight.dims.Length != 2)
            {
                diagnostics.Add(Diagnostic(index, nodeName, node.opType, "unsupported-constant-b-matmul-profile",
                    "Constant-B MatMul requires a static FP32 rank-2 B and a static rank-1 through rank-4 runtime A texture.",
                    "Export B as FP32 [K,N] and keep A within the static rank-4 texture contract.", true));
                return true;
            }
            if (!weight.TryValidatePayload(out var payloadReason))
            {
                diagnostics.Add(Diagnostic(index, nodeName, node.opType, "invalid-constant-b-matmul-payload",
                    "Constant-B MatMul weight payload is invalid: " + payloadReason + ".",
                    "Provide the complete inline or external FP32 B tensor.", true));
                return true;
            }

            var runtimeInput = inputs[0].runtimeShape;
            var k = runtimeInput[runtimeInput.Length - 1];
            var n = weight.dims[1];
            if (k <= 0 || n <= 0 || weight.dims[0] != k || k > int.MaxValue || n > int.MaxValue)
            {
                diagnostics.Add(Diagnostic(index, nodeName, node.opType, "constant-b-matmul-shape-mismatch",
                    "MatMul A's final K dimension does not match immutable B[K,N].",
                    "Re-export matching static A and B dimensions.", true));
                return true;
            }

            long rows = 1;
            for (var axis = 0; axis < runtimeInput.Length - 1; axis++)
            {
                if (runtimeInput[axis] <= 0 || rows > int.MaxValue / runtimeInput[axis])
                {
                    diagnostics.Add(Diagnostic(index, nodeName, node.opType, "constant-b-matmul-row-overflow",
                        "Flattened MatMul row count exceeds the 32-bit texture descriptor range.",
                        "Split the projection into smaller static tensors.", true));
                    return true;
                }
                rows *= runtimeInput[axis];
            }
            if (rows <= 0 || rows > int.MaxValue)
            {
                diagnostics.Add(Diagnostic(index, nodeName, node.opType, "constant-b-matmul-row-overflow",
                    "Flattened MatMul row count is not representable.", "Use a positive static projection shape.", true));
                return true;
            }

            var outputRuntimeShape = Clone(runtimeInput);
            outputRuntimeShape[outputRuntimeShape.Length - 1] = n;
            var needsFlatten = runtimeInput.Length != 2;
            var runtimeInputName = node.inputs[0];
            var gemmOutputName = node.outputs[0];
            var rewritten = new List<AexisGraphModel.Layer>();
            if (needsFlatten)
            {
                runtimeInputName = node.outputs[0] + ".aexis_projection_flat";
                var flatten = CreateLayer("Reshape", nodeName + "_flatten", new[] { node.inputs[0] }, new[] { runtimeInputName });
                SetNcnnShapeParams(flatten, new[] { rows, k });
                flatten.stringParams["aexis.synthetic"] = "constant_b_matmul_flatten";
                rewritten.Add(flatten);
                gemmOutputName = node.outputs[0] + ".aexis_projection_2d";
            }

            var gemm = CreateLayer("Gemm", nodeName, new[] { runtimeInputName }, new[] { gemmOutputName });
            gemm.intParams[0] = "1";
            gemm.intParams[1] = "0";
            gemm.intParams[2] = "0";
            gemm.intParams[3] = "0";
            gemm.intParams[4] = "0";
            gemm.intParams[5] = "1";
            gemm.intParams[6] = "0";
            gemm.intParams[7] = rows.ToString(CultureInfo.InvariantCulture);
            gemm.intParams[8] = n.ToString(CultureInfo.InvariantCulture);
            gemm.intParams[9] = k.ToString(CultureInfo.InvariantCulture);
            gemm.intParams[10] = "-1";
            gemm.stringParams["onnx.b"] = node.inputs[1];
            gemm.stringParams["onnx.original_op"] = "MatMul";
            rewritten.Add(gemm);

            if (needsFlatten)
            {
                var restore = CreateLayer("Reshape", nodeName + "_restore", new[] { gemmOutputName }, new[] { node.outputs[0] });
                SetNcnnShapeParams(restore, outputRuntimeShape);
                restore.stringParams["aexis.synthetic"] = "constant_b_matmul_restore";
                rewritten.Add(restore);
            }
            layers = rewritten.ToArray();
            return true;
        }

        private static bool TryRejectKnownUnsupportedOperator(
            int index,
            OnnxNode node,
            List<AexisOnnxLoweringDiagnostic> diagnostics)
        {
            string code;
            string message;
            string action;
            switch (node.opType)
            {
                case "GridSample":
                    code = "missing-grid-sample-texture-kernel";
                    message = "GridSample requires coordinate-grid sampling with exact interpolation, padding, and align_corners semantics; no verified Aexis CommandBuffer Pack4 kernel currently implements that contract.";
                    action = "Lower to supported static Resize only when the grid is provably affine, otherwise add a texture-native GridSample kernel.";
                    break;
                case "NonMaxSuppression":
                    code = "missing-bounded-nms-profile";
                    message = "NonMaxSuppression has a data-dependent selected-index count and no capacity-bounded deterministic Aexis texture profile.";
                    action = "Use an exporter-proven bounded NMS contract with a GPU count tensor and a texture-native kernel; CPU readback is not accepted.";
                    break;
                case "RoiAlign":
                    code = "missing-roialign-texture-kernel";
                    message = "RoiAlign requires per-ROI indexed sampling and reduction; no verified Pack4 CommandBuffer implementation is registered.";
                    action = "Implement a texture-native RoiAlign kernel for the required mode/sampling_ratio profile before import.";
                    break;
                case "LSTM":
                    code = "missing-lstm-texture-profile";
                    message = "LSTM requires recurrent state, direction/layout handling, optional sequence lengths, and immutable gate weights; no complete Aexis texture-native state contract exists.";
                    action = "Decompose to verified primitive texture operators or implement a bounded CommandBuffer LSTM state profile.";
                    break;
                case "ImageScaler":
                    code = "imagescaler-rewrite-required";
                    message = "Legacy ImageScaler is not a runtime layer; its scale and per-channel bias must be decomposed before strict Aexis import.";
                    action = "Rewrite ImageScaler as Mul plus Add/Bias with static immutable constants.";
                    break;
                case "Bernoulli":
                case "Multinomial":
                case "RandomNormal":
                case "RandomNormalLike":
                case "RandomUniform":
                case "RandomUniformLike":
                    code = "missing-deterministic-random-texture-profile";
                    message = node.opType + " requires a reproducible GPU RNG/state contract which is not part of the P0 inference runtime.";
                    action = "Fold seeded random tensors offline or implement an explicit deterministic texture-native RNG profile.";
                    break;
                default:
                    return false;
            }

            diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, code, message, action, true));
            return true;
        }

        private static bool TryValidateDirectMatMulEinsum(
            OnnxNode node,
            Dictionary<string, AexisOnnxTensorDescriptor> descriptors,
            out string reason)
        {
            reason = null;
            if (node.inputs.Count != 2 || node.outputs.Count != 1)
            {
                reason = "The P0 Einsum lowering accepts exactly two operands and one output.";
                return false;
            }

            var equation = new string((GetString(node, "equation") ?? string.Empty).Where(character => !char.IsWhiteSpace(character)).ToArray());
            var arrow = equation.IndexOf("->", StringComparison.Ordinal);
            if (arrow <= 0 || equation.IndexOf("...", StringComparison.Ordinal) >= 0)
            {
                reason = "Einsum requires an explicit equation without ellipsis for direct MatMul lowering.";
                return false;
            }
            var operands = equation.Substring(0, arrow).Split(',');
            var output = equation.Substring(arrow + 2);
            if (operands.Length != 2
                || !descriptors.TryGetValue(node.inputs[0], out var left)
                || !descriptors.TryGetValue(node.inputs[1], out var right)
                || left.shape == null || right.shape == null
                || left.shape.Length < 2 || left.shape.Length > 4
                || right.shape.Length != left.shape.Length
                || operands[0].Length != left.shape.Length
                || operands[1].Length != right.shape.Length
                || output.Length != left.shape.Length)
            {
                reason = "Einsum operand labels must match two equal static ranks in the range 2..4.";
                return false;
            }

            var rank = left.shape.Length;
            var leading = rank - 2;
            if (operands[0].Distinct().Count() != operands[0].Length
                || operands[1].Distinct().Count() != operands[1].Length
                || output.Distinct().Count() != output.Length)
            {
                reason = "Repeated labels within an Einsum operand/output are not a direct MatMul profile.";
                return false;
            }
            for (var axis = 0; axis < leading; axis++)
            {
                if (operands[0][axis] != operands[1][axis]
                    || operands[0][axis] != output[axis]
                    || left.shape[axis] != right.shape[axis])
                {
                    reason = "Direct Einsum MatMul lowering requires identical, non-broadcast leading batch labels and extents.";
                    return false;
                }
            }
            if (operands[0][leading] != output[leading]
                || operands[0][rank - 1] != operands[1][leading]
                || operands[1][rank - 1] != output[rank - 1]
                || left.shape[rank - 1] != right.shape[leading])
            {
                reason = "Einsum labels/extents do not match [...,M,K] x [...,K,N] -> [...,M,N].";
                return false;
            }
            return true;
        }

        private static string ResolveOperator(string opType)
        {
            if (BinaryOps.ContainsKey(opType)) return "BinaryOp";
            if (UnaryOps.ContainsKey(opType)) return "UnaryOp";
            return OperatorMap.TryGetValue(opType, out var canonical) ? canonical : null;
        }

        private static AexisOnnxGraphLoweringOptions RequireDeclaredOutputs(AexisOnnxGraphLoweringOptions options)
        {
            options ??= new AexisOnnxGraphLoweringOptions();
            options.requireDeclaredGraphOutputs = true;
            return options;
        }

        private static Dictionary<string, OperatorSchema> CreateOperatorSchemas()
        {
            var schemas = new Dictionary<string, OperatorSchema>(StringComparer.Ordinal);
            void Add(OperatorSchema schema, params string[] operators)
            {
                for (var index = 0; index < operators.Length; index++)
                    schemas.Add(operators[index], schema);
            }

            Add(new OperatorSchema(1, 1, 1, 1),
                "Identity", "Relu", "Sigmoid", "Tanh", "Abs", "Exp", "Log", "Celu", "Elu", "Erf", "Gelu",
                "HardSigmoid", "HardSwish", "LeakyRelu", "Selu", "Softplus", "Softsign", "Shrink", "IsInf", "IsNaN",
                "Swish", "ThresholdedRelu", "LRN", "DepthToSpace", "SpaceToDepth", "GlobalAveragePool", "GlobalMaxPool",
                "Softmax", "LogSoftmax", "Hardmax", "Transpose", "Flatten", "Shape", "Size", "ArgMax", "ArgMin",
                "NonZero", "Neg", "Floor", "Ceil", "Sqrt", "Reciprocal", "Sin", "Cos", "Tan", "Asin", "Acos",
                "Atan", "Round", "Sign", "Sinh", "Asinh", "Cosh", "Acosh", "Atanh", "Not", "ExtractImagePatches");
            Add(new OperatorSchema(2, 2, 1, 1),
                "PRelu", "MatMul", "CastLike", "Expand", "CumSum", "Compress", "Gather", "GatherElements", "GatherND",
                "Add", "Sub", "Mul", "Div", "Max", "Min", "Pow", "Mod", "Equal", "Greater", "GreaterOrEqual",
                "Less", "LessOrEqual", "And", "Or", "Xor");
            Add(new OperatorSchema(3, 3, 1, 1), "Where", "OneHot", "Range", "Scatter", "ScatterElements", "ScatterND");
            Add(new OperatorSchema(0, 0, 1, 1), "Constant");
            Add(new OperatorSchema(1, 3, 1, 2), "Dropout");
            Add(new OperatorSchema(1, 3, 1, 1), "Clip", "Pad");
            Add(new OperatorSchema(1, 4, 1, 1), "Resize");
            Add(new OperatorSchema(2, 2, 1, 1), "Upsample", "Reshape", "Tile");
            Add(new OperatorSchema(2, 3, 1, 1), "Conv", "ConvTranspose", "Gemm");
            Add(new OperatorSchema(5, 5, 1, 3), "BatchNormalization");
            Add(new OperatorSchema(3, 3, 1, 1), "InstanceNormalization");
            Add(new OperatorSchema(2, 3, 1, 3, 17), "LayerNormalization");
            Add(new OperatorSchema(1, 1, 1, 2), "MaxPool");
            Add(new OperatorSchema(1, 1, 1, 1), "AveragePool");
            Add(new OperatorSchema(1, int.MaxValue, 1, 1), "Concat", "Sum", "Mean");
            Add(new OperatorSchema(1, 2, 1, int.MaxValue), "Split");
            Add(new OperatorSchema(1, 2, 1, 1), "Squeeze", "Unsqueeze", "TopK");
            Add(new OperatorSchema(1, 5, 1, 1), "Slice");
            Add(new OperatorSchema(1, 1, 1, 1), "Cast", "ConstantOfShape");
            Add(new OperatorSchema(1, 2, 1, 1),
                "ReduceSum", "ReduceSumSquare", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1",
                "ReduceL2", "ReduceLogSum", "ReduceLogSumExp");
            Add(new OperatorSchema(1, int.MaxValue, 1, 1, 12), "Einsum");
            Add(new OperatorSchema(1, 2, 1, 1, 14), "Trilu");
            return schemas;
        }

        private static int ResolveMinimumOpset(string opType, int schemaMinimum)
        {
            switch (opType)
            {
                case "Expand": return 8;
                case "ConstantOfShape":
                case "Compress":
                case "IsNaN":
                case "NonZero":
                case "OneHot":
                case "Where": return 9;
                case "IsInf":
                case "Resize": return 10;
                case "CumSum":
                case "GatherND":
                case "Range":
                case "Round":
                case "ScatterElements":
                case "ScatterND": return 11;
                case "Celu":
                case "GreaterOrEqual":
                case "LessOrEqual": return 12;
                case "HardSwish": return 14;
                case "CastLike": return 15;
                default: return schemaMinimum;
            }
        }

        private static void ValidateGraphContract(
            OnnxModel model,
            AexisOnnxGraphLoweringOptions options,
            List<AexisOnnxLoweringDiagnostic> diagnostics)
        {
            if (model.opset == 0)
            {
                diagnostics.Add(Diagnostic(-1, model.graph.name, "Graph", "missing-default-opset",
                    "The ONNX model does not declare an ai.onnx default-domain opset.",
                    "Export a model with an ai.onnx opset in the supported 7 through 19 range.", options.requireDeclaredGraphOutputs));
            }
            else if (model.opset < MinimumSupportedOpset || model.opset > MaximumSupportedOpset)
            {
                diagnostics.Add(Diagnostic(-1, model.graph.name, "Graph", "unsupported-opset",
                    "ONNX opset " + model.opset.ToString(CultureInfo.InvariantCulture) + " is outside the verified Aexis range 7 through 19.",
                    "Re-export with opset 7 through 19 or extend the schema, lowering, and GPU golden coverage before import.", true));
            }

            var available = new HashSet<string>(StringComparer.Ordinal);
            var inputNames = new HashSet<string>(StringComparer.Ordinal);
            for (var inputIndex = 0; inputIndex < model.graph.inputs.Count; inputIndex++)
            {
                var input = model.graph.inputs[inputIndex];
                var name = input?.name ?? string.Empty;
                if (string.IsNullOrWhiteSpace(name))
                {
                    diagnostics.Add(Diagnostic(-1, "input[" + inputIndex.ToString(CultureInfo.InvariantCulture) + "]", "Input", "unnamed-graph-input",
                        "Graph input names must be non-empty.", "Assign every graph input a unique tensor name.", true));
                    continue;
                }
                if (!inputNames.Add(name))
                    diagnostics.Add(Diagnostic(-1, name, "Input", "duplicate-graph-input",
                        "Graph input " + name + " is declared more than once.", "Keep exactly one declaration for each graph input.", true));
                available.Add(name);
            }
            foreach (var pair in model.graph.initializers)
            {
                if (string.IsNullOrWhiteSpace(pair.Key))
                {
                    diagnostics.Add(Diagnostic(-1, string.Empty, "Initializer", "unnamed-initializer",
                        "Initializer names must be non-empty.", "Assign every initializer a unique tensor name.", true));
                    continue;
                }
                available.Add(pair.Key);
            }

            var nodeNames = new HashSet<string>(StringComparer.Ordinal);
            var producerNames = new HashSet<string>(available, StringComparer.Ordinal);
            for (var nodeIndex = 0; nodeIndex < model.graph.nodes.Count; nodeIndex++)
            {
                var node = model.graph.nodes[nodeIndex];
                if (node == null)
                    continue;
                var nodeName = NodeName(node, nodeIndex);
                if (!string.IsNullOrWhiteSpace(node.name) && !nodeNames.Add(node.name))
                    diagnostics.Add(Diagnostic(nodeIndex, nodeName, node.opType, "duplicate-node-name",
                        "Node name " + node.name + " is not unique.", "Assign unique non-empty names to named ONNX nodes.", true));

                if (OperatorSchemas.TryGetValue(node.opType ?? string.Empty, out var schema))
                {
                    if (node.inputs.Count < schema.minInputs || node.inputs.Count > schema.maxInputs
                        || node.outputs.Count < schema.minOutputs || node.outputs.Count > schema.maxOutputs)
                    {
                        diagnostics.Add(Diagnostic(nodeIndex, nodeName, node.opType, "invalid-operator-arity",
                            node.opType + " has " + node.inputs.Count.ToString(CultureInfo.InvariantCulture) + " inputs and "
                            + node.outputs.Count.ToString(CultureInfo.InvariantCulture) + " outputs; the verified schema accepts inputs "
                            + FormatRange(schema.minInputs, schema.maxInputs) + " and outputs " + FormatRange(schema.minOutputs, schema.maxOutputs) + ".",
                            "Re-export this node with the canonical ONNX arity for its opset.", true));
                    }
                    var minimumOpset = ResolveMinimumOpset(node.opType, schema.minimumOpset);
                    if (model.opset > 0 && model.opset < minimumOpset)
                        diagnostics.Add(Diagnostic(nodeIndex, nodeName, node.opType, "operator-before-introduction-opset",
                            node.opType + " requires opset " + minimumOpset.ToString(CultureInfo.InvariantCulture)
                            + " or newer, but the model declares opset " + model.opset.ToString(CultureInfo.InvariantCulture) + ".",
                            "Use the operator schema matching the declared model opset.", true));
                    for (var required = 0; required < Math.Min(schema.minInputs, node.inputs.Count); required++)
                        if (string.IsNullOrEmpty(node.inputs[required]))
                            diagnostics.Add(Diagnostic(nodeIndex, nodeName, node.opType, "missing-required-input",
                                node.opType + " required input " + required.ToString(CultureInfo.InvariantCulture) + " is empty.",
                                "Connect every required operator input; empty names are only valid for optional positions.", true));
                    for (var required = 0; required < Math.Min(schema.minOutputs, node.outputs.Count); required++)
                        if (string.IsNullOrEmpty(node.outputs[required]))
                            diagnostics.Add(Diagnostic(nodeIndex, nodeName, node.opType, "missing-required-output",
                                node.opType + " required output " + required.ToString(CultureInfo.InvariantCulture) + " is empty.",
                                "Name every required operator output.", true));
                }

                for (var inputIndex = 0; inputIndex < node.inputs.Count; inputIndex++)
                {
                    var input = node.inputs[inputIndex];
                    if (!string.IsNullOrEmpty(input) && !available.Contains(input))
                        diagnostics.Add(Diagnostic(nodeIndex, nodeName, node.opType, "non-topological-input",
                            "Input " + input + " is not a graph source or the output of an earlier node.",
                            "Topologically sort the graph and ensure the tensor has exactly one producer.", true));
                }
                for (var outputIndex = 0; outputIndex < node.outputs.Count; outputIndex++)
                {
                    var output = node.outputs[outputIndex];
                    if (string.IsNullOrEmpty(output))
                        continue;
                    if (!producerNames.Add(output))
                        diagnostics.Add(Diagnostic(nodeIndex, nodeName, node.opType, "duplicate-tensor-producer",
                            "Tensor " + output + " already has a graph input, initializer, or earlier node producer.",
                            "Assign each non-empty tensor name exactly one producer.", true));
                    available.Add(output);
                }
            }

            if (model.graph.outputs.Count == 0)
            {
                diagnostics.Add(Diagnostic(-1, model.graph.name, "Graph", "missing-graph-output",
                    "The ONNX graph declares no outputs.", "Declare at least one named graph output produced by the graph.", options.requireDeclaredGraphOutputs));
                return;
            }

            var graphOutputs = new HashSet<string>(StringComparer.Ordinal);
            for (var outputIndex = 0; outputIndex < model.graph.outputs.Count; outputIndex++)
            {
                var output = model.graph.outputs[outputIndex];
                var name = output?.name ?? string.Empty;
                if (string.IsNullOrWhiteSpace(name))
                {
                    diagnostics.Add(Diagnostic(-1, "output[" + outputIndex.ToString(CultureInfo.InvariantCulture) + "]", "Output", "unnamed-graph-output",
                        "Graph output names must be non-empty.", "Assign every graph output a unique produced tensor name.", true));
                    continue;
                }
                if (!graphOutputs.Add(name))
                    diagnostics.Add(Diagnostic(-1, name, "Output", "duplicate-graph-output",
                        "Graph output " + name + " is declared more than once.", "Keep one declaration for each graph output.", true));
                if (!available.Contains(name))
                    diagnostics.Add(Diagnostic(-1, name, "Output", "unproduced-graph-output",
                        "Graph output " + name + " is not a graph input, initializer, or node output.",
                        "Connect the output to an existing tensor producer and preserve topological order.", true));
            }
        }

        private static string FormatRange(int minimum, int maximum)
        {
            return maximum == int.MaxValue
                ? minimum.ToString(CultureInfo.InvariantCulture) + " or more"
                : minimum == maximum
                    ? minimum.ToString(CultureInfo.InvariantCulture)
                    : minimum.ToString(CultureInfo.InvariantCulture) + " through " + maximum.ToString(CultureInfo.InvariantCulture);
        }

        private static AexisGraphModel.Layer CreateLayer(string typeName, string name, string[] bottoms, string[] tops)
        {
            var canonical = AexisLayerFactory.ResolveCanonicalLayerTypeName(typeName);
            return new AexisGraphModel.Layer
            {
                typeName = canonical,
                type = AexisLayerTypeKey.FromString(canonical),
                name = name,
                bottoms = bottoms?.Length ?? 0,
                tops = tops?.Length ?? 0,
                bottomNames = bottoms ?? Array.Empty<string>(),
                topNames = tops ?? Array.Empty<string>()
            };
        }

        private static void CopyAttributes(OnnxNode node, AexisGraphModel.Layer layer)
        {
            foreach (var pair in node.attributes)
            {
                var attribute = pair.Value;
                if (attribute == null) continue;
                if (attribute.ints.Count > 0)
                    layer.stringParams[pair.Key] = Join(attribute.ints);
                else if (attribute.floats.Count > 0)
                    layer.stringParams[pair.Key] = Join(attribute.floats);
                else if (attribute.s != null && attribute.s.Length > 0)
                    layer.stringParams[pair.Key] = attribute.GetUtf8String();
                else if (attribute.tensor != null)
                    layer.stringParams[pair.Key] = attribute.tensor.name ?? string.Empty;
                else if (attribute.type == 1)
                    layer.stringParams[pair.Key] = attribute.f.ToString("R", CultureInfo.InvariantCulture);
                else
                    layer.stringParams[pair.Key] = attribute.i.ToString(CultureInfo.InvariantCulture);
            }
        }

        private static void ConfigureCoreNcnnContract(
            OnnxNode node,
            AexisGraphModel.Layer layer,
            List<AexisOnnxTensorDescriptor> inputs,
            Dictionary<string, OnnxTensor> initializers,
            List<AexisOnnxLoweringDiagnostic> diagnostics,
            int index)
        {
            if (string.Equals(node.opType, "Conv", StringComparison.Ordinal)
                || string.Equals(node.opType, "ConvTranspose", StringComparison.Ordinal))
            {
                if (node.inputs.Count < 2 || !initializers.TryGetValue(node.inputs[1], out var weight) || weight.dims == null || weight.dims.Length != 4)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "missing-static-convolution-weight",
                        node.opType + " requires a static rank-4 initializer for texture-native OIHW packing.",
                        "Bind a rank-4 FP32 initializer or lower this node before strict import.", true));
                    return;
                }
                if (weight.dataType != TensorDataType.Float32)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-convolution-weight-dtype",
                        "Texture-native convolution import currently requires FP32 immutable weights.",
                        "Convert the ONNX convolution initializer to FP32 before import.", true));
                    return;
                }

                var kernel = GetInts(node, "kernel_shape", new[] { weight.dims[2], weight.dims[3] });
                var strides = GetInts(node, "strides", new long[] { 1, 1 });
                var dilations = GetInts(node, "dilations", new long[] { 1, 1 });
                var inputShape = inputs.Count > 0 ? inputs[0].shape : null;
                var transpose = string.Equals(node.opType, "ConvTranspose", StringComparison.Ordinal);
                if (kernel.Length != 2 || kernel[0] != weight.dims[2] || kernel[1] != weight.dims[3])
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "convolution-kernel-weight-mismatch",
                        "kernel_shape must exactly match the immutable convolution weight spatial dimensions.",
                        "Correct kernel_shape or re-export the weight tensor.", true));
                    return;
                }
                if (transpose && node.attributes.TryGetValue("output_shape", out var outputShapeAttribute)
                    && outputShapeAttribute != null && outputShapeAttribute.ints.Count > 0)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-convtranspose-output-shape",
                        "ConvTranspose output_shape requires output cropping that is not part of the strict immutable Pack4 profile.",
                        "Express the result with explicit pads/output_padding or add a following static Slice.", true));
                    return;
                }
                if (!TryResolveAutoPads2D(node, inputShape, kernel, strides, dilations, transpose, out var pads, out var padReason))
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-convolution-padding", padReason,
                        "Use explicit pads, VALID, or SAME_UPPER/SAME_LOWER on a static Conv input.", true));
                    return;
                }
                if (kernel.Length != 2 || strides.Length != 2 || dilations.Length != 2 || pads.Length != 4
                    || kernel[0] <= 0 || kernel[1] <= 0 || strides[0] <= 0 || strides[1] <= 0
                    || dilations[0] <= 0 || dilations[1] <= 0 || pads.Any(value => value < 0))
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-convolution-attributes",
                        "Conv/ConvTranspose requires positive 2D kernel/stride/dilation values and four non-negative explicit pads.",
                        "Re-export a static 2D NCHW convolution.", true));
                    return;
                }
                var group = GetInt(node, "group", 1);
                if (group <= 0)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-convolution-group",
                        "Convolution group must be positive.", "Export a positive group value.", true));
                    return;
                }
                var outputChannels = node.opType == "Conv" ? weight.dims[0] : weight.dims[1] * group;
                if (outputChannels <= 0 || outputChannels > int.MaxValue)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-convolution-output-channels", "The convolution output channel count is invalid.", "Use a positive static channel count.", true));
                    return;
                }
                if (inputShape != null && inputShape.Length == 4 && inputShape[1] > 0)
                {
                    var expectedInputChannels = transpose ? weight.dims[0] : weight.dims[1] * group;
                    if (inputShape[1] != expectedInputChannels || outputChannels % group != 0)
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "convolution-channel-weight-mismatch",
                            "Input/output channels, group, and immutable weight dimensions are inconsistent.",
                            "Re-export a valid grouped convolution weight tensor.", true));
                        return;
                    }
                }
                layer.intParams[0] = outputChannels.ToString(CultureInfo.InvariantCulture);
                layer.intParams[1] = kernel[1].ToString(CultureInfo.InvariantCulture);
                layer.intParams[11] = kernel[0].ToString(CultureInfo.InvariantCulture);
                layer.intParams[2] = dilations[1].ToString(CultureInfo.InvariantCulture);
                layer.intParams[12] = dilations[0].ToString(CultureInfo.InvariantCulture);
                layer.intParams[3] = strides[1].ToString(CultureInfo.InvariantCulture);
                layer.intParams[13] = strides[0].ToString(CultureInfo.InvariantCulture);
                layer.intParams[4] = pads[1].ToString(CultureInfo.InvariantCulture);
                layer.intParams[15] = pads[3].ToString(CultureInfo.InvariantCulture);
                layer.intParams[14] = pads[0].ToString(CultureInfo.InvariantCulture);
                layer.intParams[16] = pads[2].ToString(CultureInfo.InvariantCulture);
                var hasBias = node.inputs.Count > 2 && !string.IsNullOrEmpty(node.inputs[2]);
                if (hasBias && (!initializers.TryGetValue(node.inputs[2], out var bias)
                    || bias.dataType != TensorDataType.Float32 || bias.dims == null || bias.dims.Length != 1 || bias.dims[0] != outputChannels))
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-convolution-bias",
                        "Convolution bias must be a static rank-1 FP32 tensor matching num_output.",
                        "Bind a valid immutable bias initializer or remove the optional bias input.", true));
                    return;
                }
                layer.intParams[5] = (hasBias ? 1 : 0).ToString(CultureInfo.InvariantCulture);
                layer.intParams[6] = ElementCount(weight.dims).ToString(CultureInfo.InvariantCulture);
                layer.intParams[7] = group.ToString(CultureInfo.InvariantCulture);
                if (string.Equals(node.opType, "ConvTranspose", StringComparison.Ordinal))
                {
                    var outputPadding = GetInts(node, "output_padding", new long[] { 0, 0 });
                    if (outputPadding.Length != 2 || outputPadding[0] < 0 || outputPadding[1] < 0)
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-output-padding",
                            "ConvTranspose output_padding must contain two non-negative values.",
                            "Use a valid static 2D output_padding attribute.", true));
                        return;
                    }
                    layer.intParams[18] = outputPadding[1].ToString(CultureInfo.InvariantCulture);
                    layer.intParams[19] = outputPadding[0].ToString(CultureInfo.InvariantCulture);
                    if (group != 1)
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-grouped-convtranspose",
                            "The strict Aexis deconvolution weight packer currently supports group=1 only.",
                            "Rewrite grouped ConvTranspose as independent group=1 nodes before import.", true));
                }
                layer.stringParams["onnx.weight"] = node.inputs[1];
                if (layer.GetInt(5, 0) != 0) layer.stringParams["onnx.bias"] = node.inputs[2];
                layer.bottomNames = node.inputs.Count > 0 ? new[] { node.inputs[0] } : Array.Empty<string>();
                layer.bottoms = layer.bottomNames.Length;
                return;
            }

            if (string.Equals(node.opType, "BatchNormalization", StringComparison.Ordinal))
            {
                if (node.inputs.Count < 5 || !initializers.TryGetValue(node.inputs[1], out var scale) || scale.dims == null || scale.dims.Length != 1)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "missing-static-batchnorm-weights",
                        "BatchNormalization requires static rank-1 scale, bias, mean, and variance initializers.",
                        "Bind the four immutable channel tensors before strict import.", true));
                    return;
                }
                var channels = scale.dims[0];
                if (channels <= 0 || GetFloat(node, "epsilon", 1e-5f) < 0f)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-batchnorm-parameters",
                        "BatchNormalization requires a positive channel count and non-negative epsilon.",
                        "Export valid inference BatchNormalization parameters.", true));
                    return;
                }
                for (var inputIndex = 1; inputIndex <= 4; inputIndex++)
                {
                    if (!initializers.TryGetValue(node.inputs[inputIndex], out var tensor)
                        || tensor.dataType != TensorDataType.Float32 || tensor.dims == null || tensor.dims.Length != 1 || tensor.dims[0] != channels)
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-batchnorm-weight",
                            "BatchNormalization scale, bias, mean, and variance must be matching rank-1 FP32 tensors.",
                            "Export four same-sized immutable FP32 channel tensors.", true));
                        return;
                    }
                }
                layer.intParams[0] = scale.dims[0].ToString(CultureInfo.InvariantCulture);
                layer.intParams[1] = GetFloat(node, "epsilon", 1e-5f).ToString("R", CultureInfo.InvariantCulture);
                layer.stringParams["onnx.scale"] = node.inputs[1];
                layer.stringParams["onnx.bias"] = node.inputs[2];
                layer.stringParams["onnx.mean"] = node.inputs[3];
                layer.stringParams["onnx.variance"] = node.inputs[4];
                layer.bottomNames = node.inputs.Count > 0 ? new[] { node.inputs[0] } : Array.Empty<string>();
                layer.bottoms = layer.bottomNames.Length;
                return;
            }

            if (string.Equals(node.opType, "Gemm", StringComparison.Ordinal))
            {
                var transA = GetInt(node, "transA", 0);
                var transB = GetInt(node, "transB", 0);
                layer.intParams[0] = GetFloat(node, "alpha", 1f).ToString("R", CultureInfo.InvariantCulture);
                layer.intParams[1] = GetFloat(node, "beta", 1f).ToString("R", CultureInfo.InvariantCulture);
                layer.intParams[2] = transA.ToString(CultureInfo.InvariantCulture);
                layer.intParams[3] = transB.ToString(CultureInfo.InvariantCulture);
                layer.intParams[4] = "0";
                OnnxTensor b = null;
                var hasB = node.inputs.Count > 1 && initializers.TryGetValue(node.inputs[1], out b) && b.dims != null && b.dims.Length == 2;
                var hasC = node.inputs.Count > 2 && initializers.ContainsKey(node.inputs[2]);
                layer.intParams[5] = hasB ? "1" : "0";
                layer.intParams[6] = hasC ? "1" : "0";
                if (!hasB || b.dataType != TensorDataType.Float32)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "missing-static-gemm-weight",
                        "Gemm requires a static rank-2 FP32 B initializer for the texture-native Aexis kernel.",
                        "Fold B into an immutable FP32 initializer before import.", true));
                    return;
                }
                if (hasB)
                {
                    var k = transB != 0 ? b.dims[1] : b.dims[0];
                    var n = transB != 0 ? b.dims[0] : b.dims[1];
                    if (k <= 0 || n <= 0 || k > int.MaxValue || n > int.MaxValue)
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-gemm-weight-shape", "Gemm constant B has an invalid static shape.", "Use a positive rank-2 FP32 initializer.", true));
                        return;
                    }
                    layer.intParams[8] = n.ToString(CultureInfo.InvariantCulture);
                    layer.intParams[9] = k.ToString(CultureInfo.InvariantCulture);
                    var m = inputs.Count > 0 && inputs[0].shape != null && inputs[0].shape.Length == 2
                        ? inputs[0].shape[transA != 0 ? 1 : 0]
                        : 0;
                    if (m > 0 && m <= int.MaxValue) layer.intParams[7] = m.ToString(CultureInfo.InvariantCulture);
                    if (node.inputs.Count > 0 && initializers.TryGetValue(node.inputs[0], out var constantA) && constantA != null)
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-constant-gemm-a",
                            "Gemm constant A is not supported by the production texture path.",
                            "Keep A as a runtime texture input.", true));
                        return;
                    }
                    layer.stringParams["onnx.b"] = node.inputs[1];
                    if (hasC)
                    {
                        var c = initializers[node.inputs[2]];
                        if (c.dataType != TensorDataType.Float32)
                        {
                            diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-gemm-bias-dtype",
                                "Gemm C must be an FP32 immutable tensor.", "Convert C to FP32 before import.", true));
                            return;
                        }
                        layer.stringParams["onnx.c"] = node.inputs[2];
                        layer.intParams[10] = ResolveGemmBroadcastType(c.dims, m, n).ToString(CultureInfo.InvariantCulture);
                        if (layer.GetInt(10, -2) == -2)
                        {
                            diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-gemm-bias-shape",
                                "Gemm C is not a scalar, N-vector, or supported matrix broadcast.",
                                "Use scalar C, shape [N], [1,N], or a statically compatible matrix.", true));
                            return;
                        }
                    }
                    else layer.intParams[10] = "-1";
                    var runtimeInputs = new List<string> { node.inputs[0] };
                    layer.bottomNames = runtimeInputs.ToArray();
                    layer.bottoms = layer.bottomNames.Length;
                }
            }
        }

        private static void ConfigureCanonicalContract(
            OnnxNode node,
            AexisGraphModel.Layer layer,
            List<AexisOnnxTensorDescriptor> inputs,
            Dictionary<string, OnnxTensor> initializers,
            List<AexisOnnxLoweringDiagnostic> diagnostics,
            int index,
            int opset)
        {
            if (string.Equals(node.opType, "Mod", StringComparison.Ordinal))
            {
                layer.intParams[0] = GetInt(node, "fmod", 0) != 0 ? "12" : "19";
            }

            if (string.Equals(node.opType, "Not", StringComparison.Ordinal))
            {
                layer.intParams[0] = "28";
            }

            if (string.Equals(node.opType, "Trilu", StringComparison.Ordinal))
            {
                var upper = GetInt(node, "upper", 1);
                if (upper != 0 && upper != 1)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-trilu-upper",
                        "Trilu upper must be 0 or 1.", "Export a valid static Trilu upper attribute.", true));
                }

                var rank = inputs.Count > 0 && inputs[0]?.shape != null ? inputs[0].shape.Length : 0;
                if (rank < 2 || rank > 4)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-trilu-rank",
                        "Trilu requires a static input rank from 2 through 4 and masks the final two axes.",
                        "Reshape the input to rank 2 through 4 before Trilu.", true));
                }

                var diagonal = 0L;
                if (node.inputs.Count > 2)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-trilu-input-count",
                        "Trilu accepts the data tensor and one optional scalar k input.",
                        "Remove extra Trilu inputs.", true));
                }
                if (node.inputs.Count > 1 && !string.IsNullOrEmpty(node.inputs[1]))
                {
                    if (!TryGetInitializerInput(node, 1, initializers, out var diagonalTensor)
                        || (diagonalTensor.onnxDataType != 6 && diagonalTensor.onnxDataType != 7)
                        || !TryGetIntValues(diagonalTensor, out var diagonalValues)
                        || diagonalValues.Length != 1
                        || diagonalValues[0] < int.MinValue
                        || diagonalValues[0] > int.MaxValue)
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "non-static-trilu-diagonal",
                            "Trilu k must be a static scalar INT32 or INT64 initializer in the Int32 range.",
                            "Fold the diagonal offset into an immutable scalar initializer.", true));
                    }
                    else
                    {
                        diagonal = diagonalValues[0];
                    }
                }

                layer.intParams[0] = upper.ToString(CultureInfo.InvariantCulture);
                layer.intParams[1] = diagonal.ToString(CultureInfo.InvariantCulture);
                layer.stringParams["onnx.upper"] = upper.ToString(CultureInfo.InvariantCulture);
                layer.stringParams["onnx.k"] = diagonal.ToString(CultureInfo.InvariantCulture);
                KeepOnlyFirstBottom(layer);
            }

            if (string.Equals(node.opType, "Sum", StringComparison.Ordinal)
                || string.Equals(node.opType, "Mean", StringComparison.Ordinal))
            {
                var valid = inputs.Count > 0 && inputs[0] != null && !HasDynamic(inputs[0].shape);
                for (var i = 1; i < inputs.Count; i++)
                    valid &= inputs[i] != null && ShapesEqual(inputs[0].shape, inputs[i].shape);
                if (!valid)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-eltwise-broadcast",
                        node.opType + " requires two or more equal, static shapes in the Pack4 Eltwise profile.",
                        "Broadcast inputs explicitly to the same static shape before import.", true));
                }
                layer.intParams[0] = "1";
                if (string.Equals(node.opType, "Mean", StringComparison.Ordinal) && inputs.Count > 0)
                {
                    var coefficients = new float[inputs.Count];
                    for (var i = 0; i < coefficients.Length; i++) coefficients[i] = 1f / coefficients.Length;
                    layer.intParams[-23301] = EncodeFloatArray(coefficients);
                }
            }

            if (BinaryOps.ContainsKey(node.opType) && node.inputs.Count != 2)
            {
                diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-variadic-binary",
                    node.opType + " requires exactly two inputs in the BinaryOp Pack4 profile.",
                    "Lower variadic operators to a chain of binary nodes before import.", true));
            }

            if (string.Equals(node.opType, "Shrink", StringComparison.Ordinal))
            {
                layer.intParams[0] = GetFloat(node, "bias", 0f).ToString("R", CultureInfo.InvariantCulture);
                layer.intParams[1] = GetFloat(node, "lambd", 0.5f).ToString("R", CultureInfo.InvariantCulture);
            }

            if (string.Equals(node.opType, "IsInf", StringComparison.Ordinal))
            {
                layer.intParams[0] = GetInt(node, "detect_negative", 1).ToString(CultureInfo.InvariantCulture);
                layer.intParams[1] = GetInt(node, "detect_positive", 1).ToString(CultureInfo.InvariantCulture);
            }

            if (string.Equals(node.opType, "Dropout", StringComparison.Ordinal))
            {
                var training = false;
                if (node.inputs.Count > 2 && !string.IsNullOrEmpty(node.inputs[2]))
                {
                    if (!TryGetInitializerInput(node, 2, initializers, out var trainingTensor)
                        || !TryGetIntValues(trainingTensor, out var trainingValues) || trainingValues.Length != 1)
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "dynamic-dropout-training-mode",
                            "Dropout training_mode must be a static false scalar for inference lowering.",
                            "Fold training_mode=false and remove the mask output before import.", true));
                    }
                    else
                    {
                        training = trainingValues[0] != 0;
                    }
                }
                if (training || node.outputs.Count > 1)
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-training-dropout",
                        "Dropout only supports inference mode with one data output.",
                        "Fold inference Dropout to Identity and remove the training mask output.", true));
                KeepOnlyFirstBottom(layer);
            }

            if (string.Equals(node.opType, "BatchNormalization", StringComparison.Ordinal))
            {
                var training = false;
                if (node.inputs.Count > 5 && !string.IsNullOrEmpty(node.inputs[5]))
                {
                    if (!TryGetInitializerInput(node, 5, initializers, out var trainingTensor)
                        || !TryGetIntValues(trainingTensor, out var trainingValues) || trainingValues.Length != 1)
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "dynamic-batchnorm-training-mode",
                            "BatchNormalization training_mode must be a static false scalar.",
                            "Export BatchNormalization in inference mode.", true));
                    }
                    else
                    {
                        training = trainingValues[0] != 0;
                    }
                }
                if (training || node.outputs.Count > 1)
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-training-batchnorm",
                        "BatchNormalization training mode or running-statistics outputs are outside the inference Pack4 profile.",
                        "Export BatchNormalization in inference mode with one output.", true));
            }

            if (string.Equals(node.opType, "MaxPool", StringComparison.Ordinal) && node.outputs.Count > 1)
                diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-maxpool-indices",
                    "MaxPool indices output is not produced by the Pooling Pack4 kernel.",
                    "Remove the unused indices output or implement a paired MaxPoolingInd texture profile.", true));

            if (string.Equals(node.opType, "Celu", StringComparison.Ordinal)
                || string.Equals(node.opType, "Elu", StringComparison.Ordinal))
            {
                layer.intParams[0] = GetFloat(node, "alpha", 1f).ToString("R", CultureInfo.InvariantCulture);
            }

            if (string.Equals(node.opType, "LeakyRelu", StringComparison.Ordinal))
                layer.intParams[0] = GetFloat(node, "alpha", 0.01f).ToString("R", CultureInfo.InvariantCulture);

            if (string.Equals(node.opType, "HardSigmoid", StringComparison.Ordinal))
            {
                layer.intParams[0] = GetFloat(node, "alpha", 0.2f).ToString("R", CultureInfo.InvariantCulture);
                layer.intParams[1] = GetFloat(node, "beta", 0.5f).ToString("R", CultureInfo.InvariantCulture);
            }

            if (string.Equals(node.opType, "Selu", StringComparison.Ordinal))
            {
                layer.intParams[0] = GetFloat(node, "alpha", 1.67326319f).ToString("R", CultureInfo.InvariantCulture);
                layer.intParams[1] = GetFloat(node, "gamma", 1.05070102f).ToString("R", CultureInfo.InvariantCulture);
            }

            if (string.Equals(node.opType, "ThresholdedRelu", StringComparison.Ordinal))
                layer.intParams[0] = GetFloat(node, "alpha", 1f).ToString("R", CultureInfo.InvariantCulture);

            if (string.Equals(node.opType, "PRelu", StringComparison.Ordinal))
            {
                if (!TryGetInitializerInput(node, 1, initializers, out var slope)
                    || slope.dims == null || slope.ElementCount <= 0
                    || slope.ElementCount > int.MaxValue || slope.dataType != TensorDataType.Float32
                    || !IsSupportedPReluSlope(inputs.Count > 0 ? inputs[0] : null, slope))
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "missing-static-prelu-slope",
                        "PRelu requires an immutable FP32 scalar slope or an NCHW channel-broadcast slope shaped [C,1,1] or [1,C,1,1].",
                        "Fold a scalar or exact channel-broadcast PRelu slope into an immutable FP32 initializer before import.", true));
                }
                else
                {
                    layer.intParams[0] = slope.ElementCount.ToString(CultureInfo.InvariantCulture);
                    layer.stringParams["onnx.slope"] = node.inputs[1];
                    KeepOnlyFirstBottom(layer);
                }
            }

            if (string.Equals(node.opType, "Clip", StringComparison.Ordinal))
            {
                var minimum = GetFloat(node, "min", float.NegativeInfinity);
                var maximum = GetFloat(node, "max", float.PositiveInfinity);
                if (node.inputs.Count > 1 && !string.IsNullOrEmpty(node.inputs[1]))
                {
                    if (!TryGetInitializerInput(node, 1, initializers, out var minTensor) || !TryGetScalarFloat(minTensor, out minimum))
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "non-static-clip-bound",
                            "Clip min must be a static scalar initializer.", "Fold the Clip bound before import.", true));
                }
                if (node.inputs.Count > 2 && !string.IsNullOrEmpty(node.inputs[2]))
                {
                    if (!TryGetInitializerInput(node, 2, initializers, out var maxTensor) || !TryGetScalarFloat(maxTensor, out maximum))
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "non-static-clip-bound",
                            "Clip max must be a static scalar initializer.", "Fold the Clip bound before import.", true));
                }
                if (minimum > maximum)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-clip-range",
                        "Clip minimum exceeds maximum.", "Export a valid static Clip range.", true));
                }
                else
                {
                    layer.intParams[0] = minimum.ToString("R", CultureInfo.InvariantCulture);
                    layer.intParams[1] = maximum.ToString("R", CultureInfo.InvariantCulture);
                    KeepOnlyFirstBottom(layer);
                }
            }

            if (string.Equals(node.opType, "LRN", StringComparison.Ordinal))
            {
                var size = GetInt(node, "size", 0);
                if (size <= 0 || (size & 1) == 0)
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-lrn-size",
                        "LRN size must be a positive odd integer.", "Export a valid across-channel ONNX LRN.", true));
                layer.intParams[0] = "0";
                layer.intParams[1] = size.ToString(CultureInfo.InvariantCulture);
                layer.intParams[2] = GetFloat(node, "alpha", 0.0001f).ToString("R", CultureInfo.InvariantCulture);
                layer.intParams[3] = GetFloat(node, "beta", 0.75f).ToString("R", CultureInfo.InvariantCulture);
                layer.intParams[4] = GetFloat(node, "bias", 1f).ToString("R", CultureInfo.InvariantCulture);
            }

            if (string.Equals(node.opType, "DepthToSpace", StringComparison.Ordinal))
            {
                var blockSize = GetInt(node, "blocksize", 0);
                var mode = GetString(node, "mode");
                if (string.IsNullOrEmpty(mode)) mode = "DCR";
                if (blockSize <= 0 || (!string.Equals(mode, "DCR", StringComparison.Ordinal) && !string.Equals(mode, "CRD", StringComparison.Ordinal)))
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-depth-to-space",
                        "DepthToSpace requires blocksize>0 and mode DCR or CRD.", "Export a valid static DepthToSpace node.", true));
                layer.intParams[0] = blockSize.ToString(CultureInfo.InvariantCulture);
                layer.intParams[1] = string.Equals(mode, "DCR", StringComparison.Ordinal) ? "1" : "0";
            }

            if (string.Equals(node.opType, "SpaceToDepth", StringComparison.Ordinal))
            {
                var blockSize = GetInt(node, "blocksize", 0);
                if (blockSize != 2)
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-space-to-depth-blocksize",
                        "The strict Pack4 Reorg kernel supports ONNX SpaceToDepth blocksize=2.",
                        "Use blocksize=2 or implement the general Pack4 reorg kernel.", true));
                layer.intParams[0] = blockSize.ToString(CultureInfo.InvariantCulture);
                layer.intParams[1] = "0";
            }

            if (string.Equals(node.opType, "Pad", StringComparison.Ordinal))
                ConfigurePadContract(node, layer, inputs, initializers, diagnostics, index);

            if (string.Equals(node.opType, "Resize", StringComparison.Ordinal)
                || string.Equals(node.opType, "Upsample", StringComparison.Ordinal))
                ConfigureResizeContract(node, layer, inputs, initializers, diagnostics, index);

            if (string.Equals(node.opType, "MaxPool", StringComparison.Ordinal)
                || string.Equals(node.opType, "AveragePool", StringComparison.Ordinal))
            {
                var kernel = GetInts(node, "kernel_shape", Array.Empty<long>());
                var strides = GetInts(node, "strides", new long[] { 1, 1 });
                var dilations = GetInts(node, "dilations", new long[] { 1, 1 });
                var inputShape = inputs.Count > 0 ? inputs[0].shape : null;
                var resolvedPadding = TryResolveAutoPads2D(node, inputShape, kernel, strides, dilations, false, out var pads, out var reason);
                if (kernel.Length != 2 || strides.Length != 2 || pads.Length != 4 || dilations.Length != 2
                    || GetInt(node, "ceil_mode", 0) != 0 || dilations[0] != 1 || dilations[1] != 1
                    || kernel.Any(value => value <= 0) || strides.Any(value => value <= 0) || pads.Any(value => value < 0)
                    || !resolvedPadding)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-pooling-attributes",
                        reason ?? "Pooling requires static 2D kernel/stride/pads, dilation=1, and ceil_mode=0.",
                        "Use an explicit ONNX floor-mode 2D pooling profile.", true));
                    return;
                }
                layer.intParams[1] = kernel[1].ToString(CultureInfo.InvariantCulture);
                layer.intParams[11] = kernel[0].ToString(CultureInfo.InvariantCulture);
                layer.intParams[2] = strides[1].ToString(CultureInfo.InvariantCulture);
                layer.intParams[12] = strides[0].ToString(CultureInfo.InvariantCulture);
                layer.intParams[3] = pads[1].ToString(CultureInfo.InvariantCulture);
                layer.intParams[14] = pads[3].ToString(CultureInfo.InvariantCulture);
                layer.intParams[13] = pads[0].ToString(CultureInfo.InvariantCulture);
                layer.intParams[15] = pads[2].ToString(CultureInfo.InvariantCulture);
                // auto_pad has already been resolved to explicit ONNX pads. ncnn
                // pad_mode=1 preserves floor-mode output instead of adding tail padding.
                layer.intParams[5] = "1";
                var countIncludePad = GetInt(node, "count_include_pad", 0);
                if (countIncludePad != 0 && countIncludePad != 1)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-count-include-pad",
                        "AveragePool count_include_pad must be 0 or 1.", "Export a valid boolean count_include_pad attribute.", true));
                }
                layer.intParams[6] = countIncludePad.ToString(CultureInfo.InvariantCulture);
            }

            if (string.Equals(node.opType, "Concat", StringComparison.Ordinal)
                || string.Equals(node.opType, "Softmax", StringComparison.Ordinal)
                || string.Equals(node.opType, "LogSoftmax", StringComparison.Ordinal)
                || string.Equals(node.opType, "Hardmax", StringComparison.Ordinal)
                || string.Equals(node.opType, "Gather", StringComparison.Ordinal)
                || string.Equals(node.opType, "GatherElements", StringComparison.Ordinal)
                || string.Equals(node.opType, "Scatter", StringComparison.Ordinal)
                || string.Equals(node.opType, "ScatterElements", StringComparison.Ordinal))
            {
                var descriptor = inputs.Count > 0 ? inputs[0] : null;
                var rank = descriptor?.shape.Length ?? 0;
                var isAxisActivation = string.Equals(node.opType, "Softmax", StringComparison.Ordinal)
                    || string.Equals(node.opType, "LogSoftmax", StringComparison.Ordinal)
                    || string.Equals(node.opType, "Hardmax", StringComparison.Ordinal);
                var axis = GetInt(node, "axis", isAxisActivation ? ResolveSoftmaxDefaultAxis(opset) : 0);
                if (isAxisActivation && opset > 0 && opset < 13)
                {
                    var normalizedAxis = NormalizeAxis(axis, rank);
                    if (descriptor == null || normalizedAxis < 0 || normalizedAxis >= rank)
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-axis",
                            node.opType + " axis is outside the static input rank.",
                            "Use an axis in [-rank, rank-1].", true));
                    }
                    else
                    {
                        layer.stringParams["onnx.legacy_axis"] = normalizedAxis.ToString(CultureInfo.InvariantCulture);
                    }
                }
                else
                {
                    if (!TryTranslateRuntimeAxis(descriptor, axis, out var runtimeAxis))
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-batch-axis",
                            node.opType + " cannot operate on the ONNX batch axis after the static batch=1 axis is removed.",
                            "Move this operation to a non-batch axis or implement a batched texture profile.", true));
                    }
                    else
                    {
                        layer.intParams[0] = runtimeAxis.ToString(CultureInfo.InvariantCulture);
                        layer.stringParams["axis"] = layer.intParams[0];
                    }
                }
                if (string.Equals(node.opType, "LogSoftmax", StringComparison.Ordinal)) layer.intParams[10] = "1";
                if (string.Equals(node.opType, "Hardmax", StringComparison.Ordinal)) layer.intParams[10] = "2";
            }

            if (string.Equals(node.opType, "Transpose", StringComparison.Ordinal))
            {
                var descriptor = inputs.Count > 0 ? inputs[0] : null;
                var rank = descriptor?.shape.Length ?? 0;
                var perm = GetInts(node, "perm", ReverseAxes(rank));
                var runtimePerm = RemoveBatchFromPermutation(perm, descriptor?.batchAxis ?? -1);
                var runtimeRank = descriptor?.runtimeShape?.Length ?? rank;
                if (!TryGetNcnnPermuteOrder(runtimeRank, runtimePerm, out var order))
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-transpose-permutation",
                        "Transpose permutation cannot be represented after removing the static ONNX batch axis.",
                        "Use a valid static permutation with rank between 2 and 4.", true));
                else
                    layer.intParams[0] = order.ToString(CultureInfo.InvariantCulture);
            }

            if (string.Equals(node.opType, "Reshape", StringComparison.Ordinal))
            {
                if (!TryGetInitializerInput(node, 1, initializers, out var shapeTensor)
                    || !TryGetIntValues(shapeTensor, out var target)
                    || target.Length < 1 || target.Length > 4)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-reshape-shape",
                        "Reshape requires a static Int32/Int64 shape initializer with rank 1..4.",
                        "Fold the shape input before Aexis import.", true));
                }
                else
                {
                    var resolvedTarget = InferReshape(node, inputs.Count > 0 ? inputs[0].shape : null, initializers);
                    if (GetInt(node, "allowzero", 0) != 0 && target.Any(value => value == 0))
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-empty-reshape",
                            "Reshape allowzero=1 would create a zero-extent tensor with no texture storage contract.",
                            "Remove zero extents or use allowzero=0 with copied static dimensions.", true));
                    }
                    if (HasDynamic(resolvedTarget))
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-reshape-shape",
                            "Reshape target contains invalid dimensions, multiple inferred dimensions, or an incompatible element count.",
                            "Use at most one -1, use 0 only with allowzero=0, and preserve the static element count.", true));
                    }
                    else if (!HasZeroExtent(resolvedTarget))
                    {
                        var outputBatchAxis = inputs.Count > 0 && inputs[0].batchAxis >= 0
                            && resolvedTarget.Length > 0 && resolvedTarget[0] == 1 ? 0 : -1;
                        SetNcnnShapeParams(layer, RemoveAxis(resolvedTarget, outputBatchAxis));
                    }
                    KeepOnlyFirstBottom(layer);
                }
            }

            if (string.Equals(node.opType, "Flatten", StringComparison.Ordinal))
            {
                var descriptor = inputs.Count > 0 ? inputs[0] : null;
                if (descriptor == null || descriptor.batchAxis != 0 || GetInt(node, "axis", 1) != 1)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-flatten-axis",
                        "The texture-native Flatten profile requires a leading static batch=1 axis and ONNX axis=1.",
                        "Reshape explicitly to the required rank-1 runtime tensor before import.", true));
                }
            }

            if (string.Equals(node.opType, "Squeeze", StringComparison.Ordinal)
                || string.Equals(node.opType, "Unsqueeze", StringComparison.Ordinal))
            {
                var axes = GetInts(node, "axes", null);
                if ((axes == null || axes.Length == 0) && TryGetInitializerInput(node, 1, initializers, out var axesTensor))
                    TryGetIntValues(axesTensor, out axes);
                if (string.Equals(node.opType, "Squeeze", StringComparison.Ordinal) && (axes == null || axes.Length == 0))
                {
                    var sourceShape = inputs.Count > 0 ? inputs[0]?.shape : null;
                    if (sourceShape != null)
                    {
                        var inferredAxes = new List<long>();
                        for (var axis = 0; axis < sourceShape.Length; axis++)
                            if (sourceShape[axis] == 1) inferredAxes.Add(axis);
                        axes = inferredAxes.ToArray();
                    }
                }
                if (string.Equals(node.opType, "Unsqueeze", StringComparison.Ordinal) && (axes == null || axes.Length == 0))
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "missing-static-axes",
                        "Unsqueeze requires static axes.", "Fold the axes input before import.", true));
                }
                else if (axes != null && axes.Length > 0)
                {
                    var sourceDescriptor = inputs.Count > 0 ? inputs[0] : null;
                    var outputRank = string.Equals(node.opType, "Unsqueeze", StringComparison.Ordinal)
                        ? (sourceDescriptor?.shape.Length ?? 0) + axes.Length
                        : sourceDescriptor?.shape.Length ?? 0;
                    var runtimeAxes = TranslateAxesRemovingBatch(
                        axes,
                        outputRank,
                        string.Equals(node.opType, "Unsqueeze", StringComparison.Ordinal)
                            ? InferUnsqueezeBatchAxis(sourceDescriptor, axes, outputRank)
                            : sourceDescriptor?.batchAxis ?? -1);
                    if (runtimeAxes.Length == 0)
                    {
                        ConvertToNoop(layer);
                    }
                    else
                    {
                        layer.intParams[-23303] = EncodeIntArray(runtimeAxes);
                        layer.stringParams["axes"] = Join(runtimeAxes);
                    }
                }
                else if (string.Equals(node.opType, "Squeeze", StringComparison.Ordinal))
                {
                    ConvertToNoop(layer);
                }
                KeepOnlyFirstBottom(layer);
            }

            if (node.opType.StartsWith("Reduce", StringComparison.Ordinal))
            {
                var axes = GetInts(node, "axes", null);
                var hasAxesInput = node.inputs.Count > 1 && !string.IsNullOrEmpty(node.inputs[1]);
                if ((axes == null || axes.Length == 0) && hasAxesInput)
                {
                    if (!TryGetInitializerInput(node, 1, initializers, out var axesTensor) || !TryGetIntValues(axesTensor, out axes))
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "dynamic-reduction-axes",
                            node.opType + " requires static axes for texture-native lowering.",
                            "Fold the reduction axes into an Int32/Int64 initializer.", true));
                    }
                }
                var noopWithEmptyAxes = GetInt(node, "noop_with_empty_axes", 0) != 0;
                if ((axes == null || axes.Length == 0) && noopWithEmptyAxes)
                {
                    ConvertToNoop(layer);
                }
                else
                {
                    layer.intParams[1] = axes == null || axes.Length == 0 ? "1" : "0";
                    if (axes != null && axes.Length > 0)
                    {
                        var descriptor = inputs.Count > 0 ? inputs[0] : null;
                        var runtimeAxes = TranslateAxesRemovingBatch(axes, descriptor?.shape.Length ?? 0, descriptor?.batchAxis ?? -1);
                        if (runtimeAxes.Length == 0)
                        {
                            ConvertToNoop(layer);
                        }
                        else
                        {
                            layer.intParams[1] = "0";
                            layer.intParams[-23303] = EncodeIntArray(runtimeAxes);
                            layer.stringParams["axes"] = Join(runtimeAxes);
                        }
                    }
                    layer.intParams[4] = GetInt(node, "keepdims", 1).ToString(CultureInfo.InvariantCulture);
                }
                KeepOnlyFirstBottom(layer);
            }

            if (string.Equals(node.opType, "Cast", StringComparison.Ordinal)
                || string.Equals(node.opType, "CastLike", StringComparison.Ordinal))
            {
                var from = inputs.Count > 0 ? ToNcnnCastType(inputs[0].dataType, inputs[0].onnxDataType) : 0;
                var castLike = string.Equals(node.opType, "CastLike", StringComparison.Ordinal);
                var targetOnnxType = castLike
                    ? inputs.Count > 1 ? inputs[1].onnxDataType : 0
                    : GetInt(node, "to", 0);
                var targetType = castLike
                    ? inputs.Count > 1 ? inputs[1].dataType : TensorDataType.Unknown
                    : FromOnnxDataType(targetOnnxType);
                var to = ToNcnnCastType(targetType, targetOnnxType);
                if (from == 0 || to == 0)
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-cast-dtype",
                        "Cast only supports FP32, FP16, Int32, Int8, and UInt8 in the Aexis texture contract.",
                        "Insert a supported explicit cast before import.", true));
                layer.intParams[0] = from.ToString(CultureInfo.InvariantCulture);
                layer.intParams[1] = to.ToString(CultureInfo.InvariantCulture);
                if (castLike) KeepOnlyFirstBottom(layer);
            }

            if (string.Equals(node.opType, "Shape", StringComparison.Ordinal))
            {
                layer.intParams[0] = GetInt(node, "start", 0).ToString(CultureInfo.InvariantCulture);
                layer.intParams[1] = GetInt(node, "end", inputs.Count > 0 ? inputs[0].shape.Length : 0).ToString(CultureInfo.InvariantCulture);
                layer.stringParams["logical_dtype"] = "Int32";
            }

            if (string.Equals(node.opType, "Size", StringComparison.Ordinal))
                layer.stringParams["logical_dtype"] = "Int32";

            if (string.Equals(node.opType, "ArgMax", StringComparison.Ordinal) || string.Equals(node.opType, "ArgMin", StringComparison.Ordinal))
            {
                var descriptor = inputs.Count > 0 ? inputs[0] : null;
                if (!TryTranslateRuntimeAxis(descriptor, GetInt(node, "axis", 0), out var runtimeAxis))
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-batch-axis",
                        node.opType + " on the removed singleton batch axis would require generating an Int32 zero tensor.",
                        "Fold this singleton-axis reduction before import.", true));
                else
                    layer.intParams[0] = runtimeAxis.ToString(CultureInfo.InvariantCulture);
                layer.intParams[1] = GetInt(node, "keepdims", 1).ToString(CultureInfo.InvariantCulture);
                layer.intParams[2] = GetInt(node, "select_last_index", 0).ToString(CultureInfo.InvariantCulture);
                layer.stringParams["logical_dtype"] = "Int32";
            }

            if (string.Equals(node.opType, "Where", StringComparison.Ordinal))
            {
                var condition = inputs.Count > 0 ? inputs[0] : null;
                var whenTrue = inputs.Count > 1 ? inputs[1] : null;
                var whenFalse = inputs.Count > 2 ? inputs[2] : null;
                if (inputs.Count != 3 || condition == null || condition.onnxDataType != 9
                    || !IsStrictFloat32Descriptor(whenTrue) || !IsStrictFloat32Descriptor(whenFalse)
                    || whenTrue.onnxDataType != 0 && whenFalse.onnxDataType != 0 && whenTrue.onnxDataType != whenFalse.onnxDataType)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-where-profile",
                        "Where texture lowering requires a BOOL condition and matching FP32 true/false tensors.",
                        "Cast the condition to BOOL and both data inputs to FP32 before import.", true));
                }
            }

            if (string.Equals(node.opType, "TopK", StringComparison.Ordinal))
            {
                if (inputs.Count == 0 || !IsStrictFloat32Descriptor(inputs[0]))
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-topk-input-dtype",
                        "TopK texture lowering requires an FP32 input tensor.", "Cast the TopK data input to FP32 before import.", true));
                if (!TryGetInitializerInput(node, 1, initializers, out var kTensor) || !TryGetIntValues(kTensor, out var k) || k.Length != 1 || k[0] <= 0 || k[0] > int.MaxValue)
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "missing-static-topk-k", "TopK requires a positive static scalar K initializer.", "Fold K before import.", true));
                else
                {
                    var descriptor = inputs.Count > 0 ? inputs[0] : null;
                    if (!TryTranslateRuntimeAxis(descriptor, GetInt(node, "axis", -1), out var runtimeAxis))
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-batch-axis",
                            "TopK on the removed singleton batch axis is not a runtime texture operation.",
                            "Fold this TopK or select a non-batch axis.", true));
                    else
                        layer.intParams[0] = runtimeAxis.ToString(CultureInfo.InvariantCulture);
                    layer.intParams[1] = k[0].ToString(CultureInfo.InvariantCulture);
                    layer.intParams[2] = GetInt(node, "largest", 1).ToString(CultureInfo.InvariantCulture);
                    layer.intParams[3] = GetInt(node, "sorted", 1).ToString(CultureInfo.InvariantCulture);
                    layer.stringParams["k"] = layer.intParams[1];
                    layer.stringParams["values_logical_dtype"] = inputs.Count > 0 && inputs[0].dataType == TensorDataType.Int32 ? "Int32" : "Float32";
                    layer.stringParams["indices_logical_dtype"] = "Int32";
                    KeepOnlyFirstBottom(layer);
                }
            }

            if (string.Equals(node.opType, "Tile", StringComparison.Ordinal)
                || string.Equals(node.opType, "Expand", StringComparison.Ordinal))
            {
                if (!TryGetInitializerInput(node, 1, initializers, out var valuesTensor) || !TryGetIntValues(valuesTensor, out var values) || values.Length == 0 || values.Length > 4)
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "missing-static-shape-input", node.opType + " requires a static rank-1 Int32/Int64 shape input.", "Fold the shape/repeats input before import.", true));
                else
                {
                    var descriptor = inputs.Count > 0 ? inputs[0] : null;
                    var batchAxis = descriptor?.batchAxis ?? -1;
                    if (batchAxis >= 0 && (batchAxis >= values.Length || values[batchAxis] != 1))
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-batch-expansion",
                            node.opType + " attempts to replicate the removed static batch axis.",
                            "Keep the ONNX batch dimension equal to 1.", true));
                    }
                    var runtimeValues = RemoveAxis(values, batchAxis);
                    if (string.Equals(node.opType, "Tile", StringComparison.Ordinal)) layer.intParams[-23302] = EncodeIntArray(runtimeValues);
                    else layer.stringParams["shape"] = Join(runtimeValues);
                    KeepOnlyFirstBottom(layer);
                }
            }

            if (string.Equals(node.opType, "ConstantOfShape", StringComparison.Ordinal))
            {
                if (!TryGetInitializerInput(node, 0, initializers, out var shapeTensor) || !TryGetIntValues(shapeTensor, out var values) || values.Length == 0 || values.Length > 4)
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "missing-static-shape-input", "ConstantOfShape requires a static rank-1 Int32/Int64 shape input.", "Fold the shape input before import.", true));
                else
                {
                    layer.stringParams["shape"] = Join(values);
                    layer.stringParams["logical_dtype"] = "Float32";
                    if (node.attributes.TryGetValue("value", out var valueAttribute))
                    {
                        if (valueAttribute?.tensor == null || !TryGetScalarFloat(valueAttribute.tensor, out var fill))
                        {
                            diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-constant-of-shape-value",
                                "ConstantOfShape value must be a supported static scalar tensor.",
                                "Use a scalar FP32 or Int32 fill value.", true));
                        }
                        else
                        {
                            layer.stringParams["value"] = fill.ToString("R", CultureInfo.InvariantCulture);
                            if (!TryResolveRFloatLogicalDtype(valueAttribute.tensor, out var logicalDtype))
                            {
                                diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-constant-of-shape-dtype",
                                    "ConstantOfShape value dtype is outside the Float32/Int32 RFloat texture contract.",
                                    "Use an FP32 or range-checked Int32 scalar value.", true));
                            }
                            else
                            {
                                layer.stringParams["logical_dtype"] = logicalDtype;
                            }
                        }
                    }
                    layer.bottomNames = Array.Empty<string>();
                    layer.bottoms = 0;
                }
            }

            if (string.Equals(node.opType, "Range", StringComparison.Ordinal))
            {
                if (!TryGetInitializerInput(node, 0, initializers, out var startTensor) || !TryGetScalarFloat(startTensor, out var start)
                    || !TryGetInitializerInput(node, 1, initializers, out var limitTensor) || !TryGetScalarFloat(limitTensor, out var limit)
                    || !TryGetInitializerInput(node, 2, initializers, out var deltaTensor) || !TryGetScalarFloat(deltaTensor, out var delta) || Math.Abs(delta) < 1e-12f)
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "missing-static-range-input", "Range requires static scalar start, limit, and non-zero delta inputs.", "Fold all Range inputs before import.", true));
                else
                {
                    if (!TryResolveRFloatLogicalDtype(startTensor, out var logicalDtype)
                        || !TryResolveRFloatLogicalDtype(limitTensor, out var limitLogicalDtype)
                        || !TryResolveRFloatLogicalDtype(deltaTensor, out var deltaLogicalDtype)
                        || !string.Equals(logicalDtype, limitLogicalDtype, StringComparison.Ordinal)
                        || !string.Equals(logicalDtype, deltaLogicalDtype, StringComparison.Ordinal))
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-range-dtype",
                            "Range start, limit, and delta must have one matching Float32 or range-checked Int32 dtype.",
                            "Cast all Range inputs to FP32 or Int32 before export.", true));
                    }
                    layer.stringParams["start"] = start.ToString("R", CultureInfo.InvariantCulture);
                    layer.stringParams["limit"] = limit.ToString("R", CultureInfo.InvariantCulture);
                    layer.stringParams["delta"] = delta.ToString("R", CultureInfo.InvariantCulture);
                    if (!string.IsNullOrEmpty(logicalDtype))
                        layer.stringParams["logical_dtype"] = logicalDtype;
                    layer.bottomNames = Array.Empty<string>();
                    layer.bottoms = 0;
                }
            }

            if (string.Equals(node.opType, "CumSum", StringComparison.Ordinal))
            {
                if (!TryGetInitializerInput(node, 1, initializers, out var axisTensor) || !TryGetIntValues(axisTensor, out var axisValues) || axisValues.Length != 1)
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "missing-static-axis", "CumSum requires a static scalar Int32/Int64 axis.", "Fold the axis input before import.", true));
                else
                {
                    var descriptor = inputs.Count > 0 ? inputs[0] : null;
                    if (!TryTranslateRuntimeAxis(descriptor, (int)axisValues[0], out var runtimeAxis))
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-batch-axis",
                            "CumSum cannot scan the removed singleton batch axis.", "Fold this CumSum before import.", true));
                    else
                        layer.intParams[0] = runtimeAxis.ToString(CultureInfo.InvariantCulture);
                    layer.intParams[1] = GetInt(node, "exclusive", 0).ToString(CultureInfo.InvariantCulture);
                    layer.intParams[2] = GetInt(node, "reverse", 0).ToString(CultureInfo.InvariantCulture);
                    KeepOnlyFirstBottom(layer);
                }
            }

            if (string.Equals(node.opType, "OneHot", StringComparison.Ordinal))
            {
                if (!TryGetInitializerInput(node, 1, initializers, out var depthTensor) || !TryGetIntValues(depthTensor, out var depth) || depth.Length != 1 || depth[0] <= 0
                    || !TryGetInitializerInput(node, 2, initializers, out var valuesTensor) || !TryGetFloatValues(valuesTensor, out var values) || values.Length != 2)
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "missing-static-onehot-input", "OneHot requires static positive depth and two static off/on values.", "Fold depth and values before import.", true));
                else
                {
                    var descriptor = inputs.Count > 0 ? inputs[0] : null;
                    var outputRank = (descriptor?.shape?.Length ?? 0) + 1;
                    var outputAxis = NormalizeAxis(GetInt(node, "axis", -1), outputRank);
                    if (outputAxis < 0 || outputAxis >= outputRank)
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-onehot-axis",
                            "OneHot axis is outside the inferred output rank.", "Use an axis in [-rank-1, rank].", true));
                    }
                    var outputBatchAxis = descriptor != null && descriptor.batchAxis >= 0
                        ? descriptor.batchAxis + (outputAxis <= descriptor.batchAxis ? 1 : 0)
                        : -1;
                    var runtimeAxis = outputBatchAxis >= 0 && outputAxis > outputBatchAxis ? outputAxis - 1 : outputAxis;
                    layer.intParams[1] = depth[0].ToString(CultureInfo.InvariantCulture);
                    layer.intParams[2] = runtimeAxis.ToString(CultureInfo.InvariantCulture);
                    layer.stringParams["off_value"] = values[0].ToString("R", CultureInfo.InvariantCulture);
                    layer.stringParams["on_value"] = values[1].ToString("R", CultureInfo.InvariantCulture);
                    if (!TryResolveRFloatLogicalDtype(valuesTensor, out var logicalDtype))
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-onehot-values-dtype",
                            "OneHot values must use Float32 or range-checked Int32 texture semantics.",
                            "Cast the two values to FP32 or Int32 before export.", true));
                    }
                    else
                    {
                        layer.stringParams["logical_dtype"] = logicalDtype;
                    }
                    KeepOnlyFirstBottom(layer);
                }
            }

            if (string.Equals(node.opType, "Split", StringComparison.Ordinal))
            {
                var descriptor = inputs.Count > 0 ? inputs[0] : null;
                var onnxAxis = GetInt(node, "axis", 0);
                var normalizedAxis = NormalizeAxis(onnxAxis, descriptor?.shape.Length ?? 0);
                if (!TryTranslateRuntimeAxis(descriptor, onnxAxis, out var runtimeAxis))
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-batch-split",
                        "Split cannot partition the removed static batch axis.", "Split a non-batch axis.", true));
                }
                else
                {
                    long[] split = GetInts(node, "split", null);
                    if ((split == null || split.Length == 0)
                        && TryGetInitializerInput(node, 1, initializers, out var splitTensor))
                        TryGetIntValues(splitTensor, out split);
                    var outputCount = node.outputs.Count;
                    var axisSize = descriptor != null && normalizedAxis >= 0 && normalizedAxis < descriptor.shape.Length
                        ? descriptor.shape[normalizedAxis] : -1;
                    if (split == null || split.Length == 0)
                    {
                        var requested = GetInt(node, "num_outputs", outputCount);
                        if (requested != outputCount || outputCount <= 0 || axisSize <= 0 || axisSize % outputCount != 0)
                        {
                            diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-equal-split",
                                "Equal Split requires a static axis size divisible by the output count.",
                                "Provide explicit static split sizes or evenly divisible outputs.", true));
                        }
                        else
                        {
                            split = new long[outputCount];
                            for (var i = 0; i < split.Length; i++) split[i] = axisSize / outputCount;
                        }
                    }
                    if (split != null)
                    {
                        long total = 0;
                        for (var i = 0; i < split.Length; i++) total += split[i];
                        if (split.Length != outputCount || total != axisSize)
                            diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-split-sizes",
                                "Split sizes must be positive, match the output count, and sum to the static axis size.",
                                "Correct the static Split sizes.", true));
                        else
                        {
                            for (var i = 0; i < split.Length; i++)
                                if (split[i] <= 0)
                                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-split-size",
                                        "Split sizes must all be positive.", "Remove empty Split outputs.", true));
                            layer.intParams[-23300] = EncodeIntArray(split);
                            layer.intParams[1] = runtimeAxis.ToString(CultureInfo.InvariantCulture);
                        }
                    }
                    KeepOnlyFirstBottom(layer);
                }
            }

            if (string.Equals(node.opType, "Slice", StringComparison.Ordinal))
            {
                if (!TryGetInitializerInput(node, 1, initializers, out var startsTensor) || !TryGetIntValues(startsTensor, out var starts)
                    || !TryGetInitializerInput(node, 2, initializers, out var endsTensor) || !TryGetIntValues(endsTensor, out var ends)
                    || starts.Length == 0 || starts.Length != ends.Length)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "missing-static-slice-input", "Slice requires matching static starts and ends.", "Fold Slice parameters before import.", true));
                }
                else
                {
                    long[] axes = null;
                    long[] steps = null;
                    if (TryGetInitializerInput(node, 3, initializers, out var axesTensor)) TryGetIntValues(axesTensor, out axes);
                    if (TryGetInitializerInput(node, 4, initializers, out var stepsTensor)) TryGetIntValues(stepsTensor, out steps);
                    if (axes == null || axes.Length == 0) axes = RangeAxes(starts.Length);
                    var supported = axes.Length == starts.Length && (steps == null || steps.Length == starts.Length);
                    if (steps != null) for (var i = 0; i < steps.Length; i++) supported &= steps[i] == 1;
                    if (!supported)
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-slice-steps", "Slice texture lowering supports static axes and step=1 only.", "Rewrite strided/reversed Slice before import.", true));
                    else
                    {
                        var descriptor = inputs.Count > 0 ? inputs[0] : null;
                        var seenAxes = new HashSet<int>();
                        var runtimeStarts = new List<long>();
                        var runtimeEnds = new List<long>();
                        var runtimeAxes = new List<long>();
                        for (var i = 0; i < axes.Length; i++)
                        {
                            var normalizedAxis = NormalizeAxis((int)axes[i], descriptor?.shape.Length ?? 0);
                            if (descriptor == null || normalizedAxis < 0 || normalizedAxis >= descriptor.shape.Length)
                            {
                                diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-slice-axis",
                                    "Slice axis is outside the input rank.", "Use unique axes in the input rank.", true));
                                continue;
                            }
                            if (!seenAxes.Add(normalizedAxis))
                            {
                                diagnostics.Add(Diagnostic(index, layer.name, node.opType, "duplicate-slice-axis",
                                    "Slice axes must be unique.", "Merge repeated slicing of the same axis before import.", true));
                                continue;
                            }
                            var dimension = descriptor.shape[normalizedAxis];
                            if (dimension <= 0)
                            {
                                diagnostics.Add(Diagnostic(index, layer.name, node.opType, "dynamic-slice-axis",
                                    "Slice requires a positive static extent for every sliced axis.", "Provide static value_info for the sliced tensor.", true));
                                continue;
                            }
                            var normalizedStart = NormalizeSliceBound(starts[i], dimension);
                            var normalizedEnd = NormalizeSliceBound(ends[i], dimension);
                            if (normalizedEnd <= normalizedStart)
                            {
                                diagnostics.Add(Diagnostic(index, layer.name, node.opType, "empty-slice-output",
                                    "Slice produces an empty axis, which has no Pack4 texture storage contract.", "Remove empty slices before Aexis import.", true));
                                continue;
                            }
                            if (descriptor != null && normalizedAxis == descriptor.batchAxis)
                            {
                                if (normalizedStart != 0 || normalizedEnd != 1)
                                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-batch-slice",
                                        "Slice changes or removes the static batch=1 element.", "Keep the complete batch axis slice [0:1].", true));
                                continue;
                            }
                            if (descriptor != null && descriptor.batchAxis >= 0 && normalizedAxis > descriptor.batchAxis) normalizedAxis--;
                            runtimeStarts.Add(normalizedStart);
                            runtimeEnds.Add(normalizedEnd);
                            runtimeAxes.Add(normalizedAxis);
                        }
                        if (runtimeAxes.Count == 0)
                        {
                            ConvertToNoop(layer);
                        }
                        else
                        {
                            layer.intParams[-23309] = EncodeIntArray(runtimeStarts);
                            layer.intParams[-23310] = EncodeIntArray(runtimeEnds);
                            layer.intParams[-23311] = EncodeIntArray(runtimeAxes);
                        }
                        KeepOnlyFirstBottom(layer);
                    }
                }
            }

            if (string.Equals(node.opType, "InstanceNormalization", StringComparison.Ordinal)
                || string.Equals(node.opType, "LayerNormalization", StringComparison.Ordinal))
            {
                if (string.Equals(node.opType, "LayerNormalization", StringComparison.Ordinal))
                {
                    var rank = inputs.Count > 0 ? inputs[0]?.shape?.Length ?? 0 : 0;
                    var axis = NormalizeAxis(GetInt(node, "axis", -1), rank);
                    if (rank == 0 || axis != rank - 1)
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-layernorm-axis",
                            "The Aexis LayerNorm kernel normalizes only the final logical width axis.",
                            "Export LayerNormalization with axis=-1 or decompose the wider normalization.", true));
                    }
                    if (node.outputs.Count != 1)
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-layernorm-auxiliary-outputs",
                            "The P0 LayerNorm texture contract publishes only Y; Mean and InvStdDev auxiliary outputs are not implemented.",
                            "Remove unused LayerNormalization auxiliary outputs or add explicit texture-native output kernels.", true));
                    }
                    if (GetInt(node, "stash_type", 1) != 1)
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-layernorm-stash-type",
                            "LayerNormalization requires stash_type=1 for FP32 accumulation.",
                            "Export LayerNormalization with the default FP32 stash type.", true));
                    }
                }
                if (!TryGetInitializerInput(node, 1, initializers, out var scale) || scale.dims == null || scale.dims.Length != 1
                    || scale.dims[0] <= 0 || scale.dataType != TensorDataType.Float32)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "missing-static-normalization-affine",
                        node.opType + " requires a static rank-1 FP32 scale initializer.",
                        "Export scale and bias as immutable FP32 initializers.", true));
                }
                else
                {
                    var inputShape = inputs.Count > 0 ? inputs[0]?.shape : null;
                    var expectedAffineSize = string.Equals(node.opType, "LayerNormalization", StringComparison.Ordinal)
                        ? inputShape != null && inputShape.Length > 0 ? inputShape[inputShape.Length - 1] : -1
                        : inputShape != null && inputShape.Length > 1 ? inputShape[1] : -1;
                    if (expectedAffineSize <= 0 || scale.dims[0] != expectedAffineSize)
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "normalization-affine-shape-mismatch",
                            node.opType + " scale length does not match the normalized logical axis.",
                            "Export immutable scale/bias vectors whose length exactly matches the normalized axis.", true));
                    }
                    if (!TryGetInitializerInput(node, 2, initializers, out var bias)
                        || bias.dataType != TensorDataType.Float32 || bias.dims == null
                        || bias.dims.Length != 1 || bias.dims[0] != scale.dims[0])
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-normalization-bias",
                            node.opType + " bias must be a rank-1 FP32 initializer matching scale.",
                            "Export matching immutable scale and bias tensors.", true));
                    }
                    layer.intParams[0] = scale.dims[0].ToString(CultureInfo.InvariantCulture);
                    layer.intParams[1] = GetFloat(node, "epsilon", string.Equals(node.opType, "InstanceNormalization", StringComparison.Ordinal) ? 1e-5f : 1e-5f).ToString("R", CultureInfo.InvariantCulture);
                    layer.intParams[2] = "1";
                    layer.stringParams["onnx.scale"] = node.inputs[1];
                    if (node.inputs.Count > 2 && initializers.ContainsKey(node.inputs[2])) layer.stringParams["onnx.bias"] = node.inputs[2];
                    KeepOnlyFirstBottom(layer);
                }
            }
        }

        private static void ConfigurePadContract(
            OnnxNode node,
            AexisGraphModel.Layer layer,
            List<AexisOnnxTensorDescriptor> inputs,
            Dictionary<string, OnnxTensor> initializers,
            List<AexisOnnxLoweringDiagnostic> diagnostics,
            int index)
        {
            var descriptor = inputs.Count > 0 ? inputs[0] : null;
            long[] pads = GetInts(node, "pads", null);
            if ((pads == null || pads.Length == 0)
                && TryGetInitializerInput(node, 1, initializers, out var padsTensor))
                TryGetIntValues(padsTensor, out pads);
            if (descriptor == null || descriptor.shape == null || descriptor.shape.Length != 4
                || descriptor.batchAxis != 0 || pads == null || pads.Length != 8)
            {
                diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-pad-profile",
                    "Pad requires a static batch=1 rank-4 NCHW input and eight static pad values.",
                    "Fold pads and export an NCHW batch=1 Pad node.", true));
                return;
            }

            var mode = GetString(node, "mode");
            if (string.IsNullOrEmpty(mode)) mode = "constant";
            int padType;
            if (string.Equals(mode, "constant", StringComparison.Ordinal)) padType = 0;
            else if (string.Equals(mode, "edge", StringComparison.Ordinal)) padType = 1;
            else if (string.Equals(mode, "reflect", StringComparison.Ordinal)) padType = 2;
            else
            {
                diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-pad-mode",
                    "Pad mode " + mode + " has no exact Pack4 Padding kernel.",
                    "Use constant, edge, or reflect padding.", true));
                return;
            }

            for (var i = 0; i < pads.Length; i++)
                if (pads[i] < 0)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "negative-pad-unsupported",
                        "Negative Pad values require a crop followed by padding and are not accepted as a single strict node.",
                        "Lower negative pads to an explicit Slice/Crop plus Pad.", true));
                    return;
                }
            if (pads[0] != 0 || pads[1] != 0 || pads[4] != 0 || pads[5] != 0)
            {
                diagnostics.Add(Diagnostic(index, layer.name, node.opType, "batch-channel-pad-unsupported",
                    "The strict Pack4 Pad profile only pads H and W; N and C pads must be zero.",
                    "Rewrite batch/channel padding using explicit texture-native operators.", true));
                return;
            }

            var value = GetFloat(node, "value", 0f);
            var hasValueInput = node.inputs.Count > 2 && !string.IsNullOrEmpty(node.inputs[2]);
            OnnxTensor valueTensor = null;
            if (hasValueInput && !TryGetInitializerInput(node, 2, initializers, out valueTensor))
            {
                diagnostics.Add(Diagnostic(index, layer.name, node.opType, "dynamic-pad-value",
                    "Pad constant_value must be a static scalar for texture-native lowering.",
                    "Fold the constant_value input before import.", true));
                return;
            }
            if (hasValueInput && !TryGetScalarFloat(valueTensor, out value))
            {
                diagnostics.Add(Diagnostic(index, layer.name, node.opType, "non-scalar-pad-value",
                    "Pad constant_value must be a static scalar.", "Fold a scalar pad value before import.", true));
                return;
            }
            layer.intParams[0] = pads[2].ToString(CultureInfo.InvariantCulture);
            layer.intParams[1] = pads[6].ToString(CultureInfo.InvariantCulture);
            layer.intParams[2] = pads[3].ToString(CultureInfo.InvariantCulture);
            layer.intParams[3] = pads[7].ToString(CultureInfo.InvariantCulture);
            layer.intParams[4] = padType.ToString(CultureInfo.InvariantCulture);
            layer.intParams[5] = value.ToString("R", CultureInfo.InvariantCulture);
            layer.intParams[6] = "0";
            layer.intParams[7] = "0";
            layer.intParams[8] = "0";
            KeepOnlyFirstBottom(layer);
        }

        private static void ConfigureResizeContract(
            OnnxNode node,
            AexisGraphModel.Layer layer,
            List<AexisOnnxTensorDescriptor> inputs,
            Dictionary<string, OnnxTensor> initializers,
            List<AexisOnnxLoweringDiagnostic> diagnostics,
            int index)
        {
            var descriptor = inputs.Count > 0 ? inputs[0] : null;
            if (descriptor == null || descriptor.shape == null || descriptor.shape.Length != 4 || descriptor.batchAxis != 0)
            {
                diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-resize-rank",
                    "Resize/Upsample strict lowering requires a static batch=1 rank-4 NCHW tensor.",
                    "Export a static NCHW batch=1 resize.", true));
                return;
            }

            var mode = GetString(node, "mode");
            if (string.IsNullOrEmpty(mode)) mode = "nearest";
            var resizeType = string.Equals(mode, "nearest", StringComparison.Ordinal) ? 1
                : string.Equals(mode, "linear", StringComparison.Ordinal) ? 2 : 0;
            if (resizeType == 0)
            {
                diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-resize-mode",
                    "Resize mode " + mode + " has no verified exact Pack4 production profile.",
                    "Use nearest or linear interpolation.", true));
                return;
            }

            var coordinateMode = GetString(node, "coordinate_transformation_mode");
            if (string.IsNullOrEmpty(coordinateMode))
                coordinateMode = string.Equals(node.opType, "Upsample", StringComparison.Ordinal) ? "asymmetric" : "half_pixel";
            var nearestMode = GetString(node, "nearest_mode");
            if (string.IsNullOrEmpty(nearestMode))
                nearestMode = string.Equals(node.opType, "Upsample", StringComparison.Ordinal) ? "floor" : "round_prefer_floor";
            var coordinateParam = -1;
            var alignCorners = 0;
            if (string.Equals(coordinateMode, "half_pixel", StringComparison.Ordinal)) coordinateParam = 0;
            else if (string.Equals(coordinateMode, "asymmetric", StringComparison.Ordinal)) coordinateParam = 1;
            else if (string.Equals(coordinateMode, "align_corners", StringComparison.Ordinal)) { coordinateParam = 0; alignCorners = 1; }
            var nearestProfileSupported = string.Equals(coordinateMode, "half_pixel", StringComparison.Ordinal)
                    && string.Equals(nearestMode, "round_prefer_floor", StringComparison.Ordinal)
                || string.Equals(coordinateMode, "asymmetric", StringComparison.Ordinal)
                    && string.Equals(nearestMode, "floor", StringComparison.Ordinal);
            var keepAspectRatioPolicy = GetString(node, "keep_aspect_ratio_policy");
            if (string.IsNullOrEmpty(keepAspectRatioPolicy)) keepAspectRatioPolicy = "stretch";
            if (coordinateParam < 0
                || resizeType == 1 && !nearestProfileSupported
                || GetInt(node, "antialias", 0) != 0
                || !string.Equals(keepAspectRatioPolicy, "stretch", StringComparison.Ordinal))
            {
                diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-resize-coordinates",
                    "Resize coordinate/nearest/antialias attributes do not match the exact Aexis Pack4 sampling profile.",
                    "Use linear half_pixel/align_corners/asymmetric, or nearest half_pixel+round_prefer_floor / asymmetric+floor, without antialias.", true));
                return;
            }

            var axes = GetInts(node, "axes", null);
            if (axes != null && axes.Length > 0)
            {
                axes = NormalizeAxes(axes, descriptor.shape.Length);
                var fullAxes = axes.Length == 4 && axes[0] == 0 && axes[1] == 1 && axes[2] == 2 && axes[3] == 3;
                var spatialAxes = axes.Length == 2 && axes[0] == 2 && axes[1] == 3;
                if (!fullAxes && !spatialAxes)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-resize-axes",
                        "Resize axes must select NCHW [0,1,2,3] or spatial [2,3] dimensions in canonical order.",
                        "Canonicalize Resize axes and preserve N/C dimensions before import.", true));
                    return;
                }
            }
            var sizesInput = string.Equals(node.opType, "Resize", StringComparison.Ordinal) ? 3 : -1;
            var scalesInput = string.Equals(node.opType, "Resize", StringComparison.Ordinal) ? 2 : 1;
            long[] sizes = null;
            float[] scales = null;
            if (sizesInput >= 0 && TryGetInitializerInput(node, sizesInput, initializers, out var sizesTensor))
                TryGetIntValues(sizesTensor, out sizes);
            if (TryGetInitializerInput(node, scalesInput, initializers, out var scalesTensor))
                TryGetFloatValues(scalesTensor, out scales);
            if (sizes != null && sizes.Length > 0 && scales != null && scales.Length > 0)
            {
                diagnostics.Add(Diagnostic(index, layer.name, node.opType, "ambiguous-resize-size-source",
                    "Resize must specify exactly one of scales or sizes, but both are non-empty immutable inputs.",
                    "Export Resize with one empty optional input and one static size source.", true));
                return;
            }
            if ((sizes == null || sizes.Length == 0) && (scales == null || scales.Length == 0))
            {
                scales = node.attributes.TryGetValue("scales", out var scalesAttribute) && scalesAttribute.floats.Count > 0
                    ? scalesAttribute.floats.ToArray() : null;
            }

            var spatialOnly = axes != null && axes.Length == 2 && axes[0] == 2 && axes[1] == 3;
            if (sizes != null && sizes.Length > 0)
            {
                if (!(sizes.Length == 4 || spatialOnly && sizes.Length == 2)
                    || sizes[sizes.Length - 2] <= 0 || sizes[sizes.Length - 1] <= 0
                    || sizes.Length == 4 && (sizes[0] != descriptor.shape[0] || sizes[1] != descriptor.shape[1]))
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-resize-sizes",
                        "Resize sizes must preserve N/C and provide positive static H/W.", "Fold valid static spatial sizes.", true));
                    return;
                }
                layer.intParams[3] = sizes[sizes.Length - 2].ToString(CultureInfo.InvariantCulture);
                layer.intParams[4] = sizes[sizes.Length - 1].ToString(CultureInfo.InvariantCulture);
            }
            else
            {
                if (scales == null || !(scales.Length == 4 || spatialOnly && scales.Length == 2)
                    || scales[scales.Length - 2] <= 0f || scales[scales.Length - 1] <= 0f
                    || scales.Length == 4 && (Math.Abs(scales[0] - 1f) > 1e-6f || Math.Abs(scales[1] - 1f) > 1e-6f))
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "invalid-resize-scales",
                        "Resize scales must preserve N/C and provide positive static H/W scales.", "Fold valid static spatial scales.", true));
                    return;
                }
                layer.intParams[1] = scales[scales.Length - 2].ToString("R", CultureInfo.InvariantCulture);
                layer.intParams[2] = scales[scales.Length - 1].ToString("R", CultureInfo.InvariantCulture);
            }
            layer.intParams[0] = resizeType.ToString(CultureInfo.InvariantCulture);
            layer.intParams[5] = "0";
            layer.intParams[6] = alignCorners.ToString(CultureInfo.InvariantCulture);
            layer.intParams[100] = coordinateParam.ToString(CultureInfo.InvariantCulture);
            KeepOnlyFirstBottom(layer);
        }

        private static bool TryResolveAutoPads2D(
            OnnxNode node,
            long[] inputShape,
            long[] kernel,
            long[] strides,
            long[] dilations,
            bool transpose,
            out long[] pads,
            out string reason)
        {
            pads = GetInts(node, "pads", new long[] { 0, 0, 0, 0 });
            var autoPad = GetString(node, "auto_pad");
            if (string.IsNullOrEmpty(autoPad) || string.Equals(autoPad, "NOTSET", StringComparison.Ordinal))
            {
                reason = pads.Length == 4 ? null : "Explicit pads must contain four values.";
                return pads.Length == 4;
            }
            if (string.Equals(autoPad, "VALID", StringComparison.Ordinal))
            {
                pads = new long[] { 0, 0, 0, 0 };
                reason = null;
                return true;
            }
            if (transpose)
            {
                reason = "ConvTranspose auto_pad=" + autoPad + " is ambiguous with output_shape/output_padding in the current immutable packer.";
                return false;
            }
            if (!string.Equals(autoPad, "SAME_UPPER", StringComparison.Ordinal)
                && !string.Equals(autoPad, "SAME_LOWER", StringComparison.Ordinal))
            {
                reason = "Unsupported auto_pad=" + autoPad + ".";
                return false;
            }
            if (inputShape == null || inputShape.Length != 4 || HasDynamic(inputShape)
                || kernel == null || kernel.Length != 2 || strides == null || strides.Length != 2
                || dilations == null || dilations.Length != 2)
            {
                reason = "SAME padding requires a fully static rank-4 input and two kernel/stride/dilation values.";
                return false;
            }

            pads = new long[4];
            var sameLower = string.Equals(autoPad, "SAME_LOWER", StringComparison.Ordinal);
            for (var axis = 0; axis < 2; axis++)
            {
                if (kernel[axis] <= 0 || strides[axis] <= 0 || dilations[axis] <= 0)
                {
                    reason = "Kernel, stride, and dilation must be positive for SAME padding.";
                    return false;
                }
                var input = inputShape[axis + 2];
                var output = (input + strides[axis] - 1) / strides[axis];
                var extent = dilations[axis] * (kernel[axis] - 1) + 1;
                var total = Math.Max(0, (output - 1) * strides[axis] + extent - input);
                var begin = sameLower ? (total + 1) / 2 : total / 2;
                pads[axis] = begin;
                pads[axis + 2] = total - begin;
            }
            reason = null;
            return true;
        }

        private static bool IsNchw2DAttributesSupported(OnnxNode node, out string reason)
        {
            var autoPad = GetString(node, "auto_pad");
            if (!string.IsNullOrEmpty(autoPad) && !string.Equals(autoPad, "NOTSET", StringComparison.Ordinal))
            {
                reason = "auto_pad=" + autoPad + " is not an explicit ncnn texture contract.";
                return false;
            }
            reason = null;
            return true;
        }

        private static void InsertRuntimeInitializerLayers(
            AexisGraphModel graph,
            Dictionary<string, OnnxTensor> initializers,
            List<AexisOnnxLoweringDiagnostic> diagnostics)
        {
            var required = new HashSet<string>(StringComparer.Ordinal);
            foreach (var layer in graph.layers)
                foreach (var bottom in layer.bottomNames ?? Array.Empty<string>())
                    if (!string.IsNullOrEmpty(bottom) && initializers.ContainsKey(bottom))
                        required.Add(bottom);
            if (required.Count == 0) return;

            var constants = new List<AexisGraphModel.Layer>();
            foreach (var name in required)
            {
                var tensor = initializers[name];
                if (!CanUploadMemoryData(tensor, out var reason))
                {
                    diagnostics.Add(Diagnostic(-1, name, "Initializer", "unsupported-runtime-initializer",
                        "Initializer " + name + " cannot be represented by a texture-native MemoryData upload: " + reason,
                        "Use FP32 or range-checked Int32/Int64 data with rank <= 4.", true));
                    continue;
                }
                var layer = CreateLayer("MemoryData", "onnx_const_" + SanitizeName(name), Array.Empty<string>(), new[] { name });
                SetNcnnShapeParams(layer, ToRuntimeConstantShape(tensor.dims));
                layer.intParams[21] = "1";
                layer.stringParams["onnx.tensor"] = name;
                layer.stringParams["logical_dtype"] = tensor.dataType == TensorDataType.Int32 ? "Int32" : "Float32";
                constants.Add(layer);
            }

            var insertAt = 0;
            while (insertAt < graph.layers.Count && graph.layers[insertAt].type == AexisLayerTypes.Input) insertAt++;
            graph.layers.InsertRange(insertAt, constants);
        }

        private static bool CanUploadMemoryData(OnnxTensor tensor, out string reason)
        {
            if (tensor == null) { reason = "tensor is null"; return false; }
            if (tensor.dims == null || tensor.dims.Length > 4) { reason = "rank exceeds 4"; return false; }
            if (HasDynamic(tensor.dims) || ElementCount(tensor.dims) <= 0) { reason = "shape is not positive and static"; return false; }
            if (!tensor.TryValidatePayload(out reason)) return false;
            if (tensor.dataType == TensorDataType.Float32) { reason = null; return true; }
            if (tensor.dataType == TensorDataType.Int32 && TryGetIntValues(tensor, out var values))
            {
                for (var index = 0; index < values.Length; index++)
                {
                    if (values[index] < int.MinValue || values[index] > int.MaxValue || Math.Abs((double)values[index]) > 16777216d)
                    {
                        reason = "integer payload is outside the FP32-exact Int32 texture range";
                        return false;
                    }
                }
                reason = null;
                return true;
            }
            reason = "dtype or payload is unsupported";
            return false;
        }

        private static long[] ToRuntimeConstantShape(long[] onnxShape)
        {
            if (onnxShape == null || onnxShape.Length == 0) return new long[] { 1 };
            if (onnxShape.Length == 4 && onnxShape[0] == 1)
                return new[] { onnxShape[1], onnxShape[2], onnxShape[3] };
            return Clone(onnxShape);
        }

        private static void SetNcnnShapeParams(AexisGraphModel.Layer layer, long[] logicalShape)
        {
            layer.intParams.Remove(0);
            layer.intParams.Remove(1);
            layer.intParams.Remove(2);
            layer.intParams.Remove(11);
            if (logicalShape == null || logicalShape.Length == 0) return;
            layer.intParams[0] = logicalShape[logicalShape.Length - 1].ToString(CultureInfo.InvariantCulture);
            if (logicalShape.Length >= 2) layer.intParams[1] = logicalShape[logicalShape.Length - 2].ToString(CultureInfo.InvariantCulture);
            if (logicalShape.Length == 3) layer.intParams[2] = logicalShape[0].ToString(CultureInfo.InvariantCulture);
            if (logicalShape.Length == 4)
            {
                layer.intParams[11] = logicalShape[1].ToString(CultureInfo.InvariantCulture);
                layer.intParams[2] = logicalShape[0].ToString(CultureInfo.InvariantCulture);
            }
        }

        private static void KeepOnlyFirstBottom(AexisGraphModel.Layer layer)
        {
            layer.bottomNames = layer.bottomNames != null && layer.bottomNames.Length > 0
                ? new[] { layer.bottomNames[0] }
                : Array.Empty<string>();
            layer.bottoms = layer.bottomNames.Length;
        }

        private static bool IsSupportedPReluSlope(AexisOnnxTensorDescriptor input, OnnxTensor slope)
        {
            if (slope == null || slope.dims == null || slope.ElementCount <= 0)
                return false;
            if (slope.ElementCount == 1)
                return true;
            if (input?.shape == null || input.shape.Length != 4 || input.batchAxis != 0
                || input.shape[1] <= 0 || slope.ElementCount != input.shape[1]
                || slope.dims.Length > input.shape.Length)
            {
                return false;
            }

            var offset = input.shape.Length - slope.dims.Length;
            for (var axis = 0; axis < input.shape.Length; axis++)
            {
                var slopeExtent = axis < offset ? 1 : slope.dims[axis - offset];
                if (slopeExtent <= 0 || slopeExtent != 1 && slopeExtent != input.shape[axis])
                    return false;
                if (axis != 1 && slopeExtent != 1)
                    return false;
            }
            return true;
        }

        private static bool TryGetInitializerInput(OnnxNode node, int inputIndex, Dictionary<string, OnnxTensor> tensors, out OnnxTensor tensor)
        {
            tensor = null;
            return node != null && node.inputs.Count > inputIndex && !string.IsNullOrEmpty(node.inputs[inputIndex])
                && tensors.TryGetValue(node.inputs[inputIndex], out tensor);
        }

        private static bool TryGetIntValues(OnnxTensor tensor, out long[] values)
        {
            values = null;
            if (tensor == null || tensor.ElementCount < 0 || tensor.ElementCount > int.MaxValue) return false;
            var count = (int)tensor.ElementCount;
            if (tensor.int64Data != null && tensor.int64Data.Length == count)
            {
                values = (long[])tensor.int64Data.Clone();
                return true;
            }
            if (tensor.int32Data != null && tensor.int32Data.Length == count)
            {
                values = new long[count];
                for (var i = 0; i < count; i++) values[i] = tensor.int32Data[i];
                return true;
            }
            if (tensor.rawData == null) return false;
            if (tensor.onnxDataType == 7 && tensor.rawData.Length == checked(count * sizeof(long)))
            {
                values = new long[count];
                for (var i = 0; i < count; i++) values[i] = BitConverter.ToInt64(tensor.rawData, i * sizeof(long));
                return true;
            }
            if (tensor.onnxDataType == 6 && tensor.rawData.Length == checked(count * sizeof(int)))
            {
                values = new long[count];
                for (var i = 0; i < count; i++) values[i] = BitConverter.ToInt32(tensor.rawData, i * sizeof(int));
                return true;
            }
            if (tensor.onnxDataType == 9 && tensor.rawData.Length == count)
            {
                values = new long[count];
                for (var i = 0; i < count; i++) values[i] = tensor.rawData[i] == 0 ? 0 : 1;
                return true;
            }
            return false;
        }

        private static bool TryGetFloatValues(OnnxTensor tensor, out float[] values)
        {
            values = null;
            if (tensor == null || tensor.ElementCount < 0 || tensor.ElementCount > int.MaxValue) return false;
            var count = (int)tensor.ElementCount;
            if (tensor.dataType == TensorDataType.Float32)
            {
                if (tensor.floatData != null && tensor.floatData.Length == count)
                {
                    values = (float[])tensor.floatData.Clone();
                    return true;
                }
                if (tensor.rawData != null && tensor.rawData.Length == checked(count * sizeof(float)))
                {
                    values = new float[count];
                    Buffer.BlockCopy(tensor.rawData, 0, values, 0, tensor.rawData.Length);
                    return BitConverter.IsLittleEndian;
                }
            }
            if (tensor.dataType == TensorDataType.Int32 && TryGetIntValues(tensor, out var integers))
            {
                values = new float[integers.Length];
                for (var i = 0; i < integers.Length; i++) values[i] = integers[i];
                return true;
            }
            return false;
        }

        private static bool TryGetScalarFloat(OnnxTensor tensor, out float value)
        {
            value = 0f;
            if (!TryGetFloatValues(tensor, out var values) || values.Length != 1) return false;
            value = values[0];
            return true;
        }

        private static bool TryResolveRFloatLogicalDtype(OnnxTensor tensor, out string logicalDtype)
        {
            logicalDtype = null;
            if (tensor == null)
                return false;
            if (tensor.onnxDataType == 1 || tensor.onnxDataType == 0 && tensor.dataType == TensorDataType.Float32)
            {
                logicalDtype = "Float32";
                return true;
            }
            if ((tensor.onnxDataType == 6 || tensor.onnxDataType == 7 || tensor.onnxDataType == 9
                    || tensor.onnxDataType == 0 && tensor.dataType == TensorDataType.Int32)
                && TryGetIntValues(tensor, out var values)
                && values.All(value => value >= int.MinValue && value <= int.MaxValue && Math.Abs((double)value) <= 16777216d))
            {
                logicalDtype = "Int32";
                return true;
            }
            return false;
        }

        private static bool IsStrictFloat32Descriptor(AexisOnnxTensorDescriptor tensor)
        {
            return tensor != null
                && tensor.dataType == TensorDataType.Float32
                && (tensor.onnxDataType == 0 || tensor.onnxDataType == 1);
        }

        private static bool IsStrictInt32IndexDescriptor(AexisOnnxTensorDescriptor tensor)
        {
            return tensor != null
                && tensor.dataType == TensorDataType.Int32
                && (tensor.onnxDataType == 0
                    || tensor.onnxDataType == 6
                    || tensor.onnxDataType == 7 && tensor.isInitializer);
        }

        private static TensorDataType FromOnnxDataType(int dataType)
        {
            switch (dataType)
            {
                case 1: return TensorDataType.Float32;
                case 10: return TensorDataType.Float16;
                case 3: return TensorDataType.Int8;
                case 2: return TensorDataType.UInt8;
                case 6:
                case 7:
                case 9: return TensorDataType.Int32;
                default: return TensorDataType.Unknown;
            }
        }

        private static int ToNcnnCastType(TensorDataType dataType, int onnxDataType = 0)
        {
            // Internal logical Bool code. Values remain physically float-backed
            // but every lane is canonicalized to exact 0 or 1.
            if (onnxDataType == 9)
                return 7;
            switch (dataType)
            {
                case TensorDataType.Float32: return 1;
                case TensorDataType.Float16: return 2;
                case TensorDataType.Int8: return 3;
                case TensorDataType.Int32: return 5;
                case TensorDataType.UInt8: return 6;
                default: return 0;
            }
        }

        private static int ResolveGemmBroadcastType(long[] shape, long m, long n)
        {
            if (shape == null || shape.Length == 0) return 0;
            if (shape.Length == 1 && shape[0] == n) return 4;
            if (shape.Length == 2 && shape[0] == 1 && shape[1] == n) return 4;
            if (shape.Length == 2 && m > 0 && shape[0] == m && shape[1] == 1) return 2;
            if (shape.Length == 2 && m > 0 && shape[0] == m && shape[1] == n) return 3;
            return -2;
        }

        private static int NormalizeAxis(int axis, int rank)
        {
            if (rank <= 0) return axis;
            if (axis < 0) axis += rank;
            return axis;
        }

        private static int ResolveSoftmaxDefaultAxis(int opset)
        {
            return opset > 0 && opset < 13 ? 1 : -1;
        }

        private static bool IsLegacyAxisActivation(string opType, int opset)
        {
            return opset > 0 && opset < 13
                && (string.Equals(opType, "Softmax", StringComparison.Ordinal)
                    || string.Equals(opType, "LogSoftmax", StringComparison.Ordinal)
                    || string.Equals(opType, "Hardmax", StringComparison.Ordinal));
        }

        private static bool TryRewriteLegacyAxisActivation(
            OnnxNode node,
            AexisGraphModel.Layer activation,
            IReadOnlyList<AexisOnnxTensorDescriptor> inputs,
            int nodeIndex,
            List<AexisOnnxLoweringDiagnostic> diagnostics,
            out AexisGraphModel.Layer[] layers)
        {
            layers = Array.Empty<AexisGraphModel.Layer>();
            var input = inputs != null && inputs.Count > 0 ? inputs[0] : null;
            if (input?.shape == null || HasDynamic(input.shape) || input.shape.Length == 0
                || input.runtimeShape == null || input.runtimeShape.Length == 0
                || node.outputs.Count != 1 || activation.bottomNames.Length == 0 || activation.topNames.Length != 1)
            {
                diagnostics.Add(Diagnostic(nodeIndex, activation.name, node.opType, "unsupported-legacy-axis-activation-shape",
                    node.opType + " opset 1-12 requires a single static input/output so Aexis can preserve the legacy flatten-from-axis contract.",
                    "Provide static shape metadata and one output, or export with ONNX opset 13 or newer.", true));
                return false;
            }

            var axis = GetInt(node, "axis", 1);
            axis = NormalizeAxis(axis, input.shape.Length);
            if (axis < 0 || axis >= input.shape.Length)
                return false;

            long outer = 1;
            long inner = 1;
            for (var dimension = 0; dimension < input.shape.Length; dimension++)
            {
                var extent = input.shape[dimension];
                if (extent <= 0 || (dimension < axis ? outer : inner) > int.MaxValue / extent)
                {
                    diagnostics.Add(Diagnostic(nodeIndex, activation.name, node.opType, "legacy-axis-activation-size-overflow",
                        node.opType + " legacy flattened extent cannot be represented by the static texture profile.",
                        "Use smaller static extents or export with ONNX opset 13 or newer.", true));
                    return false;
                }
                if (dimension < axis) outer *= extent;
                else inner *= extent;
            }

            var prefix = "__aexis_legacy_axis_" + nodeIndex.ToString(CultureInfo.InvariantCulture) + "_" + SanitizeName(activation.name);
            var flattenedInput = prefix + "_input";
            var flattenedOutput = prefix + "_output";
            var originalInput = activation.bottomNames[0];
            var originalOutput = activation.topNames[0];

            var flatten = CreateLayer("Reshape", prefix + "_flatten", new[] { originalInput }, new[] { flattenedInput });
            SetNcnnShapeParams(flatten, new[] { outer, inner });
            flatten.stringParams["onnx.synthetic"] = "legacy-axis-flatten";

            activation.bottomNames = new[] { flattenedInput };
            activation.topNames = new[] { flattenedOutput };
            activation.bottoms = 1;
            activation.tops = 1;
            activation.intParams[0] = "1";
            activation.stringParams["axis"] = "1";
            activation.stringParams["onnx.legacy_flatten_shape"] = outer.ToString(CultureInfo.InvariantCulture)
                + "," + inner.ToString(CultureInfo.InvariantCulture);

            var restore = CreateLayer("Reshape", prefix + "_restore", new[] { flattenedOutput }, new[] { originalOutput });
            SetNcnnShapeParams(restore, input.runtimeShape);
            restore.stringParams["onnx.synthetic"] = "legacy-axis-restore";
            layers = new[] { flatten, activation, restore };
            return true;
        }

        private static long[] ReverseAxes(int rank)
        {
            var axes = new long[Math.Max(0, rank)];
            for (var i = 0; i < axes.Length; i++) axes[i] = rank - i - 1;
            return axes;
        }

        private static bool TryGetNcnnPermuteOrder(int rank, long[] onnxPerm, out int order)
        {
            order = -1;
            if (rank < 2 || rank > 4 || onnxPerm == null || onnxPerm.Length != rank) return false;
            var internalAxes = new int[rank];
            var seen = new bool[rank];
            for (var i = 0; i < rank; i++)
            {
                var onnxSource = onnxPerm[rank - i - 1];
                if (onnxSource < 0 || onnxSource >= rank) return false;
                var axis = rank - (int)onnxSource - 1;
                if (seen[axis]) return false;
                seen[axis] = true;
                internalAxes[i] = axis;
            }
            var candidate = new int[rank];
            var used = new bool[rank];
            var current = 0;
            var foundOrder = -1;
            bool Visit(int depth)
            {
                if (depth == rank)
                {
                    var match = true;
                    for (var i = 0; i < rank; i++) match &= candidate[i] == internalAxes[i];
                    if (match) { foundOrder = current; return true; }
                    current++;
                    return false;
                }
                for (var axis = 0; axis < rank; axis++)
                {
                    if (used[axis]) continue;
                    used[axis] = true;
                    candidate[depth] = axis;
                    if (Visit(depth + 1)) return true;
                    used[axis] = false;
                }
                return false;
            }
            var found = Visit(0);
            order = foundOrder;
            return found;
        }

        private static string EncodeIntArray(IList<long> values)
        {
            return values.Count.ToString(CultureInfo.InvariantCulture) + "," + Join(values);
        }

        private static string EncodeFloatArray(IList<float> values)
        {
            return values.Count.ToString(CultureInfo.InvariantCulture) + "," + Join(values);
        }

        private static string SanitizeName(string value)
        {
            if (string.IsNullOrEmpty(value)) return "constant";
            var chars = value.ToCharArray();
            for (var i = 0; i < chars.Length; i++)
                if (!char.IsLetterOrDigit(chars[i]) && chars[i] != '_') chars[i] = '_';
            return new string(chars);
        }

        private static long[] GetInts(OnnxNode node, string name, long[] fallback)
        {
            if (node?.attributes != null && node.attributes.TryGetValue(name, out var attribute) && attribute?.ints != null && attribute.ints.Count > 0)
                return attribute.ints.ToArray();
            return fallback;
        }

        private static string GetString(OnnxNode node, string name)
        {
            return node?.attributes != null && node.attributes.TryGetValue(name, out var attribute) && attribute?.s != null
                ? attribute.GetUtf8String()
                : string.Empty;
        }

        private static float GetFloat(OnnxNode node, string name, float fallback)
        {
            return node?.attributes != null && node.attributes.TryGetValue(name, out var attribute) && attribute != null && attribute.type == 1
                ? attribute.f
                : fallback;
        }

        private static long ElementCount(long[] shape)
        {
            long count = 1;
            if (shape == null) return 0;
            for (var i = 0; i < shape.Length; i++)
            {
                if (shape[i] <= 0 || count > long.MaxValue / shape[i]) return 0;
                count *= shape[i];
            }
            return count;
        }

        private static bool ConfigureBoundedDataIndexNode(
            OnnxNode node,
            AexisGraphModel.Layer layer,
            List<AexisOnnxTensorDescriptor> inputs,
            IReadOnlyList<OnnxNode> graphNodes,
            AexisOnnxGraphLoweringOptions options,
            int index,
            List<AexisOnnxLoweringDiagnostic> diagnostics)
        {
            if (string.Equals(node.opType, "NonZero", StringComparison.Ordinal)
                || string.Equals(node.opType, "Compress", StringComparison.Ordinal))
            {
                if (!options.enableBoundedDataIndexLowering)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "bounded-data-index-profile-required",
                        node.opType + " has a data-dependent standard ONNX result and needs the Aexis bounded LinearMat profile.",
                        "Enable bounded data-index lowering and supply a fixed output capacity; consumers must use the generated GPU count output.", true));
                    return false;
                }

                if (inputs.Count == 0 || HasDynamic(inputs[0].shape) || inputs[0].shape.Length != 1)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-data-index-shape",
                        node.opType + " bounded texture lowering currently requires a static rank-1 input.",
                        "Lower to a rank-1 linear profile or keep this node outside strict Pack4 execution.", true));
                    return false;
                }
                if (ResolveOnnxDataType(inputs[0]) != 1)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-data-index-dtype",
                        node.opType + " bounded P0 texture lowering requires Float32 data in RFloat storage.",
                        "Insert an explicit Cast to Float32 before this node.", true));
                    return false;
                }
                if (node.outputs.Count == 0 || string.IsNullOrWhiteSpace(node.outputs[0]))
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "missing-data-index-output",
                        node.opType + " requires a named value output so the bounded value/count contract can be published.",
                        "Re-export the node with a named output.", true));
                    return false;
                }

                var capacity = ResolveCapacity(options, layer.name, node.outputs.Count > 0 ? node.outputs[0] : string.Empty, inputs[0].shape);
                if (capacity <= 0)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "missing-output-capacity",
                        node.opType + " needs a positive fixed output capacity for texture-native compaction.",
                        "Declare outputCapacities by node or output name before lowering.", true));
                    return false;
                }
                var maximum = ElementCount(inputs[0].shape);
                if (maximum <= 0 || capacity < maximum)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "insufficient-output-capacity",
                        node.opType + " capacity " + capacity.ToString(CultureInfo.InvariantCulture) + " is smaller than the static maximum " + maximum.ToString(CultureInfo.InvariantCulture) + ".",
                        "Use capacity >= the input element count so strict execution cannot truncate results.", true));
                    return false;
                }

                if (string.Equals(node.opType, "Compress", StringComparison.Ordinal)
                    && (inputs.Count != 2 || HasDynamic(inputs[1].shape) || inputs[1].shape.Length != 1 || inputs[1].shape[0] != inputs[0].shape[0]
                        || (ResolveOnnxDataType(inputs[1]) != 9 && ResolveOnnxDataType(inputs[1]) != 6)))
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-compress-condition",
                        "Compress bounded texture lowering requires a static rank-1 Bool/Int32 condition with the same element count as Float32 data.",
                        "Lower to matching rank-1 data and canonical Bool/Int32 condition tensors before import.", true));
                    return false;
                }
                if (string.Equals(node.opType, "Compress", StringComparison.Ordinal))
                {
                    var axis = GetInt(node, "axis", 0);
                    if (axis != 0 && axis != -1)
                    {
                        diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-compress-axis",
                            "Rank-1 Compress accepts only axis 0, axis -1, or an omitted axis in the bounded P0 profile.",
                            "Normalize the Compress axis to 0 before import.", true));
                        return false;
                    }
                    layer.intParams[0] = "0";
                }

                var outputName = node.outputs.Count > 0 ? node.outputs[0] : string.Empty;
                var consumer = FindTensorConsumer(graphNodes, outputName, index);
                if (consumer != null)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "bounded-data-index-consumer-requires-count",
                        node.opType + " bounded output " + outputName + " is consumed by " + NodeName(consumer.Value.node, consumer.Value.index)
                        + " without the generated GPU count tensor.",
                        "Keep bounded NonZero/Compress terminal in the P0 graph, or introduce an explicit count-aware texture-native consumer contract.", true));
                    return false;
                }

                layer.stringParams["capacity"] = capacity.ToString(CultureInfo.InvariantCulture);
                layer.intParams[30] = layer.stringParams["capacity"];
                var countName = layer.topNames.Length > 0 ? layer.topNames[0] + ".count" : layer.name + ".count";
                layer.topNames = new[] { layer.topNames[0], countName };
                layer.tops = 2;
                return true;
            }

            if (string.Equals(node.opType, "GatherND", StringComparison.Ordinal))
            {
                var batchDims = GetInt(node, "batch_dims", 0);
                layer.stringParams["batch_dims"] = batchDims.ToString(CultureInfo.InvariantCulture);
                layer.intParams[0] = layer.stringParams["batch_dims"];
                layer.stringParams["index_depth"] = "1";
                layer.intParams[1] = "1";
                if (batchDims != 0 || inputs.Count < 2 || HasDynamic(inputs[0].shape) || inputs[0].shape.Length != 1
                    || HasDynamic(inputs[1].shape) || inputs[1].shape.Length != 2 || inputs[1].shape[1] != 1
                    || !IsStrictInt32IndexDescriptor(inputs[1]) || !HasVerifiedNodeProof(options.verifiedInRangeIndexNodes, node, graphNodes, index))
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-gathernd-profile",
                        "GatherND texture lowering requires static rank-1 data, batch_dims=0, static rank-2 [N,1] Int32 indices, and exporter proof that every index is in range.",
                        "Add the node to verifiedInRangeIndexNodes only after proving the index bounds.", true));
                    return false;
                }
                layer.stringParams["index_dtype"] = "Int32";
                layer.stringParams["indices_in_range"] = "1";
                return true;
            }

            if (string.Equals(node.opType, "Gather", StringComparison.Ordinal)
                || string.Equals(node.opType, "GatherElements", StringComparison.Ordinal))
            {
                var gatherElements = string.Equals(node.opType, "GatherElements", StringComparison.Ordinal);
                var dataShape = inputs.Count > 0 ? inputs[0].runtimeShape : null;
                var indicesShape = inputs.Count > 1 ? inputs[1].runtimeShape : null;
                var axis = layer.GetInt(0, 0);
                var indexTypeSupported = inputs.Count > 1 && IsStrictInt32IndexDescriptor(inputs[1]);
                var valid = inputs.Count == 2
                    && dataShape != null && indicesShape != null
                    && !HasDynamic(dataShape) && !HasDynamic(indicesShape)
                    && dataShape.Length >= 1 && dataShape.Length <= 4
                    && indicesShape.Length >= 1 && indicesShape.Length <= 4
                    && axis >= 0 && axis < dataShape.Length
                    && indexTypeSupported
                    && HasVerifiedNodeProof(options.verifiedInRangeIndexNodes, node, graphNodes, index);
                if (valid && gatherElements)
                {
                    valid = dataShape.Length == indicesShape.Length;
                    for (var dimension = 0; valid && dimension < dataShape.Length; dimension++)
                        if (dimension != axis && indicesShape[dimension] > dataShape[dimension])
                            valid = false;
                }
                if (valid && !gatherElements)
                    valid = dataShape.Length + indicesShape.Length - 1 <= 4;
                if (!valid)
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-gather-profile",
                        gatherElements
                            ? "GatherElements texture lowering requires static rank=1..4 equal-rank data/Int32 indices, an in-range proof, and indices dimensions no larger than data outside the axis."
                            : "Gather texture lowering requires static rank=1..4 data/Int32 indices, output rank<=4, a valid static axis, and an exporter proof that every index is in range.",
                        "Add the node to verifiedInRangeIndexNodes only after proving index bounds; narrow dynamic INT64 indices explicitly before import.", true));
                    return false;
                }
                layer.stringParams["index_dtype"] = "Int32";
                layer.stringParams["indices_in_range"] = "1";
                return true;
            }

            if (string.Equals(node.opType, "Scatter", StringComparison.Ordinal)
                || string.Equals(node.opType, "ScatterElements", StringComparison.Ordinal)
                || string.Equals(node.opType, "ScatterND", StringComparison.Ordinal))
            {
                var scatterNd = string.Equals(node.opType, "ScatterND", StringComparison.Ordinal);
                var axis = GetInt(node, "axis", 0);
                var validShapes = inputs.Count == 3 && !HasDynamic(inputs[0].shape) && inputs[0].shape.Length == 1
                    && !HasDynamic(inputs[1].shape) && !HasDynamic(inputs[2].shape) && inputs[2].shape.Length == 1
                    && (scatterNd
                        ? inputs[1].shape.Length == 2 && inputs[1].shape[1] == 1 && inputs[1].shape[0] == inputs[2].shape[0]
                        : inputs[1].shape.Length == 1 && inputs[1].shape[0] == inputs[2].shape[0] && axis == 0);
                if (!validShapes
                    || !IsStrictInt32IndexDescriptor(inputs[1])
                    || !HasVerifiedNodeProof(options.verifiedUniqueScatterNodes, node, graphNodes, index)
                    || !HasVerifiedNodeProof(options.verifiedInRangeIndexNodes, node, graphNodes, index)
                    || !string.Equals(GetString(node, "reduction"), string.Empty, StringComparison.Ordinal)
                        && !string.Equals(GetString(node, "reduction"), "none", StringComparison.Ordinal))
                {
                    diagnostics.Add(Diagnostic(index, layer.name, node.opType, "unsupported-scatter-profile",
                        scatterNd
                            ? "ScatterND texture lowering requires rank-1 data/updates, static rank-2 [N,1] Int32 indices, reduction=none, and exporter proof that indices are unique and in range."
                            : "Scatter/ScatterElements texture lowering requires rank-1 data/updates, axis=0, static rank-1 Int32 indices, reduction=none, and exporter proof that indices are unique and in range.",
                        "Supply verifiedUniqueScatterNodes and verifiedInRangeIndexNodes only after proving both contracts.", true));
                    return false;
                }
                layer.stringParams["unique_indices"] = "1";
                layer.stringParams["indices_in_range"] = "1";
                layer.stringParams["index_dtype"] = "Int32";
                layer.stringParams["reduction"] = "none";
                layer.stringParams["index_depth"] = "1";
                layer.intParams[1] = "1";
                layer.intParams[-1] = "1";
            }
            return true;
        }

        private static (OnnxNode node, int index)? FindTensorConsumer(IReadOnlyList<OnnxNode> graphNodes, string tensorName, int producerIndex)
        {
            if (graphNodes == null || string.IsNullOrEmpty(tensorName))
                return null;
            for (var nodeIndex = Math.Max(0, producerIndex + 1); nodeIndex < graphNodes.Count; nodeIndex++)
            {
                var candidate = graphNodes[nodeIndex];
                if (candidate?.inputs == null)
                    continue;
                for (var inputIndex = 0; inputIndex < candidate.inputs.Count; inputIndex++)
                {
                    if (string.Equals(candidate.inputs[inputIndex], tensorName, StringComparison.Ordinal))
                        return (candidate, nodeIndex);
                }
            }
            return null;
        }

        private static int ResolveCapacity(AexisOnnxGraphLoweringOptions options, string nodeName, string outputName, long[] inputShape)
        {
            if (options.outputCapacities != null)
            {
                if (!string.IsNullOrEmpty(outputName) && options.outputCapacities.TryGetValue(outputName, out var outputCapacity)) return outputCapacity;
                if (!string.IsNullOrEmpty(nodeName) && options.outputCapacities.TryGetValue(nodeName, out var nodeCapacity)) return nodeCapacity;
            }
            if (inputShape == null || inputShape.Length != 1 || inputShape[0] <= 0 || inputShape[0] > int.MaxValue) return 0;
            return (int)inputShape[0];
        }

        private static bool Contains(string[] values, string value)
        {
            if (values == null) return false;
            for (var i = 0; i < values.Length; i++) if (string.Equals(values[i], value, StringComparison.Ordinal)) return true;
            return false;
        }

        private static bool HasVerifiedNodeProof(
            string[] values,
            OnnxNode node,
            IReadOnlyList<OnnxNode> graphNodes,
            int nodeIndex)
        {
            if (values == null || node == null)
                return false;

            var nodeName = NodeName(node, nodeIndex);
            var fingerprint = nodeIndex.ToString(CultureInfo.InvariantCulture)
                + ":" + nodeName
                + "|" + (node.opType ?? string.Empty)
                + "|" + string.Join(",", node.inputs)
                + "->" + string.Join(",", node.outputs);
            if (Contains(values, fingerprint))
                return true;

            var matchingNames = 0;
            for (var index = 0; graphNodes != null && index < graphNodes.Count; index++)
            {
                var candidate = graphNodes[index];
                if (candidate != null && string.Equals(NodeName(candidate, index), nodeName, StringComparison.Ordinal))
                    matchingNames++;
            }
            return matchingNames == 1 && Contains(values, nodeName);
        }

        private static List<AexisOnnxTensorDescriptor> ResolveInputs(OnnxNode node, Dictionary<string, AexisOnnxTensorDescriptor> descriptors, int index, List<AexisOnnxLoweringDiagnostic> diagnostics)
        {
            var result = new List<AexisOnnxTensorDescriptor>();
            foreach (var input in node.inputs)
            {
                if (string.IsNullOrEmpty(input)) continue;
                if (!descriptors.TryGetValue(input, out var descriptor))
                {
                    diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "missing-input", "Input " + input + " has no ONNX type/shape descriptor or earlier producer.", "Provide ONNX value_info or fix the graph connection.", true));
                    descriptor = new AexisOnnxTensorDescriptor { name = input };
                }
                result.Add(descriptor);
            }
            return result;
        }

        private static void ValidateInputBatchContracts(
            OnnxNode node,
            List<AexisOnnxTensorDescriptor> inputs,
            int index,
            List<AexisOnnxLoweringDiagnostic> diagnostics)
        {
            if (inputs == null || inputs.Count < 2)
                return;
            var isBroadcast = BinaryOps.ContainsKey(node.opType)
                || string.Equals(node.opType, "Where", StringComparison.Ordinal)
                || string.Equals(node.opType, "MatMul", StringComparison.Ordinal)
                || string.Equals(node.opType, "Einsum", StringComparison.Ordinal);
            var isConcat = string.Equals(node.opType, "Concat", StringComparison.Ordinal);
            var isEltwise = string.Equals(node.opType, "Sum", StringComparison.Ordinal)
                || string.Equals(node.opType, "Mean", StringComparison.Ordinal);
            if (!isBroadcast && !isConcat && !isEltwise)
                return;

            AexisOnnxTensorDescriptor batched = null;
            for (var i = 0; i < inputs.Count; i++)
            {
                var candidate = inputs[i];
                if (candidate != null && !candidate.isInitializer && candidate.batchAxis >= 0)
                {
                    batched = candidate;
                    break;
                }
            }
            if (batched == null)
                return;

            for (var i = 0; i < inputs.Count; i++)
            {
                var candidate = inputs[i];
                if (candidate == null || candidate.isInitializer || ReferenceEquals(candidate, batched))
                    continue;
                if (candidate.batchAxis >= 0 && candidate.batchAxis != batched.batchAxis)
                {
                    diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "inconsistent-batch-axis",
                        node.opType + " inputs do not agree on the static singleton batch axis.",
                        "Keep batch=1 at the same ONNX axis for every runtime input.", true));
                }
                else if ((isConcat || isEltwise) && (candidate.shape.Length != batched.shape.Length || candidate.batchAxis != batched.batchAxis))
                {
                    diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "unbatched-concat-input",
                        node.opType + " requires every runtime input to preserve the same static batch=1 contract.",
                        "Add the singleton batch axis consistently before concatenation.", true));
                }
                else if (isBroadcast && candidate.batchAxis < 0 && candidate.shape.Length == batched.shape.Length
                         && batched.batchAxis < candidate.shape.Length && candidate.shape[batched.batchAxis] != 1)
                {
                    diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "batch-broadcast-expansion",
                        node.opType + " would expand the removed batch axis beyond one.",
                        "Keep the broadcast extent on the ONNX batch axis equal to 1.", true));
                }
            }
        }

        private static void InferOutputs(
            OnnxNode node,
            List<AexisOnnxTensorDescriptor> inputs,
            Dictionary<string, AexisOnnxTensorDescriptor> descriptors,
            Dictionary<string, OnnxTensor> initializers,
            AexisOnnxGraphLoweringOptions options,
            int index,
            List<AexisOnnxLoweringDiagnostic> diagnostics)
        {
            var type = inputs.Count > 0 ? inputs[0].dataType : TensorDataType.Unknown;
            var onnxDataType = inputs.Count > 0 ? inputs[0].onnxDataType : 0;
            var shape = inputs.Count > 0 ? Clone(inputs[0].shape) : Array.Empty<long>();
            if (string.Equals(node.opType, "Split", StringComparison.Ordinal))
            {
                InferSplitOutputs(node, inputs, descriptors, initializers, options, index, diagnostics);
                return;
            }
            var boundedDataIndexCapacity = (string.Equals(node.opType, "NonZero", StringComparison.Ordinal) || string.Equals(node.opType, "Compress", StringComparison.Ordinal))
                ? ResolveCapacity(options, NodeName(node, index), node.outputs.Count > 0 ? node.outputs[0] : string.Empty, inputs.Count > 0 ? inputs[0].shape : null)
                : 0;
            if (BinaryOps.ContainsKey(node.opType) && inputs.Count > 1) shape = Broadcast(inputs[0].shape, inputs[1].shape);
            if ((string.Equals(node.opType, "Sum", StringComparison.Ordinal) || string.Equals(node.opType, "Mean", StringComparison.Ordinal)) && inputs.Count > 0)
                shape = Clone(inputs[0].shape);
            if (string.Equals(node.opType, "Where", StringComparison.Ordinal) && inputs.Count >= 3)
            {
                type = inputs[1].dataType;
                onnxDataType = inputs[1].onnxDataType;
                shape = Broadcast(Broadcast(inputs[0].shape, inputs[1].shape), inputs[2].shape);
            }
            if (string.Equals(node.opType, "Concat", StringComparison.Ordinal) && inputs.Count > 0) shape = InferConcat(node, inputs, shape);
            if (string.Equals(node.opType, "ExtractImagePatches", StringComparison.Ordinal) && inputs.Count > 0)
            {
                shape = TryResolveExtractImagePatchesSpec(
                    node, inputs[0].shape, out _, out _, out _, out _, out _, out _,
                    out _, out _, out _, out _, out var patchOutput, out _)
                    ? patchOutput
                    : Dynamic(inputs[0].shape?.Length ?? 4);
            }
            if (string.Equals(node.opType, "Transpose", StringComparison.Ordinal)) shape = InferTranspose(node, shape);
            if (string.Equals(node.opType, "Flatten", StringComparison.Ordinal)) shape = InferFlatten(node, shape);
            if (string.Equals(node.opType, "Conv", StringComparison.Ordinal) || string.Equals(node.opType, "ConvTranspose", StringComparison.Ordinal))
                shape = InferConvolution(node, inputs, initializers, string.Equals(node.opType, "ConvTranspose", StringComparison.Ordinal));
            if (string.Equals(node.opType, "MaxPool", StringComparison.Ordinal) || string.Equals(node.opType, "AveragePool", StringComparison.Ordinal)
                || string.Equals(node.opType, "GlobalAveragePool", StringComparison.Ordinal) || string.Equals(node.opType, "GlobalMaxPool", StringComparison.Ordinal))
                shape = InferPool(node, shape, string.Equals(node.opType, "GlobalAveragePool", StringComparison.Ordinal) || string.Equals(node.opType, "GlobalMaxPool", StringComparison.Ordinal));
            if (string.Equals(node.opType, "Gemm", StringComparison.Ordinal)) shape = InferGemm(node, inputs);
            if ((string.Equals(node.opType, "MatMul", StringComparison.Ordinal) || string.Equals(node.opType, "Einsum", StringComparison.Ordinal)) && inputs.Count > 1) shape = InferMatMul(inputs[0].shape, inputs[1].shape);
            if (string.Equals(node.opType, "Reshape", StringComparison.Ordinal)) shape = InferReshape(node, shape, initializers);
            if (string.Equals(node.opType, "Squeeze", StringComparison.Ordinal)) shape = InferSqueeze(node, shape, initializers);
            if (string.Equals(node.opType, "Unsqueeze", StringComparison.Ordinal)) shape = InferUnsqueeze(node, shape, initializers);
            if (string.Equals(node.opType, "Gather", StringComparison.Ordinal) && inputs.Count > 1) shape = InferGather(node, inputs[0].shape, inputs[1].shape);
            if (string.Equals(node.opType, "GatherElements", StringComparison.Ordinal) && inputs.Count > 1) shape = Clone(inputs[1].shape);
            if (string.Equals(node.opType, "GatherND", StringComparison.Ordinal) && inputs.Count > 1) shape = InferGatherND(node, inputs[0].shape, inputs[1].shape);
            if (string.Equals(node.opType, "Scatter", StringComparison.Ordinal) || string.Equals(node.opType, "ScatterElements", StringComparison.Ordinal) || string.Equals(node.opType, "ScatterND", StringComparison.Ordinal)) shape = inputs.Count > 0 ? Clone(inputs[0].shape) : shape;
            if (node.opType.StartsWith("Reduce", StringComparison.Ordinal)) shape = InferReduction(node, shape, initializers);
            if (string.Equals(node.opType, "ArgMax", StringComparison.Ordinal) || string.Equals(node.opType, "ArgMin", StringComparison.Ordinal))
            {
                type = TensorDataType.Int32;
                onnxDataType = 6;
                shape = InferAxisReduction(node, shape);
            }
            if (string.Equals(node.opType, "Cast", StringComparison.Ordinal))
            {
                onnxDataType = GetInt(node, "to", 0);
                type = FromOnnxDataType(onnxDataType);
            }
            if (string.Equals(node.opType, "CastLike", StringComparison.Ordinal) && inputs.Count > 1)
            {
                type = inputs[1].dataType;
                onnxDataType = inputs[1].onnxDataType;
            }
            if (string.Equals(node.opType, "Shape", StringComparison.Ordinal)) { type = TensorDataType.Int32; onnxDataType = 6; shape = InferShapeResult(node, shape.Length); }
            if (string.Equals(node.opType, "Size", StringComparison.Ordinal)) { type = TensorDataType.Int32; onnxDataType = 6; shape = Array.Empty<long>(); }
            if (string.Equals(node.opType, "NonZero", StringComparison.Ordinal)) { type = TensorDataType.Int32; onnxDataType = 6; shape = boundedDataIndexCapacity > 0 ? new long[] { 1, boundedDataIndexCapacity } : new long[] { shape.Length, -1 }; }
            if (string.Equals(node.opType, "Compress", StringComparison.Ordinal) && boundedDataIndexCapacity > 0) shape = new long[] { boundedDataIndexCapacity };
            if (string.Equals(node.opType, "TopK", StringComparison.Ordinal)) shape = InferTopK(node, shape, initializers);
            if (string.Equals(node.opType, "Expand", StringComparison.Ordinal)) shape = InferStaticShapeInput(node, 1, initializers, shape.Length);
            if (string.Equals(node.opType, "ConstantOfShape", StringComparison.Ordinal)) shape = InferStaticShapeInput(node, 0, initializers, 1);
            if (string.Equals(node.opType, "Tile", StringComparison.Ordinal) && TryGetInitializerInput(node, 1, initializers, out var repeatsTensor) && TryGetIntValues(repeatsTensor, out var repeats)) shape = InferTile(shape, repeats);
            if (string.Equals(node.opType, "Slice", StringComparison.Ordinal)) shape = InferSlice(node, shape, initializers);
            if (string.Equals(node.opType, "Range", StringComparison.Ordinal)) shape = InferRange(node, initializers);
            if (string.Equals(node.opType, "OneHot", StringComparison.Ordinal)) shape = InferOneHot(node, shape, initializers);
            if (string.Equals(node.opType, "Pad", StringComparison.Ordinal)) shape = InferPad(node, shape, initializers);
            if (string.Equals(node.opType, "Resize", StringComparison.Ordinal) || string.Equals(node.opType, "Upsample", StringComparison.Ordinal)) shape = InferResize(node, shape, initializers);
            if (string.Equals(node.opType, "DepthToSpace", StringComparison.Ordinal)) shape = InferDepthToSpace(node, shape);
            if (string.Equals(node.opType, "SpaceToDepth", StringComparison.Ordinal)) shape = InferSpaceToDepth(node, shape);
            if (string.Equals(node.opType, "Equal", StringComparison.Ordinal)
                || string.Equals(node.opType, "Greater", StringComparison.Ordinal)
                || string.Equals(node.opType, "GreaterOrEqual", StringComparison.Ordinal)
                || string.Equals(node.opType, "Less", StringComparison.Ordinal)
                || string.Equals(node.opType, "LessOrEqual", StringComparison.Ordinal)
                || string.Equals(node.opType, "And", StringComparison.Ordinal)
                || string.Equals(node.opType, "Or", StringComparison.Ordinal)
                || string.Equals(node.opType, "Xor", StringComparison.Ordinal)
                || string.Equals(node.opType, "Not", StringComparison.Ordinal))
            {
                type = TensorDataType.Int32;
                onnxDataType = 9;
            }
            if (string.Equals(node.opType, "IsInf", StringComparison.Ordinal)
                || string.Equals(node.opType, "IsNaN", StringComparison.Ordinal))
            {
                type = TensorDataType.Int32;
                onnxDataType = 9;
            }
            if (string.Equals(node.opType, "OneHot", StringComparison.Ordinal)
                && TryGetInitializerInput(node, 2, initializers, out var oneHotValues))
            {
                type = oneHotValues.dataType;
                onnxDataType = oneHotValues.onnxDataType;
            }
            if (string.Equals(node.opType, "ConstantOfShape", StringComparison.Ordinal))
            {
                type = TensorDataType.Float32;
                if (node.attributes.TryGetValue("value", out var valueAttribute) && valueAttribute?.tensor != null)
                {
                    type = valueAttribute.tensor.dataType;
                    onnxDataType = valueAttribute.tensor.onnxDataType;
                }
            }

            if (options.rejectDynamicShapes && HasDynamic(shape))
                diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "dynamic-shape-not-static", "Output shape cannot be proven statically for " + node.opType + ".", "Supply a capacity-bounded GPU shape/index contract or lower this dynamic node before strict import.", true));
            if (shape.Length > 4)
                diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "unsupported-output-rank",
                    "Output rank exceeds the P0 rank-4 texture descriptor contract.", "Lower or flatten this output to rank <= 4.", true));
            if (HasZeroExtent(shape))
                diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "empty-output-tensor",
                    "Output contains a zero extent with no P0 texture storage contract.", "Remove empty tensor branches before import.", true));
            var batchAxis = InferOutputBatchAxis(node, inputs, shape, initializers, index, diagnostics);
            foreach (var output in node.outputs)
            {
                if (string.IsNullOrEmpty(output)) continue;
                Register(descriptors, output, type, shape, false, onnxDataType);
                SetBatchContract(descriptors[output], batchAxis);
            }
            if (boundedDataIndexCapacity > 0 && node.outputs.Count > 0)
            {
                var output = node.outputs[0];
                Register(descriptors, output, type, string.Equals(node.opType, "NonZero", StringComparison.Ordinal)
                    ? new long[] { 1, boundedDataIndexCapacity }
                    : new long[] { boundedDataIndexCapacity }, false);
                Register(descriptors, output + ".count", TensorDataType.Int32, new long[] { 1 }, false);
            }
            if (string.Equals(node.opType, "TopK", StringComparison.Ordinal) && node.outputs.Count > 1)
            {
                Register(descriptors, node.outputs[1], TensorDataType.Int32, shape, false, 6);
                SetBatchContract(descriptors[node.outputs[1]], batchAxis);
            }
        }

        private static void InferSplitOutputs(
            OnnxNode node,
            List<AexisOnnxTensorDescriptor> inputs,
            Dictionary<string, AexisOnnxTensorDescriptor> descriptors,
            Dictionary<string, OnnxTensor> initializers,
            AexisOnnxGraphLoweringOptions options,
            int index,
            List<AexisOnnxLoweringDiagnostic> diagnostics)
        {
            var input = inputs.Count > 0 ? inputs[0] : null;
            var inputShape = input?.shape ?? Array.Empty<long>();
            var axis = NormalizeAxis(GetInt(node, "axis", 0), inputShape.Length);
            long[] split = GetInts(node, "split", null);
            if ((split == null || split.Length == 0)
                && TryGetInitializerInput(node, 1, initializers, out var splitTensor))
                TryGetIntValues(splitTensor, out split);
            if ((split == null || split.Length == 0)
                && axis >= 0 && axis < inputShape.Length && node.outputs.Count > 0
                && inputShape[axis] > 0 && inputShape[axis] % node.outputs.Count == 0)
            {
                split = new long[node.outputs.Count];
                for (var i = 0; i < split.Length; i++) split[i] = inputShape[axis] / node.outputs.Count;
            }

            var valid = axis >= 0 && axis < inputShape.Length
                && split != null && split.Length == node.outputs.Count;
            if (valid)
            {
                long total = 0;
                for (var i = 0; i < split.Length; i++)
                {
                    valid &= split[i] > 0;
                    total += split[i];
                }
                valid &= inputShape[axis] > 0 && total == inputShape[axis];
            }

            for (var outputIndex = 0; outputIndex < node.outputs.Count; outputIndex++)
            {
                var outputName = node.outputs[outputIndex];
                if (string.IsNullOrEmpty(outputName)) continue;
                var outputShape = valid ? Clone(inputShape) : Dynamic(inputShape.Length);
                if (valid) outputShape[axis] = split[outputIndex];
                if (options.rejectDynamicShapes && HasDynamic(outputShape))
                    diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "dynamic-split-shape",
                        "Split output " + outputName + " cannot be proven statically.", "Provide valid static split sizes and input value_info.", true));
                if (outputShape.Length > 4 || HasZeroExtent(outputShape))
                    diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "unsupported-split-output",
                        "Split output " + outputName + " exceeds rank 4 or contains an empty extent.",
                        "Use positive static split sizes with rank <= 4.", true));
                Register(descriptors, outputName, input?.dataType ?? TensorDataType.Unknown, outputShape, false, input?.onnxDataType ?? 0);
                SetBatchContract(descriptors[outputName], input?.batchAxis ?? -1);
            }
        }

        private static void LowerConstant(
            int index,
            OnnxNode node,
            Dictionary<string, AexisOnnxTensorDescriptor> descriptors,
            Dictionary<string, OnnxTensor> immutableTensors,
            List<AexisOnnxLoweringDiagnostic> diagnostics)
        {
            if (node.outputs.Count != 1 || string.IsNullOrEmpty(node.outputs[0]))
            {
                diagnostics.Add(Diagnostic(index, NodeName(node, index), "Constant", "invalid-constant-output",
                    "Constant requires exactly one named output.", "Repair the ONNX graph output list.", true));
                return;
            }

            OnnxTensor tensor = null;
            if (node.attributes.TryGetValue("value", out var value) && value?.tensor != null)
            {
                tensor = value.tensor;
            }
            else if (node.attributes.TryGetValue("value_float", out var scalarFloat))
            {
                tensor = new OnnxTensor
                {
                    dataType = TensorDataType.Float32,
                    onnxDataType = 1,
                    dims = Array.Empty<long>(),
                    floatData = new[] { scalarFloat.f }
                };
            }
            else if (node.attributes.TryGetValue("value_int", out var scalarInt))
            {
                if (scalarInt.i < int.MinValue || scalarInt.i > int.MaxValue)
                {
                    diagnostics.Add(Diagnostic(index, NodeName(node, index), "Constant", "constant-int64-overflow",
                        "Constant value_int is outside the Aexis Int32 texture/index range.", "Cast or clamp this constant to Int32 before import.", true));
                    return;
                }
                tensor = new OnnxTensor
                {
                    dataType = TensorDataType.Int32,
                    onnxDataType = 6,
                    dims = Array.Empty<long>(),
                    int32Data = new[] { (int)scalarInt.i }
                };
            }
            else if (node.attributes.TryGetValue("value_floats", out var floatList) && floatList.floats.Count > 0)
            {
                tensor = new OnnxTensor
                {
                    dataType = TensorDataType.Float32,
                    onnxDataType = 1,
                    dims = new long[] { floatList.floats.Count },
                    floatData = floatList.floats.ToArray()
                };
            }
            else if (node.attributes.TryGetValue("value_ints", out var intList) && intList.ints.Count > 0)
            {
                var values = new int[intList.ints.Count];
                for (var i = 0; i < values.Length; i++)
                {
                    if (intList.ints[i] < int.MinValue || intList.ints[i] > int.MaxValue)
                    {
                        diagnostics.Add(Diagnostic(index, NodeName(node, index), "Constant", "constant-int64-overflow",
                            "Constant value_ints contains a value outside the Aexis Int32 texture/index range.", "Cast or clamp this constant to Int32 before import.", true));
                        return;
                    }
                    values[i] = (int)intList.ints[i];
                }
                tensor = new OnnxTensor
                {
                    dataType = TensorDataType.Int32,
                    onnxDataType = 6,
                    dims = new long[] { values.Length },
                    int32Data = values
                };
            }

            if (tensor == null)
            {
                diagnostics.Add(Diagnostic(index, NodeName(node, index), "Constant", "unsupported-constant",
                    "Constant requires a non-empty value, value_float, value_int, value_floats, or value_ints attribute in the P0 texture profile.",
                    "Fold string, sparse, or empty constants before Aexis import.", true));
                return;
            }

            tensor.name = node.outputs[0];
            immutableTensors[node.outputs[0]] = tensor;
            Register(descriptors, node.outputs[0], tensor.dataType, tensor.dims, true, tensor.onnxDataType);
        }

        private static bool TryLowerStaticDimensionNode(
            int index,
            OnnxNode node,
            Dictionary<string, AexisOnnxTensorDescriptor> descriptors,
            Dictionary<string, OnnxTensor> immutableTensors,
            List<AexisOnnxLoweringDiagnostic> diagnostics)
        {
            if (node.inputs.Count < 1 || node.outputs.Count != 1 || string.IsNullOrEmpty(node.outputs[0]))
                return false;
            if (!descriptors.TryGetValue(node.inputs[0], out var input) || input.shape == null || HasDynamic(input.shape))
            {
                diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "dynamic-dimension-query",
                    node.opType + " requires a fully static input shape because Aexis does not read texture dimensions back to CPU during execution.",
                    "Provide static ONNX value_info so the importer can constant-fold this node.", true));
                return true;
            }

            int[] values;
            long[] tensorShape;
            if (string.Equals(node.opType, "Size", StringComparison.Ordinal))
            {
                var count = ElementCount(input.shape);
                if (count < 0 || count > int.MaxValue)
                {
                    diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "dimension-query-overflow",
                        "Size exceeds the Aexis Int32 shape/index range.", "Use a tensor with at most Int32.MaxValue elements.", true));
                    return true;
                }
                values = new[] { (int)count };
                tensorShape = Array.Empty<long>();
            }
            else
            {
                var rank = input.shape.Length;
                var start = GetInt(node, "start", 0);
                var end = GetInt(node, "end", rank);
                if (start < 0) start += rank;
                if (end < 0) end += rank;
                start = Math.Max(0, Math.Min(rank, start));
                end = Math.Max(start, Math.Min(rank, end));
                values = new int[end - start];
                if (values.Length == 0)
                {
                    diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "empty-shape-result",
                        "Shape produces an empty tensor, which has no P0 texture storage contract.",
                        "Fold or remove the empty Shape result before import.", true));
                    return true;
                }
                for (var i = 0; i < values.Length; i++)
                {
                    var dimension = input.shape[start + i];
                    if (dimension < int.MinValue || dimension > int.MaxValue)
                    {
                        diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "dimension-query-overflow",
                            "Shape dimension exceeds the Aexis Int32 shape/index range.", "Use Int32-range static dimensions.", true));
                        return true;
                    }
                    values[i] = (int)dimension;
                }
                tensorShape = new long[] { values.Length };
            }

            var outputName = node.outputs[0];
            var tensor = new OnnxTensor
            {
                name = outputName,
                dataType = TensorDataType.Int32,
                onnxDataType = 6,
                dims = tensorShape,
                int32Data = values
            };
            immutableTensors[outputName] = tensor;
            Register(descriptors, outputName, tensor.dataType, tensor.dims, true, tensor.onnxDataType);
            return true;
        }

        private static bool TryFoldStaticShapeNode(
            int index,
            OnnxNode node,
            Dictionary<string, AexisOnnxTensorDescriptor> descriptors,
            Dictionary<string, OnnxTensor> immutableTensors,
            List<AexisOnnxLoweringDiagnostic> diagnostics)
        {
            if (node == null || node.outputs.Count != 1 || string.IsNullOrEmpty(node.outputs[0])
                || !IsStaticShapeFoldOperator(node.opType))
                return false;
            for (var inputIndex = 0; inputIndex < node.inputs.Count; inputIndex++)
            {
                var inputName = node.inputs[inputIndex];
                if (!string.IsNullOrEmpty(inputName) && !immutableTensors.ContainsKey(inputName))
                    return false;
            }
            if (node.inputs.Count == 0 || !immutableTensors.TryGetValue(node.inputs[0], out var firstInput)
                || !TryGetIntValues(firstInput, out _))
                return false;

            try
            {
                if (!TryEvaluateStaticShapeNode(node, immutableTensors, out var values, out var shape, out var onnxDataType, out var reason))
                {
                    diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "static-shape-fold-failed",
                        "A fully constant shape/index node could not be folded exactly: " + reason,
                        "Rewrite the shape subgraph to the supported deterministic integer subset.", true));
                    return true;
                }
                if (shape == null || shape.Length > 4 || HasDynamic(shape))
                {
                    diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "static-shape-fold-invalid-shape",
                        "The folded shape/index tensor has an unsupported rank or dynamic extent.",
                        "Keep folded shape tensors static with rank <= 4.", true));
                    return true;
                }

                var outputName = node.outputs[0];
                var tensor = CreateFoldedIntTensor(outputName, onnxDataType, shape, values);
                immutableTensors[outputName] = tensor;
                Register(descriptors, outputName, tensor.dataType, tensor.dims, true, tensor.onnxDataType);
                return true;
            }
            catch (OverflowException)
            {
                diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "static-shape-fold-overflow",
                    "The constant shape/index expression overflows its declared integer type.",
                    "Keep shape arithmetic within the ONNX INT32/INT64 range.", true));
                return true;
            }
        }

        private static bool IsStaticShapeFoldOperator(string opType)
        {
            switch (opType)
            {
                case "Cast":
                case "Gather":
                case "Slice":
                case "Squeeze":
                case "Unsqueeze":
                case "Concat":
                case "Add":
                case "Sub":
                case "Mul":
                case "Div":
                case "Reshape":
                    return true;
                default:
                    return false;
            }
        }

        private static bool TryEvaluateStaticShapeNode(
            OnnxNode node,
            Dictionary<string, OnnxTensor> tensors,
            out long[] values,
            out long[] shape,
            out int onnxDataType,
            out string reason)
        {
            values = null;
            shape = null;
            onnxDataType = 0;
            reason = null;
            if (node.inputs.Count == 0 || !tensors.TryGetValue(node.inputs[0], out var first) || !TryGetIntValues(first, out var firstValues))
            {
                reason = "the first input is not a decoded integer tensor";
                return false;
            }
            onnxDataType = first.onnxDataType == 7 ? 7 : 6;

            switch (node.opType)
            {
                case "Cast":
                {
                    var targetType = GetInt(node, "to", 0);
                    if (targetType != 6 && targetType != 7)
                    {
                        reason = "only INT32 and INT64 shape casts are folded";
                        return false;
                    }
                    values = firstValues;
                    shape = Clone(first.dims);
                    onnxDataType = targetType;
                    return ValuesFitDeclaredType(values, onnxDataType, out reason);
                }
                case "Gather":
                {
                    if (node.inputs.Count < 2 || !tensors.TryGetValue(node.inputs[1], out var indices)
                        || !TryGetIntValues(indices, out var indexValues) || first.dims == null || first.dims.Length != 1
                        || NormalizeAxis(GetInt(node, "axis", 0), 1) != 0)
                    {
                        reason = "P0 constant Gather folding requires rank-1 integer data and integer indices on axis 0";
                        return false;
                    }
                    values = new long[indexValues.Length];
                    for (var i = 0; i < indexValues.Length; i++)
                    {
                        var source = indexValues[i] < 0 ? checked(indexValues[i] + firstValues.Length) : indexValues[i];
                        if (source < 0 || source >= firstValues.Length)
                        {
                            reason = "Gather index is out of range";
                            return false;
                        }
                        values[i] = firstValues[(int)source];
                    }
                    shape = Clone(indices.dims);
                    return true;
                }
                case "Slice":
                {
                    if (!TryEvaluateStaticRankOneSlice(node, tensors, firstValues, out values, out reason))
                        return false;
                    shape = new long[] { values.Length };
                    return true;
                }
                case "Squeeze":
                    shape = InferSqueeze(node, first.dims, tensors);
                    values = firstValues;
                    break;
                case "Unsqueeze":
                    shape = InferUnsqueeze(node, first.dims, tensors);
                    values = firstValues;
                    break;
                case "Reshape":
                    shape = InferReshape(node, first.dims, tensors);
                    values = firstValues;
                    break;
                case "Concat":
                {
                    if (GetInt(node, "axis", 0) != 0)
                    {
                        reason = "P0 constant Concat folding supports rank-1 shape tensors on axis 0";
                        return false;
                    }
                    var concatenated = new List<long>();
                    foreach (var inputName in node.inputs)
                    {
                        if (!tensors.TryGetValue(inputName, out var tensor) || tensor.dims == null || tensor.dims.Length != 1
                            || !TryGetIntValues(tensor, out var inputValues))
                        {
                            reason = "Concat input is not a rank-1 integer tensor";
                            return false;
                        }
                        concatenated.AddRange(inputValues);
                        if (tensor.onnxDataType == 7) onnxDataType = 7;
                    }
                    values = concatenated.ToArray();
                    shape = new long[] { values.Length };
                    return ValuesFitDeclaredType(values, onnxDataType, out reason);
                }
                case "Add":
                case "Sub":
                case "Mul":
                case "Div":
                {
                    if (node.inputs.Count < 2 || !tensors.TryGetValue(node.inputs[1], out var second)
                        || !TryGetIntValues(second, out var secondValues)
                        || !TryFoldIntegerBinary(node.opType, firstValues, secondValues, out values, out reason))
                        return false;
                    shape = firstValues.Length == 1 ? Clone(second.dims) : Clone(first.dims);
                    if (second.onnxDataType == 7) onnxDataType = 7;
                    return ValuesFitDeclaredType(values, onnxDataType, out reason);
                }
                default:
                    reason = "operator is not in the deterministic shape fold subset";
                    return false;
            }

            if (shape == null || HasDynamic(shape) || ElementCount(shape) != values.LongLength)
            {
                reason = "folded reshape/squeeze geometry does not preserve element count";
                return false;
            }
            return ValuesFitDeclaredType(values, onnxDataType, out reason);
        }

        private static bool TryEvaluateStaticRankOneSlice(
            OnnxNode node,
            Dictionary<string, OnnxTensor> tensors,
            long[] input,
            out long[] output,
            out string reason)
        {
            output = null;
            reason = null;
            if (node.inputs.Count < 3
                || !TryGetInitializerInput(node, 1, tensors, out var startsTensor) || !TryGetIntValues(startsTensor, out var starts)
                || !TryGetInitializerInput(node, 2, tensors, out var endsTensor) || !TryGetIntValues(endsTensor, out var ends)
                || starts.Length != 1 || ends.Length != 1)
            {
                reason = "rank-1 Slice requires one static start and end";
                return false;
            }
            long axis = 0;
            if (TryGetInitializerInput(node, 3, tensors, out var axesTensor))
            {
                if (!TryGetIntValues(axesTensor, out var axes) || axes.Length != 1) { reason = "Slice axes must contain one value"; return false; }
                axis = axes[0];
            }
            long step = 1;
            if (TryGetInitializerInput(node, 4, tensors, out var stepsTensor))
            {
                if (!TryGetIntValues(stepsTensor, out var steps) || steps.Length != 1) { reason = "Slice steps must contain one value"; return false; }
                step = steps[0];
            }
            if (axis != 0 && axis != -1 || step != 1)
            {
                reason = "P0 constant Slice folding supports axis 0 with step 1";
                return false;
            }
            var start = starts[0] < 0 ? Math.Max(0L, input.LongLength + starts[0]) : Math.Min(input.LongLength, starts[0]);
            var end = ends[0] < 0 ? Math.Max(0L, input.LongLength + ends[0]) : Math.Min(input.LongLength, ends[0]);
            end = Math.Max(start, end);
            output = new long[checked((int)(end - start))];
            Array.Copy(input, (int)start, output, 0, output.Length);
            return true;
        }

        private static bool TryFoldIntegerBinary(string opType, long[] left, long[] right, out long[] output, out string reason)
        {
            output = null;
            reason = null;
            if (left == null || right == null || left.Length != right.Length && left.Length != 1 && right.Length != 1)
            {
                reason = "integer binary fold supports equal shapes or scalar broadcasting";
                return false;
            }
            var count = Math.Max(left.Length, right.Length);
            output = new long[count];
            for (var i = 0; i < count; i++)
            {
                var a = left[left.Length == 1 ? 0 : i];
                var b = right[right.Length == 1 ? 0 : i];
                switch (opType)
                {
                    case "Add": output[i] = checked(a + b); break;
                    case "Sub": output[i] = checked(a - b); break;
                    case "Mul": output[i] = checked(a * b); break;
                    case "Div":
                        if (b == 0) { reason = "integer division by zero"; return false; }
                        if (a == long.MinValue && b == -1) throw new OverflowException();
                        output[i] = a / b;
                        break;
                }
            }
            return true;
        }

        private static bool ValuesFitDeclaredType(long[] values, int onnxDataType, out string reason)
        {
            if (onnxDataType == 7) { reason = null; return true; }
            for (var i = 0; i < values.Length; i++)
            {
                if (values[i] < int.MinValue || values[i] > int.MaxValue)
                {
                    reason = "folded value exceeds INT32 range";
                    return false;
                }
            }
            reason = null;
            return true;
        }

        private static OnnxTensor CreateFoldedIntTensor(string name, int onnxDataType, long[] shape, long[] values)
        {
            var tensor = new OnnxTensor
            {
                name = name,
                dataType = TensorDataType.Int32,
                onnxDataType = onnxDataType == 7 ? 7 : 6,
                dims = Clone(shape)
            };
            if (tensor.onnxDataType == 7)
            {
                tensor.int64Data = (long[])values.Clone();
            }
            else
            {
                tensor.int32Data = new int[values.Length];
                for (var i = 0; i < values.Length; i++) tensor.int32Data[i] = checked((int)values[i]);
            }
            return tensor;
        }

        private static void Register(
            Dictionary<string, AexisOnnxTensorDescriptor> descriptors,
            string name,
            TensorDataType type,
            long[] shape,
            bool initializer,
            int onnxDataType = 0)
        {
            if (string.IsNullOrWhiteSpace(name)) return;
            var sourceShape = Clone(shape);
            descriptors[name] = new AexisOnnxTensorDescriptor
            {
                name = name,
                dataType = type,
                onnxDataType = onnxDataType != 0 ? onnxDataType : ToOnnxDataType(type),
                shape = sourceShape,
                runtimeShape = initializer ? ToRuntimeConstantShape(sourceShape) : Clone(sourceShape),
                batchAxis = -1,
                isInitializer = initializer
            };
        }

        private static int ToOnnxDataType(TensorDataType type)
        {
            switch (type)
            {
                case TensorDataType.Float32: return 1;
                case TensorDataType.UInt8: return 2;
                case TensorDataType.Int8: return 3;
                case TensorDataType.Int32: return 6;
                case TensorDataType.Float16: return 10;
                default: return 0;
            }
        }

        private static int ResolveOnnxDataType(AexisOnnxTensorDescriptor descriptor)
        {
            return descriptor == null
                ? 0
                : descriptor.onnxDataType != 0
                    ? descriptor.onnxDataType
                    : ToOnnxDataType(descriptor.dataType);
        }

        private static void SetBatchContract(AexisOnnxTensorDescriptor descriptor, int batchAxis)
        {
            if (descriptor == null) return;
            descriptor.batchAxis = batchAxis;
            descriptor.runtimeShape = RemoveAxis(descriptor.shape, batchAxis);
        }

        private static int InferOutputBatchAxis(
            OnnxNode node,
            List<AexisOnnxTensorDescriptor> inputs,
            long[] outputShape,
            Dictionary<string, OnnxTensor> initializers,
            int index,
            List<AexisOnnxLoweringDiagnostic> diagnostics)
        {
            var source = FindActivationInput(inputs);
            var batchAxis = source?.batchAxis ?? -1;
            if (batchAxis < 0 || outputShape == null || outputShape.Length == 0)
                return -1;

            if (string.Equals(node.opType, "Transpose", StringComparison.Ordinal))
            {
                var permutation = GetInts(node, "perm", ReverseAxes(source.shape.Length));
                for (var i = 0; i < permutation.Length; i++)
                    if (permutation[i] == batchAxis) return i;
                return -1;
            }

            if (string.Equals(node.opType, "Reshape", StringComparison.Ordinal))
            {
                if (outputShape.Length > 0 && outputShape[0] == 1)
                    return 0;
                diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "unresolved-batch-reshape",
                    "Reshape does not preserve the leading static batch=1 axis.",
                    "Keep batch as the leading dimension with value 1 before Aexis import.", true));
                return -1;
            }

            if (string.Equals(node.opType, "Flatten", StringComparison.Ordinal))
            {
                if (batchAxis == 0 && GetInt(node, "axis", 1) == 1) return 0;
                diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "unsupported-batch-flatten",
                    "Flatten must use axis=1 when removing the leading ONNX batch axis.",
                    "Export Flatten(axis=1) for static batch=1 models.", true));
                return -1;
            }

            if (string.Equals(node.opType, "Squeeze", StringComparison.Ordinal))
            {
                var axes = ResolveStaticAxes(node, initializers);
                if (axes == null || axes.Length == 0)
                    return outputShape.Length > batchAxis && outputShape[batchAxis] == 1 ? batchAxis : -1;
                var normalized = NormalizeAxes(axes, source.shape.Length);
                for (var i = 0; i < normalized.Length; i++)
                    if (normalized[i] == batchAxis) return -1;
                var removedBefore = 0;
                for (var i = 0; i < normalized.Length; i++) if (normalized[i] < batchAxis) removedBefore++;
                return batchAxis - removedBefore;
            }

            if (string.Equals(node.opType, "Unsqueeze", StringComparison.Ordinal))
            {
                var normalized = NormalizeAxes(ResolveStaticAxes(node, initializers), outputShape.Length);
                var insertedBefore = 0;
                for (var i = 0; i < normalized.Length; i++) if (normalized[i] <= batchAxis) insertedBefore++;
                return batchAxis + insertedBefore;
            }

            if (string.Equals(node.opType, "OneHot", StringComparison.Ordinal))
            {
                var outputAxis = NormalizeAxis(GetInt(node, "axis", -1), outputShape.Length);
                return outputAxis <= batchAxis ? batchAxis + 1 : batchAxis;
            }

            if (node.opType.StartsWith("Reduce", StringComparison.Ordinal)
                || string.Equals(node.opType, "ArgMax", StringComparison.Ordinal)
                || string.Equals(node.opType, "ArgMin", StringComparison.Ordinal))
            {
                var keepDims = GetInt(node, "keepdims", 1) != 0;
                var axes = node.opType.StartsWith("Reduce", StringComparison.Ordinal)
                    ? ResolveStaticAxes(node, initializers)
                    : new long[] { GetInt(node, "axis", 0) };
                if (axes == null || axes.Length == 0) return -1;
                var normalized = NormalizeAxes(axes, source.shape.Length);
                var removedBefore = 0;
                var reducesBatch = false;
                for (var i = 0; i < normalized.Length; i++)
                {
                    if (normalized[i] == batchAxis) reducesBatch = true;
                    if (normalized[i] < batchAxis) removedBefore++;
                }
                if (reducesBatch) return keepDims ? batchAxis : -1;
                return keepDims ? batchAxis : batchAxis - removedBefore;
            }

            if (batchAxis >= outputShape.Length || outputShape[batchAxis] != 1)
            {
                diagnostics.Add(Diagnostic(index, NodeName(node, index), node.opType, "batch-axis-not-preserved",
                    "The operator does not preserve the static batch=1 axis in its inferred output.",
                    "Rewrite this node so batch remains a static singleton, or implement a batched texture profile.", true));
                return -1;
            }
            return batchAxis;
        }

        private static AexisOnnxTensorDescriptor FindActivationInput(List<AexisOnnxTensorDescriptor> inputs)
        {
            if (inputs == null) return null;
            for (var i = 0; i < inputs.Count; i++)
                if (inputs[i] != null && !inputs[i].isInitializer && inputs[i].batchAxis >= 0) return inputs[i];
            return null;
        }

        private static long[] ResolveStaticAxes(OnnxNode node, Dictionary<string, OnnxTensor> initializers)
        {
            var axes = GetInts(node, "axes", null);
            if ((axes == null || axes.Length == 0)
                && TryGetInitializerInput(node, 1, initializers, out var axesTensor))
                TryGetIntValues(axesTensor, out axes);
            return axes;
        }

        private static long[] NormalizeAxes(long[] axes, int rank)
        {
            if (axes == null) return Array.Empty<long>();
            var result = new long[axes.Length];
            for (var i = 0; i < axes.Length; i++) result[i] = NormalizeAxis((int)axes[i], rank);
            return result;
        }

        private static long[] RemoveAxis(long[] shape, int axis)
        {
            if (shape == null) return Array.Empty<long>();
            if (axis < 0 || axis >= shape.Length) return Clone(shape);
            var result = new long[shape.Length - 1];
            if (axis > 0) Array.Copy(shape, 0, result, 0, axis);
            if (axis + 1 < shape.Length) Array.Copy(shape, axis + 1, result, axis, shape.Length - axis - 1);
            return result;
        }

        private static bool TryTranslateRuntimeAxis(AexisOnnxTensorDescriptor descriptor, int onnxAxis, out int runtimeAxis)
        {
            var rank = descriptor?.shape?.Length ?? 0;
            runtimeAxis = NormalizeAxis(onnxAxis, rank);
            if (runtimeAxis < 0 || runtimeAxis >= rank) return false;
            var batchAxis = descriptor?.batchAxis ?? -1;
            if (runtimeAxis == batchAxis) return false;
            if (batchAxis >= 0 && runtimeAxis > batchAxis) runtimeAxis--;
            return true;
        }

        private static long[] TranslateAxesRemovingBatch(long[] axes, int rank, int batchAxis)
        {
            if (axes == null || axes.Length == 0) return Array.Empty<long>();
            var translated = new List<long>(axes.Length);
            for (var i = 0; i < axes.Length; i++)
            {
                var axis = NormalizeAxis((int)axes[i], rank);
                if (axis < 0 || axis >= rank || axis == batchAxis) continue;
                if (batchAxis >= 0 && axis > batchAxis) axis--;
                if (!translated.Contains(axis)) translated.Add(axis);
            }
            translated.Sort();
            return translated.ToArray();
        }

        private static long NormalizeSliceBound(long value, long dimension)
        {
            if (dimension <= 0) return value;
            if (value < 0)
            {
                if (value <= -dimension) return 0;
                value += dimension;
            }
            if (value < 0) return 0;
            return value > dimension ? dimension : value;
        }

        private static int InferUnsqueezeBatchAxis(AexisOnnxTensorDescriptor source, long[] axes, int outputRank)
        {
            var batchAxis = source?.batchAxis ?? -1;
            if (batchAxis < 0) return -1;
            var normalized = NormalizeAxes(axes, outputRank);
            for (var i = 0; i < normalized.Length; i++)
                if (normalized[i] <= batchAxis) batchAxis++;
            return batchAxis;
        }

        private static long[] RemoveBatchFromPermutation(long[] permutation, int batchAxis)
        {
            if (permutation == null || batchAxis < 0) return Clone(permutation);
            var translated = new List<long>(Math.Max(0, permutation.Length - 1));
            for (var i = 0; i < permutation.Length; i++)
            {
                var sourceAxis = permutation[i];
                if (sourceAxis == batchAxis) continue;
                translated.Add(sourceAxis > batchAxis ? sourceAxis - 1 : sourceAxis);
            }
            return translated.ToArray();
        }

        private static void ConvertToNoop(AexisGraphModel.Layer layer)
        {
            layer.typeName = "Noop";
            layer.type = AexisLayerTypes.Noop;
            layer.intParams.Clear();
            layer.stringParams.Clear();
        }

        private static AexisOnnxTensorDescriptor[] ToArray(Dictionary<string, AexisOnnxTensorDescriptor> descriptors)
        {
            var values = new List<AexisOnnxTensorDescriptor>(descriptors.Values);
            values.Sort((left, right) => string.CompareOrdinal(left.name, right.name));
            return values.ToArray();
        }

        private static long[] InferConvolution(OnnxNode node, List<AexisOnnxTensorDescriptor> inputs, Dictionary<string, OnnxTensor> tensors, bool transpose)
        {
            if (inputs.Count < 2 || inputs[0].shape == null || inputs[0].shape.Length != 4
                || !TryGetInitializerInput(node, 1, tensors, out var weight) || weight.dims == null || weight.dims.Length != 4)
                return Dynamic(4);
            var input = inputs[0].shape;
            var strides = GetInts(node, "strides", new long[] { 1, 1 });
            var dilations = GetInts(node, "dilations", new long[] { 1, 1 });
                var kernel = GetInts(node, "kernel_shape", new[] { weight.dims[2], weight.dims[3] });
            if (!TryResolveAutoPads2D(node, input, kernel, strides, dilations, transpose, out var pads, out _)) return Dynamic(4);
            var outputPadding = GetInts(node, "output_padding", new long[] { 0, 0 });
            if (strides.Length != 2 || dilations.Length != 2 || pads.Length != 4 || outputPadding.Length != 2) return Dynamic(4);
            var group = Math.Max(1, GetInt(node, "group", 1));
            var result = new long[4];
            result[0] = input[0];
            result[1] = transpose ? weight.dims[1] * group : weight.dims[0];
            for (var axis = 0; axis < 2; axis++)
            {
                if (input[axis + 2] < 0) { result[axis + 2] = -1; continue; }
                var extent = dilations[axis] * (weight.dims[axis + 2] - 1) + 1;
                result[axis + 2] = transpose
                    ? (input[axis + 2] - 1) * strides[axis] - pads[axis] - pads[axis + 2] + extent + outputPadding[axis]
                    : FloorDiv(input[axis + 2] + pads[axis] + pads[axis + 2] - extent, strides[axis]) + 1;
            }
            return result;
        }

        private static long[] InferPool(OnnxNode node, long[] input, bool global)
        {
            if (input == null || input.Length < 3) return Dynamic(input?.Length ?? 0);
            var result = Clone(input);
            if (global)
            {
                for (var i = 2; i < result.Length; i++) result[i] = 1;
                return result;
            }
            var spatialRank = input.Length - 2;
            var kernel = GetInts(node, "kernel_shape", Array.Empty<long>());
            var strides = GetInts(node, "strides", Ones(spatialRank));
            var dilations = GetInts(node, "dilations", Ones(spatialRank));
            long[] pads;
            if (spatialRank == 2)
            {
                if (!TryResolveAutoPads2D(node, input, kernel, strides, dilations, false, out pads, out _)) return Dynamic(input.Length);
            }
            else
            {
                pads = GetInts(node, "pads", new long[spatialRank * 2]);
            }
            if (kernel.Length != spatialRank || strides.Length != spatialRank || dilations.Length != spatialRank || pads.Length != spatialRank * 2) return Dynamic(input.Length);
            var ceil = GetInt(node, "ceil_mode", 0) != 0;
            for (var i = 0; i < spatialRank; i++)
            {
                if (input[i + 2] < 0) { result[i + 2] = -1; continue; }
                var extent = dilations[i] * (kernel[i] - 1) + 1;
                var numerator = input[i + 2] + pads[i] + pads[i + spatialRank] - extent;
                result[i + 2] = (ceil ? CeilDiv(numerator, strides[i]) : FloorDiv(numerator, strides[i])) + 1;
            }
            return result;
        }

        private static long[] InferPad(OnnxNode node, long[] input, Dictionary<string, OnnxTensor> tensors)
        {
            if (input == null) return Array.Empty<long>();
            var pads = GetInts(node, "pads", null);
            if ((pads == null || pads.Length == 0)
                && TryGetInitializerInput(node, 1, tensors, out var padsTensor))
                TryGetIntValues(padsTensor, out pads);
            if (pads == null || pads.Length != input.Length * 2) return Dynamic(input.Length);
            var output = new long[input.Length];
            for (var i = 0; i < output.Length; i++)
            {
                if (input[i] < 0) output[i] = -1;
                else output[i] = input[i] + pads[i] + pads[i + input.Length];
                if (output[i] <= 0) return Dynamic(input.Length);
            }
            return output;
        }

        private static long[] InferResize(OnnxNode node, long[] input, Dictionary<string, OnnxTensor> tensors)
        {
            if (input == null) return Array.Empty<long>();
            var output = Clone(input);
            var axes = GetInts(node, "axes", null);
            var sizesInput = string.Equals(node.opType, "Resize", StringComparison.Ordinal) ? 3 : -1;
            var scalesInput = string.Equals(node.opType, "Resize", StringComparison.Ordinal) ? 2 : 1;
            if (sizesInput >= 0 && TryGetInitializerInput(node, sizesInput, tensors, out var sizesTensor)
                && TryGetIntValues(sizesTensor, out var sizes) && sizes.Length > 0)
            {
                if (axes != null && axes.Length == sizes.Length)
                {
                    for (var i = 0; i < axes.Length; i++)
                    {
                        var axis = NormalizeAxis((int)axes[i], input.Length);
                        if (axis < 0 || axis >= input.Length || sizes[i] <= 0) return Dynamic(input.Length);
                        output[axis] = sizes[i];
                    }
                    return output;
                }
                return sizes.Length == input.Length ? Clone(sizes) : Dynamic(input.Length);
            }

            float[] scales = null;
            if (TryGetInitializerInput(node, scalesInput, tensors, out var scalesTensor))
                TryGetFloatValues(scalesTensor, out scales);
            if ((scales == null || scales.Length == 0)
                && node.attributes.TryGetValue("scales", out var scalesAttribute) && scalesAttribute.floats.Count > 0)
                scales = scalesAttribute.floats.ToArray();
            if (scales == null) return Dynamic(input.Length);
            if (axes != null && axes.Length == scales.Length)
            {
                for (var i = 0; i < axes.Length; i++)
                {
                    var axis = NormalizeAxis((int)axes[i], input.Length);
                    if (axis < 0 || axis >= input.Length || scales[i] <= 0f || input[axis] < 0) return Dynamic(input.Length);
                    output[axis] = (long)Math.Floor(input[axis] * scales[i]);
                }
                return output;
            }
            if (scales.Length != input.Length) return Dynamic(input.Length);
            for (var i = 0; i < output.Length; i++)
            {
                if (scales[i] <= 0f || input[i] < 0) return Dynamic(input.Length);
                output[i] = (long)Math.Floor(input[i] * scales[i]);
            }
            return output;
        }

        private static long[] InferDepthToSpace(OnnxNode node, long[] input)
        {
            if (input == null || input.Length != 4) return Dynamic(input?.Length ?? 4);
            var blockSize = GetInt(node, "blocksize", 0);
            var divisor = (long)blockSize * blockSize;
            if (blockSize <= 0 || input[1] <= 0 || input[1] % divisor != 0) return Dynamic(4);
            return new[] { input[0], input[1] / divisor, input[2] * blockSize, input[3] * blockSize };
        }

        private static long[] InferSpaceToDepth(OnnxNode node, long[] input)
        {
            if (input == null || input.Length != 4) return Dynamic(input?.Length ?? 4);
            var blockSize = GetInt(node, "blocksize", 0);
            if (blockSize <= 0 || input[2] <= 0 || input[3] <= 0 || input[2] % blockSize != 0 || input[3] % blockSize != 0)
                return Dynamic(4);
            return new[] { input[0], input[1] * blockSize * blockSize, input[2] / blockSize, input[3] / blockSize };
        }

        private static long[] InferGemm(OnnxNode node, List<AexisOnnxTensorDescriptor> inputs)
        {
            if (inputs.Count < 2 || inputs[0].shape == null || inputs[1].shape == null || inputs[0].shape.Length != 2 || inputs[1].shape.Length != 2) return Dynamic(2);
            var transA = GetInt(node, "transA", 0) != 0;
            var transB = GetInt(node, "transB", 0) != 0;
            var m = inputs[0].shape[transA ? 1 : 0];
            var kA = inputs[0].shape[transA ? 0 : 1];
            var kB = inputs[1].shape[transB ? 1 : 0];
            var n = inputs[1].shape[transB ? 0 : 1];
            return kA >= 0 && kB >= 0 && kA != kB ? Dynamic(2) : new[] { m, n };
        }

        private static long[] InferMatMul(long[] left, long[] right)
        {
            if (left == null || right == null || left.Length == 0 || right.Length == 0) return Array.Empty<long>();
            var leftVector = left.Length == 1;
            var rightVector = right.Length == 1;
            var a = leftVector ? new[] { 1L, left[0] } : left;
            var b = rightVector ? new[] { right[0], 1L } : right;
            var kA = a[a.Length - 1];
            var kB = b[b.Length - 2];
            var batchA = SubArray(a, 0, a.Length - 2);
            var batchB = SubArray(b, 0, b.Length - 2);
            var batch = Broadcast(batchA, batchB);
            if (kA >= 0 && kB >= 0 && kA != kB) return Dynamic(batch.Length + (leftVector ? 0 : 1) + (rightVector ? 0 : 1));
            var output = new List<long>(batch);
            if (!leftVector) output.Add(a[a.Length - 2]);
            if (!rightVector) output.Add(b[b.Length - 1]);
            return output.ToArray();
        }

        private static long[] InferReshape(OnnxNode node, long[] input, Dictionary<string, OnnxTensor> tensors)
        {
            if (!TryGetInitializerInput(node, 1, tensors, out var shapeTensor) || !TryGetIntValues(shapeTensor, out var target)) return Dynamic(Math.Max(1, input?.Length ?? 0));
            var output = Clone(target);
            var allowZero = GetInt(node, "allowzero", 0) != 0;
            long known = 1;
            var inferred = -1;
            for (var i = 0; i < output.Length; i++)
            {
                if (output[i] == 0 && !allowZero)
                {
                    if (input == null || i >= input.Length) return Dynamic(output.Length);
                    output[i] = input[i];
                }
                if (output[i] == -1) { if (inferred >= 0) return Dynamic(output.Length); inferred = i; }
                else if (output[i] < 0) return Dynamic(output.Length);
                else
                {
                    if (output[i] != 0 && known > long.MaxValue / output[i]) return Dynamic(output.Length);
                    known *= output[i];
                }
            }
            if (inferred >= 0)
            {
                var total = ElementCount(input);
                if (total <= 0 || known <= 0 || total % known != 0) return Dynamic(output.Length);
                output[inferred] = total / known;
            }
            else
            {
                var total = ElementCount(input);
                if (total < 0 || total != known)
                    return Dynamic(output.Length);
            }
            return output;
        }

        private static long[] InferSqueeze(OnnxNode node, long[] input, Dictionary<string, OnnxTensor> tensors)
        {
            if (input == null) return Array.Empty<long>();
            var axes = GetInts(node, "axes", null);
            if ((axes == null || axes.Length == 0) && TryGetInitializerInput(node, 1, tensors, out var tensor)) TryGetIntValues(tensor, out axes);
            var remove = new HashSet<int>();
            if (axes == null || axes.Length == 0)
            {
                for (var i = 0; i < input.Length; i++) if (input[i] == 1) remove.Add(i);
            }
            else
            {
                foreach (var raw in axes)
                {
                    var axis = NormalizeAxis((int)raw, input.Length);
                    if (axis < 0 || axis >= input.Length || input[axis] != 1) return Dynamic(Math.Max(0, input.Length - axes.Length));
                    remove.Add(axis);
                }
            }
            var result = new List<long>();
            for (var i = 0; i < input.Length; i++) if (!remove.Contains(i)) result.Add(input[i]);
            return result.ToArray();
        }

        private static long[] InferUnsqueeze(OnnxNode node, long[] input, Dictionary<string, OnnxTensor> tensors)
        {
            var axes = GetInts(node, "axes", null);
            if ((axes == null || axes.Length == 0) && TryGetInitializerInput(node, 1, tensors, out var tensor)) TryGetIntValues(tensor, out axes);
            if (axes == null || axes.Length == 0) return Dynamic((input?.Length ?? 0) + 1);
            var rank = (input?.Length ?? 0) + axes.Length;
            var normalized = new HashSet<int>();
            foreach (var raw in axes)
            {
                var axis = (int)raw;
                if (axis < 0) axis += rank;
                if (axis < 0 || axis >= rank || !normalized.Add(axis)) return Dynamic(rank);
            }
            var result = new long[rank];
            var source = 0;
            for (var i = 0; i < rank; i++) result[i] = normalized.Contains(i) ? 1 : input[source++];
            return result;
        }

        private static long[] InferGather(OnnxNode node, long[] data, long[] indices)
        {
            if (data == null || indices == null) return Array.Empty<long>();
            var axis = NormalizeAxis(GetInt(node, "axis", 0), data.Length);
            if (axis < 0 || axis >= data.Length) return Dynamic(data.Length + indices.Length - 1);
            var output = new List<long>();
            for (var i = 0; i < axis; i++) output.Add(data[i]);
            output.AddRange(indices);
            for (var i = axis + 1; i < data.Length; i++) output.Add(data[i]);
            return output.ToArray();
        }

        private static long[] InferGatherND(OnnxNode node, long[] data, long[] indices)
        {
            if (data == null || indices == null || indices.Length == 0) return Array.Empty<long>();
            var batchDims = GetInt(node, "batch_dims", 0);
            var depth = indices[indices.Length - 1];
            if (depth < 0 || depth > data.Length - batchDims) return Dynamic(indices.Length - 1);
            var output = new List<long>();
            output.AddRange(SubArray(indices, 0, indices.Length - 1));
            output.AddRange(SubArray(data, batchDims + (int)depth, data.Length));
            return output.ToArray();
        }

        private static long[] InferReduction(OnnxNode node, long[] input, Dictionary<string, OnnxTensor> tensors)
        {
            if (input == null) return Array.Empty<long>();
            var axes = GetInts(node, "axes", null);
            if ((axes == null || axes.Length == 0) && TryGetInitializerInput(node, 1, tensors, out var axesTensor)) TryGetIntValues(axesTensor, out axes);
            if ((axes == null || axes.Length == 0) && GetInt(node, "noop_with_empty_axes", 0) != 0) return Clone(input);
            if (axes == null || axes.Length == 0) axes = RangeAxes(input.Length);
            var reduced = new HashSet<int>();
            foreach (var raw in axes)
            {
                var axis = NormalizeAxis((int)raw, input.Length);
                if (axis < 0 || axis >= input.Length) return Dynamic(GetInt(node, "keepdims", 1) != 0 ? input.Length : Math.Max(0, input.Length - axes.Length));
                reduced.Add(axis);
            }
            var keep = GetInt(node, "keepdims", 1) != 0;
            var output = new List<long>();
            for (var i = 0; i < input.Length; i++)
                if (reduced.Contains(i)) { if (keep) output.Add(1); }
                else output.Add(input[i]);
            return output.ToArray();
        }

        private static long[] InferAxisReduction(OnnxNode node, long[] input)
        {
            if (input == null || input.Length == 0) return Array.Empty<long>();
            var axis = NormalizeAxis(GetInt(node, "axis", 0), input.Length);
            if (axis < 0 || axis >= input.Length) return Dynamic(input.Length);
            var keep = GetInt(node, "keepdims", 1) != 0;
            var result = new List<long>();
            for (var i = 0; i < input.Length; i++) if (i != axis) result.Add(input[i]); else if (keep) result.Add(1);
            return result.ToArray();
        }

        private static long[] InferTopK(OnnxNode node, long[] input, Dictionary<string, OnnxTensor> tensors)
        {
            var result = Clone(input);
            if (result.Length == 0) return result;
            var axis = NormalizeAxis(GetInt(node, "axis", -1), result.Length);
            if (axis < 0 || axis >= result.Length || !TryGetInitializerInput(node, 1, tensors, out var kTensor) || !TryGetIntValues(kTensor, out var k) || k.Length != 1 || k[0] <= 0) return Dynamic(result.Length);
            result[axis] = k[0];
            return result;
        }

        private static long[] InferShapeResult(OnnxNode node, int rank)
        {
            var start = GetInt(node, "start", 0);
            var end = GetInt(node, "end", rank);
            if (start < 0) start += rank;
            if (end < 0) end += rank;
            start = Math.Max(0, Math.Min(rank, start));
            end = Math.Max(start, Math.Min(rank, end));
            return new long[] { end - start };
        }

        private static long[] InferStaticShapeInput(OnnxNode node, int inputIndex, Dictionary<string, OnnxTensor> tensors, int dynamicRank)
        {
            return TryGetInitializerInput(node, inputIndex, tensors, out var tensor) && TryGetIntValues(tensor, out var values)
                ? values
                : Dynamic(dynamicRank);
        }

        private static long[] InferTile(long[] input, long[] repeats)
        {
            if (input == null || repeats == null || input.Length != repeats.Length) return Dynamic(input?.Length ?? 0);
            var result = new long[input.Length];
            for (var i = 0; i < result.Length; i++) result[i] = input[i] < 0 || repeats[i] < 0 ? -1 : checked(input[i] * repeats[i]);
            return result;
        }

        private static long[] InferSlice(OnnxNode node, long[] input, Dictionary<string, OnnxTensor> tensors)
        {
            if (input == null || !TryGetInitializerInput(node, 1, tensors, out var startsTensor) || !TryGetIntValues(startsTensor, out var starts)
                || !TryGetInitializerInput(node, 2, tensors, out var endsTensor) || !TryGetIntValues(endsTensor, out var ends) || starts.Length != ends.Length)
                return Dynamic(input?.Length ?? 0);
            long[] axes = null;
            long[] steps = null;
            if (TryGetInitializerInput(node, 3, tensors, out var axesTensor)) TryGetIntValues(axesTensor, out axes);
            if (TryGetInitializerInput(node, 4, tensors, out var stepsTensor)) TryGetIntValues(stepsTensor, out steps);
            if (axes == null || axes.Length == 0) axes = RangeAxes(starts.Length);
            var result = Clone(input);
            for (var i = 0; i < starts.Length; i++)
            {
                var axis = NormalizeAxis((int)axes[i], result.Length);
                var step = steps == null ? 1 : steps[i];
                if (axis < 0 || axis >= result.Length || step != 1 || result[axis] < 0) return Dynamic(result.Length);
                var start = starts[i] < 0 ? Math.Max(0, result[axis] + starts[i]) : Math.Min(result[axis], starts[i]);
                var end = ends[i] < 0 ? Math.Max(0, result[axis] + ends[i]) : Math.Min(result[axis], ends[i]);
                result[axis] = Math.Max(0, end - start);
            }
            return result;
        }

        private static long[] InferRange(OnnxNode node, Dictionary<string, OnnxTensor> tensors)
        {
            if (!TryGetInitializerInput(node, 0, tensors, out var startTensor) || !TryGetScalarFloat(startTensor, out var start)
                || !TryGetInitializerInput(node, 1, tensors, out var limitTensor) || !TryGetScalarFloat(limitTensor, out var limit)
                || !TryGetInitializerInput(node, 2, tensors, out var deltaTensor) || !TryGetScalarFloat(deltaTensor, out var delta) || Math.Abs(delta) < 1e-12f)
                return Dynamic(1);
            var count = (long)Math.Ceiling((limit - start) / delta);
            return count > 0 ? new[] { count } : Dynamic(1);
        }

        private static long[] InferOneHot(OnnxNode node, long[] indices, Dictionary<string, OnnxTensor> tensors)
        {
            if (!TryGetInitializerInput(node, 1, tensors, out var depthTensor) || !TryGetIntValues(depthTensor, out var depth) || depth.Length != 1 || depth[0] <= 0)
                return Dynamic((indices?.Length ?? 0) + 1);
            var rank = (indices?.Length ?? 0) + 1;
            var axis = GetInt(node, "axis", -1);
            if (axis < 0) axis += rank;
            if (axis < 0 || axis >= rank) return Dynamic(rank);
            var result = new List<long>(indices ?? Array.Empty<long>());
            result.Insert(axis, depth[0]);
            return result.ToArray();
        }

        private static long[] InferConcat(OnnxNode node, List<AexisOnnxTensorDescriptor> inputs, long[] shape)
        {
            var axis = GetInt(node, "axis", 0); if (axis < 0) axis += shape.Length;
            if (axis < 0 || axis >= shape.Length) return Dynamic(shape.Length);
            var total = 0L;
            foreach (var input in inputs)
            {
                if (input.shape == null || input.shape.Length != shape.Length || input.shape[axis] < 0) return Dynamic(shape.Length);
                total += input.shape[axis];
            }
            shape[axis] = total; return shape;
        }

        private static long[] InferTranspose(OnnxNode node, long[] shape)
        {
            if (!node.attributes.TryGetValue("perm", out var perm))
            {
                var reversed = Clone(shape);
                Array.Reverse(reversed);
                return reversed;
            }
            if (perm.ints.Count != shape.Length) return Dynamic(shape.Length);
            var result = new long[shape.Length];
            var seen = new bool[shape.Length];
            for (var i = 0; i < result.Length; i++)
            {
                var source = perm.ints[i];
                if (source < 0 || source >= shape.Length || seen[source]) return Dynamic(shape.Length);
                seen[source] = true;
                result[i] = shape[source];
            }
            return result;
        }

        private static bool TryResolveExtractImagePatchesSpec(
            OnnxNode node,
            long[] inputShape,
            out int kernelH,
            out int kernelW,
            out int strideH,
            out int strideW,
            out int rateH,
            out int rateW,
            out int padTop,
            out int padBottom,
            out int padLeft,
            out int padRight,
            out long[] outputShape,
            out string reason)
        {
            kernelH = kernelW = strideH = strideW = rateH = rateW = 0;
            padTop = padBottom = padLeft = padRight = 0;
            outputShape = Dynamic(4);
            reason = null;
            if (inputShape == null || inputShape.Length != 4 || HasDynamic(inputShape)
                || inputShape[0] != 1 || inputShape[1] <= 0 || inputShape[2] <= 0 || inputShape[3] <= 0)
            {
                reason = "ExtractImagePatches P0 requires static NHWC [1,H,W,C] with batch=1.";
                return false;
            }

            var kernels = GetInts(node, "ksizes", null);
            var strides = GetInts(node, "strides", null);
            var rates = GetInts(node, "rates", null);
            if (!TryReadNhwcSpatialPair(kernels, out kernelH, out kernelW)
                || !TryReadNhwcSpatialPair(strides, out strideH, out strideW)
                || !TryReadNhwcSpatialPair(rates, out rateH, out rateW))
            {
                reason = "ExtractImagePatches ksizes, strides, and rates must be positive [1,H,W,1] attributes.";
                return false;
            }

            var padding = GetString(node, "padding");
            if (!string.Equals(padding, "SAME", StringComparison.OrdinalIgnoreCase)
                && !string.Equals(padding, "VALID", StringComparison.OrdinalIgnoreCase))
            {
                reason = "ExtractImagePatches padding must be SAME or VALID.";
                return false;
            }

            try
            {
                var inputH = inputShape[1];
                var inputW = inputShape[2];
                var extentH = checked((long)rateH * (kernelH - 1L) + 1L);
                var extentW = checked((long)rateW * (kernelW - 1L) + 1L);
                long outputH;
                long outputW;
                if (string.Equals(padding, "SAME", StringComparison.OrdinalIgnoreCase))
                {
                    outputH = CeilDiv(inputH, strideH);
                    outputW = CeilDiv(inputW, strideW);
                    var totalH = Math.Max(0L, checked((outputH - 1L) * strideH + extentH - inputH));
                    var totalW = Math.Max(0L, checked((outputW - 1L) * strideW + extentW - inputW));
                    padTop = checked((int)(totalH / 2L));
                    padBottom = checked((int)(totalH - padTop));
                    padLeft = checked((int)(totalW / 2L));
                    padRight = checked((int)(totalW - padLeft));
                }
                else
                {
                    outputH = FloorDiv(inputH - extentH, strideH) + 1L;
                    outputW = FloorDiv(inputW - extentW, strideW) + 1L;
                }
                var outputChannels = checked(inputShape[3] * kernelH * kernelW);
                if (outputH <= 0 || outputW <= 0 || outputChannels <= 0
                    || outputH > int.MaxValue || outputW > int.MaxValue || outputChannels > int.MaxValue)
                {
                    reason = "ExtractImagePatches produces an empty or 32-bit-unrepresentable output.";
                    return false;
                }
                outputShape = new[] { 1L, outputH, outputW, outputChannels };
                return true;
            }
            catch (OverflowException)
            {
                reason = "ExtractImagePatches shape arithmetic overflowed the static descriptor range.";
                return false;
            }
        }

        private static bool TryReadNhwcSpatialPair(long[] values, out int height, out int width)
        {
            height = width = 0;
            if (values == null || values.Length != 4 || values[0] != 1 || values[3] != 1
                || values[1] <= 0 || values[2] <= 0 || values[1] > int.MaxValue || values[2] > int.MaxValue)
                return false;
            height = (int)values[1];
            width = (int)values[2];
            return true;
        }

        private static long FloorDiv(long numerator, long denominator)
        {
            if (denominator <= 0) throw new ArgumentOutOfRangeException(nameof(denominator));
            var quotient = numerator / denominator;
            var remainder = numerator % denominator;
            return remainder < 0 ? quotient - 1 : quotient;
        }

        private static long CeilDiv(long numerator, long denominator)
        {
            if (denominator <= 0) throw new ArgumentOutOfRangeException(nameof(denominator));
            var quotient = numerator / denominator;
            var remainder = numerator % denominator;
            return remainder > 0 ? quotient + 1 : quotient;
        }

        private static long[] InferFlatten(OnnxNode node, long[] shape)
        {
            var axis = GetInt(node, "axis", 1); if (axis < 0) axis += shape.Length;
            if (axis < 0 || axis > shape.Length || HasDynamic(shape)) return Dynamic(2);
            long left = 1, right = 1; for (var i = 0; i < axis; i++) left *= shape[i]; for (var i = axis; i < shape.Length; i++) right *= shape[i]; return new[] { left, right };
        }

        private static long[] Broadcast(long[] left, long[] right)
        {
            var rank = Math.Max(left?.Length ?? 0, right?.Length ?? 0); var result = new long[rank];
            for (var i = 0; i < rank; i++) { var a = GetTrailing(left, i); var b = GetTrailing(right, i); result[rank - i - 1] = a < 0 || b < 0 ? -1 : (a == 1 ? b : b == 1 || a == b ? a : -1); }
            return result;
        }

        private static long GetTrailing(long[] value, int index) => value == null || index >= value.Length ? 1 : value[value.Length - index - 1];
        private static long[] Ones(int count) { var result = new long[Math.Max(0, count)]; for (var i = 0; i < result.Length; i++) result[i] = 1; return result; }
        private static long[] RangeAxes(int count) { var result = new long[Math.Max(0, count)]; for (var i = 0; i < result.Length; i++) result[i] = i; return result; }
        private static long[] SubArray(long[] value, int start, int end)
        {
            if (value == null) return Array.Empty<long>();
            start = Math.Max(0, Math.Min(value.Length, start));
            end = Math.Max(start, Math.Min(value.Length, end));
            var result = new long[end - start];
            Array.Copy(value, start, result, 0, result.Length);
            return result;
        }
        private static int GetInt(OnnxNode node, string name, int defaultValue) => node.attributes.TryGetValue(name, out var attribute) ? (int)attribute.i : defaultValue;
        private static bool HasDynamic(long[] shape) { if (shape == null) return true; for (var i = 0; i < shape.Length; i++) if (shape[i] < 0) return true; return false; }
        private static bool HasZeroExtent(long[] shape) { if (shape == null) return false; for (var i = 0; i < shape.Length; i++) if (shape[i] == 0) return true; return false; }
        private static bool ShapesEqual(long[] left, long[] right)
        {
            if (left == null || right == null || left.Length != right.Length) return false;
            for (var i = 0; i < left.Length; i++) if (left[i] != right[i]) return false;
            return true;
        }
        private static long[] Dynamic(int rank) { var result = new long[Math.Max(0, rank)]; for (var i = 0; i < result.Length; i++) result[i] = -1; return result; }
        private static long[] Clone(long[] shape) => shape == null ? Array.Empty<long>() : (long[])shape.Clone();
        private static string NodeName(OnnxNode node, int index) => string.IsNullOrWhiteSpace(node.name) ? node.opType + "_" + index.ToString(CultureInfo.InvariantCulture) : node.name;
        private static AexisOnnxLoweringDiagnostic Diagnostic(int index, string node, string op, string code, string message, string action, bool blocking) => new AexisOnnxLoweringDiagnostic { nodeIndex = index, node = node, opType = op, code = code, message = message, recommendedAction = action, blocking = blocking };
        private static string Join(IList<long> values) { var result = new string[values.Count]; for (var i = 0; i < values.Count; i++) result[i] = values[i].ToString(CultureInfo.InvariantCulture); return string.Join(",", result); }
        private static string Join(IList<float> values) { var result = new string[values.Count]; for (var i = 0; i < values.Count; i++) result[i] = values[i].ToString("R", CultureInfo.InvariantCulture); return string.Join(",", result); }
    }
}
