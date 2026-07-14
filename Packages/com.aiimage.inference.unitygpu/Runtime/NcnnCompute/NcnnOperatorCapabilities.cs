using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using UnityEngine;

namespace NcnnCompute
{
    public static class NcnnOperatorCapabilityStatus
    {
        public const string Supported = "supported";
        public const string Partial = "partial";
        public const string AliasOnly = "alias-only";
        public const string DebugOnly = "debug-only";
        public const string Unsupported = "unsupported";
    }

    public static class NcnnOperatorCapabilityBackend
    {
        public const string RenderTexture = "render-texture";
        public const string CommandBuffer = "command-buffer";
    }

    [Serializable]
    public sealed class NcnnOperatorCapabilityProfile
    {
        public string backend;
        public string[] layouts;
        public string shapeProfile;
        public string[] supportedParameters;
        public string[] rejectedParameters;
    }

    [Serializable]
    public sealed class NcnnOperatorCapability
    {
        public string operatorName;
        public string canonicalOperator;
        public string[] importFormats;
        public bool importSupported;
        public bool shapeInference;
        public bool renderTexture;
        public bool commandBuffer;
        public bool fp32;
        public bool fp16;
        public bool int8;
        public string[] layouts;
        public int[] ranks;
        public string[] verifiedModels;
        public string status;
        public string limitations;
        public string[] requiredParameters;
        public NcnnOperatorCapabilityProfile[] profiles;
    }

    [Serializable]
    public sealed class NcnnOperatorCapabilityDocument
    {
        public int schemaVersion;
        public string contract;
        public NcnnOperatorCapability[] operators;
    }

    [Serializable]
    public sealed class NcnnPreflightTensorDescriptor
    {
        public string blob;
        public int[] logicalShape;
        public int[] storageShape;
        public string layout;
        public string dtype;
    }

    [Serializable]
    public sealed class NcnnModelPreflightRequest
    {
        public string modelName;
        public string targetBackend = NcnnOperatorCapabilityBackend.CommandBuffer;
        public string targetDtype = "FP32";
        public bool strict = true;
        public NcnnPreflightTensorDescriptor[] inputs = Array.Empty<NcnnPreflightTensorDescriptor>();
    }

    [Serializable]
    public sealed class NcnnModelPreflightIssue
    {
        public int layerIndex;
        public string layer;
        public string operatorName;
        public string blob;
        public string code;
        public string message;
        public string recommendedAction;
    }

    [Serializable]
    public sealed class NcnnModelPreflightNode
    {
        public int layerIndex;
        public string layer;
        public string operatorName;
        public string canonicalOperator;
        public string status;
        public bool strictEligible;
        public string[] bottomBlobs;
        public string[] topBlobs;
        public NcnnPreflightTensorDescriptor[] inputs;
        public string[] missingParameters;
        public string[] issues;
        public string recommendedAction;
    }

    [Serializable]
    public sealed class NcnnModelPreflightReport
    {
        public int schemaVersion;
        public string contract;
        public string modelName;
        public string targetBackend;
        public string targetDtype;
        public bool strict;
        public NcnnPreflightTensorDescriptor[] declaredInputs;
        public NcnnModelPreflightNode[] nodes;
        public NcnnModelPreflightIssue[] missingNodes;
        public NcnnModelPreflightIssue[] missingParameters;
        public NcnnModelPreflightIssue[] missingDependencies;
        public bool strictEligible;
        public string summary;
    }

    // This is intentionally metadata-only. It never creates a layer, NcnnRepro, texture, or buffer.
    public static class NcnnOperatorCapabilities
    {
        public const int SchemaVersion = 2;
        public const string Contract = "aiimage.operator-capabilities/v2";
        public const string PreflightContract = "aiimage.model-preflight/v1";

        private static readonly HashSet<string> UnsupportedOperators = new HashSet<string>(StringComparer.Ordinal)
        {
            "NonZero", "Compress", "GatherND", "Scatter", "ScatterElements", "ScatterND"
        };

        private static readonly HashSet<string> AliasOnlyOperators = new HashSet<string>(StringComparer.Ordinal)
        {
            "Input", "Noop", "Split", "DeepCopy", "Dropout"
        };

        private static readonly HashSet<string> SentisOperators = new HashSet<string>(StringComparer.Ordinal)
        {
            "Shape", "Size", "Range", "ConstantOfShape", "Expand", "ArgMax", "ArgMin", "Where",
            "TopK", "NonZero", "OneHot", "CumSum", "Compress", "Gather", "GatherElements", "GatherND",
            "Scatter", "ScatterElements", "ScatterND"
        };

        private static readonly HashSet<string> TextureAndCommandBufferOperators = new HashSet<string>(StringComparer.Ordinal)
        {
            "AbsVal", "AtenTo", "BatchNorm", "BinaryOp", "Cast", "Clip", "Concat", "Convolution",
            "Convolution1D", "Convolution3D", "ConvolutionDepthWise", "Crop", "Deconvolution",
            "Deconvolution3D", "DeconvolutionDepthWise", "Eltwise", "ExpandDims", "Flatten", "GELU",
            "Gemm", "GroupNorm", "InnerProduct", "Interp", "LayerNorm", "MatMul", "Packing", "Padding",
            "Permute", "PixelShuffle", "Pooling", "Pooling3D", "PReLU", "Quantize", "Dequantize",
            "Requantize", "Reduction", "ReLU", "Reorg", "Reshape", "Scale", "Sigmoid", "Slice", "Softmax",
            "Swish", "Tile", "UnaryOp", "Unfold", "MemoryData", "Shape", "Size", "Range", "ConstantOfShape", "Expand",
            "ArgMax", "ArgMin", "Where", "TopK", "OneHot", "CumSum", "Gather", "GatherElements"
        };

        private static readonly Dictionary<string, string[]> RequiredParameters = new Dictionary<string, string[]>(StringComparer.Ordinal)
        {
            { "Convolution", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size" } },
            { "ConvolutionDepthWise", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size" } },
            { "Convolution1D", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size" } },
            { "Convolution3D", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size" } },
            { "Deconvolution", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size" } },
            { "DeconvolutionDepthWise", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size" } },
            { "Deconvolution3D", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size" } },
            { "Pooling3D", new[] { "0:pooling_type", "1:kernel_w" } },
            { "Interp", new[] { "0:resize_type" } },
            { "InnerProduct", new[] { "0:num_output", "1:bias_term", "2:weight_data_size" } },
        };

        public static NcnnOperatorCapabilityDocument CreateDocument()
        {
            var operators = new List<NcnnOperatorCapability>();
            var registered = NcnnLayerFactoryRepro.GetRegisteredLayerTypes();
            for (var i = 0; i < registered.Count; i++)
                operators.Add(CreateCapability(registered[i].ToString()));

            return new NcnnOperatorCapabilityDocument
            {
                schemaVersion = SchemaVersion,
                contract = Contract,
                operators = operators.OrderBy(capability => capability.operatorName, StringComparer.Ordinal).ToArray()
            };
        }

        public static bool TryGet(string operatorName, out NcnnOperatorCapability capability)
        {
            capability = null;
            if (string.IsNullOrWhiteSpace(operatorName))
                return false;

            if (!NcnnLayerFactoryRepro.IsRegistered(NcnnLayerTypeKey.FromString(operatorName)))
                return false;

            capability = CreateCapability(operatorName);
            return true;
        }

        public static string ToStableJson(NcnnOperatorCapabilityDocument document, bool prettyPrint = true)
        {
            if (document == null)
                throw new ArgumentNullException(nameof(document));
            return JsonUtility.ToJson(document, prettyPrint) + "\n";
        }

        public static void WriteStableJson(string path, NcnnOperatorCapabilityDocument document)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("An output path is required.", nameof(path));

            var directory = Path.GetDirectoryName(Path.GetFullPath(path));
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);
            File.WriteAllText(path, ToStableJson(document), new System.Text.UTF8Encoding(false));
        }

        public static bool IsStrictlySupported(NcnnOperatorCapability capability, string targetBackend, string targetDtype)
        {
            return IsStrictlySupported(capability, targetBackend, targetDtype, null);
        }

        public static bool IsStrictlySupported(
            NcnnOperatorCapability capability,
            string targetBackend,
            string targetDtype,
            string targetLayout)
        {
            if (capability == null || !string.Equals(capability.status, NcnnOperatorCapabilityStatus.Supported, StringComparison.Ordinal))
                return false;

            if (string.Equals(targetBackend, NcnnOperatorCapabilityBackend.CommandBuffer, StringComparison.Ordinal)
                && !capability.commandBuffer)
                return false;
            if (string.Equals(targetBackend, NcnnOperatorCapabilityBackend.RenderTexture, StringComparison.Ordinal)
                && !capability.renderTexture)
                return false;

            var dtypeSupported = string.Equals(targetDtype, "FP32", StringComparison.OrdinalIgnoreCase) ? capability.fp32
                : string.Equals(targetDtype, "FP16", StringComparison.OrdinalIgnoreCase) ? capability.fp16
                : string.Equals(targetDtype, "INT8", StringComparison.OrdinalIgnoreCase) && capability.int8;
            if (!dtypeSupported)
                return false;

            return string.IsNullOrWhiteSpace(targetLayout)
                || (capability.layouts ?? Array.Empty<string>()).Any(layout => string.Equals(layout, targetLayout, StringComparison.OrdinalIgnoreCase));
        }

        private static NcnnOperatorCapability CreateCapability(string operatorName)
        {
            var isUnsupported = UnsupportedOperators.Contains(operatorName);
            var isAliasOnly = AliasOnlyOperators.Contains(operatorName);
            var hasTexturePath = TextureAndCommandBufferOperators.Contains(operatorName);
            var isSentis = SentisOperators.Contains(operatorName);
            // These two pointwise paths record a complete production contract. Other texture
            // entries may expose an FP16 Pack4 branch, but remain partial until the loaded
            // runtime profile proves that a concrete node cannot reach a fallback.
            var hasVerifiedCommandBufferPack4 = operatorName == "ReLU" || operatorName == "Sigmoid";
            var status = isUnsupported
                ? NcnnOperatorCapabilityStatus.Unsupported
                : isAliasOnly
                    ? NcnnOperatorCapabilityStatus.AliasOnly
                    : hasVerifiedCommandBufferPack4
                        ? NcnnOperatorCapabilityStatus.Supported
                    : hasTexturePath
                        ? NcnnOperatorCapabilityStatus.Partial
                        : NcnnOperatorCapabilityStatus.DebugOnly;

            return new NcnnOperatorCapability
            {
                operatorName = operatorName,
                canonicalOperator = ResolveCanonicalOperator(operatorName),
                importFormats = isSentis ? new[] { "ncnn", "Sentis/ONNX" } : new[] { "ncnn" },
                importSupported = true,
                // The current runtime resolves most shapes while executing. There is no complete static shape engine yet.
                shapeInference = isAliasOnly,
                renderTexture = hasTexturePath && !isUnsupported,
                commandBuffer = hasTexturePath && !isUnsupported,
                fp32 = hasTexturePath && !isUnsupported,
                fp16 = hasTexturePath && !isUnsupported,
                int8 = false,
                layouts = ResolveLayouts(operatorName),
                ranks = ResolveRanks(operatorName),
                verifiedModels = hasVerifiedCommandBufferPack4
                    ? new[] { "pack4-command-buffer-smoke" }
                    : Array.Empty<string>(),
                status = status,
                limitations = ResolveLimitations(operatorName, status),
                requiredParameters = RequiredParameters.TryGetValue(operatorName, out var parameters) ? parameters : Array.Empty<string>(),
                profiles = ResolveProfiles(operatorName)
            };
        }

        private static string ResolveCanonicalOperator(string operatorName)
        {
            switch (operatorName)
            {
                case "Convolution": return "Conv2D";
                case "ConvolutionDepthWise": return "DepthwiseConv2D";
                case "Convolution1D": return "Conv1D";
                case "Convolution3D": return "Conv3D";
                case "Deconvolution": return "ConvTranspose2D";
                case "DeconvolutionDepthWise": return "DepthwiseConvTranspose2D";
                case "Deconvolution3D": return "ConvTranspose3D";
                case "InnerProduct": return "FullyConnected";
                case "BinaryOp": return "BinaryElementwise";
                case "UnaryOp": return "UnaryElementwise";
                case "Interp": return "Resize";
                case "PnnxExpression": return "Expression";
                case "AtenTo": return "Cast";
                default: return operatorName;
            }
        }

        private static string[] ResolveLayouts(string operatorName)
        {
            if (operatorName == "Convolution3D" || operatorName == "Deconvolution3D" || operatorName == "Pooling3D")
                return new[] { "CDHW", "Packed4" };
            if (operatorName == "Interp")
                return new[] { "NCHW", "CDHW", "Packed4" };
            if (operatorName == "Gemm" || operatorName == "MatMul" || operatorName == "InnerProduct")
                return new[] { "Linear", "Packed4" };
            if (SentisOperators.Contains(operatorName))
                return new[] { "Scalar", "Linear", "Packed4" };
            return new[] { "NCHW", "Packed4" };
        }

        private static int[] ResolveRanks(string operatorName)
        {
            if (operatorName == "Convolution3D" || operatorName == "Deconvolution3D" || operatorName == "Pooling3D")
                return new[] { 5 };
            if (operatorName == "Gemm" || operatorName == "MatMul" || operatorName == "InnerProduct")
                return new[] { 1, 2, 3 };
            if (SentisOperators.Contains(operatorName))
                return new[] { 1, 2, 3, 4, 5 };
            return new[] { 1, 2, 3, 4 };
        }

        private static string ResolveLimitations(string operatorName, string status)
        {
            if (status == NcnnOperatorCapabilityStatus.Unsupported)
                return operatorName + " is registered only to report a deterministic error; its texture-native implementation is absent.";
            if (status == NcnnOperatorCapabilityStatus.AliasOnly)
                return "Only alias/view semantics are known. Strict planning requires a separately proven logical/storage layout match.";
            if (status == NcnnOperatorCapabilityStatus.DebugOnly)
                return "The factory entry exists, but no verified Pack4 RenderTexture and CommandBuffer production contract is recorded.";
            if (status == NcnnOperatorCapabilityStatus.Supported)
                return "Verified FP16 Pack4 CommandBuffer pointwise dispatch. Other dtype/layout combinations remain unsupported.";
            if (operatorName == "Convolution" || operatorName == "ConvolutionDepthWise"
                || operatorName == "Deconvolution" || operatorName == "DeconvolutionDepthWise")
            {
                return "A runtime-verified 2D CommandBuffer Pack4 profile supports immutable scalar OIHW weights, "
                    + "texture-native activations/outputs, positive rectangular kernel/stride/dilation, non-negative explicit padding, "
                    + "groups dividing input/output channels, optional bias, and activation none/ReLU/LeakyReLU/Sigmoid. "
                    + "Input/output tails use ceil(channel/4) packs and are zeroed. Auto/negative padding, unsupported activations, "
                    + "and invalid group/weight profiles fail strict planning; FP16 remains unvalidated by this C1 contract.";
            }
            if (operatorName == "Gemm" || operatorName == "MatMul" || operatorName == "InnerProduct")
                return "Partial CommandBuffer Pack4 support: Gemm/InnerProduct require loaded immutable FP32 weights and verified LinearMat or attention Pack4 storage; MatMul supports Pack4 rank-3/rank-4 matrices with transB and broadcast batch dimensions. Unsupported profiles fail strict planning without Buffer materialization.";
            if (operatorName == "LayerNorm" || operatorName == "Softmax" || operatorName == "Reduction"
                || operatorName == "MultiHeadAttention" || operatorName == "SDPA")
                return "Partial CommandBuffer Pack4 support: LayerNorm/Softmax use FP32 accumulation; Reduction covers scalar rank-2 and Pack4 spatial SUM/MEAN; SDPA/MHA support texture-native masks where their descriptor profiles prove it. KV-cache, unlisted axes/ranks, and unsupported dtype/layout profiles fail strict planning.";
            if (operatorName == "Convolution3D")
                return "Partial until the loaded node proves the group=1 OIDHW profile, explicit non-negative W/H/D padding, positive kernel/stride/dilation, supported activation, and TensorDescriptor CDHW Pack4 storage. Strict planning rejects every other branch.";
            if (operatorName == "Deconvolution3D")
                return "Partial until the loaded node proves the group=1 OIDHW profile, explicit non-negative W/H/D padding, zero output padding, positive kernel/stride/dilation, supported activation, and TensorDescriptor CDHW Pack4 storage. Strict planning rejects every other branch.";
            if (operatorName == "Pooling3D")
                return "Partial until the loaded node proves max/average global, adaptive, or explicit/full/SAME W/H/D pooling with TensorDescriptor CDHW Pack4 storage. Strict planning rejects invalid padding and all unlisted modes.";
            if (operatorName == "Interp")
                return "2D paths remain partial. The CDHW runtime profile is static nearest (1) or trilinear (2) resize with align_corners explicitly recorded, TensorDescriptor Pack4 storage, and no dynamic size expression; strict planning rejects other CDHW modes.";
            return "A texture branch may exist for selected shapes, but this entry has not passed full Pack4 CommandBuffer model validation. Strict planning rejects partial capability.";
        }

        private static NcnnOperatorCapabilityProfile[] ResolveProfiles(string operatorName)
        {
            const string CdhwShape = "logical [dims=4,w,h,d,c]; storage [dims=4,w,h,d,c]; Texture2DArray slices=d*ceil(c/4)";
            switch (operatorName)
            {
                case "Convolution3D":
                    return new[]
                    {
                        new NcnnOperatorCapabilityProfile
                        {
                            backend = NcnnOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "CDHW", "Packed4" },
                            shapeProfile = CdhwShape,
                            supportedParameters = new[] { "group=1", "kernel_w/kernel_h/kernel_d>0", "stride_w/stride_h/stride_d>0", "dilation_w/dilation_h/dilation_d>0", "explicit pad_left/right/top/bottom/front/behind>=0", "bias optional", "activation none/ReLU/LeakyReLU/Sigmoid" },
                            rejectedParameters = new[] { "group!=1", "negative/auto padding", "unsupported activation", "weight profile mismatch" }
                        }
                    };
                case "Deconvolution3D":
                    return new[]
                    {
                        new NcnnOperatorCapabilityProfile
                        {
                            backend = NcnnOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "CDHW", "Packed4" },
                            shapeProfile = CdhwShape,
                            supportedParameters = new[] { "group=1", "kernel_w/kernel_h/kernel_d>0", "stride_w/stride_h/stride_d>0", "dilation_w/dilation_h/dilation_d>0", "explicit pad_left/right/top/bottom/front/behind>=0", "output_padding=0", "bias optional", "activation none/ReLU/LeakyReLU/Sigmoid" },
                            rejectedParameters = new[] { "group!=1", "non-zero output padding", "negative/auto padding", "unsupported activation", "weight profile mismatch" }
                        }
                    };
                case "Pooling3D":
                    return new[]
                    {
                        new NcnnOperatorCapabilityProfile
                        {
                            backend = NcnnOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "CDHW", "Packed4" },
                            shapeProfile = CdhwShape,
                            supportedParameters = new[] { "pooling_type=max|average", "global or adaptive or explicit/full/SAME_UPPER/SAME_LOWER W/H/D kernel/stride/pad", "include_pad for average" },
                            rejectedParameters = new[] { "invalid pad mode", "negative padding", "kernel larger than padded input" }
                        }
                    };
                case "Interp":
                    return new[]
                    {
                        new NcnnOperatorCapabilityProfile
                        {
                            backend = NcnnOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "CDHW", "Packed4" },
                            shapeProfile = CdhwShape,
                            supportedParameters = new[] { "resize_type=1 nearest|2 trilinear", "static output W/H/D or positive scale W/H/D", "align_corners=0|1" },
                            rejectedParameters = new[] { "dynamic_target_size", "size_expr", "bicubic/other resize modes" }
                        }
                    };
                default:
                    return Array.Empty<NcnnOperatorCapabilityProfile>();
            }
        }
    }

    public static class NcnnModelPreflight
    {
        public static NcnnModelPreflightReport Analyze(NcnnParamModel model, NcnnModelPreflightRequest request)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            request ??= new NcnnModelPreflightRequest();

            var inputByBlob = new Dictionary<string, NcnnPreflightTensorDescriptor>(StringComparer.Ordinal);
            var declaredInputs = (request.inputs ?? Array.Empty<NcnnPreflightTensorDescriptor>())
                .Where(input => input != null && !string.IsNullOrWhiteSpace(input.blob))
                .Select(CloneDescriptor)
                .OrderBy(input => input.blob, StringComparer.Ordinal)
                .ToArray();
            for (var i = 0; i < declaredInputs.Length; i++)
                inputByBlob[declaredInputs[i].blob] = declaredInputs[i];

            var knownBlobs = new Dictionary<string, NcnnPreflightTensorDescriptor>(inputByBlob, StringComparer.Ordinal);
            var missingNodes = new List<NcnnModelPreflightIssue>();
            var missingParameters = new List<NcnnModelPreflightIssue>();
            var missingDependencies = new List<NcnnModelPreflightIssue>();
            var nodes = new List<NcnnModelPreflightNode>();
            var layers = model.layers ?? new List<NcnnParamModel.Layer>();

            for (var index = 0; index < layers.Count; index++)
            {
                var layer = layers[index];
                if (layer == null)
                {
                    missingNodes.Add(CreateIssue(index, string.Empty, string.Empty, string.Empty, "null-layer", "The model contains a null layer.", "Re-export the model graph."));
                    continue;
                }

                var operatorName = !string.IsNullOrWhiteSpace(layer.typeName) ? layer.typeName : layer.type.ToString();
                if (!NcnnOperatorCapabilities.TryGet(operatorName, out var capability))
                {
                    var issue = CreateIssue(index, layer.name, operatorName, string.Empty, "unregistered-operator", "No factory/capability entry exists for this operator.", "Add a texture-native implementation and capability entry, or re-export with a supported operator.");
                    missingNodes.Add(issue);
                    nodes.Add(CreateUnknownNode(index, layer, operatorName, issue.message, issue.recommendedAction));
                    RegisterUnknownOutputDescriptors(layer, knownBlobs);
                    continue;
                }

                var nodeIssues = new List<string>();
                var nodeInputs = ResolveInputs(layer, knownBlobs, nodeIssues, index, operatorName, missingDependencies);
                ReportMissingInputDescriptors(layer, capability, inputByBlob, nodeIssues, index, operatorName, missingDependencies);
                var missingForNode = FindMissingParameters(layer, capability.requiredParameters);
                for (var parameterIndex = 0; parameterIndex < missingForNode.Length; parameterIndex++)
                {
                    var issue = CreateIssue(index, layer.name, operatorName, string.Empty, "missing-parameter", "Missing required parameter " + missingForNode[parameterIndex] + ".", "Provide the required ncnn parameter or re-export the layer.");
                    missingParameters.Add(issue);
                    nodeIssues.Add(issue.message);
                }

                var strictEligible = !request.strict || (NcnnOperatorCapabilities.IsStrictlySupported(capability, request.targetBackend, request.targetDtype) && missingForNode.Length == 0 && nodeIssues.Count == 0);
                var recommendation = ResolveRecommendedAction(capability, request.targetBackend, request.targetDtype, missingForNode.Length > 0);
                if (request.strict && !strictEligible && nodeIssues.Count == 0)
                    nodeIssues.Add("Strict target rejects status=" + capability.status + " for backend=" + request.targetBackend + " dtype=" + request.targetDtype + ".");

                nodes.Add(new NcnnModelPreflightNode
                {
                    layerIndex = index,
                    layer = layer.name ?? string.Empty,
                    operatorName = operatorName,
                    canonicalOperator = capability.canonicalOperator,
                    status = capability.status,
                    strictEligible = strictEligible,
                    bottomBlobs = CloneStrings(layer.bottomNames),
                    topBlobs = CloneStrings(layer.topNames),
                    inputs = nodeInputs.ToArray(),
                    missingParameters = missingForNode,
                    issues = nodeIssues.ToArray(),
                    recommendedAction = recommendation
                });

                PropagateOutputDescriptors(layer, capability, nodeInputs, knownBlobs);
            }

            var report = new NcnnModelPreflightReport
            {
                schemaVersion = NcnnOperatorCapabilities.SchemaVersion,
                contract = NcnnOperatorCapabilities.PreflightContract,
                modelName = request.modelName ?? string.Empty,
                targetBackend = request.targetBackend ?? string.Empty,
                targetDtype = request.targetDtype ?? string.Empty,
                strict = request.strict,
                declaredInputs = declaredInputs,
                nodes = nodes.ToArray(),
                missingNodes = missingNodes.ToArray(),
                missingParameters = missingParameters.ToArray(),
                missingDependencies = missingDependencies.ToArray(),
                strictEligible = missingNodes.Count == 0 && missingParameters.Count == 0 && missingDependencies.Count == 0 && nodes.All(node => node.strictEligible),
            };
            report.summary = "nodes=" + report.nodes.Length.ToString(CultureInfo.InvariantCulture)
                + " missingNodes=" + report.missingNodes.Length.ToString(CultureInfo.InvariantCulture)
                + " missingParameters=" + report.missingParameters.Length.ToString(CultureInfo.InvariantCulture)
                + " missingDependencies=" + report.missingDependencies.Length.ToString(CultureInfo.InvariantCulture)
                + " strictEligible=" + (report.strictEligible ? "true" : "false");
            return report;
        }

        public static string ToStableJson(NcnnModelPreflightReport report, bool prettyPrint = true)
        {
            if (report == null)
                throw new ArgumentNullException(nameof(report));
            return JsonUtility.ToJson(report, prettyPrint) + "\n";
        }

        public static void WriteStableJson(string path, NcnnModelPreflightReport report)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("An output path is required.", nameof(path));
            var directory = Path.GetDirectoryName(Path.GetFullPath(path));
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);
            File.WriteAllText(path, ToStableJson(report), new System.Text.UTF8Encoding(false));
        }

        private static List<NcnnPreflightTensorDescriptor> ResolveInputs(
            NcnnParamModel.Layer layer,
            Dictionary<string, NcnnPreflightTensorDescriptor> knownBlobs,
            List<string> nodeIssues,
            int layerIndex,
            string operatorName,
            List<NcnnModelPreflightIssue> missingDependencies)
        {
            var result = new List<NcnnPreflightTensorDescriptor>();
            var bottoms = layer.bottomNames ?? Array.Empty<string>();
            for (var i = 0; i < bottoms.Length; i++)
            {
                var bottom = bottoms[i] ?? string.Empty;
                if (knownBlobs.TryGetValue(bottom, out var descriptor))
                {
                    result.Add(CloneDescriptor(descriptor));
                    continue;
                }

                var issue = CreateIssue(layerIndex, layer.name, operatorName, bottom, "missing-input-blob", "No declared input or earlier producer exists for bottom blob " + bottom + ".", "Declare its logical/storage shape, or fix the graph connection.");
                missingDependencies.Add(issue);
                nodeIssues.Add(issue.message);
                result.Add(new NcnnPreflightTensorDescriptor { blob = bottom, logicalShape = Array.Empty<int>(), storageShape = Array.Empty<int>(), layout = "Unknown", dtype = "Unknown" });
            }
            return result;
        }

        private static void ReportMissingInputDescriptors(
            NcnnParamModel.Layer layer,
            NcnnOperatorCapability capability,
            Dictionary<string, NcnnPreflightTensorDescriptor> inputByBlob,
            List<string> nodeIssues,
            int layerIndex,
            string operatorName,
            List<NcnnModelPreflightIssue> missingDependencies)
        {
            if (!string.Equals(capability.operatorName, "Input", StringComparison.Ordinal) || layer.topNames == null)
                return;

            for (var i = 0; i < layer.topNames.Length; i++)
            {
                var top = layer.topNames[i];
                if (string.IsNullOrWhiteSpace(top) || inputByBlob.ContainsKey(top))
                    continue;

                var issue = CreateIssue(
                    layerIndex,
                    layer.name,
                    operatorName,
                    top,
                    "missing-input-descriptor",
                    "No logical/storage shape descriptor was supplied for model input " + top + ".",
                    "Supply a Pack4 texture input descriptor with logical shape, storage shape, layout, and dtype.");
                missingDependencies.Add(issue);
                nodeIssues.Add(issue.message);
            }
        }

        private static void PropagateOutputDescriptors(
            NcnnParamModel.Layer layer,
            NcnnOperatorCapability capability,
            List<NcnnPreflightTensorDescriptor> inputs,
            Dictionary<string, NcnnPreflightTensorDescriptor> knownBlobs)
        {
            if (layer.topNames == null)
                return;

            for (var i = 0; i < layer.topNames.Length; i++)
            {
                if (!string.IsNullOrWhiteSpace(layer.topNames[i]))
                {
                    if (string.Equals(capability.operatorName, "Input", StringComparison.Ordinal)
                        && knownBlobs.ContainsKey(layer.topNames[i]))
                    {
                        continue;
                    }

                    var descriptor = capability.shapeInference && inputs.Count > 0
                        ? CloneDescriptor(inputs[0])
                        : new NcnnPreflightTensorDescriptor
                        {
                            blob = layer.topNames[i],
                            logicalShape = Array.Empty<int>(),
                            storageShape = Array.Empty<int>(),
                            layout = "Unknown",
                            dtype = "Unknown"
                        };
                    descriptor.blob = layer.topNames[i];
                    knownBlobs[descriptor.blob] = descriptor;
                }
            }
        }

        private static void RegisterUnknownOutputDescriptors(
            NcnnParamModel.Layer layer,
            Dictionary<string, NcnnPreflightTensorDescriptor> knownBlobs)
        {
            if (layer.topNames == null)
                return;

            for (var i = 0; i < layer.topNames.Length; i++)
            {
                var top = layer.topNames[i];
                if (string.IsNullOrWhiteSpace(top))
                    continue;
                knownBlobs[top] = new NcnnPreflightTensorDescriptor
                {
                    blob = top,
                    logicalShape = Array.Empty<int>(),
                    storageShape = Array.Empty<int>(),
                    layout = "Unknown",
                    dtype = "Unknown"
                };
            }
        }

        private static string[] FindMissingParameters(NcnnParamModel.Layer layer, string[] requiredParameters)
        {
            if (requiredParameters == null || requiredParameters.Length == 0)
                return Array.Empty<string>();

            var missing = new List<string>();
            for (var i = 0; i < requiredParameters.Length; i++)
            {
                var keyAndName = requiredParameters[i];
                var separator = keyAndName.IndexOf(':');
                var keyText = separator < 0 ? keyAndName : keyAndName.Substring(0, separator);
                if (!int.TryParse(keyText, NumberStyles.Integer, CultureInfo.InvariantCulture, out var key)
                    || layer.intParams == null
                    || !layer.intParams.ContainsKey(key))
                {
                    missing.Add(keyAndName);
                }
            }
            return missing.ToArray();
        }

        private static NcnnModelPreflightNode CreateUnknownNode(int index, NcnnParamModel.Layer layer, string operatorName, string message, string action)
        {
            return new NcnnModelPreflightNode
            {
                layerIndex = index,
                layer = layer.name ?? string.Empty,
                operatorName = operatorName,
                canonicalOperator = operatorName,
                status = NcnnOperatorCapabilityStatus.Unsupported,
                strictEligible = false,
                bottomBlobs = CloneStrings(layer.bottomNames),
                topBlobs = CloneStrings(layer.topNames),
                inputs = Array.Empty<NcnnPreflightTensorDescriptor>(),
                missingParameters = Array.Empty<string>(),
                issues = new[] { message },
                recommendedAction = action
            };
        }

        private static NcnnModelPreflightIssue CreateIssue(int index, string layer, string operatorName, string blob, string code, string message, string action)
        {
            return new NcnnModelPreflightIssue
            {
                layerIndex = index,
                layer = layer ?? string.Empty,
                operatorName = operatorName ?? string.Empty,
                blob = blob ?? string.Empty,
                code = code,
                message = message,
                recommendedAction = action
            };
        }

        private static string ResolveRecommendedAction(NcnnOperatorCapability capability, string backend, string dtype, bool missingParameters)
        {
            if (missingParameters)
                return "Provide the missing parameters before execution planning.";
            if (capability.status == NcnnOperatorCapabilityStatus.Unsupported)
                return "Do not execute or materialize to Buffer; implement the Pack4 RenderTexture/CommandBuffer operator first.";
            if (capability.status == NcnnOperatorCapabilityStatus.Partial)
                return "Keep this node out of strict plans until its exact Pack4 RenderTexture/CommandBuffer shape and dtype contract is implemented and validated.";
            if (capability.status == NcnnOperatorCapabilityStatus.AliasOnly)
                return "Prove compatible logical/storage shape and layout for aliasing, or implement a real Pack4 transform. Do not use a Buffer fallback.";
            if (capability.status == NcnnOperatorCapabilityStatus.DebugOnly)
                return "This is debug-only. Add a Pack4 RenderTexture/CommandBuffer path; do not select its legacy Buffer path for production.";
            return "Target backend/dtype requires further validation: " + backend + "/" + dtype + ".";
        }

        private static NcnnPreflightTensorDescriptor CloneDescriptor(NcnnPreflightTensorDescriptor source)
        {
            if (source == null)
                return new NcnnPreflightTensorDescriptor { blob = string.Empty, logicalShape = Array.Empty<int>(), storageShape = Array.Empty<int>(), layout = "Unknown", dtype = "Unknown" };
            return new NcnnPreflightTensorDescriptor
            {
                blob = source.blob ?? string.Empty,
                logicalShape = source.logicalShape != null ? (int[])source.logicalShape.Clone() : Array.Empty<int>(),
                storageShape = source.storageShape != null ? (int[])source.storageShape.Clone() : Array.Empty<int>(),
                layout = source.layout ?? "Unknown",
                dtype = source.dtype ?? "Unknown"
            };
        }

        private static string[] CloneStrings(string[] values)
        {
            return values == null ? Array.Empty<string>() : values.Select(value => value ?? string.Empty).ToArray();
        }
    }
}
