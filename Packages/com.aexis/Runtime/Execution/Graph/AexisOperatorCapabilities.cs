using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Aexis;
using Aexis.Onnx;
using UnityEngine;

namespace Aexis.Execution
{
    public static class AexisOperatorCapabilityStatus
    {
        public const string Supported = "supported";
        public const string Partial = "partial";
        public const string AliasOnly = "alias-only";
        public const string DebugOnly = "debug-only";
        public const string Unsupported = "unsupported";
    }

    public static class AexisOperatorCapabilityBackend
    {
        public const string RenderTexture = "render-texture";
        public const string CommandBuffer = "command-buffer";
    }

    [Serializable]
    public sealed class AexisOperatorCapabilityProfile
    {
        public string backend;
        public string[] layouts;
        public string shapeProfile;
        public string[] supportedParameters;
        public string[] rejectedParameters;
    }

    [Serializable]
    public sealed class AexisOperatorCapability
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
        public AexisOperatorCapabilityProfile[] profiles;
    }

    [Serializable]
    public sealed class AexisOperatorCapabilityDocument
    {
        public int schemaVersion;
        public string contract;
        public AexisOperatorCapability[] operators;
    }

    [Serializable]
    public sealed class AexisPreflightTensorDescriptor
    {
        public string blob;
        public int[] logicalShape;
        public int[] storageShape;
        public string layout;
        public string dtype;
    }

    [Serializable]
    public sealed class AexisModelPreflightRequest
    {
        public string modelName;
        public string targetBackend = AexisOperatorCapabilityBackend.CommandBuffer;
        public string targetDtype = "FP32";
        public bool strict = true;
        public AexisPreflightTensorDescriptor[] inputs = Array.Empty<AexisPreflightTensorDescriptor>();
    }

    [Serializable]
    public sealed class AexisModelPreflightIssue
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
    public sealed class AexisModelPreflightNode
    {
        public int layerIndex;
        public string layer;
        public string operatorName;
        public string canonicalOperator;
        public string status;
        public bool strictEligible;
        public string[] bottomBlobs;
        public string[] topBlobs;
        public AexisPreflightTensorDescriptor[] inputs;
        public string[] missingParameters;
        public string[] issues;
        public string recommendedAction;
    }

    [Serializable]
    public sealed class AexisModelPreflightReport
    {
        public int schemaVersion;
        public string contract;
        public string modelName;
        public string targetBackend;
        public string targetDtype;
        public bool strict;
        public AexisPreflightTensorDescriptor[] declaredInputs;
        public AexisModelPreflightNode[] nodes;
        public AexisModelPreflightIssue[] missingNodes;
        public AexisModelPreflightIssue[] missingParameters;
        public AexisModelPreflightIssue[] missingDependencies;
        public bool strictEligible;
        public string summary;
    }

    // This is intentionally metadata-only. It never creates a layer, AexisGraphSession, texture, or buffer.
    public static class AexisOperatorCapabilities
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

        private static readonly HashSet<string> Pack4LayoutOperators = new HashSet<string>(StringComparer.Ordinal)
        {
            "Reshape", "Flatten", "Squeeze", "ExpandDims", "Permute", "Slice", "Tile", "Packing", "Cast"
        };

        private static readonly HashSet<string> TextureAndCommandBufferOperators = new HashSet<string>(StringComparer.Ordinal)
        {
            "AbsVal", "aten::to", "BatchNorm", "BinaryOp", "Cast", "Clip", "Concat", "Convolution",
            "Convolution1D", "Convolution3D", "ConvolutionDepthWise", "Crop", "Deconvolution",
            "Deconvolution3D", "DeconvolutionDepthWise", "Eltwise", "ExpandDims", "Flatten", "GELU", "Squeeze",
            "Gemm", "GroupNorm", "InnerProduct", "Interp", "LayerNorm", "MatMul", "Packing", "Padding",
            "MaxPoolingInd", "MaxUnPooling", "Permute", "PixelShuffle", "Pooling", "Pooling3D", "PReLU", "Quantize", "Dequantize",
            "Requantize", "Reduction", "ReLU", "Reorg", "Reshape", "Scale", "Sigmoid", "Slice", "Softmax",
            "Swish", "Tile", "UnaryOp", "Unfold", "MemoryData", "Shape", "Size", "Range", "ConstantOfShape", "Expand",
            "ArgMax", "ArgMin", "Where", "TopK", "OneHot", "CumSum", "Gather", "GatherElements", "pnnx.Expression"
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

        public static AexisOperatorCapabilityDocument CreateDocument()
        {
            var operators = new List<AexisOperatorCapability>();
            var registered = AexisLayerFactory.GetRegisteredLayerTypes();
            for (var i = 0; i < registered.Count; i++)
                operators.Add(CreateCapability(registered[i].ToString()));

            return new AexisOperatorCapabilityDocument
            {
                schemaVersion = SchemaVersion,
                contract = Contract,
                operators = operators.OrderBy(capability => capability.operatorName, StringComparer.Ordinal).ToArray()
            };
        }

        public static bool TryGet(string operatorName, out AexisOperatorCapability capability)
        {
            capability = null;
            if (string.IsNullOrWhiteSpace(operatorName))
                return false;

            if (!AexisLayerFactory.IsRegistered(AexisLayerTypeKey.FromString(operatorName)))
                return false;

            capability = CreateCapability(operatorName);
            return true;
        }

        public static string ToStableJson(AexisOperatorCapabilityDocument document, bool prettyPrint = true)
        {
            if (document == null)
                throw new ArgumentNullException(nameof(document));
            return JsonUtility.ToJson(document, prettyPrint) + "\n";
        }

        public static void WriteStableJson(string path, AexisOperatorCapabilityDocument document)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("An output path is required.", nameof(path));

            var directory = Path.GetDirectoryName(Path.GetFullPath(path));
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);
            File.WriteAllText(path, ToStableJson(document), new System.Text.UTF8Encoding(false));
        }

        public static bool IsStrictlySupported(AexisOperatorCapability capability, string targetBackend, string targetDtype)
        {
            return IsStrictlySupported(capability, targetBackend, targetDtype, null);
        }

        public static bool IsStrictlySupported(
            AexisOperatorCapability capability,
            string targetBackend,
            string targetDtype,
            string targetLayout)
        {
            if (capability == null || !string.Equals(capability.status, AexisOperatorCapabilityStatus.Supported, StringComparison.Ordinal))
                return false;

            if (string.Equals(targetBackend, AexisOperatorCapabilityBackend.CommandBuffer, StringComparison.Ordinal)
                && !capability.commandBuffer)
                return false;
            if (string.Equals(targetBackend, AexisOperatorCapabilityBackend.RenderTexture, StringComparison.Ordinal)
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

        private static AexisOperatorCapability CreateCapability(string operatorName)
        {
            var isUnsupported = UnsupportedOperators.Contains(operatorName);
            var isAliasOnly = AliasOnlyOperators.Contains(operatorName);
            var hasTexturePath = TextureAndCommandBufferOperators.Contains(operatorName);
            var isSentis = SentisOperators.Contains(operatorName);
            var hasInt8QuantizedKernel = operatorName == "Convolution"
                || operatorName == "ConvolutionDepthWise"
                || operatorName == "Gemm"
                || operatorName == "InnerProduct";
            // These two pointwise paths record a complete production contract. Other texture
            // entries may expose an FP16 Pack4 branch, but remain partial until the loaded
            // runtime profile proves that a concrete node cannot reach a fallback.
            var hasVerifiedCommandBufferPack4 = operatorName == "ReLU" || operatorName == "Sigmoid";
            var status = isUnsupported
                ? AexisOperatorCapabilityStatus.Unsupported
                : isAliasOnly
                    ? AexisOperatorCapabilityStatus.AliasOnly
                    : hasVerifiedCommandBufferPack4
                        ? AexisOperatorCapabilityStatus.Supported
                    : hasTexturePath
                        ? AexisOperatorCapabilityStatus.Partial
                        : AexisOperatorCapabilityStatus.DebugOnly;

            return new AexisOperatorCapability
            {
                operatorName = operatorName,
                canonicalOperator = ResolveCanonicalOperator(operatorName),
                importFormats = isSentis ? new[] { "ncnn", "Sentis/ONNX" } : new[] { "ncnn" },
                importSupported = true,
                // The current runtime resolves most shapes while executing. There is no complete static shape engine yet.
                shapeInference = isAliasOnly || Pack4LayoutOperators.Contains(operatorName),
                renderTexture = hasTexturePath && !isUnsupported,
                commandBuffer = hasTexturePath && !isUnsupported,
                fp32 = hasTexturePath && !isUnsupported,
                fp16 = hasTexturePath && !isUnsupported,
                int8 = hasInt8QuantizedKernel && hasTexturePath && !isUnsupported,
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
                case "pnnx.Expression": return "Expression";
                case "aten::to": return "Cast";
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
                return new[] { 1, 2, 3, 4 };
            return new[] { 1, 2, 3, 4 };
        }

        private static string ResolveLimitations(string operatorName, string status)
        {
            if (status == AexisOperatorCapabilityStatus.Unsupported)
                return operatorName + " is registered only to report a deterministic error; its texture-native implementation is absent. "
                    + "D3 rejects this graph at plan time rather than reading GPU data back or falling back to a ComputeBuffer.";
            if (status == AexisOperatorCapabilityStatus.AliasOnly)
                return "Only alias/view semantics are known. Strict planning requires a separately proven logical/storage layout match.";
            if (status == AexisOperatorCapabilityStatus.DebugOnly)
                return "The factory entry exists, but no verified Pack4 RenderTexture and CommandBuffer production contract is recorded.";
            if (status == AexisOperatorCapabilityStatus.Supported)
                return "Verified FP16 Pack4 CommandBuffer pointwise dispatch. Other dtype/layout combinations remain unsupported.";
            if (operatorName == "pnnx.Expression")
                return "Partial CommandBuffer Pack4 support: a no-input constant scalar/list with at most four values is written by the real FillScalarTexture dispatch. Dynamic expressions and wider lists fail strict planning.";
            if (operatorName == "Convolution" || operatorName == "ConvolutionDepthWise"
                || operatorName == "Deconvolution" || operatorName == "DeconvolutionDepthWise")
            {
                return "A runtime-verified 2D CommandBuffer Pack4 profile supports immutable scalar OIHW weights, "
                    + "texture-native activations/outputs, positive rectangular kernel/stride/dilation, non-negative explicit padding, "
                    + "groups dividing input/output channels, optional bias, and activation none/ReLU/LeakyReLU/Sigmoid. "
                    + "Input/output tails use ceil(channel/4) packs and are zeroed. Auto/negative padding, unsupported activations, "
                    + "and invalid group/weight profiles fail strict planning. INT8 additionally requires packed signed INT8 OIHW, per-output-channel symmetric scales, "
                    + "optional calibrated W8A8 activation quantization from ModelManifest nodePlans, and FP32 texture accumulation.";
            }
            if (operatorName == "Gemm" || operatorName == "MatMul" || operatorName == "InnerProduct")
                return "Partial CommandBuffer Pack4 support: Gemm/InnerProduct use verified LinearMat or attention Pack4 storage. D2 accepts immutable packed INT8 weights, optional calibrated W8A8 activation quantization from ModelManifest nodePlans, per-output-channel symmetric scales, and FP32 accumulation; MatMul remains FP-only. Unsupported profiles fail strict planning without Buffer materialization.";
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
            if (Pack4LayoutOperators.Contains(operatorName))
                return "Partial CommandBuffer Pack4 layout profile: every branch validates logicalShape, storageShape, and layout. "
                    + "Only descriptor-proven identity/view branches alias; Permute, non-identity Slice/Tile/Packing/Cast, and storage-changing Reshape/Flatten dispatch a real texture transform. "
                    + "Dynamic data-dependent lengths and unsupported rank/axis/dtype profiles fail strict planning. Placeholder publication is not a production path.";
            if (operatorName == "TopK")
                return "Partial LinearMat texture profile: rank=1..4, FP32/FP16 values, Int32 logical indices, static axis and static k only. "
                    + "GPU-driven k requires a capacity-bounded GPU shape tensor; the current backend rejects it at plan time and never reads back k.";
            if (operatorName == "OneHot")
                return "Partial LinearMat texture profile: rank=1..3 indices, static positive depth and axis, FP32/FP16 values, and Int32 logical indices. "
                    + "GPU-driven depth requires a capacity-bounded GPU shape tensor; the current backend rejects it at plan time and never reads back depth.";
            if (operatorName == "RotaryEmbed")
                return "RotaryEmbed has no verified CommandBuffer Pack4 production profile and remains unavailable to strict plans; it must not be reported as a production placeholder.";
            return "A texture branch may exist for selected shapes, but this entry has not passed full Pack4 CommandBuffer model validation. Strict planning rejects partial capability.";
        }

        private static AexisOperatorCapabilityProfile[] ResolveProfiles(string operatorName)
        {
            const string CdhwShape = "logical [dims=4,w,h,d,c]; storage [dims=4,w,h,d,c]; Texture2DArray slices=d*ceil(c/4)";
            const string LayoutShape = "logical [dims,w,h,d,c]; storage explicitly records LinearMat or Pack4 Texture2DArray physical mapping; descriptor alias requires unchanged storage/layout/dtype";
            switch (operatorName)
            {
                case "pnnx.Expression":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Packed4" },
                            shapeProfile = "logical [dims=1,w=value_count,h=1,d=1,c=1]; storage [dims=3,w=value_count,h=1,d=1,c=1]; real scalar-fill texture dispatch",
                            supportedParameters = new[] { "constant expr", "no input blobs", "value_count=1..4" },
                            rejectedParameters = new[] { "dynamic expression", "input blobs", "value_count>4" }
                        }
                    };
                case "Reshape":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Packed4", "Linear" },
                            shapeProfile = LayoutShape,
                            supportedParameters = new[] { "static shape metadata", "supported texture shape tensor", "equal element count", "descriptor alias only when physical mapping is unchanged" },
                            rejectedParameters = new[] { "data-dependent output length", "element-count change", "unproven alias mapping" }
                        }
                    };
                case "Flatten":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Packed4", "Linear" },
                            shapeProfile = LayoutShape,
                            supportedParameters = new[] { "rank=1..4", "descriptor alias only for already-flat matching storage", "otherwise reshape Pack4 texture kernel" },
                            rejectedParameters = new[] { "buffer materialization", "placeholder output" }
                        }
                    };
                case "Squeeze":
                case "ExpandDims":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Packed4", "Linear" },
                            shapeProfile = LayoutShape,
                            supportedParameters = new[] { "static axes", "singleton-axis view", "rank remains within 1..4" },
                            rejectedParameters = new[] { "missing axes", "non-singleton squeeze axis", "rank outside 1..4" }
                        }
                    };
                case "Permute":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Packed4", "Linear" },
                            shapeProfile = LayoutShape,
                            supportedParameters = new[] { "rank=2..4", "order=0 descriptor alias", "non-identity Pack4/LinearMat texture transpose" },
                            rejectedParameters = new[] { "unsupported order", "unproven storage mapping", "placeholder output" }
                        }
                    };
                case "Slice":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Packed4", "Linear" },
                            shapeProfile = LayoutShape,
                            supportedParameters = new[] { "static split sizes or indices", "rank=1..4", "identity alias", "Pack4/CDHW/LinearMat texture copy" },
                            rejectedParameters = new[] { "data-dependent split length", "invalid axis or empty output", "placeholder output" }
                        }
                    };
                case "Tile":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Packed4", "Linear" },
                            shapeProfile = LayoutShape,
                            supportedParameters = new[] { "static repeats", "all repeats=1 descriptor alias", "Pack4/LinearMat texture tile" },
                            rejectedParameters = new[] { "data-dependent repeats", "rank outside 1..4", "placeholder output" }
                        }
                    };
                case "Packing":
                case "Cast":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Packed4", "Linear" },
                            shapeProfile = LayoutShape,
                            supportedParameters = new[] { "identity descriptor alias", "non-identity Pack4 texture repack/cast" },
                            rejectedParameters = new[] { "unsupported elempack/dtype", "buffer materialization", "placeholder output" }
                        }
                    };
                case "Convolution3D":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "CDHW", "Packed4" },
                            shapeProfile = CdhwShape,
                            supportedParameters = new[] { "group=1", "kernel_w/kernel_h/kernel_d>0", "stride_w/stride_h/stride_d>0", "dilation_w/dilation_h/dilation_d>0", "explicit pad_left/right/top/bottom/front/behind>=0", "bias optional", "activation none/ReLU/LeakyReLU/Sigmoid" },
                            rejectedParameters = new[] { "group!=1", "negative/auto padding", "unsupported activation", "weight profile mismatch" }
                        }
                    };
                case "Deconvolution3D":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "CDHW", "Packed4" },
                            shapeProfile = CdhwShape,
                            supportedParameters = new[] { "group=1", "kernel_w/kernel_h/kernel_d>0", "stride_w/stride_h/stride_d>0", "dilation_w/dilation_h/dilation_d>0", "explicit pad_left/right/top/bottom/front/behind>=0", "output_padding=0", "bias optional", "activation none/ReLU/LeakyReLU/Sigmoid" },
                            rejectedParameters = new[] { "group!=1", "non-zero output padding", "negative/auto padding", "unsupported activation", "weight profile mismatch" }
                        }
                    };
                case "Pooling3D":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "CDHW", "Packed4" },
                            shapeProfile = CdhwShape,
                            supportedParameters = new[] { "pooling_type=max|average", "global or adaptive or explicit/full/SAME_UPPER/SAME_LOWER W/H/D kernel/stride/pad", "include_pad for average" },
                            rejectedParameters = new[] { "invalid pad mode", "negative padding", "kernel larger than padded input" }
                        }
                    };
                case "Interp":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "CDHW", "Packed4" },
                            shapeProfile = CdhwShape,
                            supportedParameters = new[] { "resize_type=1 nearest|2 trilinear", "static output W/H/D or positive scale W/H/D", "align_corners=0|1" },
                            rejectedParameters = new[] { "dynamic_target_size", "size_expr", "bicubic/other resize modes" }
                        }
                    };
                case "TopK":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Linear" },
                            shapeProfile = "rank=1..4; output axis has static k; LinearMat values plus logical Int32 indices",
                            supportedParameters = new[] { "axis in [-rank,rank-1]", "static k in [1,axis_size]", "largest=0|1", "FP32/FP16 values" },
                            rejectedParameters = new[] { "dynamic k", "rank>4", "k>axis_size", "CPU readback for output shape" }
                        }
                    };
                case "OneHot":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Linear" },
                            shapeProfile = "indices rank=1..3; output rank=2..4; static depth axis inserted into output",
                            supportedParameters = new[] { "Int32 logical indices", "static depth>0", "axis in [-rank-1,rank]", "FP32/FP16 on/off values" },
                            rejectedParameters = new[] { "dynamic depth", "indices rank>3", "output rank>4", "CPU readback for output shape" }
                        }
                    };
                case "NonZero":
                case "Compress":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Linear" },
                            shapeProfile = "data-dependent output requires GPU prefix-sum/compaction and a capacity-bounded Int32 shape tensor",
                            supportedParameters = Array.Empty<string>(),
                            rejectedParameters = new[] { "all runtime profiles until compaction kernel is present", "CPU count readback", "unbounded output", "overflow without reject policy" }
                        }
                    };
                case "GatherND":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Linear" },
                            shapeProfile = "planned subset: data rank=1..4, Int32 indices, batch_dims=0",
                            supportedParameters = Array.Empty<string>(),
                            rejectedParameters = new[] { "all runtime profiles until GatherND texture kernel is present", "batch_dims!=0", "non-Int32 indices", "rank>4" }
                        }
                    };
                case "Scatter":
                case "ScatterElements":
                case "ScatterND":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Linear" },
                            shapeProfile = "planned subset: rank=1..4, Int32 indices, provably unique destinations; conflict semantics are reject",
                            supportedParameters = Array.Empty<string>(),
                            rejectedParameters = new[] { "all runtime profiles until deterministic texture writes are present", "duplicate destinations", "non-Int32 indices", "atomic or unspecified last-write semantics" }
                        }
                    };
                default:
                    return Array.Empty<AexisOperatorCapabilityProfile>();
            }
        }
    }

    public static class AexisModelPreflight
    {
        public static AexisModelPreflightReport Analyze(AexisGraphModel model, AexisModelPreflightRequest request)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            request ??= new AexisModelPreflightRequest();

            var inputByBlob = new Dictionary<string, AexisPreflightTensorDescriptor>(StringComparer.Ordinal);
            var declaredInputs = (request.inputs ?? Array.Empty<AexisPreflightTensorDescriptor>())
                .Where(input => input != null && !string.IsNullOrWhiteSpace(input.blob))
                .Select(CloneDescriptor)
                .OrderBy(input => input.blob, StringComparer.Ordinal)
                .ToArray();
            for (var i = 0; i < declaredInputs.Length; i++)
                inputByBlob[declaredInputs[i].blob] = declaredInputs[i];

            var knownBlobs = new Dictionary<string, AexisPreflightTensorDescriptor>(inputByBlob, StringComparer.Ordinal);
            var missingNodes = new List<AexisModelPreflightIssue>();
            var missingParameters = new List<AexisModelPreflightIssue>();
            var missingDependencies = new List<AexisModelPreflightIssue>();
            var nodes = new List<AexisModelPreflightNode>();
            var layers = model.layers ?? new List<AexisGraphModel.Layer>();

            for (var index = 0; index < layers.Count; index++)
            {
                var layer = layers[index];
                if (layer == null)
                {
                    missingNodes.Add(CreateIssue(index, string.Empty, string.Empty, string.Empty, "null-layer", "The model contains a null layer.", "Re-export the model graph."));
                    continue;
                }

                var operatorName = !string.IsNullOrWhiteSpace(layer.typeName) ? layer.typeName : layer.type.ToString();
                if (!AexisOperatorCapabilities.TryGet(operatorName, out var capability))
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
                var d3Diagnostic = ValidateD3ShapeIndexNode(layer, operatorName, nodeInputs);
                if (d3Diagnostic != null)
                    nodeIssues.Add(d3Diagnostic.code + ": " + d3Diagnostic.message);
                var missingForNode = FindMissingParameters(layer, capability.requiredParameters);
                for (var parameterIndex = 0; parameterIndex < missingForNode.Length; parameterIndex++)
                {
                    var issue = CreateIssue(index, layer.name, operatorName, string.Empty, "missing-parameter", "Missing required parameter " + missingForNode[parameterIndex] + ".", "Provide the required ncnn parameter or re-export the layer.");
                    missingParameters.Add(issue);
                    nodeIssues.Add(issue.message);
                }

                var strictEligible = !request.strict || (AexisOperatorCapabilities.IsStrictlySupported(capability, request.targetBackend, request.targetDtype) && missingForNode.Length == 0 && nodeIssues.Count == 0);
                var recommendation = d3Diagnostic != null
                    ? d3Diagnostic.recommendedAction
                    : ResolveRecommendedAction(capability, request.targetBackend, request.targetDtype, missingForNode.Length > 0);
                if (request.strict && !strictEligible && nodeIssues.Count == 0)
                    nodeIssues.Add("Strict target rejects status=" + capability.status + " for backend=" + request.targetBackend + " dtype=" + request.targetDtype + ".");

                nodes.Add(new AexisModelPreflightNode
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

            var report = new AexisModelPreflightReport
            {
                schemaVersion = AexisOperatorCapabilities.SchemaVersion,
                contract = AexisOperatorCapabilities.PreflightContract,
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

        public static string ToStableJson(AexisModelPreflightReport report, bool prettyPrint = true)
        {
            if (report == null)
                throw new ArgumentNullException(nameof(report));
            return JsonUtility.ToJson(report, prettyPrint) + "\n";
        }

        public static void WriteStableJson(string path, AexisModelPreflightReport report)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("An output path is required.", nameof(path));
            var directory = Path.GetDirectoryName(Path.GetFullPath(path));
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);
            File.WriteAllText(path, ToStableJson(report), new System.Text.UTF8Encoding(false));
        }

        private static List<AexisPreflightTensorDescriptor> ResolveInputs(
            AexisGraphModel.Layer layer,
            Dictionary<string, AexisPreflightTensorDescriptor> knownBlobs,
            List<string> nodeIssues,
            int layerIndex,
            string operatorName,
            List<AexisModelPreflightIssue> missingDependencies)
        {
            var result = new List<AexisPreflightTensorDescriptor>();
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
                result.Add(new AexisPreflightTensorDescriptor { blob = bottom, logicalShape = Array.Empty<int>(), storageShape = Array.Empty<int>(), layout = "Unknown", dtype = "Unknown" });
            }
            return result;
        }

        private static OnnxExecutionPlanDiagnostic ValidateD3ShapeIndexNode(
            AexisGraphModel.Layer layer,
            string operatorName,
            List<AexisPreflightTensorDescriptor> inputs)
        {
            switch (operatorName)
            {
                case "TopK":
                case "OneHot":
                case "NonZero":
                case "Compress":
                case "GatherND":
                case "Scatter":
                case "ScatterElements":
                case "ScatterND":
                    break;
                default:
                    return null;
            }

            var inputRank = 1;
            if (inputs != null && inputs.Count > 0 && inputs[0]?.logicalShape != null && inputs[0].logicalShape.Length > 0)
                inputRank = inputs[0].logicalShape.Length;
            var hasStaticParameter = operatorName == "TopK"
                ? HasLayerParameter(layer, "k", 1)
                : operatorName == "OneHot"
                    ? HasLayerParameter(layer, "depth", 1)
                    : true;
            var batchDims = ReadLayerInt(layer, "batch_dims", 0, 0);
            var uniqueIndices = ReadLayerInt(layer, "unique_indices", -1, 0) != 0;
            return OnnxExecutionShapePlanner.Validate(new OnnxExecutionNodeContract
            {
                name = layer.name ?? string.Empty,
                opType = operatorName,
                inputRank = inputRank,
                batchDims = batchDims,
                indexDataType = TensorDataType.Int32,
                dynamicParameter = !hasStaticParameter,
                uniqueIndices = uniqueIndices
            });
        }

        private static bool HasLayerParameter(AexisGraphModel.Layer layer, string name, int key)
        {
            return (layer?.stringParams != null && layer.stringParams.ContainsKey(name))
                || (layer?.intParams != null && layer.intParams.ContainsKey(key));
        }

        private static int ReadLayerInt(AexisGraphModel.Layer layer, string name, int key, int defaultValue)
        {
            if (layer?.stringParams != null
                && layer.stringParams.TryGetValue(name, out var named)
                && int.TryParse(named, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsedNamed))
            {
                return parsedNamed;
            }
            if (layer?.intParams != null
                && layer.intParams.TryGetValue(key, out var keyed)
                && int.TryParse(keyed, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsedKeyed))
            {
                return parsedKeyed;
            }
            return defaultValue;
        }

        private static void ReportMissingInputDescriptors(
            AexisGraphModel.Layer layer,
            AexisOperatorCapability capability,
            Dictionary<string, AexisPreflightTensorDescriptor> inputByBlob,
            List<string> nodeIssues,
            int layerIndex,
            string operatorName,
            List<AexisModelPreflightIssue> missingDependencies)
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
            AexisGraphModel.Layer layer,
            AexisOperatorCapability capability,
            List<AexisPreflightTensorDescriptor> inputs,
            Dictionary<string, AexisPreflightTensorDescriptor> knownBlobs)
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
                        : new AexisPreflightTensorDescriptor
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
            AexisGraphModel.Layer layer,
            Dictionary<string, AexisPreflightTensorDescriptor> knownBlobs)
        {
            if (layer.topNames == null)
                return;

            for (var i = 0; i < layer.topNames.Length; i++)
            {
                var top = layer.topNames[i];
                if (string.IsNullOrWhiteSpace(top))
                    continue;
                knownBlobs[top] = new AexisPreflightTensorDescriptor
                {
                    blob = top,
                    logicalShape = Array.Empty<int>(),
                    storageShape = Array.Empty<int>(),
                    layout = "Unknown",
                    dtype = "Unknown"
                };
            }
        }

        private static string[] FindMissingParameters(AexisGraphModel.Layer layer, string[] requiredParameters)
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

        private static AexisModelPreflightNode CreateUnknownNode(int index, AexisGraphModel.Layer layer, string operatorName, string message, string action)
        {
            return new AexisModelPreflightNode
            {
                layerIndex = index,
                layer = layer.name ?? string.Empty,
                operatorName = operatorName,
                canonicalOperator = operatorName,
                status = AexisOperatorCapabilityStatus.Unsupported,
                strictEligible = false,
                bottomBlobs = CloneStrings(layer.bottomNames),
                topBlobs = CloneStrings(layer.topNames),
                inputs = Array.Empty<AexisPreflightTensorDescriptor>(),
                missingParameters = Array.Empty<string>(),
                issues = new[] { message },
                recommendedAction = action
            };
        }

        private static AexisModelPreflightIssue CreateIssue(int index, string layer, string operatorName, string blob, string code, string message, string action)
        {
            return new AexisModelPreflightIssue
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

        private static string ResolveRecommendedAction(AexisOperatorCapability capability, string backend, string dtype, bool missingParameters)
        {
            if (missingParameters)
                return "Provide the missing parameters before execution planning.";
            if (capability.status == AexisOperatorCapabilityStatus.Unsupported)
                return "Do not execute or materialize to Buffer; implement the Pack4 RenderTexture/CommandBuffer operator first.";
            if (capability.status == AexisOperatorCapabilityStatus.Partial)
                return "Keep this node out of strict plans until its exact Pack4 RenderTexture/CommandBuffer shape and dtype contract is implemented and validated.";
            if (capability.status == AexisOperatorCapabilityStatus.AliasOnly)
                return "Prove compatible logical/storage shape and layout for aliasing, or implement a real Pack4 transform. Do not use a Buffer fallback.";
            if (capability.status == AexisOperatorCapabilityStatus.DebugOnly)
                return "This is debug-only. Add a Pack4 RenderTexture/CommandBuffer path; do not select its legacy Buffer path for production.";
            return "Target backend/dtype requires further validation: " + backend + "/" + dtype + ".";
        }

        private static AexisPreflightTensorDescriptor CloneDescriptor(AexisPreflightTensorDescriptor source)
        {
            if (source == null)
                return new AexisPreflightTensorDescriptor { blob = string.Empty, logicalShape = Array.Empty<int>(), storageShape = Array.Empty<int>(), layout = "Unknown", dtype = "Unknown" };
            return new AexisPreflightTensorDescriptor
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
