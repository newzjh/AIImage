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
        public const string SupportedByProfile = "supported-by-profile";
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
        public string profileId;
        public string backend;
        // Logical layouts describe tensor interpretation; storageLayouts describe the
        // physical runtime target selected by the planner (currently Packed4).
        public string[] layouts;
        public string[] storageLayouts;
        public string[] dtypes;
        public int[] inputRanks;
        public int[] outputRanks;
        public int minInputs;
        // -1 means variadic.
        public int maxInputs;
        public int minOutputs;
        public int maxOutputs;
        public bool requiresTextureBackedInputs;
        public bool requiresImmutableWeights;
        public bool requiresLoadedRuntimeVerification;
        public string predicateEvaluator;
        public string weightContract;
        public string[] parameterPredicates;
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
        public bool bf16;
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
        // Physical texture dtype and logical tensor dtype differ for Int32
        // shape/index tensors stored exactly in FP32 textures.
        public string dtype;
        public string logicalDtype;
    }

    [Serializable]
    public sealed class AexisModelPreflightRequest
    {
        public string modelName;
        public string targetBackend = AexisOperatorCapabilityBackend.CommandBuffer;
        public string targetDtype = "FP32";
        public string targetLayout = AexisTexturePlanLayout.Packed4;
        public bool strict = true;
        public AexisPreflightTensorDescriptor[] inputs = Array.Empty<AexisPreflightTensorDescriptor>();
        // Optional exact physical descriptors. Supplying these lets the same report carry
        // the strict Pack4 planner result without allocating a texture or buffer.
        public AexisTexturePlanTensorDescriptor[] textureInputs = Array.Empty<AexisTexturePlanTensorDescriptor>();
        [NonSerialized] public AexisTextureExecutionPlanNodeVerifier nodeVerifier;
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
        public string matchedProfileId;
        public bool strictEligible;
        public string[] bottomBlobs;
        public string[] topBlobs;
        public AexisPreflightTensorDescriptor[] inputs;
        public AexisPreflightTensorDescriptor[] outputs;
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
        public AexisTextureExecutionPlan texturePlan;
        public bool strictEligible;
        public string summary;
    }

    // This is intentionally metadata-only. It never creates a layer, AexisGraphSession, texture, or buffer.
    public static class AexisOperatorCapabilities
    {
        public const int SchemaVersion = 3;
        public const string Contract = "aiimage.operator-capabilities/v3";
        public const string PreflightContract = "aiimage.model-preflight/v2";

        // Kept for raw registry names whose runtime intentionally has no
        // texture-native profile. P2 recurrent operators are no longer here:
        // they have a bounded Pack4 CommandBuffer implementation below.
        private static readonly HashSet<string> UnsupportedOperators = new HashSet<string>(StringComparer.Ordinal)
        {
        };

        private static readonly HashSet<string> P1VisualOperators = new HashSet<string>(StringComparer.Ordinal)
        {
            "DeformableConv2D", "DetectionOutput", "Diag", "Einsum", "Flip", "Fold", "GLU", "GridSample",
            "Proposal", "PSROIPooling", "ROIAlign", "ROIPooling", "SPP", "YoloDetectionOutput", "Yolov3DetectionOutput",
            "YoloDetectOut", "Yolov3DetectOut"
        };

        private static readonly HashSet<string> AliasOnlyOperators = new HashSet<string>(StringComparer.Ordinal)
        {
            "Input", "Noop", "Split", "DeepCopy", "Dropout"
        };

        private static readonly HashSet<string> SentisOperators = new HashSet<string>(StringComparer.Ordinal)
        {
            "Shape", "Size", "Range", "ConstantOfShape", "Expand", "ArgMax", "ArgMin", "Where",
            "TopK", "Nms", "NonMaxSuppression", "NonZero", "OneHot", "CumSum", "Compress", "Gather", "GatherElements", "GatherND",
            "Scatter", "ScatterElements", "ScatterND", "Trilu"
        };

        private static readonly HashSet<string> Pack4LayoutOperators = new HashSet<string>(StringComparer.Ordinal)
        {
            "Reshape", "Flatten", "Squeeze", "ExpandDims", "Permute", "Slice", "Tile", "Packing", "Cast"
        };

        private static readonly HashSet<string> ShapePreservingOperators = new HashSet<string>(StringComparer.Ordinal)
        {
            "AbsVal", "TanH", "Exp", "Log", "BNLL", "Power", "Threshold", "ThresholdedRelu", "Softsign", "IsInf", "IsNaN",
            "UnaryOp", "BinaryOp", "ReLU", "Sigmoid", "GELU", "ELU", "Erf", "HardSigmoid", "HardSwish",
            "Mish", "Swish", "SELU", "Shrink", "Softplus", "CELU", "Clip", "Bias", "BatchNorm", "InstanceNorm", "MVN",
            "LayerNorm", "PReLU", "LRN", "Softmax", "CopyTo", "Cast", "Trilu"
        };

        private static readonly HashSet<string> TextureAndCommandBufferOperators = new HashSet<string>(StringComparer.Ordinal)
        {
            "AbsVal", "aten::to", "BatchNorm", "Bias", "BinaryOp", "Cast", "Clip", "Concat", "Convolution", "InstanceNorm", "MVN", "Normalize", "PriorBox",
            "Convolution1D", "Convolution3D", "ConvolutionDepthWise", "ConvolutionDepthWise3D", "Crop", "Deconvolution",
            "Deconvolution3D", "DeconvolutionDepthWise", "DeconvolutionDepthWise3D", "Eltwise", "ExpandDims", "Flatten", "GELU", "Squeeze",
            "Embed", "Gemm", "GroupNorm", "InnerProduct", "Interp", "LayerNorm", "MatMul", "Packing", "Padding",
            "MaxPoolingInd", "MaxUnPooling", "Permute", "PixelShuffle", "Pooling", "Pooling1D", "Pooling3D", "PReLU", "LRN", "Quantize", "Dequantize",
            "Requantize", "Reduction", "ReLU", "Reorg", "Reshape", "Scale", "ShuffleChannel", "Sigmoid", "Slice", "Softmax",
            "Swish", "Tile", "UnaryOp", "Unfold", "MemoryData", "Shape", "Size", "Range", "ConstantOfShape", "Expand",
            "ArgMax", "ArgMin", "Where", "TopK", "OneHot", "CumSum", "Gather", "GatherElements", "pnnx.Expression",
            "AbsVal", "TanH", "Exp", "Log", "BNLL", "Power", "Threshold", "ThresholdedRelu", "ELU", "Erf", "HardSigmoid",
            "HardSwish", "Mish", "SELU", "Shrink", "Softplus", "Softsign", "IsInf", "IsNaN", "CELU"
            , "CopyTo", "Nms", "NonMaxSuppression", "NonZero", "Compress", "GatherND", "Scatter", "ScatterElements", "ScatterND"
            , "ConvolutionDepthWise1D", "Deconvolution1D", "DeconvolutionDepthWise1D", "ExtractPatches", "Softsign", "IsInf", "IsNaN", "Trilu"
            , "RotaryEmbed", "MultiHeadAttention", "DeepFillV2ContextualAttention"
            , "GridSample", "DeformableConv2D", "Fold", "Flip", "GLU", "Einsum", "Diag", "SPP"
            , "ROIAlign", "ROIPooling", "PSROIPooling", "Proposal", "DetectionOutput", "YoloDetectionOutput", "Yolov3DetectionOutput", "YoloDetectOut", "Yolov3DetectOut"
            , "RandomLike", "RandomUniformLike", "RandomNormalLike", "RandomUniform", "RandomNormal", "Bernoulli", "Multinomial", "StatisticsPooling", "StatsPooling", "Spectrogram", "InverseSpectrogram", "InvSpectrogram"
            , "RNN", "GRU", "LSTM"
        };

        // These operators have a loaded-runtime verifier which proves the exact node
        // profile, including immutable constants, parameters, logical/storage shape,
        // and the concrete CommandBuffer Pack4 execution path.
        private static readonly HashSet<string> RuntimeVerifiedProfileOperators = new HashSet<string>(StringComparer.Ordinal)
        {
            "Convolution", "Convolution1D", "ConvolutionDepthWise", "ConvolutionDepthWise1D", "ConvolutionDepthWise3D", "Convolution3D",
            "Deconvolution", "Deconvolution1D", "DeconvolutionDepthWise1D", "DeconvolutionDepthWise", "DeconvolutionDepthWise3D", "Deconvolution3D",
            "Eltwise", "Concat", "BinaryOp", "Interp", "PixelShuffle", "UnaryOp", "AbsVal", "TanH",
            "Exp", "Log", "BNLL", "Power", "Threshold", "ThresholdedRelu", "ELU", "Erf", "HardSigmoid", "HardSwish", "Mish",
            "SELU", "Shrink", "Softplus", "Softsign", "IsInf", "IsNaN", "CELU", "Swish", "Clip", "GELU",
            "pnnx.Expression", "MemoryData", "Embed", "InnerProduct", "Pooling", "Pooling1D", "MaxPoolingInd",
            "MaxUnPooling", "Pooling3D", "Reduction", "BatchNorm", "PReLU", "LRN", "InstanceNorm", "MVN", "Bias", "CopyTo", "Normalize", "PriorBox", "Reshape",
            "Flatten", "Squeeze", "ExpandDims", "Permute", "ShuffleChannel", "Gemm", "LayerNorm", "RMSNorm", "Slice", "Tile",
            "Packing", "Cast", "MatMul", "Softmax", "SDPA", "MultiHeadAttention", "NonZero", "Compress",
            "GatherND", "Scatter", "ScatterElements", "ScatterND", "CumSum", "CumulativeSum", "ReLU", "Sigmoid", "Trilu",
            "RotaryEmbed", "DeepFillV2ContextualAttention", "ShortConv", "GatedDeltaRule",
            "Shape", "Size", "Range", "ConstantOfShape", "Expand", "Where", "Gather", "GatherElements", "Nms", "NonMaxSuppression",
            "ArgMax", "ArgMin", "TopK", "OneHot", "aten::to", "Crop", "GroupNorm", "Padding",
            "Quantize", "Dequantize", "Requantize", "Reorg", "Scale", "Unfold", "ExtractPatches"
            , "GridSample", "DeformableConv2D", "Fold", "Flip", "GLU", "Einsum", "Diag", "SPP"
            , "ROIAlign", "ROIPooling", "PSROIPooling", "Proposal", "DetectionOutput", "YoloDetectionOutput", "Yolov3DetectionOutput", "YoloDetectOut", "Yolov3DetectOut"
            , "RandomLike", "RandomUniformLike", "RandomNormalLike", "RandomUniform", "RandomNormal", "Bernoulli", "Multinomial", "StatisticsPooling", "StatsPooling", "Spectrogram", "InverseSpectrogram", "InvSpectrogram"
            , "RNN", "GRU", "LSTM"
        };

        private static readonly Dictionary<string, string[]> RequiredParameters = new Dictionary<string, string[]>(StringComparer.Ordinal)
        {
            { "Convolution", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size" } },
            { "ConvolutionDepthWise", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size" } },
            { "Convolution1D", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size" } },
            { "ConvolutionDepthWise1D", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size" } },
            { "Convolution3D", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size" } },
            { "ConvolutionDepthWise3D", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size", "7:group" } },
            { "Deconvolution", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size" } },
            { "DeconvolutionDepthWise", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size" } },
            { "DeconvolutionDepthWise3D", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size", "7:group" } },
            { "Deconvolution3D", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size" } },
            { "Deconvolution1D", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size" } },
            { "DeconvolutionDepthWise1D", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size", "7:group" } },
            { "Pooling3D", new[] { "0:pooling_type", "1:kernel_w" } },
            { "Bias", new[] { "0:bias_data_size" } },
            { "PReLU", new[] { "0:num_slope" } },
            { "LRN", new[] { "0:region_type", "1:local_size" } },
            { "InstanceNorm", new[] { "0:channels", "1:eps", "2:affine" } },
            { "GroupNorm", new[] { "0:group", "1:channels", "2:eps" } },
            { "Scale", new[] { "0:scale_data_size", "1:bias_term" } },
            { "Quantize", new[] { "0:scale_data_size" } },
            { "Dequantize", new[] { "0:scale_data_size", "1:bias_data_size" } },
            { "Requantize", new[] { "0:scale_in_data_size", "1:scale_out_data_size", "2:bias_data_size" } },
            { "Multinomial", new[] { "0:sample_size", "1:seed" } },
            { "Reorg", new[] { "0:stride", "1:mode" } },
            { "Unfold", new[] { "1:kernel_w", "11:kernel_h" } },
            { "ExtractPatches", new[] { "1:kernel_w", "11:kernel_h" } },
            { "Interp", new[] { "0:resize_type" } },
            { "InnerProduct", new[] { "0:num_output", "1:bias_term", "2:weight_data_size" } },
            { "GridSample", new[] { "0:sample_type", "1:padding_mode", "2:align_corner" } },
            { "DeformableConv2D", new[] { "0:num_output", "1:kernel_w", "6:weight_data_size" } },
            { "Fold", new[] { "1:kernel_w", "20:output_w" } },
            { "SPP", new[] { "0:pooling_type", "1:pooling_kernel" } },
            { "ROIAlign", new[] { "0:pooled_width", "1:pooled_height", "2:spatial_scale" } },
            { "ROIPooling", new[] { "0:pooled_width", "1:pooled_height", "2:spatial_scale" } },
            { "PSROIPooling", new[] { "0:pooled_width", "1:pooled_height", "2:spatial_scale" } },
            { "Proposal", new[] { "0:feat_stride", "1:base_size", "2:pre_nms_topN", "3:after_nms_topN" } },
            { "DetectionOutput", new[] { "0:num_class", "3:keep_top_k" } },
            { "YoloDetectionOutput", new[] { "0:num_class", "1:num_box" } },
            { "Yolov3DetectionOutput", new[] { "0:num_class", "1:num_box" } },
            { "RandomLike", new[] { "0:seed" } },
            { "RNN", new[] { "0:input_size", "1:hidden_size", "2:direction", "3:initial_state", "4:state_outputs" } },
            { "GRU", new[] { "0:input_size", "1:hidden_size", "2:direction", "3:initial_state", "4:state_outputs" } },
            { "LSTM", new[] { "0:input_size", "1:hidden_size", "2:direction", "3:initial_state", "4:state_outputs" } },
            { "Nms", new[] { "0:capacity", "1:max_output_boxes_per_class", "2:center_point_box", "3:iou_threshold", "4:score_threshold" } },
            { "NonMaxSuppression", new[] { "0:capacity", "1:max_output_boxes_per_class", "2:center_point_box", "3:iou_threshold", "4:score_threshold" } },
            { "YoloDetectOut", new[] { "0:num_class", "1:num_box" } },
            { "Yolov3DetectOut", new[] { "0:num_class", "1:num_box" } },
        };

        public static AexisOperatorCapabilityDocument CreateDocument()
        {
            var operatorNames = new HashSet<string>(StringComparer.Ordinal);
            var registered = AexisLayerFactory.GetRegisteredLayerTypes();
            for (var i = 0; i < registered.Count; i++)
                operatorNames.Add(registered[i].ToString());
            foreach (var alias in new[] { "CumulativeSum", "ConvolutionDepthWise1D", "ConvolutionDepthWise3D", "Deconvolution1D", "DeconvolutionDepthWise1D", "DeconvolutionDepthWise3D", "StatisticsPooling", "InverseSpectrogram" })
                operatorNames.Add(alias);
            // AexisLayerTypeKey is deliberately limited to 16 bytes, so keys such
            // as MultiHeadAttention cannot round-trip through ToString(). Keep the
            // public capability document in sync with the verified runtime names.
            foreach (var runtimeVerified in RuntimeVerifiedProfileOperators)
                operatorNames.Add(runtimeVerified);
            foreach (var unsupported in UnsupportedOperators)
                operatorNames.Add(unsupported);
            foreach (var p1 in P1VisualOperators)
                operatorNames.Add(p1);

            return new AexisOperatorCapabilityDocument
            {
                schemaVersion = SchemaVersion,
                contract = Contract,
                operators = operatorNames
                    .OrderBy(operatorName => operatorName, StringComparer.Ordinal)
                    .Select(CreateCapability)
                    .ToArray()
            };
        }

        public static bool TryGet(string operatorName, out AexisOperatorCapability capability)
        {
            capability = null;
            if (string.IsNullOrWhiteSpace(operatorName))
                return false;

            var canonical = AexisLayerFactory.ResolveCanonicalLayerTypeName(operatorName);
            if (!AexisLayerFactory.IsRegistered(canonical) && !UnsupportedOperators.Contains(operatorName))
                return false;

            // Preserve the requested spelling. Long NCNN names have distinct parameter
            // and verifier contracts even when the fixed-size factory key is canonicalized.
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
                : string.Equals(targetDtype, "BF16", StringComparison.OrdinalIgnoreCase) ? capability.bf16
                : string.Equals(targetDtype, "INT8", StringComparison.OrdinalIgnoreCase) && capability.int8;
            if (!dtypeSupported)
                return false;

            return string.IsNullOrWhiteSpace(targetLayout)
                || (capability.layouts ?? Array.Empty<string>()).Any(layout => string.Equals(layout, targetLayout, StringComparison.OrdinalIgnoreCase));
        }

        private static AexisOperatorCapability CreateCapability(string operatorName)
        {
            var isUnsupported = UnsupportedOperators.Contains(operatorName);
            var isP1Visual = P1VisualOperators.Contains(operatorName);
            var isAliasOnly = AliasOnlyOperators.Contains(operatorName);
            var hasTexturePath = TextureAndCommandBufferOperators.Contains(operatorName)
                || TextureAndCommandBufferOperators.Contains(AexisLayerFactory.ResolveCanonicalLayerTypeName(operatorName));
            var isSentis = SentisOperators.Contains(operatorName)
                || SentisOperators.Contains(AexisLayerFactory.ResolveCanonicalLayerTypeName(operatorName));
            var hasInt8QuantizedKernel = operatorName == "Convolution"
                || operatorName == "ConvolutionDepthWise"
                || operatorName == "Gemm"
                || operatorName == "InnerProduct";
            // Every production node is admitted against its concrete loaded descriptor.
            // A global operator flag cannot prove storage shape, dtype, or loaded weights.
            var hasVerifiedCommandBufferPack4 = false;
            var hasRuntimeVerifiedProfile = RuntimeVerifiedProfileOperators.Contains(operatorName);
            var status = isUnsupported
                ? AexisOperatorCapabilityStatus.Unsupported
                : isAliasOnly
                    ? AexisOperatorCapabilityStatus.AliasOnly
                    : isP1Visual && !hasRuntimeVerifiedProfile
                        ? AexisOperatorCapabilityStatus.Partial
                     : hasVerifiedCommandBufferPack4
                         ? AexisOperatorCapabilityStatus.Supported
                    : hasRuntimeVerifiedProfile
                        ? AexisOperatorCapabilityStatus.SupportedByProfile
                    : hasTexturePath
                        ? AexisOperatorCapabilityStatus.Partial
                        : AexisOperatorCapabilityStatus.DebugOnly;

            var profiles = FinalizeProfiles(operatorName, ResolveProfiles(operatorName), hasRuntimeVerifiedProfile);
            return new AexisOperatorCapability
            {
                operatorName = operatorName,
                canonicalOperator = ResolveCanonicalOperator(operatorName),
                importFormats = isSentis ? new[] { "ncnn", "Sentis/ONNX" } : new[] { "ncnn" },
                importSupported = !isUnsupported,
                // The current runtime resolves most shapes while executing. There is no complete static shape engine yet.
                shapeInference = isAliasOnly || ShapePreservingOperators.Contains(operatorName) || Pack4LayoutOperators.Contains(operatorName),
                renderTexture = hasTexturePath && !isUnsupported,
                commandBuffer = hasTexturePath && !isUnsupported,
                fp32 = hasTexturePath && !isUnsupported,
                fp16 = hasTexturePath && !isUnsupported,
                bf16 = hasTexturePath && !isUnsupported,
                int8 = hasInt8QuantizedKernel && hasTexturePath && !isUnsupported,
                layouts = ResolveLayouts(operatorName),
                ranks = ResolveRanks(operatorName),
                verifiedModels = operatorName == "ROIAlign"
                    ? new[] { "command-buffer-pack4-roialign-multi-roi-golden" }
                    : hasVerifiedCommandBufferPack4
                    ? new[] { "pack4-command-buffer-smoke" }
                    : Array.Empty<string>(),
                status = status,
                limitations = ResolveLimitations(operatorName, status),
                requiredParameters = RequiredParameters.TryGetValue(operatorName, out var parameters) ? parameters : Array.Empty<string>(),
                profiles = profiles
            };
        }

        private static string ResolveCanonicalOperator(string operatorName)
        {
            switch (operatorName)
            {
                case "Convolution": return "Conv2D";
                case "ConvolutionDepthWise": return "DepthwiseConv2D";
                case "Convolution1D": return "Conv1D";
                case "ConvolutionDepthWise1D": return "DepthwiseConv1D";
                case "Convolution3D": return "Conv3D";
                case "Deconvolution": return "ConvTranspose2D";
                case "Deconvolution1D": return "ConvTranspose1D";
                case "DeconvolutionDepthWise1D": return "DepthwiseConvTranspose1D";
                case "DeconvolutionDepthWise": return "DepthwiseConvTranspose2D";
                case "DeconvolutionDepthWise3D": return "DepthwiseConvTranspose3D";
                case "Deconvolution3D": return "ConvTranspose3D";
                case "InnerProduct": return "FullyConnected";
                case "BinaryOp": return "BinaryElementwise";
                case "UnaryOp": return "UnaryElementwise";
                case "Interp": return "Resize";
                case "pnnx.Expression": return "Expression";
                case "aten::to": return "Cast";
                case "CumulativeSum": return "CumSum";
                case "Nms": return "NonMaxSuppression";
                case "StatisticsPooling":
                case "StatsPooling": return "TemporalStatisticsPooling";
                case "InverseSpectrogram":
                case "InvSpectrogram": return "InverseSpectrogram";
                default: return operatorName;
            }
        }

        private static string[] ResolveLayouts(string operatorName)
        {
            if (operatorName == "Multinomial")
                return new[] { "Packed4" };
            if (operatorName == "Convolution3D" || operatorName == "Deconvolution3D" || operatorName == "Pooling3D")
                return new[] { "CDHW", "Packed4" };
            if (operatorName == "Interp")
                return new[] { "NCHW", "CDHW", "Packed4" };
            if (operatorName == "Gemm" || operatorName == "MatMul" || operatorName == "InnerProduct")
                return new[] { "Linear", "Packed4" };
            if (operatorName == "Embed")
                return new[] { "Index", "Linear", "Packed4" };
            if (SentisOperators.Contains(operatorName))
                return new[] { "Scalar", "Linear", "Packed4" };
            return new[] { "NCHW", "Packed4" };
        }

        private static int[] ResolveRanks(string operatorName)
        {
            if (operatorName == "Convolution3D" || operatorName == "Deconvolution3D" || operatorName == "Pooling3D")
                return new[] { 4 };
            if (operatorName == "Convolution1D")
                return new[] { 2 };
            if (operatorName == "StatisticsPooling" || operatorName == "StatsPooling")
                return new[] { 2 };
            if (operatorName == "Spectrogram" || operatorName == "InverseSpectrogram" || operatorName == "InvSpectrogram")
                return new[] { 2 };
            if (operatorName == "RNN" || operatorName == "GRU" || operatorName == "LSTM")
                return new[] { 3 };
            if (operatorName == "Multinomial")
                return new[] { 2, 3 };
            if (operatorName == "ConvolutionDepthWise1D" || operatorName == "Deconvolution1D" || operatorName == "DeconvolutionDepthWise1D")
                return new[] { 2 };
            if (operatorName == "ConvolutionDepthWise3D" || operatorName == "DeconvolutionDepthWise3D")
                return new[] { 4 };
            if (operatorName == "Pooling1D")
                return new[] { 3 };
            if (operatorName == "ShuffleChannel")
                return new[] { 3 };
            if (operatorName == "Bias")
                return new[] { 3, 4 };
            if (operatorName == "ExtractPatches")
                return new[] { 3, 4 };
            if (operatorName == "CopyTo" || operatorName == "MVN")
                return new[] { 3, 4 };
            if (operatorName == "Convolution" || operatorName == "ConvolutionDepthWise"
                || operatorName == "Deconvolution" || operatorName == "DeconvolutionDepthWise"
                || operatorName == "Pooling" || operatorName == "Bias")
                return new[] { 3 };
            if (operatorName == "Gemm" || operatorName == "MatMul" || operatorName == "InnerProduct")
                return new[] { 1, 2, 3 };
            if (operatorName == "Embed")
                return new[] { 1, 2 };
            if (operatorName == "NonZero" || operatorName == "Compress"
                || operatorName == "Scatter" || operatorName == "ScatterElements")
                return new[] { 1 };
            if (operatorName == "Nms" || operatorName == "NonMaxSuppression")
                return new[] { 2 };
            if (operatorName == "GatherND" || operatorName == "ScatterND")
                return new[] { 1, 2 };
            if (operatorName == "Trilu")
                return new[] { 2, 3, 4 };
            if (SentisOperators.Contains(operatorName))
                return new[] { 1, 2, 3, 4 };
            return new[] { 1, 2, 3, 4 };
        }

        private static string ResolveLimitations(string operatorName, string status)
        {
            if (status == AexisOperatorCapabilityStatus.Unsupported)
                return operatorName + " is registered only to report a deterministic error; its texture-native implementation is absent. "
                    + "D3 rejects this graph at plan time rather than reading GPU data back or falling back to a ComputeBuffer.";
            if (P1VisualOperators.Contains(operatorName) && !RuntimeVerifiedProfileOperators.Contains(operatorName))
                return "P1 visual operator ABI is registered but its exact Pack4 profile has not been installed. The runtime rejects the node instead of falling back to a ComputeBuffer.";
            if (P1VisualOperators.Contains(operatorName))
                return "Built-in P1 Pack4 RenderTexture and CommandBuffer implementation. Inputs may be exact Pack4 or LinearMat materialized through a texture transform; unsupported dynamic ranks, unbounded detection capacity, and unlisted equation forms are rejected before dispatch.";
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
            if (operatorName == "Bias")
                return "CommandBuffer Pack4 support: ncnn bias_data is loaded once as immutable float4 channel constants and dispatched across exact rank-3 or rank-4 Pack4 storage. Strict planning requires a loaded constant pack whose channel count matches the activation.";
            if (operatorName == "CopyTo")
                return "Texture-native ncnn ROI write: the first Pack4 input is copied and the second is written at static W/H/D/C offsets. "
                    + "Strict planning requires equal rank-3 or rank-4 FP16/FP32 inputs, matching layout/dtype, and an in-bounds ROI. "
                    + "Numpy-style static starts/axes and non-pack-aligned channel offsets are supported without buffer materialization.";
            if (operatorName == "Pooling1D")
                return "Texture-native ncnn Pooling1D for dims=3,height=1 Pack4 tensors. Max/average, global, adaptive out_w, "
                    + "pad_mode full/valid/SAME_UPPER/SAME_LOWER, asymmetric explicit padding, and average include/exclude-pad semantics are supported.";
            if (operatorName == "StatisticsPooling" || operatorName == "StatsPooling")
                return "CommandBuffer LinearMat statistics pooling for a static rank-2 [frames,channels] FP32 texture. "
                    + "It writes per-channel mean and optional population standard deviation to a bounded FP32 LinearMat output entirely on GPU. "
                    + "Only include_std=0|1 and finite non-negative epsilon are accepted; dynamic sequence lengths and buffer materialization are rejected.";
            if (operatorName == "Spectrogram" || operatorName == "InverseSpectrogram" || operatorName == "InvSpectrogram")
                return "CommandBuffer LinearMat real DFT/iDFT profile with static rank-2 FP32 audio textures. "
                    + "n_fft is even and bounded to 256, hop is positive, channels are explicit, and full-frame coverage is required. "
                    + "Spectrogram emits one-sided complex bins; inverse overlap-averages the matching texture profile entirely on GPU. "
                    + "Window functions, center padding, dynamic lengths, and buffer materialization are rejected.";
            if (operatorName == "RNN" || operatorName == "GRU" || operatorName == "LSTM")
                return "Bounded FP32 CommandBuffer Pack4 recurrent profile with immutable gate weights and zero initial state. "
                    + "It accepts only a forward rank-3 [sequence<=256,1,input_channels] Texture2DArray and writes [sequence,1,hidden<=256] "
                    + "without activation buffers, CPU state, readback, bidirectional execution, sequence lengths, or optional state outputs.";
            if (operatorName == "Multinomial")
                return "Bounded deterministic categorical sampling over a static FP32 Pack4 logits texture. ONNX [batch,classes] is represented as logical/storage [dims=3,w=1,h=batch,c=classes]; results use exact FP32 Pack4 lanes with logical Int32 indices. Fixed seed, batch<=256, classes<=4096, sample_size<=256, and sampling with replacement are mandatory. FP16/BF16, dynamic sizes, activation buffers, CPU sampling, and readback-driven state are rejected.";
            if (operatorName == "ExtractPatches")
                return "Texture-native ExtractPatches for exact rank-3 Pack4 or rank-4 Fold-D storage. ONNX ExtractImagePatches lowers static FP32 NHWC batch=1 with SAME/VALID padding through real Pack4 permutations; dynamic shapes and unsupported layouts fail preflight.";
            if (operatorName == "LayerNorm" || operatorName == "Softmax" || operatorName == "Reduction"
                || operatorName == "MultiHeadAttention" || operatorName == "SDPA")
                return "Partial CommandBuffer Pack4 support: LayerNorm/Softmax use FP32 accumulation; Reduction covers scalar rank-2 and Pack4 spatial SUM/MEAN; SDPA supports texture-native masks and rank-3 KV-cache append/concat with logical sequence length separated from reserved Texture2DArray height. MultiHeadAttention KV-cache, unlisted axes/ranks, and unsupported dtype/layout profiles fail strict planning.";
            if (operatorName == "Convolution3D")
                return "Partial until the loaded node proves the group=1 OIDHW profile, explicit non-negative W/H/D padding, positive kernel/stride/dilation, supported activation, and TensorDescriptor CDHW Pack4 storage. Strict planning rejects every other branch.";
            if (operatorName == "Deconvolution3D")
                return "Partial until the loaded node proves the group=1 OIDHW profile, explicit non-negative W/H/D padding, zero output padding, positive kernel/stride/dilation, supported activation, and TensorDescriptor CDHW Pack4 storage. Strict planning rejects every other branch.";
            if (operatorName == "Pooling3D")
                return "Partial until the loaded node proves max/average global, adaptive, or explicit/full/SAME W/H/D pooling with TensorDescriptor CDHW Pack4 storage. Strict planning rejects invalid padding and all unlisted modes.";
            if (operatorName == "Interp")
                return "2D Pack4 Interp supports static size/scale and descriptor-only size_expr (for example 1w,1h) with GPU texture metadata only; no activation readback or buffer materialization is used. The CDHW runtime profile remains static nearest (1) or trilinear (2) resize with TensorDescriptor Pack4 storage and rejects dynamic size expressions.";
            if (Pack4LayoutOperators.Contains(operatorName))
                return "Partial CommandBuffer Pack4 layout profile: every branch validates logicalShape, storageShape, and layout. "
                    + "Only descriptor-proven identity/view branches alias; Permute, non-identity Slice/Tile/Packing/Cast, and storage-changing Reshape/Flatten dispatch a real texture transform. "
                    + "Dynamic data-dependent lengths and unsupported rank/axis/dtype profiles fail strict planning. Placeholder publication is not a production path.";
            if (operatorName == "TopK")
                return "Partial LinearMat texture profile: rank=1..4, FP32 values, Int32 logical indices, static axis and static k only. "
                    + "GPU-driven k requires a capacity-bounded GPU shape tensor; the current backend rejects it at plan time and never reads back k.";
            if (operatorName == "OneHot")
                return "Partial LinearMat texture profile: rank=1..3 indices, static positive depth and axis, FP32 values, and Int32 logical indices. "
                    + "GPU-driven depth requires a capacity-bounded GPU shape tensor; the current backend rejects it at plan time and never reads back depth.";
            if (operatorName == "NonZero" || operatorName == "Compress")
                return "Fixed-capacity LinearMat texture compaction. A second GPU-resident count output is mandatory; P0 admits the bounded value/count pair only as terminal graph outputs, rejects every ordinary value consumer, and never materializes the result to a ComputeBuffer.";
            if (operatorName == "Nms" || operatorName == "NonMaxSuppression")
                return "Static batch-one ONNX NonMaxSuppression with exact FP32 LinearMat box/score textures, a fixed padded [capacity,3] Int32 index texture, and a second GPU-resident Int32 count texture. Selection is deterministic on GPU; unbounded output, runtime threshold inputs, ordinary consumers that ignore count, and CPU/readback paths are rejected.";
            if (operatorName == "GatherND")
                return "LinearMat texture profile: rank-1 data, batch_dims=0, rank-2 [N,1] Int32 indices, and index_depth=1. Other GatherND layouts are rejected before dispatch.";
            if (operatorName == "ScatterND")
                return "LinearMat texture profile: rank-1 data/updates and rank-2 [N,1] unique Int32 indices. Serial texture dispatch is admitted only with explicit in-range and uniqueness proofs and reduction=none.";
            if (operatorName == "Scatter" || operatorName == "ScatterElements")
                return "LinearMat texture profile: rank-1 data/updates, axis=0, and rank-1 unique Int32 indices. Serial texture dispatch is admitted only with explicit in-range and uniqueness proofs and reduction=none.";
            if (operatorName == "Trilu")
                return "Texture-native ONNX Trilu over the final two axes. Static scalar k and upper=0|1 are required; exact rank-2 LinearMat/scalar-Pack4 or rank-3/rank-4 Pack4 storage is verified before dispatch.";
            if (operatorName == "RotaryEmbed")
                return "CommandBuffer Pack4 RotaryEmbed supports an even-width rank-3 [embed,sequence,head] source and exact rank-3 Texture2DArray cosine/sine caches with at least sequence*embed/2 values. Cache/state growth and unproven storage layouts fail strict planning.";
            if (operatorName == "DeepFillV2ContextualAttention")
                return "CommandBuffer Pack4 DeepFillV2 contextual attention supports only the verified HiFill case-1 texture profile: feature d3:100x128x96, mask d3:400x512x1, ksize=3, rate=2, stride=1, mask_downsample=8, and finite positive softmax scale/patch epsilon. All intermediate tensors are Texture2DArray resources; buffer activations and materialization fallbacks are rejected.";
            if (status == AexisOperatorCapabilityStatus.SupportedByProfile)
                return "Production support is conditional on a matching machine-readable profile and an accepted loaded-runtime node verifier result. A profile match alone never authorizes dispatch or a Buffer/materialization fallback.";
            return "A texture branch may exist for selected shapes, but this entry has not passed full Pack4 CommandBuffer model validation. Strict planning rejects partial capability.";
        }

        public static bool TryMatchTextureProfile(
            AexisOperatorCapability capability,
            AexisGraphModel.Layer layer,
            string targetBackend,
            string targetDtype,
            string targetStorageLayout,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            out AexisOperatorCapabilityProfile profile,
            out string reason)
        {
            profile = null;
            reason = null;
            if (capability == null)
            {
                reason = "No operator capability exists.";
                return false;
            }

            var profiles = capability.profiles ?? Array.Empty<AexisOperatorCapabilityProfile>();
            if (profiles.Length == 0)
            {
                reason = "The operator has no machine-readable production profile.";
                return false;
            }

            var rejectionReasons = new List<string>();
            for (var profileIndex = 0; profileIndex < profiles.Length; profileIndex++)
            {
                var candidate = profiles[profileIndex];
                if (candidate == null)
                    continue;
                if (!string.Equals(candidate.backend, targetBackend, StringComparison.OrdinalIgnoreCase))
                {
                    rejectionReasons.Add(candidate.profileId + ":backend");
                    continue;
                }
                if (!(candidate.dtypes ?? Array.Empty<string>()).Any(value => string.Equals(value, targetDtype, StringComparison.OrdinalIgnoreCase)))
                {
                    rejectionReasons.Add(candidate.profileId + ":dtype");
                    continue;
                }
                if (!(candidate.storageLayouts ?? Array.Empty<string>()).Any(value => string.Equals(value, targetStorageLayout, StringComparison.OrdinalIgnoreCase)))
                {
                    rejectionReasons.Add(candidate.profileId + ":storage-layout");
                    continue;
                }

                var inputCount = inputs?.Count ?? 0;
                var outputCount = layer?.topNames?.Length ?? Math.Max(0, layer?.tops ?? 0);
                if (inputCount < candidate.minInputs || (candidate.maxInputs >= 0 && inputCount > candidate.maxInputs))
                {
                    rejectionReasons.Add(candidate.profileId + ":input-count");
                    continue;
                }
                if (outputCount < candidate.minOutputs || (candidate.maxOutputs >= 0 && outputCount > candidate.maxOutputs))
                {
                    rejectionReasons.Add(candidate.profileId + ":output-count");
                    continue;
                }

                var ranks = candidate.inputRanks ?? Array.Empty<int>();
                var inputsMatch = true;
                for (var inputIndex = 0; inputIndex < inputCount; inputIndex++)
                {
                    var input = inputs[inputIndex];
                    if (input == null
                        || (candidate.requiresTextureBackedInputs && !input.textureBacked)
                        || !(candidate.dtypes ?? Array.Empty<string>()).Any(value => string.Equals(value, input.dtype, StringComparison.OrdinalIgnoreCase))
                        || !string.Equals(input.layout, targetStorageLayout, StringComparison.OrdinalIgnoreCase)
                        || !TryReadTextureDescriptorRank(input.logicalShape, out var inputRank)
                        || (ranks.Length > 0 && !ranks.Contains(inputRank)))
                    {
                        inputsMatch = false;
                        break;
                    }
                }
                if (!inputsMatch)
                {
                    rejectionReasons.Add(candidate.profileId + ":input-descriptor");
                    continue;
                }

                profile = candidate;
                return true;
            }

            reason = "No profile matched the concrete backend/dtype/storage-layout/I/O/rank contract"
                + (rejectionReasons.Count == 0 ? "." : ": " + string.Join(",", rejectionReasons) + ".");
            return false;
        }

        private static bool TryReadTextureDescriptorRank(int[] shape, out int rank)
        {
            rank = 0;
            if (shape == null || shape.Length != 5 || shape[0] < 1 || shape[0] > 4)
                return false;
            rank = shape[0];
            return shape.Skip(1).All(value => value > 0);
        }

        private static AexisOperatorCapabilityProfile[] FinalizeProfiles(
            string operatorName,
            AexisOperatorCapabilityProfile[] profiles,
            bool runtimeVerified)
        {
            profiles ??= Array.Empty<AexisOperatorCapabilityProfile>();
            if (profiles.Length == 0 && runtimeVerified)
            {
                profiles = new[]
                {
                    new AexisOperatorCapabilityProfile
                    {
                        backend = AexisOperatorCapabilityBackend.CommandBuffer,
                        layouts = ResolveLayouts(operatorName),
                        shapeProfile = "Exact logical/storage shape and parameter constraints are evaluated by the loaded-runtime node verifier.",
                        supportedParameters = Array.Empty<string>(),
                        rejectedParameters = new[] { "profile mismatch", "missing immutable constants", "placeholder", "buffer materialization" }
                    }
                };
            }

            ResolveIoContract(operatorName, out var minInputs, out var maxInputs, out var minOutputs, out var maxOutputs);
            var immutableWeights = RequiresImmutableWeights(operatorName);
            var ranks = ResolveRanks(operatorName);
            for (var index = 0; index < profiles.Length; index++)
            {
                var profile = profiles[index];
                if (profile == null)
                    continue;
                profile.profileId ??= ToProfileId(operatorName) + ".cmd-pack4." + index.ToString(CultureInfo.InvariantCulture);
                profile.backend ??= AexisOperatorCapabilityBackend.CommandBuffer;
                profile.layouts ??= ResolveLayouts(operatorName);
                profile.storageLayouts ??= new[] { AexisTexturePlanLayout.Packed4 };
                // BF16 is logically distinct but uses ARGBFloat physical storage.
                // The loaded verifier still proves the concrete Pack4 path before a
                // profile can be dispatched.
                profile.dtypes ??= new[] { "FP16", "BF16", "FP32" };
                profile.inputRanks ??= ranks;
                profile.outputRanks ??= ranks;
                profile.minInputs = minInputs;
                profile.maxInputs = maxInputs;
                profile.minOutputs = minOutputs;
                profile.maxOutputs = maxOutputs;
                profile.requiresTextureBackedInputs = minInputs > 0;
                profile.requiresImmutableWeights = immutableWeights;
                profile.requiresLoadedRuntimeVerification = runtimeVerified;
                profile.predicateEvaluator = runtimeVerified ? "aexis.loaded-runtime-node-verifier/v1" : "aexis.static-profile/v1";
                profile.weightContract = immutableWeights ? ResolveWeightContract(operatorName) : "none";
                profile.parameterPredicates ??= RequiredParameters.TryGetValue(operatorName, out var required)
                    ? required.Select(value => "required:" + value).ToArray()
                    : Array.Empty<string>();
                profile.supportedParameters ??= Array.Empty<string>();
                profile.rejectedParameters ??= Array.Empty<string>();
            }
            return profiles;
        }

        private static void ResolveIoContract(
            string operatorName,
            out int minInputs,
            out int maxInputs,
            out int minOutputs,
            out int maxOutputs)
        {
            minInputs = 1;
            maxInputs = 1;
            minOutputs = 1;
            maxOutputs = 1;
            switch (operatorName)
            {
                case "pnnx.Expression":
                case "MemoryData":
                case "Range":
                case "ConstantOfShape":
                case "RandomUniform":
                case "RandomNormal":
                    minInputs = maxInputs = 0;
                    break;
                case "Concat":
                case "Eltwise":
                    minInputs = 2;
                    maxInputs = -1;
                    break;
                case "BinaryOp":
                    minInputs = 1;
                    maxInputs = 2;
                    break;
                case "aten::to":
                    minInputs = 1;
                    maxInputs = -1;
                    break;
                case "Crop":
                    minInputs = 1;
                    maxInputs = 2;
                    break;
                case "CopyTo":
                case "MatMul":
                case "Gather":
                case "GatherElements":
                case "GatherND":
                    minInputs = maxInputs = 2;
                    break;
                case "Gemm":
                    // Gemm has both the usual constant-B one-input form and the
                    // texture-native two-input attention matmul form.
                    minInputs = 1;
                    maxInputs = 2;
                    break;
                case "Interp":
                    // The second input is an optional texture-resident shape
                    // reference for descriptor-only size_expr. It is metadata
                    // only: the runtime never reads its activation values back.
                    minInputs = 1;
                    maxInputs = 2;
                    break;
                case "PriorBox":
                    // ncnn supports both the MXNet one-feature profile and the
                    // Caffe profile with a second static image-shape texture.
                    // The loaded verifier proves the exact immutable parameters
                    // and RFloat LinearMat output capacity before dispatch.
                    minInputs = 1;
                    maxInputs = 2;
                    break;
                case "GridSample":
                case "ROIAlign":
                case "ROIPooling":
                case "PSROIPooling":
                    minInputs = maxInputs = 2;
                    break;
                case "DeformableConv2D":
                case "Einsum":
                    minInputs = 2;
                    maxInputs = 3;
                    break;
                case "Proposal":
                case "DetectionOutput":
                    minInputs = maxInputs = 3;
                    break;
                case "YoloDetectionOutput":
                case "Yolov3DetectionOutput":
                case "YoloDetectOut":
                case "Yolov3DetectOut":
                    minInputs = 1;
                    maxInputs = -1;
                    break;
                case "Where":
                    minInputs = maxInputs = 3;
                    break;
                case "Compress":
                    minInputs = maxInputs = 2;
                    minOutputs = maxOutputs = 2;
                    break;
                case "Nms":
                case "NonMaxSuppression":
                    minInputs = maxInputs = 2;
                    minOutputs = maxOutputs = 2;
                    break;
                case "Scatter":
                case "ScatterElements":
                case "ScatterND":
                    minInputs = maxInputs = 3;
                    break;
                case "SDPA":
                    minInputs = 3;
                    maxInputs = 6;
                    minOutputs = 1;
                    maxOutputs = 3;
                    break;
                case "MultiHeadAttention":
                    // ncnn accepts self-attention as one source blob. Cross-attention
                    // and attention-mask variants add K/V and mask blobs up to four.
                    minInputs = 1;
                    maxInputs = 4;
                    break;
                case "RotaryEmbed":
                    minInputs = maxInputs = 3;
                    break;
                case "DeepFillV2ContextualAttention":
                    minInputs = maxInputs = 2;
                    break;
                case "ShortConv":
                    minInputs = maxInputs = 3;
                    minOutputs = maxOutputs = 2;
                    break;
                case "GatedDeltaRule":
                    minInputs = maxInputs = 8;
                    minOutputs = maxOutputs = 2;
                    break;
                case "NonZero":
                    minOutputs = maxOutputs = 2;
                    break;
                case "Slice":
                    // ncnn Slice is a static split operation.  Unlike Split, it can
                    // produce any positive number of individually materialized
                    // texture outputs from one descriptor-backed input.
                    minOutputs = 1;
                    maxOutputs = -1;
                    break;
                case "MaxPoolingInd":
                    minOutputs = maxOutputs = 2;
                    break;
                case "MaxUnPooling":
                    // Values and the GPU-resident indices tensor are distinct
                    // texture inputs.  Do not describe this as a unary operator:
                    // that would reject the real Pack4 CommandBuffer implementation
                    // before its loaded-runtime shape verifier runs.
                    minInputs = maxInputs = 2;
                    break;
                case "TopK":
                    minOutputs = 1;
                    maxOutputs = 2;
                    break;
            }
        }

        private static bool RequiresImmutableWeights(string operatorName)
        {
            return operatorName.IndexOf("Convolution", StringComparison.Ordinal) >= 0
                || operatorName.IndexOf("Deconvolution", StringComparison.Ordinal) >= 0
                || operatorName == "Gemm" || operatorName == "InnerProduct" || operatorName == "BatchNorm" || operatorName == "InstanceNorm"
                || operatorName == "Bias" || operatorName == "LayerNorm" || operatorName == "RMSNorm" || operatorName == "GroupNorm" || operatorName == "Scale"
                || operatorName == "Quantize" || operatorName == "Dequantize" || operatorName == "Requantize" || operatorName == "MultiHeadAttention"
                || operatorName == "MemoryData" || operatorName == "Normalize" || operatorName == "PriorBox"
                || operatorName == "RNN" || operatorName == "GRU" || operatorName == "LSTM";
        }

        private static string ResolveWeightContract(string operatorName)
        {
            if (operatorName.IndexOf("Convolution", StringComparison.Ordinal) >= 0
                || operatorName.IndexOf("Deconvolution", StringComparison.Ordinal) >= 0)
                return "immutable-packed-kernel-and-optional-bias";
            if (operatorName == "Gemm" || operatorName == "InnerProduct")
                return "immutable-packed-matrix-and-optional-bias";
            if (operatorName == "RNN" || operatorName == "GRU" || operatorName == "LSTM")
                return "immutable-gate-input-recurrent-weights-and-bias";
            if (operatorName == "MemoryData")
                return "immutable-texture-constant";
            return "immutable-channel-constants";
        }

        private static string ToProfileId(string operatorName)
        {
            return new string((operatorName ?? "operator").Select(character => char.IsLetterOrDigit(character)
                ? char.ToLowerInvariant(character)
                : '-').ToArray()).Trim('-');
        }

        private static AexisOperatorCapabilityProfile[] ResolveProfiles(string operatorName)
        {
            const string CdhwShape = "logical [dims=4,w,h,d,c]; storage [dims=4,w,h,d,c]; Texture2DArray slices=d*ceil(c/4)";
            const string LayoutShape = "logical [dims,w,h,d,c]; storage explicitly records LinearMat or Pack4 Texture2DArray physical mapping; descriptor alias requires unchanged storage/layout/dtype";
            switch (operatorName)
            {
                case "RNN":
                case "GRU":
                case "LSTM":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Packed4" },
                            dtypes = new[] { "FP32" },
                            inputRanks = new[] { 3 },
                            outputRanks = new[] { 3 },
                            shapeProfile = "input logical/storage [dims=3,w=sequence<=256,h=1,d=1,c=input_size<=256]; output [dims=3,w=sequence,h=1,d=1,c=hidden_size<=256]; Texture2DArray slices=ceil(channels/4)",
                            supportedParameters = new[] { "forward direction", "zero implicit initial state", "one sequence output", "immutable gate W/R/bias", "RNN tanh", "GRU reset-after-matmul disabled", "LSTM IOFC gates" },
                            rejectedParameters = new[] { "bidirectional", "reverse", "initial_h/initial_c", "sequence_lens", "peepholes", "clip", "linear_before_reset", "state output", "FP16", "activation ComputeBuffer", "CPU state/readback" }
                        }
                    };
                case "RandomUniform":
                case "RandomNormal":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Packed4" },
                            dtypes = new[] { "FP32" },
                            inputRanks = Array.Empty<int>(),
                            outputRanks = new[] { 3, 4 },
                            shapeProfile = "zero-input static output; immutable dims=3|4,w,h,d,c and seed; storage Texture2DArray slices=depth*ceil(channels/4)",
                            supportedParameters = new[] { "explicit integral immutable seed", "static rank-3/4 Pack4 shape", "finite uniform low<high or normal mean/scale>0", "zero Pack4 tail lanes" },
                            rejectedParameters = new[] { "implicit seed", "rank<3/rank>4", "dynamic output shape", "FP16", "activation ComputeBuffer", "CPU RNG", "readback-driven state" }
                        }
                    };
                case "RandomLike":
                case "RandomUniformLike":
                case "RandomNormalLike":
                case "Bernoulli":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Packed4" },
                            dtypes = new[] { "FP32" },
                            inputRanks = new[] { 3, 4 },
                            outputRanks = new[] { 3, 4 },
                            shapeProfile = "one exact Pack4 Texture2DArray input/output; logical rank-3 CHW or rank-4 CDHW; storage slices=depth*ceil(channels/4)",
                            supportedParameters = new[] { "explicit integral immutable seed", "RandomUniformLike finite low<high", "RandomNormalLike finite mean and scale>0", "Bernoulli FP32 probability source", "zero Pack4 tail lanes" },
                            rejectedParameters = new[] { "implicit seed", "non-FP32 dtype", "LinearMat", "activation ComputeBuffer", "CPU RNG", "readback-driven state" }
                        }
                    };
                case "Multinomial":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Packed4" },
                            dtypes = new[] { "FP32" },
                            inputRanks = new[] { 3 },
                            outputRanks = new[] { 3 },
                            shapeProfile = "ONNX [batch,classes] maps to exact Pack4 logical/storage [dims=3,w=1,h=batch<=256,d=1,c=classes<=4096]; output [dims=3,w=1,h=batch,d=1,c=sample_size<=256] with exact FP32 lanes and logical Int32 category ids",
                            supportedParameters = new[] { "explicit integral immutable seed", "static sample_size", "sampling with replacement", "FP32 logits", "zero output Pack4 tail lanes" },
                            rejectedParameters = new[] { "implicit seed", "replacement=false", "FP16/BF16", "dynamic batch/classes/sample_size", "activation ComputeBuffer", "CPU RNG", "readback-driven state" }
                        }
                    };
                case "Nms":
                case "NonMaxSuppression":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Linear", "Packed4" },
                            dtypes = new[] { "FP32" },
                            inputRanks = new[] { 2 },
                            outputRanks = new[] { 1, 2 },
                            shapeProfile = "boxes logical/storage [dims=2,w=4,h=num_boxes,d=1,c=1] and scores [dims=2,w=num_boxes,h=num_classes,d=1,c=1] as exact RFloat LinearMat; output padded [capacity,3] plus Int32 count[1] as RFloat textures",
                            supportedParameters = new[] { "ONNX static batch=1", "immutable max_output_boxes_per_class/iou_threshold/score_threshold", "capacity contract", "center_point_box=0|1", "deterministic score tie lower-index-first" },
                            rejectedParameters = new[] { "dynamic thresholds", "batch>1", "unbounded output", "count readback", "count-unaware consumer", "activation ComputeBuffer" }
                        }
                    };
                case "PriorBox":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Packed4", "Linear" },
                            dtypes = new[] { "FP16", "FP32", "BF16" },
                            inputRanks = new[] { 3, 4 },
                            outputRanks = new[] { 1, 2 },
                            shapeProfile = "one Pack4 feature input for MXNet MultiBoxPrior, or feature plus static Pack4 image-shape input for Caffe PriorBox; output is capacity-proven FP32 RFloat LinearMat [4*w*h*num_prior] or [2,4*w*h*num_prior]",
                            supportedParameters = new[] { "immutable finite min_size/max_size/aspect_ratio/variance", "static feature and optional image dimensions", "flip/clip", "Caffe or MXNet parameter mode", "RFloat LinearMat output" },
                            rejectedParameters = new[] { "missing immutable parameter buffer", "dynamic feature/image shape", "invalid size/ratio", "unproven output capacity", "activation ComputeBuffer", "CPU box generation/readback" }
                        }
                    };
                case "ROIAlign":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "NCHW", "Packed4" },
                            dtypes = new[] { "FP32" },
                            inputRanks = new[] { 2, 3 },
                            outputRanks = new[] { 4 },
                            shapeProfile = "feature logical/storage [dims=3,w,h,d=1,c]; ROI logical/storage [dims=2,w=4,h=num_rois,d=1,c=1]; output [dims=4,w=pooled_w,h=pooled_h,d=num_rois,c] as Texture2DArray slices=num_rois*ceil(c/4)",
                            supportedParameters = new[] { "ONNX static batch=1", "static FP32 rois[num_rois,4]", "immutable zero INT32/INT64 batch_indices", "mode=avg", "coordinate_transformation_mode=half_pixel", "sampling_ratio>=0" },
                            rejectedParameters = new[] { "dynamic ROI count", "batch_indices activation", "batch_indices!=0", "batch>1", "mode=max", "non-half-pixel coordinate transform", "CPU/readback count" }
                        }
                    };
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
                            supportedParameters = new[] { "identity descriptor alias", "non-identity Pack4 texture repack/cast", "FP32/FP16/Int8/Int32/UInt8/logical Bool" },
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
                            layouts = new[] { "NCHW", "Packed4" },
                            shapeProfile = "logical [dims=3,w,h,1,c]; storage [dims=3,w,h,1,c]; Texture2DArray slices=ceil(c/4)",
                            supportedParameters = new[] { "resize_type=0..3", "static output W/H or positive scale W/H", "descriptor-only size_expr with one source and optional shape-reference texture", "align_corners=0|1" },
                            rejectedParameters = new[] { "data-dependent target sizes", "missing/mismatched descriptor for size_expr bottom", "size_expr with more than two output extents", "buffer materialization" }
                        },
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "CDHW", "Packed4" },
                            shapeProfile = CdhwShape,
                            supportedParameters = new[] { "resize_type=1 nearest|2 trilinear", "static output W/H/D or positive scale W/H/D", "align_corners=0|1" },
                            rejectedParameters = new[] { "dynamic_target_size", "size_expr", "bicubic/other resize modes" }
                        }
                    };
                case "PReLU":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "NCHW", "CDHW", "Packed4" },
                            shapeProfile = "logical rank 1..4 in exact Pack4 or LinearMat texture storage; non-scalar slopes require rank 3/4 and one immutable value per channel",
                            supportedParameters = new[] { "immutable FP32 scalar slope", "immutable FP32 channel slope with num_slope=logical channels", "texture-only LinearMat-to-Pack4 transform" },
                            rejectedParameters = new[] { "arbitrary ONNX broadcast slope", "missing/mismatched immutable slopes", "buffer materialization" }
                        }
                    };
                case "LRN":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "NCHW", "Packed4" },
                            shapeProfile = "static logical rank-3 CHW activation in exact Pack4 or LinearMat texture storage",
                            supportedParameters = new[] { "region_type=0 across channels", "region_type=1 within channel", "local_size>0", "finite alpha/beta/bias" },
                            rejectedParameters = new[] { "rank outside 3", "invalid region/local_size", "buffer materialization" }
                        }
                    };
                case "BNLL":
                case "Exp":
                case "Log":
                case "Power":
                case "Threshold":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Linear", "NCHW", "CDHW", "Packed4" },
                            inputRanks = new[] { 1, 2, 3, 4 },
                            outputRanks = new[] { 1, 2, 3, 4 },
                            shapeProfile = "rank-1/rank-2 require exact scalar-Pack4 storage; rank-3/rank-4 require exact CHW/CDHW Pack4 storage; output preserves logical/storage shape and zeroes invalid channel lanes",
                            supportedParameters = new[] { "FP16 or FP32 texture storage", "static finite formula parameters", "ceil(channel/4) packs with zero tail lanes" },
                            rejectedParameters = new[] { "LinearMat rematerialization", "logical/storage mismatch", "rank outside 1..4", "buffer materialization" }
                        }
                    };
                case "Bias":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "NCHW", "Packed4" },
                            shapeProfile = "logical [dims=3,w,h,d=1,c] or [dims=4,w,h,d,c]; immutable float4 bias constants; exact Texture2DArray Pack4 activation/output",
                            supportedParameters = new[] { "bias_data_size>0", "loaded immutable float4 bias", "input channels=bias_data_size", "rank-4 bias reused for every depth slice" },
                            rejectedParameters = new[] { "missing constants", "channel mismatch", "logical/storage mismatch", "transient ComputeBuffer input or output" }
                        }
                    };
                case "CopyTo":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "NCHW", "CDHW", "Packed4" },
                            shapeProfile = "two equal-rank logical inputs (dims=3 or dims=4); output shape/storage equals destination input",
                            supportedParameters = new[] { "static woffset/hoffset/doffset/coffset", "static numpy starts/axes", "negative starts", "non-pack-aligned channel offset", "in-bounds ROI" },
                            rejectedParameters = new[] { "rank mismatch", "rank outside 3..4", "layout or dtype mismatch", "out-of-bounds ROI", "buffer materialization" }
                        }
                    };
                case "Pooling1D":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "NCHW", "Packed4" },
                            shapeProfile = "logical [dims=3,w,h=1,d=1,c]; output [dims=3,out_w,h=1,d=1,c]",
                            supportedParameters = new[] { "pooling_type=max|average", "global_pooling", "adaptive_pooling with out_w>0", "pad_mode=0..3", "avgpool_count_include_pad=0|1", "asymmetric explicit padding" },
                            rejectedParameters = new[] { "rank/layout mismatch", "invalid pool or pad mode", "non-positive kernel/stride/out_w", "buffer materialization" }
                        }
                    };
                case "CumSum":
                case "CumulativeSum":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Linear", "NCHW", "CDHW", "Packed4" },
                            shapeProfile = "logical rank=1..4 with exact LinearMat or direct Pack4 texture storage; output preserves logical shape in LinearMat storage",
                            supportedParameters = new[] { "static axis in [-rank,rank-1]", "exclusive=0|1", "reverse=0|1", "NCNN CumulativeSum defaults axis=0, exclusive=0, reverse=0" },
                            rejectedParameters = new[] { "dynamic axis", "rank outside 1..4", "invalid boolean flags", "logical/storage mismatch", "buffer materialization" }
                        }
                    };
                case "Shape":
                case "Size":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            dtypes = new[] { "FP32" },
                            layouts = new[] { "Linear", "NCHW", "CDHW", "Packed4" },
                            shapeProfile = "static descriptor-backed input rank=1..4; output is a GPU-resident logical Int32 RFloat LinearMat tensor",
                            supportedParameters = new[] { "Shape static start/end slice", "Size exact static element count", "logical Int32 output", "no CPU shape readback" },
                            rejectedParameters = new[] { "dynamic rank", "empty Shape slice", "invalid logical/storage descriptor", "CPU readback", "buffer materialization" }
                        }
                    };
                case "Range":
                case "ConstantOfShape":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            dtypes = new[] { "FP32" },
                            layouts = new[] { "Linear" },
                            shapeProfile = "zero runtime inputs; all shape/value parameters are statically folded; output uses exact RFloat LinearMat storage",
                            supportedParameters = new[] { "static positive output extent", "finite FP32 values", "range delta!=0", "FP32-exact logical Int32 values" },
                            rejectedParameters = new[] { "dynamic shape/value input", "empty or rank>4 output", "non-finite value", "non-exact logical Int32 encoding", "buffer materialization" }
                        }
                    };
                case "Expand":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            dtypes = new[] { "FP32" },
                            layouts = new[] { "Linear", "NCHW", "CDHW", "Packed4" },
                            shapeProfile = "rank=1..4 input with static broadcast-compatible output shape; output is exact RFloat LinearMat storage",
                            supportedParameters = new[] { "static shape", "right-aligned broadcast", "texture-to-texture Pack4-to-LinearMat input transform" },
                            rejectedParameters = new[] { "dynamic shape", "rank>4", "zero/negative unsupported extent", "incompatible broadcast", "buffer materialization" }
                        }
                    };
                case "Where":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            dtypes = new[] { "FP32" },
                            layouts = new[] { "Linear", "NCHW", "CDHW", "Packed4" },
                            shapeProfile = "three rank=1..4 broadcast-compatible inputs; condition is logical Bool/Int32 and data logical dtypes match; output is RFloat LinearMat",
                            supportedParameters = new[] { "static multidirectional broadcast", "logical Bool or Int32 condition", "matching FP32/Int32 data semantics" },
                            rejectedParameters = new[] { "dtype mismatch", "incompatible broadcast", "rank>4", "buffer materialization" }
                        }
                    };
                case "Gather":
                case "GatherElements":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            dtypes = new[] { "FP32" },
                            layouts = new[] { "Linear", "NCHW", "CDHW", "Packed4" },
                            shapeProfile = "rank=1..4 data and logical Int32 indices with static axis; output rank<=4 in exact RFloat LinearMat storage",
                            supportedParameters = new[] { "negative indices normalized once", "exporter-proven indices in range", "GatherElements equal ranks and bounded non-axis dimensions" },
                            rejectedParameters = new[] { "missing in-range proof", "non-Int32 logical indices", "invalid axis", "output rank>4", "buffer materialization" }
                        }
                    };
                case "ArgMax":
                case "ArgMin":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            dtypes = new[] { "FP32" },
                            layouts = new[] { "Linear", "NCHW", "CDHW", "Packed4" },
                            shapeProfile = "FP32 rank=1..4 input, static axis, keepdims=0|1; logical Int32 output in exact RFloat LinearMat storage",
                            supportedParameters = new[] { "select_last_index=0|1", "axis in [-rank,rank-1]", "deterministic first/last tie handling" },
                            rejectedParameters = new[] { "non-FP32 data", "invalid axis/boolean flags", "buffer materialization" }
                        }
                    };
                case "TopK":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            dtypes = new[] { "FP32" },
                            layouts = new[] { "Linear" },
                            shapeProfile = "rank=1..4; output axis has static k; LinearMat values plus logical Int32 indices",
                            supportedParameters = new[] { "axis in [-rank,rank-1]", "static k in [1,axis_size]", "largest=0|1", "FP32 values" },
                            rejectedParameters = new[] { "dynamic k", "rank>4", "k>axis_size", "CPU readback for output shape" }
                        }
                    };
                case "OneHot":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            dtypes = new[] { "FP32" },
                            layouts = new[] { "Linear" },
                            shapeProfile = "indices rank=1..3; output rank=2..4; static depth axis inserted into output",
                            supportedParameters = new[] { "Int32 logical indices", "static depth>0", "axis in [-rank-1,rank]", "FP32 on/off values" },
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
                            dtypes = new[] { "FP32" },
                            layouts = new[] { "Linear" },
                            shapeProfile = "rank-1 static input; NonZero logical output [1,capacity], Compress logical output [capacity], plus second GPU-resident Int32 count output; physical storage is exact RFloat LinearMat",
                            supportedParameters = new[] { "bounded output capacity", "terminal value/count graph outputs", "Compress condition rank=1 and same length" },
                            rejectedParameters = new[] { "ordinary consumer of the bounded value tensor", "CPU count readback", "unbounded output", "rank!=1", "overflow without reject policy" }
                        }
                    };
                case "GatherND":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            dtypes = new[] { "FP32" },
                            layouts = new[] { "Linear" },
                            shapeProfile = "rank-1 data and rank-2 [N,1] Int32 indices in exact RFloat LinearMat storage",
                            supportedParameters = new[] { "batch_dims=0", "index_depth=1", "static rank-2 [N,1] indices", "exporter-proven in-range indices" },
                            rejectedParameters = new[] { "batch_dims!=0", "non-Int32 logical indices", "index_depth!=1", "data rank!=1", "indices rank!=2 or final dimension!=1" }
                        }
                    };
                case "Scatter":
                case "ScatterElements":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            dtypes = new[] { "FP32" },
                            layouts = new[] { "Linear" },
                            shapeProfile = "rank-1 data/updates, axis=0, and rank-1 unique Int32 indices in exact RFloat LinearMat storage",
                            supportedParameters = new[] { "reduction=none", "exporter-proven unique and in-range indices", "static matching update and index lengths" },
                            rejectedParameters = new[] { "axis!=0", "duplicate or out-of-range destinations", "non-Int32 logical indices", "rank!=1", "reduction other than none" }
                        }
                    };
                case "ScatterND":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            dtypes = new[] { "FP32" },
                            layouts = new[] { "Linear" },
                            shapeProfile = "rank-1 data/updates and rank-2 [N,1] unique Int32 indices in exact RFloat LinearMat storage",
                            supportedParameters = new[] { "reduction=none", "index_depth=1", "exporter-proven unique and in-range indices" },
                            rejectedParameters = new[] { "duplicate or out-of-range destinations", "non-Int32 logical indices", "indices rank!=2 or final dimension!=1", "reduction other than none" }
                        }
                    };
                case "Trilu":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Linear", "NCHW", "CDHW", "Packed4" },
                            shapeProfile = "logical rank=2..4; final axes map exactly to texture Y/X; output logical and storage descriptors equal the input",
                            supportedParameters = new[] { "upper=0|1", "static scalar Int32-range k", "rank-2 LinearMat/scalar Pack4", "rank-3/rank-4 exact Pack4 Texture2DArray" },
                            rejectedParameters = new[] { "dynamic k", "rank<2 or rank>4", "packed-lane matrix storage", "logical/storage X/Y mismatch", "buffer materialization" }
                        }
                    };
                case "DeepFillV2ContextualAttention":
                    return new[]
                    {
                        new AexisOperatorCapabilityProfile
                        {
                            backend = AexisOperatorCapabilityBackend.CommandBuffer,
                            layouts = new[] { "Packed4" },
                            inputRanks = new[] { 3 },
                            outputRanks = new[] { 3 },
                            shapeProfile = "feature logical/storage d3:100x128x96 and mask logical/storage d3:400x512x1 as exact Pack4 Texture2DArray resources; output preserves the feature descriptor",
                            supportedParameters = new[] { "ksize=3", "rate=2", "stride=1", "mask_downsample=8", "finite positive patch_epsilon", "finite positive softmax_scale", "four real CommandBuffer texture dispatches" },
                            rejectedParameters = new[] { "other shapes or contextual-attention geometry", "LinearMat/buffer activations", "ComputeBuffer materialization", "placeholder output" }
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

            AexisTextureExecutionPlan texturePlan = null;
            // An explicitly supplied empty input list denotes a statically closed
            // graph (for example seeded RandomUniform). It needs the same plan
            // proof as a graph with external Pack4 textures.
            if (request.textureInputs != null && (request.textureInputs.Length > 0 || request.strict || request.nodeVerifier != null))
            {
                texturePlan = AexisTextureExecutionPlanner.Analyze(model, new AexisTextureExecutionPlanRequest
                {
                    modelName = request.modelName,
                    targetBackend = request.targetBackend,
                    targetDtype = request.targetDtype,
                    targetLayout = request.targetLayout,
                    strict = request.strict,
                    inputs = request.textureInputs,
                    nodeVerifier = request.nodeVerifier
                });
            }

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

                var operatorName = ResolveCapabilityOperatorName(layer);
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
                var textureNode = texturePlan?.nodes?.FirstOrDefault(node => node != null && node.layerIndex == index);
                if (textureNode?.inputs != null && textureNode.inputs.Length > 0)
                    nodeInputs = textureNode.inputs.Select(ToPreflightDescriptor).ToList();
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

                var textureNodeInputs = textureNode?.inputs ?? Array.Empty<AexisTexturePlanTensorDescriptor>();
                AexisOperatorCapabilities.TryMatchTextureProfile(
                    capability,
                    layer,
                    request.targetBackend,
                    request.targetDtype,
                    request.targetLayout,
                    textureNodeInputs,
                    out var matchedProfile,
                    out var profileReason);
                // Exact model admission comes from the texture plan. Conditional entries
                // additionally require the loaded session verifier to accept this node.
                var strictCapabilityEligible = textureNode?.accepted == true;
                var strictEligible = !request.strict || (strictCapabilityEligible && missingForNode.Length == 0 && nodeIssues.Count == 0);
                var recommendation = d3Diagnostic != null
                    ? d3Diagnostic.recommendedAction
                    : ResolveRecommendedAction(capability, request.targetBackend, request.targetDtype, missingForNode.Length > 0);
                if (request.strict && !strictEligible && nodeIssues.Count == 0)
                {
                    var planDiagnostic = texturePlan?.diagnostics?.FirstOrDefault(diagnostic => diagnostic != null && diagnostic.layerIndex == index && diagnostic.blocking);
                    nodeIssues.Add(string.Equals(capability.status, AexisOperatorCapabilityStatus.Unsupported, StringComparison.Ordinal)
                        ? capability.limitations
                        : planDiagnostic?.reason
                        ?? (texturePlan == null
                            ? "Strict model admission requires exact textureInputs and a model-level texture execution plan."
                            : !string.IsNullOrWhiteSpace(profileReason)
                                ? profileReason
                                : "Strict target rejected the concrete node profile."));
                }

                nodes.Add(new AexisModelPreflightNode
                {
                    layerIndex = index,
                    layer = layer.name ?? string.Empty,
                    operatorName = operatorName,
                    canonicalOperator = capability.canonicalOperator,
                    status = capability.status,
                    matchedProfileId = matchedProfile?.profileId ?? string.Empty,
                    strictEligible = strictEligible,
                    bottomBlobs = CloneStrings(layer.bottomNames),
                    topBlobs = CloneStrings(layer.topNames),
                    inputs = nodeInputs.ToArray(),
                    outputs = textureNode?.outputs == null
                        ? Array.Empty<AexisPreflightTensorDescriptor>()
                        : textureNode.outputs.Select(ToPreflightDescriptor).ToArray(),
                    missingParameters = missingForNode,
                    issues = nodeIssues.ToArray(),
                    recommendedAction = recommendation
                });

                PropagateOutputDescriptors(layer, capability, nodeInputs, textureNode?.outputs, knownBlobs);
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
            if (request.textureInputs != null && (request.textureInputs.Length > 0 || request.strict))
            {
                report.texturePlan = texturePlan;
                report.strictEligible &= texturePlan != null && texturePlan.strictEligible;
            }
            report.summary = "nodes=" + report.nodes.Length.ToString(CultureInfo.InvariantCulture)
                + " missingNodes=" + report.missingNodes.Length.ToString(CultureInfo.InvariantCulture)
                + " missingParameters=" + report.missingParameters.Length.ToString(CultureInfo.InvariantCulture)
                + " missingDependencies=" + report.missingDependencies.Length.ToString(CultureInfo.InvariantCulture)
                + " texturePlan=" + (report.texturePlan == null ? "not-requested" : (report.texturePlan.strictEligible ? "eligible" : "rejected"))
                + " strictEligible=" + (report.strictEligible ? "true" : "false");
            return report;
        }

        private static string ResolveCapabilityOperatorName(AexisGraphModel.Layer layer)
        {
            var operatorName = !string.IsNullOrWhiteSpace(layer?.typeName) ? layer.typeName : layer?.type.ToString();
            if (!string.Equals(operatorName, "RandomLike", StringComparison.Ordinal)
                || layer?.stringParams == null
                || !layer.stringParams.TryGetValue("aexis.random.operator", out var randomOperator)
                || string.IsNullOrWhiteSpace(randomOperator))
                return operatorName;

            switch (randomOperator)
            {
                case "RandomUniform":
                case "RandomNormal":
                case "RandomUniformLike":
                case "RandomNormalLike":
                case "Bernoulli":
                    return randomOperator;
                default:
                    return operatorName;
            }
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

            var inputRank = ResolveLogicalRank(inputs != null && inputs.Count > 0 ? inputs[0]?.logicalShape : null);
            var hasStaticParameter = operatorName == "TopK"
                ? HasLayerParameter(layer, "k", 1)
                : operatorName == "OneHot"
                    ? HasLayerParameter(layer, "depth", 1)
                    : true;
            var batchDims = ReadLayerInt(layer, "batch_dims", 0, 0);
            var uniqueIndices = ReadLayerInt(layer, "unique_indices", -1, 0) != 0;
            var outputCapacity = ReadLayerInt(layer, "capacity", 30, 0);
            var isBoundedCompaction = operatorName == "NonZero" || operatorName == "Compress";
            var indexInput = operatorName == "GatherND" || operatorName == "Scatter" || operatorName == "ScatterElements" || operatorName == "ScatterND" ? 1 : 0;
            var indexDataType = inputs != null && inputs.Count > indexInput
                && string.Equals(inputs[indexInput]?.logicalDtype, "Int32", StringComparison.Ordinal)
                    ? TensorDataType.Int32
                    : TensorDataType.Unknown;
            return OnnxExecutionShapePlanner.Validate(new OnnxExecutionNodeContract
            {
                name = layer.name ?? string.Empty,
                opType = operatorName,
                inputRank = inputRank,
                batchDims = batchDims,
                indexDataType = indexDataType,
                dynamicParameter = !hasStaticParameter,
                uniqueIndices = uniqueIndices
                , outputCapacity = outputCapacity
                , outputShape = isBoundedCompaction ? new GpuShapeTensorContract
                {
                    rank = operatorName == "NonZero" ? 2 : 1,
                    capacity = outputCapacity,
                    lengthPolicy = GpuShapeLengthPolicy.CapacityBounded,
                    overflowPolicy = "reject",
                    lengthTensor = (layer.topNames != null && layer.topNames.Length > 1 ? layer.topNames[1] : string.Empty)
                } : null
            });
        }

        private static int ResolveLogicalRank(int[] shape)
        {
            if (shape == null || shape.Length == 0)
                return 0;
            if (shape.Length == 5 && shape[0] >= 1 && shape[0] <= 4)
                return shape[0];
            return shape.Length;
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
            AexisTexturePlanTensorDescriptor[] verifiedOutputs,
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

                    var descriptor = verifiedOutputs != null && i < verifiedOutputs.Length && verifiedOutputs[i] != null
                        ? ToPreflightDescriptor(verifiedOutputs[i])
                        : capability.shapeInference && inputs.Count > 0
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

        private static AexisPreflightTensorDescriptor ToPreflightDescriptor(AexisTexturePlanTensorDescriptor source)
        {
            if (source == null)
                return new AexisPreflightTensorDescriptor { blob = string.Empty, logicalShape = Array.Empty<int>(), storageShape = Array.Empty<int>(), layout = "Unknown", dtype = "Unknown" };
            return new AexisPreflightTensorDescriptor
            {
                blob = source.blob ?? string.Empty,
                logicalShape = source.logicalShape == null ? Array.Empty<int>() : (int[])source.logicalShape.Clone(),
                storageShape = source.storageShape == null ? Array.Empty<int>() : (int[])source.storageShape.Clone(),
                layout = source.layout ?? "Unknown",
                dtype = source.dtype ?? "Unknown",
                logicalDtype = source.logicalDtype ?? "Unknown"
            };
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
                matchedProfileId = string.Empty,
                strictEligible = false,
                bottomBlobs = CloneStrings(layer.bottomNames),
                topBlobs = CloneStrings(layer.topNames),
                inputs = Array.Empty<AexisPreflightTensorDescriptor>(),
                outputs = Array.Empty<AexisPreflightTensorDescriptor>(),
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
            if (capability.status == AexisOperatorCapabilityStatus.SupportedByProfile)
                return "Supply exact texture descriptors and use the loaded-runtime verifier; dispatch is allowed only when the concrete profile is accepted.";
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
                dtype = source.dtype ?? "Unknown",
                logicalDtype = source.logicalDtype ?? "Unknown"
            };
        }

        private static string[] CloneStrings(string[] values)
        {
            return values == null ? Array.Empty<string>() : values.Select(value => value ?? string.Empty).ToArray();
        }
    }
}
