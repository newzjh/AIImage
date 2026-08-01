using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Aexis.Async;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    public static class AexisLayerFactory
    {
        private static readonly Dictionary<string, string> CanonicalAliases = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            { "CumulativeSum", "CumSum" },
            { "CumSum", "CumSum" },
            { "ConvolutionDepthWise1D", "Convolution1D" },
            { "ConvolutionDepthWise3D", "ConvDw3D" },
            { "Deconvolution1D", "Deconvolution" },
            { "DeconvolutionDepthWise1D", "DeconvDw1D" },
            { "DeconvolutionDepthWise3D", "DeconvDw3D" },
            { "YoloDetectionOutput", "YoloDetectOut" },
            { "Yolov3DetectionOutput", "Yolov3DetectOut" },
            { "NonMaxSuppression", "Nms" },
            { "RandomUniformLike", "RandomLike" },
            { "RandomNormalLike", "RandomLike" },
            { "RandomUniform", "RandomLike" },
            { "RandomNormal", "RandomLike" },
            { "Bernoulli", "RandomLike" },
            { "Multinomial", "Multinomial" },
            { "StatisticsPooling", "StatsPooling" },
            { "InverseSpectrogram", "InvSpectrogram" },
        };

        private static readonly Dictionary<AexisLayerTypeKey, Func<AexisBaseLayer>> Registry = new Dictionary<AexisLayerTypeKey, Func<AexisBaseLayer>>
        {
            { AexisLayerTypes.Input, () => new AexisInputLayer() },
            { AexisLayerTypes.PnnxExpression, () => new AexisPnnxExpressionLayer() },
            { AexisLayerTypes.AtenTo, () => new AexisAtenToLayer() },
            { AexisLayerTypes.AbsVal, () => new AexisUnaryOpAliasLayer(AexisLayerTypes.AbsVal, 0) },
            { AexisLayerTypes.Split, () => new AexisSplitLayer() },
            { AexisLayerTypes.Concat, () => new AexisConcatLayer() },
            { AexisLayerTypes.TanH, () => new AexisUnaryOpAliasLayer(AexisLayerTypes.TanH, 16) },
            { AexisLayerTypes.Reshape, () => new AexisReshapeLayer() },
            { AexisLayerTypes.ShuffleChannel, () => new AexisShuffleChannelLayer() },
            { AexisLayerTypes.Permute, () => new AexisPermuteLayer() },
            { AexisLayerTypes.Slice, () => new AexisSliceLayer() },
            { AexisLayerTypes.ExpandDims, () => new AexisExpandDimsLayer() },
            { AexisLayerTypes.Squeeze, () => new AexisSqueezeLayer() },
            { AexisLayerTypes.Crop, () => new AexisCropLayer() },
            { AexisLayerTypes.Convolution, () => new AexisConvolutionLayer() },
            { AexisLayerTypes.Convolution3D, () => new AexisConvolution3DLayer() },
            { AexisLayerTypes.ConvDw3D, () => new AexisConvolutionDepthWise3DLayer() },
            { AexisLayerTypes.Convolution1D, () => new AexisConvolution1DLayer() },
            { AexisLayerTypes.ConvolutionDepthWise, () => new AexisConvolutionDepthWiseLayer() },
            { AexisLayerTypes.Deconvolution, () => new AexisDeconvolutionLayer() },
            { AexisLayerTypes.Deconvolution3D, () => new AexisDeconvolution3DLayer() },
            { AexisLayerTypes.DeconvolutionDepthWise, () => new AexisDeconvolutionDepthWiseLayer() },
            { AexisLayerTypes.DeconvDw3D, () => new AexisDeconvolutionDepthWise3DLayer() },
            { AexisLayerTypes.DeconvDw1D, () => new AexisDeconvolutionDepthWise1DLayer() },
            { AexisLayerTypes.Interp, () => new AexisInterpLayer() },
            { AexisLayerTypes.Dropout, () => new AexisDropoutLayer() },
            { AexisLayerTypes.Eltwise, () => new AexisEltwiseLayer() },
            { AexisLayerTypes.ELU, () => new AexisPointwiseFormulaLayer(AexisLayerTypes.ELU) },
            { AexisLayerTypes.Erf, () => new AexisPointwiseFormulaLayer(AexisLayerTypes.Erf) },
            { AexisLayerTypes.Flatten, () => new AexisFlattenLayer() },
            { AexisLayerTypes.BinaryOp, () => new AexisBinaryOpLayer() },
            { AexisLayerTypes.UnaryOp, () => new AexisUnaryOpLayer() },
            { AexisLayerTypes.HardSigmoid, () => new AexisPointwiseFormulaLayer(AexisLayerTypes.HardSigmoid) },
            { AexisLayerTypes.HardSwish, () => new AexisPointwiseFormulaLayer(AexisLayerTypes.HardSwish) },
            { AexisLayerTypes.InstanceNorm, () => new AexisInstanceNormLayer() },
            { AexisLayerTypes.LRN, () => new AexisLRNLayer() },
            { AexisLayerTypes.Mish, () => new AexisPointwiseFormulaLayer(AexisLayerTypes.Mish) },
            { AexisLayerTypes.Swish, () => new AexisSwishLayer() },
            { AexisLayerTypes.Noop, () => new AexisNoopLayer() },
            { AexisLayerTypes.Normalize, () => new AexisNormalizeLayer() },
            { AexisLayerTypes.Packing, () => new AexisPackingLayer() },
            { AexisLayerTypes.PixelShuffle, () => new AexisPixelShuffleLayer() },
            { AexisLayerTypes.PReLU, () => new AexisPReLULayer() },
            { AexisLayerTypes.PriorBox, () => new AexisPriorBoxLayer() },
            { AexisLayerTypes.Quantize, () => new AexisQuantizeLayer() },
            { AexisLayerTypes.Dequantize, () => new AexisDequantizeLayer() },
            { AexisLayerTypes.Requantize, () => new AexisRequantizeLayer() },
            { AexisLayerTypes.Reorg, () => new AexisReorgLayer() },
            { AexisLayerTypes.Sigmoid, () => new AexisSigmoidLayer() },
            { AexisLayerTypes.RMSNorm, () => new AexisRMSNormLayer() },
            { AexisLayerTypes.RotaryEmbed, () => new AexisRotaryEmbedLayer() },
            { AexisLayerTypes.Scale, () => new AexisScaleLayer() },
            { AexisLayerTypes.SDPA, () => new AexisSdpaLayer() },
            { AexisLayerTypes.SELU, () => new AexisPointwiseFormulaLayer(AexisLayerTypes.SELU) },
            { AexisLayerTypes.Shrink, () => new AexisPointwiseFormulaLayer(AexisLayerTypes.Shrink) },
            { AexisLayerTypes.Softplus, () => new AexisPointwiseFormulaLayer(AexisLayerTypes.Softplus) },
            { AexisLayerTypes.Softsign, () => new AexisPointwiseFormulaLayer(AexisLayerTypes.Softsign) },
            { AexisLayerTypes.IsInf, () => new AexisPointwiseFormulaLayer(AexisLayerTypes.IsInf) },
            { AexisLayerTypes.IsNaN, () => new AexisPointwiseFormulaLayer(AexisLayerTypes.IsNaN) },
            { AexisLayerTypes.GELU, () => new AexisGeluLayer() },
            { AexisLayerTypes.Cast, () => new AexisCastLayer() },
            { AexisLayerTypes.CELU, () => new AexisPointwiseFormulaLayer(AexisLayerTypes.CELU) },
            { AexisLayerTypes.Clip, () => new AexisClipLayer() },
            { AexisLayerTypes.Trilu, () => new AexisTriluLayer() },
            { AexisLayerTypes.Softmax, () => new AexisSoftmaxLayer() },
            { AexisLayerTypes.Padding, () => new AexisPaddingLayer() },
            { AexisLayerTypes.Pooling, () => new AexisPoolingLayer() },
            { AexisLayerTypes.Pooling3D, () => new AexisPooling3DLayer() },
            { AexisLayerTypes.InnerProduct, () => new AexisInnerProductLayer() },
            { AexisLayerTypes.MatMul, () => new AexisMatMulLayer() },
            { AexisLayerTypes.Gemm, () => new AexisGemmLayer() },
            { AexisLayerTypes.MultiHeadAttention, () => new AexisMultiHeadAttentionLayer() },
            { AexisLayerTypes.LayerNorm, () => new AexisLayerNormLayer() },
            { AexisLayerTypes.GroupNorm, () => new AexisGroupNormLayer() },
            { AexisLayerTypes.BatchNorm, () => new AexisBatchNormLayer() },
            { AexisLayerTypes.Embed, () => new AexisEmbedLayer() },
            { AexisLayerTypes.Reduction, () => new AexisReductionLayer() },
            { AexisLayerTypes.MemoryData, () => new AexisMemoryDataLayer() },
            { AexisLayerTypes.ReLU, () => new AexisReLULayer() },
            { AexisLayerTypes.DeepCopy, () => new AexisDeepCopyLayer() },
            { AexisLayerTypes.MaxPoolingInd, () => new AexisMaxPoolingIndLayer() },
            { AexisLayerTypes.MaxUnPooling, () => new AexisMaxUnPoolingLayer() },
            { AexisLayerTypes.Unfold, () => new AexisUnfoldLayer() },
            { AexisLayerTypes.ExtractPatches, () => new AexisExtractPatchesLayer() },
            { AexisLayerTypes.GridSample, () => new AexisNativeP1VisionLayer(AexisLayerTypes.GridSample) },
            { AexisLayerTypes.DeformableConv2D, () => new AexisNativeP1VisionLayer(AexisLayerTypes.DeformableConv2D) },
            { AexisLayerTypes.Fold, () => new AexisNativeP1VisionLayer(AexisLayerTypes.Fold) },
            { AexisLayerTypes.Flip, () => new AexisFlipLayer() },
            { AexisLayerTypes.GLU, () => new AexisNativeP1VisionLayer(AexisLayerTypes.GLU) },
            { AexisLayerTypes.Einsum, () => new AexisNativeP1VisionLayer(AexisLayerTypes.Einsum) },
            { AexisLayerTypes.Diag, () => new AexisNativeP1VisionLayer(AexisLayerTypes.Diag) },
            { AexisLayerTypes.SPP, () => new AexisNativeP1VisionLayer(AexisLayerTypes.SPP) },
            { AexisLayerTypes.ROIAlign, () => new AexisNativeP1VisionLayer(AexisLayerTypes.ROIAlign) },
            { AexisLayerTypes.ROIPooling, () => new AexisNativeP1VisionLayer(AexisLayerTypes.ROIPooling) },
            { AexisLayerTypes.PSROIPooling, () => new AexisNativeP1VisionLayer(AexisLayerTypes.PSROIPooling) },
            { AexisLayerTypes.Proposal, () => new AexisNativeP1VisionLayer(AexisLayerTypes.Proposal) },
            { AexisLayerTypes.DetectionOutput, () => new AexisNativeP1VisionLayer(AexisLayerTypes.DetectionOutput) },
            { AexisLayerTypes.YoloDetectOut, () => new AexisNativeP1VisionLayer(AexisLayerTypes.YoloDetectOut) },
            { AexisLayerTypes.Yolov3DetectOut, () => new AexisNativeP1VisionLayer(AexisLayerTypes.Yolov3DetectOut) },
            { AexisLayerTypes.DeepFillV2ContextualAttention, () => new AexisDeepFillV2ContextualAttentionLayer() },
            { AexisLayerTypes.Tile, () => new AexisTileLayer() },
            { AexisLayerTypes.Shape, () => new AexisShapeLayer() },
            { AexisLayerTypes.Size, () => new AexisSizeLayer() },
            { AexisLayerTypes.Range, () => new AexisRangeLayer() },
            { AexisLayerTypes.ConstantOfShape, () => new AexisConstantOfShapeLayer() },
            { AexisLayerTypes.Expand, () => new AexisExpandLayer() },
            { AexisLayerTypes.ArgMax, () => new AexisArgReduceLayer(AexisLayerTypes.ArgMax, reduceMax: true) },
            { AexisLayerTypes.ArgMin, () => new AexisArgReduceLayer(AexisLayerTypes.ArgMin, reduceMax: false) },
            { AexisLayerTypes.Where, () => new AexisWhereLayer() },
            { AexisLayerTypes.TopK, () => new AexisTopKLayer() },
            { AexisLayerTypes.Nms, () => new AexisNonMaxSuppressionLayer() },
            { AexisLayerTypes.NonZero, () => new AexisNonZeroLayer() },
            { AexisLayerTypes.OneHot, () => new AexisOneHotLayer() },
            { AexisLayerTypes.CumSum, () => new AexisCumSumLayer() },
            { AexisLayerTypes.Bias, () => new AexisBiasLayer() },
            { AexisLayerTypes.BNLL, () => new AexisPointwiseFormulaLayer(AexisLayerTypes.BNLL) },
            { AexisLayerTypes.CopyTo, () => new AexisCopyToLayer() },
            { AexisLayerTypes.Exp, () => new AexisPointwiseFormulaLayer(AexisLayerTypes.Exp) },
            { AexisLayerTypes.Log, () => new AexisPointwiseFormulaLayer(AexisLayerTypes.Log) },
            { AexisLayerTypes.Power, () => new AexisPointwiseFormulaLayer(AexisLayerTypes.Power) },
            { AexisLayerTypes.Threshold, () => new AexisPointwiseFormulaLayer(AexisLayerTypes.Threshold) },
            { AexisLayerTypes.ThresholdedRelu, () => new AexisPointwiseFormulaLayer(AexisLayerTypes.ThresholdedRelu) },
            { AexisLayerTypes.MVN, () => new AexisMvnLayer() },
            { AexisLayerTypes.Pooling1D, () => new AexisPooling1DLayer() },
            { AexisLayerTypes.Compress, () => new AexisCompressLayer() },
            { AexisLayerTypes.Gather, () => new AexisGatherLayer() },
            { AexisLayerTypes.GatherElements, () => new AexisGatherElementsLayer() },
            { AexisLayerTypes.GatherND, () => new AexisGatherNDLayer() },
            { AexisLayerTypes.ScatterElements, () => new AexisScatterLayer(AexisLayerTypes.ScatterElements) },
            { AexisLayerTypes.ScatterND, () => new AexisScatterLayer(AexisLayerTypes.ScatterND) },
            { AexisLayerTypes.Scatter, () => new AexisScatterLayer(AexisLayerTypes.Scatter) },
            { AexisLayerTypes.ShortConv, () => new AexisShortConvLayer() },
            { AexisLayerTypes.GatedDeltaRule, () => new AexisGatedDeltaRuleLayer() },
            { AexisLayerTypes.RandomLike, () => new AexisDeterministicRandomLayer() },
            { AexisLayerTypes.Multinomial, () => new AexisMultinomialLayer() },
            { AexisLayerTypes.StatsPooling, () => new AexisStatisticsPoolingLayer() },
            { AexisLayerTypes.Spectrogram, () => new AexisSpectrogramLayer(inverse: false) },
            { AexisLayerTypes.InvSpectrogram, () => new AexisSpectrogramLayer(inverse: true) },
            { AexisLayerTypes.RNN, () => new AexisRecurrentLayer(AexisRecurrentKind.Rnn) },
            { AexisLayerTypes.GRU, () => new AexisRecurrentLayer(AexisRecurrentKind.Gru) },
            { AexisLayerTypes.LSTM, () => new AexisRecurrentLayer(AexisRecurrentKind.Lstm) },
        };

        public static IReadOnlyList<AexisBaseLayer> CreateModelLayers(IList<AexisGraphModel.Layer> layers)
        {
            if (layers == null || layers.Count == 0)
                return Array.Empty<AexisBaseLayer>();

            var result = new AexisBaseLayer[layers.Count];
            for (var i = 0; i < layers.Count; i++)
                result[i] = Create(layers[i]);
            return result;
        }

        // Metadata consumers must inspect registration without instantiating or executing a layer.
        public static IReadOnlyList<AexisLayerTypeKey> GetRegisteredLayerTypes()
        {
            var types = new List<AexisLayerTypeKey>(Registry.Keys);
            types.Sort((left, right) => string.CompareOrdinal(left.ToString(), right.ToString()));
            return types;
        }

        public static IReadOnlyList<string> GetRegisteredLayerTypeNames()
        {
            var names = new List<string>(Registry.Count + AexisCustomLayerRegistry.GetRegisteredTypeNames().Length);
            foreach (var type in Registry.Keys)
                names.Add(type.ToString());
            names.AddRange(AexisCustomLayerRegistry.GetRegisteredTypeNames());
            names.Sort(StringComparer.Ordinal);
            return names;
        }

        public static bool IsRegistered(AexisLayerTypeKey typeKey)
        {
            return Registry.ContainsKey(typeKey);
        }

        public static AexisBaseLayer Create(AexisGraphModel.Layer layer)
        {
            if (layer == null)
                return new AexisUnknownLayer(default);

            if (AexisCustomLayerRegistry.TryCreate(layer, out var customLayer))
                return customLayer;

            // The existing 2D/depthwise loaders do not share the NCNN 1D
            // parameter/weight contract. Keep long aliases recognized, but fail
            // before dispatch rather than silently changing their semantics.
            if (string.Equals(layer.typeName, "ConvolutionDepthWise1D", StringComparison.Ordinal)
                )
                return new AexisConvolutionDepthWise1DLayer();
            if (string.Equals(layer.typeName, "Deconvolution1D", StringComparison.Ordinal))
                return new AexisDeconvolution1DLayer();
            if (string.Equals(layer.typeName, "DeconvolutionDepthWise1D", StringComparison.Ordinal))
                return new AexisDeconvolutionDepthWise1DLayer();

            var canonicalName = ResolveCanonicalLayerTypeName(layer.typeName);
            if (!string.IsNullOrEmpty(canonicalName)
                && Registry.TryGetValue(AexisLayerTypeKey.FromString(canonicalName), out var factory))
                return factory();

            if (Registry.TryGetValue(layer.type, out factory))
                return factory();

            return new AexisUnknownLayer(layer.type);
        }

        public static string ResolveCanonicalLayerTypeName(string typeName)
        {
            if (string.IsNullOrWhiteSpace(typeName))
                return string.Empty;
            return CanonicalAliases.TryGetValue(typeName, out var canonical) ? canonical : typeName;
        }

        public static bool IsRegistered(string typeName)
        {
            var canonical = ResolveCanonicalLayerTypeName(typeName);
            return AexisCustomLayerRegistry.IsRegistered(typeName)
                || (!string.IsNullOrEmpty(canonical) && Registry.ContainsKey(AexisLayerTypeKey.FromString(canonical)));
        }

        private sealed class AexisUnknownLayer : AexisBaseLayer
        {
            public AexisUnknownLayer(AexisLayerTypeKey typeKey)
                : base(typeKey, supportsBufferPath: false, supportsCommandBufferPath: false)
            {
            }
        }
    }

    public partial class AexisGraphSession
    {
        private static int _fixedInputTextureDumpSequence;

        private static void TryDumpFixedInputTexture(string blobName, RenderTexture texture, AexisTensorBuffer view)
        {
            if (texture == null || view == null || view.dims > 2)
                return;

            string dumpDir;
            try
            {
                dumpDir = Environment.GetEnvironmentVariable("AIIMAGE_NCNN_DUMP_FIXED_INPUT_TEXTURE_DIR");
            }
            catch
            {
                return;
            }

            if (string.IsNullOrWhiteSpace(dumpDir))
                return;

            try
            {
                Directory.CreateDirectory(dumpDir);
                var sequence = ++_fixedInputTextureDumpSequence;
                var safeName = SanitizeDumpFileName(blobName);
                var prefix = Path.Combine(dumpDir, sequence.ToString("0000") + "_" + safeName);
                File.WriteAllText(
                    prefix + "_contract.txt",
                    "blob=" + (blobName ?? string.Empty) + Environment.NewLine
                    + "dims=" + view.dims + " w=" + view.w + " h=" + view.h + " d=" + view.d + " c=" + view.c + Environment.NewLine
                    + "texture_width=" + texture.width + " height=" + texture.height + " depth=" + Mathf.Max(1, texture.volumeDepth) + Environment.NewLine
                    + "dimension=" + texture.dimension + " format=" + texture.format + Environment.NewLine);

                var width = Mathf.Max(1, texture.width);
                var height = Mathf.Max(1, texture.height);
                var logicalCount = Mathf.Max(1, view.w) * Mathf.Max(1, view.h) * Mathf.Max(1, view.d) * Mathf.Max(1, view.c);
                var readCount = Mathf.Min(logicalCount, width * height);
                var previousActive = RenderTexture.active;
                Texture2D readback = null;
                try
                {
                    readback = new Texture2D(width, height, TextureFormat.RGBAFloat, false, true);
                    RenderTexture.active = texture;
                    readback.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
                    readback.Apply(false, false);
                    var raw = readback.GetRawTextureData<float>();
                    var values = new float[readCount];
                    for (var i = 0; i < readCount; i++)
                        values[i] = raw[i * 4];

                    using var stream = new FileStream(prefix + "_f32.bin", FileMode.Create, FileAccess.Write, FileShare.Read);
                    using var writer = new BinaryWriter(stream);
                    for (var i = 0; i < values.Length; i++)
                        writer.Write(values[i]);
                }
                finally
                {
                    RenderTexture.active = previousActive;
                    if (readback != null)
                        UnityEngine.Object.DestroyImmediate(readback);
                }
            }
            catch
            {
            }
        }

        private static string SanitizeDumpFileName(string value)
        {
            if (string.IsNullOrWhiteSpace(value))
                return "blob";

            var chars = value.ToCharArray();
            for (var i = 0; i < chars.Length; i++)
            {
                var ch = chars[i];
                if (!(char.IsLetterOrDigit(ch) || ch == '_' || ch == '-' || ch == '.'))
                    chars[i] = '_';
            }
            return new string(chars);
        }

        private static bool HasStrideBlob(string[] names)
        {
            if (names == null)
                return false;

            for (var i = 0; i < names.Length; i++)
            {
                var name = names[i];
                if (!string.IsNullOrEmpty(name) && name.StartsWith("stride_", StringComparison.Ordinal))
                    return true;
            }

            return false;
        }

        private static int GetReplayConsumedBottomCount(AexisGraphModel.Layer layer)
        {
            if (layer?.bottomNames == null || layer.bottomNames.Length == 0)
                return 0;

            if (layer.type == AexisLayerTypes.AtenTo)
                return 1;

            return layer.bottomNames.Length;
        }

        private HashSet<int> ResolveReplayRequiredLayerIndices(
            ICollection<string> availableBlobNames,
            string stopAfterTopName,
            int startLayerIndex)
        {
            if (string.IsNullOrWhiteSpace(stopAfterTopName))
                return null;

            if (Model?.layers == null || Model.layers.Count == 0)
                return null;

            var available = new HashSet<string>(StringComparer.Ordinal);
            if (availableBlobNames != null)
            {
                foreach (var name in availableBlobNames)
                {
                    if (!string.IsNullOrWhiteSpace(name))
                        available.Add(name);
                }
            }

            if (available.Contains(stopAfterTopName))
                return new HashSet<int>();

            var producerByTop = new Dictionary<string, int>(StringComparer.Ordinal);
            for (var li = 0; li < Model.layers.Count; li++)
            {
                var topNames = Model.layers[li]?.topNames;
                if (topNames == null)
                    continue;

                for (var ti = 0; ti < topNames.Length; ti++)
                {
                    var topName = topNames[ti];
                    if (string.IsNullOrWhiteSpace(topName) || producerByTop.ContainsKey(topName))
                        continue;
                    producerByTop[topName] = li;
                }
            }

            var required = new HashSet<int>();
            var visiting = new HashSet<string>(StringComparer.Ordinal);

            void RequireBlob(string blobName)
            {
                if (string.IsNullOrWhiteSpace(blobName) || available.Contains(blobName))
                    return;

                if (!producerByTop.TryGetValue(blobName, out var producerIndex))
                {
                    throw new InvalidOperationException(
                        "Replay target depends on unresolved blob: " + blobName
                        + " | target=" + stopAfterTopName);
                }

                if (producerIndex < startLayerIndex)
                {
                    throw new InvalidOperationException(
                        "Replay target requires producer before replay start"
                        + " | target=" + stopAfterTopName
                        + " | blob=" + blobName
                        + " | producer_idx=" + producerIndex
                        + " | start_idx=" + startLayerIndex);
                }

                if (!visiting.Add(blobName))
                {
                    throw new InvalidOperationException(
                        "Replay dependency cycle detected"
                        + " | target=" + stopAfterTopName
                        + " | blob=" + blobName);
                }

                try
                {
                    if (!required.Add(producerIndex))
                        return;

                    var layer = Model.layers[producerIndex];
                    var consumedBottomCount = GetReplayConsumedBottomCount(layer);
                    if (layer?.bottomNames == null || consumedBottomCount <= 0)
                        return;

                    for (var bi = 0; bi < layer.bottomNames.Length && bi < consumedBottomCount; bi++)
                        RequireBlob(layer.bottomNames[bi]);
                }
                finally
                {
                    visiting.Remove(blobName);
                }
            }

            RequireBlob(stopAfterTopName);
            return required;
        }

        private Dictionary<string, int> BuildScopedBlobUseCount(
            int startLayerIndex,
            HashSet<int> replayRequiredLayerIndices,
            string stopAfterTopName)
        {
            if (Model?.layers == null || Model.layers.Count == 0)
            {
                return _blobUseCount != null
                    ? new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal)
                    : new Dictionary<string, int>(StringComparer.Ordinal);
            }

            var use = new Dictionary<string, int>(StringComparer.Ordinal);
            var stopReached = string.IsNullOrWhiteSpace(stopAfterTopName);
            startLayerIndex = Mathf.Clamp(startLayerIndex, 0, Model.layers.Count);

            for (var li = startLayerIndex; li < Model.layers.Count; li++)
            {
                if (replayRequiredLayerIndices != null && !replayRequiredLayerIndices.Contains(li))
                    continue;

                var layer = Model.layers[li];
                var consumedBottomCount = GetReplayConsumedBottomCount(layer);
                var bottomNames = layer?.bottomNames;
                if (bottomNames != null && consumedBottomCount > 0)
                {
                    for (var bi = 0; bi < bottomNames.Length && bi < consumedBottomCount; bi++)
                    {
                        var name = bottomNames[bi];
                        if (string.IsNullOrEmpty(name))
                            continue;

                        use.TryGetValue(name, out var count);
                        use[name] = count + 1;
                    }
                }

                if (!stopReached
                    && layer?.topNames != null
                    && Array.IndexOf(layer.topNames, stopAfterTopName) >= 0)
                {
                    stopReached = true;
                    break;
                }
            }

            if (!stopReached && !string.IsNullOrWhiteSpace(stopAfterTopName) && _blobUseCount != null)
                return new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);

            return use;
        }

        private sealed class LayerInferenceExecution
        {
            public InferResult Result;
        }

        internal InferResult InferWithMultiInputsByLayerRepros(
            Dictionary<string, RenderTexture> textureInputs,
            Dictionary<string, AexisTensorBuffer> bufferInputs,
            ICollection<string> pinnedNames = null,
            Dictionary<string, BufferShape> textureInputShapes = null,
            string stopAfterTopName = null,
            string startAtTopName = null)
        {
            var execution = new LayerInferenceExecution();
            using (var steps = EnumerateInferWithMultiInputsByLayerRepros(
                textureInputs,
                bufferInputs,
                pinnedNames,
                textureInputShapes,
                stopAfterTopName,
                startAtTopName,
                execution).GetEnumerator())
            {
                while (steps.MoveNext())
                {
                }
            }

            return execution.Result ?? throw new InvalidOperationException("Texture inference completed without producing a result.");
        }

        internal async Task<InferResult> InferWithMultiInputsByLayerReprosAsync(
            Dictionary<string, RenderTexture> textureInputs,
            Dictionary<string, AexisTensorBuffer> bufferInputs,
            ICollection<string> pinnedNames = null,
            Dictionary<string, BufferShape> textureInputShapes = null,
            string stopAfterTopName = null,
            string startAtTopName = null,
            CancellationToken cancellationToken = default,
            int yieldEveryLayers = 12,
            Action<InferenceProgress> progress = null)
        {
            yieldEveryLayers = Mathf.Max(1, yieldEveryLayers);
            var layersSinceYield = 0;
            var execution = new LayerInferenceExecution();
            using (var steps = EnumerateInferWithMultiInputsByLayerRepros(
                textureInputs,
                bufferInputs,
                pinnedNames,
                textureInputShapes,
                stopAfterTopName,
                startAtTopName,
                execution,
                progress).GetEnumerator())
            {
                while (true)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    if (!steps.MoveNext())
                        break;

                    layersSinceYield++;
                    if (!Application.isBatchMode && layersSinceYield >= yieldEveryLayers)
                    {
                        layersSinceYield = 0;
                        await AexisAsync.YieldFrame();
                        cancellationToken.ThrowIfCancellationRequested();
                    }
                }
            }

            return execution.Result ?? throw new InvalidOperationException("Texture inference completed without producing a result.");
        }

        private IEnumerable<bool> EnumerateInferWithMultiInputsByLayerRepros(
            Dictionary<string, RenderTexture> textureInputs,
            Dictionary<string, AexisTensorBuffer> bufferInputs,
            ICollection<string> pinnedNames = null,
            Dictionary<string, BufferShape> textureInputShapes = null,
            string stopAfterTopName = null,
            string startAtTopName = null,
            LayerInferenceExecution execution = null,
            Action<InferenceProgress> progress = null)
        {
            if (execution == null)
                throw new ArgumentNullException(nameof(execution));
            static string JoinNames(string[] names)
            {
                if (names == null || names.Length == 0)
                    return "-";
                return string.Join(",", names);
            }

            static string DescribeBlobState(
                string name,
                Dictionary<string, TensorRef> textureBlobs,
                Dictionary<string, BufferShape> textureShapes,
                Dictionary<string, ComputeBuffer> bufferBlobs,
                Dictionary<string, AexisTensorBuffer> bufferViews,
                Dictionary<string, IndexRef> indexBlobs)
            {
                if (string.IsNullOrWhiteSpace(name))
                    return "<empty>";

                if (textureBlobs != null
                    && textureBlobs.TryGetValue(name, out var tex)
                    && tex != null
                    && tex.texture != null)
                {
                    var shapeText = string.Empty;
                    if (textureShapes != null && textureShapes.TryGetValue(name, out var shape))
                    {
                        shapeText =
                            " logical=d" + shape.dims
                            + ":" + shape.w + "x" + shape.h + "x" + shape.d + "x" + shape.c;
                    }
                    else
                    {
                        try
                        {
                            var inferredShape = AexisGraphSession.GetTextureShape(textureShapes, tex, name);
                            shapeText =
                                " logical=d" + inferredShape.dims
                                + ":" + inferredShape.w + "x" + inferredShape.h + "x" + inferredShape.d + "x" + inferredShape.c;
                        }
                        catch
                        {
                        }
                    }

                    return name
                        + "=tex:"
                        + tex.width + "x" + tex.height + "x" + tex.packs + "p"
                        + shapeText;
                }

                if (bufferViews != null
                    && bufferViews.TryGetValue(name, out var view)
                    && view != null
                    && view.buffer != null)
                {
                    return name
                        + "=buf:d" + view.dims
                        + ":" + view.w + "x" + view.h + "x" + view.d + "x" + view.c
                        + " count=" + view.buffer.count;
                }

                if (bufferBlobs != null
                    && bufferBlobs.TryGetValue(name, out var buffer)
                    && buffer != null)
                {
                    return name + "=buf:count=" + buffer.count;
                }

                if (indexBlobs != null
                    && indexBlobs.TryGetValue(name, out var index)
                    && index != null)
                {
                    if (index.view != null && index.view.buffer != null)
                    {
                        return name
                            + "=idxbuf:d" + index.view.dims
                            + ":" + index.view.w + "x" + index.view.h + "x" + index.view.d + "x" + index.view.c
                            + " count=" + index.view.buffer.count;
                    }

                    if (index.texture != null)
                        return name + "=idxtex:" + index.width + "x" + index.height + "x" + index.packs + "p";
                }

                return name + "=missing";
            }

            static string DescribeBlobStates(
                string[] names,
                Dictionary<string, TensorRef> textureBlobs,
                Dictionary<string, BufferShape> textureShapes,
                Dictionary<string, ComputeBuffer> bufferBlobs,
                Dictionary<string, AexisTensorBuffer> bufferViews,
                Dictionary<string, IndexRef> indexBlobs)
            {
                if (names == null || names.Length == 0)
                    return "-";

                var parts = new string[names.Length];
                for (var i = 0; i < names.Length; i++)
                {
                    parts[i] = DescribeBlobState(names[i], textureBlobs, textureShapes, bufferBlobs, bufferViews, indexBlobs);
                }

                return string.Join("; ", parts);
            }

            var startLayerIndex = 0;
            HashSet<int> replayRequiredLayerIndices = null;
            if (!string.IsNullOrWhiteSpace(startAtTopName))
            {
                var foundStart = false;
                for (var li = 0; li < Model.layers.Count; li++)
                {
                    var topNames = Model.layers[li]?.topNames;
                    if (topNames == null || topNames.Length == 0)
                        continue;
                    if (Array.IndexOf(topNames, startAtTopName) < 0)
                        continue;

                    startLayerIndex = li + 1;
                    foundStart = true;
                    break;
                }

                if (!foundStart)
                    throw new InvalidOperationException("start top not found in model: " + startAtTopName);

                var availableReplayInputs = new HashSet<string>(StringComparer.Ordinal);
                if (textureInputs != null)
                {
                    foreach (var kv in textureInputs)
                    {
                        if (!string.IsNullOrWhiteSpace(kv.Key))
                            availableReplayInputs.Add(kv.Key);
                    }
                }

                if (bufferInputs != null)
                {
                    foreach (var kv in bufferInputs)
                    {
                        if (!string.IsNullOrWhiteSpace(kv.Key))
                            availableReplayInputs.Add(kv.Key);
                    }
                }

                replayRequiredLayerIndices = ResolveReplayRequiredLayerIndices(
                    availableReplayInputs,
                    stopAfterTopName,
                    startLayerIndex);

                if (DebugLog != null)
                {
                    var requiredCount = replayRequiredLayerIndices == null ? 0 : replayRequiredLayerIndices.Count;
                    DebugLog(
                        "[ReplaySlice]"
                        + " | start=" + startAtTopName
                        + " | stop=" + (stopAfterTopName ?? string.Empty)
                        + " | start_idx=" + startLayerIndex
                        + " | required_layers=" + requiredCount);
                }
            }

            var layerWork = new float[Model.layers.Count];
            var totalWork = 0f;
            for (var li = startLayerIndex; li < Model.layers.Count; li++)
            {
                if (replayRequiredLayerIndices != null && !replayRequiredLayerIndices.Contains(li))
                    continue;

                var planLayer = Model.layers[li];
                var planRepro = LayerRepros != null && li < LayerRepros.Count ? LayerRepros[li] : null;
                var work = planRepro != null ? planRepro.GetIncrementalWorkEstimate(this, planLayer) : 1;
                layerWork[li] = Mathf.Max(1, work);
                totalWork += layerWork[li];
                if (!string.IsNullOrWhiteSpace(stopAfterTopName)
                    && planLayer?.topNames != null
                    && Array.IndexOf(planLayer.topNames, stopAfterTopName) >= 0)
                {
                    break;
                }
            }
            totalWork = Mathf.Max(1f, totalWork);

            var remaining = BuildScopedBlobUseCount(startLayerIndex, replayRequiredLayerIndices, stopAfterTopName);
            var textureBlobs = new Dictionary<string, TensorRef>(StringComparer.Ordinal);
            var textureShapes = new Dictionary<string, BufferShape>(StringComparer.Ordinal);
            var bufferBlobs = new Dictionary<string, ComputeBuffer>(StringComparer.Ordinal);
            var bufferRefs = new Dictionary<string, BufferRef>(StringComparer.Ordinal);
            var bufferViews = new Dictionary<string, AexisTensorBuffer>(StringComparer.Ordinal);
            var indexBlobs = new Dictionary<string, IndexRef>(StringComparer.Ordinal);
            var tempOwned = new List<IDisposable>();
            var fixedBufferInputBlobs = new HashSet<string>(StringComparer.Ordinal);

            RegisterTextureInputs(textureInputs, textureInputShapes, textureBlobs, textureShapes);

            if (bufferInputs != null)
            {
                foreach (var kv in bufferInputs)
                {
                    if (kv.Value == null || kv.Value.buffer == null)
                        throw new ArgumentNullException("bufferInputs[\"" + kv.Key + "\"]");

                    if (!IsDebugOracleExecution)
                    {
                        if (textureBlobs.ContainsKey(kv.Key))
                            throw new InvalidOperationException("A production input cannot bind both texture and buffer storage"
                                + " | blob=" + kv.Key
                                + " | rejected_fallback=buffer-activation");

                        // A buffer can be an immutable upload source at the graph edge,
                        // but no layer receives it. All model activations, including
                        // Embed token ids, are texture-backed before execution starts.
                        var logicalShape = new BufferShape(kv.Value.dims, kv.Value.w, kv.Value.h, kv.Value.d, kv.Value.c);
                        var texture = UploadFixedInputTexture(kv.Key, kv.Value.buffer, kv.Value);
                        var storageShape = ResolveExternalTextureInputStorageShape(
                            logicalShape,
                            texture.width,
                            texture.height,
                            texture.dimension,
                            Mathf.Max(1, texture.volumeDepth),
                            texture.format);
                        var useCount = remaining.TryGetValue(kv.Key, out var count) ? Mathf.Max(1, count) : 1;
                        textureBlobs[kv.Key] = CreateTextureRef(
                            texture,
                            logicalShape,
                            storageShape,
                            owned: true,
                            refs: useCount,
                            blobName: kv.Key);
                        textureShapes[kv.Key] = logicalShape;
                        fixedBufferInputBlobs.Add(kv.Key);
                        continue;
                    }

                    // DebugOracle is the only mode that retains legacy Buffer inputs.
                    bufferBlobs[kv.Key] = kv.Value.buffer;
                    bufferRefs[kv.Key] = new BufferRef
                    {
                        buffer = kv.Value.buffer,
                        refs = 1,
                        owned = false
                    };
                    bufferViews[kv.Key] = kv.Value;
                }
            }

            // Immediate RenderTexture execution must admit the same strict Pack4
            // CommandBuffer plan as asynchronous execution. Replay slices start from a
            // retained intermediate descriptor and are planned by their caller.
            if (!IsDebugOracleExecution && string.IsNullOrWhiteSpace(startAtTopName))
            {
                try
                {
                    EnsureImmediateTextureExecutionPlan(textureBlobs, textureShapes, fixedBufferInputBlobs, stopAfterTopName);
                }
                catch
                {
                    // A strict plan may reject after boundary upload. Return only
                    // pool-owned RTs; do not touch caller-owned texture inputs.
                    var released = new HashSet<TensorRef>();
                    foreach (var tensor in textureBlobs.Values)
                    {
                        if (tensor != null && tensor.owned && tensor.texture != null && released.Add(tensor))
                            ReturnTempArray(tensor.texture);
                    }
                    throw;
                }
            }

            var context = new AexisLayerBufferContext
            {
                textureBlobs = textureBlobs,
                textureShapes = textureShapes,
                bufferBlobs = bufferBlobs,
                bufferRefs = bufferRefs,
                bufferViews = bufferViews,
                indexBlobs = indexBlobs,
                remaining = remaining,
                pinnedNames = pinnedNames,
                tempOwned = tempOwned
            };
            SetCurrentBufferExecutionContext(context);

            bool TryLogFirstNonFiniteLayerOutput(int layerIndex, AexisGraphModel.Layer layer)
            {
                if (!DebugBreakOnFirstNonFiniteLayerOutput || DebugLog == null || layer?.topNames == null)
                    return false;

                for (var i = 0; i < layer.topNames.Length; i++)
                {
                    var topName = layer.topNames[i];
                    if (string.IsNullOrWhiteSpace(topName))
                        continue;

                    if (bufferViews.TryGetValue(topName, out var view) && view != null && view.buffer != null)
                    {
                        var logicalCount = Mathf.Max(1, view.w) * Mathf.Max(1, view.h) * Mathf.Max(1, view.d) * Mathf.Max(1, view.c);
                        if (BufferHasAnyNonFinite(view.buffer, logicalCount, out var finiteCount, out var nanCount, out var infCount))
                        {
                            DebugLog("[LayerNonFinite] idx=" + layerIndex
                                + " | name=" + (layer.name ?? string.Empty)
                                + " | type=" + (layer.typeName ?? string.Empty)
                                + " | top=" + topName
                                + " | shape=d" + view.dims + ":" + view.w + "x" + view.h + "x" + view.d + "x" + view.c
                                + " | finite=" + finiteCount
                                + " | nan=" + nanCount
                                + " | inf=" + infCount
                                + " | path=" + DescribeLayerOutputPath(layer, textureBlobs, textureShapes, bufferBlobs, bufferViews, indexBlobs));
                            return true;
                        }
                    }
                }

                return false;
            }

            void TryDebugReadbackLayerOutputs(AexisGraphModel.Layer layer)
            {
                if (DebugLayerTextureReadback == null
                    || DebugLayerReadbackBlobs == null
                    || DebugLayerReadbackBlobs.Count == 0
                    || layer?.topNames == null)
                {
                    return;
                }

                for (var topIndex = 0; topIndex < layer.topNames.Length; topIndex++)
                {
                    var topName = layer.topNames[topIndex];
                    if (string.IsNullOrWhiteSpace(topName) || !DebugLayerReadbackBlobs.Contains(topName))
                        continue;
                    if (!TryGetExistingTextureContract(textureBlobs, textureShapes, topName, out var textureRef, out var contract))
                    {
                        if (string.Equals(layer.typeName, "Input", StringComparison.Ordinal))
                        {
                            DebugLog?.Invoke("[DebugCheckpointSkip] optional input is not bound | layer=" + (layer.name ?? string.Empty) + " | top=" + topName);
                            continue;
                        }
                        throw new InvalidOperationException("Debug texture checkpoint is not texture-backed: " + topName);
                    }
                    var values = InferResult.ReadExistingTextureData(
                        textureRef.texture,
                        contract.LogicalShape,
                        contract.StorageShape,
                        contract.LayoutKind);
                    DebugLayerTextureReadback(layer.name ?? string.Empty, topName, values);
                }
            }

            bool MoveNextIncrementalLayer(
                IEnumerator<bool> layerSteps,
                int layerIndex,
                AexisGraphModel.Layer layer)
            {
                if (layerSteps == null)
                    throw new InvalidOperationException("Layer incremental execution returned no steps: " + (layer?.name ?? string.Empty));

                try
                {
                    return layerSteps.MoveNext();
                }
                catch (Exception e)
                {
                    throw new InvalidOperationException(
                        "Layer execution failed"
                        + " | idx=" + layerIndex + "/" + Model.layers.Count
                        + " | name=" + (layer?.name ?? string.Empty)
                        + " | type=" + (layer?.typeName ?? string.Empty)
                        + " | bottoms=" + DescribeBlobStates(layer?.bottomNames, textureBlobs, textureShapes, bufferBlobs, bufferViews, indexBlobs)
                        + " | tops=" + JoinNames(layer?.topNames)
                        + " | inner=" + e.Message,
                        e);
                }
            }

            BeginInferenceTempResourceTracking();
            try
            {
                var runtimeProfile = BeginLayerRuntimeProfile("buffer");
                var completedWork = 0f;
                for (var li = startLayerIndex; li < Model.layers.Count; li++)
                {
                    if (replayRequiredLayerIndices != null && !replayRequiredLayerIndices.Contains(li))
                        continue;

                    var layer = Model.layers[li];
                    var currentLayerWork = Mathf.Max(1f, layerWork[li]);
                    context.BeginInferenceProgress(
                        progress,
                        li,
                        Model.layers.Count,
                        layer,
                        completedWork,
                        currentLayerWork,
                        totalWork);
                    var layerOutputPath = string.Empty;
                    var emitHeartbeat = DebugLog != null
                        && (DebugLogAllLayerHeartbeats
                            || li < 8
                            || ((li + 1) % 32) == 0
                            || HasStrideBlob(layer?.topNames)
                            || HasStrideBlob(layer?.bottomNames));
                    if (emitHeartbeat)
                    {
                        DebugLog("[LayerHeartbeat] idx=" + li + "/" + Model.layers.Count
                            + " | name=" + (layer?.name ?? string.Empty)
                            + " | type=" + (layer?.typeName ?? string.Empty)
                            + " | bottoms=" + JoinNames(layer?.bottomNames)
                            + " | tops=" + JoinNames(layer?.topNames));
                    }

                    if (AreAllLayerTopsAlreadyAvailable(layer, textureBlobs, bufferBlobs, indexBlobs))
                    {
                        if (emitHeartbeat)
                            DebugLog("[LayerOutput] idx=" + li + " | name=" + (layer?.name ?? string.Empty) + " | path=skip-already-available");
                        Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                        context.ReportInferenceProgress(1f);
                        completedWork += currentLayerWork;
                        yield return true;
                        continue;
                    }

                    var layerRepro = LayerRepros[li];
                    if (layerRepro == null)
                        throw new InvalidOperationException("layer repro missing: " + layer?.name);
                    if (runtimeProfile == null)
                    {
                        SetCurrentExecutingLayer(layer);
                        ResetInt8ActivationQuantization();
                        try
                        {
                            using (var layerSteps = layerRepro.ExecuteBufferIncremental(this, layer, context).GetEnumerator())
                            {
                                while (MoveNextIncrementalLayer(layerSteps, li, layer))
                                    yield return true;
                            }
                        }
                        finally
                        {
                            ResetInt8ActivationQuantization();
                            ClearCurrentExecutingLayer();
                        }
                        TryDebugReadbackLayerOutputs(layer);
                        layerOutputPath = DescribeLayerOutputPath(layer, textureBlobs, textureShapes, bufferBlobs, bufferViews, indexBlobs);
                        if (DebugLog != null && (DebugLogAllLayerOutputs || HasStrideBlob(layer?.topNames)))
                        {
                            DebugLog("[LayerOutput] idx=" + li
                                + " | name=" + (layer?.name ?? string.Empty)
                                + " | path=" + layerOutputPath);
                        }
                        TryLogFirstNonFiniteLayerOutput(li, layer);
                        context.ReportInferenceProgress(1f);
                        completedWork += currentLayerWork;
                        if (!string.IsNullOrWhiteSpace(stopAfterTopName)
                            && layer?.topNames != null
                            && Array.IndexOf(layer.topNames, stopAfterTopName) >= 0
                            && AreAllLayerTopsAlreadyAvailable(layer, textureBlobs, bufferBlobs, indexBlobs))
                        {
                            break;
                        }
                        yield return true;
                        continue;
                    }

                    var layerSw = Stopwatch.StartNew();
                    SetCurrentExecutingLayer(layer);
                    ResetInt8ActivationQuantization();
                    try
                    {
                        using (var layerSteps = layerRepro.ExecuteBufferIncremental(this, layer, context).GetEnumerator())
                        {
                            while (MoveNextIncrementalLayer(layerSteps, li, layer))
                                yield return true;
                        }
                    }
                    finally
                    {
                        ResetInt8ActivationQuantization();
                        ClearCurrentExecutingLayer();
                    }
                    TryDebugReadbackLayerOutputs(layer);
                    if (LayerRuntimeProfileSyncGpu)
                        Ops.DebugSyncGpu();
                    layerSw.Stop();
                    layerOutputPath = DescribeLayerOutputPath(layer, textureBlobs, textureShapes, bufferBlobs, bufferViews, indexBlobs);
                    RecordLayerRuntime(
                        runtimeProfile,
                        li,
                        layer,
                        layerOutputPath,
                        layerSw.ElapsedTicks);

                    if (DebugLog != null && (DebugLogAllLayerOutputs || HasStrideBlob(layer?.topNames)))
                    {
                        DebugLog("[LayerOutput] idx=" + li
                            + " | name=" + (layer?.name ?? string.Empty)
                            + " | path=" + layerOutputPath);
                    }
                    if (!string.IsNullOrWhiteSpace(TimingSplitSyncAfterTopName)
                        && layer?.topNames != null
                        && Array.IndexOf(layer.topNames, TimingSplitSyncAfterTopName) >= 0)
                    {
                        var splitSyncSw = Stopwatch.StartNew();
                        Ops.DebugSyncGpu();
                        splitSyncSw.Stop();
                        NotifyTimingSplitSyncPoint(
                            TimingSplitSyncAfterTopName,
                            splitSyncSw.ElapsedTicks * 1000d / Stopwatch.Frequency);
                    }
                    TryLogFirstNonFiniteLayerOutput(li, layer);
                    context.ReportInferenceProgress(1f);
                    completedWork += currentLayerWork;
                    if (!string.IsNullOrWhiteSpace(stopAfterTopName)
                        && layer?.topNames != null
                        && Array.IndexOf(layer.topNames, stopAfterTopName) >= 0
                        && AreAllLayerTopsAlreadyAvailable(layer, textureBlobs, bufferBlobs, indexBlobs))
                    {
                        break;
                    }
                    yield return true;
                }

                FinishLayerRuntimeProfile(runtimeProfile);
                var disallowTextureToBufferFallback =
                    DisallowBufferAccess
                    || DisallowBufferOutputs
                    || DisallowBufferToTextureMaterialization
                    || DisallowInferenceTempComputeBuffers;
                execution.Result = new InferResult(
                    textureBlobs,
                    textureShapes,
                    bufferBlobs,
                    bufferRefs,
                    bufferViews,
                    tempOwned,
                    this,
                    disallowTextureToBufferFallback);
                yield break;
            }
            finally
            {
                ClearCurrentBufferExecutionContext();
                EndInferenceTempResourceTracking();
            }
        }

        internal ComputeTexture ForwardPack4ByLayerRepros(
            CommandBuffer cmd,
            ComputeTexture inputPack4,
            int inputPacks,
            string inputBlobName = "data",
            ICollection<string> pinnedNames = null)
        {
            if (inputPack4 == null)
                throw new ArgumentNullException(nameof(inputPack4));
            var inputShape = new BufferShape(3, inputPack4.width, inputPack4.height, 1, ResolveInputLogicalChannels(inputBlobName, inputPacks * 4));
            EnsureCommandBufferTextureExecutionPlan(
                new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { [inputBlobName] = inputPack4 },
                new Dictionary<string, BufferShape>(StringComparer.Ordinal) { [inputBlobName] = inputShape });
            BeginInferenceTempResourceTracking();
            Dictionary<string, CmdTensorRef> blobs = null;
            var returned = false;
            try
            {
                var remaining = new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);
                var shapes = new Dictionary<string, BufferShape>(StringComparer.Ordinal)
                {
                    [inputBlobName] = inputShape
                };
                blobs = new Dictionary<string, CmdTensorRef>(StringComparer.Ordinal)
                {
                    [inputBlobName] = new CmdTensorRef
                    {
                        texture = inputPack4,
                        width = inputPack4.width,
                        height = inputPack4.height,
                        packs = inputPacks,
                        refs = 1,
                        owned = false
                    }
                };

                var context = new AexisLayerCommandBufferContext
                {
                    commandBuffer = cmd,
                    blobs = blobs,
                    shapes = shapes,
                    remaining = remaining,
                    pinnedNames = pinnedNames
                };

                var runtimeProfile = BeginLayerRuntimeProfile("cmd");
                for (var li = 0; li < Model.layers.Count; li++)
                {
                    var layer = Model.layers[li];
                    var layerRepro = LayerRepros[li];
                    if (layerRepro == null)
                        throw new InvalidOperationException("layer repro missing: " + layer?.name);
                    if (runtimeProfile == null)
                    {
                        SetCurrentExecutingLayer(layer);
                        ResetInt8ActivationQuantization();
                        try
                        {
                            layerRepro.ExecuteCommandBuffer(this, layer, context);
                        }
                        finally
                        {
                            ResetInt8ActivationQuantization();
                            ClearCurrentExecutingLayer();
                        }
                        if (DebugLog != null && (DebugLogAllLayerOutputs || HasStrideBlob(layer?.topNames)))
                        {
                            DebugLog("[LayerOutput] idx=" + li
                                + " | name=" + (layer?.name ?? string.Empty)
                                + " | path=" + DescribeCmdLayerOutputPath(layer, blobs, shapes));
                        }
                        continue;
                    }

                    var layerSw = Stopwatch.StartNew();
                    SetCurrentExecutingLayer(layer);
                    ResetInt8ActivationQuantization();
                    try
                    {
                        layerRepro.ExecuteCommandBuffer(this, layer, context);
                    }
                    finally
                    {
                        ResetInt8ActivationQuantization();
                        ClearCurrentExecutingLayer();
                    }
                    if (DebugLog != null && (DebugLogAllLayerOutputs || HasStrideBlob(layer?.topNames)))
                    {
                        DebugLog("[LayerOutput] idx=" + li
                            + " | name=" + (layer?.name ?? string.Empty)
                            + " | path=" + DescribeCmdLayerOutputPath(layer, blobs, shapes));
                    }
                    layerSw.Stop();
                    RecordLayerRuntime(runtimeProfile, li, layer, "cmd", layerSw.ElapsedTicks);
                }

                FinishLayerRuntimeProfile(runtimeProfile);
                var outBlobName = ResolveDefaultOutputBlobName();
                var outRef = GetCmdTensor(blobs, outBlobName);
                var keep = outRef.texture;
                DetachReturnedCmdTextureOwnership(blobs, keep);

                var visited = new HashSet<CmdTensorRef>();
                foreach (var kv in blobs)
                {
                    var tr = kv.Value;
                    if (tr == null || !visited.Add(tr))
                        continue;
                    if (tr.owned && tr.texture != null)
                        ReturnTempArray(cmd, tr.texture);
                }
                returned = true;
                return keep;
            }
            finally
            {
                if (!returned && blobs != null)
                    ReleaseAllCmdTemporaryTensors(cmd, blobs);
                EndInferenceTempResourceTracking();
            }
        }

        internal ComputeTexture ForwardPack4ByLayerRepros(
            CommandBuffer cmd,
            Dictionary<string, ComputeTexture> textureInputs,
            Dictionary<string, BufferShape> textureInputShapes,
            out BufferShape outputLogicalShape,
            ICollection<string> pinnedNames = null,
            string outputBlobName = null,
            string stopAfterTopName = null,
            ICollection<string> retainedOutputNames = null,
            Dictionary<string, ComputeTexture> retainedOutputs = null,
            Dictionary<string, BufferShape> retainedOutputShapes = null,
            Dictionary<string, AexisTensorBuffer> fixedBufferInputs = null)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if (textureInputs == null)
                throw new ArgumentNullException(nameof(textureInputs));
            if (textureInputs.Count == 0 && (fixedBufferInputs == null || fixedBufferInputs.Count == 0)
                && Model != null && Model.layers != null
                && Model.layers.Any(layer => layer?.bottomNames != null && layer.bottomNames.Any(name => !string.IsNullOrWhiteSpace(name))))
            {
                throw new ArgumentException(
                    "A zero-input CommandBuffer invocation is valid only for a statically closed texture plan; this model declares external activation bottoms.",
                    nameof(textureInputs));
            }

            var resolvedInputs = textureInputs == null
                ? new Dictionary<string, ComputeTexture>(StringComparer.Ordinal)
                : new Dictionary<string, ComputeTexture>(textureInputs, StringComparer.Ordinal);
            var resolvedShapes = textureInputShapes == null
                ? new Dictionary<string, BufferShape>(StringComparer.Ordinal)
                : new Dictionary<string, BufferShape>(textureInputShapes, StringComparer.Ordinal);
            var fixedUploadNames = new HashSet<string>(StringComparer.Ordinal);
            var fixedUploads = new List<ComputeTexture>();
            try
            {
                if (fixedBufferInputs != null)
                {
                    foreach (var kv in fixedBufferInputs)
                    {
                        if (kv.Value == null || kv.Value.buffer == null)
                            throw new ArgumentNullException("fixedBufferInputs[\"" + kv.Key + "\"]");
                        if (resolvedInputs.ContainsKey(kv.Key))
                            throw new InvalidOperationException("A CommandBuffer input cannot bind both texture and buffer storage"
                                + " | blob=" + kv.Key
                                + " | rejected_fallback=buffer-activation");

                        var upload = UploadFixedInputCmdTexture(cmd, kv.Key, kv.Value.buffer, kv.Value);
                        resolvedInputs.Add(kv.Key, upload);
                        resolvedShapes[kv.Key] = new BufferShape(kv.Value.dims, kv.Value.w, kv.Value.h, kv.Value.d, kv.Value.c);
                        fixedUploadNames.Add(kv.Key);
                        fixedUploads.Add(upload);
                    }
                }

                EnsureCommandBufferTextureExecutionPlan(
                    resolvedInputs,
                    resolvedShapes,
                    fixedUploadNames,
                    stopAfterTopName);
            }
            catch
            {
                foreach (var upload in fixedUploads)
                    ReturnTempArray(cmd, upload);
                throw;
            }
            BeginCommandBufferRtArena(cmd);
            BeginInferenceTempResourceTracking();
            Dictionary<string, CmdTensorRef> blobs = null;
            var returned = false;
            try
            {
                var remaining = new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);
                var shapes = new Dictionary<string, BufferShape>(StringComparer.Ordinal);
                blobs = new Dictionary<string, CmdTensorRef>(StringComparer.Ordinal);
                RegisterCmdTextureInputs(resolvedInputs, resolvedShapes, blobs, shapes);
                foreach (var fixedUploadName in fixedUploadNames)
                {
                    if (blobs.TryGetValue(fixedUploadName, out var fixedUpload) && fixedUpload != null)
                        fixedUpload.owned = true;
                }

                var context = new AexisLayerCommandBufferContext
                {
                    commandBuffer = cmd,
                    blobs = blobs,
                    shapes = shapes,
                    remaining = remaining,
                    pinnedNames = pinnedNames
                };

                var runtimeProfile = BeginLayerRuntimeProfile("cmd");
                for (var li = 0; li < Model.layers.Count; li++)
                {
                    var layer = Model.layers[li];
                    var layerRepro = LayerRepros[li];
                    if (layerRepro == null)
                        throw new InvalidOperationException("layer repro missing: " + layer?.name);
                    if (runtimeProfile == null)
                    {
                        SetCurrentExecutingLayer(layer);
                        ResetInt8ActivationQuantization();
                        try
                        {
                            layerRepro.ExecuteCommandBuffer(this, layer, context);
                        }
                        finally
                        {
                            ResetInt8ActivationQuantization();
                            ClearCurrentExecutingLayer();
                        }
                        if (DebugLog != null && (DebugLogAllLayerOutputs || HasStrideBlob(layer?.topNames)))
                        {
                            DebugLog("[LayerOutput] idx=" + li
                                + " | name=" + (layer?.name ?? string.Empty)
                                + " | path=" + DescribeCmdLayerOutputPath(layer, blobs, shapes));
                        }
                    }
                    else
                    {
                        var layerSw = Stopwatch.StartNew();
                        SetCurrentExecutingLayer(layer);
                        ResetInt8ActivationQuantization();
                        try
                        {
                            layerRepro.ExecuteCommandBuffer(this, layer, context);
                        }
                        finally
                        {
                            ResetInt8ActivationQuantization();
                            ClearCurrentExecutingLayer();
                        }
                        if (DebugLog != null && (DebugLogAllLayerOutputs || HasStrideBlob(layer?.topNames)))
                        {
                            DebugLog("[LayerOutput] idx=" + li
                                + " | name=" + (layer?.name ?? string.Empty)
                                + " | path=" + DescribeCmdLayerOutputPath(layer, blobs, shapes));
                        }
                        layerSw.Stop();
                        RecordLayerRuntime(runtimeProfile, li, layer, "cmd", layerSw.ElapsedTicks);
                    }

                    if (!string.IsNullOrWhiteSpace(stopAfterTopName)
                        && layer?.topNames != null
                        && Array.IndexOf(layer.topNames, stopAfterTopName) >= 0
                        && TryGetCmdShape(shapes, blobs, stopAfterTopName, out _))
                    {
                        break;
                    }
                }

                FinishLayerRuntimeProfile(runtimeProfile);
                CompleteCommandBufferRtArena();
                var outBlobName = !string.IsNullOrWhiteSpace(stopAfterTopName)
                    ? stopAfterTopName
                    : (!string.IsNullOrWhiteSpace(outputBlobName) ? outputBlobName : ResolveDefaultOutputBlobName());
                var outRef = GetCmdTensor(blobs, outBlobName);
                outputLogicalShape = GetCmdShape(shapes, blobs, outBlobName);
                var keep = outRef.texture;
                if (retainedOutputNames == null)
                {
                    DetachReturnedCmdTextureOwnership(blobs, keep);
                }
                else
                {
                    if (retainedOutputs == null || retainedOutputShapes == null)
                        throw new ArgumentException("Retained CommandBuffer outputs require destination dictionaries.", nameof(retainedOutputs));
                    foreach (var retainedName in retainedOutputNames)
                    {
                        if (string.IsNullOrWhiteSpace(retainedName))
                            throw new ArgumentException("Retained CommandBuffer output names cannot be empty.", nameof(retainedOutputNames));
                        var retained = GetCmdTensor(blobs, retainedName);
                        retainedOutputs[retainedName] = retained.texture;
                        retainedOutputShapes[retainedName] = GetCmdShape(shapes, blobs, retainedName);
                        DetachReturnedCmdTextureOwnership(blobs, retained.texture);
                    }
                }

                var visited = new HashSet<CmdTensorRef>();
                foreach (var kv in blobs)
                {
                    var tr = kv.Value;
                    if (tr == null || !visited.Add(tr))
                        continue;
                    if (tr.owned && tr.texture != null)
                        ReturnTempArray(cmd, tr.texture);
                }
                returned = true;
                return keep;
            }
            finally
            {
                if (!returned && blobs != null)
                    ReleaseAllCmdTemporaryTensors(cmd, blobs);
                EndInferenceTempResourceTracking();
                EndCommandBufferRtArena();
            }
        }
    }
}
