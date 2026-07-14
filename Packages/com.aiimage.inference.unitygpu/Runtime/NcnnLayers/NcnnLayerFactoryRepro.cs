using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public static class NcnnLayerFactoryRepro
    {
        private static readonly Dictionary<NcnnLayerTypeKey, Func<NcnnBaseLayerRepro>> Registry = new Dictionary<NcnnLayerTypeKey, Func<NcnnBaseLayerRepro>>
        {
            { NcnnLayerTypes.Input, () => new NcnnInputLayerRepro() },
            { NcnnLayerTypes.PnnxExpression, () => new NcnnPnnxExpressionLayerRepro() },
            { NcnnLayerTypes.AtenTo, () => new NcnnAtenToLayerRepro() },
            { NcnnLayerTypes.AbsVal, () => new NcnnUnaryOpAliasLayerRepro(NcnnLayerTypes.AbsVal, 0) },
            { NcnnLayerTypes.Split, () => new NcnnSplitLayerRepro() },
            { NcnnLayerTypes.Concat, () => new NcnnConcatLayerRepro() },
            { NcnnLayerTypes.TanH, () => new NcnnUnaryOpAliasLayerRepro(NcnnLayerTypes.TanH, 16) },
            { NcnnLayerTypes.Reshape, () => new NcnnReshapeLayerRepro() },
            { NcnnLayerTypes.ShuffleChannel, () => new NcnnShuffleChannelLayerRepro() },
            { NcnnLayerTypes.Permute, () => new NcnnPermuteLayerRepro() },
            { NcnnLayerTypes.Slice, () => new NcnnSliceLayerRepro() },
            { NcnnLayerTypes.ExpandDims, () => new NcnnExpandDimsLayerRepro() },
            { NcnnLayerTypes.Squeeze, () => new NcnnSqueezeLayerRepro() },
            { NcnnLayerTypes.Crop, () => new NcnnCropLayerRepro() },
            { NcnnLayerTypes.Convolution, () => new NcnnConvolutionLayerRepro() },
            { NcnnLayerTypes.Convolution3D, () => new NcnnConvolution3DLayerRepro() },
            { NcnnLayerTypes.Convolution1D, () => new NcnnConvolution1DLayerRepro() },
            { NcnnLayerTypes.ConvolutionDepthWise, () => new NcnnConvolutionDepthWiseLayerRepro() },
            { NcnnLayerTypes.Deconvolution, () => new NcnnDeconvolutionLayerRepro() },
            { NcnnLayerTypes.Deconvolution3D, () => new NcnnDeconvolution3DLayerRepro() },
            { NcnnLayerTypes.DeconvolutionDepthWise, () => new NcnnDeconvolutionDepthWiseLayerRepro() },
            { NcnnLayerTypes.Interp, () => new NcnnInterpLayerRepro() },
            { NcnnLayerTypes.Dropout, () => new NcnnDropoutLayerRepro() },
            { NcnnLayerTypes.Eltwise, () => new NcnnEltwiseLayerRepro() },
            { NcnnLayerTypes.ELU, () => new NcnnPointwiseFormulaLayerRepro(NcnnLayerTypes.ELU) },
            { NcnnLayerTypes.Erf, () => new NcnnPointwiseFormulaLayerRepro(NcnnLayerTypes.Erf) },
            { NcnnLayerTypes.Flatten, () => new NcnnFlattenLayerRepro() },
            { NcnnLayerTypes.BinaryOp, () => new NcnnBinaryOpLayerRepro() },
            { NcnnLayerTypes.UnaryOp, () => new NcnnUnaryOpLayerRepro() },
            { NcnnLayerTypes.HardSigmoid, () => new NcnnPointwiseFormulaLayerRepro(NcnnLayerTypes.HardSigmoid) },
            { NcnnLayerTypes.HardSwish, () => new NcnnPointwiseFormulaLayerRepro(NcnnLayerTypes.HardSwish) },
            { NcnnLayerTypes.InstanceNorm, () => new NcnnInstanceNormLayerRepro() },
            { NcnnLayerTypes.LRN, () => new NcnnLRNLayerRepro() },
            { NcnnLayerTypes.Mish, () => new NcnnPointwiseFormulaLayerRepro(NcnnLayerTypes.Mish) },
            { NcnnLayerTypes.Swish, () => new NcnnSwishLayerRepro() },
            { NcnnLayerTypes.Noop, () => new NcnnNoopLayerRepro() },
            { NcnnLayerTypes.Normalize, () => new NcnnNormalizeLayerRepro() },
            { NcnnLayerTypes.Packing, () => new NcnnPackingLayerRepro() },
            { NcnnLayerTypes.PixelShuffle, () => new NcnnPixelShuffleLayerRepro() },
            { NcnnLayerTypes.PReLU, () => new NcnnPReLULayerRepro() },
            { NcnnLayerTypes.PriorBox, () => new NcnnPriorBoxLayerRepro() },
            { NcnnLayerTypes.Quantize, () => new NcnnQuantizeLayerRepro() },
            { NcnnLayerTypes.Dequantize, () => new NcnnDequantizeLayerRepro() },
            { NcnnLayerTypes.Requantize, () => new NcnnRequantizeLayerRepro() },
            { NcnnLayerTypes.Reorg, () => new NcnnReorgLayerRepro() },
            { NcnnLayerTypes.Sigmoid, () => new NcnnSigmoidLayerRepro() },
            { NcnnLayerTypes.RMSNorm, () => new NcnnRMSNormLayerRepro() },
            { NcnnLayerTypes.RotaryEmbed, () => new NcnnRotaryEmbedLayerRepro() },
            { NcnnLayerTypes.Scale, () => new NcnnScaleLayerRepro() },
            { NcnnLayerTypes.SDPA, () => new NcnnSdpaLayerRepro() },
            { NcnnLayerTypes.SELU, () => new NcnnPointwiseFormulaLayerRepro(NcnnLayerTypes.SELU) },
            { NcnnLayerTypes.Shrink, () => new NcnnPointwiseFormulaLayerRepro(NcnnLayerTypes.Shrink) },
            { NcnnLayerTypes.Softplus, () => new NcnnPointwiseFormulaLayerRepro(NcnnLayerTypes.Softplus) },
            { NcnnLayerTypes.GELU, () => new NcnnGeluLayerRepro() },
            { NcnnLayerTypes.Cast, () => new NcnnCastLayerRepro() },
            { NcnnLayerTypes.CELU, () => new NcnnPointwiseFormulaLayerRepro(NcnnLayerTypes.CELU) },
            { NcnnLayerTypes.Clip, () => new NcnnClipLayerRepro() },
            { NcnnLayerTypes.Softmax, () => new NcnnSoftmaxLayerRepro() },
            { NcnnLayerTypes.Padding, () => new NcnnPaddingLayerRepro() },
            { NcnnLayerTypes.Pooling, () => new NcnnPoolingLayerRepro() },
            { NcnnLayerTypes.Pooling3D, () => new NcnnPooling3DLayerRepro() },
            { NcnnLayerTypes.InnerProduct, () => new NcnnInnerProductLayerRepro() },
            { NcnnLayerTypes.MatMul, () => new NcnnMatMulLayerRepro() },
            { NcnnLayerTypes.Gemm, () => new NcnnGemmLayerRepro() },
            { NcnnLayerTypes.MultiHeadAttention, () => new NcnnMultiHeadAttentionLayerRepro() },
            { NcnnLayerTypes.LayerNorm, () => new NcnnLayerNormLayerRepro() },
            { NcnnLayerTypes.GroupNorm, () => new NcnnGroupNormLayerRepro() },
            { NcnnLayerTypes.BatchNorm, () => new NcnnBatchNormLayerRepro() },
            { NcnnLayerTypes.Embed, () => new NcnnEmbedLayerRepro() },
            { NcnnLayerTypes.Reduction, () => new NcnnReductionLayerRepro() },
            { NcnnLayerTypes.MemoryData, () => new NcnnMemoryDataLayerRepro() },
            { NcnnLayerTypes.ReLU, () => new NcnnReLULayerRepro() },
            { NcnnLayerTypes.DeepCopy, () => new NcnnDeepCopyLayerRepro() },
            { NcnnLayerTypes.MaxPoolingInd, () => new NcnnMaxPoolingIndLayerRepro() },
            { NcnnLayerTypes.MaxUnPooling, () => new NcnnMaxUnPoolingLayerRepro() },
            { NcnnLayerTypes.Unfold, () => new NcnnUnfoldLayerRepro() },
            { NcnnLayerTypes.Tile, () => new NcnnTileLayerRepro() },
            { NcnnLayerTypes.Shape, () => new NcnnShapeLayerRepro() },
            { NcnnLayerTypes.Size, () => new NcnnSizeLayerRepro() },
            { NcnnLayerTypes.Range, () => new NcnnRangeLayerRepro() },
            { NcnnLayerTypes.ConstantOfShape, () => new NcnnConstantOfShapeLayerRepro() },
            { NcnnLayerTypes.Expand, () => new NcnnExpandLayerRepro() },
            { NcnnLayerTypes.ArgMax, () => new NcnnArgReduceLayerRepro(NcnnLayerTypes.ArgMax, reduceMax: true) },
            { NcnnLayerTypes.ArgMin, () => new NcnnArgReduceLayerRepro(NcnnLayerTypes.ArgMin, reduceMax: false) },
            { NcnnLayerTypes.Where, () => new NcnnWhereLayerRepro() },
            { NcnnLayerTypes.TopK, () => new NcnnTopKLayerRepro() },
            { NcnnLayerTypes.NonZero, () => new NcnnUnsupportedSentisLayerRepro(NcnnLayerTypes.NonZero, "NonZero has data-dependent output length and needs GPU compaction/shape-tensor support.") },
            { NcnnLayerTypes.OneHot, () => new NcnnOneHotLayerRepro() },
            { NcnnLayerTypes.CumSum, () => new NcnnCumSumLayerRepro() },
            { NcnnLayerTypes.Compress, () => new NcnnUnsupportedSentisLayerRepro(NcnnLayerTypes.Compress, "Compress has data-dependent output length unless the condition is statically folded.") },
            { NcnnLayerTypes.Gather, () => new NcnnGatherLayerRepro() },
            { NcnnLayerTypes.GatherElements, () => new NcnnGatherElementsLayerRepro() },
            { NcnnLayerTypes.GatherND, () => new NcnnUnsupportedSentisLayerRepro(NcnnLayerTypes.GatherND, "GatherND is registered but not yet implemented on the texture-backed path.") },
            { NcnnLayerTypes.ScatterElements, () => new NcnnUnsupportedSentisLayerRepro(NcnnLayerTypes.ScatterElements, "ScatterElements needs conflict-safe texture writes/reductions before it can run without buffer fallback.") },
            { NcnnLayerTypes.ScatterND, () => new NcnnUnsupportedSentisLayerRepro(NcnnLayerTypes.ScatterND, "ScatterND needs conflict-safe texture writes/reductions before it can run without buffer fallback.") },
            { NcnnLayerTypes.Scatter, () => new NcnnUnsupportedSentisLayerRepro(NcnnLayerTypes.Scatter, "Scatter needs conflict-safe texture writes/reductions before it can run without buffer fallback.") },
        };

        public static IReadOnlyList<NcnnBaseLayerRepro> CreateModelLayers(IList<NcnnParamModel.Layer> layers)
        {
            if (layers == null || layers.Count == 0)
                return Array.Empty<NcnnBaseLayerRepro>();

            var result = new NcnnBaseLayerRepro[layers.Count];
            for (var i = 0; i < layers.Count; i++)
                result[i] = Create(layers[i]);
            return result;
        }

        // Metadata consumers must inspect registration without instantiating or executing a layer.
        public static IReadOnlyList<NcnnLayerTypeKey> GetRegisteredLayerTypes()
        {
            var types = new List<NcnnLayerTypeKey>(Registry.Keys);
            types.Sort((left, right) => string.CompareOrdinal(left.ToString(), right.ToString()));
            return types;
        }

        public static bool IsRegistered(NcnnLayerTypeKey typeKey)
        {
            return Registry.ContainsKey(typeKey);
        }

        public static NcnnBaseLayerRepro Create(NcnnParamModel.Layer layer)
        {
            if (layer == null)
                return new NcnnUnknownLayerRepro(default);

            if (Registry.TryGetValue(layer.type, out var factory))
                return factory();

            return new NcnnUnknownLayerRepro(layer.type);
        }

        private sealed class NcnnUnknownLayerRepro : NcnnBaseLayerRepro
        {
            public NcnnUnknownLayerRepro(NcnnLayerTypeKey typeKey)
                : base(typeKey, supportsBufferPath: false, supportsCommandBufferPath: false)
            {
            }
        }
    }

    public partial class NcnnRepro
    {
        private static int _fixedInputTextureDumpSequence;

        private static void TryDumpFixedInputTexture(string blobName, RenderTexture texture, NcnnTensorBuffer view)
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

        private static int GetReplayConsumedBottomCount(NcnnParamModel.Layer layer)
        {
            if (layer?.bottomNames == null || layer.bottomNames.Length == 0)
                return 0;

            if (layer.type == NcnnLayerTypes.AtenTo)
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

        internal InferResult InferWithMultiInputsByLayerRepros(
            Dictionary<string, RenderTexture> textureInputs,
            Dictionary<string, NcnnTensorBuffer> bufferInputs,
            ICollection<string> pinnedNames = null,
            Dictionary<string, BufferShape> textureInputShapes = null,
            string stopAfterTopName = null,
            string startAtTopName = null)
        {
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
                Dictionary<string, NcnnTensorBuffer> bufferViews,
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
                            var inferredShape = NcnnRepro.GetTextureShape(textureShapes, tex, name);
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
                Dictionary<string, NcnnTensorBuffer> bufferViews,
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

            var remaining = BuildScopedBlobUseCount(startLayerIndex, replayRequiredLayerIndices, stopAfterTopName);
            var textureBlobs = new Dictionary<string, TensorRef>(StringComparer.Ordinal);
            var textureShapes = new Dictionary<string, BufferShape>(StringComparer.Ordinal);
            var bufferBlobs = new Dictionary<string, ComputeBuffer>(StringComparer.Ordinal);
            var bufferRefs = new Dictionary<string, BufferRef>(StringComparer.Ordinal);
            var bufferViews = new Dictionary<string, NcnnTensorBuffer>(StringComparer.Ordinal);
            var indexBlobs = new Dictionary<string, IndexRef>(StringComparer.Ordinal);
            var tempOwned = new List<IDisposable>();

            RegisterTextureInputs(textureInputs, textureInputShapes, textureBlobs, textureShapes);

            if (bufferInputs != null)
            {
                foreach (var kv in bufferInputs)
                {
                    if (kv.Value == null || kv.Value.buffer == null)
                        throw new ArgumentNullException("bufferInputs[\"" + kv.Key + "\"]");
                    bufferBlobs[kv.Key] = kv.Value.buffer;
                    bufferRefs[kv.Key] = new BufferRef
                    {
                        buffer = kv.Value.buffer,
                        refs = 1,
                        owned = false
                    };
                    bufferViews[kv.Key] = kv.Value;

                    // A fixed Buffer input is a valid graph boundary, but it is not an
                    // activation fallback. Production layers must consume it directly
                    // (for example, Embed token indices) or reject it with the tensor
                    // contract; only an explicitly requested debug-oracle run may upload
                    // it to a texture for legacy numerical comparison.
                    if (IsDebugOracleExecution && !textureBlobs.ContainsKey(kv.Key))
                    {
                        var texture = MaterializeScratchTextureFromBufferView(kv.Value.buffer, kv.Value);
                        if (texture != null)
                        {
                            var shape = new BufferShape(kv.Value.dims, kv.Value.w, kv.Value.h, kv.Value.d, kv.Value.c);
                            SetTextureBlob(textureBlobs, textureShapes, kv.Key, texture, shape);
                            TryDumpFixedInputTexture(kv.Key, texture, kv.Value);
                        }
                    }
                }
            }

            var context = new NcnnLayerBufferContext
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

            bool TryLogFirstNonFiniteLayerOutput(int layerIndex, NcnnParamModel.Layer layer)
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

            BeginInferenceTempResourceTracking();
            try
            {
                var runtimeProfile = BeginLayerRuntimeProfile("buffer");
                for (var li = startLayerIndex; li < Model.layers.Count; li++)
                {
                    if (replayRequiredLayerIndices != null && !replayRequiredLayerIndices.Contains(li))
                        continue;

                    var layer = Model.layers[li];
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
                        continue;
                    }

                    var layerRepro = LayerRepros[li];
                    if (layerRepro == null)
                        throw new InvalidOperationException("layer repro missing: " + layer?.name);
                    if (runtimeProfile == null)
                    {
                        SetCurrentExecutingLayer(layer);
                        try
                        {
                            layerRepro.ExecuteBuffer(this, layer, context);
                        }
                        catch (Exception e)
                        {
                            throw new InvalidOperationException(
                                "Layer execution failed"
                                + " | idx=" + li + "/" + Model.layers.Count
                                + " | name=" + (layer?.name ?? string.Empty)
                                + " | type=" + (layer?.typeName ?? string.Empty)
                                + " | bottoms=" + DescribeBlobStates(layer?.bottomNames, textureBlobs, textureShapes, bufferBlobs, bufferViews, indexBlobs)
                                + " | tops=" + JoinNames(layer?.topNames)
                                + " | inner=" + e.Message,
                                e);
                        }
                        finally
                        {
                            ClearCurrentExecutingLayer();
                        }
                        layerOutputPath = DescribeLayerOutputPath(layer, textureBlobs, textureShapes, bufferBlobs, bufferViews, indexBlobs);
                        if (DebugLog != null && (DebugLogAllLayerOutputs || HasStrideBlob(layer?.topNames)))
                        {
                            DebugLog("[LayerOutput] idx=" + li
                                + " | name=" + (layer?.name ?? string.Empty)
                                + " | path=" + layerOutputPath);
                        }
                        TryLogFirstNonFiniteLayerOutput(li, layer);
                        if (!string.IsNullOrWhiteSpace(stopAfterTopName)
                            && layer?.topNames != null
                            && Array.IndexOf(layer.topNames, stopAfterTopName) >= 0
                            && AreAllLayerTopsAlreadyAvailable(layer, textureBlobs, bufferBlobs, indexBlobs))
                        {
                            break;
                        }
                        continue;
                    }

                    var layerSw = Stopwatch.StartNew();
                    SetCurrentExecutingLayer(layer);
                    try
                    {
                        layerRepro.ExecuteBuffer(this, layer, context);
                    }
                    catch (Exception e)
                    {
                        throw new InvalidOperationException(
                            "Layer execution failed"
                            + " | idx=" + li + "/" + Model.layers.Count
                            + " | name=" + (layer?.name ?? string.Empty)
                            + " | type=" + (layer?.typeName ?? string.Empty)
                            + " | bottoms=" + DescribeBlobStates(layer?.bottomNames, textureBlobs, textureShapes, bufferBlobs, bufferViews, indexBlobs)
                            + " | tops=" + JoinNames(layer?.topNames)
                            + " | inner=" + e.Message,
                            e);
                    }
                    finally
                    {
                        ClearCurrentExecutingLayer();
                    }
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
                    if (!string.IsNullOrWhiteSpace(stopAfterTopName)
                        && layer?.topNames != null
                        && Array.IndexOf(layer.topNames, stopAfterTopName) >= 0
                        && AreAllLayerTopsAlreadyAvailable(layer, textureBlobs, bufferBlobs, indexBlobs))
                    {
                        break;
                    }
                }

                FinishLayerRuntimeProfile(runtimeProfile);
                var disallowTextureToBufferFallback =
                    DisallowBufferAccess
                    || DisallowBufferOutputs
                    || DisallowBufferToTextureMaterialization
                    || DisallowInferenceTempComputeBuffers;
                return new InferResult(
                    textureBlobs,
                    textureShapes,
                    bufferBlobs,
                    bufferRefs,
                    bufferViews,
                    tempOwned,
                    this,
                    disallowTextureToBufferFallback);
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

                var context = new NcnnLayerCommandBufferContext
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
                        try
                        {
                            layerRepro.ExecuteCommandBuffer(this, layer, context);
                        }
                        finally
                        {
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
                    try
                    {
                        layerRepro.ExecuteCommandBuffer(this, layer, context);
                    }
                    finally
                    {
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
            string stopAfterTopName = null)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if (textureInputs == null || textureInputs.Count == 0)
                throw new ArgumentNullException(nameof(textureInputs));

            EnsureCommandBufferTextureExecutionPlan(textureInputs, textureInputShapes);
            BeginInferenceTempResourceTracking();
            Dictionary<string, CmdTensorRef> blobs = null;
            var returned = false;
            try
            {
                var remaining = new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);
                var shapes = new Dictionary<string, BufferShape>(StringComparer.Ordinal);
                blobs = new Dictionary<string, CmdTensorRef>(StringComparer.Ordinal);
                RegisterCmdTextureInputs(textureInputs, textureInputShapes, blobs, shapes);

                var context = new NcnnLayerCommandBufferContext
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
                        try
                        {
                            layerRepro.ExecuteCommandBuffer(this, layer, context);
                        }
                        finally
                        {
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
                        try
                        {
                            layerRepro.ExecuteCommandBuffer(this, layer, context);
                        }
                        finally
                        {
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
                var outBlobName = !string.IsNullOrWhiteSpace(stopAfterTopName)
                    ? stopAfterTopName
                    : (!string.IsNullOrWhiteSpace(outputBlobName) ? outputBlobName : ResolveDefaultOutputBlobName());
                var outRef = GetCmdTensor(blobs, outBlobName);
                outputLogicalShape = GetCmdShape(shapes, blobs, outBlobName);
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
    }
}
