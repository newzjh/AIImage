using System;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public static class NcnnLayerFactoryRepro
    {
        private static readonly Dictionary<NcnnLayerTypeKey, Func<NcnnBaseLayerRepro>> Registry = new Dictionary<NcnnLayerTypeKey, Func<NcnnBaseLayerRepro>>
        {
            { NcnnLayerTypes.Input, () => new NcnnInputLayerRepro() },
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
            { NcnnLayerTypes.Convolution1D, () => new NcnnConvolution1DLayerRepro() },
            { NcnnLayerTypes.ConvolutionDepthWise, () => new NcnnConvolutionDepthWiseLayerRepro() },
            { NcnnLayerTypes.Deconvolution, () => new NcnnDeconvolutionLayerRepro() },
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
        internal InferResult InferWithMultiInputsByLayerRepros(
            Dictionary<string, RenderTexture> textureInputs,
            Dictionary<string, NcnnTensorBuffer> bufferInputs,
            ICollection<string> pinnedNames = null,
            Dictionary<string, BufferShape> textureInputShapes = null)
        {
            static bool HasStrideBlob(string[] names)
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

            static string JoinNames(string[] names)
            {
                if (names == null || names.Length == 0)
                    return "-";
                return string.Join(",", names);
            }

            var remaining = new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);
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

            var runtimeProfile = BeginLayerRuntimeProfile("buffer");
            for (var li = 0; li < Model.layers.Count; li++)
            {
                var layer = Model.layers[li];
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
                    finally
                    {
                        ClearCurrentExecutingLayer();
                    }
                    if (DebugLog != null && (DebugLogAllLayerOutputs || HasStrideBlob(layer?.topNames)))
                    {
                        DebugLog("[LayerOutput] idx=" + li
                            + " | name=" + (layer?.name ?? string.Empty)
                            + " | path=" + DescribeLayerOutputPath(layer, textureBlobs, textureShapes, bufferBlobs, bufferViews, indexBlobs));
                    }
                    continue;
                }

                var layerSw = Stopwatch.StartNew();
                SetCurrentExecutingLayer(layer);
                try
                {
                    layerRepro.ExecuteBuffer(this, layer, context);
                }
                finally
                {
                    ClearCurrentExecutingLayer();
                }
                if (LayerRuntimeProfileSyncGpu)
                    Ops.DebugSyncGpu();
                layerSw.Stop();
                RecordLayerRuntime(
                    runtimeProfile,
                    li,
                    layer,
                    DescribeLayerOutputPath(layer, textureBlobs, textureShapes, bufferBlobs, bufferViews, indexBlobs),
                    layerSw.ElapsedTicks);

                if (DebugLog != null && (DebugLogAllLayerOutputs || HasStrideBlob(layer?.topNames)))
                {
                    DebugLog("[LayerOutput] idx=" + li
                        + " | name=" + (layer?.name ?? string.Empty)
                        + " | path=" + DescribeLayerOutputPath(layer, textureBlobs, textureShapes, bufferBlobs, bufferViews, indexBlobs));
                }
            }

            FinishLayerRuntimeProfile(runtimeProfile);
            return new InferResult(textureBlobs, textureShapes, bufferBlobs, bufferRefs, bufferViews, tempOwned, this);
        }

        internal ComputeTexture ForwardPack4ByLayerRepros(
            CommandBuffer cmd,
            ComputeTexture inputPack4,
            int inputPacks,
            string inputBlobName = "data",
            ICollection<string> pinnedNames = null)
        {
            var remaining = new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);
            var shapes = new Dictionary<string, BufferShape>(StringComparer.Ordinal)
            {
                [inputBlobName] = new BufferShape(3, inputPack4.width, inputPack4.height, 1, ResolveInputLogicalChannels(inputBlobName, inputPacks * 4))
            };
            var blobs = new Dictionary<string, CmdTensorRef>(StringComparer.Ordinal)
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
                layerSw.Stop();
                RecordLayerRuntime(runtimeProfile, li, layer, "cmd", layerSw.ElapsedTicks);
            }

            FinishLayerRuntimeProfile(runtimeProfile);
            var outBlobName = ResolveDefaultOutputBlobName();
            var outRef = GetCmdTensor(blobs, outBlobName);
            var keep = outRef.texture;
            outRef.texture = null;
            outRef.owned = false;

            var visited = new HashSet<CmdTensorRef>();
            foreach (var kv in blobs)
            {
                var tr = kv.Value;
                if (tr == null || !visited.Add(tr))
                    continue;
                if (tr.owned && tr.texture != null)
                    ReturnTempArray(cmd, tr.texture);
            }

            return keep;
        }
    }
}
