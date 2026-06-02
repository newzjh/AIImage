using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public static class NcnnLayerFactoryRepro
    {
        private static readonly Dictionary<NcnnLayerTypeKey, Func<NcnnBaseLayerRepro>> Registry = new Dictionary<NcnnLayerTypeKey, Func<NcnnBaseLayerRepro>>
        {
            { NcnnLayerTypes.Input, () => new NcnnInputLayerRepro() },
            { NcnnLayerTypes.Split, () => new NcnnSplitLayerRepro() },
            { NcnnLayerTypes.Concat, () => new NcnnConcatLayerRepro() },
            { NcnnLayerTypes.Reshape, () => new NcnnReshapeLayerRepro() },
            { NcnnLayerTypes.ShuffleChannel, () => new NcnnShuffleChannelLayerRepro() },
            { NcnnLayerTypes.Permute, () => new NcnnPermuteLayerRepro() },
            { NcnnLayerTypes.Slice, () => new NcnnSliceLayerRepro() },
            { NcnnLayerTypes.ExpandDims, () => new NcnnExpandDimsLayerRepro() },
            { NcnnLayerTypes.Squeeze, () => new NcnnSqueezeLayerRepro() },
            { NcnnLayerTypes.Crop, () => new NcnnCropLayerRepro() },
            { NcnnLayerTypes.Convolution, () => new NcnnConvolutionLayerRepro() },
            { NcnnLayerTypes.ConvolutionDepthWise, () => new NcnnConvolutionDepthWiseLayerRepro() },
            { NcnnLayerTypes.Deconvolution, () => new NcnnDeconvolutionLayerRepro() },
            { NcnnLayerTypes.Interp, () => new NcnnInterpLayerRepro() },
            { NcnnLayerTypes.Eltwise, () => new NcnnEltwiseLayerRepro() },
            { NcnnLayerTypes.BinaryOp, () => new NcnnBinaryOpLayerRepro() },
            { NcnnLayerTypes.UnaryOp, () => new NcnnUnaryOpLayerRepro() },
            { NcnnLayerTypes.Swish, () => new NcnnSwishLayerRepro() },
            { NcnnLayerTypes.Sigmoid, () => new NcnnSigmoidLayerRepro() },
            { NcnnLayerTypes.GELU, () => new NcnnGeluLayerRepro() },
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
            { NcnnLayerTypes.MaxPoolingInd, () => new NcnnMaxPoolingIndLayerRepro() },
            { NcnnLayerTypes.MaxUnPooling, () => new NcnnMaxUnPoolingLayerRepro() },
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
            ICollection<string> pinnedNames = null)
        {
            var remaining = new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);
            var textureBlobs = new Dictionary<string, TensorRef>(StringComparer.Ordinal);
            var textureShapes = new Dictionary<string, BufferShape>(StringComparer.Ordinal);
            var bufferBlobs = new Dictionary<string, ComputeBuffer>(StringComparer.Ordinal);
            var bufferRefs = new Dictionary<string, BufferRef>(StringComparer.Ordinal);
            var bufferViews = new Dictionary<string, NcnnTensorBuffer>(StringComparer.Ordinal);
            var indexBlobs = new Dictionary<string, IndexRef>(StringComparer.Ordinal);
            var tempOwned = new List<IDisposable>();

            if (textureInputs != null)
            {
                foreach (var kv in textureInputs)
                {
                    if (kv.Value == null)
                        throw new ArgumentNullException("textureInputs[\"" + kv.Key + "\"]");
                    var rt = kv.Value;
                    var packs = rt.volumeDepth > 0 ? rt.volumeDepth : 1;
                    var useCount = _blobUseCount.TryGetValue(kv.Key, out var c) ? c : 1;
                    var logicalChannels = ResolveInputLogicalChannels(kv.Key, packs * 4);
                    textureBlobs[kv.Key] = new TensorRef
                    {
                        texture = rt,
                        width = rt.width,
                        height = rt.height,
                        packs = packs,
                        refs = useCount,
                        owned = false
                    };
                    textureShapes[kv.Key] = new BufferShape(3, rt.width, rt.height, 1, logicalChannels);
                }
            }

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

            for (var li = 0; li < Model.layers.Count; li++)
            {
                var layer = Model.layers[li];
                var layerRepro = LayerRepros[li];
                if (layerRepro == null)
                    throw new InvalidOperationException("layer repro missing: " + layer?.name);
                layerRepro.ExecuteBuffer(this, layer, context);
            }

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
                remaining = remaining,
                pinnedNames = pinnedNames
            };

            for (var li = 0; li < Model.layers.Count; li++)
            {
                var layer = Model.layers[li];
                var layerRepro = LayerRepros[li];
                if (layerRepro == null)
                    throw new InvalidOperationException("layer repro missing: " + layer?.name);
                layerRepro.ExecuteCommandBuffer(this, layer, context);
            }

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
