using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Ncnn
{
    /// Texture-only causal depthwise convolution used by recurrent linear blocks.
    /// The cache is a graph tensor and is never materialized through a buffer.
    public sealed class NcnnShortConvLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnShortConvLayerRepro()
            : base(NcnnLayerTypes.ShortConv, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
        }

        public override NcnnGraphSession.LayerLoadMetrics LoadLayer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            owner._extraPacks[layer.name] = new NcnnGraphSession.ShortConvPack();
            return default;
        }

        public override void ExecuteBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            // InferWithMultiInputs uses this dispatch entry for both storage modes.
            // Route it to the strict texture implementation; there is no buffer kernel.
            ExecuteRenderTexturePath(owner, layer, context);
        }

        public override void ExecuteRenderTexturePath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (layer.bottomNames == null || layer.bottomNames.Length < 3 || layer.topNames == null || layer.topNames.Length < 2)
                throw new InvalidOperationException("ShortConv requires weight, mixed input, cache and two outputs: " + layer.name);

            var weightRef = RequireTexture(context, layer.bottomNames[0], layer.name);
            var mixedRef = RequireTexture(context, layer.bottomNames[1], layer.name);
            var cacheRef = RequireTexture(context, layer.bottomNames[2], layer.name);
            var weight = ToArray(owner, weightRef, context.textureShapes, layer.bottomNames[0]);
            var mixed = ToArray(owner, mixedRef, context.textureShapes, layer.bottomNames[1]);
            var cache = ToArray(owner, cacheRef, context.textureShapes, layer.bottomNames[2]);
            var mixedShape = NcnnGraphSession.GetTextureShape(context.textureShapes, mixedRef, layer.bottomNames[1]);
            var cacheShape = NcnnGraphSession.GetTextureShape(context.textureShapes, cacheRef, layer.bottomNames[2]);
            var groups = mixedShape.w;
            var kernel = cacheShape.h > 0 ? cacheShape.h : 1;
            var cacheLength = Mathf.Max(kernel, cacheShape.h);
            var output = owner.RentTempArray(mixed.texture.width, mixed.texture.height, mixed.texture.volumeDepth, owner.TensorTextureFormat);
            var cacheOut = owner.RentTempArray(cache.texture.width, cache.texture.height, cache.texture.volumeDepth, owner.TensorTextureFormat);
            try
            {
                owner.Ops.ShortConvPack4(weight.texture, mixed.texture, cache.texture, output, cacheOut, groups, kernel, mixedShape.h, cacheLength);
                NcnnGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, mixedShape, StorageShape(mixedShape, output));
                NcnnGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[1], cacheOut, cacheShape, StorageShape(cacheShape, cacheOut));
                output = null;
                cacheOut = null;
            }
            finally
            {
                if (output != null) owner.ReturnTempArray(output);
                if (cacheOut != null) owner.ReturnTempArray(cacheOut);
                if (weight.temporary) owner.ReturnTempArray(weight.texture);
                if (mixed.temporary) owner.ReturnTempArray(mixed.texture);
                if (cache.temporary) owner.ReturnTempArray(cache.texture);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            if (layer.bottomNames == null || layer.bottomNames.Length < 3 || layer.topNames == null || layer.topNames.Length < 2)
                throw new InvalidOperationException("ShortConv command contract is invalid: " + layer.name);
            var weight = RequireCmdArray(context, layer.bottomNames[0], layer.name);
            var mixed = RequireCmdArray(context, layer.bottomNames[1], layer.name);
            var cache = RequireCmdArray(context, layer.bottomNames[2], layer.name);
            var mixedShape = NcnnGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[1]);
            var cacheShape = NcnnGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[2]);
            var output = owner.RentTempArray(context.commandBuffer, mixed.texture.width, mixed.texture.height, mixed.texture.depth, owner.TensorTextureFormat);
            var cacheOut = owner.RentTempArray(context.commandBuffer, cache.texture.width, cache.texture.height, cache.texture.depth, owner.TensorTextureFormat);
            owner.Ops.ShortConvPack4(context.commandBuffer, weight.texture, mixed.texture, cache.texture, output, cacheOut, mixedShape.w, Mathf.Max(1, cacheShape.h), mixedShape.h, Mathf.Max(1, cacheShape.h));
            context.blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorRef(output, mixedShape, new NcnnGraphSession.BufferShape(3, output.width, output.height, 1, output.depth * 4), owned: true);
            context.blobs[layer.topNames[1]] = NcnnGraphSession.CreateCmdTensorRef(cacheOut, cacheShape, new NcnnGraphSession.BufferShape(3, cacheOut.width, cacheOut.height, 1, cacheOut.depth * 4), owned: true);
            if (context.shapes != null) { context.shapes[layer.topNames[0]] = mixedShape; context.shapes[layer.topNames[1]] = cacheShape; }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static NcnnGraphSession.TensorRef RequireTexture(NcnnLayerBufferContext context, string name, string layer)
        {
            if (context.textureBlobs == null || !context.textureBlobs.TryGetValue(name, out var tensor) || tensor == null || tensor.texture == null)
                throw new InvalidOperationException("ShortConv requires texture input (buffer fallback prohibited): layer=" + layer + " blob=" + name);
            return tensor;
        }

        private static NcnnGraphSession.CmdTensorRef RequireCmdArray(NcnnLayerCommandBufferContext context, string name, string layer)
        {
            var tensor = NcnnGraphSession.GetCmdTensor(context.blobs, name);
            if (tensor == null || tensor.texture == null || tensor.texture.dimension != TextureDimension.Tex2DArray)
                throw new InvalidOperationException("ShortConv command path requires Texture2DArray input: layer=" + layer + " blob=" + name);
            return tensor;
        }

        private struct TextureView { public RenderTexture texture; public bool temporary; }

        private static TextureView ToArray(NcnnGraphSession owner, NcnnGraphSession.TensorRef source, Dictionary<string, NcnnGraphSession.BufferShape> shapes, string name)
        {
            if (source.texture.dimension == TextureDimension.Tex2DArray)
                return new TextureView { texture = source.texture, temporary = false };
            var logical = NcnnGraphSession.GetTextureShape(shapes, source, name);
            var storage = NcnnGraphSession.GetTextureStorageShape(source, logical);
            var array = owner.RentTempArray(Mathf.CeilToInt(Mathf.Max(1, logical.w) / 4f), Mathf.Max(1, logical.h), 1, owner.TensorTextureFormat);
            owner.Ops.ReshapeLinearMatToPack4(source.texture, storage.w, storage.h, logical.w, logical.h, Mathf.Max(1, logical.d), Mathf.Max(1, logical.c), logical.dims, array);
            return new TextureView { texture = array, temporary = true };
        }

        private static NcnnGraphSession.BufferShape StorageShape(NcnnGraphSession.BufferShape logical, RenderTexture texture)
        {
            return logical.dims <= 2
                ? new NcnnGraphSession.BufferShape(3, texture.width, texture.height, 1, Mathf.Max(1, texture.volumeDepth) * 4)
                : new NcnnGraphSession.BufferShape(logical.dims, texture.width, texture.height, logical.d, Mathf.Max(1, texture.volumeDepth) * 4);
        }
    }
}
