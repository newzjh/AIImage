using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnRotaryEmbedLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnRotaryEmbedLayerRepro()
            : base(NcnnLayerTypes.RotaryEmbed, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            owner._extraPacks[layer.name] = new NcnnRepro.RotaryEmbedPack
            {
                interleaved = layer.GetInt(0, 0) != 0
            };
            return default;
        }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (!owner.ShouldForceCurrentLayerBufferPath() && HasAnyTextureBottom(layer, context))
            {
                ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.RotaryEmbedPack rp)
                throw new InvalidOperationException("RotaryEmbed pack not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            var cosBuf = owner.GetOrConvertToBuffer(layer.bottomNames[1], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var sinBuf = owner.GetOrConvertToBuffer(layer.bottomNames[2], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var cosView = NcnnRepro.TryGetBufferView(layer.bottomNames[1], bufferBlobs, bufferViews);
            var sinView = NcnnRepro.TryGetBufferView(layer.bottomNames[2], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null || srcView.dims != 3)
                throw new InvalidOperationException("RotaryEmbed expects dims=3 source: " + layer.name);
            if (cosBuf == null || sinBuf == null || cosView == null || sinView == null)
                throw new InvalidOperationException("RotaryEmbed cache input missing: " + layer.name);

            var embedDim = srcView.w;
            var seqLen = srcView.h;
            var numHeads = srcView.c;
            if ((embedDim & 1) != 0)
                throw new InvalidOperationException("RotaryEmbed requires even embed_dim: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(3, srcView.w, srcView.h, 1, srcView.c);
            owner.Ops.RotaryEmbedBuf(srcBuf, embedDim, seqLen, numHeads, rp.interleaved, cosBuf, sinBuf, outTensor.buffer);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: true,
                textureBlobs,
                textureShapes,
                bufferBlobs,
                bufferRefs,
                bufferViews,
                tempOwned);
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.RotaryEmbedPack rp)
                throw new InvalidOperationException("RotaryEmbed pack not found: " + layer.name);

            var src = GetRenderTextureBottom(layer, context, 0, "source");
            var cos = GetRenderTextureBottom(layer, context, 1, "cos");
            var sin = GetRenderTextureBottom(layer, context, 2, "sin");
            var srcShape = NcnnRepro.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
            var cosShape = NcnnRepro.GetTextureShape(textureShapes, cos, layer.bottomNames[1]);
            var sinShape = NcnnRepro.GetTextureShape(textureShapes, sin, layer.bottomNames[2]);
            ValidateRenderTextureInputs(layer, src, srcShape, cos, cosShape, sin, sinShape);

            var storageShape = NcnnRepro.GetTextureStorageShape(src, srcShape);
            RenderTexture output = null;
            try
            {
                output = owner.RentTempArray(
                    src.texture.width,
                    src.texture.height,
                    Mathf.Max(1, src.texture.volumeDepth),
                    NcnnRepro.ResolveTensorTextureFormat(srcShape.dims));
                owner.Ops.RotaryEmbedPack4(src.texture, srcShape, cos.texture, cosShape, sin.texture, sinShape, rp.interleaved, output);
                NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], output, srcShape, storageShape);
                output = null;
            }
            finally
            {
                if (output != null)
                    owner.ReturnTempArray(output);
            }

            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.RotaryEmbedPack rp)
                throw new InvalidOperationException("RotaryEmbed pack not found: " + layer.name);

            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var cos = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[1]);
            var cosShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[1]);
            var sin = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[2]);
            var sinShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[2]);
            ValidateCommandBufferInputs(layer, src, srcShape, cos, cosShape, sin, sinShape);

            var storageShape = NcnnRepro.GetCmdStorageShape(src, srcShape);
            ComputeTexture output = null;
            try
            {
                output = owner.RentTempArray(
                    cmd,
                    src.texture.width,
                    src.texture.height,
                    Mathf.Max(1, src.texture.depth),
                    NcnnRepro.ResolveTensorTextureFormat(srcShape.dims));
                owner.Ops.RotaryEmbedPack4(cmd, src.texture, srcShape, cos.texture, cosShape, sin.texture, sinShape, rp.interleaved, output);
                blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(output, srcShape, storageShape, owned: true);
                output = null;
                if (shapes != null)
                    shapes[layer.topNames[0]] = srcShape;
            }
            finally
            {
                if (output != null)
                    owner.ReturnTempArray(cmd, output);
            }

            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool HasAnyTextureBottom(NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (layer.bottomNames == null || context.textureBlobs == null)
                return false;

            for (var i = 0; i < layer.bottomNames.Length; i++)
            {
                if (context.textureBlobs.TryGetValue(layer.bottomNames[i], out var tensor) && tensor != null && tensor.texture != null)
                    return true;
            }

            return false;
        }

        private static NcnnRepro.TensorRef GetRenderTextureBottom(NcnnParamModel.Layer layer, NcnnLayerBufferContext context, int index, string role)
        {
            var name = layer.bottomNames[index];
            if (context.textureBlobs == null
                || !context.textureBlobs.TryGetValue(name, out var tensor)
                || tensor == null
                || tensor.texture == null)
                throw new InvalidOperationException("RotaryEmbed render-texture path requires texture " + role + " input: " + layer.name + " | bottom=" + name);
            return tensor;
        }

        private static void ValidateRenderTextureInputs(
            NcnnParamModel.Layer layer,
            NcnnRepro.TensorRef src,
            NcnnRepro.BufferShape srcShape,
            NcnnRepro.TensorRef cos,
            NcnnRepro.BufferShape cosShape,
            NcnnRepro.TensorRef sin,
            NcnnRepro.BufferShape sinShape)
        {
            if (srcShape.dims != 3)
                throw new InvalidOperationException("RotaryEmbed render-texture path requires dims=3 source: " + layer.name + " | shape=" + FormatShape(srcShape));
            if ((srcShape.w & 1) != 0)
                throw new InvalidOperationException("RotaryEmbed requires even embed_dim: " + layer.name + " | embed_dim=" + srcShape.w);
            ValidatePack4TextureArray(layer, "source", src, srcShape);
            ValidatePack4TextureArray(layer, "cos", cos, cosShape);
            ValidatePack4TextureArray(layer, "sin", sin, sinShape);
            ValidateCacheShape(layer, srcShape, cosShape, sinShape);
        }

        private static void ValidateCommandBufferInputs(
            NcnnParamModel.Layer layer,
            NcnnRepro.CmdTensorRef src,
            NcnnRepro.BufferShape srcShape,
            NcnnRepro.CmdTensorRef cos,
            NcnnRepro.BufferShape cosShape,
            NcnnRepro.CmdTensorRef sin,
            NcnnRepro.BufferShape sinShape)
        {
            if (src == null || src.texture == null || cos == null || cos.texture == null || sin == null || sin.texture == null)
                throw new InvalidOperationException("RotaryEmbed command-buffer path requires source/cos/sin texture inputs: " + layer.name);
            if (srcShape.dims != 3)
                throw new InvalidOperationException("RotaryEmbed command-buffer path requires dims=3 source: " + layer.name + " | shape=" + FormatShape(srcShape));
            if ((srcShape.w & 1) != 0)
                throw new InvalidOperationException("RotaryEmbed requires even embed_dim: " + layer.name + " | embed_dim=" + srcShape.w);
            ValidatePack4TextureArray(layer, "source", src, srcShape);
            ValidatePack4TextureArray(layer, "cos", cos, cosShape);
            ValidatePack4TextureArray(layer, "sin", sin, sinShape);
            ValidateCacheShape(layer, srcShape, cosShape, sinShape);
        }

        private static void ValidatePack4TextureArray(NcnnParamModel.Layer layer, string role, NcnnRepro.TensorRef tensor, NcnnRepro.BufferShape logicalShape)
        {
            if (tensor == null || tensor.texture == null)
                throw new InvalidOperationException("RotaryEmbed render-texture path requires texture " + role + " input: " + layer.name);
            if (NcnnRepro.IsStrictLinearMatTexture(tensor) || tensor.texture.dimension != TextureDimension.Tex2DArray)
                throw new InvalidOperationException("RotaryEmbed render-texture path requires texture-array " + role + " input: " + layer.name + " | shape=" + FormatShape(logicalShape));
            var storageShape = NcnnRepro.GetTextureStorageShape(tensor, logicalShape);
            if (!NcnnRepro.BufferShapeEquals(logicalShape, storageShape))
                throw new InvalidOperationException(
                    "RotaryEmbed render-texture path requires matching logical/storage shape for " + role
                    + ": " + layer.name
                    + " | logical=" + FormatShape(logicalShape)
                    + " | storage=" + FormatShape(storageShape));
        }

        private static void ValidatePack4TextureArray(NcnnParamModel.Layer layer, string role, NcnnRepro.CmdTensorRef tensor, NcnnRepro.BufferShape logicalShape)
        {
            if (tensor == null || tensor.texture == null)
                throw new InvalidOperationException("RotaryEmbed command-buffer path requires texture " + role + " input: " + layer.name);
            if (NcnnRepro.IsStrictLinearMatTexture(tensor) || tensor.texture.dimension != TextureDimension.Tex2DArray)
                throw new InvalidOperationException("RotaryEmbed command-buffer path requires texture-array " + role + " input: " + layer.name + " | shape=" + FormatShape(logicalShape));
            var storageShape = NcnnRepro.GetCmdStorageShape(tensor, logicalShape);
            if (!NcnnRepro.BufferShapeEquals(logicalShape, storageShape))
                throw new InvalidOperationException(
                    "RotaryEmbed command-buffer path requires matching logical/storage shape for " + role
                    + ": " + layer.name
                    + " | logical=" + FormatShape(logicalShape)
                    + " | storage=" + FormatShape(storageShape));
        }

        private static void ValidateCacheShape(NcnnParamModel.Layer layer, NcnnRepro.BufferShape srcShape, NcnnRepro.BufferShape cosShape, NcnnRepro.BufferShape sinShape)
        {
            var required = (long)Mathf.Max(1, srcShape.h) * Mathf.Max(1, srcShape.w / 2);
            if (GetElementCount(cosShape) < required || GetElementCount(sinShape) < required)
                throw new InvalidOperationException(
                    "RotaryEmbed cache texture is smaller than seq_len * embed_dim/2: " + layer.name
                    + " | required=" + required
                    + " | cos=" + FormatShape(cosShape)
                    + " | sin=" + FormatShape(sinShape));
        }

        private static long GetElementCount(NcnnRepro.BufferShape shape)
        {
            var w = Mathf.Max(1, shape.w);
            var h = shape.dims >= 2 ? Mathf.Max(1, shape.h) : 1;
            var d = shape.dims >= 4 ? Mathf.Max(1, shape.d) : 1;
            var c = shape.dims >= 3 ? Mathf.Max(1, shape.c) : 1;
            return (long)w * h * d * c;
        }

        private static string FormatShape(NcnnRepro.BufferShape shape)
        {
            return "d" + shape.dims + ":" + shape.w + "x" + shape.h + "x" + shape.d + "x" + shape.c;
        }
    }
}
