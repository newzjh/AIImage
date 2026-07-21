using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisRotaryEmbedLayer : AexisBaseLayer
    {
        public AexisRotaryEmbedLayer()
            : base(AexisLayerTypes.RotaryEmbed, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
            owner._extraPacks[layer.name] = new AexisGraphSession.RotaryEmbedPack
            {
                interleaved = layer.GetInt(0, 0) != 0
            };
            return default;
        }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
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
        public override void ExecuteComputeBufferPath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.RotaryEmbedPack rp)
                throw new InvalidOperationException("RotaryEmbed pack not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            var cosBuf = owner.GetOrConvertToBuffer(layer.bottomNames[1], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var sinBuf = owner.GetOrConvertToBuffer(layer.bottomNames[2], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var cosView = AexisGraphSession.TryGetBufferView(layer.bottomNames[1], bufferBlobs, bufferViews);
            var sinView = AexisGraphSession.TryGetBufferView(layer.bottomNames[2], bufferBlobs, bufferViews);
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

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.RotaryEmbedPack rp)
                throw new InvalidOperationException("RotaryEmbed pack not found: " + layer.name);

            var src = GetRenderTextureBottom(layer, context, 0, "source");
            var cos = GetRenderTextureBottom(layer, context, 1, "cos");
            var sin = GetRenderTextureBottom(layer, context, 2, "sin");
            var srcShape = AexisGraphSession.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
            var cosShape = AexisGraphSession.GetTextureShape(textureShapes, cos, layer.bottomNames[1]);
            var sinShape = AexisGraphSession.GetTextureShape(textureShapes, sin, layer.bottomNames[2]);
            ValidateRenderTextureInputs(layer, src, srcShape, cos, cosShape, sin, sinShape);

            var storageShape = AexisGraphSession.GetTextureStorageShape(src, srcShape);
            RenderTexture output = null;
            try
            {
                output = owner.RentTempArray(
                    src.texture.width,
                    src.texture.height,
                    Mathf.Max(1, src.texture.volumeDepth),
                    AexisGraphSession.ResolveTensorTextureFormat(srcShape.dims));
                owner.Ops.RotaryEmbedPack4(src.texture, srcShape, cos.texture, cosShape, sin.texture, sinShape, rp.interleaved, output);
                AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], output, srcShape, storageShape);
                output = null;
            }
            finally
            {
                if (output != null)
                    owner.ReturnTempArray(output);
            }

            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.RotaryEmbedPack rp)
                throw new InvalidOperationException("RotaryEmbed pack not found: " + layer.name);

            var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var cos = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[1]);
            var cosShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[1]);
            var sin = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[2]);
            var sinShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[2]);
            ValidateCommandBufferInputs(layer, src, srcShape, cos, cosShape, sin, sinShape);

            var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
            ComputeTexture output = null;
            try
            {
                output = owner.RentTempArray(
                    cmd,
                    src.texture.width,
                    src.texture.height,
                    Mathf.Max(1, src.texture.depth),
                    AexisGraphSession.ResolveTensorTextureFormat(srcShape.dims));
                owner.Ops.RotaryEmbedPack4(cmd, src.texture, srcShape, cos.texture, cosShape, sin.texture, sinShape, rp.interleaved, output);
                blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, srcShape, storageShape, owned: true);
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

        private static bool HasAnyTextureBottom(AexisGraphModel.Layer layer, AexisLayerBufferContext context)
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

        private static AexisGraphSession.TensorRef GetRenderTextureBottom(AexisGraphModel.Layer layer, AexisLayerBufferContext context, int index, string role)
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
            AexisGraphModel.Layer layer,
            AexisGraphSession.TensorRef src,
            AexisGraphSession.BufferShape srcShape,
            AexisGraphSession.TensorRef cos,
            AexisGraphSession.BufferShape cosShape,
            AexisGraphSession.TensorRef sin,
            AexisGraphSession.BufferShape sinShape)
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
            AexisGraphModel.Layer layer,
            AexisGraphSession.CmdTensorRef src,
            AexisGraphSession.BufferShape srcShape,
            AexisGraphSession.CmdTensorRef cos,
            AexisGraphSession.BufferShape cosShape,
            AexisGraphSession.CmdTensorRef sin,
            AexisGraphSession.BufferShape sinShape)
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

        private static void ValidatePack4TextureArray(AexisGraphModel.Layer layer, string role, AexisGraphSession.TensorRef tensor, AexisGraphSession.BufferShape logicalShape)
        {
            if (tensor == null || tensor.texture == null)
                throw new InvalidOperationException("RotaryEmbed render-texture path requires texture " + role + " input: " + layer.name);
            if (AexisGraphSession.IsStrictLinearMatTexture(tensor) || tensor.texture.dimension != TextureDimension.Tex2DArray)
                throw new InvalidOperationException("RotaryEmbed render-texture path requires texture-array " + role + " input: " + layer.name + " | shape=" + FormatShape(logicalShape));
            var storageShape = AexisGraphSession.GetTextureStorageShape(tensor, logicalShape);
            if (!AexisGraphSession.BufferShapeEquals(logicalShape, storageShape))
                throw new InvalidOperationException(
                    "RotaryEmbed render-texture path requires matching logical/storage shape for " + role
                    + ": " + layer.name
                    + " | logical=" + FormatShape(logicalShape)
                    + " | storage=" + FormatShape(storageShape));
        }

        private static void ValidatePack4TextureArray(AexisGraphModel.Layer layer, string role, AexisGraphSession.CmdTensorRef tensor, AexisGraphSession.BufferShape logicalShape)
        {
            if (tensor == null || tensor.texture == null)
                throw new InvalidOperationException("RotaryEmbed command-buffer path requires texture " + role + " input: " + layer.name);
            if (AexisGraphSession.IsStrictLinearMatTexture(tensor) || tensor.texture.dimension != TextureDimension.Tex2DArray)
                throw new InvalidOperationException("RotaryEmbed command-buffer path requires texture-array " + role + " input: " + layer.name + " | shape=" + FormatShape(logicalShape));
            var storageShape = AexisGraphSession.GetCmdStorageShape(tensor, logicalShape);
            if (!AexisGraphSession.BufferShapeEquals(logicalShape, storageShape))
                throw new InvalidOperationException(
                    "RotaryEmbed command-buffer path requires matching logical/storage shape for " + role
                    + ": " + layer.name
                    + " | logical=" + FormatShape(logicalShape)
                    + " | storage=" + FormatShape(storageShape));
        }

        private static void ValidateCacheShape(AexisGraphModel.Layer layer, AexisGraphSession.BufferShape srcShape, AexisGraphSession.BufferShape cosShape, AexisGraphSession.BufferShape sinShape)
        {
            var required = (long)Mathf.Max(1, srcShape.h) * Mathf.Max(1, srcShape.w / 2);
            if (GetElementCount(cosShape) < required || GetElementCount(sinShape) < required)
                throw new InvalidOperationException(
                    "RotaryEmbed cache texture is smaller than seq_len * embed_dim/2: " + layer.name
                    + " | required=" + required
                    + " | cos=" + FormatShape(cosShape)
                    + " | sin=" + FormatShape(sinShape));
        }

        private static long GetElementCount(AexisGraphSession.BufferShape shape)
        {
            var w = Mathf.Max(1, shape.w);
            var h = shape.dims >= 2 ? Mathf.Max(1, shape.h) : 1;
            var d = shape.dims >= 4 ? Mathf.Max(1, shape.d) : 1;
            var c = shape.dims >= 3 ? Mathf.Max(1, shape.c) : 1;
            return (long)w * h * d * c;
        }

        private static string FormatShape(AexisGraphSession.BufferShape shape)
        {
            return "d" + shape.dims + ":" + shape.w + "x" + shape.h + "x" + shape.d + "x" + shape.c;
        }
    }
}
