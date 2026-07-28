using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    public sealed class AexisMvnLayer : AexisBaseLayer
    {
        private sealed class Pack : IDisposable
        {
            public ComputeBuffer dummyAffine;
            public void Dispose() { try { dummyAffine?.Dispose(); } catch { } }
        }

        public AexisMvnLayer() : base(AexisLayerTypes.MVN, false, true) { }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
            owner._extraPacks[layer.name] = new Pack { dummyAffine = AexisGraphSession.NewBuffer(new[] { 1f }) };
            return default;
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (!owner.TryGetPack4Texture(layer.bottomNames[0], context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, out var src, out var shape))
                throw new InvalidOperationException("MVN requires an existing Pack4 texture input: " + layer.name);
            var pack = GetPack(owner, layer);
            Validate(shape, src.width, src.height, src.packs, layer);
            var groups = layer.GetInt(1, 0) != 0 ? 1 : shape.c;
            var depth = shape.dims == 4 ? shape.d : 1;
            var statsA = owner.RentTempArray(groups, 1, 1, RenderTextureFormat.ARGBFloat);
            var statsB = owner.RentTempArray(groups, 1, 1, RenderTextureFormat.ARGBFloat);
            var output = owner.RentTempArray(src.width, src.height, depth * src.packs, src.texture.format);
            try
            {
                owner.Ops.GroupNormPack4Tex(src.texture, shape.w, shape.h, depth, shape.c, src.packs, groups, layer.GetFloat(2, 0.0001f), pack.dummyAffine, pack.dummyAffine, statsA, statsB, output, true, layer.GetInt(0, 0) != 0);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, shape, shape);
                output = null;
            }
            finally
            {
                owner.ReturnTempArray(statsA);
                owner.ReturnTempArray(statsB);
                if (output != null) owner.ReturnTempArray(output);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var src = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var shape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            var pack = GetPack(owner, layer);
            Validate(shape, src.width, src.height, src.packs, layer);
            var groups = layer.GetInt(1, 0) != 0 ? 1 : shape.c;
            var depth = shape.dims == 4 ? shape.d : 1;
            var statsA = owner.RentTempArrayExactFormat(
                context.commandBuffer,
                groups,
                1,
                1,
                RenderTextureFormat.ARGBFloat,
                AexisGraphSession.GetStrictPack4ScratchIdentity(layer, "mvn-stats-a"));
            var statsB = owner.RentTempArrayExactFormat(
                context.commandBuffer,
                groups,
                1,
                1,
                RenderTextureFormat.ARGBFloat,
                AexisGraphSession.GetStrictPack4ScratchIdentity(layer, "mvn-stats-b"));
            var output = owner.RentTempArray(context.commandBuffer, src.width, src.height, depth * src.packs, src.texture.format);
            owner.Ops.GroupNormPack4Tex(context.commandBuffer, src.texture, shape.w, shape.h, depth, shape.c, src.packs, groups, layer.GetFloat(2, 0.0001f), pack.dummyAffine, pack.dummyAffine, statsA, statsB, output, true, layer.GetInt(0, 0) != 0);
            owner.ReturnTempArray(context.commandBuffer, statsA);
            owner.ReturnTempArray(context.commandBuffer, statsB);
            context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, shape, shape, true);
            if (context.shapes != null) context.shapes[layer.topNames[0]] = shape;
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static Pack GetPack(AexisGraphSession owner, AexisGraphModel.Layer layer)
        {
            if (!owner._extraPacks.TryGetValue(layer.name, out var value) || value is not Pack pack || pack.dummyAffine == null)
                throw new InvalidOperationException("MVN immutable dispatch constants are not loaded: " + layer.name);
            return pack;
        }

        private static void Validate(AexisGraphSession.BufferShape shape, int width, int height, int packs, AexisGraphModel.Layer layer)
        {
            if ((shape.dims != 3 && shape.dims != 4) || shape.w != width || shape.h != height || shape.c <= 0 || packs != (shape.c + 3) / 4)
                throw new InvalidOperationException("MVN requires matching dims=3/4 Pack4 logical and storage shapes: " + layer.name);
            if (layer.GetFloat(2, 0.0001f) < 0f)
                throw new InvalidOperationException("MVN epsilon must be non-negative: " + layer.name);
            if ((layer.GetInt(0, 0) != 0 && layer.GetInt(0, 0) != 1)
                || (layer.GetInt(1, 0) != 0 && layer.GetInt(1, 0) != 1))
                throw new InvalidOperationException("MVN normalize_variance and across_channels must be 0 or 1: " + layer.name);
        }
    }
}
