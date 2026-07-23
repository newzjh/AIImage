using System;
using System.Collections.Generic;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    public sealed class AexisFlipLayer : AexisBaseLayer
    {
        public AexisFlipLayer() : base(AexisLayerTypes.Flip, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (owner == null) throw new ArgumentNullException(nameof(owner));
            if (layer == null) throw new ArgumentNullException(nameof(layer));
            if (context == null) throw new ArgumentNullException(nameof(context));
            if (!AexisGraphSession.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var source, out var shape))
                throw new InvalidOperationException("Flip render-texture path requires a texture input: " + layer.name);

            ResolveFlags(layer, shape, out var flipWidth, out var flipHeight, out var flipDepth, out var flipChannels);
            if (!flipWidth && !flipHeight && !flipDepth && !flipChannels)
            {
                new AexisNoopLayer().ExecuteRenderTexturePath(owner, layer, context);
                return;
            }
            RequirePack4(shape, source.packs, source.texture.dimension, source.texture.volumeDepth, layer.name);
            var output = owner.RentTempArray(shape.w, shape.h, shape.d * source.packs, owner.ResolveActivationTextureFormat(shape.dims));
            try
            {
                owner.Ops.FlipPack4(source.texture, shape.w, shape.h, shape.d, shape.c, flipWidth, flipHeight, flipDepth, flipChannels, output);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, shape);
                output = null;
            }
            finally
            {
                if (output != null)
                    owner.ReturnTempArray(output);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (owner == null) throw new ArgumentNullException(nameof(owner));
            if (layer == null) throw new ArgumentNullException(nameof(layer));
            if (context == null) throw new ArgumentNullException(nameof(context));
            var source = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var shape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            ResolveFlags(layer, shape, out var flipWidth, out var flipHeight, out var flipDepth, out var flipChannels);
            if (!flipWidth && !flipHeight && !flipDepth && !flipChannels)
            {
                context.blobs[layer.topNames[0]] = source;
                if (context.shapes != null)
                    context.shapes[layer.topNames[0]] = shape;
                owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
                return;
            }
            RequirePack4(shape, source.packs, source.texture.dimension, source.texture.depth, layer.name);
            var output = owner.RentTempArray(context.commandBuffer, shape.w, shape.h, shape.d * source.packs, owner.ResolveActivationTextureFormat(shape.dims));
            try
            {
                owner.Ops.FlipPack4(context.commandBuffer, source.texture, shape.w, shape.h, shape.d, shape.c, flipWidth, flipHeight, flipDepth, flipChannels, output);
                context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, shape, shape, owned: true);
                output = null;
                if (context.shapes != null)
                    context.shapes[layer.topNames[0]] = shape;
            }
            finally
            {
                if (output != null)
                    owner.ReturnTempArray(context.commandBuffer, output);
            }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static void ResolveFlags(AexisGraphModel.Layer layer, AexisGraphSession.BufferShape shape, out bool width, out bool height, out bool depth, out bool channels)
        {
            width = height = depth = channels = false;
            if (shape.dims != 3 && shape.dims != 4)
                throw new InvalidOperationException("Flip Pack4 supports rank-3 or rank-4 tensors only: " + layer.name);

            foreach (var axis in ReadAxes(layer))
            {
                var positive = axis < 0 ? axis + shape.dims : axis;
                if (positive < 0 || positive >= shape.dims)
                    throw new InvalidOperationException("Flip axis is outside tensor rank: " + layer.name);
                if (shape.dims == 3)
                {
                    if (positive == 0) channels = true;
                    else if (positive == 1) height = true;
                    else width = true;
                }
                else
                {
                    if (positive == 0) channels = true;
                    else if (positive == 1) depth = true;
                    else if (positive == 2) height = true;
                    else width = true;
                }
            }
        }

        // Strict planning invokes this before any texture allocation. The execution
        // methods still resolve the flags themselves because the layer is public and
        // can be called outside graph planning.
        internal static void ValidatePack4Profile(AexisGraphModel.Layer layer, AexisGraphSession.BufferShape shape)
        {
            ResolveFlags(layer, shape, out _, out _, out _, out _);
        }

        private static int[] ReadAxes(AexisGraphModel.Layer layer)
        {
            var raw = layer.GetString(0, string.Empty);
            if (string.IsNullOrWhiteSpace(raw))
                return Array.Empty<int>();
            var parts = raw.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
            var start = 0;
            if (parts.Length > 1 && int.TryParse(parts[0], NumberStyles.Integer, CultureInfo.InvariantCulture, out var encodedCount)
                && encodedCount <= -23300 && -23300 - encodedCount == parts.Length - 1)
            {
                start = 1;
            }
            var axes = new List<int>(parts.Length - start);
            for (var index = start; index < parts.Length; index++)
            {
                if (!int.TryParse(parts[index], NumberStyles.Integer, CultureInfo.InvariantCulture, out var axis))
                    throw new InvalidOperationException("Flip axes parameter is not an integer: " + layer.name);
                if (!axes.Contains(axis))
                    axes.Add(axis);
            }
            return axes.ToArray();
        }

        private static void RequirePack4(AexisGraphSession.BufferShape shape, int packs, TextureDimension dimension, int textureDepth, string layerName)
        {
            if (dimension != TextureDimension.Tex2DArray
                || packs != Mathf.CeilToInt(shape.c / 4f)
                || textureDepth < shape.d * packs)
            {
                throw new InvalidOperationException("Flip requires exact Pack4 Texture2DArray storage: " + layerName);
            }
        }
    }
}
