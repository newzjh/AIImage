using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisShuffleChannelLayer : AexisBaseLayer
    {
        public AexisShuffleChannelLayer() : base(AexisLayerTypes.ShuffleChannel, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var group = Mathf.Max(1, layer.GetInt(0, 1));
            var reverse = layer.GetInt(1, 0) != 0;

            if (owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out var srcTex,
                    out var srcShape)
                && CanUsePack4Path(srcTex, srcShape, group, reverse, out _))
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

            var group = Mathf.Max(1, layer.GetInt(0, 1));
            var reverse = layer.GetInt(1, 0) != 0;
            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("ShuffleChannel source not found: " + layer.name);

            var outBuf = owner.ShuffleChannelCpu(srcBuf, srcView, group, reverse);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outBuf,
                preferTexture: outBuf.dims <= 3,
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
            var bufferBlobs = context.bufferBlobs;
            var bufferViews = context.bufferViews;

            var group = Mathf.Max(1, layer.GetInt(0, 1));
            var reverse = layer.GetInt(1, 0) != 0;
            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape)
                || !CanUsePack4Path(srcTex, srcShape, group, reverse, out var actualGroup))
            {
                throw new InvalidOperationException("ShuffleChannel render-texture path requires supported pack4 input: " + layer.name);
            }

            var outRt = owner.RentTempArray(srcTex.width, srcTex.height, srcTex.packs, RenderTextureFormat.ARGBHalf);
            owner.Ops.ShuffleChannelPack4(srcTex.texture, srcTex.packs, srcShape.c, actualGroup, outRt);
            AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var group = Mathf.Max(1, layer.GetInt(0, 1));
            var reverse = layer.GetInt(1, 0) != 0;
            if (!CanUsePack4Path(src, srcShape, group, reverse, out var actualGroup))
            {
                owner.CopyCmdTensor(cmd, src, layer.topNames[0], blobs, src.width, src.height, src.packs, shapes, srcShape);
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            var outArr = owner.RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
            owner.Ops.ShuffleChannelPack4(cmd, src.texture, src.packs, srcShape.c, actualGroup, outArr);
            blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
            {
                texture = outArr,
                width = src.width,
                height = src.height,
                packs = src.packs,
                refs = 1,
                owned = true
            };
            shapes[layer.topNames[0]] = srcShape;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool CanUsePack4Path(AexisGraphSession.TensorRef srcTex, AexisGraphSession.BufferShape srcShape, int group, bool reverse, out int actualGroup)
        {
            actualGroup = 0;
            if (srcTex == null || srcTex.texture == null)
                return false;
            return CanUsePack4Path(srcTex.width, srcTex.height, srcTex.packs, srcShape, group, reverse, out actualGroup);
        }

        private static bool CanUsePack4Path(AexisGraphSession.CmdTensorRef src, AexisGraphSession.BufferShape srcShape, int group, bool reverse, out int actualGroup)
        {
            actualGroup = 0;
            if (src == null || src.texture == null)
                return false;
            return CanUsePack4Path(src.width, src.height, src.packs, srcShape, group, reverse, out actualGroup);
        }

        private static bool CanUsePack4Path(int width, int height, int packs, AexisGraphSession.BufferShape srcShape, int group, bool reverse, out int actualGroup)
        {
            actualGroup = 0;
            if (srcShape.dims != 3)
                return false;
            if (srcShape.w != width || srcShape.h != height)
                return false;
            if (srcShape.c <= 0 || srcShape.c > packs * 4)
                return false;
            if (group <= 0 || srcShape.c % group != 0)
                return false;

            actualGroup = reverse ? srcShape.c / group : group;
            return actualGroup > 0 && srcShape.c % actualGroup == 0;
        }
    }
}
