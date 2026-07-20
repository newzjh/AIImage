using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Ncnn
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnPixelShuffleLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnPixelShuffleLayerRepro()
            : base(NcnnLayerTypes.PixelShuffle, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override NcnnGraphSession.LayerLoadMetrics LoadLayer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            return default;
        }

        public override void ExecuteBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferViews = context.bufferViews;

            var upscaleFactor = layer.GetInt(0, 1);
            if (upscaleFactor <= 0)
                throw new InvalidOperationException("PixelShuffle upscale_factor must be positive: " + layer.name);

            if (owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape)
                && CanUsePack4PixelShuffle(srcTex, srcShape, upscaleFactor))
            {
                ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null || srcView.dims != 3)
                throw new InvalidOperationException("PixelShuffle expects dims=3 source: " + layer.name);

            var upscaleFactor = layer.GetInt(0, 1);
            var mode = layer.GetInt(1, 0);
            if (upscaleFactor <= 0)
                throw new InvalidOperationException("PixelShuffle upscale_factor must be positive: " + layer.name);

            var divisor = upscaleFactor * upscaleFactor;
            if (srcView.c % divisor != 0)
                throw new InvalidOperationException("PixelShuffle channel count is not divisible by upscale_factor^2: " + layer.name);

            var outW = srcView.w * upscaleFactor;
            var outH = srcView.h * upscaleFactor;
            var outC = srcView.c / divisor;
            var outTensor = owner.RentTempTensorBuffer(3, outW, outH, 1, outC);
            owner.Ops.PixelShuffleBuf(srcBuf, srcView.w, srcView.h, srcView.c, upscaleFactor, mode, outTensor.buffer);
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

        public override void ExecuteRenderTexturePath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferViews = context.bufferViews;

            var upscaleFactor = layer.GetInt(0, 1);
            var mode = layer.GetInt(1, 0);
            if (upscaleFactor <= 0)
                throw new InvalidOperationException("PixelShuffle upscale_factor must be positive: " + layer.name);

            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape)
                || !CanUsePack4PixelShuffle(srcTex, srcShape, upscaleFactor))
            {
                throw new InvalidOperationException("PixelShuffle render-texture path requires supported pack4 input: " + layer.name);
            }

            if (srcShape.dims != 3)
                throw new InvalidOperationException("PixelShuffle expects dims=3 source: " + layer.name);

            var divisor = upscaleFactor * upscaleFactor;
            if (srcShape.c % divisor != 0)
                throw new InvalidOperationException("PixelShuffle channel count is not divisible by upscale_factor^2: " + layer.name);

            var outW = srcShape.w * upscaleFactor;
            var outH = srcShape.h * upscaleFactor;
            var outC = srcShape.c / divisor;
            var outPacks = Mathf.Max(1, Mathf.CeilToInt(outC / 4f));
            var outRt = owner.RentTempArray(outW, outH, outPacks, RenderTextureFormat.ARGBHalf);
            owner.Ops.PixelShufflePack4(srcTex.texture, outC, upscaleFactor, mode, outRt);
            NcnnGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, new NcnnGraphSession.BufferShape(3, outW, outH, 1, outC));
            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var srcShape = NcnnGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            if (srcShape.dims != 3)
                throw new InvalidOperationException("PixelShuffle expects dims=3 source: " + layer.name);

            var upscaleFactor = layer.GetInt(0, 1);
            if (upscaleFactor <= 0)
                throw new InvalidOperationException("PixelShuffle upscale_factor must be positive: " + layer.name);

            var divisor = upscaleFactor * upscaleFactor;
            var outShape = new NcnnGraphSession.BufferShape(
                3,
                srcShape.w * upscaleFactor,
                srcShape.h * upscaleFactor,
                1,
                Mathf.Max(1, srcShape.c / Mathf.Max(1, divisor)));

            var mode = layer.GetInt(1, 0);
            var src = NcnnGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            if (CanUsePack4PixelShuffle(src, srcShape, upscaleFactor))
            {
                var outArr = owner.RentTempArray(cmd, outShape.w, outShape.h, Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f)), RenderTextureFormat.ARGBHalf);
                owner.Ops.PixelShufflePack4(cmd, src.texture, outShape.c, upscaleFactor, mode, outArr);
                blobs[layer.topNames[0]] = new NcnnGraphSession.CmdTensorRef
                {
                    texture = outArr,
                    width = outShape.w,
                    height = outShape.h,
                    packs = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f)),
                    refs = 1,
                    owned = true
                };
                if (shapes != null)
                    shapes[layer.topNames[0]] = outShape;
            }
            else
            {
                owner.PublishCmdPlaceholder(cmd, layer.topNames[0], outShape, blobs, shapes);
            }
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool CanUsePack4PixelShuffle(NcnnGraphSession.TensorRef src, NcnnGraphSession.BufferShape srcShape, int upscaleFactor)
        {
            return src != null
                && src.texture != null
                && srcShape.dims == 3
                && srcShape.d == 1
                && srcShape.w == src.width
                && srcShape.h == src.height
                && upscaleFactor > 0
                && srcShape.c > 0
                && srcShape.c <= src.packs * 4
                && (srcShape.c % (upscaleFactor * upscaleFactor)) == 0;
        }

        private static bool CanUsePack4PixelShuffle(NcnnGraphSession.CmdTensorRef src, NcnnGraphSession.BufferShape srcShape, int upscaleFactor)
        {
            return src != null
                && src.texture != null
                && srcShape.dims == 3
                && srcShape.d == 1
                && srcShape.w == src.width
                && srcShape.h == src.height
                && upscaleFactor > 0
                && srcShape.c > 0
                && srcShape.c <= src.packs * 4
                && (srcShape.c % (upscaleFactor * upscaleFactor)) == 0;
        }
    }
}
