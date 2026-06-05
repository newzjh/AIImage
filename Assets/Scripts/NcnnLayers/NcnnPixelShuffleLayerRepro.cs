using System;
using UnityEngine;

namespace NcnnCompute
{
    public sealed class NcnnPixelShuffleLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnPixelShuffleLayerRepro()
            : base(NcnnLayerTypes.PixelShuffle, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            return default;
        }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
            var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
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

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            var outArr = owner.RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
            owner.Ops.CopyPack4(cmd, src.texture, 0, outArr, 0, src.packs);
            blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
            {
                texture = outArr,
                width = src.width,
                height = src.height,
                packs = src.packs,
                refs = 1,
                owned = true
            };
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
        }
    }
}
