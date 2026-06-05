using System;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnPReLULayerRepro : NcnnBaseLayerRepro
    {
        public NcnnPReLULayerRepro()
            : base(NcnnLayerTypes.PReLU, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            var pp = new NcnnRepro.PReluPack
            {
                numSlope = Mathf.Max(1, layer.GetInt(0, 0))
            };

            phaseSw.Restart();
            pp.slopeCpu = br.ReadNcnnMatAsFloat32(pp.numSlope, 0, 0, 0, 1);
            phaseSw.Stop();
            readMs += phaseSw.ElapsedMilliseconds;

            phaseSw.Restart();
            pp.slope = NcnnRepro.NewBuffer(pp.slopeCpu);
            phaseSw.Stop();
            uploadMs += phaseSw.ElapsedMilliseconds;

            owner._extraPacks[layer.name] = pp;
            return new NcnnRepro.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
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

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.PReluPack pp)
                throw new InvalidOperationException("PReLU pack not found: " + layer.name);

            if (pp.numSlope == 1
                && owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape))
            {
                var outRt = owner.RentTempArray(srcTex.width, srcTex.height, srcTex.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.LeakyReluPack4(srcTex.texture, pp.slopeCpu[0], srcTex.packs, outRt);
                NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("PReLU source not found: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c);
            owner.Ops.PReluBuf(srcBuf, srcView, pp.slope, pp.numSlope, outTensor.buffer);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: srcView.dims <= 3,
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
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.PReluPack pp)
                throw new InvalidOperationException("PReLU pack not found: " + layer.name);

            if (pp.numSlope != 1)
            {
                var cmdFallback = context.commandBuffer;
                var blobsFallback = context.blobs;
                var remainingFallback = context.remaining;
                var pinnedFallback = context.pinnedNames;
                var srcFallback = NcnnRepro.GetCmdTensor(blobsFallback, layer.bottomNames[0]);
                var outFallback = owner.RentTempArray(cmdFallback, srcFallback.width, srcFallback.height, srcFallback.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.CopyPack4(cmdFallback, srcFallback.texture, 0, outFallback, 0, srcFallback.packs);
                blobsFallback[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
                {
                    texture = outFallback,
                    width = srcFallback.width,
                    height = srcFallback.height,
                    packs = srcFallback.packs,
                    refs = 1,
                    owned = true
                };
                owner.ConsumeCmd(cmdFallback, blobsFallback, remainingFallback, layer.bottomNames, pinnedFallback);
                return;
            }

            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            var outArr = owner.RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
            owner.Ops.LeakyReluPack4(cmd, src.texture, pp.slopeCpu[0], src.packs, outArr);
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
