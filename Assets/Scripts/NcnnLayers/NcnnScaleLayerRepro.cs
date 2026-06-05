using System;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnScaleLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnScaleLayerRepro()
            : base(NcnnLayerTypes.Scale, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            var sp = new NcnnRepro.ScalePack
            {
                scaleDataSize = layer.GetInt(0, 0),
                dynamic = layer.GetInt(0, 0) == -233
            };
            sp.biasTerm = !sp.dynamic && layer.GetInt(1, 0) != 0;

            if (!sp.dynamic)
            {
                phaseSw.Restart();
                sp.scaleCpu = br.ReadNcnnMatAsFloat32(sp.scaleDataSize, 0, 0, 0, 1);
                if (sp.biasTerm)
                    sp.biasCpu = br.ReadNcnnMatAsFloat32(sp.scaleDataSize, 0, 0, 0, 1);
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                sp.scale = NcnnRepro.NewBuffer(sp.scaleCpu);
                if (sp.biasCpu != null)
                    sp.bias = NcnnRepro.NewBuffer(sp.biasCpu);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;
            }

            owner._extraPacks[layer.name] = sp;
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

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.ScalePack sp)
                throw new InvalidOperationException("Scale pack not found: " + layer.name);

            ComputeBuffer scaleBuf = null;
            ComputeBuffer biasBuf = null;
            var scaleCount = sp.scaleDataSize;
            var hasBias = sp.biasTerm;

            if (sp.dynamic)
            {
                if (layer.bottomNames.Length < 2)
                    throw new InvalidOperationException("Dynamic scale input missing: " + layer.name);
                scaleBuf = owner.GetOrConvertToBuffer(layer.bottomNames[1], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                if (scaleBuf == null)
                    throw new InvalidOperationException("Dynamic scale buffer missing: " + layer.name);
                var scaleView = NcnnRepro.TryGetBufferView(layer.bottomNames[1], bufferBlobs, bufferViews);
                scaleCount = scaleView != null ? scaleView.elementCount : scaleBuf.count;
                if (layer.bottomNames.Length > 2)
                {
                    biasBuf = owner.GetOrConvertToBuffer(layer.bottomNames[2], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                    hasBias = biasBuf != null;
                }
                else
                {
                    hasBias = false;
                }
            }
            else
            {
                scaleBuf = sp.scale;
                biasBuf = sp.bias;
            }

            if (!sp.dynamic
                && sp.scaleDataSize == 1
                && !sp.biasTerm
                && owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape))
            {
                var outRt = owner.RentTempArray(srcTex.width, srcTex.height, srcTex.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.ScalePack4(srcTex.texture, sp.scaleCpu[0], srcTex.packs, outRt);
                NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("Scale source not found: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c);
            owner.Ops.ScaleBuf(srcBuf, srcView, scaleBuf, scaleCount, hasBias, biasBuf, outTensor.buffer);
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
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.ScalePack sp)
                throw new InvalidOperationException("Scale pack not found: " + layer.name);

            if (!sp.dynamic && sp.scaleDataSize == 1 && !sp.biasTerm)
            {
                var cmd = context.commandBuffer;
                var blobs = context.blobs;
                var remaining = context.remaining;
                var pinnedNames = context.pinnedNames;
                var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
                var outArr = owner.RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.PointwisePack4(cmd, src.texture, src.packs, NcnnOps.PointwiseType.ScaleScalar, sp.scaleCpu[0], 0f, outArr);
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
                return;
            }

            var fallbackCmd = context.commandBuffer;
            var fallbackBlobs = context.blobs;
            var fallbackRemaining = context.remaining;
            var fallbackPinnedNames = context.pinnedNames;
            var fallbackSrc = NcnnRepro.GetCmdTensor(fallbackBlobs, layer.bottomNames[0]);
            var outArrFallback = owner.RentTempArray(fallbackCmd, fallbackSrc.width, fallbackSrc.height, fallbackSrc.packs, RenderTextureFormat.ARGBHalf);
            owner.Ops.CopyPack4(fallbackCmd, fallbackSrc.texture, 0, outArrFallback, 0, fallbackSrc.packs);
            fallbackBlobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
            {
                texture = outArrFallback,
                width = fallbackSrc.width,
                height = fallbackSrc.height,
                packs = fallbackSrc.packs,
                refs = 1,
                owned = true
            };
            owner.ConsumeCmd(fallbackCmd, fallbackBlobs, fallbackRemaining, layer.bottomNames, fallbackPinnedNames);
        }
    }
}
