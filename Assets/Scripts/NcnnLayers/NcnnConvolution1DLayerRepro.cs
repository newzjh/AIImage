using System;
using System.Diagnostics;
using UnityEngine;

namespace NcnnCompute
{
    public sealed class NcnnConvolution1DLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnConvolution1DLayerRepro()
            : base(NcnnLayerTypes.Convolution1D, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            var pack = new NcnnRepro.ConvPack
            {
                outC = layer.GetInt(0, 0),
                kernelW = layer.GetInt(1, 0),
                kernelH = 1,
                dilationW = layer.GetInt(2, 1),
                dilationH = 1,
                strideW = layer.GetInt(3, 1),
                strideH = 1,
                padLeft = layer.GetInt(4, 0),
                padRight = layer.GetInt(15, layer.GetInt(4, 0)),
                padTop = 0,
                padBottom = 0,
                biasTerm = layer.GetInt(5, 0),
                weightSize = layer.GetInt(6, 0),
                activationType = layer.GetInt(9, 0),
                group = 1,
                useBufferPath = true
            };

            var actParams = NcnnRepro.ParseActivationParams(layer);
            pack.activationSlope = actParams.Length > 0 ? actParams[0] : NcnnRepro.ParseLeakySlope(layer);
            pack.inC = Mathf.Max(1, pack.weightSize / Mathf.Max(1, pack.outC * pack.kernelW));

            phaseSw.Restart();
            var w = NcnnRepro.ReadPackedOrRawWeightArray(br, pack.weightSize, layer.name);
            var b = pack.biasTerm != 0 ? br.ReadFloat32Array(pack.outC) : new float[pack.outC];
            phaseSw.Stop();
            readMs += phaseSw.ElapsedMilliseconds;

            phaseSw.Restart();
            pack.rawWeight = NcnnRepro.NewBuffer(w);
            pack.rawBias = NcnnRepro.NewBuffer(b);
            phaseSw.Stop();
            uploadMs += phaseSw.ElapsedMilliseconds;

            owner._extraPacks[layer.name] = pack;
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

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.ConvPack conv)
                throw new InvalidOperationException("Convolution1D pack not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null || srcView.dims != 2)
                throw new InvalidOperationException("Convolution1D expects dims=2 tensor input: " + layer.name);

            var inputW = srcView.w;
            var inputC = srcView.h;
            var outW = NcnnRepro.ComputeConvOut(inputW, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
            var outTensor = owner.RentTempTensorBuffer(2, outW, conv.outC);
            var actParams = NcnnRepro.ParseActivationParams(layer);
            var param0 = actParams.Length > 0 ? actParams[0] : conv.activationSlope;
            owner.Ops.Conv1dBuf(
                srcBuf,
                conv.rawWeight,
                conv.rawBias,
                inputW,
                inputC,
                outW,
                conv.outC,
                conv.kernelW,
                conv.strideW,
                conv.dilationW,
                conv.padLeft,
                conv.activationType,
                param0,
                outTensor.buffer);
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
