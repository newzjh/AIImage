using System;
using System.Diagnostics;
using UnityEngine;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnRequantizeLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnRequantizeLayerRepro()
            : base(NcnnLayerTypes.Requantize, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            var rp = new NcnnRepro.RequantizePack
            {
                scaleInDataSize = layer.GetInt(0, 1),
                scaleOutDataSize = layer.GetInt(1, 1),
                biasDataSize = layer.GetInt(2, 0),
                activationType = layer.GetInt(3, 0)
            };

            var activationParams = NcnnRepro.ParseActivationParams(layer);
            rp.activationParam0 = activationParams.Length > 0 ? activationParams[0] : 0f;
            rp.activationParam1 = activationParams.Length > 1 ? activationParams[1] : 0f;

            phaseSw.Restart();
            rp.scaleInCpu = br.ReadNcnnMatAsFloat32(rp.scaleInDataSize, 0, 0, 0, 1);
            rp.scaleOutCpu = br.ReadNcnnMatAsFloat32(rp.scaleOutDataSize, 0, 0, 0, 1);
            if (rp.biasDataSize > 0)
                rp.biasCpu = br.ReadNcnnMatAsFloat32(rp.biasDataSize, 0, 0, 0, 1);
            phaseSw.Stop();
            readMs += phaseSw.ElapsedMilliseconds;

            phaseSw.Restart();
            rp.scaleIn = NcnnRepro.NewBuffer(rp.scaleInCpu);
            rp.scaleOut = NcnnRepro.NewBuffer(rp.scaleOutCpu);
            if (rp.biasCpu != null)
                rp.bias = NcnnRepro.NewBuffer(rp.biasCpu);
            phaseSw.Stop();
            uploadMs += phaseSw.ElapsedMilliseconds;

            owner._extraPacks[layer.name] = rp;
            return new NcnnRepro.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
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

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.RequantizePack rp)
                throw new InvalidOperationException("Requantize pack not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("Requantize source not found: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c);
            owner.Ops.RequantizeBuf(
                srcBuf,
                srcView,
                rp.scaleIn,
                rp.scaleInDataSize,
                rp.scaleOut,
                rp.scaleOutDataSize,
                rp.bias,
                rp.biasDataSize,
                rp.activationType,
                rp.activationParam0,
                rp.activationParam1,
                outTensor.buffer);
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

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            owner.PublishCmdPlaceholder(cmd, layer.topNames[0], srcShape, blobs, shapes);
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }
    }
}
