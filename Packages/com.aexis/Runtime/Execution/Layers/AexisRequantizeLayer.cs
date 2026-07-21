using System;
using System.Diagnostics;
using UnityEngine;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisRequantizeLayer : AexisBaseLayer
    {
        public AexisRequantizeLayer()
            : base(AexisLayerTypes.Requantize, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            var rp = new AexisGraphSession.RequantizePack
            {
                scaleInDataSize = layer.GetInt(0, 1),
                scaleOutDataSize = layer.GetInt(1, 1),
                biasDataSize = layer.GetInt(2, 0),
                activationType = layer.GetInt(3, 0)
            };

            var activationParams = AexisGraphSession.ParseActivationParams(layer);
            rp.activationParam0 = activationParams.Length > 0 ? activationParams[0] : 0f;
            rp.activationParam1 = activationParams.Length > 1 ? activationParams[1] : 0f;

            phaseSw.Restart();
            rp.scaleInCpu = br.ReadTensorAsFloat32(rp.scaleInDataSize, 0, 0, 0, 1);
            rp.scaleOutCpu = br.ReadTensorAsFloat32(rp.scaleOutDataSize, 0, 0, 0, 1);
            if (rp.biasDataSize > 0)
                rp.biasCpu = br.ReadTensorAsFloat32(rp.biasDataSize, 0, 0, 0, 1);
            phaseSw.Stop();
            readMs += phaseSw.ElapsedMilliseconds;

            phaseSw.Restart();
            rp.scaleIn = AexisGraphSession.NewBuffer(rp.scaleInCpu);
            rp.scaleOut = AexisGraphSession.NewBuffer(rp.scaleOutCpu);
            if (rp.biasCpu != null)
                rp.bias = AexisGraphSession.NewBuffer(rp.biasCpu);
            phaseSw.Stop();
            uploadMs += phaseSw.ElapsedMilliseconds;

            owner._extraPacks[layer.name] = rp;
            return new AexisGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out _,
                    out _))
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

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.RequantizePack rp)
                throw new InvalidOperationException("Requantize pack not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
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

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.RequantizePack rp)
                throw new InvalidOperationException("Requantize pack not found: " + layer.name);

            AexisPack4LayerHelpers.ExecuteShapePreservingRenderTexture(
                owner,
                layer,
                context,
                "Requantize",
                (input, shape, output) => owner.Ops.RequantizePack4(
                    input,
                    shape,
                    rp.scaleIn,
                    rp.scaleInDataSize,
                    rp.scaleOut,
                    rp.scaleOutDataSize,
                    rp.bias,
                    rp.biasDataSize,
                    rp.activationType,
                    rp.activationParam0,
                    rp.activationParam1,
                    output));
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.RequantizePack rp)
                throw new InvalidOperationException("Requantize pack not found: " + layer.name);

            AexisPack4LayerHelpers.ExecuteShapePreservingCommandBuffer(
                owner,
                layer,
                context,
                "Requantize",
                (cmd, input, shape, output) => owner.Ops.RequantizePack4(
                    cmd,
                    input,
                    shape,
                    rp.scaleIn,
                    rp.scaleInDataSize,
                    rp.scaleOut,
                    rp.scaleOutDataSize,
                    rp.bias,
                    rp.biasDataSize,
                    rp.activationType,
                    rp.activationParam0,
                    rp.activationParam1,
                    output));
        }
    }
}
