using System;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Ncnn
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnQuantizeLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnQuantizeLayerRepro()
            : base(NcnnLayerTypes.Quantize, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override NcnnGraphSession.LayerLoadMetrics LoadLayer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            var qp = new NcnnGraphSession.QuantizePack
            {
                scaleDataSize = layer.GetInt(0, 1)
            };

            phaseSw.Restart();
            qp.scaleCpu = br.ReadNcnnMatAsFloat32(qp.scaleDataSize, 0, 0, 0, 1);
            phaseSw.Stop();
            readMs += phaseSw.ElapsedMilliseconds;

            phaseSw.Restart();
            qp.scale = NcnnGraphSession.NewBuffer(qp.scaleCpu);
            phaseSw.Stop();
            uploadMs += phaseSw.ElapsedMilliseconds;

            owner._extraPacks[layer.name] = qp;
            return new NcnnGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        public override void ExecuteBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnGraphSession.QuantizePack qp)
                throw new InvalidOperationException("Quantize pack not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("Quantize source not found: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c);
            owner.Ops.QuantizeBuf(srcBuf, srcView, qp.scale, qp.scaleDataSize, outTensor.buffer);
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

        public override void ExecuteRenderTexturePath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnGraphSession.QuantizePack qp)
                throw new InvalidOperationException("Quantize pack not found: " + layer.name);

            NcnnPack4LayerHelpers.ExecuteShapePreservingRenderTexture(
                owner,
                layer,
                context,
                "Quantize",
                (input, shape, output) => owner.Ops.QuantizePack4(input, shape, qp.scale, qp.scaleDataSize, output));
        }

        public override void ExecuteCommandBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnGraphSession.QuantizePack qp)
                throw new InvalidOperationException("Quantize pack not found: " + layer.name);

            NcnnPack4LayerHelpers.ExecuteShapePreservingCommandBuffer(
                owner,
                layer,
                context,
                "Quantize",
                (cmd, input, shape, output) => owner.Ops.QuantizePack4(cmd, input, shape, qp.scale, qp.scaleDataSize, output));
        }
    }
}
