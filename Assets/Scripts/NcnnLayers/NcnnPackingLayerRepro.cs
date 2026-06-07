using System;
using UnityEngine;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnPackingLayerRepro : NcnnBaseLayerRepro
    {
        private sealed class PackingPack : IDisposable
        {
            public int outElemPack;
            public bool usePadding;
            public int castTypeFrom;
            public int castTypeTo;

            public void Dispose()
            {
            }
        }

        public NcnnPackingLayerRepro()
            : base(NcnnLayerTypes.Packing, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            owner._extraPacks[layer.name] = new PackingPack
            {
                outElemPack = layer.GetInt(0, 1),
                usePadding = layer.GetInt(1, 0) != 0,
                castTypeFrom = layer.GetInt(2, 0),
                castTypeTo = layer.GetInt(3, 0)
            };
            return default;
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not PackingPack pack)
                throw new InvalidOperationException("Packing pack not found: " + layer.name);

            var requiresCast = pack.castTypeTo != 0 && pack.castTypeFrom != pack.castTypeTo;
            if (!requiresCast)
            {
                new NcnnNoopLayerRepro().ExecuteBuffer(owner, layer, context);
                return;
            }

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
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("Packing source not found: " + layer.name);

            // This repo stores logical unpacked CHW/CHWD floats in ComputeBuffer.
            // ncnn Packing's elempack conversion therefore becomes metadata/no-op here,
            // except for cast requests that affect value rounding.
            var outTensor = owner.RentTempTensorBuffer(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c);
            owner.Ops.CopyBuf(srcBuf, outTensor.buffer, outTensor.elementCount);
            if (requiresCast)
                owner.Ops.CastBuf(outTensor.buffer, outTensor.elementCount, pack.castTypeFrom, pack.castTypeTo, outTensor.buffer);
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
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not PackingPack pack)
                throw new InvalidOperationException("Packing pack not found: " + layer.name);

            var requiresCast = pack.castTypeTo != 0 && pack.castTypeFrom != pack.castTypeTo;
            if (!requiresCast)
            {
                new NcnnNoopLayerRepro().ExecuteCommandBuffer(owner, layer, context);
                return;
            }

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
