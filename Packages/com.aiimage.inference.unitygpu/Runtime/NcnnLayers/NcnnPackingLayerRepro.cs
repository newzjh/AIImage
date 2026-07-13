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

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not PackingPack pack)
                throw new InvalidOperationException("Packing pack not found: " + layer.name);

            var requiresCast = pack.castTypeTo != 0 && pack.castTypeFrom != pack.castTypeTo;
            if (!requiresCast)
            {
                new NcnnNoopLayerRepro().ExecuteBuffer(owner, layer, context);
                return;
            }

            NcnnPack4LayerHelpers.ExecuteShapePreservingRenderTexture(
                owner,
                layer,
                context,
                "Packing",
                (input, shape, output) => owner.Ops.CastPack4(input, shape, pack.castTypeFrom, pack.castTypeTo, output));
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

            NcnnPack4LayerHelpers.ExecuteShapePreservingCommandBuffer(
                owner,
                layer,
                context,
                "Packing",
                (cmd, input, shape, output) => owner.Ops.CastPack4(cmd, input, shape, pack.castTypeFrom, pack.castTypeTo, output));
        }
    }
}
