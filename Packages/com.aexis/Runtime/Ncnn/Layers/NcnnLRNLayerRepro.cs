using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Ncnn
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnLRNLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnLRNLayerRepro()
            : base(NcnnLayerTypes.LRN, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override NcnnGraphSession.LayerLoadMetrics LoadLayer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            owner._extraPacks[layer.name] = new NcnnGraphSession.LrnPack
            {
                regionType = layer.GetInt(0, 0),
                localSize = layer.GetInt(1, 5),
                alpha = layer.GetFloat(2, 1f),
                beta = layer.GetFloat(3, 0.75f),
                bias = layer.GetFloat(4, 1f)
            };
            return default;
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

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnGraphSession.LrnPack lp)
                throw new InvalidOperationException("LRN pack not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null || srcView.dims != 3)
                throw new InvalidOperationException("LRN expects dims=3 source: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c);
            owner.Ops.LrnBuf(srcBuf, srcView.w, srcView.h, srcView.c, lp.regionType, lp.localSize, lp.alpha, lp.beta, lp.bias, outTensor.buffer);
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
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnGraphSession.LrnPack lp)
                throw new InvalidOperationException("LRN pack not found: " + layer.name);

            NcnnPack4LayerHelpers.ExecuteShapePreservingRenderTexture(
                owner,
                layer,
                context,
                "LRN",
                (input, shape, output) =>
                {
                    if (shape.dims != 3)
                        throw new InvalidOperationException("LRN pack4 render-texture path expects dims=3 source: " + layer.name);
                    owner.Ops.LrnPack4(input, shape.w, shape.h, shape.c, lp.regionType, lp.localSize, lp.alpha, lp.beta, lp.bias, output);
                });
        }

        public override void ExecuteCommandBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnGraphSession.LrnPack lp)
                throw new InvalidOperationException("LRN pack not found: " + layer.name);

            NcnnPack4LayerHelpers.ExecuteShapePreservingCommandBuffer(
                owner,
                layer,
                context,
                "LRN",
                (cmd, input, shape, output) =>
                {
                    if (shape.dims != 3)
                        throw new InvalidOperationException("LRN pack4 command-buffer path expects dims=3 source: " + layer.name);
                    owner.Ops.LrnPack4(cmd, input, shape.w, shape.h, shape.c, lp.regionType, lp.localSize, lp.alpha, lp.beta, lp.bias, output);
                });
        }
    }
}
