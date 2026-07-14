using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnCastLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnCastLayerRepro()
            : base(NcnnLayerTypes.Cast, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
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
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            var typeFrom = layer.GetInt(0, 0);
            var typeTo = layer.GetInt(1, 0);
            if (typeFrom == typeTo)
            {
                new NcnnNoopLayerRepro().ExecuteBuffer(owner, layer, context);
                return;
            }

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("Cast source not found: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c);
            owner.Ops.CastBuf(srcBuf, srcView.elementCount, typeFrom, typeTo, outTensor.buffer);
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
            var typeFrom = layer.GetInt(0, 0);
            var typeTo = layer.GetInt(1, 0);
            if (IsTextureNoopCast(typeFrom, typeTo))
            {
                new NcnnNoopLayerRepro().ExecuteBuffer(owner, layer, context);
                return;
            }

            NcnnPack4LayerHelpers.ExecuteShapePreservingRenderTexture(
                owner,
                layer,
                context,
                "Cast",
                (input, shape, output) => owner.Ops.CastPack4(input, shape, typeFrom, typeTo, output));
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var typeFrom = layer.GetInt(0, 0);
            var typeTo = layer.GetInt(1, 0);
            if (IsTextureNoopCast(typeFrom, typeTo))
            {
                new NcnnNoopLayerRepro().ExecuteCommandBuffer(owner, layer, context);
                return;
            }

            NcnnPack4LayerHelpers.ExecuteShapePreservingCommandBuffer(
                owner,
                layer,
                context,
                "Cast",
                (cmd, input, shape, output) => owner.Ops.CastPack4(cmd, input, shape, typeFrom, typeTo, output));
        }

        private static bool IsTextureNoopCast(int typeFrom, int typeTo)
        {
            return typeFrom == typeTo;
        }
    }
}
