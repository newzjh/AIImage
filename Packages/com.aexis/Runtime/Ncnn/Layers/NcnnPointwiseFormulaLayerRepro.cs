using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Ncnn
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnPointwiseFormulaLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnPointwiseFormulaLayerRepro(NcnnLayerTypeKey typeKey)
            : base(typeKey, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
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

            ResolveFormula(TypeKey, layer, out var type, out var a, out var b);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null)
                throw new InvalidOperationException(TypeKey + " source not found: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(
                srcView?.dims ?? 1,
                srcView?.w ?? srcBuf.count,
                srcView?.h ?? 1,
                srcView?.d ?? 1,
                srcView?.c ?? 1);
            owner.Ops.PointwiseBuf(srcBuf, srcBuf.count, type, a, b, outTensor.buffer);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: srcView != null && srcView.dims <= 3,
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
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferViews = context.bufferViews;

            ResolveFormula(TypeKey, layer, out var type, out var a, out var b);
            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape))
                throw new InvalidOperationException(TypeKey + " render-texture path requires pack4 texture input: " + layer.name);

            var outRt = owner.RentTempArray(srcTex.width, srcTex.height, srcTex.packs, RenderTextureFormat.ARGBHalf);
            owner.Ops.PointwisePack4(srcTex.texture, srcTex.packs, type, a, b, outRt);
            NcnnGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            ResolveFormula(TypeKey, layer, out var type, out var a, out var b);

            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = NcnnGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = NcnnGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var outArr = owner.RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
            owner.Ops.PointwisePack4(cmd, src.texture, src.packs, type, a, b, outArr);
            blobs[layer.topNames[0]] = new NcnnGraphSession.CmdTensorRef
            {
                texture = outArr,
                width = src.width,
                height = src.height,
                packs = src.packs,
                refs = 1,
                owned = true
            };
            if (shapes != null)
                shapes[layer.topNames[0]] = srcShape;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static void ResolveFormula(
            NcnnLayerTypeKey typeKey,
            NcnnParamModel.Layer layer,
            out NcnnOps.PointwiseType pointwiseType,
            out float a,
            out float b)
        {
            a = 0f;
            b = 0f;

            if (typeKey == NcnnLayerTypes.ELU)
            {
                pointwiseType = NcnnOps.PointwiseType.Elu;
                a = layer.GetFloat(0, 0.1f);
                return;
            }

            if (typeKey == NcnnLayerTypes.Erf)
            {
                pointwiseType = NcnnOps.PointwiseType.Erf;
                return;
            }

            if (typeKey == NcnnLayerTypes.HardSigmoid)
            {
                pointwiseType = NcnnOps.PointwiseType.HardSigmoid;
                a = layer.GetFloat(0, 0.2f);
                b = layer.GetFloat(1, 0.5f);
                return;
            }

            if (typeKey == NcnnLayerTypes.HardSwish)
            {
                pointwiseType = NcnnOps.PointwiseType.HardSwish;
                a = layer.GetFloat(0, 0.2f);
                b = layer.GetFloat(1, 0.5f);
                return;
            }

            if (typeKey == NcnnLayerTypes.Mish)
            {
                pointwiseType = NcnnOps.PointwiseType.Mish;
                return;
            }

            if (typeKey == NcnnLayerTypes.SELU)
            {
                pointwiseType = NcnnOps.PointwiseType.Selu;
                a = layer.GetFloat(0, 1.67326324f);
                b = layer.GetFloat(1, 1.050700987f);
                return;
            }

            if (typeKey == NcnnLayerTypes.Shrink)
            {
                pointwiseType = NcnnOps.PointwiseType.Shrink;
                a = layer.GetFloat(0, 0f);
                b = layer.GetFloat(1, 0.5f);
                return;
            }

            if (typeKey == NcnnLayerTypes.Softplus)
            {
                pointwiseType = NcnnOps.PointwiseType.Softplus;
                return;
            }

            if (typeKey == NcnnLayerTypes.CELU)
            {
                pointwiseType = NcnnOps.PointwiseType.Celu;
                a = layer.GetFloat(0, 1f);
                return;
            }

            throw new InvalidOperationException("Unsupported pointwise formula layer: " + typeKey);
        }
    }
}
