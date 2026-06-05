using System;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnPointwiseFormulaLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnPointwiseFormulaLayerRepro(NcnnLayerTypeKey typeKey)
            : base(typeKey, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
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

            ResolveFormula(TypeKey, layer, out var type, out var a, out var b);

            if (owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape))
            {
                var outRt = owner.RentTempArray(srcTex.width, srcTex.height, srcTex.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.PointwisePack4(srcTex.texture, srcTex.packs, type, a, b, outRt);
                NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
            }
            else
            {
                var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                if (srcBuf == null)
                    throw new InvalidOperationException(TypeKey + " source not found: " + layer.name);

                var outBuf = owner.RentTempBuffer(srcBuf.count, sizeof(float));
                owner.Ops.PointwiseBuf(srcBuf, srcBuf.count, type, a, b, outBuf);
                bufferBlobs[layer.topNames[0]] = outBuf;
                bufferRefs[layer.topNames[0]] = owner.NewOwnedBufferRef(layer.topNames[0], outBuf);
                if (srcView != null)
                    bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c, false);
                tempOwned.Add(outBuf);
            }

            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            ResolveFormula(TypeKey, layer, out var type, out var a, out var b);

            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            var outArr = owner.RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
            owner.Ops.PointwisePack4(cmd, src.texture, src.packs, type, a, b, outArr);
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
