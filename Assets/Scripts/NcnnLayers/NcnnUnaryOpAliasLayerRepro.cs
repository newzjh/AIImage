using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnUnaryOpAliasLayerRepro : NcnnBaseLayerRepro
    {
        private readonly int _opType;

        public NcnnUnaryOpAliasLayerRepro(NcnnLayerTypeKey typeKey, int opType)
            : base(typeKey, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
            _opType = opType;
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

            if (owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape))
            {
                var outRt = owner.RentTempArray(srcTex.width, srcTex.height, srcTex.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.UnaryOpPack4(srcTex.texture, srcTex.packs, _opType, outRt);
                NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
            }
            else
            {
                var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                if (srcBuf == null)
                    throw new InvalidOperationException(TypeKey + " source not found: " + layer.name);

                var outBuf = owner.RentTempBuffer(srcBuf.count, sizeof(float));
                owner.Ops.UnaryOpBuf(srcBuf, srcBuf.count, _opType, outBuf);
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
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            var outArr = owner.RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
            owner.Ops.UnaryOpPack4(cmd, src.texture, src.packs, _opType, outArr);
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
