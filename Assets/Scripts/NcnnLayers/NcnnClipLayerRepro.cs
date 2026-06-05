using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnClipLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnClipLayerRepro() : base(NcnnLayerTypes.Clip, supportsBufferPath: true, supportsCommandBufferPath: true) { }

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

                        do
                        {
                                                var minValue = layer.GetFloat(0, -1e30f);
                                                var maxValue = layer.GetFloat(1, 1e30f);

                                                if (owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape))
                                                {
                                                    var outRt = owner.RentTempArray(srcTex.width, srcTex.height, srcTex.packs, RenderTextureFormat.ARGBHalf);
                                                    owner.Ops.ClipPack4(srcTex.texture, minValue, maxValue, srcTex.packs, outRt);
                                                    NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
                                                }
                                                else
                                                {
                                                    var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                    var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                    if (srcBuf == null)
                                                        throw new InvalidOperationException("Clip source not found: " + layer.name);

                                                    var tmpBuf = owner.RentTempBuffer(srcBuf.count, sizeof(float));
                                                    var outBuf = owner.RentTempBuffer(srcBuf.count, sizeof(float));
                                                    owner.Ops.BinaryOpScalarBuf(srcBuf, minValue, srcBuf.count, 4, tmpBuf);
                                                    owner.Ops.BinaryOpScalarBuf(tmpBuf, maxValue, srcBuf.count, 5, outBuf);
                                                    bufferBlobs[layer.topNames[0]] = outBuf;
                                                    bufferRefs[layer.topNames[0]] = owner.NewOwnedBufferRef(layer.topNames[0], outBuf);
                                                    if (srcView != null)
                                                        bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c, false);
                                                    tempOwned.Add(tmpBuf);
                                                    tempOwned.Add(outBuf);
                                                }

                                                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var minValue = layer.GetFloat(0, -1e30f);
            var maxValue = layer.GetFloat(1, 1e30f);
            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            var outArr = owner.RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
            owner.Ops.ClipPack4(cmd, src.texture, minValue, maxValue, src.packs, outArr);
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
