using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnGeluLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnGeluLayerRepro() : base(NcnnLayerTypes.GELU, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context) => owner.ExecuteGELUBufferLayer(layer, context);
        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context) => owner.ExecuteGELUCommandBufferLayer(layer, context);
    }

    public partial class NcnnRepro
    {
        internal void ExecuteGELUBufferLayer(NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var indexBlobs = context.indexBlobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            do
            {
                                    if (!ForceBufferGeluAll
                                        && TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape))
                                    {
                                        var outRt = RentTempArray(srcTex.width, srcTex.height, srcTex.packs, RenderTextureFormat.ARGBHalf);
                                        _ops.GeluPack4(srcTex.texture, srcTex.packs, false, outRt);
                                        SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
                                    }
                                    else
                                    {
                                        var srcBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                        var srcView = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                        if (srcBuf == null)
                                            throw new InvalidOperationException("GELU source not found: " + layer.name);
                                        var outBuf = RentTempBuffer(srcBuf.count, sizeof(float));
                                        var srcData = ReadFloatBuffer(srcBuf);
                                        var outData = new float[srcData.Length];
                                        for (var i = 0; i < srcData.Length; i++)
                                        {
                                            var x = srcData[i];
                                            var t = 0.7978845608f * (x + 0.044715f * x * x * x);
                                            outData[i] = 0.5f * x * (1f + (float)Math.Tanh(t));
                                        }
                                        outBuf.SetData(outData);
                                        bufferBlobs[layer.topNames[0]] = outBuf;
                                        if (srcView != null)
                                            bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c, false);
                                        tempOwned.Add(outBuf);
                                    }
                                    Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }

        internal void ExecuteGELUCommandBufferLayer(NcnnParamModel.Layer l, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            do
            {
                                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                                    var fast = l.GetInt(0, 0) != 0;
                                    var outArr = RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                                    _ops.GeluPack4(cmd, src.texture, src.packs, fast, outArr);
                                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = src.width, height = src.height, packs = src.packs, refs = 1, owned = true };
                                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }
    }
}
