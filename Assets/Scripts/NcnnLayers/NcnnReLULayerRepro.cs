using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnReLULayerRepro : NcnnBaseLayerRepro
    {
        public NcnnReLULayerRepro() : base(NcnnLayerTypes.ReLU, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                                                var slope = layer.GetFloat(0, 0f);
                                                if (owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape))
                                                {
                                                    var outRt = owner.RentTempArray(srcTex.width, srcTex.height, srcTex.packs, RenderTextureFormat.ARGBHalf);
                                                    owner.Ops.LeakyReluPack4(srcTex.texture, slope, srcTex.packs, outRt);
                                                    textureBlobs[layer.topNames[0]] = new NcnnRepro.TensorRef
                                                    {
                                                        texture = outRt,
                                                        width = outRt.width,
                                                        height = outRt.height,
                                                        packs = srcTex.packs,
                                                        refs = 1,
                                                        owned = true
                                                    };
                                                    textureShapes[layer.topNames[0]] = srcShape;
                                                }
                                                else
                                                {
                                                    var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                    var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                    if (srcBuf == null || srcView == null)
                                                        throw new InvalidOperationException("ReLU source not found: " + layer.name);

                                                    var outTensor = owner.RentTempTensorBuffer(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c);
                                                    owner.Ops.LeakyReluBuf(srcBuf, srcView.elementCount, slope, outTensor.buffer);
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

                        do
                        {
                                                var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
                                                var slope = layer.GetFloat(0, 0f);
                                                var outArr = owner.RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                                                owner.Ops.LeakyReluPack4(cmd, src.texture, slope, src.packs, outArr);
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
                                                continue;
                        } while (false);
        }
    }
}
