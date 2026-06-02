using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnPaddingLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnPaddingLayerRepro() : base(NcnnLayerTypes.Padding, supportsBufferPath: true, supportsCommandBufferPath: true) { }

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
                                                var src = owner.GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                                var top = layer.GetInt(0, 0);
                                                var bottom = layer.GetInt(1, 0);
                                                var left = layer.GetInt(2, 0);
                                                var right = layer.GetInt(3, 0);
                                                var type = layer.GetInt(4, 0);
                                                var value = layer.GetFloat(5, 0f);

                                                var outRt = owner.RentTempArray(src.width + left + right, src.height + top + bottom, src.packs, RenderTextureFormat.ARGBHalf);
                                                owner.Ops.PaddingPack4(src.texture, src.packs, left, right, top, bottom, type, new Vector4(value, value, value, value), outRt);
                                                textureBlobs[layer.topNames[0]] = new NcnnRepro.TensorRef
                                                {
                                                    texture = outRt,
                                                    width = outRt.width,
                                                    height = outRt.height,
                                                    packs = src.packs,
                                                    refs = 1,
                                                    owned = true
                                                };
                                                var srcShape = NcnnRepro.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                                                textureShapes[layer.topNames[0]] = new NcnnRepro.BufferShape(3, outRt.width, outRt.height, 1, srcShape.c);
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
                                                var top = layer.GetInt(0, 0);
                                                var bottom = layer.GetInt(1, 0);
                                                var left = layer.GetInt(2, 0);
                                                var right = layer.GetInt(3, 0);
                                                var type = layer.GetInt(4, 0);
                                                var value = layer.GetFloat(5, 0f);

                                                var outW = src.width + left + right;
                                                var outH = src.height + top + bottom;
                                                if (outW <= 0 || outH <= 0)
                                                    throw new InvalidOperationException("Padding invalid out size: " + outW + "x" + outH);

                                                var outArr = owner.RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                                                owner.Ops.PaddingPack4(cmd, src.texture, src.packs, left, right, top, bottom, type, new Vector4(value, value, value, value), outArr);
                                                blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef { texture = outArr, width = outW, height = outH, packs = src.packs, refs = 1, owned = true };
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }
    }
}
