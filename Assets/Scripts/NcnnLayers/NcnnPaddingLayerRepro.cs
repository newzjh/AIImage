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

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context) => owner.ExecutePaddingBufferLayer(layer, context);
        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context) => owner.ExecutePaddingCommandBufferLayer(layer, context);
    }

    public partial class NcnnRepro
    {
        internal void ExecutePaddingBufferLayer(NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                                    var src = GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                    var top = layer.GetInt(0, 0);
                                    var bottom = layer.GetInt(1, 0);
                                    var left = layer.GetInt(2, 0);
                                    var right = layer.GetInt(3, 0);
                                    var type = layer.GetInt(4, 0);
                                    var value = layer.GetFloat(5, 0f);

                                    var outRt = RentTempArray(src.width + left + right, src.height + top + bottom, src.packs, RenderTextureFormat.ARGBHalf);
                                    _ops.PaddingPack4(src.texture, src.packs, left, right, top, bottom, type, new Vector4(value, value, value, value), outRt);
                                    textureBlobs[layer.topNames[0]] = new TensorRef
                                    {
                                        texture = outRt,
                                        width = outRt.width,
                                        height = outRt.height,
                                        packs = src.packs,
                                        refs = 1,
                                        owned = true
                                    };
                                    var srcShape = GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                                    textureShapes[layer.topNames[0]] = new BufferShape(3, outRt.width, outRt.height, 1, srcShape.c);
                                    Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }

        internal void ExecutePaddingCommandBufferLayer(NcnnParamModel.Layer l, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            do
            {
                                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                                    var top = l.GetInt(0, 0);
                                    var bottom = l.GetInt(1, 0);
                                    var left = l.GetInt(2, 0);
                                    var right = l.GetInt(3, 0);
                                    var type = l.GetInt(4, 0);
                                    var value = l.GetFloat(5, 0f);

                                    var outW = src.width + left + right;
                                    var outH = src.height + top + bottom;
                                    if (outW <= 0 || outH <= 0)
                                        throw new InvalidOperationException("Padding invalid out size: " + outW + "x" + outH);

                                    var outArr = RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                                    _ops.PaddingPack4(cmd, src.texture, src.packs, left, right, top, bottom, type, new Vector4(value, value, value, value), outArr);
                                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = outW, height = outH, packs = src.packs, refs = 1, owned = true };
                                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }
    }
}
