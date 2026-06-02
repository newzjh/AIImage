using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnInterpLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnInterpLayerRepro() : base(NcnnLayerTypes.Interp, supportsBufferPath: true, supportsCommandBufferPath: true) { }

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
                                                var srcShape = NcnnRepro.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                                                var resizeType = layer.GetInt(0, 2);
                                                var sx = layer.GetFloat(1, 1f);
                                                var sy = layer.GetFloat(2, 1f);

                                                if (Mathf.Abs(sx - 2f) < 1e-3f && Mathf.Abs(sy - 2f) < 1e-3f)
                                                {
                                                    var outRt = owner.RentTempArray(src.width * 2, src.height * 2, src.packs, RenderTextureFormat.ARGBHalf);
                                                    if (resizeType == 1)
                                                        owner.Ops.Interp2xNearestPack4(src.texture, src.packs, outRt);
                                                    else
                                                        owner.Ops.Interp2xPack4(src.texture, src.packs, outRt);
                                                    textureBlobs[layer.topNames[0]] = new NcnnRepro.TensorRef
                                                    {
                                                        texture = outRt,
                                                        width = outRt.width,
                                                        height = outRt.height,
                                                        packs = src.packs,
                                                        refs = 1,
                                                        owned = true
                                                    };
                                                    textureShapes[layer.topNames[0]] = new NcnnRepro.BufferShape(3, outRt.width, outRt.height, 1, srcShape.c);
                                                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                    continue;
                                                }

                                                if (Mathf.Abs(sx - 0.5f) < 1e-3f && Mathf.Abs(sy - 0.5f) < 1e-3f)
                                                {
                                                    var outRt = owner.RentTempArray(Mathf.Max(1, src.width / 2), Mathf.Max(1, src.height / 2), src.packs, RenderTextureFormat.ARGBHalf);
                                                    if (resizeType == 1)
                                                        owner.Ops.InterpDown2NearestPack4(src.texture, src.packs, outRt);
                                                    else
                                                        owner.Ops.InterpDown2Pack4(src.texture, src.packs, outRt);
                                                    textureBlobs[layer.topNames[0]] = new NcnnRepro.TensorRef
                                                    {
                                                        texture = outRt,
                                                        width = outRt.width,
                                                        height = outRt.height,
                                                        packs = src.packs,
                                                        refs = 1,
                                                        owned = true
                                                    };
                                                    textureShapes[layer.topNames[0]] = new NcnnRepro.BufferShape(3, outRt.width, outRt.height, 1, srcShape.c);
                                                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                    continue;
                                                }

                                                if (resizeType == 1)
                                                    throw new InvalidOperationException("unsupported nearest interp scale: " + sx.ToString("0.###", CultureInfo.InvariantCulture) + "," + sy.ToString("0.###", CultureInfo.InvariantCulture));

                                                {
                                                    var outW = Mathf.Max(1, Mathf.RoundToInt(src.width * sx));
                                                    var outH = Mathf.Max(1, Mathf.RoundToInt(src.height * sy));
                                                    var outRt = owner.RentTempArray(outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                                                    owner.Ops.InterpPack4(src.texture, src.packs, sx, sy, outRt);
                                                    textureBlobs[layer.topNames[0]] = new NcnnRepro.TensorRef
                                                    {
                                                        texture = outRt,
                                                        width = outRt.width,
                                                        height = outRt.height,
                                                        packs = src.packs,
                                                        refs = 1,
                                                        owned = true
                                                    };
                                                    textureShapes[layer.topNames[0]] = new NcnnRepro.BufferShape(3, outRt.width, outRt.height, 1, srcShape.c);
                                                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                    continue;
                                                }
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
                                                var resizeType = layer.GetInt(0, 2);
                                                var sx = layer.GetFloat(1, 1f);
                                                var sy = layer.GetFloat(2, 1f);

                                                if (Mathf.Abs(sx - 2f) < 1e-3f && Mathf.Abs(sy - 2f) < 1e-3f)
                                                {
                                                    var outArr = owner.RentTempArray(cmd, src.width * 2, src.height * 2, src.packs, RenderTextureFormat.ARGBHalf);
                                                    if (resizeType == 1)
                                                        owner.Ops.Interp2xNearestPack4(cmd, src.texture, src.packs, outArr);
                                                    else
                                                        owner.Ops.Interp2xPack4(cmd, src.texture, src.packs, outArr);
                                                    blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef { texture = outArr, width = src.width * 2, height = src.height * 2, packs = src.packs, refs = 1, owned = true };
                                                    owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
                                                    continue;
                                                }

                                                if (Mathf.Abs(sx - 0.5f) < 1e-3f && Mathf.Abs(sy - 0.5f) < 1e-3f)
                                                {
                                                    var outArr = owner.RentTempArray(cmd, src.width / 2, src.height / 2, src.packs, RenderTextureFormat.ARGBHalf);
                                                    if (resizeType == 1)
                                                        owner.Ops.InterpDown2NearestPack4(cmd, src.texture, src.packs, outArr);
                                                    else
                                                        owner.Ops.InterpDown2Pack4(cmd, src.texture, src.packs, outArr);
                                                    blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef { texture = outArr, width = src.width / 2, height = src.height / 2, packs = src.packs, refs = 1, owned = true };
                                                    owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
                                                    continue;
                                                }

                                                throw new InvalidOperationException("unsupported interp scale: " + sx.ToString("0.###", CultureInfo.InvariantCulture) + "," + sy.ToString("0.###", CultureInfo.InvariantCulture));
                        } while (false);
        }
    }
}
