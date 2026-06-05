using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnSoftmaxLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnSoftmaxLayerRepro() : base(NcnnLayerTypes.Softmax, supportsBufferPath: true, supportsCommandBufferPath: true) { }

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
                                                if (bufferBlobs.TryGetValue(layer.bottomNames[0], out var softBuf) && softBuf != null)
                                                {
                                                    var srcTensor = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                    var outBuf = owner.RentTempBuffer(softBuf.count, sizeof(float));
                                                    if (srcTensor != null && srcTensor.dims == 3)
                                                    {
                                                        var batch = srcTensor.c;
                                                        var rows = srcTensor.h;
                                                        var cols = srcTensor.w;
                                                        var matrixCount = rows * cols;
                                                        var sliceIn = owner.RentTempBuffer(matrixCount, sizeof(float));
                                                        var sliceOut = owner.RentTempBuffer(matrixCount, sizeof(float));
                                                        try
                                                        {
                                                            for (var p = 0; p < batch; p++)
                                                            {
                                                                var offset = p * matrixCount;
                                                                owner.Ops.CopyBufPartial(softBuf, offset, sliceIn, matrixCount);
                                                                owner.Ops.Softmax2D(sliceIn, sliceOut, rows, cols);
                                                                owner.Ops.CopyBufPartial(sliceOut, 0, outBuf, matrixCount, offset);
                                                            }
                                                        }
                                                        finally
                                                        {
                                                            owner.ReturnTempBuffer(sliceIn);
                                                            owner.ReturnTempBuffer(sliceOut);
                                                        }
                                                    }
                                                    else
                                                    {
                                                        var rows = srcTensor != null && srcTensor.dims == 2 ? srcTensor.h : 1;
                                                        var cols = srcTensor != null && srcTensor.dims == 2 ? srcTensor.w : softBuf.count;
                                                        owner.Ops.Softmax2D(softBuf, outBuf, rows, cols);
                                                    }
                                                    bufferBlobs[layer.topNames[0]] = outBuf;
                                                    if (srcTensor != null)
                                                        bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, srcTensor.dims, srcTensor.w, srcTensor.h, srcTensor.d, srcTensor.c, false);
                                                    tempOwned.Add(outBuf);
                                                }
                                                else
                                                {
                                                    var src = NcnnRepro.GetTexture(textureBlobs, layer.bottomNames[0]);
                                                    var outRt = owner.RentTempArray(src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                                                    owner.Ops.SoftmaxChannelPack4(src.texture, src.packs, outRt);
                                                    textureBlobs[layer.topNames[0]] = new NcnnRepro.TensorRef
                                                    {
                                                        texture = outRt,
                                                        width = src.width,
                                                        height = src.height,
                                                        packs = src.packs,
                                                        refs = 1,
                                                        owned = true
                                                    };
                                                    var softShape = NcnnRepro.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                                                    textureShapes[layer.topNames[0]] = new NcnnRepro.BufferShape(3, src.width, src.height, 1, softShape.c);
                                                }
                                                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }
        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
                        var cmd = context.commandBuffer;
                        var blobs = context.blobs;
                        var shapes = context.shapes;
                        var remaining = context.remaining;
                        var pinnedNames = context.pinnedNames;

                        do
                        {
                                                var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
                                                var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
                                                var axis = layer.GetInt(0, 0);
                                                if (axis != 0 || !CanUsePack4Softmax(src, srcShape))
                                                {
                                                    NcnnRepro.ResolveCmdTextureLayout(srcShape, out var width, out var height, out var packs);
                                                    owner.PublishCmdTensorLikeInput(cmd, layer.topNames[0], width, height, packs, blobs, shapes, srcShape);
                                                    owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                    continue;
                                                }
                                                var outArr = owner.RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                                                owner.Ops.SoftmaxChannelPack4(cmd, src.texture, src.packs, outArr);
                                                blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef { texture = outArr, width = src.width, height = src.height, packs = src.packs, refs = 1, owned = true };
                                                if (shapes != null)
                                                    shapes[layer.topNames[0]] = srcShape;
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                continue;
                        } while (false);
        }

        private static bool CanUsePack4Softmax(NcnnRepro.CmdTensorRef src, NcnnRepro.BufferShape srcShape)
        {
            return src != null
                && src.texture != null
                && srcShape.dims == 3
                && srcShape.d == 1
                && srcShape.w == src.width
                && srcShape.h == src.height
                && srcShape.c > 0
                && srcShape.c <= src.packs * 4;
        }
    }
}
