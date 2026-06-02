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

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context) => owner.ExecuteSoftmaxBufferLayer(layer, context);
        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context) => owner.ExecuteSoftmaxCommandBufferLayer(layer, context);
    }

    public partial class NcnnRepro
    {
        internal void ExecuteSoftmaxBufferLayer(NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                                        var srcTensor = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                        var outBuf = RentTempBuffer(softBuf.count, sizeof(float));
                                        if (srcTensor != null && srcTensor.dims == 3)
                                        {
                                            var batch = srcTensor.c;
                                            var rows = srcTensor.h;
                                            var cols = srcTensor.w;
                                            var matrixCount = rows * cols;
                                            var sliceIn = RentTempBuffer(matrixCount, sizeof(float));
                                            var sliceOut = RentTempBuffer(matrixCount, sizeof(float));
                                            try
                                            {
                                                for (var p = 0; p < batch; p++)
                                                {
                                                    var offset = p * matrixCount;
                                                    _ops.CopyBufPartial(softBuf, offset, sliceIn, matrixCount);
                                                    _ops.Softmax2D(sliceIn, sliceOut, rows, cols);
                                                    _ops.CopyBufPartial(sliceOut, 0, outBuf, matrixCount, offset);
                                                }
                                            }
                                            finally
                                            {
                                                ReturnTempBuffer(sliceIn);
                                                ReturnTempBuffer(sliceOut);
                                            }
                                        }
                                        else
                                        {
                                            var rows = srcTensor != null && srcTensor.dims == 2 ? srcTensor.h : 1;
                                            var cols = srcTensor != null && srcTensor.dims == 2 ? srcTensor.w : softBuf.count;
                                            _ops.Softmax2D(softBuf, outBuf, rows, cols);
                                        }
                                        bufferBlobs[layer.topNames[0]] = outBuf;
                                        if (srcTensor != null)
                                            bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, srcTensor.dims, srcTensor.w, srcTensor.h, srcTensor.d, srcTensor.c, false);
                                        tempOwned.Add(outBuf);
                                    }
                                    else
                                    {
                                        var src = GetTexture(textureBlobs, layer.bottomNames[0]);
                                        var outRt = RentTempArray(src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                                        _ops.SoftmaxChannelPack4(src.texture, src.packs, outRt);
                                        textureBlobs[layer.topNames[0]] = new TensorRef
                                        {
                                            texture = outRt,
                                            width = src.width,
                                            height = src.height,
                                            packs = src.packs,
                                            refs = 1,
                                            owned = true
                                        };
                                        var softShape = GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                                        textureShapes[layer.topNames[0]] = new BufferShape(3, src.width, src.height, 1, softShape.c);
                                    }
                                    Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }

        internal void ExecuteSoftmaxCommandBufferLayer(NcnnParamModel.Layer l, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            do
            {
                                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                                    var axis = l.GetInt(0, 0);
                                    if (axis != 0)
                                        throw new InvalidOperationException("Softmax axis not supported: " + axis);
                                    var outArr = RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                                    _ops.SoftmaxChannelPack4(cmd, src.texture, src.packs, outArr);
                                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = src.width, height = src.height, packs = src.packs, refs = 1, owned = true };
                                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }
    }
}
