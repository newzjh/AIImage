using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnMaxUnPoolingLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnMaxUnPoolingLayerRepro() : base(NcnnLayerTypes.MaxUnPooling, supportsBufferPath: true, supportsCommandBufferPath: true) { }

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
                                                if (!indexBlobs.TryGetValue(layer.bottomNames[1], out var idx) || idx == null || (idx.texture == null && idx.buffer == null))
                                                    throw new InvalidOperationException("MaxUnPooling index source not found: " + layer.name);

                                                if (owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var src, out var srcShape))
                                                {
                                                    RenderTexture tempIdxRt = null;
                                                    if (idx.texture == null)
                                                    {
                                                        tempIdxRt = owner.RentTempArray(idx.width, idx.height, Mathf.Max(1, Mathf.CeilToInt(srcShape.c / 4f)), RenderTextureFormat.ARGBFloat);
                                                        owner.Ops.FillPack4FromBufferCHW(idx.buffer, idx.width, idx.height, srcShape.c, tempIdxRt);
                                                        idx.texture = tempIdxRt;
                                                        idx.packs = tempIdxRt.volumeDepth > 0 ? tempIdxRt.volumeDepth : 1;
                                                    }

                                                    var outRt = owner.RentTempArray(idx.sourceWidth, idx.sourceHeight, src.packs, RenderTextureFormat.ARGBHalf);
                                                    owner.ApplyMaxUnPoolingCpu(src, srcShape, idx, idx.sourceWidth, idx.sourceHeight, outRt);
                                                    textureBlobs[layer.topNames[0]] = new NcnnRepro.TensorRef
                                                    {
                                                        texture = outRt,
                                                        width = outRt.width,
                                                        height = outRt.height,
                                                        packs = src.packs,
                                                        refs = 1,
                                                        owned = true
                                                    };
                                                    textureShapes[layer.topNames[0]] = new NcnnRepro.BufferShape(3, idx.sourceWidth, idx.sourceHeight, 1, srcShape.c);
                                                    if (tempIdxRt != null)
                                                    {
                                                        idx.texture = null;
                                                        idx.packs = 0;
                                                        owner.ReturnTempArray(tempIdxRt);
                                                    }
                                                }
                                                else
                                                {
                                                    var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                    var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                    if (srcBuf == null || srcView == null || srcView.dims != 3)
                                                        throw new InvalidOperationException("MaxUnPooling expects dims=3 buffer input: " + layer.name);

                                                    ComputeBuffer indexBuf = idx.buffer;
                                                    NcnnTensorBuffer indexView = idx.view;
                                                    if (indexBuf == null || indexView == null)
                                                    {
                                                        indexBuf = owner.RentTempBuffer(idx.width * idx.height * srcView.c, sizeof(float));
                                                        owner.Ops.Pack4ToBufferCHW(idx.texture, idx.width, idx.height, srcView.c, indexBuf);
                                                        indexView = new NcnnTensorBuffer(indexBuf, 3, idx.width, idx.height, 1, srcView.c, false);
                                                        tempOwned.Add(indexBuf);
                                                    }

                                                    var outTensor = owner.RentTempTensorBuffer(3, idx.sourceWidth, idx.sourceHeight, 1, srcView.c);
                                                    owner.ApplyMaxUnPoolingCpu(srcBuf, srcView, indexBuf, indexView, idx.sourceWidth, idx.sourceHeight, outTensor);
                                                    owner.PublishTensorBufferOutput(
                                                        layer.topNames[0],
                                                        outTensor,
                                                        preferTexture: true,
                                                        textureBlobs,
                                                        textureShapes,
                                                        bufferBlobs,
                                                        bufferRefs,
                                                        bufferViews,
                                                        tempOwned);
                                                }
                                                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, new[] { layer.bottomNames[0] }, pinnedNames);
                                                owner.ConsumeIndex(indexBlobs, remaining, new[] { layer.bottomNames[1] }, pinnedNames);
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

            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var idx = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[1]);
            var idxShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[1]);
            var kernelW = layer.GetInt(1, 0);
            var kernelH = layer.GetInt(11, kernelW);
            var strideW = layer.GetInt(2, 1);
            var strideH = layer.GetInt(12, strideW);
            var padLeft = layer.GetInt(3, 0);
            var padRight = layer.GetInt(14, padLeft);
            var padTop = layer.GetInt(13, padLeft);
            var padBottom = layer.GetInt(15, padTop);
            var outW = Mathf.Max(srcShape.w, idxShape.w);
            var outH = Mathf.Max(srcShape.h, idxShape.h);
            if (strideW > 1 || strideH > 1 || kernelW > 1 || kernelH > 1 || padLeft != 0 || padRight != 0 || padTop != 0 || padBottom != 0)
            {
                outW = Mathf.Max(1, (srcShape.w - 1) * strideW - padLeft - padRight + kernelW);
                outH = Mathf.Max(1, (srcShape.h - 1) * strideH - padTop - padBottom + kernelH);
            }

            var outShape = new NcnnRepro.BufferShape(3, outW, outH, 1, srcShape.c);
            if (idx == null || idx.texture == null)
            {
                owner.PublishCmdPlaceholder(cmd, layer.topNames[0], outShape, blobs, shapes);
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            var outArr = owner.RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
            owner.Ops.MaxUnPoolingPack4(cmd, src.texture, idx.texture, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, outArr);
            blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
            {
                texture = outArr,
                width = outW,
                height = outH,
                packs = src.packs,
                refs = 1,
                owned = true
            };
            if (shapes != null)
                shapes[layer.topNames[0]] = outShape;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }
    }
}
