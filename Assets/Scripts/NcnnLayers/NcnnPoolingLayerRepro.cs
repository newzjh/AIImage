using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnPoolingLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnPoolingLayerRepro() : base(NcnnLayerTypes.Pooling, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context) => owner.ExecutePoolingBufferLayer(layer, context);
        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context) => owner.ExecutePoolingCommandBufferLayer(layer, context);
    }

    public partial class NcnnRepro
    {
        internal void ExecutePoolingBufferLayer(NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                                    var poolType = layer.GetInt(0, 0);
                                    var kernelW = layer.GetInt(1, 0);
                                    var kernelH = layer.GetInt(11, kernelW);
                                    var strideW = layer.GetInt(2, 1);
                                    var strideH = layer.GetInt(12, strideW);
                                    var padLeft = layer.GetInt(3, 0);
                                    var padRight = layer.GetInt(14, padLeft);
                                    var padTop = layer.GetInt(13, padLeft);
                                    var padBottom = layer.GetInt(15, padTop);
                                    var globalPooling = layer.GetInt(4, 0) != 0;
                                    var padMode = layer.GetInt(5, 0);

                                    var pooledInput = src.texture;
                                    var pooledInputW = src.width;
                                    var pooledInputH = src.height;
                                    RenderTexture paddedRt = null;

                                    try
                                    {
                                        if (globalPooling)
                                        {
                                            kernelW = pooledInputW;
                                            kernelH = pooledInputH;
                                            strideW = 1;
                                            strideH = 1;
                                            padLeft = 0;
                                            padRight = 0;
                                            padTop = 0;
                                            padBottom = 0;
                                        }
                                        else
                                        {
                                            var padW = padLeft + padRight;
                                            var padH = padTop + padBottom;
                                            var extraRight = 0;
                                            var extraBottom = 0;

                                            if (padMode == 0)
                                            {
                                                var wtail = (src.width + padW - kernelW) % Mathf.Max(1, strideW);
                                                var htail = (src.height + padH - kernelH) % Mathf.Max(1, strideH);
                                                if (wtail != 0)
                                                    extraRight = strideW - wtail;
                                                if (htail != 0)
                                                    extraBottom = strideH - htail;
                                            }
                                            else if (padMode == 2 || padMode == 3)
                                            {
                                                var wpad = kernelW + (src.width - 1) / Mathf.Max(1, strideW) * strideW - src.width;
                                                var hpad = kernelH + (src.height - 1) / Mathf.Max(1, strideH) * strideH - src.height;
                                                if (wpad > 0)
                                                {
                                                    padLeft = 0;
                                                    padRight = wpad;
                                                    padTop = 0;
                                                    padBottom = hpad > 0 ? hpad : 0;
                                                }
                                                else if (hpad > 0)
                                                {
                                                    padLeft = 0;
                                                    padRight = 0;
                                                    padTop = 0;
                                                    padBottom = hpad;
                                                }
                                            }

                                            var totalPadLeft = Mathf.Max(0, padLeft);
                                            var totalPadRight = Mathf.Max(0, padRight + extraRight);
                                            var totalPadTop = Mathf.Max(0, padTop);
                                            var totalPadBottom = Mathf.Max(0, padBottom + extraBottom);
                                            if (totalPadLeft != 0 || totalPadRight != 0 || totalPadTop != 0 || totalPadBottom != 0)
                                            {
                                                var padValue = poolType == 0
                                                    ? ((src.texture.format == RenderTextureFormat.ARGBHalf || src.texture.format == RenderTextureFormat.ARGB2101010 || src.texture.format == RenderTextureFormat.ARGB64)
                                                        ? -65000f
                                                        : -3.402823466e+38f)
                                                    : 0f;

                                                paddedRt = RentTempArray(src.width + totalPadLeft + totalPadRight, src.height + totalPadTop + totalPadBottom, src.packs, RenderTextureFormat.ARGBHalf);
                                                _ops.PaddingPack4(src.texture, src.packs, totalPadLeft, totalPadRight, totalPadTop, totalPadBottom, 0, new Vector4(padValue, padValue, padValue, padValue), paddedRt);
                                                pooledInput = paddedRt;
                                                pooledInputW = paddedRt.width;
                                                pooledInputH = paddedRt.height;
                                            }
                                        }

                                        var outW = Mathf.Max(1, (pooledInputW - kernelW) / Mathf.Max(1, strideW) + 1);
                                        var outH = Mathf.Max(1, (pooledInputH - kernelH) / Mathf.Max(1, strideH) + 1);
                                        var outRt = RentTempArray(outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                                        _ops.PoolingPack4(pooledInput, src.packs, kernelW, kernelH, strideW, strideH, 0, 0, poolType, outRt);
                                        textureBlobs[layer.topNames[0]] = new TensorRef
                                        {
                                            texture = outRt,
                                            width = outW,
                                            height = outH,
                                            packs = src.packs,
                                            refs = 1,
                                            owned = true
                                        };
                                        var poolSrcShape = GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                                        textureShapes[layer.topNames[0]] = new BufferShape(3, outW, outH, 1, poolSrcShape.c);
                                    }
                                    finally
                                    {
                                        if (paddedRt != null)
                                            ReturnTempArray(paddedRt);
                                    }
                                    Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }

        internal void ExecutePoolingCommandBufferLayer(NcnnParamModel.Layer l, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            do
            {
                                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                                    var poolingType = l.GetInt(0, 0);
                                    var kernelW = l.GetInt(1, 0);
                                    var kernelH = l.GetInt(11, kernelW);
                                    var strideW = l.GetInt(2, 1);
                                    var strideH = l.GetInt(12, strideW);
                                    var padLeft = l.GetInt(3, 0);
                                    var padTop = l.GetInt(13, padLeft);
                                    var globalPooling = l.GetInt(4, 0);
                                    var adaptivePooling = l.GetInt(7, 0);
                                    if (globalPooling != 0 || adaptivePooling != 0)
                                        throw new InvalidOperationException("Pooling(global/adaptive) not supported");

                                    var outW = (src.width + padLeft * 2 - kernelW) / strideW + 1;
                                    var outH = (src.height + padTop * 2 - kernelH) / strideH + 1;
                                    outW = Mathf.Max(1, outW);
                                    outH = Mathf.Max(1, outH);
                                    var outArr = RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                                    _ops.PoolingPack4(cmd, src.texture, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, poolingType, outArr);
                                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = outW, height = outH, packs = src.packs, refs = 1, owned = true };
                                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }
    }
}
