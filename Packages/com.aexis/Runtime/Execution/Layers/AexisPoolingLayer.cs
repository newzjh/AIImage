using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisPoolingLayer : AexisBaseLayer
    {
        public AexisPoolingLayer() : base(AexisLayerTypes.Pooling, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (CanExecuteRenderTexturePath(owner, layer, context))
            {
                ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

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

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null || srcView.dims != 3)
                throw new InvalidOperationException("Pooling expects dims=3 buffer input: " + layer.name);

            var includePad = layer.GetInt(6, 0) != 0;
            var adaptivePooling = layer.GetInt(7, 0) != 0;
            var adaptiveOutW = layer.GetInt(8, 0);
            var adaptiveOutH = layer.GetInt(18, adaptiveOutW);
            int outW;
            int outH;
            if (adaptivePooling)
            {
                outW = adaptiveOutW == -233 ? srcView.w : adaptiveOutW;
                outH = adaptiveOutH == -233 ? srcView.h : adaptiveOutH;
                if (outW <= 0) outW = srcView.w;
                if (outH <= 0) outH = srcView.h;
            }
            else if (globalPooling)
            {
                kernelW = srcView.w;
                kernelH = srcView.h;
                strideW = 1;
                strideH = 1;
                padLeft = 0;
                padRight = 0;
                padTop = 0;
                padBottom = 0;
                outW = 1;
                outH = 1;
            }
            else
            {
                outW = Mathf.Max(1, (srcView.w + padLeft + padRight - kernelW) / Mathf.Max(1, strideW) + 1);
                outH = Mathf.Max(1, (srcView.h + padTop + padBottom - kernelH) / Mathf.Max(1, strideH) + 1);
            }

            var outTensor = owner.RentTempTensorBuffer(3, outW, outH, 1, srcView.c);
            var srcData = new float[srcView.elementCount];
            srcBuf.GetData(srcData);
            var outData = new float[outTensor.elementCount];
            var inPlane = srcView.w * srcView.h;
            var outPlane = outW * outH;

            for (var c = 0; c < srcView.c; c++)
            {
                var srcBase = c * inPlane;
                var dstBase = c * outPlane;
                for (var oy = 0; oy < outH; oy++)
                {
                    int sy0;
                    int sy1;
                    if (adaptivePooling)
                    {
                        sy0 = srcView.h * oy / outH;
                        sy1 = (srcView.h * (oy + 1) + outH - 1) / outH;
                    }
                    else
                    {
                        sy0 = oy * strideH - padTop;
                        sy1 = sy0 + kernelH;
                    }

                    for (var ox = 0; ox < outW; ox++)
                    {
                        int sx0;
                        int sx1;
                        if (adaptivePooling)
                        {
                            sx0 = srcView.w * ox / outW;
                            sx1 = (srcView.w * (ox + 1) + outW - 1) / outW;
                        }
                        else
                        {
                            sx0 = ox * strideW - padLeft;
                            sx1 = sx0 + kernelW;
                        }

                        var dstIndex = dstBase + oy * outW + ox;
                        if (poolType == 0)
                        {
                            var best = float.NegativeInfinity;
                            for (var sy = sy0; sy < sy1; sy++)
                            {
                                if (sy < 0 || sy >= srcView.h)
                                    continue;
                                for (var sx = sx0; sx < sx1; sx++)
                                {
                                    if (sx < 0 || sx >= srcView.w)
                                        continue;
                                    best = Mathf.Max(best, srcData[srcBase + sy * srcView.w + sx]);
                                }
                            }
                            outData[dstIndex] = best;
                        }
                        else
                        {
                            double sum = 0d;
                            var count = 0;
                            for (var sy = sy0; sy < sy1; sy++)
                            {
                                var validY = sy >= 0 && sy < srcView.h;
                                for (var sx = sx0; sx < sx1; sx++)
                                {
                                    var valid = validY && sx >= 0 && sx < srcView.w;
                                    if (valid)
                                        sum += srcData[srcBase + sy * srcView.w + sx];
                                    if (includePad || adaptivePooling || valid)
                                        count++;
                                }
                            }
                            outData[dstIndex] = count > 0 ? (float)(sum / count) : 0f;
                        }
                    }
                }
            }

            outTensor.buffer.SetData(outData);
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
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

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

            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var src, out var srcShape))
                throw new InvalidOperationException("Pooling render-texture path requires pack4 texture input: " + layer.name);

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

                        paddedRt = owner.RentTempArray(src.width + totalPadLeft + totalPadRight, src.height + totalPadTop + totalPadBottom, src.packs, RenderTextureFormat.ARGBHalf);
                        owner.Ops.PaddingPack4(src.texture, src.packs, totalPadLeft, totalPadRight, totalPadTop, totalPadBottom, 0, new Vector4(padValue, padValue, padValue, padValue), paddedRt);
                        pooledInput = paddedRt;
                        pooledInputW = paddedRt.width;
                        pooledInputH = paddedRt.height;
                    }
                }

                var outW = Mathf.Max(1, (pooledInputW - kernelW) / Mathf.Max(1, strideW) + 1);
                var outH = Mathf.Max(1, (pooledInputH - kernelH) / Mathf.Max(1, strideH) + 1);
                var outRt = owner.RentTempArray(outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.PoolingPack4(pooledInput, src.packs, kernelW, kernelH, strideW, strideH, 0, 0, poolType, outRt);
                var outShape = new AexisGraphSession.BufferShape(3, outW, outH, 1, srcShape.c);
                AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape);
            }
            finally
            {
                if (paddedRt != null)
                    owner.ReturnTempArray(paddedRt);
            }

            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }
        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
                        var cmd = context.commandBuffer;
                        var blobs = context.blobs;
                        var shapes = context.shapes;
                        var remaining = context.remaining;
                        var pinnedNames = context.pinnedNames;

                        do
                        {
                                                var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
                                                var srcShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
                                                var poolingType = layer.GetInt(0, 0);
                                                var kernelW = layer.GetInt(1, 0);
                                                var kernelH = layer.GetInt(11, kernelW);
                                                var strideW = layer.GetInt(2, 1);
                                                var strideH = layer.GetInt(12, strideW);
                                                var padLeft = layer.GetInt(3, 0);
                                                var padRight = layer.GetInt(14, padLeft);
                                                var padTop = layer.GetInt(13, padLeft);
                                                var padBottom = layer.GetInt(15, padTop);
                                                var globalPooling = layer.GetInt(4, 0);
                                                var adaptivePooling = layer.GetInt(7, 0);
                                                var outShape = ResolveCmdOutputShape(srcShape, layer, kernelW, kernelH, strideW, strideH, padLeft, padRight, padTop, padBottom, globalPooling != 0, adaptivePooling != 0);
                                                if (adaptivePooling != 0 || !CanUsePack4CmdPath(src, srcShape))
                                                {
                                                    AexisGraphSession.ResolveCmdTextureLayout(outShape, out var placeholderW, out var placeholderH, out var placeholderPacks);
                                                    owner.PublishCmdTensorLikeInput(cmd, layer.topNames[0], placeholderW, placeholderH, placeholderPacks, blobs, shapes, outShape);
                                                    owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                    continue;
                                                }

                                                var outW = globalPooling != 0 ? 1 : Mathf.Max(1, AexisGraphSession.ComputeConvOut(srcShape.w, kernelW, 1, strideW, padLeft, padRight));
                                                var outH = globalPooling != 0 ? 1 : Mathf.Max(1, AexisGraphSession.ComputeConvOut(srcShape.h, kernelH, 1, strideH, padTop, padBottom));
                                                var outArr = owner.RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                                                if (globalPooling != 0)
                                                    owner.Ops.PoolingPack4(cmd, src.texture, src.packs, src.width, src.height, 1, 1, 0, 0, poolingType, outArr);
                                                else
                                                    owner.Ops.PoolingPack4(cmd, src.texture, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, poolingType, outArr);
                                                blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef { texture = outArr, width = outW, height = outH, packs = src.packs, refs = 1, owned = true };
                                                if (shapes != null)
                                                    shapes[layer.topNames[0]] = outShape;
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                continue;
                        } while (false);
        }

        private static bool CanUsePack4CmdPath(AexisGraphSession.CmdTensorRef src, AexisGraphSession.BufferShape srcShape)
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

        private static bool CanExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            return owner.TryGetPack4Texture(
                layer.bottomNames[0],
                context.textureBlobs,
                context.textureShapes,
                context.bufferBlobs,
                context.bufferViews,
                out _,
                out _);
        }

        private static AexisGraphSession.BufferShape ResolveCmdOutputShape(
            AexisGraphSession.BufferShape srcShape,
            AexisGraphModel.Layer layer,
            int kernelW,
            int kernelH,
            int strideW,
            int strideH,
            int padLeft,
            int padRight,
            int padTop,
            int padBottom,
            bool globalPooling,
            bool adaptivePooling)
        {
            var outC = Mathf.Max(1, srcShape.c);
            if (srcShape.dims != 3)
                return new AexisGraphSession.BufferShape(srcShape.dims, Mathf.Max(1, srcShape.w), Mathf.Max(1, srcShape.h), Mathf.Max(1, srcShape.d), outC);

            if (globalPooling)
                return new AexisGraphSession.BufferShape(3, 1, 1, 1, outC);

            if (adaptivePooling)
            {
                var adaptiveOutW = layer.GetInt(8, 0);
                var adaptiveOutH = layer.GetInt(18, adaptiveOutW);
                var outW = adaptiveOutW == -233 || adaptiveOutW <= 0 ? srcShape.w : adaptiveOutW;
                var outH = adaptiveOutH == -233 || adaptiveOutH <= 0 ? srcShape.h : adaptiveOutH;
                return new AexisGraphSession.BufferShape(3, Mathf.Max(1, outW), Mathf.Max(1, outH), 1, outC);
            }

            var resolvedW = Mathf.Max(1, AexisGraphSession.ComputeConvOut(srcShape.w, kernelW, 1, strideW, padLeft, padRight));
            var resolvedH = Mathf.Max(1, AexisGraphSession.ComputeConvOut(srcShape.h, kernelH, 1, strideH, padTop, padBottom));
            return new AexisGraphSession.BufferShape(3, resolvedW, resolvedH, 1, outC);
        }
    }
}
