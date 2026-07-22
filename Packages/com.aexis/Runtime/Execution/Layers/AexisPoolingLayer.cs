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
            var includePad = layer.GetInt(6, 0) != 0;
            var adaptivePooling = layer.GetInt(7, 0) != 0;

            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var src, out var srcShape))
                throw new InvalidOperationException("Pooling render-texture path requires pack4 texture input: " + layer.name);

            if (adaptivePooling)
                throw new InvalidOperationException("Pooling render-texture Pack4 path does not implement adaptive 2D pooling: " + layer.name);

            int outW;
            int outH;
            if (globalPooling)
            {
                kernelW = src.width;
                kernelH = src.height;
                strideW = 1;
                strideH = 1;
                padLeft = 0;
                padTop = 0;
                outW = 1;
                outH = 1;
            }
            else if (!TryResolvePack4Geometry(
                src.width, src.height, kernelW, kernelH, strideW, strideH,
                padLeft, padRight, padTop, padBottom, padMode,
                out padLeft, out _, out padTop, out _, out outW, out outH, out var reason))
            {
                throw new InvalidOperationException("Pooling render-texture Pack4 geometry is invalid | layer=" + layer.name + " | reason=" + reason);
            }

            RenderTexture outRt = null;
            try
            {
                outRt = owner.RentTempArray(outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.PoolingPack4(src.texture, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, poolType, outRt, includePad);
                var outShape = new AexisGraphSession.BufferShape(3, outW, outH, 1, srcShape.c);
                AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape);
                outRt = null;
            }
            finally
            {
                if (outRt != null)
                    owner.ReturnTempArray(outRt);
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
                                                var padMode = layer.GetInt(5, 0);
                                                var includePad = layer.GetInt(6, 0) != 0;
                                                var adaptivePooling = layer.GetInt(7, 0);
                                                if (adaptivePooling != 0 || !CanUsePack4CmdPath(src, srcShape))
                                                {
                                                    throw new InvalidOperationException(
                                                        "Pooling command-buffer Pack4 profile rejected the input descriptor or adaptive mode"
                                                        + " | layer=" + layer.name
                                                        + " | logical=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                                                        + " | texture=" + src.width + "x" + src.height + "x" + src.packs
                                                        + " | adaptive=" + adaptivePooling
                                                        + " | rejectedFallback=placeholder");
                                                }

                                                int outW;
                                                int outH;
                                                if (globalPooling != 0)
                                                {
                                                    kernelW = srcShape.w;
                                                    kernelH = srcShape.h;
                                                    strideW = 1;
                                                    strideH = 1;
                                                    padLeft = 0;
                                                    padTop = 0;
                                                    outW = 1;
                                                    outH = 1;
                                                }
                                                else if (!TryResolvePack4Geometry(
                                                    srcShape.w, srcShape.h, kernelW, kernelH, strideW, strideH,
                                                    padLeft, padRight, padTop, padBottom, padMode,
                                                    out padLeft, out _, out padTop, out _, out outW, out outH, out var reason))
                                                {
                                                    throw new InvalidOperationException("Pooling command-buffer Pack4 geometry is invalid | layer=" + layer.name + " | reason=" + reason);
                                                }
                                                var outShape = new AexisGraphSession.BufferShape(3, outW, outH, 1, srcShape.c);
                                                var outArr = owner.RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                                                owner.Ops.PoolingPack4(cmd, src.texture, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, poolingType, outArr, includePad);
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

        internal static bool TryResolvePack4Geometry(
            int inputW,
            int inputH,
            int kernelW,
            int kernelH,
            int strideW,
            int strideH,
            int declaredPadLeft,
            int declaredPadRight,
            int declaredPadTop,
            int declaredPadBottom,
            int padMode,
            out int padLeft,
            out int padRight,
            out int padTop,
            out int padBottom,
            out int outW,
            out int outH,
            out string reason)
        {
            padLeft = declaredPadLeft;
            padRight = declaredPadRight;
            padTop = declaredPadTop;
            padBottom = declaredPadBottom;
            outW = 0;
            outH = 0;
            reason = null;
            if (inputW <= 0 || inputH <= 0 || kernelW <= 0 || kernelH <= 0 || strideW <= 0 || strideH <= 0)
            {
                reason = "input, kernel, and stride extents must be positive";
                return false;
            }
            if (padMode < 0 || padMode > 3)
            {
                reason = "pad_mode must be full(0), valid(1), SAME_UPPER(2), or SAME_LOWER(3)";
                return false;
            }
            if ((padMode == 0 || padMode == 1)
                && (padLeft < 0 || padRight < 0 || padTop < 0 || padBottom < 0))
            {
                reason = "explicit pooling pads must be non-negative";
                return false;
            }

            long left = padLeft;
            long right = padRight;
            long top = padTop;
            long bottom = padBottom;
            if (padMode == 0)
            {
                var widthTail = ((long)inputW + left + right - kernelW) % strideW;
                var heightTail = ((long)inputH + top + bottom - kernelH) % strideH;
                if (widthTail != 0) right += strideW - widthTail;
                if (heightTail != 0) bottom += strideH - heightTail;
            }
            else if (padMode == 2 || padMode == 3)
            {
                var widthPad = (long)kernelW + ((long)inputW - 1) / strideW * strideW - inputW;
                var heightPad = (long)kernelH + ((long)inputH - 1) / strideH * strideH - inputH;
                widthPad = Math.Max(0L, widthPad);
                heightPad = Math.Max(0L, heightPad);
                if (padMode == 2)
                {
                    left = widthPad / 2;
                    right = widthPad - left;
                    top = heightPad / 2;
                    bottom = heightPad - top;
                }
                else
                {
                    right = widthPad / 2;
                    left = widthPad - right;
                    bottom = heightPad / 2;
                    top = heightPad - bottom;
                }
            }

            var width = ((long)inputW + left + right - kernelW) / strideW + 1;
            var height = ((long)inputH + top + bottom - kernelH) / strideH + 1;
            if (left > int.MaxValue || right > int.MaxValue || top > int.MaxValue || bottom > int.MaxValue
                || width <= 0 || width > int.MaxValue || height <= 0 || height > int.MaxValue)
            {
                reason = "resolved pooling geometry is empty or exceeds the Int32 descriptor range";
                return false;
            }

            padLeft = (int)left;
            padRight = (int)right;
            padTop = (int)top;
            padBottom = (int)bottom;
            outW = (int)width;
            outH = (int)height;
            return true;
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
