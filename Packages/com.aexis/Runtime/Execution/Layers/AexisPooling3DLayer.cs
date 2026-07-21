using System;
using UnityEngine;

namespace Aexis.Execution
{
    public sealed class AexisPooling3DLayer : AexisBaseLayer
    {
        public AexisPooling3DLayer()
            : base(AexisLayerTypes.Pooling3D, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

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
            var kernelW = Mathf.Max(1, layer.GetInt(1, 0));
            var kernelH = Mathf.Max(1, layer.GetInt(11, kernelW));
            var kernelD = Mathf.Max(1, layer.GetInt(21, kernelW));
            var strideW = Mathf.Max(1, layer.GetInt(2, 1));
            var strideH = Mathf.Max(1, layer.GetInt(12, strideW));
            var strideD = Mathf.Max(1, layer.GetInt(22, strideW));
            var padLeft = layer.GetInt(3, 0);
            var padRight = layer.GetInt(14, padLeft);
            var padTop = layer.GetInt(13, padLeft);
            var padBottom = layer.GetInt(15, padTop);
            var padFront = layer.GetInt(23, padLeft);
            var padBehind = layer.GetInt(16, padFront);
            var globalPooling = layer.GetInt(4, 0) != 0;
            var padMode = layer.GetInt(5, 0);
            var includePad = layer.GetInt(6, 0) != 0;
            var adaptivePooling = layer.GetInt(7, 0) != 0;
            var adaptiveOutW = layer.GetInt(8, 0);
            var adaptiveOutH = layer.GetInt(18, adaptiveOutW);
            var adaptiveOutD = layer.GetInt(28, adaptiveOutW);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null || srcView.dims != 4)
                throw new InvalidOperationException("Pooling3D expects dims=4 tensor input: " + layer.name);

            ResolveOutputShape(
                srcView,
                globalPooling,
                adaptivePooling,
                kernelW,
                kernelH,
                kernelD,
                strideW,
                strideH,
                strideD,
                padLeft,
                padRight,
                padTop,
                padBottom,
                padFront,
                padBehind,
                padMode,
                adaptiveOutW,
                adaptiveOutH,
                adaptiveOutD,
                out var outW,
                out var outH,
                out var outD,
                out var resolvedPadLeft,
                out var resolvedPadTop,
                out var resolvedPadFront);

            var srcData = AexisGraphSession.ReadFloatBuffer(srcBuf);
            var outTensor = owner.RentTempTensorBuffer(4, outW, outH, outD, srcView.c);
            var outData = new float[outTensor.elementCount];

            var inPlane = srcView.w * srcView.h;
            var inVolume = inPlane * srcView.d;
            var outPlane = outW * outH;
            var outVolume = outPlane * outD;
            var maxPadValue = float.NegativeInfinity;

            for (var c = 0; c < srcView.c; c++)
            {
                var srcChannelBase = c * inVolume;
                var dstChannelBase = c * outVolume;

                for (var oz = 0; oz < outD; oz++)
                {
                    int sz0;
                    int sz1;
                    if (adaptivePooling)
                    {
                        sz0 = srcView.d * oz / outD;
                        sz1 = (srcView.d * (oz + 1) + outD - 1) / outD;
                    }
                    else if (globalPooling)
                    {
                        sz0 = 0;
                        sz1 = srcView.d;
                    }
                    else
                    {
                        sz0 = oz * strideD - resolvedPadFront;
                        sz1 = sz0 + kernelD;
                    }

                    for (var oy = 0; oy < outH; oy++)
                    {
                        int sy0;
                        int sy1;
                        if (adaptivePooling)
                        {
                            sy0 = srcView.h * oy / outH;
                            sy1 = (srcView.h * (oy + 1) + outH - 1) / outH;
                        }
                        else if (globalPooling)
                        {
                            sy0 = 0;
                            sy1 = srcView.h;
                        }
                        else
                        {
                            sy0 = oy * strideH - resolvedPadTop;
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
                            else if (globalPooling)
                            {
                                sx0 = 0;
                                sx1 = srcView.w;
                            }
                            else
                            {
                                sx0 = ox * strideW - resolvedPadLeft;
                                sx1 = sx0 + kernelW;
                            }

                            var dstIndex = dstChannelBase + (oz * outPlane) + (oy * outW) + ox;
                            if (poolType == 0)
                            {
                                var best = maxPadValue;
                                for (var sz = sz0; sz < sz1; sz++)
                                {
                                    if (sz < 0 || sz >= srcView.d)
                                        continue;

                                    var srcDepthBase = srcChannelBase + sz * inPlane;
                                    for (var sy = sy0; sy < sy1; sy++)
                                    {
                                        if (sy < 0 || sy >= srcView.h)
                                            continue;

                                        var srcRowBase = srcDepthBase + sy * srcView.w;
                                        for (var sx = sx0; sx < sx1; sx++)
                                        {
                                            if (sx < 0 || sx >= srcView.w)
                                                continue;

                                            best = Mathf.Max(best, srcData[srcRowBase + sx]);
                                        }
                                    }
                                }

                                outData[dstIndex] = best;
                            }
                            else
                            {
                                double sum = 0d;
                                var count = 0;
                                for (var sz = sz0; sz < sz1; sz++)
                                {
                                    var validZ = sz >= 0 && sz < srcView.d;
                                    var srcDepthBase = validZ ? srcChannelBase + sz * inPlane : 0;
                                    for (var sy = sy0; sy < sy1; sy++)
                                    {
                                        var validZY = validZ && sy >= 0 && sy < srcView.h;
                                        var srcRowBase = validZY ? srcDepthBase + sy * srcView.w : 0;
                                        for (var sx = sx0; sx < sx1; sx++)
                                        {
                                            var valid = validZY && sx >= 0 && sx < srcView.w;
                                            if (valid)
                                                sum += srcData[srcRowBase + sx];

                                            if (adaptivePooling || includePad || valid)
                                                count++;
                                        }
                                    }
                                }

                                outData[dstIndex] = count > 0 ? (float)(sum / count) : 0f;
                            }
                        }
                    }
                }
            }

            outTensor.buffer.SetData(outData);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: false,
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

            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape)
                || srcShape.dims != 4)
            {
                throw new InvalidOperationException("Pooling3D render-texture path requires dims=4 pack4 input: " + layer.name);
            }

            var poolType = layer.GetInt(0, 0);
            var kernelW = Mathf.Max(1, layer.GetInt(1, 0));
            var kernelH = Mathf.Max(1, layer.GetInt(11, kernelW));
            var kernelD = Mathf.Max(1, layer.GetInt(21, kernelW));
            var strideW = Mathf.Max(1, layer.GetInt(2, 1));
            var strideH = Mathf.Max(1, layer.GetInt(12, strideW));
            var strideD = Mathf.Max(1, layer.GetInt(22, strideW));
            var padLeft = layer.GetInt(3, 0);
            var padRight = layer.GetInt(14, padLeft);
            var padTop = layer.GetInt(13, padLeft);
            var padBottom = layer.GetInt(15, padTop);
            var padFront = layer.GetInt(23, padLeft);
            var padBehind = layer.GetInt(16, padFront);
            var globalPooling = layer.GetInt(4, 0) != 0;
            var padMode = layer.GetInt(5, 0);
            var includePad = layer.GetInt(6, 0) != 0;
            var adaptivePooling = layer.GetInt(7, 0) != 0;
            var adaptiveOutW = layer.GetInt(8, 0);
            var adaptiveOutH = layer.GetInt(18, adaptiveOutW);
            var adaptiveOutD = layer.GetInt(28, adaptiveOutW);

            ResolveOutputShape(
                srcShape,
                globalPooling,
                adaptivePooling,
                kernelW,
                kernelH,
                kernelD,
                strideW,
                strideH,
                strideD,
                padLeft,
                padRight,
                padTop,
                padBottom,
                padFront,
                padBehind,
                padMode,
                adaptiveOutW,
                adaptiveOutH,
                adaptiveOutD,
                out var outW,
                out var outH,
                out var outD,
                out var resolvedPadLeft,
                out var resolvedPadTop,
                out var resolvedPadFront);

            var outPacks = Mathf.Max(1, Mathf.CeilToInt(srcShape.c / 4f));
            var outSlices = Mathf.Max(1, outD) * outPacks;
            var outRt = owner.RentTempArray(outW, outH, outSlices, AexisGraphSession.ResolveTensorTextureFormat(4));
            owner.Ops.PoolingPack4Cdhw(
                srcTex.texture,
                srcShape.w,
                srcShape.h,
                srcShape.d,
                srcShape.c,
                kernelW,
                kernelH,
                kernelD,
                strideW,
                strideH,
                strideD,
                resolvedPadLeft,
                resolvedPadTop,
                resolvedPadFront,
                poolType,
                includePad,
                adaptivePooling,
                globalPooling,
                outW,
                outH,
                outD,
                srcShape.c,
                outRt);
            AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, new AexisGraphSession.BufferShape(4, outW, outH, outD, srcShape.c), new AexisGraphSession.BufferShape(4, outW, outH, outD, srcShape.c));
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcContract = AexisGraphSession.GetCmdTensorContract(src);
            var srcShape = srcContract.LogicalShape;
            if (srcShape.dims != 4)
                throw new InvalidOperationException("Pooling3D command-buffer path expects dims=4 input: " + layer.name);
            if (!srcContract.IsPack4Image || !AexisGraphSession.MatchesPack4TextureStorage(src, srcShape))
                throw new InvalidOperationException("Pooling3D command-buffer path requires a TensorDescriptor-backed CDHW Pack4 Texture2DArray: " + layer.name);

            var poolType = layer.GetInt(0, 0);
            var kernelW = Mathf.Max(1, layer.GetInt(1, 0));
            var kernelH = Mathf.Max(1, layer.GetInt(11, kernelW));
            var kernelD = Mathf.Max(1, layer.GetInt(21, kernelW));
            var strideW = Mathf.Max(1, layer.GetInt(2, 1));
            var strideH = Mathf.Max(1, layer.GetInt(12, strideW));
            var strideD = Mathf.Max(1, layer.GetInt(22, strideW));
            var padLeft = layer.GetInt(3, 0);
            var padRight = layer.GetInt(14, padLeft);
            var padTop = layer.GetInt(13, padLeft);
            var padBottom = layer.GetInt(15, padTop);
            var padFront = layer.GetInt(23, padLeft);
            var padBehind = layer.GetInt(16, padFront);
            var globalPooling = layer.GetInt(4, 0) != 0;
            var padMode = layer.GetInt(5, 0);
            var includePad = layer.GetInt(6, 0) != 0;
            var adaptivePooling = layer.GetInt(7, 0) != 0;
            var adaptiveOutW = layer.GetInt(8, 0);
            var adaptiveOutH = layer.GetInt(18, adaptiveOutW);
            var adaptiveOutD = layer.GetInt(28, adaptiveOutW);

            ResolveOutputShape(
                srcShape,
                globalPooling,
                adaptivePooling,
                kernelW,
                kernelH,
                kernelD,
                strideW,
                strideH,
                strideD,
                padLeft,
                padRight,
                padTop,
                padBottom,
                padFront,
                padBehind,
                padMode,
                adaptiveOutW,
                adaptiveOutH,
                adaptiveOutD,
                out var outW,
                out var outH,
                out var outD,
                out var resolvedPadLeft,
                out var resolvedPadTop,
                out var resolvedPadFront);

            var outShape = new AexisGraphSession.BufferShape(4, outW, outH, outD, srcShape.c);
            var outRt = owner.RentTempArray(cmd, outW, outH, outD * src.packs, AexisGraphSession.ResolveTensorTextureFormat(4));
            owner.Ops.PoolingPack4Cdhw(
                cmd,
                src.texture,
                srcShape.w,
                srcShape.h,
                srcShape.d,
                srcShape.c,
                kernelW,
                kernelH,
                kernelD,
                strideW,
                strideH,
                strideD,
                resolvedPadLeft,
                resolvedPadTop,
                resolvedPadFront,
                poolType,
                includePad,
                adaptivePooling,
                globalPooling,
                outW,
                outH,
                outD,
                srcShape.c,
                outRt);

            blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(
                outRt,
                outShape,
                outShape,
                owned: true,
                blobName: layer.topNames[0]);
            if (shapes != null)
                shapes[layer.topNames[0]] = outShape;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static void ResolveOutputShape(
            AexisTensorBuffer src,
            bool globalPooling,
            bool adaptivePooling,
            int kernelW,
            int kernelH,
            int kernelD,
            int strideW,
            int strideH,
            int strideD,
            int padLeft,
            int padRight,
            int padTop,
            int padBottom,
            int padFront,
            int padBehind,
            int padMode,
            int adaptiveOutW,
            int adaptiveOutH,
            int adaptiveOutD,
            out int outW,
            out int outH,
            out int outD,
            out int resolvedPadLeft,
            out int resolvedPadTop,
            out int resolvedPadFront)
        {
            ResolveOutputShape(
                new AexisGraphSession.BufferShape(src.dims, src.w, src.h, src.d, src.c),
                globalPooling,
                adaptivePooling,
                kernelW,
                kernelH,
                kernelD,
                strideW,
                strideH,
                strideD,
                padLeft,
                padRight,
                padTop,
                padBottom,
                padFront,
                padBehind,
                padMode,
                adaptiveOutW,
                adaptiveOutH,
                adaptiveOutD,
                out outW,
                out outH,
                out outD,
                out resolvedPadLeft,
                out resolvedPadTop,
                out resolvedPadFront);
        }

        private static void ResolveOutputShape(
            AexisGraphSession.BufferShape src,
            bool globalPooling,
            bool adaptivePooling,
            int kernelW,
            int kernelH,
            int kernelD,
            int strideW,
            int strideH,
            int strideD,
            int padLeft,
            int padRight,
            int padTop,
            int padBottom,
            int padFront,
            int padBehind,
            int padMode,
            int adaptiveOutW,
            int adaptiveOutH,
            int adaptiveOutD,
            out int outW,
            out int outH,
            out int outD,
            out int resolvedPadLeft,
            out int resolvedPadTop,
            out int resolvedPadFront)
        {
            if (globalPooling)
            {
                outW = 1;
                outH = 1;
                outD = 1;
                resolvedPadLeft = 0;
                resolvedPadTop = 0;
                resolvedPadFront = 0;
                return;
            }

            if (adaptivePooling)
            {
                outW = adaptiveOutW == -233 || adaptiveOutW <= 0 ? src.w : adaptiveOutW;
                outH = adaptiveOutH == -233 || adaptiveOutH <= 0 ? src.h : adaptiveOutH;
                outD = adaptiveOutD == -233 || adaptiveOutD <= 0 ? src.d : adaptiveOutD;
                resolvedPadLeft = 0;
                resolvedPadTop = 0;
                resolvedPadFront = 0;
                return;
            }

            var totalPadLeft = padLeft;
            var totalPadRight = padRight;
            var totalPadTop = padTop;
            var totalPadBottom = padBottom;
            var totalPadFront = padFront;
            var totalPadBehind = padBehind;

            if (padMode == 0)
            {
                var wtail = (src.w + padLeft + padRight - kernelW) % strideW;
                var htail = (src.h + padTop + padBottom - kernelH) % strideH;
                var dtail = (src.d + padFront + padBehind - kernelD) % strideD;
                if (wtail != 0)
                    totalPadRight += strideW - wtail;
                if (htail != 0)
                    totalPadBottom += strideH - htail;
                if (dtail != 0)
                    totalPadBehind += strideD - dtail;
            }
            else if (padMode == 2)
            {
                var wpad = kernelW + (src.w - 1) / strideW * strideW - src.w;
                var hpad = kernelH + (src.h - 1) / strideH * strideH - src.h;
                var dpad = kernelD + (src.d - 1) / strideD * strideD - src.d;
                totalPadLeft = wpad / 2;
                totalPadRight = wpad - totalPadLeft;
                totalPadTop = hpad / 2;
                totalPadBottom = hpad - totalPadTop;
                totalPadFront = dpad / 2;
                totalPadBehind = dpad - totalPadFront;
            }
            else if (padMode == 3)
            {
                var wpad = kernelW + (src.w - 1) / strideW * strideW - src.w;
                var hpad = kernelH + (src.h - 1) / strideH * strideH - src.h;
                var dpad = kernelD + (src.d - 1) / strideD * strideD - src.d;
                totalPadLeft = wpad - wpad / 2;
                totalPadRight = wpad / 2;
                totalPadTop = hpad - hpad / 2;
                totalPadBottom = hpad / 2;
                totalPadFront = dpad - dpad / 2;
                totalPadBehind = dpad / 2;
            }

            outW = Mathf.Max(1, (src.w + totalPadLeft + totalPadRight - kernelW) / strideW + 1);
            outH = Mathf.Max(1, (src.h + totalPadTop + totalPadBottom - kernelH) / strideH + 1);
            outD = Mathf.Max(1, (src.d + totalPadFront + totalPadBehind - kernelD) / strideD + 1);
            resolvedPadLeft = totalPadLeft;
            resolvedPadTop = totalPadTop;
            resolvedPadFront = totalPadFront;
        }

        private static bool CanExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            return !owner.ShouldForceCurrentLayerBufferPath()
                && owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out _,
                    out var srcShape)
                && srcShape.dims == 4;
        }
    }
}
