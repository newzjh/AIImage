using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
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
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            if (!owner.ShouldForceCurrentLayerBufferPath()
                && NcnnRepro.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[0], out var srcTex, out var srcShape))
            {
                ResolveTargetShape(layer, srcShape, textureBlobs, textureShapes, bufferViews, out var outW, out var outH, out var outC);

                if (srcShape.dims == 2 && outW == srcShape.w)
                {
                    new NcnnNoopLayerRepro().ExecuteBuffer(owner, layer, context);
                    return;
                }

                if (srcShape.dims >= 3 && outW == srcShape.w && outH == srcShape.h)
                {
                    new NcnnNoopLayerRepro().ExecuteBuffer(owner, layer, context);
                    return;
                }

                if (srcShape.dims == 3
                    && outC == srcShape.c
                    && CanUsePack4Interp(srcTex, srcShape))
                {
                    var resizeTypePack = layer.GetInt(0, 0);
                    var sxPack = layer.GetFloat(2, 1f);
                    var syPack = layer.GetFloat(1, 1f);
                    var outShape = new NcnnRepro.BufferShape(3, outW, outH, 1, outC);
                    var outRt = owner.RentTempArray(outW, outH, srcTex.packs, RenderTextureFormat.ARGBHalf);
                    var executed = false;

                    if (Mathf.Abs(sxPack - 2f) < 1e-3f && Mathf.Abs(syPack - 2f) < 1e-3f)
                    {
                        if (resizeTypePack == 1)
                            owner.Ops.Interp2xNearestPack4(srcTex.texture, srcTex.packs, outRt);
                        else
                            owner.Ops.Interp2xPack4(srcTex.texture, srcTex.packs, outRt);
                        executed = true;
                    }
                    else if (Mathf.Abs(sxPack - 0.5f) < 1e-3f && Mathf.Abs(syPack - 0.5f) < 1e-3f)
                    {
                        if (resizeTypePack == 1)
                            owner.Ops.InterpDown2NearestPack4(srcTex.texture, srcTex.packs, outRt);
                        else
                            owner.Ops.InterpDown2Pack4(srcTex.texture, srcTex.packs, outRt);
                        executed = true;
                    }
                    else if (resizeTypePack != 1 && resizeTypePack != 3)
                    {
                        var scaleX = outW / (float)Mathf.Max(1, srcTex.width);
                        var scaleY = outH / (float)Mathf.Max(1, srcTex.height);
                        owner.Ops.InterpPack4(srcTex.texture, srcTex.packs, scaleX, scaleY, outRt);
                        executed = true;
                    }

                    if (executed)
                    {
                        NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape);
                        owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                        return;
                    }

                    owner.ReturnTempArray(outRt);
                }
            }

            using var srcReadable = owner.GetReadableTensorInput(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcBuf = srcReadable?.buffer;
            var srcView = srcReadable;
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("Interp source not found: " + layer.name);

            ResolveTargetShape(
                layer,
                new NcnnRepro.BufferShape(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c),
                textureBlobs,
                textureShapes,
                bufferViews,
                out var fallbackOutW,
                out var fallbackOutH,
                out var fallbackOutC);

            if (srcView.dims == 2 && fallbackOutW == srcView.w)
            {
                new NcnnNoopLayerRepro().ExecuteBuffer(owner, layer, context);
                return;
            }

            if (srcView.dims >= 3 && fallbackOutW == srcView.w && fallbackOutH == srcView.h)
            {
                new NcnnNoopLayerRepro().ExecuteBuffer(owner, layer, context);
                return;
            }

            var resizeType = layer.GetInt(0, 0);
            var alignCorner = layer.GetInt(6, 0) != 0;
            var outDims = srcView.dims == 1 ? 3 : srcView.dims;
            var outTensor = srcView.dims == 1
                ? owner.RentTempTensorBuffer(3, fallbackOutW, fallbackOutH, 1, fallbackOutC)
                : owner.RentTempTensorBuffer(srcView.dims, fallbackOutW, fallbackOutH, srcView.d, fallbackOutC);

            var srcData = NcnnRepro.ReadFloatBuffer(srcBuf);
            var outData = new float[outTensor.elementCount];

            if (srcView.dims == 1)
            {
                ApplyInterpDims1(srcData, srcView, fallbackOutW, fallbackOutH, resizeType, alignCorner, outData);
            }
            else if (srcView.dims == 2)
            {
                ApplyInterpDims2(srcData, srcView, fallbackOutW, resizeType, alignCorner, layer, outData);
            }
            else
            {
                ApplyInterpDims3(srcData, srcView, fallbackOutW, fallbackOutH, resizeType, alignCorner, layer, outData);
            }

            outTensor.buffer.SetData(outData);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: outTensor.dims <= 3,
                textureBlobs,
                textureShapes,
                bufferBlobs,
                bufferRefs,
                bufferViews,
                tempOwned);
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
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
            var resizeType = layer.GetInt(0, 0);
            var sy = layer.GetFloat(1, 1f);
            var sx = layer.GetFloat(2, 1f);

            ResolveCmdTargetShape(layer, shapes, blobs, srcShape, sx, sy, out var outW, out var outH);
            var outShape = ResolveCmdOutputShape(srcShape, outW, outH);

            if (IsCmdInterpNoop(srcShape, outShape))
            {
                blobs[layer.topNames[0]] = src;
                src.refs++;
                if (shapes != null)
                    shapes[layer.topNames[0]] = srcShape;
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            if (!CanUsePack4Interp(src, srcShape))
            {
                NcnnRepro.ResolveCmdTextureLayout(outShape, out var placeholderW, out var placeholderH, out var placeholderPacks);
                owner.PublishCmdTensorLikeInput(cmd, layer.topNames[0], placeholderW, placeholderH, placeholderPacks, blobs, shapes, outShape);
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            if (Mathf.Abs(sx - 2f) < 1e-3f && Mathf.Abs(sy - 2f) < 1e-3f)
            {
                var outArr = owner.RentTempArray(cmd, src.width * 2, src.height * 2, src.packs, RenderTextureFormat.ARGBHalf);
                if (resizeType == 1)
                    owner.Ops.Interp2xNearestPack4(cmd, src.texture, src.packs, outArr);
                else
                    owner.Ops.Interp2xPack4(cmd, src.texture, src.packs, outArr);
                blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef { texture = outArr, width = src.width * 2, height = src.height * 2, packs = src.packs, refs = 1, owned = true };
                if (shapes != null)
                    shapes[layer.topNames[0]] = outShape;
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            if (Mathf.Abs(sx - 0.5f) < 1e-3f && Mathf.Abs(sy - 0.5f) < 1e-3f)
            {
                var outArr = owner.RentTempArray(cmd, Mathf.Max(1, src.width / 2), Mathf.Max(1, src.height / 2), src.packs, RenderTextureFormat.ARGBHalf);
                if (resizeType == 1)
                    owner.Ops.InterpDown2NearestPack4(cmd, src.texture, src.packs, outArr);
                else
                    owner.Ops.InterpDown2Pack4(cmd, src.texture, src.packs, outArr);
                blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef { texture = outArr, width = Mathf.Max(1, src.width / 2), height = Mathf.Max(1, src.height / 2), packs = src.packs, refs = 1, owned = true };
                if (shapes != null)
                    shapes[layer.topNames[0]] = outShape;
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            if (resizeType != 1 && resizeType != 3)
            {
                var outArr = owner.RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                var scaleX = outW / (float)Mathf.Max(1, src.width);
                var scaleY = outH / (float)Mathf.Max(1, src.height);
                owner.Ops.InterpPack4(cmd, src.texture, src.packs, scaleX, scaleY, outArr);
                blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef { texture = outArr, width = outW, height = outH, packs = src.packs, refs = 1, owned = true };
                if (shapes != null)
                    shapes[layer.topNames[0]] = outShape;
            }
            else
            {
                NcnnRepro.ResolveCmdTextureLayout(outShape, out var placeholderW, out var placeholderH, out var placeholderPacks);
                owner.PublishCmdTensorLikeInput(cmd, layer.topNames[0], placeholderW, placeholderH, placeholderPacks, blobs, shapes, outShape);
            }

            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static void ResolveCmdTargetShape(
            NcnnParamModel.Layer layer,
            System.Collections.Generic.Dictionary<string, NcnnRepro.BufferShape> shapes,
            System.Collections.Generic.Dictionary<string, NcnnRepro.CmdTensorRef> blobs,
            NcnnRepro.BufferShape srcShape,
            float sx,
            float sy,
            out int outW,
            out int outH)
        {
            var sizeExpr = layer.GetString(9, null);
            if (!string.IsNullOrWhiteSpace(sizeExpr))
            {
                var bottomShapes = new System.Collections.Generic.List<NcnnRepro.BufferShape>(layer.bottomNames.Length);
                for (var i = 0; i < layer.bottomNames.Length; i++)
                    bottomShapes.Add(NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[i]));

                var sizes = NcnnRepro.EvaluateExpressionList(sizeExpr, bottomShapes, layer);
                if (sizes.Count <= 0 || sizes.Count > 2)
                    throw new InvalidOperationException("Interp cmd size_expr rank unsupported: " + layer.name + " | " + sizeExpr);
                outW = Mathf.Max(1, sizes[0]);
                outH = sizes.Count == 1 ? (srcShape.dims >= 2 ? srcShape.h : 1) : Mathf.Max(1, sizes[1]);
                return;
            }

            outW = Mathf.Max(1, layer.GetInt(4, 0));
            outH = Mathf.Max(1, layer.GetInt(3, 0));
            var srcW = srcShape.dims == 1 ? 1 : srcShape.w;
            var srcH = srcShape.dims == 1 ? 1 : (srcShape.dims >= 2 ? srcShape.h : 1);
            if (layer.GetInt(4, 0) == 0)
                outW = Mathf.Max(1, (int)(srcW * Mathf.Max(0f, sx)));
            if (layer.GetInt(3, 0) == 0)
                outH = Mathf.Max(1, (int)(srcH * Mathf.Max(0f, sy)));
        }

        private static bool CanUsePack4Interp(NcnnRepro.CmdTensorRef src, NcnnRepro.BufferShape srcShape)
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

        private static bool CanUsePack4Interp(NcnnRepro.TensorRef src, NcnnRepro.BufferShape srcShape)
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

        private static bool IsCmdInterpNoop(NcnnRepro.BufferShape srcShape, NcnnRepro.BufferShape outShape)
        {
            if (srcShape.dims == 2)
                return outShape.w == srcShape.w;
            if (srcShape.dims >= 3)
                return outShape.w == srcShape.w && outShape.h == srcShape.h;
            return false;
        }

        private static NcnnRepro.BufferShape ResolveCmdOutputShape(NcnnRepro.BufferShape srcShape, int outW, int outH)
        {
            if (srcShape.dims == 1)
                return new NcnnRepro.BufferShape(3, outW, outH, 1, srcShape.w);
            if (srcShape.dims == 2)
                return new NcnnRepro.BufferShape(2, outW, srcShape.h, 1, 1);
            return new NcnnRepro.BufferShape(srcShape.dims, outW, outH, srcShape.d, srcShape.c);
        }

        private static void ResolveTargetShape(
            NcnnParamModel.Layer layer,
            NcnnRepro.BufferShape srcShape,
            Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            out int outW,
            out int outH,
            out int outC)
        {
            var resizeType = layer.GetInt(0, 0);
            var sx = layer.GetFloat(2, 1f);
            var sy = layer.GetFloat(1, 1f);
            var outputHeight = layer.GetInt(3, 0);
            var outputWidth = layer.GetInt(4, 0);
            var dynamicTargetSize = layer.GetInt(5, 0) != 0;
            var sizeExpr = layer.GetString(9, null);

            if (!string.IsNullOrWhiteSpace(sizeExpr))
            {
                var bottomShapes = new List<NcnnRepro.BufferShape>(layer.bottomNames.Length);
                for (var i = 0; i < layer.bottomNames.Length; i++)
                {
                    if (!TryGetBottomShape(layer.bottomNames[i], textureBlobs, textureShapes, bufferViews, out var bottomShape))
                        throw new InvalidOperationException("Interp size_expr bottom shape unavailable: " + layer.name + " | " + layer.bottomNames[i]);
                    bottomShapes.Add(bottomShape);
                }

                var sizes = NcnnRepro.EvaluateExpressionList(sizeExpr, bottomShapes, layer);
                if (sizes.Count <= 0 || sizes.Count > 2)
                    throw new InvalidOperationException("Interp size_expr rank unsupported: " + layer.name + " | " + sizeExpr);

                outW = Mathf.Max(1, sizes[0]);
                outH = sizes.Count == 1
                    ? (srcShape.dims >= 2 ? srcShape.h : 1)
                    : Mathf.Max(1, sizes[1]);
                outC = srcShape.dims >= 3 ? srcShape.c : 1;
                if (srcShape.dims == 1)
                    outC = srcShape.w;
                return;
            }

            if (srcShape.dims == 1)
            {
                var srcW = 1;
                var srcH = 1;
                outW = outputWidth;
                outH = outputHeight;

                if (dynamicTargetSize && layer.bottomNames.Length > 1 && TryGetBottomShape(layer.bottomNames[1], textureBlobs, textureShapes, bufferViews, out var refShape1))
                {
                    outW = refShape1.w;
                    outH = refShape1.h;
                }

                if (outW == 0 || outH == 0)
                {
                    outW = Mathf.Max(1, (int)(srcW * Mathf.Max(0f, sx)));
                    outH = Mathf.Max(1, (int)(srcH * Mathf.Max(0f, sy)));
                }

                outC = srcShape.w;
                return;
            }

            if (srcShape.dims == 2)
            {
                outW = outputWidth;
                if (dynamicTargetSize && layer.bottomNames.Length > 1 && TryGetBottomShape(layer.bottomNames[1], textureBlobs, textureShapes, bufferViews, out var refShape2))
                    outW = refShape2.w;
                if (outW == 0)
                    outW = Mathf.Max(1, (int)(srcShape.w * Mathf.Max(0f, sx)));
                outH = srcShape.h;
                outC = 1;
                return;
            }

            outW = outputWidth;
            outH = outputHeight;
            if (dynamicTargetSize && layer.bottomNames.Length > 1 && TryGetBottomShape(layer.bottomNames[1], textureBlobs, textureShapes, bufferViews, out var refShape3))
            {
                outW = refShape3.w;
                outH = refShape3.h;
            }

            if (outW == 0)
                outW = Mathf.Max(1, (int)(srcShape.w * Mathf.Max(0f, sx)));
            if (outH == 0)
                outH = Mathf.Max(1, (int)(srcShape.h * Mathf.Max(0f, sy)));
            outC = srcShape.c;

            _ = resizeType;
        }

        private static bool TryGetBottomShape(
            string name,
            Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            out NcnnRepro.BufferShape shape)
        {
            shape = default;
            if (bufferViews != null && bufferViews.TryGetValue(name, out var view) && view != null)
            {
                shape = new NcnnRepro.BufferShape(view.dims, view.w, view.h, view.d, view.c);
                return true;
            }

            if (NcnnRepro.TryGetExistingTexture(textureBlobs, textureShapes, name, out _, out shape))
                return true;

            return false;
        }

        private static void ApplyInterpDims1(float[] srcData, NcnnTensorBuffer srcView, int outW, int outH, int resizeType, bool alignCorner, float[] outData)
        {
            var channels = srcView.w;
            var outPlane = outW * outH;
            for (var c = 0; c < channels; c++)
            {
                var value = srcData[c];
                var baseIndex = c * outPlane;
                for (var i = 0; i < outPlane; i++)
                    outData[baseIndex + i] = value;
            }
        }

        private static void ApplyInterpDims2(float[] srcData, NcnnTensorBuffer srcView, int outW, int resizeType, bool alignCorner, NcnnParamModel.Layer layer, float[] outData)
        {
            var useTargetWidth = layer.GetInt(4, 0) != 0;
            var ws = (resizeType == 2 || useTargetWidth) ? srcView.w / (float)outW : 1f / Mathf.Max(layer.GetFloat(2, 1f), 1e-6f);
            if (resizeType == 2 && alignCorner && outW > 1 && srcView.w > 1)
                ws = (srcView.w - 1) / (float)(outW - 1);

            for (var y = 0; y < srcView.h; y++)
            {
                var srcRow = y * srcView.w;
                var outRow = y * outW;
                for (var x = 0; x < outW; x++)
                {
                    if (resizeType == 1)
                    {
                        var wsNearest = useTargetWidth ? srcView.w / (float)outW : 1f / Mathf.Max(layer.GetFloat(2, 1f), 1e-6f);
                        var inX = Mathf.Min((int)(x * wsNearest), srcView.w - 1);
                        outData[outRow + x] = srcData[srcRow + inX];
                    }
                    else if (resizeType == 3)
                    {
                        outData[outRow + x] = SampleBicubic1D(srcData, srcRow, srcView.w, x, outW, alignCorner);
                    }
                    else
                    {
                        outData[outRow + x] = SampleLinear1D(srcData, srcRow, srcView.w, x, outW, alignCorner);
                    }
                }
            }
        }

        private static void ApplyInterpDims3(float[] srcData, NcnnTensorBuffer srcView, int outW, int outH, int resizeType, bool alignCorner, NcnnParamModel.Layer layer, float[] outData)
        {
            var srcPlane = srcView.w * srcView.h;
            var outPlane = outW * outH;
            var useTargetHeight = layer.GetInt(3, 0) != 0;
            var useTargetWidth = layer.GetInt(4, 0) != 0;
            var hsNearest = useTargetHeight ? srcView.h / (float)outH : 1f / Mathf.Max(layer.GetFloat(1, 1f), 1e-6f);
            var wsNearest = useTargetWidth ? srcView.w / (float)outW : 1f / Mathf.Max(layer.GetFloat(2, 1f), 1e-6f);

            for (var c = 0; c < srcView.c; c++)
            {
                var srcBase = c * srcPlane;
                var outBase = c * outPlane;
                for (var oy = 0; oy < outH; oy++)
                {
                    for (var ox = 0; ox < outW; ox++)
                    {
                        var outIndex = outBase + oy * outW + ox;
                        if (resizeType == 1)
                        {
                            var inY = Mathf.Min((int)(oy * hsNearest), srcView.h - 1);
                            var inX = Mathf.Min((int)(ox * wsNearest), srcView.w - 1);
                            outData[outIndex] = srcData[srcBase + inY * srcView.w + inX];
                        }
                        else if (resizeType == 3)
                        {
                            outData[outIndex] = SampleBicubic2D(srcData, srcBase, srcView.w, srcView.h, ox, oy, outW, outH, alignCorner);
                        }
                        else
                        {
                            outData[outIndex] = SampleLinear2D(srcData, srcBase, srcView.w, srcView.h, ox, oy, outW, outH, alignCorner);
                        }
                    }
                }
            }
        }

        private static float SampleLinear1D(float[] data, int rowBase, int w, int x, int outW, bool alignCorner)
        {
            if (w <= 1 || outW <= 1)
                return data[rowBase];

            var scale = alignCorner && outW > 1 ? (w - 1) / (float)(outW - 1) : w / (float)outW;
            var fx = alignCorner ? x * scale : ((x + 0.5f) * scale - 0.5f);
            var sx = Mathf.FloorToInt(fx);
            fx -= sx;
            if (sx < 0)
            {
                sx = 0;
                fx = 0f;
            }
            if (sx >= w - 1)
            {
                sx = Mathf.Max(0, w - 2);
                fx = 1f;
            }

            var a0 = 1f - fx;
            var a1 = fx;
            return data[rowBase + sx] * a0 + data[rowBase + sx + 1] * a1;
        }

        private static float SampleLinear2D(float[] data, int baseIndex, int w, int h, int x, int y, int outW, int outH, bool alignCorner)
        {
            if (w <= 1 && h <= 1)
                return data[baseIndex];

            var scaleX = alignCorner && outW > 1 ? (w - 1) / (float)(outW - 1) : w / (float)outW;
            var scaleY = alignCorner && outH > 1 ? (h - 1) / (float)(outH - 1) : h / (float)outH;
            var fx = alignCorner ? x * scaleX : ((x + 0.5f) * scaleX - 0.5f);
            var fy = alignCorner ? y * scaleY : ((y + 0.5f) * scaleY - 0.5f);
            var x0 = Mathf.FloorToInt(fx);
            var y0 = Mathf.FloorToInt(fy);
            var tx = fx - x0;
            var ty = fy - y0;

            if (w <= 1)
            {
                x0 = 0;
                tx = 0f;
            }
            else
            {
                if (x0 < 0) { x0 = 0; tx = 0f; }
                if (x0 >= w - 1) { x0 = Mathf.Max(0, w - 2); tx = 1f; }
            }

            if (h <= 1)
            {
                y0 = 0;
                ty = 0f;
            }
            else
            {
                if (y0 < 0) { y0 = 0; ty = 0f; }
                if (y0 >= h - 1) { y0 = Mathf.Max(0, h - 2); ty = 1f; }
            }

            var x1 = Mathf.Min(x0 + 1, w - 1);
            var y1 = Mathf.Min(y0 + 1, h - 1);
            var v00 = data[baseIndex + y0 * w + x0];
            var v10 = data[baseIndex + y0 * w + x1];
            var v01 = data[baseIndex + y1 * w + x0];
            var v11 = data[baseIndex + y1 * w + x1];
            var vx0 = Mathf.Lerp(v00, v10, tx);
            var vx1 = Mathf.Lerp(v01, v11, tx);
            return Mathf.Lerp(vx0, vx1, ty);
        }

        private static float SampleBicubic1D(float[] data, int rowBase, int w, int x, int outW, bool alignCorner)
        {
            if (w <= 1)
                return data[rowBase];
            if (w <= 3)
                return SampleLinear1D(data, rowBase, w, x, outW, alignCorner);

            var coeffs = new float[4];
            var scale = alignCorner && outW > 1 ? (w - 1) / (float)(outW - 1) : w / (float)outW;
            var fx = alignCorner ? x * scale : ((x + 0.5f) * scale - 0.5f);
            var sx = Mathf.FloorToInt(fx);
            fx -= sx;
            InterpolateCubic(fx, coeffs);
            AdjustCubicIndex(ref sx, coeffs, w);

            return data[rowBase + sx - 1] * coeffs[0]
                + data[rowBase + sx + 0] * coeffs[1]
                + data[rowBase + sx + 1] * coeffs[2]
                + data[rowBase + sx + 2] * coeffs[3];
        }

        private static float SampleBicubic2D(float[] data, int baseIndex, int w, int h, int x, int y, int outW, int outH, bool alignCorner)
        {
            if (w <= 3 || h <= 3)
                return SampleLinear2D(data, baseIndex, w, h, x, y, outW, outH, alignCorner);

            var alpha = new float[4];
            var beta = new float[4];
            var scaleX = alignCorner && outW > 1 ? (w - 1) / (float)(outW - 1) : w / (float)outW;
            var scaleY = alignCorner && outH > 1 ? (h - 1) / (float)(outH - 1) : h / (float)outH;
            var fx = alignCorner ? x * scaleX : ((x + 0.5f) * scaleX - 0.5f);
            var fy = alignCorner ? y * scaleY : ((y + 0.5f) * scaleY - 0.5f);
            var sx = Mathf.FloorToInt(fx);
            var sy = Mathf.FloorToInt(fy);
            fx -= sx;
            fy -= sy;
            InterpolateCubic(fx, alpha);
            InterpolateCubic(fy, beta);
            AdjustCubicIndex(ref sx, alpha, w);
            AdjustCubicIndex(ref sy, beta, h);

            float sum = 0f;
            for (var ky = 0; ky < 4; ky++)
            {
                var py = sy + ky - 1;
                for (var kx = 0; kx < 4; kx++)
                {
                    var px = sx + kx - 1;
                    sum += data[baseIndex + py * w + px] * alpha[kx] * beta[ky];
                }
            }
            return sum;
        }

        private static void InterpolateCubic(float fx, float[] coeffs)
        {
            const float a = -0.75f;
            var fx0 = fx + 1f;
            var fx1 = fx;
            var fx2 = 1f - fx;

            coeffs[0] = a * fx0 * fx0 * fx0 - 5f * a * fx0 * fx0 + 8f * a * fx0 - 4f * a;
            coeffs[1] = (a + 2f) * fx1 * fx1 * fx1 - (a + 3f) * fx1 * fx1 + 1f;
            coeffs[2] = (a + 2f) * fx2 * fx2 * fx2 - (a + 3f) * fx2 * fx2 + 1f;
            coeffs[3] = 1f - coeffs[0] - coeffs[1] - coeffs[2];
        }

        private static void AdjustCubicIndex(ref int sx, float[] coeffs, int w)
        {
            if (sx <= -1)
            {
                sx = 1;
                coeffs[0] = 1f - coeffs[3];
                coeffs[1] = coeffs[3];
                coeffs[2] = 0f;
                coeffs[3] = 0f;
            }
            if (sx == 0)
            {
                sx = 1;
                coeffs[0] = coeffs[0] + coeffs[1];
                coeffs[1] = coeffs[2];
                coeffs[2] = coeffs[3];
                coeffs[3] = 0f;
            }
            if (sx == w - 2)
            {
                sx = w - 3;
                coeffs[3] = coeffs[2] + coeffs[3];
                coeffs[2] = coeffs[1];
                coeffs[1] = coeffs[0];
                coeffs[0] = 0f;
            }
            if (sx >= w - 1)
            {
                sx = w - 3;
                coeffs[3] = 1f - coeffs[0];
                coeffs[2] = coeffs[0];
                coeffs[1] = 0f;
                coeffs[0] = 0f;
            }
        }
    }
}
