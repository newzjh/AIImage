using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Ncnn
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnUnfoldLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnUnfoldLayerRepro()
            : base(NcnnLayerTypes.Unfold, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override NcnnGraphSession.LayerLoadMetrics LoadLayer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            owner._extraPacks[layer.name] = new NcnnGraphSession.UnfoldPack
            {
                kernelW = layer.GetInt(1, 0),
                kernelH = layer.GetInt(11, layer.GetInt(1, 0)),
                dilationW = layer.GetInt(2, 1),
                dilationH = layer.GetInt(12, layer.GetInt(2, 1)),
                strideW = layer.GetInt(3, 1),
                strideH = layer.GetInt(13, layer.GetInt(3, 1)),
                padLeft = layer.GetInt(4, 0),
                padRight = layer.GetInt(15, layer.GetInt(4, 0)),
                padTop = layer.GetInt(14, layer.GetInt(4, 0)),
                padBottom = layer.GetInt(16, layer.GetInt(14, layer.GetInt(4, 0))),
                padValue = layer.GetFloat(18, 0f)
            };
            return default;
        }

        public override void ExecuteBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnGraphSession.UnfoldPack up)
                throw new InvalidOperationException("Unfold pack not found: " + layer.name);

            if (!owner.ShouldForceCurrentLayerBufferPath() && CanUseTexturePath(layer, context, up))
            {
                ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnGraphSession.UnfoldPack up)
                throw new InvalidOperationException("Unfold pack not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null || (srcView.dims != 2 && srcView.dims != 3))
                throw new InvalidOperationException("Unfold expects dims=2 or dims=3 source: " + layer.name);

            var inW = srcView.w;
            var inH = srcView.h;
            var inC = srcView.dims == 3 ? srcView.c : 1;
            var kernelExtentW = up.dilationW * (up.kernelW - 1) + 1;
            var kernelExtentH = up.dilationH * (up.kernelH - 1) + 1;

            ResolvePadding(inW, inH, kernelExtentW, kernelExtentH, up, out var padLeft, out var padRight, out var padTop, out var padBottom);

            var paddedW = inW + padLeft + padRight;
            var paddedH = inH + padTop + padBottom;
            var outw = (paddedW - kernelExtentW) / up.strideW + 1;
            var outh = (paddedH - kernelExtentH) / up.strideH + 1;
            var size = outw * outh;
            var maxk = up.kernelW * up.kernelH;
            var outRows = maxk * inC;

            var outTensor = owner.RentTempTensorBuffer(2, size, outRows);
            owner.Ops.UnfoldBuf(
                srcBuf,
                inW,
                inH,
                inC,
                outw,
                outh,
                up.kernelW,
                up.kernelH,
                up.dilationW,
                up.dilationH,
                up.strideW,
                up.strideH,
                padLeft,
                padTop,
                up.padValue,
                outTensor.buffer);
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

        public override void ExecuteRenderTexturePath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnGraphSession.UnfoldPack up)
                throw new InvalidOperationException("Unfold pack not found: " + layer.name);
            if (!textureBlobs.TryGetValue(layer.bottomNames[0], out var src) || src == null || src.texture == null)
                throw new InvalidOperationException("Unfold render-texture path requires texture input: " + layer.name);

            var srcShape = NcnnGraphSession.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
            if (!CanUseTexturePath(src, srcShape, up))
                throw new InvalidOperationException("Unfold render-texture path requires dims=2/3 supported texture input: " + layer.name);

            ResolveOutputGeometry(srcShape, up, layer.name, out var inW, out var inH, out var inC, out var outw, out var outh, out var size, out var outRows, out var padLeft, out var padTop);
            var outShape = new NcnnGraphSession.BufferShape(2, size, outRows, 1, 1);
            var storageShape = NcnnGraphSession.GetTextureStorageShape(src, srcShape);
            RenderTexture materializedInput = null;
            RenderTexture output = null;
            try
            {
                var input = src.texture;
                if (NcnnGraphSession.IsStrictLinearMatTexture(src))
                {
                    materializedInput = owner.RentTempArray(inW, inH, Mathf.Max(1, Mathf.CeilToInt(inC / 4f)), NcnnGraphSession.ResolveTensorTextureFormat(3));
                    owner.Ops.ReshapeLinearMatToPack4(
                        src.texture,
                        storageShape.w,
                        storageShape.h,
                        inW,
                        inH,
                        1,
                        inC,
                        3,
                        materializedInput);
                    input = materializedInput;
                }

                output = owner.RentTempArray(outShape.w, outShape.h, 1, NcnnGraphSession.ResolveTensorTextureFormat(outShape.dims));
                owner.Ops.UnfoldPack4(
                    input,
                    inW,
                    inH,
                    inC,
                    outw,
                    outh,
                    up.kernelW,
                    up.kernelH,
                    up.dilationW,
                    up.dilationH,
                    up.strideW,
                    up.strideH,
                    padLeft,
                    padTop,
                    up.padValue,
                    output);

                NcnnGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], output, outShape);
                output = null;
            }
            finally
            {
                if (materializedInput != null)
                    owner.ReturnTempArray(materializedInput);
                if (output != null)
                    owner.ReturnTempArray(output);
            }

            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnGraphSession.UnfoldPack up)
                throw new InvalidOperationException("Unfold pack not found: " + layer.name);

            var srcShape = NcnnGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var src = NcnnGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            if (!CanUseTexturePath(src, srcShape, up))
                throw new InvalidOperationException("Unfold command-buffer path requires dims=2/3 supported texture input: " + layer.name);

            ResolveOutputGeometry(srcShape, up, layer.name, out var inW, out var inH, out var inC, out var outw, out var outh, out var size, out var outRows, out var padLeft, out var padTop);
            var outShape = new NcnnGraphSession.BufferShape(2, size, outRows, 1, 1);
            var storageShape = NcnnGraphSession.GetCmdStorageShape(src, srcShape);
            ComputeTexture materializedInput = null;
            ComputeTexture output = null;
            try
            {
                var input = src.texture;
                if (NcnnGraphSession.IsStrictLinearMatTexture(src))
                {
                    materializedInput = owner.RentTempArray(cmd, inW, inH, Mathf.Max(1, Mathf.CeilToInt(inC / 4f)), NcnnGraphSession.ResolveTensorTextureFormat(3));
                    owner.Ops.ReshapeLinearMatToPack4(
                        cmd,
                        src.texture,
                        storageShape.w,
                        storageShape.h,
                        inW,
                        inH,
                        1,
                        inC,
                        3,
                        materializedInput);
                    input = materializedInput;
                }

                output = owner.RentTempArray(cmd, outShape.w, outShape.h, 1, NcnnGraphSession.ResolveTensorTextureFormat(outShape.dims));
                owner.Ops.UnfoldPack4(
                    cmd,
                    input,
                    inW,
                    inH,
                    inC,
                    outw,
                    outh,
                    up.kernelW,
                    up.kernelH,
                    up.dilationW,
                    up.dilationH,
                    up.strideW,
                    up.strideH,
                    padLeft,
                    padTop,
                    up.padValue,
                    output);

                blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorRef(output, outShape, outShape, owned: true);
                output = null;
                if (shapes != null)
                    shapes[layer.topNames[0]] = outShape;
            }
            finally
            {
                if (materializedInput != null)
                    owner.ReturnTempArray(cmd, materializedInput);
                if (output != null)
                    owner.ReturnTempArray(cmd, output);
            }

            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool CanUseTexturePath(NcnnParamModel.Layer layer, NcnnLayerBufferContext context, NcnnGraphSession.UnfoldPack up)
        {
            if (context?.textureBlobs == null || layer?.bottomNames == null || layer.bottomNames.Length == 0)
                return false;
            if (!context.textureBlobs.TryGetValue(layer.bottomNames[0], out var src) || src == null || src.texture == null)
                return false;
            var shape = NcnnGraphSession.GetTextureShape(context.textureShapes, src, layer.bottomNames[0]);
            return CanUseTexturePath(src, shape, up);
        }

        private static bool CanUseTexturePath(NcnnGraphSession.TensorRef src, NcnnGraphSession.BufferShape srcShape, NcnnGraphSession.UnfoldPack up)
        {
            if (src == null || src.texture == null || up == null)
                return false;
            if (srcShape.dims != 2 && srcShape.dims != 3)
                return false;
            if (srcShape.w <= 0 || srcShape.h <= 0 || up.kernelW <= 0 || up.kernelH <= 0)
                return false;
            if (up.strideW <= 0 || up.strideH <= 0 || up.dilationW <= 0 || up.dilationH <= 0)
                return false;
            if (NcnnGraphSession.IsStrictLinearMatTexture(src))
                return true;
            if (srcShape.dims == 2)
            {
                var storageShape = NcnnGraphSession.GetTextureStorageShape(src, srcShape);
                return storageShape.dims == 2
                    && storageShape.w == src.width
                    && storageShape.h == src.height
                    && src.packs == 1;
            }

            return NcnnGraphSession.MatchesPack4TextureStorage(src, srcShape);
        }

        private static bool CanUseTexturePath(NcnnGraphSession.CmdTensorRef src, NcnnGraphSession.BufferShape srcShape, NcnnGraphSession.UnfoldPack up)
        {
            if (src == null || src.texture == null || up == null)
                return false;
            if (srcShape.dims != 2 && srcShape.dims != 3)
                return false;
            if (srcShape.w <= 0 || srcShape.h <= 0 || up.kernelW <= 0 || up.kernelH <= 0)
                return false;
            if (up.strideW <= 0 || up.strideH <= 0 || up.dilationW <= 0 || up.dilationH <= 0)
                return false;
            if (NcnnGraphSession.IsStrictLinearMatTexture(src))
                return true;
            if (srcShape.dims == 2)
            {
                var storageShape = NcnnGraphSession.GetCmdStorageShape(src, srcShape);
                return storageShape.dims == 2
                    && storageShape.w == src.width
                    && storageShape.h == src.height
                    && src.packs == 1;
            }

            return NcnnGraphSession.MatchesPack4TextureStorage(src, srcShape);
        }

        private static void ResolveOutputGeometry(
            NcnnGraphSession.BufferShape srcShape,
            NcnnGraphSession.UnfoldPack up,
            string layerName,
            out int inW,
            out int inH,
            out int inC,
            out int outw,
            out int outh,
            out int size,
            out int outRows,
            out int padLeft,
            out int padTop)
        {
            inW = srcShape.w;
            inH = srcShape.h;
            inC = srcShape.dims == 3 ? Mathf.Max(1, srcShape.c) : 1;
            var kernelExtentW = up.dilationW * (up.kernelW - 1) + 1;
            var kernelExtentH = up.dilationH * (up.kernelH - 1) + 1;
            ResolvePadding(inW, inH, kernelExtentW, kernelExtentH, up, out padLeft, out var padRight, out padTop, out var padBottom);
            var paddedW = inW + padLeft + padRight;
            var paddedH = inH + padTop + padBottom;
            outw = (paddedW - kernelExtentW) / up.strideW + 1;
            outh = (paddedH - kernelExtentH) / up.strideH + 1;
            if (outw <= 0 || outh <= 0)
                throw new InvalidOperationException("Unfold output shape is empty: " + layerName);
            size = Mathf.Max(1, outw * outh);
            outRows = Mathf.Max(1, up.kernelW * up.kernelH * inC);
        }

        private static void ResolvePadding(
            int w,
            int h,
            int kernelExtentW,
            int kernelExtentH,
            NcnnGraphSession.UnfoldPack pack,
            out int padLeft,
            out int padRight,
            out int padTop,
            out int padBottom)
        {
            padLeft = pack.padLeft;
            padRight = pack.padRight;
            padTop = pack.padTop;
            padBottom = pack.padBottom;

            if (padLeft > 0 || padRight > 0 || padTop > 0 || padBottom > 0)
                return;

            if (padLeft == -233 && padRight == -233 && padTop == -233 && padBottom == -233)
            {
                var wpad = kernelExtentW + (w - 1) / pack.strideW * pack.strideW - w;
                var hpad = kernelExtentH + (h - 1) / pack.strideH * pack.strideH - h;
                padLeft = Mathf.Max(0, wpad / 2);
                padRight = Mathf.Max(0, wpad - padLeft);
                padTop = Mathf.Max(0, hpad / 2);
                padBottom = Mathf.Max(0, hpad - padTop);
                return;
            }

            if (padLeft == -234 && padRight == -234 && padTop == -234 && padBottom == -234)
            {
                var wpad = kernelExtentW + (w - 1) / pack.strideW * pack.strideW - w;
                var hpad = kernelExtentH + (h - 1) / pack.strideH * pack.strideH - h;
                padRight = Mathf.Max(0, wpad / 2);
                padLeft = Mathf.Max(0, wpad - padRight);
                padBottom = Mathf.Max(0, hpad / 2);
                padTop = Mathf.Max(0, hpad - padBottom);
                return;
            }

            padLeft = 0;
            padRight = 0;
            padTop = 0;
            padBottom = 0;
        }
    }
}
