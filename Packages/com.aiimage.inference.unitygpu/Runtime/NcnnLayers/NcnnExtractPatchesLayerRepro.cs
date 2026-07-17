using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // TensorFlow/ONNX ExtractImagePatches lowered as a native texture layer.
    // Output layout matches the sparse Conv lowering used by the reference
    // DeepFillV2 HiFill graph: dims=3, w=outW, h=outH, c=kh*kw*inC, with
    // channel order [ky, kx, channel].
    public sealed class NcnnExtractPatchesLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnExtractPatchesLayerRepro()
            : base(NcnnLayerTypes.ExtractPatches, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
        }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            owner._extraPacks[layer.name] = new NcnnRepro.UnfoldPack
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

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            ExecuteRenderTexturePath(owner, layer, context);
        }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.UnfoldPack pack)
                throw new InvalidOperationException("ExtractPatches pack not found: " + layer.name);

            if (!owner.TryGetPack4Texture(layer.bottomNames[0], context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, out var src, out var srcShape))
            {
                throw new InvalidOperationException(
                    "ExtractPatches render-texture path requires pack4 texture input"
                    + " | layer=" + layer.name
                    + " | bottom=" + (layer.bottomNames != null && layer.bottomNames.Length > 0 ? layer.bottomNames[0] : string.Empty));
            }

            if (!CanUsePack4(src, srcShape, pack))
                throw new InvalidOperationException(DescribeUnsupported("ExtractPatches render-texture path", layer, src, srcShape, pack));

            ResolveOutputShape(srcShape, pack, layer.name, out var outShape);
            var foldedD = srcShape.dims == 4 && IsFoldDStorage(src, srcShape);
            var outStorageShape = foldedD ? ResolveFoldDStorageShape(outShape) : outShape;
            var outDepth = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            RenderTexture output = null;
            try
            {
                output = owner.RentTempArray(outStorageShape.w, outStorageShape.h, outDepth, NcnnRepro.ResolveTensorTextureFormat(outShape.dims));
                if (foldedD)
                {
                    owner.Ops.ExtractPatchesFoldDPack4(
                        src.texture,
                        srcShape.w,
                        srcShape.h,
                        srcShape.d,
                        srcShape.c,
                        outShape.w,
                        outShape.h,
                        pack.kernelW,
                        pack.kernelH,
                        pack.dilationW,
                        pack.dilationH,
                        pack.strideW,
                        pack.strideH,
                        ResolveSameOrExplicitPadLeft(srcShape, pack),
                        ResolveSameOrExplicitPadTop(srcShape, pack),
                        pack.padValue,
                        output);
                }
                else
                {
                    owner.Ops.ExtractPatchesPack4(
                        src.texture,
                        srcShape.w,
                        srcShape.h,
                        srcShape.c,
                        outShape.w,
                        outShape.h,
                        pack.kernelW,
                        pack.kernelH,
                        pack.dilationW,
                        pack.dilationH,
                        pack.strideW,
                        pack.strideH,
                        ResolveSameOrExplicitPadLeft(srcShape, pack),
                        ResolveSameOrExplicitPadTop(srcShape, pack),
                        pack.padValue,
                        output);
                }

                NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, outShape, outStorageShape);
                output = null;
            }
            finally
            {
                if (output != null)
                    owner.ReturnTempArray(output);
            }

            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.UnfoldPack pack)
                throw new InvalidOperationException("ExtractPatches pack not found: " + layer.name);

            var src = NcnnRepro.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var srcShape = NcnnRepro.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            if (!CanUsePack4(src, srcShape, pack))
                throw new InvalidOperationException(DescribeUnsupported("ExtractPatches command-buffer path", layer, src, srcShape, pack));

            ResolveOutputShape(srcShape, pack, layer.name, out var outShape);
            var foldedD = srcShape.dims == 4 && IsFoldDStorage(src, srcShape);
            var outStorageShape = foldedD ? ResolveFoldDStorageShape(outShape) : outShape;
            var outDepth = Mathf.Max(1, Mathf.CeilToInt(outShape.c / 4f));
            ComputeTexture output = null;
            try
            {
                output = owner.RentTempArray(context.commandBuffer, outStorageShape.w, outStorageShape.h, outDepth, NcnnRepro.ResolveTensorTextureFormat(outShape.dims));
                if (foldedD)
                {
                    owner.Ops.ExtractPatchesFoldDPack4(
                        context.commandBuffer,
                        src.texture,
                        srcShape.w,
                        srcShape.h,
                        srcShape.d,
                        srcShape.c,
                        outShape.w,
                        outShape.h,
                        pack.kernelW,
                        pack.kernelH,
                        pack.dilationW,
                        pack.dilationH,
                        pack.strideW,
                        pack.strideH,
                        ResolveSameOrExplicitPadLeft(srcShape, pack),
                        ResolveSameOrExplicitPadTop(srcShape, pack),
                        pack.padValue,
                        output);
                }
                else
                {
                    owner.Ops.ExtractPatchesPack4(
                        context.commandBuffer,
                        src.texture,
                        srcShape.w,
                        srcShape.h,
                        srcShape.c,
                        outShape.w,
                        outShape.h,
                        pack.kernelW,
                        pack.kernelH,
                        pack.dilationW,
                        pack.dilationH,
                        pack.strideW,
                        pack.strideH,
                        ResolveSameOrExplicitPadLeft(srcShape, pack),
                        ResolveSameOrExplicitPadTop(srcShape, pack),
                        pack.padValue,
                        output);
                }

                context.blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(output, outShape, outStorageShape, owned: true, blobName: layer.topNames[0]);
                if (context.shapes != null)
                    context.shapes[layer.topNames[0]] = outShape;
                output = null;
            }
            finally
            {
                if (output != null)
                    owner.ReturnTempArray(context.commandBuffer, output);
            }

            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private static bool CanUsePack4(NcnnRepro.TensorRef src, NcnnRepro.BufferShape srcShape, NcnnRepro.UnfoldPack pack)
        {
            return src != null
                   && src.texture != null
                   && (srcShape.dims == 3 || srcShape.dims == 4)
                   && srcShape.w > 0
                   && srcShape.h > 0
                   && srcShape.c > 0
                   && pack != null
                   && pack.kernelW > 0
                   && pack.kernelH > 0
                   && pack.dilationW > 0
                   && pack.dilationH > 0
                   && pack.strideW > 0
                   && pack.strideH > 0
                   && (srcShape.dims == 3
                       ? NcnnRepro.MatchesPack4TextureStorage(src, srcShape)
                       : IsFoldDStorage(src, srcShape));
        }

        private static bool CanUsePack4(NcnnRepro.CmdTensorRef src, NcnnRepro.BufferShape srcShape, NcnnRepro.UnfoldPack pack)
        {
            return src != null
                   && src.texture != null
                   && (srcShape.dims == 3 || srcShape.dims == 4)
                   && srcShape.w > 0
                   && srcShape.h > 0
                   && srcShape.c > 0
                   && pack != null
                   && pack.kernelW > 0
                   && pack.kernelH > 0
                   && pack.dilationW > 0
                   && pack.dilationH > 0
                   && pack.strideW > 0
                   && pack.strideH > 0
                   && (srcShape.dims == 3
                       ? NcnnRepro.MatchesPack4TextureStorage(src, srcShape)
                       : IsFoldDStorage(src, srcShape));
        }

        private static void ResolveOutputShape(NcnnRepro.BufferShape srcShape, NcnnRepro.UnfoldPack pack, string layerName, out NcnnRepro.BufferShape outShape)
        {
            var kernelExtentW = pack.dilationW * (pack.kernelW - 1) + 1;
            var kernelExtentH = pack.dilationH * (pack.kernelH - 1) + 1;
            ResolvePadding(srcShape.w, srcShape.h, kernelExtentW, kernelExtentH, pack, out var padLeft, out var padRight, out var padTop, out var padBottom);
            var outW = (srcShape.w + padLeft + padRight - kernelExtentW) / pack.strideW + 1;
            var outH = (srcShape.h + padTop + padBottom - kernelExtentH) / pack.strideH + 1;
            if (outW <= 0 || outH <= 0)
                throw new InvalidOperationException("ExtractPatches output shape is empty: " + layerName);
            var outC = Mathf.Max(1, srcShape.c * pack.kernelW * pack.kernelH);
            outShape = srcShape.dims == 4
                ? new NcnnRepro.BufferShape(4, outW, outH, Mathf.Max(1, srcShape.d), outC)
                : new NcnnRepro.BufferShape(3, outW, outH, 1, outC);
        }

        private static int ResolveSameOrExplicitPadLeft(NcnnRepro.BufferShape srcShape, NcnnRepro.UnfoldPack pack)
        {
            var kernelExtentW = pack.dilationW * (pack.kernelW - 1) + 1;
            ResolvePadding(srcShape.w, srcShape.h, kernelExtentW, pack.dilationH * (pack.kernelH - 1) + 1, pack, out var padLeft, out _, out _, out _);
            return padLeft;
        }

        private static int ResolveSameOrExplicitPadTop(NcnnRepro.BufferShape srcShape, NcnnRepro.UnfoldPack pack)
        {
            var kernelExtentH = pack.dilationH * (pack.kernelH - 1) + 1;
            ResolvePadding(srcShape.w, srcShape.h, pack.dilationW * (pack.kernelW - 1) + 1, kernelExtentH, pack, out _, out _, out var padTop, out _);
            return padTop;
        }

        private static void ResolvePadding(
            int w,
            int h,
            int kernelExtentW,
            int kernelExtentH,
            NcnnRepro.UnfoldPack pack,
            out int padLeft,
            out int padRight,
            out int padTop,
            out int padBottom)
        {
            padLeft = pack.padLeft;
            padRight = pack.padRight;
            padTop = pack.padTop;
            padBottom = pack.padBottom;

            // Keep the converter free to emit explicit NCNN pads. When only the
            // left/top side is present, this also matches NCNN's default symmetric
            // interpretation for older Unfold-style params.
            if (padRight < 0 || padBottom < 0)
                throw new InvalidOperationException("ExtractPatches does not support negative padding.");
        }

        private static string DescribeUnsupported(string prefix, NcnnParamModel.Layer layer, NcnnRepro.TensorRef src, NcnnRepro.BufferShape shape, NcnnRepro.UnfoldPack pack)
        {
            return prefix
                   + " requires dims=3 pack4 texture input"
                   + " | layer=" + (layer != null ? layer.name : string.Empty)
                   + " | texture=" + (src != null && src.texture != null ? (src.width + "x" + src.height + "x" + src.packs + "p") : "null")
                   + " | shape=d" + shape.dims + ":" + shape.w + "x" + shape.h + "x" + shape.d + "x" + shape.c
                   + " | kernel=" + (pack != null ? (pack.kernelW + "x" + pack.kernelH) : "null")
                   + " | stride=" + (pack != null ? (pack.strideW + "x" + pack.strideH) : "null");
        }

        private static string DescribeUnsupported(string prefix, NcnnParamModel.Layer layer, NcnnRepro.CmdTensorRef src, NcnnRepro.BufferShape shape, NcnnRepro.UnfoldPack pack)
        {
            return prefix
                   + " requires dims=3 pack4 texture input"
                   + " | layer=" + (layer != null ? layer.name : string.Empty)
                   + " | texture=" + (src != null && src.texture != null ? (src.width + "x" + src.height + "x" + src.packs + "p") : "null")
                   + " | shape=d" + shape.dims + ":" + shape.w + "x" + shape.h + "x" + shape.d + "x" + shape.c
                   + " | kernel=" + (pack != null ? (pack.kernelW + "x" + pack.kernelH) : "null")
                   + " | stride=" + (pack != null ? (pack.strideW + "x" + pack.strideH) : "null");
        }

        private static bool IsFoldDStorage(NcnnRepro.TensorRef tensor, NcnnRepro.BufferShape logicalShape)
        {
            if (tensor == null || tensor.texture == null || !TryResolveFoldDStorageShape(logicalShape, out var foldStorageShape))
                return false;
            var storageShape = NcnnRepro.GetTextureStorageShape(tensor, logicalShape);
            var expectedPacks = Mathf.Max(1, Mathf.CeilToInt(logicalShape.c / 4f));
            return ShapesEqual(storageShape, foldStorageShape)
                && tensor.width == foldStorageShape.w
                && tensor.height == foldStorageShape.h
                && tensor.packs == expectedPacks
                && Mathf.Max(1, tensor.texture.volumeDepth) == expectedPacks;
        }

        private static bool IsFoldDStorage(NcnnRepro.CmdTensorRef tensor, NcnnRepro.BufferShape logicalShape)
        {
            if (tensor == null || tensor.texture == null || !TryResolveFoldDStorageShape(logicalShape, out var foldStorageShape))
                return false;
            var storageShape = NcnnRepro.GetCmdStorageShape(tensor, logicalShape);
            var expectedPacks = Mathf.Max(1, Mathf.CeilToInt(logicalShape.c / 4f));
            return ShapesEqual(storageShape, foldStorageShape)
                && tensor.width == foldStorageShape.w
                && tensor.height == foldStorageShape.h
                && tensor.packs == expectedPacks
                && Mathf.Max(1, tensor.texture.depth) == expectedPacks;
        }

        private static NcnnRepro.BufferShape ResolveFoldDStorageShape(NcnnRepro.BufferShape logicalShape)
        {
            if (!TryResolveFoldDStorageShape(logicalShape, out var storageShape))
                throw new InvalidOperationException(
                    "ExtractPatches Fold-D storage shape is not representable"
                    + " | logical=d" + logicalShape.dims + ":" + logicalShape.w + "x" + logicalShape.h + "x" + logicalShape.d + "x" + logicalShape.c);
            return storageShape;
        }

        private static bool TryResolveFoldDStorageShape(NcnnRepro.BufferShape logicalShape, out NcnnRepro.BufferShape storageShape)
        {
            storageShape = default;
            if (logicalShape.dims != 4
                || logicalShape.w <= 0
                || logicalShape.h <= 0
                || logicalShape.d <= 0
                || logicalShape.c <= 0)
            {
                return false;
            }

            var foldedHeight = checked(logicalShape.h * logicalShape.d);
            var outPacks = Mathf.Max(1, Mathf.CeilToInt(logicalShape.c / 4f));
            if (foldedHeight > GetMaxTextureSizeSafe() || outPacks > GetMaxTextureArraySlicesSafe())
                return false;
            storageShape = new NcnnRepro.BufferShape(4, logicalShape.w, foldedHeight, 1, logicalShape.c);
            return true;
        }

        private static bool ShapesEqual(NcnnRepro.BufferShape a, NcnnRepro.BufferShape b)
        {
            return a.dims == b.dims
                && a.w == b.w
                && a.h == b.h
                && a.d == b.d
                && a.c == b.c;
        }

        private static int GetMaxTextureArraySlicesSafe()
        {
            try
            {
                return Mathf.Max(1, SystemInfo.maxTextureArraySlices);
            }
            catch
            {
                return 2048;
            }
        }

        private static int GetMaxTextureSizeSafe()
        {
            try
            {
                return Mathf.Max(1, SystemInfo.maxTextureSize);
            }
            catch
            {
                return 16384;
            }
        }
    }
}
