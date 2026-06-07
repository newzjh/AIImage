using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnUnfoldLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnUnfoldLayerRepro()
            : base(NcnnLayerTypes.Unfold, supportsBufferPath: true, supportsCommandBufferPath: true)
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

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.UnfoldPack up)
                throw new InvalidOperationException("Unfold pack not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
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

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.UnfoldPack up)
                throw new InvalidOperationException("Unfold pack not found: " + layer.name);

            var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            if (srcShape.dims != 2 && srcShape.dims != 3)
                throw new InvalidOperationException("Unfold expects dims=2 or dims=3 source: " + layer.name);

            var inW = srcShape.w;
            var inH = srcShape.h;
            var inC = srcShape.dims == 3 ? srcShape.c : 1;
            var kernelExtentW = up.dilationW * (up.kernelW - 1) + 1;
            var kernelExtentH = up.dilationH * (up.kernelH - 1) + 1;
            ResolvePadding(inW, inH, kernelExtentW, kernelExtentH, up, out var padLeft, out var padRight, out var padTop, out var padBottom);
            var paddedW = inW + padLeft + padRight;
            var paddedH = inH + padTop + padBottom;
            var outw = (paddedW - kernelExtentW) / up.strideW + 1;
            var outh = (paddedH - kernelExtentH) / up.strideH + 1;
            var size = Mathf.Max(1, outw * outh);
            var outRows = Mathf.Max(1, up.kernelW * up.kernelH * inC);

            owner.PublishCmdPlaceholder(
                cmd,
                layer.topNames[0],
                new NcnnRepro.BufferShape(2, size, outRows, 1, 1),
                blobs,
                shapes);
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
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
