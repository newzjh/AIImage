using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnPriorBoxLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnPriorBoxLayerRepro()
            : base(NcnnLayerTypes.PriorBox, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            owner._extraPacks[layer.name] = new NcnnRepro.PriorBoxPack
            {
                minSizes = layer.GetFloats(-23300, Array.Empty<float>()),
                maxSizes = layer.GetFloats(-23301, Array.Empty<float>()),
                aspectRatios = layer.GetFloats(-23302, Array.Empty<float>()),
                variances = new[]
                {
                    layer.GetFloat(3, 0.1f),
                    layer.GetFloat(4, 0.1f),
                    layer.GetFloat(5, 0.2f),
                    layer.GetFloat(6, 0.2f)
                },
                flip = layer.GetInt(7, 1) != 0,
                clip = layer.GetInt(8, 0) != 0,
                imageWidth = layer.GetInt(9, 0),
                imageHeight = layer.GetInt(10, 0),
                stepWidth = layer.GetFloat(11, -233f),
                stepHeight = layer.GetFloat(12, -233f),
                offset = layer.GetFloat(13, 0f),
                stepMmdetection = layer.GetInt(14, 0) != 0,
                centerMmdetection = layer.GetInt(15, 0) != 0
            };
            return default;
        }

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

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.PriorBoxPack pp)
                throw new InvalidOperationException("PriorBox pack not found: " + layer.name);

            if (!TryResolveLogicalShape(layer.bottomNames[0], textureShapes, bufferBlobs, bufferViews, out var featDims, out var featW, out var featH, out _, out _))
                throw new InvalidOperationException("PriorBox feature shape missing: " + layer.name);
            if (featDims < 2)
                throw new InvalidOperationException("PriorBox expects at least 2d feature shape: " + layer.name);

            if (layer.bottomNames.Length == 1 && pp.imageWidth == -233 && pp.imageHeight == -233 && (pp.maxSizes == null || pp.maxSizes.Length == 0))
            {
                var output = BuildMxnetPrior(featW, featH, pp);
                var tensor = owner.RentTempTensorBuffer(1, output.Length);
                tensor.buffer.SetData(output);
                owner.PublishTensorBufferOutput(
                    layer.topNames[0],
                    tensor,
                    preferTexture: false,
                    textureBlobs,
                    textureShapes,
                    bufferBlobs,
                    bufferRefs,
                    bufferViews,
                    tempOwned);
                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                return;
            }

            var imageW = pp.imageWidth;
            var imageH = pp.imageHeight;
            if (imageW == -233 || imageH == -233)
            {
                if (layer.bottomNames.Length < 2 || !TryResolveLogicalShape(layer.bottomNames[1], textureShapes, bufferBlobs, bufferViews, out _, out var iw, out var ih, out _, out _))
                    throw new InvalidOperationException("PriorBox image shape missing: " + layer.name);
                if (imageW == -233) imageW = iw;
                if (imageH == -233) imageH = ih;
            }

            var output2 = BuildCaffePrior(featW, featH, imageW, imageH, pp);
            var tensor2 = owner.RentTempTensorBuffer(2, output2.Length / 2, 2);
            tensor2.buffer.SetData(output2);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                tensor2,
                preferTexture: true,
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

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.PriorBoxPack pp)
                throw new InvalidOperationException("PriorBox pack not found: " + layer.name);

            var featShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var featW = featShape.w;
            var featH = featShape.h;

            if (layer.bottomNames.Length == 1 && pp.imageWidth == -233 && pp.imageHeight == -233 && (pp.maxSizes == null || pp.maxSizes.Length == 0))
            {
                var numSizes = pp.minSizes?.Length ?? 0;
                var numRatios = pp.aspectRatios?.Length ?? 0;
                var numPrior = Mathf.Max(1, numSizes - 1 + numRatios);
                owner.PublishCmdPlaceholder(
                    cmd,
                    layer.topNames[0],
                    new NcnnRepro.BufferShape(1, Mathf.Max(1, 4 * featW * featH * numPrior), 1, 1, 1),
                    blobs,
                    shapes);
            }
            else
            {
                var imageW = pp.imageWidth;
                var imageH = pp.imageHeight;
                if (imageW == -233 || imageH == -233)
                {
                    if (layer.bottomNames.Length < 2)
                        throw new InvalidOperationException("PriorBox image shape missing: " + layer.name);
                    var imageShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[1]);
                    if (imageW == -233) imageW = imageShape.w;
                    if (imageH == -233) imageH = imageShape.h;
                }

                var numMinSize = pp.minSizes?.Length ?? 0;
                var numMaxSize = pp.maxSizes?.Length ?? 0;
                var numAspectRatio = pp.aspectRatios?.Length ?? 0;
                var numPrior = numMinSize * numAspectRatio + numMinSize + numMaxSize;
                if (pp.flip)
                    numPrior += numMinSize * numAspectRatio;
                var total = Mathf.Max(1, 4 * featW * featH * numPrior);
                owner.PublishCmdPlaceholder(
                    cmd,
                    layer.topNames[0],
                    new NcnnRepro.BufferShape(2, total, 2, 1, 1),
                    blobs,
                    shapes);
            }

            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool TryResolveLogicalShape(
            string name,
            System.Collections.Generic.Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            System.Collections.Generic.Dictionary<string, ComputeBuffer> bufferBlobs,
            System.Collections.Generic.Dictionary<string, NcnnTensorBuffer> bufferViews,
            out int dims,
            out int w,
            out int h,
            out int d,
            out int c)
        {
            if (bufferViews.TryGetValue(name, out var view) && view != null)
            {
                dims = view.dims;
                w = view.w;
                h = view.h;
                d = view.d;
                c = view.c;
                return true;
            }

            if (textureShapes.TryGetValue(name, out var shape))
            {
                dims = shape.dims;
                w = shape.w;
                h = shape.h;
                d = shape.d;
                c = shape.c;
                return true;
            }

            if (bufferBlobs.TryGetValue(name, out var buf) && buf != null)
            {
                dims = 1;
                w = buf.count;
                h = 1;
                d = 1;
                c = 1;
                return true;
            }

            dims = 0;
            w = 0;
            h = 0;
            d = 0;
            c = 0;
            return false;
        }

        private static float[] BuildMxnetPrior(int w, int h, NcnnRepro.PriorBoxPack pp)
        {
            var stepW = pp.stepWidth == -233f ? 1f / w : pp.stepWidth;
            var stepH = pp.stepHeight == -233f ? 1f / h : pp.stepHeight;
            var numSizes = pp.minSizes.Length;
            var numRatios = pp.aspectRatios.Length;
            var numPrior = numSizes - 1 + numRatios;
            var output = new float[4 * w * h * numPrior];
            var cursor = 0;

            for (var i = 0; i < h; i++)
            {
                var centerX = pp.offset * stepW;
                var centerY = pp.offset * stepH + i * stepH;
                for (var j = 0; j < w; j++)
                {
                    for (var k = 0; k < numSizes; k++)
                    {
                        var size = pp.minSizes[k];
                        var cw = size * h / (float)w / 2f;
                        var ch = size / 2f;
                        output[cursor++] = centerX - cw;
                        output[cursor++] = centerY - ch;
                        output[cursor++] = centerX + cw;
                        output[cursor++] = centerY + ch;
                    }

                    var baseSize = pp.minSizes[0];
                    for (var p = 1; p < numRatios; p++)
                    {
                        var ratio = Mathf.Sqrt(pp.aspectRatios[p]);
                        var cw = baseSize * h / (float)w * ratio / 2f;
                        var ch = baseSize / ratio / 2f;
                        output[cursor++] = centerX - cw;
                        output[cursor++] = centerY - ch;
                        output[cursor++] = centerX + cw;
                        output[cursor++] = centerY + ch;
                    }

                    centerX += stepW;
                }
            }

            if (pp.clip)
            {
                for (var i = 0; i < output.Length; i++)
                    output[i] = Mathf.Clamp01(output[i]);
            }

            return output;
        }

        private static float[] BuildCaffePrior(int w, int h, int imageW, int imageH, NcnnRepro.PriorBoxPack pp)
        {
            var stepW = pp.stepWidth;
            var stepH = pp.stepHeight;
            if (stepW == -233f)
            {
                stepW = imageW / (float)w;
                if (pp.stepMmdetection)
                    stepW = Mathf.Ceil(imageW / (float)w);
            }
            if (stepH == -233f)
            {
                stepH = imageH / (float)h;
                if (pp.stepMmdetection)
                    stepH = Mathf.Ceil(imageH / (float)h);
            }

            var numMinSize = pp.minSizes.Length;
            var numMaxSize = pp.maxSizes == null ? 0 : pp.maxSizes.Length;
            var numAspectRatio = pp.aspectRatios == null ? 0 : pp.aspectRatios.Length;
            var numPrior = numMinSize * numAspectRatio + numMinSize + numMaxSize;
            if (pp.flip)
                numPrior += numMinSize * numAspectRatio;

            var total = 4 * w * h * numPrior;
            var output = new float[total * 2];
            var cursor = 0;

            for (var i = 0; i < h; i++)
            {
                var centerX = pp.offset * stepW;
                var centerY = pp.offset * stepH + i * stepH;
                if (pp.centerMmdetection)
                {
                    centerX = pp.offset * (stepW - 1f);
                    centerY = pp.offset * (stepH - 1f) + i * stepH;
                }

                for (var j = 0; j < w; j++)
                {
                    for (var k = 0; k < numMinSize; k++)
                    {
                        var minSize = pp.minSizes[k];
                        WriteBox(output, ref cursor, centerX, centerY, minSize, minSize, imageW, imageH);

                        if (numMaxSize > 0)
                        {
                            var maxSize = pp.maxSizes[k];
                            var boxSize = Mathf.Sqrt(minSize * maxSize);
                            WriteBox(output, ref cursor, centerX, centerY, boxSize, boxSize, imageW, imageH);
                        }

                        for (var p = 0; p < numAspectRatio; p++)
                        {
                            var ar = pp.aspectRatios[p];
                            var boxW = minSize * Mathf.Sqrt(ar);
                            var boxH = minSize / Mathf.Sqrt(ar);
                            WriteBox(output, ref cursor, centerX, centerY, boxW, boxH, imageW, imageH);

                            if (pp.flip)
                                WriteBox(output, ref cursor, centerX, centerY, boxH, boxW, imageW, imageH);
                        }
                    }

                    centerX += stepW;
                }
            }

            if (pp.clip)
            {
                for (var i = 0; i < total; i++)
                    output[i] = Mathf.Clamp01(output[i]);
            }

            var varianceOffset = total;
            for (var i = 0; i < total / 4; i++)
            {
                output[varianceOffset++] = pp.variances[0];
                output[varianceOffset++] = pp.variances[1];
                output[varianceOffset++] = pp.variances[2];
                output[varianceOffset++] = pp.variances[3];
            }

            return output;
        }

        private static void WriteBox(float[] output, ref int cursor, float centerX, float centerY, float boxW, float boxH, int imageW, int imageH)
        {
            output[cursor++] = (centerX - boxW * 0.5f) / imageW;
            output[cursor++] = (centerY - boxH * 0.5f) / imageH;
            output[cursor++] = (centerX + boxW * 0.5f) / imageW;
            output[cursor++] = (centerY + boxH * 0.5f) / imageH;
        }
    }
}
