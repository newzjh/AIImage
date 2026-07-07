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
            var pack = new NcnnRepro.PriorBoxPack
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
            pack.minSizeBuffer = NewParamBuffer(pack.minSizes);
            pack.maxSizeBuffer = NewParamBuffer(pack.maxSizes);
            pack.aspectRatioBuffer = NewParamBuffer(pack.aspectRatios);
            pack.varianceBuffer = NewParamBuffer(pack.variances);
            owner._extraPacks[layer.name] = pack;
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

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not NcnnRepro.PriorBoxPack pp)
                throw new InvalidOperationException("PriorBox pack not found: " + layer.name);
            var spec = ResolvePriorBoxSpec(
                layer,
                pp,
                (string name, out NcnnRepro.BufferShape shape) =>
                {
                    if (TryResolveLogicalShape(name, textureShapes, bufferBlobs, bufferViews, out var dims, out var w, out var h, out var d, out var c))
                    {
                        shape = new NcnnRepro.BufferShape(dims, w, h, d, c);
                        return true;
                    }

                    shape = default;
                    return false;
                });

            RenderTexture output = null;
            try
            {
                output = owner.RentTempMat(spec.storageShape.w, spec.storageShape.h, NcnnRepro.ResolveLinearMatTextureFormat());
                owner.Ops.PriorBox(
                    output,
                    spec.mode,
                    spec.featW,
                    spec.featH,
                    spec.imageW,
                    spec.imageH,
                    spec.numPrior,
                    pp.minSizeBuffer,
                    pp.minSizes?.Length ?? 0,
                    pp.maxSizeBuffer,
                    pp.maxSizes?.Length ?? 0,
                    pp.aspectRatioBuffer,
                    pp.aspectRatios?.Length ?? 0,
                    pp.varianceBuffer,
                    pp.flip,
                    pp.clip,
                    pp.stepMmdetection,
                    pp.centerMmdetection,
                    spec.stepW,
                    spec.stepH,
                    pp.offset);
                NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], output, spec.logicalShape, spec.storageShape);
                output = null;
            }
            finally
            {
                if (output != null)
                    owner.ReturnTempArray(output);
            }

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

            var spec = ResolvePriorBoxSpec(
                layer,
                pp,
                (string name, out NcnnRepro.BufferShape shape) =>
                {
                    shape = NcnnRepro.GetCmdShape(shapes, blobs, name);
                    return true;
                });

            ComputeTexture output = null;
            try
            {
                output = owner.RentTempMat(cmd, spec.storageShape.w, spec.storageShape.h, NcnnRepro.ResolveLinearMatTextureFormat());
                owner.Ops.PriorBox(
                    cmd,
                    output,
                    spec.mode,
                    spec.featW,
                    spec.featH,
                    spec.imageW,
                    spec.imageH,
                    spec.numPrior,
                    pp.minSizeBuffer,
                    pp.minSizes?.Length ?? 0,
                    pp.maxSizeBuffer,
                    pp.maxSizes?.Length ?? 0,
                    pp.aspectRatioBuffer,
                    pp.aspectRatios?.Length ?? 0,
                    pp.varianceBuffer,
                    pp.flip,
                    pp.clip,
                    pp.stepMmdetection,
                    pp.centerMmdetection,
                    spec.stepW,
                    spec.stepH,
                    pp.offset);
                blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(output, spec.logicalShape, spec.storageShape, owned: true);
                output = null;
                if (shapes != null)
                    shapes[layer.topNames[0]] = spec.logicalShape;
            }
            finally
            {
                if (output != null)
                    owner.ReturnTempArray(cmd, output);
            }

            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private readonly struct PriorBoxSpec
        {
            public readonly int mode;
            public readonly int featW;
            public readonly int featH;
            public readonly int imageW;
            public readonly int imageH;
            public readonly int numPrior;
            public readonly float stepW;
            public readonly float stepH;
            public readonly NcnnRepro.BufferShape logicalShape;
            public readonly NcnnRepro.BufferShape storageShape;

            public PriorBoxSpec(int mode, int featW, int featH, int imageW, int imageH, int numPrior, float stepW, float stepH, NcnnRepro.BufferShape logicalShape)
            {
                this.mode = mode;
                this.featW = featW;
                this.featH = featH;
                this.imageW = imageW;
                this.imageH = imageH;
                this.numPrior = numPrior;
                this.stepW = stepW;
                this.stepH = stepH;
                this.logicalShape = logicalShape;
                storageShape = NcnnRepro.ResolveLinearMatStorageShape(logicalShape);
            }
        }

        private delegate bool TryGetShapeDelegate(string name, out NcnnRepro.BufferShape shape);

        private static PriorBoxSpec ResolvePriorBoxSpec(NcnnParamModel.Layer layer, NcnnRepro.PriorBoxPack pp, TryGetShapeDelegate tryGetShape)
        {
            if (!tryGetShape(layer.bottomNames[0], out var featShape))
                throw new InvalidOperationException("PriorBox feature shape missing: " + layer.name);
            if (featShape.dims < 2)
                throw new InvalidOperationException("PriorBox expects at least 2d feature shape: " + layer.name);

            var featW = Mathf.Max(1, featShape.w);
            var featH = Mathf.Max(1, featShape.h);
            var useMxnet = layer.bottomNames.Length == 1
                && pp.imageWidth == -233
                && pp.imageHeight == -233
                && (pp.maxSizes == null || pp.maxSizes.Length == 0);
            if (useMxnet)
            {
                var numPrior = Mathf.Max(1, (pp.minSizes?.Length ?? 0) - 1 + (pp.aspectRatios?.Length ?? 0));
                var total = Mathf.Max(1, 4 * featW * featH * numPrior);
                var stepW = pp.stepWidth == -233f ? 1f / featW : pp.stepWidth;
                var stepH = pp.stepHeight == -233f ? 1f / featH : pp.stepHeight;
                return new PriorBoxSpec(0, featW, featH, 1, 1, numPrior, stepW, stepH, new NcnnRepro.BufferShape(1, total, 1, 1, 1));
            }

            var imageW = pp.imageWidth;
            var imageH = pp.imageHeight;
            if (imageW == -233 || imageH == -233)
            {
                if (layer.bottomNames.Length < 2 || !tryGetShape(layer.bottomNames[1], out var imageShape))
                    throw new InvalidOperationException("PriorBox image shape missing: " + layer.name);
                if (imageW == -233) imageW = imageShape.w;
                if (imageH == -233) imageH = imageShape.h;
            }

            imageW = Mathf.Max(1, imageW);
            imageH = Mathf.Max(1, imageH);
            var stepWidth = pp.stepWidth;
            var stepHeight = pp.stepHeight;
            if (stepWidth == -233f)
            {
                stepWidth = imageW / (float)featW;
                if (pp.stepMmdetection)
                    stepWidth = Mathf.Ceil(imageW / (float)featW);
            }
            if (stepHeight == -233f)
            {
                stepHeight = imageH / (float)featH;
                if (pp.stepMmdetection)
                    stepHeight = Mathf.Ceil(imageH / (float)featH);
            }

            var numMinSize = pp.minSizes?.Length ?? 0;
            var numMaxSize = pp.maxSizes?.Length ?? 0;
            var numAspectRatio = pp.aspectRatios?.Length ?? 0;
            var caffeNumPrior = numMinSize * numAspectRatio + numMinSize + numMaxSize;
            if (pp.flip)
                caffeNumPrior += numMinSize * numAspectRatio;
            caffeNumPrior = Mathf.Max(1, caffeNumPrior);
            var caffeTotal = Mathf.Max(1, 4 * featW * featH * caffeNumPrior);
            return new PriorBoxSpec(1, featW, featH, imageW, imageH, caffeNumPrior, stepWidth, stepHeight, new NcnnRepro.BufferShape(2, caffeTotal, 2, 1, 1));
        }

        private static ComputeBuffer NewParamBuffer(float[] values)
        {
            return NcnnRepro.NewBuffer(values != null && values.Length > 0 ? values : new[] { 0f });
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
