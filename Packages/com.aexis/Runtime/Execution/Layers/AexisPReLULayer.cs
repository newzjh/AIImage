using System;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisPReLULayer : AexisBaseLayer
    {
        public AexisPReLULayer()
            : base(AexisLayerTypes.PReLU, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            var pp = new AexisGraphSession.PReluPack
            {
                numSlope = Mathf.Max(1, layer.GetInt(0, 0))
            };

            phaseSw.Restart();
            pp.slopeCpu = br.ReadTensorAsFloat32(pp.numSlope, 0, 0, 0, 1);
            phaseSw.Stop();
            readMs += phaseSw.ElapsedMilliseconds;

            phaseSw.Restart();
            pp.slope = AexisGraphSession.NewBuffer(pp.slopeCpu);
            phaseSw.Stop();
            uploadMs += phaseSw.ElapsedMilliseconds;

            owner._extraPacks[layer.name] = pp;
            return new AexisGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.PReluPack pp)
                throw new InvalidOperationException("PReLU pack not found: " + layer.name);

            if (owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out _,
                    out _))
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

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.PReluPack pp)
                throw new InvalidOperationException("PReLU pack not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("PReLU source not found: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c);
            owner.Ops.PReluBuf(srcBuf, srcView, pp.slope, pp.numSlope, outTensor.buffer);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: srcView.dims <= 3,
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
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.PReluPack pp)
                throw new InvalidOperationException("PReLU pack not found: " + layer.name);

            AexisPack4LayerHelpers.ExecuteShapePreservingRenderTexture(
                owner,
                layer,
                context,
                "PReLU",
                (input, shape, output) =>
                {
                    if (pp.numSlope == 1)
                    {
                        owner.Ops.LeakyReluPack4(input, pp.slopeCpu[0], output.volumeDepth, output);
                        return;
                    }
                    ValidateChannelSlope(shape, pp.numSlope, layer.name);
                    owner.Ops.PReluPack4(input, pp.slope, pp.numSlope, Mathf.Max(1, Mathf.CeilToInt(shape.c / 4f)), output);
                });
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.PReluPack pp)
                throw new InvalidOperationException("PReLU pack not found: " + layer.name);

            AexisPack4LayerHelpers.ExecuteShapePreservingCommandBuffer(
                owner,
                layer,
                context,
                "PReLU",
                (cmd, input, shape, output) =>
                {
                    if (pp.numSlope == 1)
                    {
                        owner.Ops.LeakyReluPack4(cmd, input, pp.slopeCpu[0], output.depth, output);
                        return;
                    }
                    ValidateChannelSlope(shape, pp.numSlope, layer.name);
                    owner.Ops.PReluPack4(cmd, input, pp.slope, pp.numSlope, Mathf.Max(1, Mathf.CeilToInt(shape.c / 4f)), output);
                });
        }

        private static void ValidateChannelSlope(AexisGraphSession.BufferShape shape, int slopeCount, string layerName)
        {
            if ((shape.dims != 3 && shape.dims != 4) || slopeCount != shape.c)
            {
                throw new InvalidOperationException(
                    "PReLU non-scalar slope requires one immutable value per logical channel"
                    + " | layer=" + layerName
                    + " | dims=" + shape.dims
                    + " | channels=" + shape.c
                    + " | slopes=" + slopeCount);
            }
        }
    }
}
