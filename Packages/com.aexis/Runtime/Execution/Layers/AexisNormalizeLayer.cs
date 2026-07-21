using System;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisNormalizeLayer : AexisBaseLayer
    {
        public AexisNormalizeLayer()
            : base(AexisLayerTypes.Normalize, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            var np = new AexisGraphSession.NormalizePack
            {
                acrossSpatial = layer.GetInt(0, 0) != 0,
                channelShared = layer.GetInt(1, 0) != 0,
                eps = layer.GetFloat(2, 0.0001f),
                scaleDataSize = layer.GetInt(3, 0),
                acrossChannel = layer.GetInt(4, 1) != 0,
                epsMode = layer.GetInt(9, 0)
            };

            if (np.scaleDataSize > 0)
            {
                phaseSw.Restart();
                np.scaleCpu = br.ReadTensorAsFloat32(np.scaleDataSize, 0, 0, 0, 1);
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                np.scale = AexisGraphSession.NewBuffer(np.scaleCpu);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;
            }

            owner._extraPacks[layer.name] = np;
            return new AexisGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
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

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.NormalizePack np)
                throw new InvalidOperationException("Normalize pack not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("Normalize source not found: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c);
            owner.Ops.NormalizeBuf(
                srcBuf,
                srcView,
                np.scale,
                np.scaleDataSize,
                np.acrossSpatial,
                np.acrossChannel,
                np.channelShared,
                np.eps,
                np.epsMode,
                outTensor.buffer);
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
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.NormalizePack np)
                throw new InvalidOperationException("Normalize pack not found: " + layer.name);

            AexisPack4LayerHelpers.ExecuteShapePreservingRenderTexture(
                owner,
                layer,
                context,
                "Normalize",
                (input, shape, output) => owner.Ops.NormalizePack4(
                    input,
                    shape,
                    np.scale,
                    np.scaleDataSize,
                    np.acrossSpatial,
                    np.acrossChannel,
                    np.channelShared,
                    np.eps,
                    np.epsMode,
                    output));
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.NormalizePack np)
                throw new InvalidOperationException("Normalize pack not found: " + layer.name);

            AexisPack4LayerHelpers.ExecuteShapePreservingCommandBuffer(
                owner,
                layer,
                context,
                "Normalize",
                (cmd, input, shape, output) => owner.Ops.NormalizePack4(
                    cmd,
                    input,
                    shape,
                    np.scale,
                    np.scaleDataSize,
                    np.acrossSpatial,
                    np.acrossChannel,
                    np.channelShared,
                    np.eps,
                    np.epsMode,
                    output));
        }
    }
}
