using System;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisRMSNormLayer : AexisBaseLayer
    {
        public AexisRMSNormLayer()
            : base(AexisLayerTypes.RMSNorm, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            var rp = new AexisGraphSession.RmsNormPack
            {
                affineSize = layer.GetInt(0, 0),
                eps = layer.GetFloat(1, 0.001f),
                affine = layer.GetInt(2, 1) != 0
            };

            if (rp.affine && rp.affineSize > 0)
            {
                phaseSw.Restart();
                rp.gammaCpu = br.ReadTensorAsFloat32(rp.affineSize, 0, 0, 0, 1);
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                rp.gamma = AexisGraphSession.NewBuffer(rp.gammaCpu);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;
            }

            owner._extraPacks[layer.name] = rp;
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

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.RmsNormPack rp)
                throw new InvalidOperationException("RMSNorm pack not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("RMSNorm source not found: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c);
            owner.Ops.RmsNormBuf(srcBuf, srcView, rp.gamma, rp.affineSize, rp.affine, rp.eps, outTensor.buffer);
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
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.RmsNormPack rp)
                throw new InvalidOperationException("RMSNorm pack not found: " + layer.name);

            if (AexisGraphSession.TryGetExistingTexture(
                    context.textureBlobs,
                    context.textureShapes,
                    layer.bottomNames[0],
                    out var src,
                    out var shape)
                && AexisGraphSession.IsPack4LinearMatTexture(src, shape))
            {
                var storageShape = AexisGraphSession.GetTextureStorageShape(src, shape);
                var output = owner.RentTempArray(storageShape.w, storageShape.h, 1, src.texture.format);
                owner.Ops.RmsNormPack4LinearMat(src.texture, shape, rp.gamma, rp.affineSize, rp.affine, rp.eps, output);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, shape, storageShape);
                owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
                return;
            }

            AexisPack4LayerHelpers.ExecuteShapePreservingRenderTexture(
                owner,
                layer,
                context,
                "RMSNorm",
                (input, shape, output) => owner.Ops.RmsNormPack4(input, shape, rp.gamma, rp.affineSize, rp.affine, rp.eps, output));
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not AexisGraphSession.RmsNormPack rp)
                throw new InvalidOperationException("RMSNorm pack not found: " + layer.name);

            var src = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var shape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            if (AexisGraphSession.IsPack4LinearMatTexture(src, shape))
            {
                var storageShape = AexisGraphSession.GetCmdStorageShape(src, shape);
                var output = owner.RentTempArray(context.commandBuffer, storageShape.w, storageShape.h, 1, src.texture.format);
                owner.Ops.RmsNormPack4LinearMat(context.commandBuffer, src.texture, shape, rp.gamma, rp.affineSize, rp.affine, rp.eps, output);
                context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, shape, storageShape, owned: true);
                if (context.shapes != null)
                    context.shapes[layer.topNames[0]] = shape;
                owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
                return;
            }

            AexisPack4LayerHelpers.ExecuteShapePreservingCommandBuffer(
                owner,
                layer,
                context,
                "RMSNorm",
                (cmd, input, shape, output) => owner.Ops.RmsNormPack4(cmd, input, shape, rp.gamma, rp.affineSize, rp.affine, rp.eps, output));
        }
    }
}
