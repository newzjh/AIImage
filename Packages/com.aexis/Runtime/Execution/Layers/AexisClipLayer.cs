using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisClipLayer : AexisBaseLayer
    {
        public AexisClipLayer() : base(AexisLayerTypes.Clip, supportsBufferPath: true, supportsCommandBufferPath: true) { }

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

            var minValue = layer.GetFloat(0, -1e30f);
            var maxValue = layer.GetFloat(1, 1e30f);
            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null)
                throw new InvalidOperationException("Clip source not found: " + layer.name);

            var tmpBuf = owner.RentTempBuffer(srcBuf.count, sizeof(float));
            var outTensor = owner.RentTempTensorBuffer(
                srcView?.dims ?? 1,
                srcView?.w ?? srcBuf.count,
                srcView?.h ?? 1,
                srcView?.d ?? 1,
                srcView?.c ?? 1);
            owner.Ops.BinaryOpScalarBuf(srcBuf, minValue, srcBuf.count, 4, tmpBuf);
            owner.Ops.BinaryOpScalarBuf(tmpBuf, maxValue, srcBuf.count, 5, outTensor.buffer);
            tempOwned.Add(tmpBuf);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: srcView != null && srcView.dims <= 3,
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
            var bufferViews = context.bufferViews;

            var minValue = layer.GetFloat(0, -1e30f);
            var maxValue = layer.GetFloat(1, 1e30f);
            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape))
                throw new InvalidOperationException("Clip render-texture path requires pack4 texture input: " + layer.name);

            if (minValue > maxValue)
                throw new InvalidOperationException("Clip minimum exceeds maximum: " + layer.name);
            var storageShape = AexisGraphSession.GetTextureStorageShape(srcTex, srcShape);
            var outputDepth = srcShape.dims == 4 ? srcShape.d * srcTex.packs : srcTex.packs;
            var outRt = owner.RentTempArray(srcTex.width, srcTex.height, outputDepth, AexisGraphSession.ResolveTensorTextureFormat(srcShape.dims));
            owner.Ops.ClipPack4(srcTex.texture, minValue, maxValue, outputDepth, outRt);
            AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape, storageShape);
            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var minValue = layer.GetFloat(0, -1e30f);
            var maxValue = layer.GetFloat(1, 1e30f);
            if (minValue > maxValue)
                throw new InvalidOperationException("Clip minimum exceeds maximum: " + layer.name);
            var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
            var outputDepth = srcShape.dims == 4 ? srcShape.d * src.packs : src.packs;
            var outArr = owner.RentTempArray(cmd, src.width, src.height, outputDepth, AexisGraphSession.ResolveTensorTextureFormat(srcShape.dims));
            owner.Ops.ClipPack4(cmd, src.texture, minValue, maxValue, outputDepth, outArr);
            blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outArr, srcShape, storageShape, owned: true, blobName: layer.topNames[0]);
            if (shapes != null)
                shapes[layer.topNames[0]] = srcShape;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }
    }
}
