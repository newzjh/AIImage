using System;
using UnityEngine;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisFlattenLayer : AexisBaseLayer
    {
        public AexisFlattenLayer()
            : base(AexisLayerTypes.Flatten, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
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

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("Flatten source not found: " + layer.name);

            if (TryAliasExistingBuffer(layer.bottomNames[0], layer.topNames[0], bufferBlobs, bufferRefs, bufferViews, srcView, out var aliased))
            {
                if (aliased)
                {
                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                    return;
                }
            }

            var outTensor = owner.RentTempTensorBuffer(1, srcView.elementCount);
            owner.Ops.CopyBufPartial(srcBuf, 0, outTensor.buffer, srcView.elementCount);
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

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var src, out var srcShape))
                throw new InvalidOperationException("Flatten render-texture path requires pack4 texture input: " + layer.name);

            var sourceContract = AexisGraphSession.GetTextureContract(textureShapes, src, layer.bottomNames[0]);
            var outShape = new AexisGraphSession.BufferShape(1, srcShape.w * srcShape.h * srcShape.d * srcShape.c, 1, 1, 1);
            var outStorageShape = ResolveFlattenStorageShape(outShape);
            if (CanAliasFlatten(sourceContract, outShape, outStorageShape))
            {
                textureBlobs[layer.topNames[0]] = AexisGraphSession.CreateTextureAlias(src, outShape, outStorageShape);
            }
            else
            {
                var output = owner.RentTempArray(
                    outStorageShape.w,
                    outStorageShape.h,
                    1,
                    AexisGraphSession.ResolveTensorTextureFormat(outShape.dims));
                if (AexisGraphSession.IsStrictLinearMatTexture(src))
                {
                    var storage = sourceContract.StorageShape;
                    owner.Ops.ReshapeLinearMatToPack4(
                        src.texture,
                        storage.w,
                        storage.h,
                        outShape.w,
                        outShape.h,
                        outShape.d,
                        outShape.c,
                        outShape.dims,
                        output);
                }
                else
                {
                    owner.Ops.ReshapePack4ToPack4(
                        src.texture,
                        srcShape.w,
                        srcShape.h,
                        srcShape.d,
                        srcShape.c,
                        srcShape.dims,
                        outShape.w,
                        outShape.h,
                        outShape.d,
                        outShape.c,
                        outShape.dims,
                        output);
                }

                AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], output, outShape, outStorageShape);
            }
            if (textureShapes != null)
                textureShapes[layer.topNames[0]] = outShape;
            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var sourceContract = AexisGraphSession.GetCmdTensorContract(src);
            var srcShape = sourceContract.LogicalShape;
            var outShape = new AexisGraphSession.BufferShape(1, srcShape.w * srcShape.h * srcShape.d * srcShape.c, 1, 1, 1);
            var outStorageShape = ResolveFlattenStorageShape(outShape);
            if (CanAliasFlatten(sourceContract, outShape, outStorageShape))
            {
                blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorAlias(src, outShape, outStorageShape);
            }
            else
            {
                var output = owner.RentTempArray(
                    cmd,
                    outStorageShape.w,
                    outStorageShape.h,
                    1,
                    AexisGraphSession.ResolveTensorTextureFormat(outShape.dims));
                if (AexisGraphSession.IsStrictLinearMatTexture(src))
                {
                    var storage = sourceContract.StorageShape;
                    owner.Ops.ReshapeLinearMatToPack4(
                        cmd,
                        src.texture,
                        storage.w,
                        storage.h,
                        outShape.w,
                        outShape.h,
                        outShape.d,
                        outShape.c,
                        outShape.dims,
                        output);
                }
                else
                {
                    owner.Ops.ReshapePack4ToPack4(
                        cmd,
                        src.texture,
                        srcShape.w,
                        srcShape.h,
                        srcShape.d,
                        srcShape.c,
                        srcShape.dims,
                        outShape.w,
                        outShape.h,
                        outShape.d,
                        outShape.c,
                        outShape.dims,
                        output);
                }

                blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, outShape, outStorageShape, owned: true, blobName: layer.topNames[0]);
            }
            if (shapes != null)
                shapes[layer.topNames[0]] = outShape;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static AexisGraphSession.BufferShape ResolveFlattenStorageShape(AexisGraphSession.BufferShape shape)
        {
            return new AexisGraphSession.BufferShape(3, shape.w, 1, 1, 1);
        }

        private static bool CanAliasFlatten(
            AexisGraphSession.AexisTextureTensorContract source,
            AexisGraphSession.BufferShape outShape,
            AexisGraphSession.BufferShape outStorageShape)
        {
            return AexisGraphSession.BufferShapeEquals(source.LogicalShape, outShape)
                && AexisGraphSession.BufferShapeEquals(source.StorageShape, outStorageShape);
        }

        private static bool TryAliasExistingBuffer(
            string bottomName,
            string topName,
            System.Collections.Generic.Dictionary<string, ComputeBuffer> bufferBlobs,
            System.Collections.Generic.Dictionary<string, AexisGraphSession.BufferRef> bufferRefs,
            System.Collections.Generic.Dictionary<string, AexisTensorBuffer> bufferViews,
            AexisTensorBuffer srcView,
            out bool aliased)
        {
            aliased = false;
            if (!bufferBlobs.TryGetValue(bottomName, out var existing) || existing == null)
                return false;
            if (!bufferRefs.TryGetValue(bottomName, out var existingRef) || existingRef == null || !existingRef.owned)
                return false;

            bufferBlobs[topName] = existing;
            bufferRefs[topName] = existingRef;
            existingRef.refs++;
            bufferViews[topName] = srcView.Reshape(1, srcView.elementCount);
            aliased = true;
            return true;
        }
    }
}
