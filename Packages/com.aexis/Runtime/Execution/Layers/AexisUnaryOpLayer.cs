using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisUnaryOpLayer : AexisBaseLayer
    {
        public AexisUnaryOpLayer() : base(AexisLayerTypes.UnaryOp, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (!owner.ShouldForceCurrentLayerBufferPath()
                && AexisGraphSession.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var srcTex, out var srcShape)
                && ((!AexisGraphSession.IsStrictLinearMatTexture(srcTex) && srcShape.dims <= 4)
                    || (AexisGraphSession.IsStrictLinearMatTexture(srcTex) && srcShape.dims <= 2)))
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
            if (srcBuf == null)
                throw new InvalidOperationException("UnaryOp source not found: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(
                srcView?.dims ?? 1,
                srcView?.w ?? srcBuf.count,
                srcView?.h ?? 1,
                srcView?.d ?? 1,
                srcView?.c ?? 1);
            owner.Ops.UnaryOpBuf(srcBuf, srcBuf.count, layer.GetInt(0, 0), outTensor.buffer);
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

            if (!AexisGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[0], out var srcTex, out var srcShape) || srcShape.dims > 4)
                throw new InvalidOperationException("UnaryOp render-texture path requires existing <=4D texture input: " + layer.name);

            if (AexisGraphSession.IsStrictLinearMatTexture(srcTex))
            {
                var storageShape = AexisGraphSession.GetTextureStorageShape(srcTex, srcShape);
                var outRt = owner.RentTempMat(storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.UnaryOpLinearMat(srcTex.texture, layer.GetInt(0, 0), outRt);
                AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape, storageShape);
            }
            else
            {
                var storageShape = AexisGraphSession.GetTextureStorageShape(srcTex, srcShape);
                var outputDepth = srcShape.dims == 4 ? srcShape.d * srcTex.packs : srcTex.packs;
                var outRt = owner.RentTempArray(srcTex.width, srcTex.height, outputDepth, AexisGraphSession.ResolveTensorTextureFormat(srcShape.dims));
                owner.Ops.UnaryOpPack4(srcTex.texture, outputDepth, layer.GetInt(0, 0), outRt, srcShape.dims >= 3 ? srcShape.c : 0);
                AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape, storageShape);
            }
            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }
        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
                        var cmd = context.commandBuffer;
                        var blobs = context.blobs;
                        var shapes = context.shapes;
                        var remaining = context.remaining;
                        var pinnedNames = context.pinnedNames;

                        do
                        {
                                                var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
                                                var srcShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
                                                var opType = layer.GetInt(0, 0);
                                                if (AexisGraphSession.IsStrictLinearMatTexture(src) && srcShape.dims <= 2)
                                                {
                                                    var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
                                                    var outMat = owner.RentTempMat(cmd, storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                                                    owner.Ops.UnaryOpLinearMat(cmd, src.texture, opType, outMat);
                                                    blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outMat, srcShape, storageShape, owned: true);
                                                }
                                                else
                                                {
                                                    var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
                                                    var outputDepth = srcShape.dims == 4 ? srcShape.d * src.packs : src.packs;
                                                    var outArr = owner.RentTempArray(cmd, src.width, src.height, outputDepth, AexisGraphSession.ResolveTensorTextureFormat(srcShape.dims));
                                                    owner.Ops.UnaryOpPack4(cmd, src.texture, outputDepth, opType, outArr, srcShape.dims >= 3 ? srcShape.c : 0);
                                                    blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outArr, srcShape, storageShape, owned: true, blobName: layer.topNames[0]);
                                                }
                                                if (shapes != null)
                                                    shapes[layer.topNames[0]] = srcShape;
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                continue;
                        } while (false);
        }
    }
}
