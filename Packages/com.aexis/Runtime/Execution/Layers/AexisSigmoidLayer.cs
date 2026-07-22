using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisSigmoidLayer : AexisBaseLayer
    {
        public AexisSigmoidLayer() : base(AexisLayerTypes.Sigmoid, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (!owner.ShouldForceCurrentLayerBufferPath()
                && AexisGraphSession.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var srcTex, out var srcShape)
                && ((!AexisGraphSession.IsStrictLinearMatTexture(srcTex) && srcShape.dims <= 3)
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
                throw new InvalidOperationException("Sigmoid source not found: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(
                srcView?.dims ?? 1,
                srcView?.w ?? srcBuf.count,
                srcView?.h ?? 1,
                srcView?.d ?? 1,
                srcView?.c ?? 1);
            owner.Ops.SigmoidBuf(srcBuf, srcBuf.count, outTensor.buffer);
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

            if (!AexisGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[0], out var srcTex, out var srcShape) || srcShape.dims > 3)
                throw new InvalidOperationException("Sigmoid render-texture path requires existing <=3D texture input: " + layer.name);

            if (AexisGraphSession.IsPack4LinearMatTexture(srcTex, srcShape))
            {
                // Pack4 LinearMat tensors are still logically scalar matrices.  Preserve their
                // packed storage contract so a following exact Pack4 binary operator can
                // recognize the texture rather than treating it as a generic rank-2 array.
                var storageShape = AexisGraphSession.GetTextureStorageShape(srcTex, srcShape);
                var outRt = owner.RentTempArray(storageShape.w, storageShape.h, 1, srcTex.texture.format);
                owner.Ops.SigmoidPack4(srcTex.texture, 1, outRt);
                AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape, storageShape);
            }
            else if (AexisGraphSession.IsStrictLinearMatTexture(srcTex))
            {
                var storageShape = AexisGraphSession.GetTextureStorageShape(srcTex, srcShape);
                var outRt = owner.RentTempMat(storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.SigmoidLinearMat(srcTex.texture, outRt);
                AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape, storageShape);
            }
            else
            {
                var outRt = owner.RentTempArray(srcTex.width, srcTex.height, srcTex.packs, srcTex.texture.format);
                owner.Ops.SigmoidPack4(srcTex.texture, srcTex.packs, outRt);
                AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
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
                                                if (AexisGraphSession.IsPack4LinearMatTexture(src, srcShape))
                                                {
                                                    var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
                                                    var outArr = owner.RentTempArray(cmd, storageShape.w, storageShape.h, 1, src.texture.format);
                                                    owner.Ops.SigmoidPack4(cmd, src.texture, 1, outArr);
                                                    blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outArr, srcShape, storageShape, owned: true);
                                                }
                                                else if (AexisGraphSession.IsStrictLinearMatTexture(src) && srcShape.dims <= 2)
                                                {
                                                    var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
                                                    var outMat = owner.RentTempMat(cmd, storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                                                    owner.Ops.SigmoidLinearMat(cmd, src.texture, outMat);
                                                    blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outMat, srcShape, storageShape, owned: true);
                                                }
                                                else
                                                {
                                                    var outDepth = srcShape.dims == 4 ? srcShape.d * src.packs : src.packs;
                                                    var outArr = owner.RentTempArray(cmd, src.width, src.height, outDepth, src.texture.format);
                                                    owner.Ops.SigmoidPack4(cmd, src.texture, src.packs, outArr);
                                                    blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef { texture = outArr, width = src.width, height = src.height, packs = src.packs, refs = 1, owned = true };
                                                }
                                                if (shapes != null)
                                                    shapes[layer.topNames[0]] = srcShape;
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                continue;
                        } while (false);
        }
    }
}
