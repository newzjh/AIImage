using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Ncnn
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnGeluLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnGeluLayerRepro() : base(NcnnLayerTypes.GELU, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (!owner.ForceBufferGeluAll
                && !owner.ShouldForceCurrentLayerBufferPath()
                && NcnnGraphSession.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var srcTex, out var srcShape)
                && ((NcnnGraphSession.IsPack4LinearMatTexture(srcTex, srcShape) && srcShape.dims == 2)
                    || (!NcnnGraphSession.IsStrictLinearMatTexture(srcTex) && srcShape.dims <= 3)
                    || (NcnnGraphSession.IsStrictLinearMatTexture(srcTex) && srcShape.dims <= 2)))
            {
                ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
            var srcView = NcnnGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null)
                throw new InvalidOperationException("GELU source not found: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(
                srcView?.dims ?? 1,
                srcView?.w ?? srcBuf.count,
                srcView?.h ?? 1,
                srcView?.d ?? 1,
                srcView?.c ?? 1);
            owner.Ops.GeluBuf(srcBuf, srcBuf.count, outTensor.buffer);
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

        public override void ExecuteRenderTexturePath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;

            if (!NcnnGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, layer.bottomNames[0], out var srcTex, out var srcShape) || srcShape.dims > 3)
                throw new InvalidOperationException("GELU render-texture path requires existing <=3D texture input: " + layer.name);

            if (NcnnGraphSession.IsPack4LinearMatTexture(srcTex, srcShape))
            {
                var storageShape = NcnnGraphSession.GetTextureStorageShape(srcTex, srcShape);
                var outRt = owner.RentTempArray(storageShape.w, storageShape.h, 1, srcTex.texture.format);
                owner.Ops.GeluPack4(srcTex.texture, 1, false, outRt);
                NcnnGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape, storageShape);
            }
            else if (NcnnGraphSession.IsStrictLinearMatTexture(srcTex))
            {
                var storageShape = NcnnGraphSession.GetTextureStorageShape(srcTex, srcShape);
                var outRt = owner.RentTempMat(storageShape.w, storageShape.h, NcnnGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.GeluLinearMat(srcTex.texture, false, outRt);
                NcnnGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape, storageShape);
            }
            else
            {
                var outRt = owner.RentTempArray(srcTex.width, srcTex.height, srcTex.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.GeluPack4(srcTex.texture, srcTex.packs, false, outRt);
                NcnnGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
            }
            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }
        public override void ExecuteCommandBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
                        var cmd = context.commandBuffer;
                        var blobs = context.blobs;
                        var shapes = context.shapes;
                        var remaining = context.remaining;
                        var pinnedNames = context.pinnedNames;

                        do
                        {
                                                var src = NcnnGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
                                                var srcShape = NcnnGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
                                                var fast = layer.GetInt(0, 0) != 0;
                                                if (NcnnGraphSession.IsPack4LinearMatTexture(src, srcShape))
                                                {
                                                    var storageShape = NcnnGraphSession.GetCmdStorageShape(src, srcShape);
                                                    var outMat = owner.RentTempArray(cmd, storageShape.w, storageShape.h, 1, src.texture.format);
                                                    owner.Ops.GeluPack4(cmd, src.texture, 1, fast, outMat);
                                                    blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorRef(outMat, srcShape, storageShape, owned: true);
                                                }
                                                else if (NcnnGraphSession.IsStrictLinearMatTexture(src) && srcShape.dims <= 2)
                                                {
                                                    var storageShape = NcnnGraphSession.GetCmdStorageShape(src, srcShape);
                                                    var outMat = owner.RentTempMat(cmd, storageShape.w, storageShape.h, NcnnGraphSession.ResolveLinearMatTextureFormat());
                                                    owner.Ops.GeluLinearMat(cmd, src.texture, fast, outMat);
                                                    blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorRef(outMat, srcShape, storageShape, owned: true);
                                                }
                                                else
                                                {
                                                    var outArr = owner.RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                                                    owner.Ops.GeluPack4(cmd, src.texture, src.packs, fast, outArr);
                                                    blobs[layer.topNames[0]] = new NcnnGraphSession.CmdTensorRef { texture = outArr, width = src.width, height = src.height, packs = src.packs, refs = 1, owned = true };
                                                }
                                                if (shapes != null)
                                                    shapes[layer.topNames[0]] = srcShape;
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                continue;
                        } while (false);
        }
    }
}
