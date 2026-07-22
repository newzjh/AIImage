using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisReLULayer : AexisBaseLayer
    {
        public AexisReLULayer() : base(AexisLayerTypes.ReLU, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (!owner.ShouldForceCurrentLayerBufferPath()
                && owner.TryGetPack4Texture(
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

            var slope = layer.GetFloat(0, 0f);
            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("ReLU source not found: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c);
            owner.Ops.LeakyReluBuf(srcBuf, srcView.elementCount, slope, outTensor.buffer);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: srcView.dims <= 4,
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
            var bufferRefs = context.bufferRefs;
            var tempOwned = context.tempOwned;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var slope = layer.GetFloat(0, 0f);
            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape))
                throw new InvalidOperationException("ReLU render-texture path requires pack4 texture input: " + layer.name);

            var outDepth = srcShape.dims == 4 ? srcShape.d * srcTex.packs : srcTex.packs;
            var outRt = owner.RentTempArray(srcTex.width, srcTex.height, outDepth, srcTex.texture.format);
            owner.Ops.LeakyReluPack4(srcTex.texture, slope, srcTex.packs, outRt);
            textureBlobs[layer.topNames[0]] = new AexisGraphSession.TensorRef
            {
                texture = outRt,
                width = outRt.width,
                height = outRt.height,
                packs = srcTex.packs,
                refs = 1,
                owned = true
            };
            textureShapes[layer.topNames[0]] = srcShape;
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
                                                var slope = layer.GetFloat(0, 0f);
                                                var outDepth = srcShape.dims == 4 ? srcShape.d * src.packs : src.packs;
                                                var outArr = owner.RentTempArray(cmd, src.width, src.height, outDepth, src.texture.format);
                                                owner.Ops.LeakyReluPack4(cmd, src.texture, slope, src.packs, outArr);
                                                var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
                                                blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outArr, srcShape, storageShape, owned: true);
                                                if (shapes != null)
                                                    shapes[layer.topNames[0]] = srcShape;
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                continue;
                        } while (false);
        }
    }
}
