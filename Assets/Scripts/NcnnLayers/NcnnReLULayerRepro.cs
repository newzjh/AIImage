using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnReLULayerRepro : NcnnBaseLayerRepro
    {
        public NcnnReLULayerRepro() : base(NcnnLayerTypes.ReLU, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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

            var slope = layer.GetFloat(0, 0f);
            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
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

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
            var outRt = owner.RentTempArray(srcTex.width, srcTex.height, outDepth, RenderTextureFormat.ARGBHalf);
            owner.Ops.LeakyReluPack4(srcTex.texture, slope, srcTex.packs, outRt);
            textureBlobs[layer.topNames[0]] = new NcnnRepro.TensorRef
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

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
                        var cmd = context.commandBuffer;
                        var blobs = context.blobs;
                        var shapes = context.shapes;
                        var remaining = context.remaining;
                        var pinnedNames = context.pinnedNames;

                        do
                        {
                                                var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
                                                var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
                                                var slope = layer.GetFloat(0, 0f);
                                                var outDepth = srcShape.dims == 4 ? srcShape.d * src.packs : src.packs;
                                                var outArr = owner.RentTempArray(cmd, src.width, src.height, outDepth, RenderTextureFormat.ARGBHalf);
                                                owner.Ops.LeakyReluPack4(cmd, src.texture, slope, src.packs, outArr);
                                                blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
                                                {
                                                    texture = outArr,
                                                    width = src.width,
                                                    height = src.height,
                                                    packs = src.packs,
                                                    refs = 1,
                                                    owned = true
                                                };
                                                if (shapes != null)
                                                    shapes[layer.topNames[0]] = srcShape;
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                continue;
                        } while (false);
        }
    }
}
