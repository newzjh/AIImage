using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnSqueezeLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnSqueezeLayerRepro() : base(NcnnLayerTypes.Squeeze, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
                        var textureBlobs = context.textureBlobs;
                        var textureShapes = context.textureShapes;
                        var bufferBlobs = context.bufferBlobs;
                        var bufferRefs = context.bufferRefs;
                        var bufferViews = context.bufferViews;
                        var indexBlobs = context.indexBlobs;
                        var remaining = context.remaining;
                        var pinnedNames = context.pinnedNames;
                        var tempOwned = context.tempOwned;

                        do
                        {
                                                var axes = layer.GetInts(-23303, null);
                                                if (axes == null || axes.Length == 0)
                                                    axes = layer.GetInts(3, Array.Empty<int>());
                                                if (axes == null || axes.Length != 1 || axes[0] != 1)
                                                    throw new InvalidOperationException("Squeeze currently only supports axes=[1]: " + layer.name);

                                                if (bufferBlobs.TryGetValue(layer.bottomNames[0], out var squeezeBuf) && squeezeBuf != null)
                                                {
                                                    bufferBlobs[layer.topNames[0]] = squeezeBuf;
                                                    if (bufferRefs.TryGetValue(layer.bottomNames[0], out var squeezeRef) && squeezeRef != null)
                                                    {
                                                        bufferRefs[layer.topNames[0]] = squeezeRef;
                                                        squeezeRef.refs++;
                                                    }

                                                    var srcTensor = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                    if (srcTensor == null || srcTensor.dims != 3 || srcTensor.h != 1)
                                                        throw new InvalidOperationException("Squeeze expects dims=3 buffer input with h=1: " + layer.name);
                                                    bufferViews[layer.topNames[0]] = srcTensor.View(2, srcTensor.w, srcTensor.c, 1, 1);
                                                }
                                                else
                                                {
                                                    var src = owner.GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                                    var srcShape = NcnnRepro.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                                                    if (srcShape.dims != 3 || srcShape.h != 1)
                                                        throw new InvalidOperationException("Squeeze expects dims=3 texture input with h=1: " + layer.name);

                                                    textureBlobs[layer.topNames[0]] = src;
                                                    textureShapes[layer.topNames[0]] = new NcnnRepro.BufferShape(2, srcShape.w, srcShape.c, 1, 1);
                                                    src.refs++;
                                                }

                                                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            blobs[layer.topNames[0]] = src;
            src.refs++;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
        }
    }
}
