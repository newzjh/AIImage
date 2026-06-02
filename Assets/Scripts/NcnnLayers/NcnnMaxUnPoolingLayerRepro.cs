using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnMaxUnPoolingLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnMaxUnPoolingLayerRepro() : base(NcnnLayerTypes.MaxUnPooling, supportsBufferPath: true, supportsCommandBufferPath: false) { }

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
                                                var src = owner.GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                                if (!indexBlobs.TryGetValue(layer.bottomNames[1], out var idx) || idx == null || idx.texture == null)
                                                    throw new InvalidOperationException("MaxUnPooling index source not found: " + layer.name);

                                                var srcShape = NcnnRepro.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                                                var outRt = owner.RentTempArray(idx.sourceWidth, idx.sourceHeight, src.packs, RenderTextureFormat.ARGBHalf);
                                                owner.ApplyMaxUnPoolingCpu(src, srcShape, idx, idx.sourceWidth, idx.sourceHeight, outRt);
                                                textureBlobs[layer.topNames[0]] = new NcnnRepro.TensorRef
                                                {
                                                    texture = outRt,
                                                    width = outRt.width,
                                                    height = outRt.height,
                                                    packs = src.packs,
                                                    refs = 1,
                                                    owned = true
                                                };
                                                textureShapes[layer.topNames[0]] = new NcnnRepro.BufferShape(3, idx.sourceWidth, idx.sourceHeight, 1, srcShape.c);
                                                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, new[] { layer.bottomNames[0] }, pinnedNames);
                                                owner.ConsumeIndex(indexBlobs, remaining, new[] { layer.bottomNames[1] }, pinnedNames);
                                                continue;
                        } while (false);
        }
    }
}
