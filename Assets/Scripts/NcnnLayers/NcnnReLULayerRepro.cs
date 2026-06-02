using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnReLULayerRepro : NcnnBaseLayerRepro
    {
        public NcnnReLULayerRepro() : base(NcnnLayerTypes.ReLU, supportsBufferPath: true, supportsCommandBufferPath: false) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context) => owner.ExecuteReLUBufferLayer(layer, context);
    }

    public partial class NcnnRepro
    {
        internal void ExecuteReLUBufferLayer(NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                                    var src = GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                    var srcShape = GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                                    var slope = layer.GetFloat(0, 0f);
                                    var outRt = RentTempArray(src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                                    _ops.LeakyReluPack4(src.texture, slope, src.packs, outRt);
                                    textureBlobs[layer.topNames[0]] = new TensorRef
                                    {
                                        texture = outRt,
                                        width = outRt.width,
                                        height = outRt.height,
                                        packs = src.packs,
                                        refs = 1,
                                        owned = true
                                    };
                                    textureShapes[layer.topNames[0]] = srcShape;
                                    Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }
    }
}
