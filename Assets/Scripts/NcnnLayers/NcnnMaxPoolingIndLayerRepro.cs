using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnMaxPoolingIndLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnMaxPoolingIndLayerRepro() : base(NcnnLayerTypes.MaxPoolingInd, supportsBufferPath: true, supportsCommandBufferPath: false) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context) => owner.ExecuteMaxPoolingIndBufferLayer(layer, context);
    }

    public partial class NcnnRepro
    {
        internal void ExecuteMaxPoolingIndBufferLayer(NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                                    var kernelW = layer.GetInt(1, 0);
                                    var kernelH = layer.GetInt(11, kernelW);
                                    var strideW = layer.GetInt(2, 1);
                                    var strideH = layer.GetInt(12, strideW);
                                    var padLeft = layer.GetInt(3, 0);
                                    var padRight = layer.GetInt(14, padLeft);
                                    var padTop = layer.GetInt(13, padLeft);
                                    var padBottom = layer.GetInt(15, padTop);

                                    var outW = Mathf.Max(1, (src.width + padLeft + padRight - kernelW) / Mathf.Max(1, strideW) + 1);
                                    var outH = Mathf.Max(1, (src.height + padTop + padBottom - kernelH) / Mathf.Max(1, strideH) + 1);
                                    var outRt = RentTempArray(outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                                    var idxRt = RentTempArray(outW, outH, src.packs, RenderTextureFormat.ARGBFloat);
                                    if (UseTextureMaxPoolingInd)
                                    {
                                        _ops.PoolingPack4(src.texture, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, 0, outRt);
                                        _ops.MaxPoolingIndicesFromValuePack4(src.texture, outRt, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, idxRt);
                                    }
                                    else
                                    {
                                        ApplyMaxPoolingIndCpu(src, srcShape, kernelW, kernelH, strideW, strideH, padLeft, padTop, outW, outH, outRt, idxRt);
                                    }

                                    if (DebugCompareMaxPoolingLayers != null
                                        && (DebugCompareMaxPoolingLayers.Contains(layer.name) || DebugCompareMaxPoolingLayers.Contains("*")))
                                    {
                                        CompareMaxPoolingIndPath(layer.name, src, srcShape, outRt, idxRt, kernelW, kernelH, strideW, strideH, padLeft, padTop, outW, outH);
                                    }

                                    textureBlobs[layer.topNames[0]] = new TensorRef
                                    {
                                        texture = outRt,
                                        width = outW,
                                        height = outH,
                                        packs = src.packs,
                                        refs = 1,
                                        owned = true
                                    };
                                    textureShapes[layer.topNames[0]] = new BufferShape(3, outW, outH, 1, srcShape.c);
                                    indexBlobs[layer.topNames[1]] = new IndexRef
                                    {
                                        texture = idxRt,
                                        width = outW,
                                        height = outH,
                                        packs = src.packs,
                                        sourceWidth = src.width,
                                        sourceHeight = src.height,
                                        refs = _blobUseCount.TryGetValue(layer.topNames[1], out var idxUseCount) ? idxUseCount : 1,
                                        owned = true
                                    };
                                    Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }
    }
}
