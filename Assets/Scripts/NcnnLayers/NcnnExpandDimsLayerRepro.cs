using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnExpandDimsLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnExpandDimsLayerRepro() : base(NcnnLayerTypes.ExpandDims, supportsBufferPath: true, supportsCommandBufferPath: false) { }

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
                                                    throw new InvalidOperationException("ExpandDims currently only supports axes=[1]: " + layer.name);

                                                if (bufferBlobs.TryGetValue(layer.bottomNames[0], out var expandBuf) && expandBuf != null)
                                                {
                                                    bufferBlobs[layer.topNames[0]] = expandBuf;
                                                    if (bufferRefs.TryGetValue(layer.bottomNames[0], out var expandRef) && expandRef != null)
                                                    {
                                                        bufferRefs[layer.topNames[0]] = expandRef;
                                                        expandRef.refs++;
                                                    }

                                                    var srcTensor = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                    if (srcTensor == null || srcTensor.dims != 2)
                                                        throw new InvalidOperationException("ExpandDims expects dims=2 buffer input: " + layer.name);
                                                    bufferViews[layer.topNames[0]] = srcTensor.View(3, srcTensor.w, 1, 1, srcTensor.h);
                                                }
                                                else
                                                {
                                                    var src = owner.GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                                    var srcShape = NcnnRepro.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                                                    if (srcShape.dims != 2)
                                                        throw new InvalidOperationException("ExpandDims expects dims=2 texture input: " + layer.name);

                                                    textureBlobs[layer.topNames[0]] = src;
                                                    textureShapes[layer.topNames[0]] = new NcnnRepro.BufferShape(3, srcShape.w, 1, 1, srcShape.h);
                                                    src.refs++;
                                                }

                                                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }
    }
}
