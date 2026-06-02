using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnSplitLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnSplitLayerRepro() : base(NcnnLayerTypes.Split, supportsBufferPath: true, supportsCommandBufferPath: true) { }

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
                                                if (bufferBlobs.TryGetValue(layer.bottomNames[0], out var srcBuf) && srcBuf != null)
                                                {
                                                    var srcTensor = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                    for (var i = 0; i < layer.topNames.Length; i++)
                                                    {
                                                        bufferBlobs[layer.topNames[i]] = srcBuf;
                                                        if (bufferRefs.TryGetValue(layer.bottomNames[0], out var srcRef) && srcRef != null)
                                                        {
                                                            bufferRefs[layer.topNames[i]] = srcRef;
                                                            srcRef.refs++;
                                                        }
                                                        if (srcTensor != null)
                                                            bufferViews[layer.topNames[i]] = srcTensor;
                                                    }
                                                }
                                                else
                                                {
                                                    var src = owner.GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                                    var shape = NcnnRepro.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                                                    for (var i = 0; i < layer.topNames.Length; i++)
                                                    {
                                                        textureBlobs[layer.topNames[i]] = src;
                                                        textureShapes[layer.topNames[i]] = shape;
                                                        src.refs++;
                                                    }
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

                        do
                        {
                                                var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
                                                for (var i = 0; i < layer.topNames.Length; i++)
                                                {
                                                    blobs[layer.topNames[i]] = src;
                                                    src.refs++;
                                                }
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }
    }
}
