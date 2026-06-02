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

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context) => owner.ExecuteSplitBufferLayer(layer, context);
        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context) => owner.ExecuteSplitCommandBufferLayer(layer, context);
    }

    public partial class NcnnRepro
    {
        internal void ExecuteSplitBufferLayer(NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                                        var srcTensor = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
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
                                        var src = GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                        var shape = GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                                        for (var i = 0; i < layer.topNames.Length; i++)
                                        {
                                            textureBlobs[layer.topNames[i]] = src;
                                            textureShapes[layer.topNames[i]] = shape;
                                            src.refs++;
                                        }
                                    }

                                    Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }

        internal void ExecuteSplitCommandBufferLayer(NcnnParamModel.Layer l, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            do
            {
                                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                                    for (var i = 0; i < l.topNames.Length; i++)
                                    {
                                        blobs[l.topNames[i]] = src;
                                        src.refs++;
                                    }
                                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }
    }
}
