using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnReshapeLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnReshapeLayerRepro() : base(NcnnLayerTypes.Reshape, supportsBufferPath: true, supportsCommandBufferPath: true) { }

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
                                                if (bufferBlobs.TryGetValue(layer.bottomNames[0], out var reshapeBuf) && reshapeBuf != null)
                                                {
                                                    bufferBlobs[layer.topNames[0]] = reshapeBuf;
                                                    if (bufferRefs.TryGetValue(layer.bottomNames[0], out var reshapeRef) && reshapeRef != null)
                                                    {
                                                        bufferRefs[layer.topNames[0]] = reshapeRef;
                                                        reshapeRef.refs++;
                                                    }
                                                    var srcTensor = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                    if (srcTensor != null)
                                                        bufferViews[layer.topNames[0]] = NcnnRepro.ResolveReshapeTensor(srcTensor, layer);
                                                }
                                                else
                                                {
                                                    var src = owner.GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                                    var srcShape = NcnnRepro.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
                                                    var outShape = NcnnRepro.ResolveReshapeShape(srcShape, layer);

                                                    // If logical channels do not fill whole pack4 lanes, keeping the texture view
                                                    // would preserve padded channels and break later buffer consumers such as Permute.
                                                    if (srcShape.dims == 3 && (srcShape.c % 4) != 0)
                                                    {
                                                        var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                        if (srcBuf == null)
                                                            throw new InvalidOperationException("Reshape source not found: " + layer.bottomNames[0]);
                                                        bufferBlobs[layer.topNames[0]] = srcBuf;
                                                        if (NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews) is { } srcTensor)
                                                            bufferViews[layer.topNames[0]] = NcnnRepro.ResolveReshapeTensor(srcTensor, layer);
                                                    }
                                                    else
                                                    {
                                                        textureBlobs[layer.topNames[0]] = src;
                                                        textureShapes[layer.topNames[0]] = outShape;
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
                                                blobs[layer.topNames[0]] = src;
                                                src.refs++;
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }
    }
}
