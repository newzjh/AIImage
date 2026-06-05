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
        public NcnnExpandDimsLayerRepro() : base(NcnnLayerTypes.ExpandDims, supportsBufferPath: true, supportsCommandBufferPath: true) { }

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
                                                if (axes == null || axes.Length == 0)
                                                    throw new InvalidOperationException("ExpandDims missing axes: " + layer.name);

                                                static NcnnTensorBuffer ExpandBufferView(NcnnTensorBuffer input, int[] expandAxes)
                                                {
                                                    if (input == null)
                                                        throw new ArgumentNullException(nameof(input));

                                                    var current = input;
                                                    for (var i = 0; i < expandAxes.Length; i++)
                                                    {
                                                        var outDims = current.dims + 1;
                                                        if (outDims > 4)
                                                            throw new InvalidOperationException("ExpandDims would exceed dims=4");

                                                        var ncnnAxis = expandAxes[i];
                                                        if (ncnnAxis < 0)
                                                            ncnnAxis += outDims;
                                                        if (ncnnAxis < 0 || ncnnAxis >= outDims)
                                                            throw new InvalidOperationException("ExpandDims axis out of range");

                                                        var tensorAxis = NcnnRepro.MapNcnnAxisToTensorAxis(outDims, ncnnAxis);
                                                        current = current.ExpandDims(tensorAxis);
                                                    }

                                                    return current;
                                                }

                                                static NcnnRepro.BufferShape ExpandTextureShape(NcnnRepro.BufferShape input, int[] expandAxes)
                                                {
                                                    var dims = input.dims;
                                                    var w = input.w;
                                                    var h = input.h;
                                                    var d = input.d;
                                                    var c = input.c;

                                                    for (var i = 0; i < expandAxes.Length; i++)
                                                    {
                                                        var outDims = dims + 1;
                                                        if (outDims > 4)
                                                            throw new InvalidOperationException("ExpandDims would exceed dims=4");

                                                        var ncnnAxis = expandAxes[i];
                                                        if (ncnnAxis < 0)
                                                            ncnnAxis += outDims;
                                                        if (ncnnAxis < 0 || ncnnAxis >= outDims)
                                                            throw new InvalidOperationException("ExpandDims axis out of range");

                                                        var tensorAxis = NcnnRepro.MapNcnnAxisToTensorAxis(outDims, ncnnAxis);
                                                        var sizes = new[] { w, h, dims == 4 ? d : c, dims == 4 ? c : 1 };
                                                        var expanded = new[] { 1, 1, 1, 1 };
                                                        for (var axisIndex = 0; axisIndex < outDims; axisIndex++)
                                                        {
                                                            if (axisIndex < tensorAxis)
                                                                expanded[axisIndex] = sizes[axisIndex];
                                                            else if (axisIndex == tensorAxis)
                                                                expanded[axisIndex] = 1;
                                                            else
                                                                expanded[axisIndex] = sizes[axisIndex - 1];
                                                        }

                                                        dims = outDims;
                                                        w = expanded[0];
                                                        h = expanded[1];
                                                        if (dims == 3)
                                                        {
                                                            d = 1;
                                                            c = expanded[2];
                                                        }
                                                        else
                                                        {
                                                            d = expanded[2];
                                                            c = expanded[3];
                                                        }
                                                    }

                                                    return new NcnnRepro.BufferShape(dims, w, h, d, c);
                                                }

                                                if (bufferBlobs.TryGetValue(layer.bottomNames[0], out var expandBuf) && expandBuf != null)
                                                {
                                                    bufferBlobs[layer.topNames[0]] = expandBuf;
                                                    if (bufferRefs.TryGetValue(layer.bottomNames[0], out var expandRef) && expandRef != null)
                                                    {
                                                        bufferRefs[layer.topNames[0]] = expandRef;
                                                        expandRef.refs++;
                                                    }

                                                    var srcTensor = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                    if (srcTensor == null)
                                                        throw new InvalidOperationException("ExpandDims expects buffer input: " + layer.name);
                                                    bufferViews[layer.topNames[0]] = ExpandBufferView(srcTensor, axes);
                                                }
                                                else
                                                {
                                                    var src = owner.GetOrMaterializeTexture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews);
                                                    var srcShape = NcnnRepro.GetTextureShape(textureShapes, src, layer.bottomNames[0]);

                                                    textureBlobs[layer.topNames[0]] = src;
                                                    textureShapes[layer.topNames[0]] = ExpandTextureShape(srcShape, axes);
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
