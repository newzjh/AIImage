using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnSliceLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnSliceLayerRepro() : base(NcnnLayerTypes.Slice, supportsBufferPath: true, supportsCommandBufferPath: false) { }

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
                                                var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                if (srcBuf == null || srcView == null)
                                                    throw new InvalidOperationException("Slice source not found: " + layer.name);

                                                var sliceParams = layer.GetInts(-23300, null);
                                                var indices = layer.GetInts(-23302, null);
                                                var ncnnAxis = layer.GetInt(1, 0);
                                                if (ncnnAxis < 0)
                                                    ncnnAxis += srcView.dims;
                                                var axis = NcnnRepro.MapNcnnAxisToTensorAxis(srcView.dims, ncnnAxis);
                                                var axisSize = NcnnRepro.GetAxisSize(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c, axis);

                                                var begin = 0;
                                                for (var i = 0; i < layer.topNames.Length; i++)
                                                {
                                                    int sliceSize;
                                                    if (indices != null && indices.Length > 0)
                                                    {
                                                        if (i == layer.topNames.Length - 1)
                                                        {
                                                            sliceSize = axisSize - begin;
                                                        }
                                                        else
                                                        {
                                                            var indice = indices[Mathf.Min(i, indices.Length - 1)];
                                                            if (indice < 0)
                                                                indice += axisSize;
                                                            sliceSize = indice - begin;
                                                        }
                                                    }
                                                    else
                                                    {
                                                        if (sliceParams == null || sliceParams.Length == 0)
                                                            throw new InvalidOperationException("Slice missing params: " + layer.name);
                                                        sliceSize = sliceParams[Mathf.Min(i, sliceParams.Length - 1)];
                                                        if (sliceSize == -233)
                                                            sliceSize = (axisSize - begin) / Mathf.Max(1, layer.topNames.Length - i);
                                                    }

                                                    if (sliceSize <= 0)
                                                        throw new InvalidOperationException("Slice produced empty output: " + layer.name + " top=" + i);

                                                    var outW = srcView.w;
                                                    var outH = srcView.h;
                                                    var outD = srcView.d;
                                                    var outC = srcView.c;
                                                    if (axis == 0) outW = sliceSize;
                                                    else if (axis == 1) outH = sliceSize;
                                                    else if (axis == 2 && srcView.dims == 4) outD = sliceSize;
                                                    else if (axis == 2 || axis == 3) outC = sliceSize;

                                                    var outCount = outW * outH * outD * outC;
                                                    var outBuf = owner.RentTempBuffer(outCount, sizeof(float));
                                                    owner.Ops.Slice(srcBuf, srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c, axis, begin, outW, outH, outD, outC, outBuf);

                                                    bufferBlobs[layer.topNames[i]] = outBuf;
                                                    bufferRefs[layer.topNames[i]] = owner.NewOwnedBufferRef(layer.topNames[i], outBuf);
                                                    bufferViews[layer.topNames[i]] = new NcnnTensorBuffer(outBuf, srcView.dims, outW, outH, outD, outC, false);
                                                    begin += sliceSize;
                                                }

                                                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }
    }
}
