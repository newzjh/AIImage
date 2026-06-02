using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnReductionLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnReductionLayerRepro() : base(NcnnLayerTypes.Reduction, supportsBufferPath: true, supportsCommandBufferPath: false) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context) => owner.ExecuteReductionBufferLayer(layer, context);
    }

    public partial class NcnnRepro
    {
        internal void ExecuteReductionBufferLayer(NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                                    var srcBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                    if (srcBuf == null)
                                        throw new InvalidOperationException("Reduction source not found: " + layer.bottomNames[0]);

                                    var srcTensor = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                    var reduceAll = layer.GetInt(1, 1) != 0;
                                    var coeff = layer.GetFloat(2, 1f);
                                    var axes = layer.GetInts(-23303, null);
                                    var keepDims = layer.GetInt(4, 0) != 0;

                                    if (srcTensor == null)
                                        throw new InvalidOperationException("Reduction shape missing: " + layer.name);

                                    if (srcTensor.dims == 3 && !reduceAll && axes != null && axes.Length == 2)
                                    {
                                        var axisA = axes[0] < 0 ? axes[0] + srcTensor.dims : axes[0];
                                        var axisB = axes[1] < 0 ? axes[1] + srcTensor.dims : axes[1];
                                        var reduceHW = (axisA == 1 && axisB == 2) || (axisA == 2 && axisB == 1);
                                        if (reduceHW)
                                        {
                                            var op = layer.GetInt(0, 0);
                                            if (op != 0 && op != 3)
                                                throw new InvalidOperationException("Reduction dims=3 currently supports SUM/MEAN only: " + layer.name);

                                            var data = new float[srcBuf.count];
                                            srcBuf.GetData(data);
                                            var plane = srcTensor.w * srcTensor.h;
                                            var reduced = new float[srcTensor.c];
                                            var scale = op == 3 ? (coeff / Mathf.Max(1, plane)) : coeff;
                                            for (var channelIndex = 0; channelIndex < srcTensor.c; channelIndex++)
                                            {
                                                var sum = 0f;
                                                var offset = channelIndex * plane;
                                                for (var i = 0; i < plane; i++)
                                                    sum += data[offset + i];
                                                reduced[channelIndex] = sum * scale;
                                            }

                                            var outBuf3 = RentTempBuffer(reduced.Length, sizeof(float));
                                            outBuf3.SetData(reduced);
                                            bufferBlobs[layer.topNames[0]] = outBuf3;
                                            bufferViews[layer.topNames[0]] = keepDims
                                                ? new NcnnTensorBuffer(outBuf3, 3, 1, 1, 1, srcTensor.c, false)
                                                : new NcnnTensorBuffer(outBuf3, 1, srcTensor.c, 1, 1, 1, false);
                                            tempOwned.Add(outBuf3);
                                            Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                            continue;
                                        }
                                    }

                                    if (srcTensor.dims != 2)
                                        throw new InvalidOperationException("Reduction currently expects dims=2 buffer input: " + layer.name);

                                    if (reduceAll)
                                    {
                                        var outBufAll = RentTempBuffer(1, sizeof(float));
                                        _ops.ReductionBuf(srcBuf, srcBuf.count, 1, layer.GetInt(0, 0), coeff, outBufAll);
                                        bufferBlobs[layer.topNames[0]] = outBufAll;
                                        bufferViews[layer.topNames[0]] = keepDims
                                            ? new NcnnTensorBuffer(outBufAll, 2, 1, 1, 1, 1, false)
                                            : new NcnnTensorBuffer(outBufAll, 1, 1, 1, 1, 1, false);
                                        tempOwned.Add(outBufAll);
                                        Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                        continue;
                                    }

                                    if (axes == null || axes.Length == 0)
                                        throw new InvalidOperationException("Reduction axes missing: " + layer.name);
                                    if (axes.Length != 1)
                                        throw new InvalidOperationException("Reduction axes length > 1 not supported yet: " + layer.name);

                                    var axis = axes[0];
                                    var positiveAxis = axis < 0 ? axis + srcTensor.dims : axis;
                                    int reduceElems;
                                    int outCount;
                                    NcnnTensorBuffer outView;
                                    if (positiveAxis == 1)
                                    {
                                        reduceElems = srcTensor.w;
                                        outCount = srcTensor.h;
                                        outView = keepDims ? null : null;
                                    }
                                    else if (positiveAxis == 0)
                                    {
                                        reduceElems = srcTensor.h;
                                        outCount = srcTensor.w;
                                        var tempTranspose = RentTempBuffer(srcTensor.buffer.count, sizeof(float));
                                        _ops.Permute(srcBuf, 2, srcTensor.w, srcTensor.h, 1, 1, 1, tempTranspose);
                                        srcBuf = tempTranspose;
                                        tempOwned.Add(tempTranspose);
                                        outView = keepDims ? null : null;
                                    }
                                    else
                                    {
                                        throw new InvalidOperationException("Reduction axis not supported for dims=2: " + axis + " | " + layer.name);
                                    }

                                    var outBuf = RentTempBuffer(outCount, sizeof(float));
                                    _ops.ReductionBuf(srcBuf, reduceElems, outCount, layer.GetInt(0, 0), coeff, outBuf);
                                    bufferBlobs[layer.topNames[0]] = outBuf;
                                    bufferViews[layer.topNames[0]] = positiveAxis == 1
                                        ? (keepDims
                                            ? new NcnnTensorBuffer(outBuf, 2, 1, srcTensor.h, 1, 1, false)
                                            : new NcnnTensorBuffer(outBuf, 1, srcTensor.h, 1, 1, 1, false))
                                        : (keepDims
                                            ? new NcnnTensorBuffer(outBuf, 2, srcTensor.w, 1, 1, 1, false)
                                            : new NcnnTensorBuffer(outBuf, 1, srcTensor.w, 1, 1, 1, false));
                                    tempOwned.Add(outBuf);
                                    Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }
    }
}
