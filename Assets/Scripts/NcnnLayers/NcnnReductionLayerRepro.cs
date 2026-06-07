using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnReductionLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnReductionLayerRepro() : base(NcnnLayerTypes.Reduction, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                                                using var srcReadable = owner.GetReadableTensorInput(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                var srcBuf = srcReadable?.buffer;
                                                if (srcBuf == null)
                                                    throw new InvalidOperationException("Reduction source not found: " + layer.bottomNames[0]);

                                                var srcTensor = srcReadable;
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

                                                        var plane = srcTensor.w * srcTensor.h;
                                                        var scale = op == 3 ? (coeff / Mathf.Max(1, plane)) : coeff;
                                                        var outBuf3 = owner.RentTempBuffer(srcTensor.c, sizeof(float));
                                                        owner.Ops.ReductionRowsBuf(srcBuf, plane, srcTensor.c, 0, scale, outBuf3);
                                                        var outTensor3 = keepDims
                                                            ? new NcnnTensorBuffer(outBuf3, 3, 1, 1, 1, srcTensor.c, true, owner.ReturnTempBuffer)
                                                            : new NcnnTensorBuffer(outBuf3, 1, srcTensor.c, 1, 1, 1, true, owner.ReturnTempBuffer);
                                                        owner.PublishTensorBufferOutput(
                                                            layer.topNames[0],
                                                            outTensor3,
                                                            preferTexture: outTensor3.dims <= 3,
                                                            textureBlobs,
                                                            textureShapes,
                                                            bufferBlobs,
                                                            bufferRefs,
                                                            bufferViews,
                                                            tempOwned);
                                                        owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                        continue;
                                                    }
                                                }

                                                if (srcTensor.dims != 2)
                                                    throw new InvalidOperationException("Reduction currently expects dims=2 buffer input: " + layer.name);

                                                if (reduceAll)
                                                {
                                                    var outBufAll = owner.RentTempBuffer(1, sizeof(float));
                                                    owner.Ops.ReductionBuf(srcBuf, srcBuf.count, 1, layer.GetInt(0, 0), coeff, outBufAll);
                                                    var outTensorAll = keepDims
                                                        ? new NcnnTensorBuffer(outBufAll, 2, 1, 1, 1, 1, true, owner.ReturnTempBuffer)
                                                        : new NcnnTensorBuffer(outBufAll, 1, 1, 1, 1, 1, true, owner.ReturnTempBuffer);
                                                    owner.PublishTensorBufferOutput(
                                                        layer.topNames[0],
                                                        outTensorAll,
                                                        preferTexture: outTensorAll.dims <= 3,
                                                        textureBlobs,
                                                        textureShapes,
                                                        bufferBlobs,
                                                        bufferRefs,
                                                        bufferViews,
                                                        tempOwned);
                                                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
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
                                                if (positiveAxis == 1)
                                                {
                                                    reduceElems = srcTensor.w;
                                                    outCount = srcTensor.h;
                                                }
                                                else if (positiveAxis == 0)
                                                {
                                                    reduceElems = srcTensor.h;
                                                    outCount = srcTensor.w;
                                                    var tempTranspose = owner.RentTempBuffer(srcTensor.buffer.count, sizeof(float));
                                                    owner.Ops.Permute(srcBuf, 2, srcTensor.w, srcTensor.h, 1, 1, 1, tempTranspose);
                                                    srcBuf = tempTranspose;
                                                    tempOwned.Add(tempTranspose);
                                                }
                                                else
                                                {
                                                    throw new InvalidOperationException("Reduction axis not supported for dims=2: " + axis + " | " + layer.name);
                                                }

                                                var outBuf = owner.RentTempBuffer(outCount, sizeof(float));
                                                owner.Ops.ReductionRowsBuf(srcBuf, reduceElems, outCount, layer.GetInt(0, 0), coeff, outBuf);
                                                var outTensor = positiveAxis == 1
                                                    ? (keepDims
                                                        ? new NcnnTensorBuffer(outBuf, 2, 1, srcTensor.h, 1, 1, true, owner.ReturnTempBuffer)
                                                        : new NcnnTensorBuffer(outBuf, 1, srcTensor.h, 1, 1, 1, true, owner.ReturnTempBuffer))
                                                    : (keepDims
                                                        ? new NcnnTensorBuffer(outBuf, 2, srcTensor.w, 1, 1, 1, true, owner.ReturnTempBuffer)
                                                        : new NcnnTensorBuffer(outBuf, 1, srcTensor.w, 1, 1, 1, true, owner.ReturnTempBuffer));
                                                owner.PublishTensorBufferOutput(
                                                    layer.topNames[0],
                                                    outTensor,
                                                    preferTexture: outTensor.dims <= 3,
                                                    textureBlobs,
                                                    textureShapes,
                                                    bufferBlobs,
                                                    bufferRefs,
                                                    bufferViews,
                                                    tempOwned);
                                                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            static int[] ToNcnnAxisSizes(NcnnRepro.BufferShape shape)
            {
                return shape.dims switch
                {
                    1 => new[] { shape.w },
                    2 => new[] { shape.h, shape.w },
                    3 => new[] { shape.c, shape.h, shape.w },
                    4 => new[] { shape.c, shape.d, shape.h, shape.w },
                    _ => throw new InvalidOperationException("Unsupported reduction dims: " + shape.dims)
                };
            }

            static NcnnRepro.BufferShape FromNcnnAxisSizes(int[] sizes)
            {
                if (sizes == null || sizes.Length == 0)
                    return new NcnnRepro.BufferShape(1, 1, 1, 1, 1);

                return sizes.Length switch
                {
                    1 => new NcnnRepro.BufferShape(1, Mathf.Max(1, sizes[0]), 1, 1, 1),
                    2 => new NcnnRepro.BufferShape(2, Mathf.Max(1, sizes[1]), Mathf.Max(1, sizes[0]), 1, 1),
                    3 => new NcnnRepro.BufferShape(3, Mathf.Max(1, sizes[2]), Mathf.Max(1, sizes[1]), 1, Mathf.Max(1, sizes[0])),
                    4 => new NcnnRepro.BufferShape(4, Mathf.Max(1, sizes[3]), Mathf.Max(1, sizes[2]), Mathf.Max(1, sizes[1]), Mathf.Max(1, sizes[0])),
                    _ => throw new InvalidOperationException("Unsupported reduction output dims: " + sizes.Length)
                };
            }

            var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var reduceAll = layer.GetInt(1, 1) != 0;
            var keepDims = layer.GetInt(4, 0) != 0;
            var axes = layer.GetInts(-23303, null);
            var axisSizes = ToNcnnAxisSizes(srcShape);
            var reduceMask = new bool[axisSizes.Length];

            if (reduceAll)
            {
                for (var i = 0; i < reduceMask.Length; i++)
                    reduceMask[i] = true;
            }
            else
            {
                if (axes == null || axes.Length == 0)
                    throw new InvalidOperationException("Reduction axes missing: " + layer.name);

                for (var i = 0; i < axes.Length; i++)
                {
                    var axis = axes[i];
                    if (axis < 0)
                        axis += axisSizes.Length;
                    if (axis < 0 || axis >= axisSizes.Length)
                        throw new InvalidOperationException("Reduction axis out of range: " + layer.name);
                    reduceMask[axis] = true;
                }
            }

            NcnnRepro.BufferShape outShape;
            if (keepDims)
            {
                var kept = new int[axisSizes.Length];
                for (var i = 0; i < axisSizes.Length; i++)
                    kept[i] = reduceMask[i] ? 1 : Mathf.Max(1, axisSizes[i]);
                outShape = FromNcnnAxisSizes(kept);
            }
            else
            {
                var keptCount = 0;
                for (var i = 0; i < axisSizes.Length; i++)
                {
                    if (!reduceMask[i])
                        keptCount++;
                }

                if (keptCount == 0)
                {
                    outShape = new NcnnRepro.BufferShape(1, 1, 1, 1, 1);
                }
                else
                {
                    var kept = new int[keptCount];
                    var cursor = 0;
                    for (var i = 0; i < axisSizes.Length; i++)
                    {
                        if (!reduceMask[i])
                            kept[cursor++] = Mathf.Max(1, axisSizes[i]);
                    }
                    outShape = FromNcnnAxisSizes(kept);
                }
            }

            owner.PublishCmdPlaceholder(cmd, layer.topNames[0], outShape, blobs, shapes);
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }
    }
}
