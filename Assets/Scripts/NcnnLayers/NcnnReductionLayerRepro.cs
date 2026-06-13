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

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (TryExecuteRenderTexturePath(owner, layer, context))
                return;

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

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
            if (TryExecuteRenderTexturePath(owner, layer, context))
                return;

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            if (TryExecuteCommandBufferTexturePath(owner, layer, context))
                return;

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

            owner.DebugLog?.Invoke(
                "[CmdPlaceholder][Reduction]"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c
                + " | reduceAll=" + (reduceAll ? "1" : "0")
                + " | keepDims=" + (keepDims ? "1" : "0"));
            owner.PublishCmdPlaceholder(cmd, layer.topNames[0], outShape, blobs, shapes);
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool TryExecuteCommandBufferTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            if (owner == null || layer == null || context == null)
                return false;
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;

            var srcTex = NcnnRepro.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var srcShape = NcnnRepro.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            if (TryExecuteCommandBufferScalar2DTexturePath(owner, layer, srcTex, srcShape, context))
                return true;
            if (srcTex == null
                || srcTex.texture == null
                || srcShape.dims != 3
                || srcShape.d != 1
                || !NcnnRepro.MatchesPack4TextureStorage(srcTex, srcShape))
            {
                return false;
            }

            var reduceAll = layer.GetInt(1, 1) != 0;
            var keepDims = layer.GetInt(4, 0) != 0;
            var op = layer.GetInt(0, 0);
            var coeff = layer.GetFloat(2, 1f);
            var axes = layer.GetInts(-23303, null);
            if (!TryResolveSpatialReductionAxes(srcShape, reduceAll, axes, out var reduceSpatialOnly, out var reduceSpatialAndChannels))
                return false;
            if (op != 3 && op != 0)
                return false;
            if (!reduceSpatialOnly && !reduceSpatialAndChannels)
                return false;

            var cmd = context.commandBuffer;
            var area = Mathf.Max(1, srcShape.w * srcShape.h);
            var outTop = layer.topNames[0];

            ComputeTexture pooled = null;
            if (reduceSpatialOnly || reduceSpatialAndChannels)
            {
                pooled = owner.RentTempArray(cmd, 1, 1, srcTex.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.PoolingPack4(cmd, srcTex.texture, srcTex.packs, srcShape.w, srcShape.h, 1, 1, 0, 0, 1, pooled);
                if (op == 0 && Mathf.Abs(coeff - area) > 1e-6f)
                {
                    var scaled = owner.RentTempArray(cmd, 1, 1, srcTex.packs, RenderTextureFormat.ARGBHalf);
                    owner.Ops.PointwisePack4(cmd, pooled, srcTex.packs, NcnnOps.PointwiseType.ScaleScalar, coeff * area, 0f, scaled);
                    owner.ReturnTempArray(cmd, pooled);
                    pooled = scaled;
                }
                else if (op == 3 && Mathf.Abs(coeff - 1f) > 1e-6f)
                {
                    var scaled = owner.RentTempArray(cmd, 1, 1, srcTex.packs, RenderTextureFormat.ARGBHalf);
                    owner.Ops.PointwisePack4(cmd, pooled, srcTex.packs, NcnnOps.PointwiseType.ScaleScalar, coeff, 0f, scaled);
                    owner.ReturnTempArray(cmd, pooled);
                    pooled = scaled;
                }
            }

            if (reduceSpatialOnly && keepDims)
            {
                context.blobs[outTop] = new NcnnRepro.CmdTensorRef
                {
                    texture = pooled,
                    width = 1,
                    height = 1,
                    packs = srcTex.packs,
                    refs = 1,
                    owned = true,
                    hasLogicalShape = true,
                    logicalShape = new NcnnRepro.BufferShape(3, 1, 1, 1, srcShape.c),
                    hasStorageShape = true,
                    storageShape = new NcnnRepro.BufferShape(3, 1, 1, 1, srcShape.c)
                };
                context.shapes[outTop] = new NcnnRepro.BufferShape(3, 1, 1, 1, srcShape.c);
                owner.ConsumeCmd(cmd, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
                return true;
            }

            if (reduceSpatialOnly && !keepDims)
            {
                var outRt = owner.RentTempArray(cmd, srcShape.c, 1, 1, RenderTextureFormat.ARGBHalf);
                owner.Ops.Pack4ChannelsToWidth(cmd, pooled, srcShape.c, outRt);
                owner.ReturnTempArray(cmd, pooled);
                context.blobs[outTop] = new NcnnRepro.CmdTensorRef
                {
                    texture = outRt,
                    width = srcShape.c,
                    height = 1,
                    packs = 1,
                    refs = 1,
                    owned = true,
                    hasLogicalShape = true,
                    logicalShape = new NcnnRepro.BufferShape(1, srcShape.c, 1, 1, 1),
                    hasStorageShape = true,
                    storageShape = new NcnnRepro.BufferShape(3, srcShape.c, 1, 1, 1)
                };
                context.shapes[outTop] = new NcnnRepro.BufferShape(1, srcShape.c, 1, 1, 1);
                owner.ConsumeCmd(cmd, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
                return true;
            }

            if (reduceSpatialAndChannels && !keepDims)
            {
                var outRt = owner.RentTempArray(cmd, srcShape.c, 1, 1, RenderTextureFormat.ARGBHalf);
                owner.Ops.Pack4ChannelsToWidth(cmd, pooled, srcShape.c, outRt);
                owner.ReturnTempArray(cmd, pooled);
                context.blobs[outTop] = new NcnnRepro.CmdTensorRef
                {
                    texture = outRt,
                    width = srcShape.c,
                    height = 1,
                    packs = 1,
                    refs = 1,
                    owned = true,
                    hasLogicalShape = true,
                    logicalShape = new NcnnRepro.BufferShape(1, srcShape.c, 1, 1, 1),
                    hasStorageShape = true,
                    storageShape = new NcnnRepro.BufferShape(3, srcShape.c, 1, 1, 1)
                };
                context.shapes[outTop] = new NcnnRepro.BufferShape(1, srcShape.c, 1, 1, 1);
                owner.ConsumeCmd(cmd, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
                return true;
            }

            if (pooled != null)
                owner.ReturnTempArray(cmd, pooled);
            return false;
        }

        private static bool TryExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (owner == null || layer == null || context == null)
                return false;
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;
            if (TryExecuteRenderTextureScalar2DPath(owner, layer, context))
                return true;
            if (!owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out var srcTex,
                    out var srcShape))
                return false;

            var reduceAll = layer.GetInt(1, 1) != 0;
            var keepDims = layer.GetInt(4, 0) != 0;
            var op = layer.GetInt(0, 0);
            var coeff = layer.GetFloat(2, 1f);
            var axes = layer.GetInts(-23303, null);

            if (srcShape.dims != 3 || srcShape.d != 1 || !NcnnRepro.MatchesPack4TextureStorage(srcTex, srcShape))
                return false;
            if (!TryResolveSpatialReductionAxes(srcShape, reduceAll, axes, out var reduceSpatialOnly, out var reduceSpatialAndChannels))
                return false;
            if (op != 3 && op != 0)
                return false;
            if (!reduceSpatialOnly && !reduceSpatialAndChannels)
                return false;

            var area = Mathf.Max(1, srcShape.w * srcShape.h);
            var outTop = layer.topNames[0];

            if (reduceSpatialOnly && keepDims)
            {
                var outRt = owner.RentTempArray(1, 1, srcTex.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.PoolingPack4(srcTex.texture, srcTex.packs, srcShape.w, srcShape.h, 1, 1, 0, 0, 1, outRt);
                if (op == 0 && Mathf.Abs(coeff - area) > 1e-6f)
                {
                    var scaledRt = owner.RentTempArray(1, 1, srcTex.packs, RenderTextureFormat.ARGBHalf);
                    owner.Ops.PointwisePack4(outRt, srcTex.packs, NcnnOps.PointwiseType.ScaleScalar, coeff * area, 0f, scaledRt);
                    owner.ReturnTempArray(outRt);
                    outRt = scaledRt;
                }
                else if (op == 3 && Mathf.Abs(coeff - 1f) > 1e-6f)
                {
                    var scaledRt = owner.RentTempArray(1, 1, srcTex.packs, RenderTextureFormat.ARGBHalf);
                    owner.Ops.PointwisePack4(outRt, srcTex.packs, NcnnOps.PointwiseType.ScaleScalar, coeff, 0f, scaledRt);
                    owner.ReturnTempArray(outRt);
                    outRt = scaledRt;
                }

                var outShape = new NcnnRepro.BufferShape(3, 1, 1, 1, srcShape.c);
                NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, outTop, outRt, outShape, outShape);
                owner.Consume(
                    context.textureBlobs,
                    context.bufferBlobs,
                    context.bufferRefs,
                    context.bufferViews,
                    context.remaining,
                    layer.bottomNames,
                    context.pinnedNames);
                return true;
            }

            if (reduceSpatialAndChannels && !keepDims)
            {
                var pooledRt = owner.RentTempArray(1, 1, srcTex.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.PoolingPack4(srcTex.texture, srcTex.packs, srcShape.w, srcShape.h, 1, 1, 0, 0, 1, pooledRt);
                if (op == 0 && Mathf.Abs(coeff - area) > 1e-6f)
                {
                    var scaledRt = owner.RentTempArray(1, 1, srcTex.packs, RenderTextureFormat.ARGBHalf);
                    owner.Ops.PointwisePack4(pooledRt, srcTex.packs, NcnnOps.PointwiseType.ScaleScalar, coeff * area, 0f, scaledRt);
                    owner.ReturnTempArray(pooledRt);
                    pooledRt = scaledRt;
                }
                else if (op == 3 && Mathf.Abs(coeff - 1f) > 1e-6f)
                {
                    var scaledRt = owner.RentTempArray(1, 1, srcTex.packs, RenderTextureFormat.ARGBHalf);
                    owner.Ops.PointwisePack4(pooledRt, srcTex.packs, NcnnOps.PointwiseType.ScaleScalar, coeff, 0f, scaledRt);
                    owner.ReturnTempArray(pooledRt);
                    pooledRt = scaledRt;
                }

                var outRt = owner.RentTempArray(srcShape.c, 1, 1, RenderTextureFormat.ARGBHalf);
                owner.Ops.Pack4ChannelsToWidth(pooledRt, srcShape.c, outRt);
                owner.ReturnTempArray(pooledRt);
                var logicalShape = new NcnnRepro.BufferShape(1, srcShape.c, 1, 1, 1);
                var storageShape = new NcnnRepro.BufferShape(3, srcShape.c, 1, 1, 1);
                NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, outTop, outRt, logicalShape, storageShape);
                owner.Consume(
                    context.textureBlobs,
                    context.bufferBlobs,
                    context.bufferRefs,
                    context.bufferViews,
                    context.remaining,
                    layer.bottomNames,
                    context.pinnedNames);
                return true;
            }

            return false;
        }

        private static bool TryExecuteRenderTextureScalar2DPath(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnLayerBufferContext context)
        {
            if (!NcnnRepro.TryGetExistingTexture(
                    context.textureBlobs,
                    context.textureShapes,
                    layer.bottomNames[0],
                    out var srcTex,
                    out var srcShape))
            {
                return false;
            }

            if (!CanUseScalar2DTexturePath(srcTex, srcShape))
                return false;

            var reduceAll = layer.GetInt(1, 1) != 0;
            var keepDims = layer.GetInt(4, 0) != 0;
            var op = layer.GetInt(0, 0);
            var coeff = layer.GetFloat(2, 1f);
            var axes = layer.GetInts(-23303, null);
            if (!TryResolveScalar2DReduction(srcShape, reduceAll, keepDims, axes, out var reduceAlongWidth, out var outShape))
                return false;
            if (!CanUseScalarTextureReductionOp(op))
                return false;

            var outRt = owner.RentTempArray(Mathf.Max(1, outShape.w), Mathf.Max(1, outShape.h), 1, RenderTextureFormat.ARGBHalf);
            ExecuteScalar2DReduction(owner.Ops, srcTex.texture, srcShape, reduceAlongWidth, op, coeff, outRt);
            var storageShape = new NcnnRepro.BufferShape(3, outRt.width, outRt.height, 1, 1);
            NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], outRt, outShape, storageShape);
            owner.Consume(
                context.textureBlobs,
                context.bufferBlobs,
                context.bufferRefs,
                context.bufferViews,
                context.remaining,
                layer.bottomNames,
                context.pinnedNames);
            return true;
        }

        private static bool TryExecuteCommandBufferScalar2DTexturePath(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnRepro.CmdTensorRef srcTex,
            NcnnRepro.BufferShape srcShape,
            NcnnLayerCommandBufferContext context)
        {
            if (!CanUseScalar2DTexturePath(srcTex, srcShape))
                return false;

            var reduceAll = layer.GetInt(1, 1) != 0;
            var keepDims = layer.GetInt(4, 0) != 0;
            var op = layer.GetInt(0, 0);
            var coeff = layer.GetFloat(2, 1f);
            var axes = layer.GetInts(-23303, null);
            if (!TryResolveScalar2DReduction(srcShape, reduceAll, keepDims, axes, out var reduceAlongWidth, out var outShape))
                return false;
            if (!CanUseScalarTextureReductionOp(op))
                return false;

            var outRt = owner.RentTempArray(context.commandBuffer, Mathf.Max(1, outShape.w), Mathf.Max(1, outShape.h), 1, RenderTextureFormat.ARGBHalf);
            ExecuteScalar2DReduction(owner.Ops, context.commandBuffer, srcTex.texture, srcShape, reduceAlongWidth, op, coeff, outRt);
            var storageShape = new NcnnRepro.BufferShape(3, outRt.width, outRt.height, 1, 1);
            context.blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
            {
                texture = outRt,
                width = outRt.width,
                height = outRt.height,
                packs = 1,
                refs = 1,
                owned = true,
                hasLogicalShape = true,
                logicalShape = outShape,
                hasStorageShape = true,
                storageShape = storageShape
            };
            context.shapes[layer.topNames[0]] = outShape;
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
            return true;
        }

        private static bool CanUseScalarTextureReductionOp(int op)
        {
            return op == 0
                || op == 1
                || op == 2
                || op == 3
                || op == 4
                || op == 5
                || op == 6
                || op == 7
                || op == 8
                || op == 9
                || op == 10;
        }

        private static bool CanUseScalar2DTexturePath(NcnnRepro.TensorRef srcTex, NcnnRepro.BufferShape srcShape)
        {
            return srcTex != null
                && srcTex.texture != null
                && srcShape.dims == 2
                && srcShape.w > 0
                && srcShape.h > 0
                && srcTex.width == srcShape.w
                && srcTex.height == srcShape.h
                && srcTex.packs == 1;
        }

        private static bool CanUseScalar2DTexturePath(NcnnRepro.CmdTensorRef srcTex, NcnnRepro.BufferShape srcShape)
        {
            return srcTex != null
                && srcTex.texture != null
                && srcShape.dims == 2
                && srcShape.w > 0
                && srcShape.h > 0
                && srcTex.width == srcShape.w
                && srcTex.height == srcShape.h
                && srcTex.packs == 1;
        }

        private static bool TryResolveScalar2DReduction(
            NcnnRepro.BufferShape srcShape,
            bool reduceAll,
            bool keepDims,
            int[] axes,
            out bool reduceAlongWidth,
            out NcnnRepro.BufferShape outShape)
        {
            reduceAlongWidth = false;
            outShape = default;
            if (srcShape.dims != 2 || srcShape.w <= 0 || srcShape.h <= 0)
                return false;

            if (reduceAll)
            {
                outShape = keepDims
                    ? new NcnnRepro.BufferShape(2, 1, 1, 1, 1)
                    : new NcnnRepro.BufferShape(1, 1, 1, 1, 1);
                reduceAlongWidth = true;
                return true;
            }

            if (axes == null || axes.Length != 1)
                return false;

            var axis = axes[0];
            if (axis < 0)
                axis += srcShape.dims;
            if (axis == 1)
            {
                reduceAlongWidth = true;
                outShape = keepDims
                    ? new NcnnRepro.BufferShape(2, 1, srcShape.h, 1, 1)
                    : new NcnnRepro.BufferShape(1, srcShape.h, 1, 1, 1);
                return true;
            }

            if (axis == 0)
            {
                reduceAlongWidth = false;
                outShape = keepDims
                    ? new NcnnRepro.BufferShape(2, srcShape.w, 1, 1, 1)
                    : new NcnnRepro.BufferShape(1, srcShape.w, 1, 1, 1);
                return true;
            }

            return false;
        }

        private static void ExecuteScalar2DReduction(
            NcnnOps ops,
            RenderTexture input,
            NcnnRepro.BufferShape srcShape,
            bool reduceAlongWidth,
            int op,
            float coeff,
            RenderTexture output)
        {
            var axis = reduceAlongWidth ? 1 : 0;
            ops.ReductionScalar2D(input, srcShape.w, srcShape.h, axis, op, coeff, output);
        }

        private static void ExecuteScalar2DReduction(
            NcnnOps ops,
            CommandBuffer cmd,
            ComputeTexture input,
            NcnnRepro.BufferShape srcShape,
            bool reduceAlongWidth,
            int op,
            float coeff,
            ComputeTexture output)
        {
            var axis = reduceAlongWidth ? 1 : 0;
            ops.ReductionScalar2D(cmd, input, srcShape.w, srcShape.h, axis, op, coeff, output);
        }

        private static bool TryResolveSpatialReductionAxes(
            NcnnRepro.BufferShape srcShape,
            bool reduceAll,
            int[] axes,
            out bool reduceSpatialOnly,
            out bool reduceSpatialAndChannels)
        {
            reduceSpatialOnly = false;
            reduceSpatialAndChannels = false;
            if (srcShape.dims != 3)
                return false;

            if (reduceAll)
            {
                reduceSpatialAndChannels = true;
                return true;
            }

            if (axes == null || axes.Length == 0)
                return false;

            var normalized = new HashSet<int>();
            for (var i = 0; i < axes.Length; i++)
            {
                var axis = axes[i];
                if (axis < 0)
                    axis += srcShape.dims;
                if (axis < 0 || axis >= srcShape.dims)
                    return false;
                normalized.Add(axis);
            }

            reduceSpatialOnly = normalized.SetEquals(new[] { 1, 2 });
            reduceSpatialAndChannels = normalized.SetEquals(new[] { 0, 1, 2 });
            return reduceSpatialOnly || reduceSpatialAndChannels;
        }
    }
}
