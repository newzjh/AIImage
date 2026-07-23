using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisReductionLayer : AexisBaseLayer
    {
        public AexisReductionLayer() : base(AexisLayerTypes.Reduction, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (TryExecuteRenderTexturePath(owner, layer, context))
                return;

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
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
                                                            ? new AexisTensorBuffer(outBuf3, 3, 1, 1, 1, srcTensor.c, true, owner.ReturnTempBuffer)
                                                            : new AexisTensorBuffer(outBuf3, 1, srcTensor.c, 1, 1, 1, true, owner.ReturnTempBuffer);
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
                                                        ? new AexisTensorBuffer(outBufAll, 2, 1, 1, 1, 1, true, owner.ReturnTempBuffer)
                                                        : new AexisTensorBuffer(outBufAll, 1, 1, 1, 1, 1, true, owner.ReturnTempBuffer);
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
                                                        ? new AexisTensorBuffer(outBuf, 2, 1, srcTensor.h, 1, 1, true, owner.ReturnTempBuffer)
                                                        : new AexisTensorBuffer(outBuf, 1, srcTensor.h, 1, 1, 1, true, owner.ReturnTempBuffer))
                                                    : (keepDims
                                                        ? new AexisTensorBuffer(outBuf, 2, srcTensor.w, 1, 1, 1, true, owner.ReturnTempBuffer)
                                                        : new AexisTensorBuffer(outBuf, 1, srcTensor.w, 1, 1, 1, true, owner.ReturnTempBuffer));
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

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (TryExecuteRenderTexturePath(owner, layer, context))
                return;

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (TryExecuteCommandBufferTexturePath(owner, layer, context))
                return;
            var srcShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            throw new NotSupportedException("CommandBuffer Reduction supports texture-native scalar rank-2 reductions and Pack4 spatial SUM/MEAN only"
                + " | layer=" + layer.name
                + " | input=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | rejectedFallback=placeholder-or-buffer-materialization");
        }

        private static bool TryExecuteCommandBufferTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (owner == null || layer == null || context == null)
                return false;
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;

            var srcTex = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var srcShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            if (TryExecuteCommandBufferScalar2DTexturePath(owner, layer, srcTex, srcShape, context))
                return true;
            if (srcTex == null
                || srcTex.texture == null
                || srcShape.dims != 3
                || srcShape.d != 1
                || !AexisGraphSession.MatchesPack4TextureStorage(srcTex, srcShape))
            {
                return false;
            }

            var reduceAll = layer.GetInt(1, 1) != 0;
            var keepDims = layer.GetInt(4, 0) != 0;
            var op = layer.GetInt(0, 0);
            var coeff = layer.GetFloat(2, 1f);
            var axes = layer.GetInts(-23303, null);
            var cmd = context.commandBuffer;
            var outTop = layer.topNames[0];
            if (TryResolveWidthReductionAxes(srcShape, reduceAll, keepDims, axes, out var reduceWidthOnly)
                && reduceWidthOnly
                && CanUseScalarTextureReductionOp(op))
            {
                var outShape = new AexisGraphSession.BufferShape(3, 1, srcShape.h, 1, srcShape.c);
                var outStorage = new AexisGraphSession.BufferShape(3, 1, srcShape.h, 1, srcShape.c);
                var outRt = owner.RentTempArray(cmd, outStorage.w, outStorage.h, srcTex.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.ReductionPack4Width(cmd, srcTex.texture, srcShape.w, srcShape.h, op, coeff, outRt);
                context.blobs[outTop] = AexisGraphSession.CreateCmdTensorRef(outRt, outShape, outStorage, owned: true, blobName: outTop);
                context.shapes[outTop] = outShape;
                owner.ConsumeCmd(cmd, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
                return true;
            }

            if (TryResolveChannelReductionAxes(srcShape, reduceAll, axes, out var reduceChannelsOnly)
                && reduceChannelsOnly
                && CanUseScalarTextureReductionOp(op))
            {
                var outShape = keepDims
                    ? new AexisGraphSession.BufferShape(3, srcShape.w, srcShape.h, 1, 1)
                    : new AexisGraphSession.BufferShape(2, srcShape.w, srcShape.h, 1, 1);
                var outStorage = new AexisGraphSession.BufferShape(3, srcShape.w, srcShape.h, 1, 1);
                var outRt = owner.RentTempArray(cmd, outStorage.w, outStorage.h, 1, RenderTextureFormat.ARGBHalf);
                owner.Ops.ReductionPack4Channel(cmd, srcTex.texture, srcShape.w, srcShape.h, srcShape.c, op, coeff, outRt);
                context.blobs[outTop] = AexisGraphSession.CreateCmdTensorRef(outRt, outShape, outStorage, owned: true, blobName: outTop);
                context.shapes[outTop] = outShape;
                owner.ConsumeCmd(cmd, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
                return true;
            }

            if (!TryResolveSpatialReductionAxes(srcShape, reduceAll, axes, out var reduceSpatialOnly, out var reduceSpatialAndChannels))
                return false;
            if (op != 3 && op != 0)
                return false;
            if (!reduceSpatialOnly && !reduceSpatialAndChannels)
                return false;

            var area = Mathf.Max(1, srcShape.w * srcShape.h);

            ComputeTexture pooled = null;
            if (reduceSpatialOnly || reduceSpatialAndChannels)
            {
                pooled = owner.RentTempArray(cmd, 1, 1, srcTex.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.PoolingPack4(cmd, srcTex.texture, srcTex.packs, srcShape.w, srcShape.h, 1, 1, 0, 0, 1, pooled);
                if (op == 0 && Mathf.Abs(coeff - area) > 1e-6f)
                {
                    var scaled = owner.RentTempArray(cmd, 1, 1, srcTex.packs, RenderTextureFormat.ARGBHalf);
                    owner.Ops.PointwisePack4(cmd, pooled, srcTex.packs, AexisOps.PointwiseType.ScaleScalar, coeff * area, 0f, scaled);
                    owner.ReturnTempArray(cmd, pooled);
                    pooled = scaled;
                }
                else if (op == 3 && Mathf.Abs(coeff - 1f) > 1e-6f)
                {
                    var scaled = owner.RentTempArray(cmd, 1, 1, srcTex.packs, RenderTextureFormat.ARGBHalf);
                    owner.Ops.PointwisePack4(cmd, pooled, srcTex.packs, AexisOps.PointwiseType.ScaleScalar, coeff, 0f, scaled);
                    owner.ReturnTempArray(cmd, pooled);
                    pooled = scaled;
                }
            }

            if (reduceSpatialOnly && keepDims)
            {
                context.blobs[outTop] = new AexisGraphSession.CmdTensorRef
                {
                    texture = pooled,
                    width = 1,
                    height = 1,
                    packs = srcTex.packs,
                    refs = 1,
                    owned = true,
                    hasLogicalShape = true,
                    logicalShape = new AexisGraphSession.BufferShape(3, 1, 1, 1, srcShape.c),
                    hasStorageShape = true,
                    storageShape = new AexisGraphSession.BufferShape(3, 1, 1, 1, srcShape.c)
                };
                context.shapes[outTop] = new AexisGraphSession.BufferShape(3, 1, 1, 1, srcShape.c);
                owner.ConsumeCmd(cmd, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
                return true;
            }

            if (reduceSpatialOnly && !keepDims)
            {
                var logicalShape = new AexisGraphSession.BufferShape(1, srcShape.c, 1, 1, 1);
                var storageShape = AexisGraphSession.ResolveLinearMatStorageShape(logicalShape);
                var outRt = owner.RentTempMat(cmd, storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.ReshapePack4ToLinearMat(cmd, pooled, 1, 1, 1, srcShape.c, 3, outRt);
                owner.ReturnTempArray(cmd, pooled);
                context.blobs[outTop] = new AexisGraphSession.CmdTensorRef
                {
                    texture = outRt,
                    width = outRt.width,
                    height = outRt.height,
                    packs = 1,
                    refs = 1,
                    owned = true,
                    hasLogicalShape = true,
                    logicalShape = logicalShape,
                    hasStorageShape = true,
                    storageShape = storageShape
                };
                context.shapes[outTop] = logicalShape;
                owner.ConsumeCmd(cmd, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
                return true;
            }

            if (reduceSpatialAndChannels && !keepDims)
            {
                var logicalShape = new AexisGraphSession.BufferShape(1, srcShape.c, 1, 1, 1);
                var storageShape = AexisGraphSession.ResolveLinearMatStorageShape(logicalShape);
                var outRt = owner.RentTempMat(cmd, storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.ReshapePack4ToLinearMat(cmd, pooled, 1, 1, 1, srcShape.c, 3, outRt);
                owner.ReturnTempArray(cmd, pooled);
                context.blobs[outTop] = new AexisGraphSession.CmdTensorRef
                {
                    texture = outRt,
                    width = outRt.width,
                    height = outRt.height,
                    packs = 1,
                    refs = 1,
                    owned = true,
                    hasLogicalShape = true,
                    logicalShape = logicalShape,
                    hasStorageShape = true,
                    storageShape = storageShape
                };
                context.shapes[outTop] = logicalShape;
                owner.ConsumeCmd(cmd, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
                return true;
            }

            if (pooled != null)
                owner.ReturnTempArray(cmd, pooled);
            return false;
        }

        private static bool TryExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
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
            var outTop = layer.topNames[0];

            if (srcShape.dims != 3 || srcShape.d != 1 || !AexisGraphSession.MatchesPack4TextureStorage(srcTex, srcShape))
                return false;
            if (TryResolveWidthReductionAxes(srcShape, reduceAll, keepDims, axes, out var reduceWidthOnly)
                && reduceWidthOnly
                && CanUseScalarTextureReductionOp(op))
            {
                var outShape = new AexisGraphSession.BufferShape(3, 1, srcShape.h, 1, srcShape.c);
                var outStorage = new AexisGraphSession.BufferShape(3, 1, srcShape.h, 1, srcShape.c);
                var outRt = owner.RentTempArray(outStorage.w, outStorage.h, srcTex.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.ReductionPack4Width(srcTex.texture, srcShape.w, srcShape.h, op, coeff, outRt);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, outTop, outRt, outShape, outStorage);
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

            if (TryResolveChannelReductionAxes(srcShape, reduceAll, axes, out var reduceChannelsOnly)
                && reduceChannelsOnly
                && CanUseScalarTextureReductionOp(op))
            {
                var outShape = keepDims
                    ? new AexisGraphSession.BufferShape(3, srcShape.w, srcShape.h, 1, 1)
                    : new AexisGraphSession.BufferShape(2, srcShape.w, srcShape.h, 1, 1);
                var outStorage = new AexisGraphSession.BufferShape(3, srcShape.w, srcShape.h, 1, 1);
                var outRt = owner.RentTempArray(outStorage.w, outStorage.h, 1, RenderTextureFormat.ARGBHalf);
                owner.Ops.ReductionPack4Channel(srcTex.texture, srcShape.w, srcShape.h, srcShape.c, op, coeff, outRt);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, outTop, outRt, outShape, outStorage);
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

            if (!TryResolveSpatialReductionAxes(srcShape, reduceAll, axes, out var reduceSpatialOnly, out var reduceSpatialAndChannels))
                return false;
            if (op != 3 && op != 0)
                return false;
            if (!reduceSpatialOnly && !reduceSpatialAndChannels)
                return false;

            var area = Mathf.Max(1, srcShape.w * srcShape.h);
            if (reduceSpatialOnly && keepDims)
            {
                var outRt = owner.RentTempArray(1, 1, srcTex.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.PoolingPack4(srcTex.texture, srcTex.packs, srcShape.w, srcShape.h, 1, 1, 0, 0, 1, outRt);
                if (op == 0 && Mathf.Abs(coeff - area) > 1e-6f)
                {
                    var scaledRt = owner.RentTempArray(1, 1, srcTex.packs, RenderTextureFormat.ARGBHalf);
                    owner.Ops.PointwisePack4(outRt, srcTex.packs, AexisOps.PointwiseType.ScaleScalar, coeff * area, 0f, scaledRt);
                    owner.ReturnTempArray(outRt);
                    outRt = scaledRt;
                }
                else if (op == 3 && Mathf.Abs(coeff - 1f) > 1e-6f)
                {
                    var scaledRt = owner.RentTempArray(1, 1, srcTex.packs, RenderTextureFormat.ARGBHalf);
                    owner.Ops.PointwisePack4(outRt, srcTex.packs, AexisOps.PointwiseType.ScaleScalar, coeff, 0f, scaledRt);
                    owner.ReturnTempArray(outRt);
                    outRt = scaledRt;
                }

                var outShape = new AexisGraphSession.BufferShape(3, 1, 1, 1, srcShape.c);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, outTop, outRt, outShape, outShape);
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

            if (reduceSpatialOnly && !keepDims)
            {
                var pooledRt = owner.RentTempArray(1, 1, srcTex.packs, RenderTextureFormat.ARGBHalf);
                owner.Ops.PoolingPack4(srcTex.texture, srcTex.packs, srcShape.w, srcShape.h, 1, 1, 0, 0, 1, pooledRt);
                if (op == 0 && Mathf.Abs(coeff - area) > 1e-6f)
                {
                    var scaledRt = owner.RentTempArray(1, 1, srcTex.packs, RenderTextureFormat.ARGBHalf);
                    owner.Ops.PointwisePack4(pooledRt, srcTex.packs, AexisOps.PointwiseType.ScaleScalar, coeff * area, 0f, scaledRt);
                    owner.ReturnTempArray(pooledRt);
                    pooledRt = scaledRt;
                }
                else if (op == 3 && Mathf.Abs(coeff - 1f) > 1e-6f)
                {
                    var scaledRt = owner.RentTempArray(1, 1, srcTex.packs, RenderTextureFormat.ARGBHalf);
                    owner.Ops.PointwisePack4(pooledRt, srcTex.packs, AexisOps.PointwiseType.ScaleScalar, coeff, 0f, scaledRt);
                    owner.ReturnTempArray(pooledRt);
                    pooledRt = scaledRt;
                }

                var logicalShape = new AexisGraphSession.BufferShape(1, srcShape.c, 1, 1, 1);
                var storageShape = AexisGraphSession.ResolveLinearMatStorageShape(logicalShape);
                var outRt = owner.RentTempMat(storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.ReshapePack4ToLinearMat(pooledRt, 1, 1, 1, srcShape.c, 3, outRt);
                owner.ReturnTempArray(pooledRt);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, outTop, outRt, logicalShape, storageShape);
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
                    owner.Ops.PointwisePack4(pooledRt, srcTex.packs, AexisOps.PointwiseType.ScaleScalar, coeff * area, 0f, scaledRt);
                    owner.ReturnTempArray(pooledRt);
                    pooledRt = scaledRt;
                }
                else if (op == 3 && Mathf.Abs(coeff - 1f) > 1e-6f)
                {
                    var scaledRt = owner.RentTempArray(1, 1, srcTex.packs, RenderTextureFormat.ARGBHalf);
                    owner.Ops.PointwisePack4(pooledRt, srcTex.packs, AexisOps.PointwiseType.ScaleScalar, coeff, 0f, scaledRt);
                    owner.ReturnTempArray(pooledRt);
                    pooledRt = scaledRt;
                }

                var logicalShape = new AexisGraphSession.BufferShape(1, srcShape.c, 1, 1, 1);
                var storageShape = AexisGraphSession.ResolveLinearMatStorageShape(logicalShape);
                var outRt = owner.RentTempMat(storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.ReshapePack4ToLinearMat(pooledRt, 1, 1, 1, srcShape.c, 3, outRt);
                owner.ReturnTempArray(pooledRt);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, outTop, outRt, logicalShape, storageShape);
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
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisLayerBufferContext context)
        {
            if (!AexisGraphSession.TryGetExistingTexture(
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
            var reductionAxis = reduceAll ? 2 : reduceAlongWidth ? 1 : 0;

            RenderTexture outRt;
            AexisGraphSession.BufferShape storageShape;
            if (AexisGraphSession.IsStrictLinearMatTexture(srcTex))
            {
                storageShape = AexisGraphSession.ResolveLinearMatStorageShape(outShape);
                outRt = owner.RentTempMat(storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                ExecuteLinearMatReduction(owner.Ops, srcTex.texture, srcShape, reductionAxis, op, coeff, outRt);
            }
            else
            {
                outRt = owner.RentTempArray(Mathf.Max(1, outShape.w), Mathf.Max(1, outShape.h), 1, RenderTextureFormat.ARGBHalf);
                ExecuteScalar2DReduction(owner.Ops, srcTex.texture, srcShape, reductionAxis, op, coeff, outRt);
                storageShape = new AexisGraphSession.BufferShape(3, outRt.width, outRt.height, 1, 1);
            }
            AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], outRt, outShape, storageShape);
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
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.CmdTensorRef srcTex,
            AexisGraphSession.BufferShape srcShape,
            AexisLayerCommandBufferContext context)
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
            var reductionAxis = reduceAll ? 2 : reduceAlongWidth ? 1 : 0;

            ComputeTexture outRt;
            AexisGraphSession.BufferShape storageShape;
            if (AexisGraphSession.IsStrictLinearMatTexture(srcTex))
            {
                storageShape = AexisGraphSession.ResolveLinearMatStorageShape(outShape);
                outRt = owner.RentTempMat(context.commandBuffer, storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                ExecuteLinearMatReduction(owner.Ops, context.commandBuffer, srcTex.texture, srcShape, reductionAxis, op, coeff, outRt);
            }
            else
            {
                outRt = owner.RentTempArray(context.commandBuffer, Mathf.Max(1, outShape.w), Mathf.Max(1, outShape.h), 1, RenderTextureFormat.ARGBHalf);
                ExecuteScalar2DReduction(owner.Ops, context.commandBuffer, srcTex.texture, srcShape, reductionAxis, op, coeff, outRt);
                storageShape = new AexisGraphSession.BufferShape(3, outRt.width, outRt.height, 1, 1);
            }
            context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(
                outRt,
                outShape,
                storageShape,
                owned: true,
                blobName: layer.topNames[0]);
            context.shapes[layer.topNames[0]] = outShape;
            owner.DebugLog?.Invoke(
                "[CmdTexture][Reduction]"
                + " | layer=" + layer.name
                + " | strictLinear=" + (AexisGraphSession.IsStrictLinearMatTexture(srcTex) ? "1" : "0")
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c
                + " | outFormat=" + outRt.format
                + " | reduceAlongWidth=" + (reduceAlongWidth ? "1" : "0"));
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

        private static bool CanUseScalar2DTexturePath(AexisGraphSession.TensorRef srcTex, AexisGraphSession.BufferShape srcShape)
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

        private static bool CanUseScalar2DTexturePath(AexisGraphSession.CmdTensorRef srcTex, AexisGraphSession.BufferShape srcShape)
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
            AexisGraphSession.BufferShape srcShape,
            bool reduceAll,
            bool keepDims,
            int[] axes,
            out bool reduceAlongWidth,
            out AexisGraphSession.BufferShape outShape)
        {
            reduceAlongWidth = false;
            outShape = default;
            if (srcShape.dims != 2 || srcShape.w <= 0 || srcShape.h <= 0)
                return false;

            if (reduceAll)
            {
                outShape = keepDims
                    ? new AexisGraphSession.BufferShape(2, 1, 1, 1, 1)
                    : new AexisGraphSession.BufferShape(1, 1, 1, 1, 1);
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
                    ? new AexisGraphSession.BufferShape(2, 1, srcShape.h, 1, 1)
                    : new AexisGraphSession.BufferShape(1, srcShape.h, 1, 1, 1);
                return true;
            }

            if (axis == 0)
            {
                reduceAlongWidth = false;
                outShape = keepDims
                    ? new AexisGraphSession.BufferShape(2, srcShape.w, 1, 1, 1)
                    : new AexisGraphSession.BufferShape(1, srcShape.w, 1, 1, 1);
                return true;
            }

            return false;
        }

        private static void ExecuteScalar2DReduction(
            AexisOps ops,
            RenderTexture input,
            AexisGraphSession.BufferShape srcShape,
            int axis,
            int op,
            float coeff,
            RenderTexture output)
        {
            ops.ReductionScalar2D(input, srcShape.w, srcShape.h, axis, op, coeff, output);
        }

        private static void ExecuteScalar2DReduction(
            AexisOps ops,
            CommandBuffer cmd,
            ComputeTexture input,
            AexisGraphSession.BufferShape srcShape,
            int axis,
            int op,
            float coeff,
            ComputeTexture output)
        {
            ops.ReductionScalar2D(cmd, input, srcShape.w, srcShape.h, axis, op, coeff, output);
        }

        private static void ExecuteLinearMatReduction(
            AexisOps ops,
            RenderTexture input,
            AexisGraphSession.BufferShape srcShape,
            int axis,
            int op,
            float coeff,
            RenderTexture output)
        {
            ops.ReductionLinearMat2D(input, srcShape.w, srcShape.h, axis, op, coeff, output);
        }

        private static void ExecuteLinearMatReduction(
            AexisOps ops,
            CommandBuffer cmd,
            ComputeTexture input,
            AexisGraphSession.BufferShape srcShape,
            int axis,
            int op,
            float coeff,
            ComputeTexture output)
        {
            ops.ReductionLinearMat2D(cmd, input, srcShape.w, srcShape.h, axis, op, coeff, output);
        }

        private static bool TryResolveSpatialReductionAxes(
            AexisGraphSession.BufferShape srcShape,
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

        private static bool TryResolveChannelReductionAxes(
            AexisGraphSession.BufferShape srcShape,
            bool reduceAll,
            int[] axes,
            out bool reduceChannelsOnly)
        {
            reduceChannelsOnly = false;
            if (srcShape.dims != 3)
                return false;
            if (reduceAll)
                return false;
            if (axes == null || axes.Length != 1)
                return false;

            var axis = axes[0];
            if (axis < 0)
                axis += srcShape.dims;
            if (axis != 0)
                return false;
            reduceChannelsOnly = true;
            return true;
        }

        private static bool TryResolveWidthReductionAxes(
            AexisGraphSession.BufferShape srcShape,
            bool reduceAll,
            bool keepDims,
            int[] axes,
            out bool reduceWidthOnly)
        {
            reduceWidthOnly = false;
            if (srcShape.dims != 3)
                return false;
            if (reduceAll || !keepDims)
                return false;
            if (axes == null || axes.Length != 1)
                return false;

            var axis = axes[0];
            if (axis < 0)
                axis += srcShape.dims;
            if (axis != 0)
                return false;

            reduceWidthOnly = true;
            return true;
        }
    }
}
