using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnInnerProductLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnInnerProductLayerRepro() : base(NcnnLayerTypes.InnerProduct, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (TryExecuteRenderTexturePath(owner, layer, context))
                return;

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var ip = new NcnnRepro.InnerProductPack();
                                        ip.outFeatures = layer.GetInt(0, 0);
                                        ip.biasTerm = layer.GetInt(1, 0);
                                        ip.weightSize = layer.GetInt(2, 0);
                                        ip.inFeatures = ip.outFeatures > 0 ? ip.weightSize / ip.outFeatures : 0;

                                        phaseSw.Restart();
                                        var w = NcnnRepro.ReadPackedOrRawWeightArray(br, ip.weightSize, layer.name);
                                        var b = ip.biasTerm != 0 ? br.ReadFloat32Array(ip.outFeatures) : new float[ip.outFeatures];
                                        phaseSw.Stop();
                                        readMs += phaseSw.ElapsedMilliseconds;

                                        phaseSw.Restart();
                                        ip.w = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                                        ip.b = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                                        ip.w.SetData(w);
                                        ip.b.SetData(b);
                                        phaseSw.Stop();
                                        uploadMs += phaseSw.ElapsedMilliseconds;

                                        owner._innerProduct[layer.name] = ip;
                                        return new NcnnRepro.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
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
                                                if (!owner._innerProduct.TryGetValue(layer.name, out var ip))
                                                    throw new InvalidOperationException("InnerProduct not found: " + layer.name);

                                                using var srcTensor = owner.GetReadableTensorInput(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                if (srcTensor == null || srcTensor.buffer == null)
                                                    throw new InvalidOperationException("InnerProduct source not found: " + layer.bottomNames[0]);

                                                var rows = srcTensor.dims == 2 && srcTensor.w == ip.inFeatures ? srcTensor.h : 1;
                                                var outTensor = rows > 1
                                                    ? owner.RentTempTensorBuffer(2, ip.outFeatures, rows)
                                                    : owner.RentTempTensorBuffer(1, ip.outFeatures);
                                                if (rows > 1)
                                                    owner.Ops.InnerProduct2D(srcTensor.buffer, rows, ip.inFeatures, ip.w, ip.b, ip.outFeatures, outTensor.buffer);
                                                else
                                                    owner.Ops.InnerProduct(srcTensor.buffer, ip.inFeatures, ip.w, ip.b, ip.outFeatures, outTensor.buffer);

                                                owner.PublishTensorBufferOutput(
                                                    layer.topNames[0],
                                                    outTensor,
                                                    preferTexture: true,
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

            if (!owner._innerProduct.TryGetValue(layer.name, out var ip))
                throw new InvalidOperationException("InnerProduct not found: " + layer.name);

            var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var rows = srcShape.dims == 2 && srcShape.w == ip.inFeatures ? srcShape.h : 1;
            var outShape = rows > 1
                ? new NcnnRepro.BufferShape(2, Mathf.Max(1, ip.outFeatures), rows, 1, 1)
                : new NcnnRepro.BufferShape(1, Mathf.Max(1, ip.outFeatures), 1, 1, 1);
            owner.PublishCmdPlaceholder(cmd, layer.topNames[0], outShape, blobs, shapes);
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool TryExecuteCommandBufferTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            if (owner == null || layer == null || context == null)
                return false;
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;
            if (!owner._innerProduct.TryGetValue(layer.name, out var ip))
                return false;
            if (ip.w == null || ip.b == null)
                return false;

            var srcTex = NcnnRepro.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var srcShape = NcnnRepro.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            if (srcTex == null || srcTex.texture == null)
                return false;
            if (srcShape.w != ip.inFeatures || srcTex.width != ip.inFeatures || srcTex.packs != 1)
                return false;

            var rows = 0;
            var outLogicalShape = default(NcnnRepro.BufferShape);
            if (srcShape.dims == 1)
            {
                if (srcTex.height != 1)
                    return false;
                rows = 1;
                outLogicalShape = new NcnnRepro.BufferShape(1, Mathf.Max(1, ip.outFeatures), 1, 1, 1);
            }
            else if (srcShape.dims == 2)
            {
                if (srcTex.height != srcShape.h || srcShape.h <= 0)
                    return false;
                rows = srcShape.h;
                outLogicalShape = new NcnnRepro.BufferShape(2, Mathf.Max(1, ip.outFeatures), rows, 1, 1);
            }
            else
                return false;

            var useStrictLinearMat = NcnnRepro.IsStrictLinearMatTexture(srcTex);
            var outStorageShape = useStrictLinearMat
                ? NcnnRepro.ResolveLinearMatStorageShape(outLogicalShape)
                : new NcnnRepro.BufferShape(3, Mathf.Max(1, outLogicalShape.w), Mathf.Max(1, outLogicalShape.h), 1, 1);
            var outRt = useStrictLinearMat
                ? owner.RentTempMat(context.commandBuffer, outStorageShape.w, outStorageShape.h, NcnnRepro.ResolveLinearMatTextureFormat())
                : owner.RentTempArray(context.commandBuffer, outStorageShape.w, outStorageShape.h, 1, RenderTextureFormat.ARGBHalf);
            if (useStrictLinearMat)
            {
                owner.Ops.Gemm2DLinearTextureA(
                    context.commandBuffer,
                    srcTex.texture,
                    ip.w,
                    ip.b,
                    rows,
                    ip.outFeatures,
                    ip.inFeatures,
                    transB: true,
                    alpha: 1f,
                    beta: 1f,
                    useC: true,
                    broadcastTypeC: 4,
                    output: outRt);
            }
            else
            {
                owner.Ops.Gemm2DTextureA(
                    context.commandBuffer,
                    srcTex.texture,
                    ip.w,
                    ip.b,
                    rows,
                    ip.outFeatures,
                    ip.inFeatures,
                    transB: true,
                    alpha: 1f,
                    beta: 1f,
                    useC: true,
                    broadcastTypeC: 4,
                    output: outRt);
            }

            context.blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
            {
                texture = outRt,
                width = outStorageShape.w,
                height = outStorageShape.h,
                packs = 1,
                refs = 1,
                owned = true,
                hasLogicalShape = true,
                logicalShape = outLogicalShape,
                hasStorageShape = true,
                storageShape = outStorageShape
            };
            context.shapes[layer.topNames[0]] = outLogicalShape;
            owner.DebugLog?.Invoke(
                "[CmdTexture][InnerProduct]"
                + " | layer=" + layer.name
                + " | strictLinear=" + (useStrictLinearMat ? "1" : "0")
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | out=d" + outLogicalShape.dims + ":" + outLogicalShape.w + "x" + outLogicalShape.h + "x" + outLogicalShape.d + "x" + outLogicalShape.c
                + " | outFormat=" + outRt.format);
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
            return true;
        }

        private static bool TryExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (owner == null || layer == null || context == null)
                return false;
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;
            if (!owner._innerProduct.TryGetValue(layer.name, out var ip))
                return false;
            if (ip.w == null || ip.b == null)
                return false;
            if (!NcnnRepro.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var srcTex, out var srcShape))
                return false;
            if (srcTex == null || srcTex.texture == null)
                return false;
            if (srcShape.w != ip.inFeatures || srcTex.width != ip.inFeatures || srcTex.packs != 1)
                return false;

            var rows = 0;
            var logicalShape = default(NcnnRepro.BufferShape);
            if (srcShape.dims == 1)
            {
                if (srcTex.height != 1)
                    return false;
                rows = 1;
                logicalShape = new NcnnRepro.BufferShape(1, Mathf.Max(1, ip.outFeatures), 1, 1, 1);
            }
            else if (srcShape.dims == 2)
            {
                if (srcTex.height != srcShape.h || srcShape.h <= 0)
                    return false;
                rows = srcShape.h;
                logicalShape = new NcnnRepro.BufferShape(2, Mathf.Max(1, ip.outFeatures), rows, 1, 1);
            }
            else
                return false;

            var useStrictLinearMat = NcnnRepro.IsStrictLinearMatTexture(srcTex);
            var storageShape = useStrictLinearMat
                ? NcnnRepro.ResolveLinearMatStorageShape(logicalShape)
                : new NcnnRepro.BufferShape(3, Mathf.Max(1, logicalShape.w), Mathf.Max(1, logicalShape.h), 1, 1);
            var outRt = useStrictLinearMat
                ? owner.RentTempMat(storageShape.w, storageShape.h, NcnnRepro.ResolveLinearMatTextureFormat())
                : owner.RentTempArray(storageShape.w, storageShape.h, 1, RenderTextureFormat.ARGBHalf);
            if (useStrictLinearMat)
            {
                owner.Ops.Gemm2DLinearTextureA(
                    srcTex.texture,
                    ip.w,
                    ip.b,
                    rows,
                    ip.outFeatures,
                    ip.inFeatures,
                    transB: true,
                    alpha: 1f,
                    beta: 1f,
                    useC: true,
                    broadcastTypeC: 4,
                    output: outRt);
            }
            else
            {
                owner.Ops.Gemm2DTextureA(
                    srcTex.texture,
                    ip.w,
                    ip.b,
                    rows,
                    ip.outFeatures,
                    ip.inFeatures,
                    transB: true,
                    alpha: 1f,
                    beta: 1f,
                    useC: true,
                    broadcastTypeC: 4,
                    output: outRt);
            }

            if (!useStrictLinearMat && owner.ShouldCompareTextureLayer(layer.name))
            {
                owner.CompareTextureInnerProductPath(
                    layer.name,
                    layer.bottomNames[0],
                    ip,
                    outRt,
                    context.textureBlobs,
                    context.bufferBlobs,
                    context.textureShapes,
                    context.bufferViews,
                    context.tempOwned);
            }

            owner.DebugLog?.Invoke(
                "[Texture][InnerProduct]"
                + " | layer=" + layer.name
                + " | strictLinear=" + (useStrictLinearMat ? "1" : "0")
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | out=d" + logicalShape.dims + ":" + logicalShape.w + "x" + logicalShape.h + "x" + logicalShape.d + "x" + logicalShape.c
                + " | outFormat=" + outRt.format);
            NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], outRt, logicalShape, storageShape);
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
    }
}
