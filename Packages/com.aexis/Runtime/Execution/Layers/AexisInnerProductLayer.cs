using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisInnerProductLayer : AexisBaseLayer
    {
        public AexisInnerProductLayer() : base(AexisLayerTypes.InnerProduct, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            ConfigureGemmWeightBindings(owner, layer);
            if (TryExecuteRenderTexturePath(owner, layer, context))
                return;

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var ip = new AexisGraphSession.InnerProductPack();
                                        ip.outFeatures = layer.GetInt(0, 0);
                                        ip.biasTerm = layer.GetInt(1, 0);
                                        ip.weightSize = layer.GetInt(2, 0);
                                        ip.inFeatures = ip.outFeatures > 0 ? ip.weightSize / ip.outFeatures : 0;

                                        phaseSw.Restart();
                                        var w = AexisGraphSession.ReadPackedOrRawWeightArray(br, ip.weightSize, layer.name);
                                        var b = ip.biasTerm != 0 ? br.ReadFloat32Array(ip.outFeatures) : new float[ip.outFeatures];
                                        phaseSw.Stop();
                                        readMs += phaseSw.ElapsedMilliseconds;

                                        phaseSw.Restart();
                                        ip.b = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                                        ip.b.SetData(b);
                                        if (owner.UsesInt4WeightOnlyForLayer(layer))
                                        {
                                            var quantized = AexisGraphSession.NewInt4WeightOnlyUpload(
                                                w,
                                                ip.outFeatures,
                                                ip.inFeatures,
                                                outputChannelsAreContiguous: true,
                                                "AexisGraphSession.InnerProductInt4WeightOnly:" + layer.name);
                                            ip.wInt4Packed = quantized.packedWeights;
                                            ip.wInt4Scales = quantized.scales;
                                        }
                                        else if (owner.UsesInt8WeightOnlyForLayer(layer))
                                        {
                                            var quantized = AexisGraphSession.NewInt8WeightOnlyUpload(
                                                w,
                                                ip.outFeatures,
                                                ip.inFeatures,
                                                outputChannelsAreContiguous: true,
                                                "AexisGraphSession.InnerProductInt8WeightOnly:" + layer.name);
                                            ip.wInt8Packed = quantized.packedWeights;
                                            ip.wInt8Scales = quantized.scales;
                                        }
                                        else
                                        {
                                            ip.w = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                                            ip.w.SetData(w);
                                        }
                                        if (owner.UsesFp16WeightsForLayer(layer))
                                            ip.wFp16 = AexisGraphSession.NewFp16Buffer(w, "AexisGraphSession.InnerProductWeightFp16:" + layer.name);
                                        phaseSw.Stop();
                                        uploadMs += phaseSw.ElapsedMilliseconds;

                                        owner._innerProduct[layer.name] = ip;
                                        return new AexisGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
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
                                                    owner.Ops.InnerProduct2D(srcTensor.buffer, rows, ip.inFeatures, ip.TextureWeightBinding, ip.b, ip.outFeatures, outTensor.buffer);
                                                else
                                                    owner.Ops.InnerProduct(srcTensor.buffer, ip.inFeatures, ip.TextureWeightBinding, ip.b, ip.outFeatures, outTensor.buffer);

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

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            ConfigureGemmWeightBindings(owner, layer);
            if (TryExecuteRenderTexturePath(owner, layer, context))
                return;

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            ConfigureGemmWeightBindings(owner, layer);
            if (TryExecuteCommandBufferTexturePath(owner, layer, context))
                return;

            if (!owner._innerProduct.TryGetValue(layer.name, out var ip))
                throw new InvalidOperationException("InnerProduct not found: " + layer.name);

            var srcShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            throw new NotSupportedException("CommandBuffer InnerProduct requires a LinearMat vector/matrix or a verified Pack4 attention output"
                + " | layer=" + layer.name
                + " | input=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | inFeatures=" + ip.inFeatures
                + " | rejectedFallback=placeholder-or-buffer-materialization");
        }

        private static void ConfigureGemmWeightBindings(AexisGraphSession owner, AexisGraphModel.Layer layer)
        {
            var hasPack = owner._innerProduct.TryGetValue(layer.name, out var pack);
            var useInt8WeightOnly = owner.UsesInt8WeightsForLayer(layer);
            var useInt4WeightOnly = owner.UsesInt4WeightsForLayer(layer);
            var useFp16Weights = owner.UsesFp16WeightsForLayer(layer) && !owner.UsesQuantizedWeightsForLayer(layer);
            owner.Ops.SetFp16GemmWeights(useFp16Weights && hasPack ? pack.wFp16 : null);
            owner.Ops.SetInt8GemmWeights(
                useInt8WeightOnly && hasPack ? pack.wInt8Packed : null,
                useInt8WeightOnly && hasPack ? pack.wInt8Scales : null);
            owner.Ops.SetInt4GemmWeights(
                useInt4WeightOnly && hasPack ? pack.wInt4Packed : null,
                useInt4WeightOnly && hasPack ? pack.wInt4Scales : null);
            owner.ConfigureInt8ActivationQuantization(layer);
        }

        private static bool HasDirectSoftmaxConsumer(AexisGraphSession owner, AexisGraphModel.Layer layer)
        {
            if (owner?.Model?.layers == null || layer?.topNames == null)
                return false;

            for (var topIndex = 0; topIndex < layer.topNames.Length; topIndex++)
            {
                var topName = layer.topNames[topIndex];
                if (string.IsNullOrWhiteSpace(topName))
                    continue;

                for (var layerIndex = 0; layerIndex < owner.Model.layers.Count; layerIndex++)
                {
                    var consumer = owner.Model.layers[layerIndex];
                    if (consumer?.type != AexisLayerTypes.Softmax || consumer.bottomNames == null)
                        continue;
                    for (var bottomIndex = 0; bottomIndex < consumer.bottomNames.Length; bottomIndex++)
                    {
                        if (string.Equals(consumer.bottomNames[bottomIndex], topName, StringComparison.Ordinal))
                            return true;
                    }
                }
            }

            return false;
        }

        private static bool TryExecuteCommandBufferTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (owner == null || layer == null || context == null)
                return false;
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;
            if (!owner._innerProduct.TryGetValue(layer.name, out var ip))
                return false;
            if (ip.TextureWeightBinding == null || ip.b == null)
                return false;

            var srcTex = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var srcShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            if (srcTex == null || srcTex.texture == null)
                return false;
            if (!owner.PreserveLegacyFp32Execution
                && !owner.UseLegacyPack4AttentionLayout
                && TryResolveAttentionPack4ToLinearInput(srcTex, srcShape, ip, out var attentionStorageShape, out var attentionRows, out var headDim, out var numHeads))
            {
                var attentionOutShape = attentionRows > 1
                    ? new AexisGraphSession.BufferShape(2, Mathf.Max(1, ip.outFeatures), attentionRows, 1, 1)
                    : new AexisGraphSession.BufferShape(1, Mathf.Max(1, ip.outFeatures), 1, 1, 1);
                var attentionOutStorage = AexisGraphSession.ResolveLinearMatStorageShape(attentionOutShape);
                var attentionOut = owner.RentTempMat(context.commandBuffer, attentionOutStorage.w, attentionOutStorage.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.Gemm2DAttentionPack4ToLinearTextureA(
                    context.commandBuffer,
                    srcTex.texture,
                    ip.TextureWeightBinding,
                    ip.b,
                    attentionRows,
                    ip.outFeatures,
                    ip.inFeatures,
                    transB: true,
                    alpha: 1f,
                    beta: 1f,
                    useC: true,
                    broadcastTypeC: 4,
                    headDim: headDim,
                    numHeads: numHeads,
                    output: attentionOut);
                context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(attentionOut, attentionOutShape, attentionOutStorage, owned: true);
                context.shapes[layer.topNames[0]] = attentionOutShape;
                owner.DebugLog?.Invoke(
                    "[CmdTexture][InnerProductAttentionPack4ToLinear]"
                    + " | layer=" + layer.name
                    + " | headDim=" + headDim
                    + " | heads=" + numHeads
                    + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | storage=d" + attentionStorageShape.dims + ":" + attentionStorageShape.w + "x" + attentionStorageShape.h + "x" + attentionStorageShape.d + "x" + attentionStorageShape.c);
                owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
                return true;
            }
            var srcIsStrictLinear = AexisGraphSession.IsStrictLinearMatTexture(srcTex);
            var srcIsPack4Linear = AexisGraphSession.IsPack4LinearMatTexture(srcTex, srcShape);
            if (srcShape.w != ip.inFeatures
                || (!srcIsPack4Linear && (srcTex.width != ip.inFeatures || srcTex.packs != 1)))
                return false;

            var rows = 0;
            var outLogicalShape = default(AexisGraphSession.BufferShape);
            if (srcShape.dims == 1)
            {
                if (srcTex.height != 1)
                    return false;
                rows = 1;
                outLogicalShape = new AexisGraphSession.BufferShape(1, Mathf.Max(1, ip.outFeatures), 1, 1, 1);
            }
            else if (srcShape.dims == 2)
            {
                if (srcTex.height != srcShape.h || srcShape.h <= 0)
                    return false;
                rows = srcShape.h;
                outLogicalShape = new AexisGraphSession.BufferShape(2, Mathf.Max(1, ip.outFeatures), rows, 1, 1);
            }
            else
                return false;

            // A Pack4-linear input must stay in the Pack4 matrix contract.  The
            // shader reads the four logical input values from each texel; routing
            // it through the scalar kernel would silently reinterpret width=K/4.
            var usePack4LinearMat = (srcIsStrictLinear || srcIsPack4Linear)
                && outLogicalShape.dims == 2
                && (ip.outFeatures & 3) == 0;
            if (srcIsPack4Linear && !usePack4LinearMat)
                return false;
            var useStrictLinearMat = srcIsStrictLinear && !usePack4LinearMat;
            var outStorageShape = usePack4LinearMat
                ? AexisGraphSession.ResolvePack4LinearMatStorageShape(outLogicalShape)
                : useStrictLinearMat
                    ? AexisGraphSession.ResolveLinearMatStorageShape(outLogicalShape)
                    : new AexisGraphSession.BufferShape(3, Mathf.Max(1, outLogicalShape.w), Mathf.Max(1, outLogicalShape.h), 1, 1);
            var outRt = usePack4LinearMat
                ? owner.RentTempArray(context.commandBuffer, outStorageShape.w, outStorageShape.h, 1, owner.ResolveActivationTextureFormat(layer, outLogicalShape.dims))
                : useStrictLinearMat
                    ? owner.RentTempMat(context.commandBuffer, outStorageShape.w, outStorageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat())
                    : owner.RentTempArray(context.commandBuffer, outStorageShape.w, outStorageShape.h, 1, owner.ResolveActivationTextureFormat(layer, outLogicalShape.dims));
            if (usePack4LinearMat)
            {
                owner.Ops.Gemm2DPack4LinearTextureA(
                    context.commandBuffer,
                    srcTex.texture,
                    srcIsPack4Linear,
                    ip.TextureWeightBinding,
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
            else if (useStrictLinearMat)
            {
                owner.Ops.Gemm2DLinearTextureA(
                    context.commandBuffer,
                    srcTex.texture,
                    ip.TextureWeightBinding,
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
                    ip.TextureWeightBinding,
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

            context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(
                outRt,
                outLogicalShape,
                outStorageShape,
                owned: true,
                blobName: layer.topNames[0]);
            context.shapes[layer.topNames[0]] = outLogicalShape;
            owner.DebugLog?.Invoke(
                "[CmdTexture][InnerProduct]"
                + " | layer=" + layer.name
                + " | strictLinear=" + (useStrictLinearMat ? "1" : "0")
                + " | srcPack4Linear=" + (srcIsPack4Linear ? "1" : "0")
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | out=d" + outLogicalShape.dims + ":" + outLogicalShape.w + "x" + outLogicalShape.h + "x" + outLogicalShape.d + "x" + outLogicalShape.c
                + " | outFormat=" + outRt.format);
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
            return true;
        }

        private static bool TryExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (owner == null || layer == null || context == null)
                return false;
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;
            if (!owner._innerProduct.TryGetValue(layer.name, out var ip))
                return false;
            if (ip.TextureWeightBinding == null || ip.b == null)
                return false;
            if (!AexisGraphSession.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var srcTex, out var srcShape))
                return false;
            if (srcTex == null || srcTex.texture == null)
                return false;
            if (!owner.PreserveLegacyFp32Execution
                && !owner.UseLegacyPack4AttentionLayout
                && TryResolveAttentionPack4ToLinearInput(srcTex, srcShape, ip, out var attentionStorageShape, out var attentionRows, out var headDim, out var numHeads))
            {
                var attentionOutShape = attentionRows > 1
                    ? new AexisGraphSession.BufferShape(2, Mathf.Max(1, ip.outFeatures), attentionRows, 1, 1)
                    : new AexisGraphSession.BufferShape(1, Mathf.Max(1, ip.outFeatures), 1, 1, 1);
                var attentionOutStorage = AexisGraphSession.ResolveLinearMatStorageShape(attentionOutShape);
                var attentionOut = owner.RentTempMat(attentionOutStorage.w, attentionOutStorage.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                owner.Ops.Gemm2DAttentionPack4ToLinearTextureA(
                    srcTex.texture,
                    ip.TextureWeightBinding,
                    ip.b,
                    attentionRows,
                    ip.outFeatures,
                    ip.inFeatures,
                    transB: true,
                    alpha: 1f,
                    beta: 1f,
                    useC: true,
                    broadcastTypeC: 4,
                    headDim: headDim,
                    numHeads: numHeads,
                    output: attentionOut);
                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], attentionOut, attentionOutShape, attentionOutStorage);
                owner.DebugLog?.Invoke(
                    "[Texture][InnerProductAttentionPack4ToLinear]"
                    + " | layer=" + layer.name
                    + " | headDim=" + headDim
                    + " | heads=" + numHeads
                    + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | storage=d" + attentionStorageShape.dims + ":" + attentionStorageShape.w + "x" + attentionStorageShape.h + "x" + attentionStorageShape.d + "x" + attentionStorageShape.c);
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
            var srcIsStrictLinear = AexisGraphSession.IsStrictLinearMatTexture(srcTex);
            var srcIsPack4Linear = AexisGraphSession.IsPack4LinearMatTexture(srcTex, srcShape);
            if (srcShape.w != ip.inFeatures
                || (!srcIsPack4Linear && (srcTex.width != ip.inFeatures || srcTex.packs != 1)))
                return false;

            var rows = 0;
            var logicalShape = default(AexisGraphSession.BufferShape);
            if (srcShape.dims == 1)
            {
                if (srcTex.height != 1)
                    return false;
                rows = 1;
                logicalShape = new AexisGraphSession.BufferShape(1, Mathf.Max(1, ip.outFeatures), 1, 1, 1);
            }
            else if (srcShape.dims == 2)
            {
                if (srcTex.height != srcShape.h || srcShape.h <= 0)
                    return false;
                rows = srcShape.h;
                logicalShape = new AexisGraphSession.BufferShape(2, Mathf.Max(1, ip.outFeatures), rows, 1, 1);
            }
            else
                return false;

            // Keep Pack4-linear matrices in their native texture layout.  This
            // is required for both immediate RT and CommandBuffer inference and
            // never materializes an activation into a ComputeBuffer.
            var usePack4LinearMat = (srcIsStrictLinear || srcIsPack4Linear)
                && logicalShape.dims == 2
                && (ip.outFeatures & 3) == 0;
            if (srcIsPack4Linear && !usePack4LinearMat)
                return false;
            var useStrictLinearMat = srcIsStrictLinear && !usePack4LinearMat;
            var storageShape = usePack4LinearMat
                ? AexisGraphSession.ResolvePack4LinearMatStorageShape(logicalShape)
                : useStrictLinearMat
                    ? AexisGraphSession.ResolveLinearMatStorageShape(logicalShape)
                    : new AexisGraphSession.BufferShape(3, Mathf.Max(1, logicalShape.w), Mathf.Max(1, logicalShape.h), 1, 1);
            var outRt = usePack4LinearMat
                ? owner.RentTempArray(storageShape.w, storageShape.h, 1, owner.ResolveActivationTextureFormat(layer, logicalShape.dims))
                : useStrictLinearMat
                    ? owner.RentTempMat(storageShape.w, storageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat())
                    : owner.RentTempArray(storageShape.w, storageShape.h, 1, owner.ResolveActivationTextureFormat(layer, logicalShape.dims));
            if (usePack4LinearMat)
            {
                owner.Ops.Gemm2DPack4LinearTextureA(
                    srcTex.texture,
                    srcIsPack4Linear,
                    ip.TextureWeightBinding,
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
            else if (useStrictLinearMat)
            {
                owner.Ops.Gemm2DLinearTextureA(
                    srcTex.texture,
                    ip.TextureWeightBinding,
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
                    ip.TextureWeightBinding,
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
                + " | srcPack4Linear=" + (srcIsPack4Linear ? "1" : "0")
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | out=d" + logicalShape.dims + ":" + logicalShape.w + "x" + logicalShape.h + "x" + logicalShape.d + "x" + logicalShape.c
                + " | outFormat=" + outRt.format);
            AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], outRt, logicalShape, storageShape);
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

        private static bool TryResolveAttentionPack4ToLinearInput(
            AexisGraphSession.TensorRef src,
            AexisGraphSession.BufferShape logicalShape,
            AexisGraphSession.InnerProductPack ip,
            out AexisGraphSession.BufferShape storageShape,
            out int rows,
            out int headDim,
            out int numHeads)
        {
            storageShape = default;
            rows = 0;
            headDim = 0;
            numHeads = 0;
            if (src == null || src.texture == null || ip == null || logicalShape.dims != 2)
                return false;

            storageShape = AexisGraphSession.GetTextureStorageShape(src, logicalShape);
            if (storageShape.dims != 3
                || storageShape.d != 1
                || storageShape.w <= 0
                || storageShape.h <= 0
                || storageShape.c <= 1
                || !AexisGraphSession.MatchesPack4TextureStorage(src, storageShape))
                return false;

            headDim = storageShape.w;
            numHeads = storageShape.c;
            rows = storageShape.h;
            return logicalShape.w == ip.inFeatures
                && logicalShape.w == headDim * numHeads
                && logicalShape.h == rows;
        }

        private static bool TryResolveAttentionPack4ToLinearInput(
            AexisGraphSession.CmdTensorRef src,
            AexisGraphSession.BufferShape logicalShape,
            AexisGraphSession.InnerProductPack ip,
            out AexisGraphSession.BufferShape storageShape,
            out int rows,
            out int headDim,
            out int numHeads)
        {
            storageShape = default;
            rows = 0;
            headDim = 0;
            numHeads = 0;
            if (src == null || src.texture == null || ip == null || logicalShape.dims != 2)
                return false;

            storageShape = AexisGraphSession.GetCmdStorageShape(src, logicalShape);
            if (storageShape.dims != 3
                || storageShape.d != 1
                || storageShape.w <= 0
                || storageShape.h <= 0
                || storageShape.c <= 1
                || !AexisGraphSession.MatchesPack4TextureStorage(src, storageShape))
                return false;

            headDim = storageShape.w;
            numHeads = storageShape.c;
            rows = storageShape.h;
            return logicalShape.w == ip.inFeatures
                && logicalShape.w == headDim * numHeads
                && logicalShape.h == rows;
        }
    }
}
