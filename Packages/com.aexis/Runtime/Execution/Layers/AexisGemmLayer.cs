using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisGemmLayer : AexisBaseLayer
    {
        public AexisGemmLayer() : base(AexisLayerTypes.Gemm, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var gp = new AexisGraphSession.GemmPack
                                        {
                                            alpha = layer.GetFloat(0, 1f),
                                            beta = layer.GetFloat(1, 1f),
                                            transA = layer.GetInt(2, 0) != 0,
                                            transB = layer.GetInt(3, 0) != 0,
                                            constantA = layer.GetInt(4, 0) != 0,
                                            constantB = layer.GetInt(5, 0) != 0,
                                            constantC = layer.GetInt(6, 0) != 0,
                                            constantM = layer.GetInt(7, 0),
                                            constantN = layer.GetInt(8, 0),
                                            constantK = layer.GetInt(9, 0),
                                            broadcastTypeC = layer.GetInt(10, 0)
                                        };

                                        if (gp.constantA)
                                            throw new InvalidOperationException("Gemm constantA is not supported in AexisGraphSession: " + layer.name);
                                        if (gp.constantB)
                                        {
                                            var bw = gp.transB ? gp.constantK : gp.constantN;
                                            var bh = gp.transB ? gp.constantN : gp.constantK;
                                            var weightCount = checked(bw * bh);
                                            var hasSharedFp32 = owner.SharedTokenEmbeddingWeights != null
                                                && !owner.UsesQuantizedWeightsForLayer(layer)
                                                && !owner.UsesFp16WeightsForLayer(layer);
                                            var hasSharedInt8 = owner.SharedTokenEmbeddingWeightsInt8Packed != null
                                                && owner.SharedTokenEmbeddingWeightsInt8Scales != null
                                                && owner.UsesInt8WeightOnlyForLayer(layer);
                                            var useSharedWeights = owner.SharedTokenEmbeddingElementCount == weightCount
                                                && (hasSharedFp32 || hasSharedInt8);
                                            AexisQuantizedTensor directQ8 = null;

                                            phaseSw.Restart();
                                            if (useSharedWeights)
                                                br.SkipTensor(bw, bh, 0, 0, 0);
                                            else if (!(owner.UsesInt8WeightOnlyForLayer(layer)
                                                && gp.transB
                                                && br.TryReadQuantizedTensor(weightCount, gp.constantK, out directQ8)))
                                                gp.bDataCpu = AexisGraphSession.ReadClipMatAsFloat32(br, bw, bh, 0, 0, 0);
                                            if (gp.constantC && gp.broadcastTypeC != -1)
                                            {
                                                int cw;
                                                int ch;
                                                switch (gp.broadcastTypeC)
                                                {
                                                    case 0: cw = 1; ch = 0; break;
                                                    case 1: cw = gp.constantM; ch = 0; break;
                                                    case 2: cw = 1; ch = gp.constantM; break;
                                                    case 3: cw = gp.constantN; ch = gp.constantM; break;
                                                    case 4: cw = gp.constantN; ch = 1; break;
                                                    default:
                                                        throw new InvalidOperationException("Gemm broadcast_type_C unsupported: " + gp.broadcastTypeC + " | " + layer.name);
                                                }

                                                var c = AexisGraphSession.ReadClipMatAsFloat32(br, cw, ch, 0, 0, 0);
                                                gp.cDataCpu = c;
                                            }
                                            phaseSw.Stop();
                                            readMs += phaseSw.ElapsedMilliseconds;

                                            phaseSw.Restart();
                                            if (useSharedWeights)
                                            {
                                                if (hasSharedInt8)
                                                {
                                                    gp.bDataInt8Packed = owner.SharedTokenEmbeddingWeightsInt8Packed;
                                                    gp.bDataInt8Scales = owner.SharedTokenEmbeddingWeightsInt8Scales;
                                                    gp.ownsBDataInt8 = false;
                                                }
                                                else
                                                {
                                                    gp.bData = owner.SharedTokenEmbeddingWeights;
                                                    gp.ownsBData = false;
                                                }
                                            }
                                            else if (owner.UsesInt4WeightOnlyForLayer(layer))
                                            {
                                                var quantized = AexisGraphSession.NewInt4WeightOnlyUpload(
                                                    gp.bDataCpu,
                                                    gp.constantN,
                                                    gp.constantK,
                                                    outputChannelsAreContiguous: gp.transB,
                                                    "AexisGraphSession.GemmInt4WeightOnly:" + layer.name);
                                                gp.bDataInt4Packed = quantized.packedWeights;
                                                gp.bDataInt4Scales = quantized.scales;
                                            }
                                            else if (owner.UsesInt8WeightOnlyForLayer(layer))
                                            {
                                                var quantized = directQ8 != null
                                                    ? AexisGraphSession.NewInt8WeightOnlyUpload(
                                                        directQ8,
                                                        gp.constantN,
                                                        "AexisGraphSession.GemmInt8WeightOnlyDirect:" + layer.name)
                                                    : AexisGraphSession.NewInt8WeightOnlyUpload(
                                                        gp.bDataCpu,
                                                        gp.constantN,
                                                        gp.constantK,
                                                        outputChannelsAreContiguous: gp.transB,
                                                        "AexisGraphSession.GemmInt8WeightOnly:" + layer.name);
                                                gp.bDataInt8Packed = quantized.packedWeights;
                                                gp.bDataInt8Scales = quantized.scales;
                                            }
                                            else
                                            {
                                                gp.bData = AexisGraphSession.NewBuffer(gp.bDataCpu);
                                            }
                                            if (owner.UsesFp16WeightsForLayer(layer))
                                                gp.bDataFp16 = AexisGraphSession.NewFp16Buffer(gp.bDataCpu, "AexisGraphSession.GemmWeightFp16:" + layer.name);
                                            if (gp.cDataCpu != null)
                                                gp.cData = AexisGraphSession.NewBuffer(gp.cDataCpu);
                                            if (owner.UsesQuantizedWeightsForLayer(layer))
                                                gp.bDataCpu = null;
                                            phaseSw.Stop();
                                            uploadMs += phaseSw.ElapsedMilliseconds;
                                        }

                                        owner._gemm[layer.name] = gp;
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
                                                if (!owner._gemm.TryGetValue(layer.name, out var gp))
                                                    throw new InvalidOperationException("Gemm not found: " + layer.name);

                                                var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                var srcView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                if (srcBuf == null || srcView == null)
                                                    throw new InvalidOperationException("Gemm source not found: " + layer.name);
                                                if (gp.transA)
                                                    throw new InvalidOperationException("Gemm transA is not supported in AexisGraphSession: " + layer.name);
                                                if (srcView.dims != 1 && srcView.dims != 2)
                                                    throw new InvalidOperationException("Gemm expects dims<=2 source tensor: " + layer.name);

                                                var m = srcView.dims == 1 ? 1 : srcView.h;
                                                var k = srcView.w;
                                                ComputeBuffer bBuf;
                                                int bRows;
                                                int bCols;
                                                if (gp.constantB)
                                                {
                                                    if (gp.bData == null)
                                                        throw new InvalidOperationException("Gemm constantB buffer missing: " + layer.name);
                                                    bBuf = gp.bData;
                                                    bRows = gp.transB ? gp.constantN : gp.constantK;
                                                    bCols = gp.transB ? gp.constantK : gp.constantN;
                                                }
                                                else
                                                {
                                                    if (layer.bottomNames.Length < 2)
                                                        throw new InvalidOperationException("Gemm B input missing: " + layer.name);
                                                    bBuf = owner.GetOrConvertToBuffer(layer.bottomNames[1], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                    var bView = AexisGraphSession.TryGetBufferView(layer.bottomNames[1], bufferBlobs, bufferViews);
                                                    if (bBuf == null || bView == null || (bView.dims != 1 && bView.dims != 2))
                                                        throw new InvalidOperationException("Gemm B input invalid: " + layer.name);
                                                    bRows = bView.dims == 1 ? 1 : bView.h;
                                                    bCols = bView.w;
                                                }

                                                var kFromB = gp.transB ? bCols : bRows;
                                                var n = gp.transB ? bRows : bCols;
                                                if (gp.constantK > 0 && k != gp.constantK)
                                                    throw new InvalidOperationException("Gemm input K mismatch: " + layer.name + " | " + k + " vs " + gp.constantK);
                                                if (k != kFromB)
                                                    throw new InvalidOperationException("Gemm K mismatch: " + layer.name + " | " + k + " vs " + kFromB);

                                                var useC = false;
                                                ComputeBuffer cBuf = null;
                                                float[] cCpu = null;
                                                if (gp.constantC && gp.broadcastTypeC != -1 && gp.cData != null)
                                                {
                                                    useC = true;
                                                    cBuf = gp.cData;
                                                    cCpu = gp.cDataCpu;
                                                }
                                                else if (!gp.constantC && layer.bottomNames.Length > 2)
                                                {
                                                    cBuf = owner.GetOrConvertToBuffer(layer.bottomNames[2], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                    useC = cBuf != null;
                                                    if (useC)
                                                        cCpu = AexisGraphSession.ReadFloatBuffer(cBuf);
                                                }

                                                var outTensor = m == 1 && srcView.dims == 1
                                                    ? owner.RentTempTensorBuffer(1, n)
                                                    : owner.RentTempTensorBuffer(2, n, m);
                                                if (owner.ForceCpuGemmAll)
                                                {
                                                    if (!gp.constantB)
                                                        throw new InvalidOperationException("ForceCpuGemmAll currently requires constantB=1: " + layer.name);

                                                    var cpuOut = AexisGraphSession.RunGemmCpu(srcBuf, srcView, gp, cCpu);
                                                    outTensor.buffer.SetData(cpuOut);
                                                    owner.DebugLog?.Invoke(
                                                        "ForceCpuGemmAll applied"
                                                        + " | layer=" + layer.name
                                                        + " | m=" + m.ToString(CultureInfo.InvariantCulture)
                                                        + " | n=" + n.ToString(CultureInfo.InvariantCulture)
                                                        + " | k=" + k.ToString(CultureInfo.InvariantCulture)
                                                        + " | useC=" + useC.ToString());
                                                }
                                                else
                                                {
                                                    owner.Ops.Gemm2D(srcBuf, bBuf, cBuf, m, n, k, gp.transB, gp.alpha, gp.beta, useC, gp.broadcastTypeC, outTensor.buffer);
                                                }
                                                var textureFormatOverride = ShouldPromoteAttentionGemmOutputTexture(owner, layer)
                                                    ? RenderTextureFormat.ARGBFloat
                                                    : (RenderTextureFormat?)null;
                                                owner.PublishTensorBufferOutput(
                                                    layer.topNames[0],
                                                    outTensor,
                                                    preferTexture: true,
                                                    textureBlobs,
                                                    textureShapes,
                                                    bufferBlobs,
                                                    bufferRefs,
                                                    bufferViews,
                                                    tempOwned,
                                                    textureFormatOverride);
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

            if (!owner._gemm.TryGetValue(layer.name, out var gp))
                throw new InvalidOperationException("Gemm not found: " + layer.name);

            var aShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            throw new NotSupportedException("CommandBuffer Gemm requires non-transposed A and a verified texture-native constant-B profile"
                + " | layer=" + layer.name
                + " | input=d" + aShape.dims + ":" + aShape.w + "x" + aShape.h + "x" + aShape.d + "x" + aShape.c
                + " | constantB=" + (gp.constantB ? "1" : "0")
                + " | transA=" + (gp.transA ? "1" : "0")
                + " | transB=" + (gp.transB ? "1" : "0")
                + " | rejectedFallback=placeholder-or-buffer-materialization");
        }

        private static void ConfigureGemmWeightBindings(AexisGraphSession owner, AexisGraphModel.Layer layer)
        {
            var hasPack = owner._gemm.TryGetValue(layer.name, out var pack);
            var useInt8WeightOnly = owner.UsesInt8WeightsForLayer(layer);
            var useInt4WeightOnly = owner.UsesInt4WeightsForLayer(layer);
            var useFp16Weights = owner.UsesFp16WeightsForLayer(layer) && !owner.UsesQuantizedWeightsForLayer(layer);
            owner.Ops.SetFp16GemmWeights(useFp16Weights && hasPack ? pack.bDataFp16 : null);
            owner.Ops.SetInt8GemmWeights(
                useInt8WeightOnly && hasPack ? pack.bDataInt8Packed : null,
                useInt8WeightOnly && hasPack ? pack.bDataInt8Scales : null);
            owner.Ops.SetInt4GemmWeights(
                useInt4WeightOnly && hasPack ? pack.bDataInt4Packed : null,
                useInt4WeightOnly && hasPack ? pack.bDataInt4Scales : null);
            owner.ConfigureInt8ActivationQuantization(layer);
        }

        private static bool TryExecuteCommandBufferTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (owner == null || layer == null || context == null)
                return false;
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;
            if (!owner._gemm.TryGetValue(layer.name, out var gp))
                return false;
            if (gp.transA)
                return false;

            if (TryExecuteCommandBufferAttentionProjectionTexturePath(owner, layer, context, gp))
                return true;
            if (TryExecuteCommandBufferTextureMatMulPath(owner, layer, context, gp))
                return true;
            if (!gp.constantB)
                return false;

            var srcTex = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var srcShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            if (srcTex == null || srcTex.texture == null)
                return false;
            if (srcShape.dims != 2)
                return false;
            var srcIsStrictLinear = AexisGraphSession.IsStrictLinearMatTexture(srcTex);
            var srcIsPack4Linear = IsPackedLogical2DTexture(srcTex, srcShape);
            if (!srcIsPack4Linear && (srcTex.width != srcShape.w || srcTex.height != srcShape.h || srcTex.packs != 1))
                return false;
            if (srcShape.w <= 0 || srcShape.h <= 0)
                return false;

            var m = srcShape.h;
            var k = srcShape.w;
            var bRows = gp.transB ? gp.constantN : gp.constantK;
            var bCols = gp.transB ? gp.constantK : gp.constantN;
            var kFromB = gp.transB ? bCols : bRows;
            var n = gp.transB ? bRows : bCols;
            if (gp.constantK > 0 && k != gp.constantK)
                return false;
            if (k != kFromB || n <= 0)
                return false;

            var useC = gp.constantC && gp.broadcastTypeC != -1 && gp.cData != null;
            var outShape = new AexisGraphSession.BufferShape(2, Mathf.Max(1, n), Mathf.Max(1, m), 1, 1);
            var usePack4LinearMat = (srcIsStrictLinear || srcIsPack4Linear) && n % 4 == 0;
            var usePack4TiledMat = usePack4LinearMat
                && Mathf.CeilToInt(n / 4f) > Mathf.Max(1, SystemInfo.maxTextureSize);
            var useStrictLinearMat = srcIsStrictLinear && !usePack4LinearMat;
            var outStorageShape = usePack4TiledMat
                ? AexisGraphSession.ResolvePack4TiledLinearMatStorageShape(outShape)
                : usePack4LinearMat
                ? AexisGraphSession.ResolvePack4LinearMatStorageShape(outShape)
                : useStrictLinearMat
                ? AexisGraphSession.ResolveLinearMatStorageShape(outShape)
                : new AexisGraphSession.BufferShape(3, outShape.w, outShape.h, 1, 1);
            var outRt = usePack4LinearMat
                ? owner.RentTempArray(
                    context.commandBuffer,
                    outStorageShape.w,
                    outStorageShape.h,
                    1,
                    ShouldPromoteAttentionGemmOutputTexture(owner, layer) ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGBHalf)
                : useStrictLinearMat
                ? owner.RentTempMat(context.commandBuffer, outStorageShape.w, outStorageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat())
                : owner.RentTempArray(
                    context.commandBuffer,
                    outShape.w,
                    outShape.h,
                    1,
                    ShouldPromoteAttentionGemmOutputTexture(owner, layer) ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGBHalf);
            if (usePack4LinearMat)
            {
                if (usePack4TiledMat)
                {
                    if (srcIsPack4Linear)
                        owner.Ops.Gemm2DPack4TiledTextureAFromPack4(context.commandBuffer, srcTex.texture, gp.TextureWeightBinding, useC ? gp.cData : null, m, n, k, gp.transB, gp.alpha, gp.beta, useC, gp.broadcastTypeC, outStorageShape.h / m, outRt);
                    else
                        owner.Ops.Gemm2DPack4TiledTextureAFromLinear(context.commandBuffer, srcTex.texture, gp.TextureWeightBinding, useC ? gp.cData : null, m, n, k, gp.transB, gp.alpha, gp.beta, useC, gp.broadcastTypeC, outStorageShape.h / m, outRt);
                }
                else
                    owner.Ops.Gemm2DPack4LinearTextureA(context.commandBuffer, srcTex.texture, srcIsPack4Linear, gp.TextureWeightBinding, useC ? gp.cData : null, m, n, k, gp.transB, gp.alpha, gp.beta, useC, gp.broadcastTypeC, outRt);
            }
            else if (useStrictLinearMat)
            {
                owner.Ops.Gemm2DLinearTextureA(
                    context.commandBuffer,
                    srcTex.texture,
                    gp.TextureWeightBinding,
                    useC ? gp.cData : null,
                    m,
                    n,
                    k,
                    gp.transB,
                    gp.alpha,
                    gp.beta,
                    useC,
                    gp.broadcastTypeC,
                    outRt);
            }
            else
            {
                owner.Ops.Gemm2DTextureA(
                    context.commandBuffer,
                    srcTex.texture,
                    gp.TextureWeightBinding,
                    useC ? gp.cData : null,
                    m,
                    n,
                    k,
                    gp.transB,
                    gp.alpha,
                    gp.beta,
                    useC,
                    gp.broadcastTypeC,
                    outRt);
            }

            context.blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
            {
                texture = outRt,
                width = outStorageShape.w,
                height = outStorageShape.h,
                packs = 1,
                refs = 1,
                owned = true,
                hasLogicalShape = true,
                logicalShape = outShape,
                hasStorageShape = true,
                storageShape = outStorageShape
            };
            context.blobs[layer.topNames[0]].layoutKind = AexisGraphSession.ResolveNcnnTextureLayoutKind(outShape, outStorageShape, 1);
            context.shapes[layer.topNames[0]] = outShape;
            owner.DebugLog?.Invoke(
                "[CmdTexture][Gemm]"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c
                + " | outFormat=" + outRt.format
                + " | linear=" + (useStrictLinearMat ? "1" : "0")
                + " | pack4Linear=" + (usePack4LinearMat ? "1" : "0"));
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
            return true;
        }

        private static bool TryExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (owner == null || layer == null || context == null)
                return false;
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;
            if (!owner._gemm.TryGetValue(layer.name, out var gp))
                return false;
            if (gp.transA)
                return false;
            if (TryExecuteRenderTextureAttentionProjectionTexturePath(owner, layer, context, gp))
                return true;
            if (TryExecuteRenderTextureTextureMatMulPath(owner, layer, context, gp))
                return true;
            if (!gp.constantB)
                return false;
            if (!AexisGraphSession.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var srcTex, out var srcShape))
                return false;
            if (srcTex == null || srcTex.texture == null)
                return false;
            if (srcShape.dims != 2)
                return false;
            var srcIsStrictLinear = AexisGraphSession.IsStrictLinearMatTexture(srcTex);
            var srcIsPack4Linear = IsPackedLogical2DTexture(srcTex, srcShape);
            if (!srcIsPack4Linear && (srcTex.width != srcShape.w || srcTex.height != srcShape.h || srcTex.packs != 1))
                return false;
            if (srcShape.w <= 0 || srcShape.h <= 0)
                return false;
            if (layer.bottomNames == null || layer.bottomNames.Length < 1 || layer.topNames == null || layer.topNames.Length < 1)
                return false;

            var m = srcShape.h;
            var k = srcShape.w;
            var bRows = gp.transB ? gp.constantN : gp.constantK;
            var bCols = gp.transB ? gp.constantK : gp.constantN;
            var kFromB = gp.transB ? bCols : bRows;
            var n = gp.transB ? bRows : bCols;
            if (gp.constantK > 0 && k != gp.constantK)
                return false;
            if (k != kFromB || n <= 0)
                return false;

            var useC = gp.constantC && gp.broadcastTypeC != -1 && gp.cData != null;
            var outShape = new AexisGraphSession.BufferShape(2, Mathf.Max(1, n), Mathf.Max(1, m), 1, 1);
            var usePack4LinearMat = (srcIsStrictLinear || srcIsPack4Linear) && n % 4 == 0;
            var usePack4TiledMat = usePack4LinearMat
                && Mathf.CeilToInt(n / 4f) > Mathf.Max(1, SystemInfo.maxTextureSize);
            var useStrictLinearMat = srcIsStrictLinear && !usePack4LinearMat;
            var outStorageShape = usePack4TiledMat
                ? AexisGraphSession.ResolvePack4TiledLinearMatStorageShape(outShape)
                : usePack4LinearMat
                ? AexisGraphSession.ResolvePack4LinearMatStorageShape(outShape)
                : useStrictLinearMat
                ? AexisGraphSession.ResolveLinearMatStorageShape(outShape)
                : new AexisGraphSession.BufferShape(3, outShape.w, outShape.h, 1, 1);
            var outRt = usePack4LinearMat
                ? owner.RentTempArray(
                    outStorageShape.w,
                    outStorageShape.h,
                    1,
                    ShouldPromoteAttentionGemmOutputTexture(owner, layer) ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGBHalf)
                : useStrictLinearMat
                ? owner.RentTempMat(outStorageShape.w, outStorageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat())
                : owner.RentTempArray(
                    outShape.w,
                    outShape.h,
                    1,
                    ShouldPromoteAttentionGemmOutputTexture(owner, layer) ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGBHalf);
            if (usePack4LinearMat)
            {
                if (usePack4TiledMat)
                {
                    if (srcIsPack4Linear)
                        owner.Ops.Gemm2DPack4TiledTextureAFromPack4(srcTex.texture, gp.TextureWeightBinding, useC ? gp.cData : null, m, n, k, gp.transB, gp.alpha, gp.beta, useC, gp.broadcastTypeC, outStorageShape.h / m, outRt);
                    else
                        owner.Ops.Gemm2DPack4TiledTextureAFromLinear(srcTex.texture, gp.TextureWeightBinding, useC ? gp.cData : null, m, n, k, gp.transB, gp.alpha, gp.beta, useC, gp.broadcastTypeC, outStorageShape.h / m, outRt);
                }
                else
                    owner.Ops.Gemm2DPack4LinearTextureA(srcTex.texture, srcIsPack4Linear, gp.TextureWeightBinding, useC ? gp.cData : null, m, n, k, gp.transB, gp.alpha, gp.beta, useC, gp.broadcastTypeC, outRt);
            }
            else if (useStrictLinearMat)
            {
                owner.Ops.Gemm2DLinearTextureA(
                    srcTex.texture,
                    gp.TextureWeightBinding,
                    useC ? gp.cData : null,
                    m,
                    n,
                    k,
                    gp.transB,
                    gp.alpha,
                    gp.beta,
                    useC,
                    gp.broadcastTypeC,
                    outRt);
            }
            else
            {
                owner.Ops.Gemm2DTextureA(
                    srcTex.texture,
                    gp.TextureWeightBinding,
                    useC ? gp.cData : null,
                    m,
                    n,
                    k,
                    gp.transB,
                    gp.alpha,
                    gp.beta,
                    useC,
                    gp.broadcastTypeC,
                    outRt);
            }
            AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], outRt, outShape, outStorageShape);
            owner.DebugLog?.Invoke(
                "[Texture][Gemm] layer=" + layer.name
                + " srcLinear=" + (srcIsStrictLinear ? "1" : "0")
                + " srcPack4=" + (srcIsPack4Linear ? "1" : "0")
                + " tiled=" + (usePack4TiledMat ? "1" : "0")
                + " m=" + m + " n=" + n + " k=" + k
                + " srcTexture=" + srcTex.texture.width + "x" + srcTex.texture.height + "x" + srcTex.texture.volumeDepth
                + " outTexture=" + outRt.width + "x" + outRt.height + "x" + outRt.volumeDepth);
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

        private static bool IsPackedLogical2DTexture(AexisGraphSession.TensorRef tensor, AexisGraphSession.BufferShape shape)
        {
            if (AexisGraphSession.IsPack4LinearMatTexture(tensor, shape))
                return true;
            if (tensor == null
                || tensor.texture == null
                || shape.dims != 2
                || tensor.texture.dimension != TextureDimension.Tex2DArray
                || tensor.height != shape.h)
            {
                return false;
            }
            var slices = Mathf.Max(1, tensor.texture.volumeDepth > 0 ? tensor.texture.volumeDepth : tensor.packs);
            if (slices <= 1 || tensor.width >= shape.w)
                return false;
            var logicalPacks = checked((shape.w + 3) / 4);
            return checked(tensor.width * slices) >= logicalPacks
                && checked(tensor.width * (slices - 1)) < logicalPacks;
        }

        private static bool IsPackedLogical2DTexture(AexisGraphSession.CmdTensorRef tensor, AexisGraphSession.BufferShape shape)
        {
            if (AexisGraphSession.IsPack4LinearMatTexture(tensor, shape))
                return true;
            if (tensor == null || tensor.texture == null || shape.dims != 2 || tensor.height != shape.h)
                return false;
            var slices = Mathf.Max(1, tensor.packs);
            if (slices <= 1 || tensor.width >= shape.w)
                return false;
            var logicalPacks = checked((shape.w + 3) / 4);
            return checked(tensor.width * slices) >= logicalPacks
                && checked(tensor.width * (slices - 1)) < logicalPacks;
        }

        private static bool TryExecuteRenderTextureAttentionProjectionTexturePath(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisLayerBufferContext context,
            AexisGraphSession.GemmPack gp)
        {
            if (owner == null || layer == null || context == null || gp == null || !gp.constantB)
                return false;
            if (layer.bottomNames == null || layer.bottomNames.Length < 1 || layer.topNames == null || layer.topNames.Length < 1)
                return false;
            if (!AexisGraphSession.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var srcTex, out var srcShape))
                return false;
            if (srcTex == null || srcTex.texture == null || srcShape.dims != 2 || srcShape.w <= 0 || srcShape.h <= 0)
                return false;

            var bRows = gp.transB ? gp.constantN : gp.constantK;
            var bCols = gp.transB ? gp.constantK : gp.constantN;
            var kFromB = gp.transB ? bCols : bRows;
            var m = srcShape.h;
            var k = srcShape.w;
            var n = gp.transB ? bRows : bCols;
            if (gp.constantK > 0 && k != gp.constantK)
                return false;
            if (k != kFromB || n <= 0)
                return false;

            var logicalOutShape = new AexisGraphSession.BufferShape(2, Mathf.Max(1, n), Mathf.Max(1, m), 1, 1);
            var useC = gp.constantC && gp.broadcastTypeC != -1 && gp.cData != null;
            var outFormat = ShouldPromoteAttentionGemmOutputTexture(owner, layer)
                ? RenderTextureFormat.ARGBFloat
                : owner.ResolveActivationTextureFormat(layer, logicalOutShape.dims);

            if (TryResolveAttentionQkvProjectionSpec(owner, layer, logicalOutShape, out var packedQkvShape))
            {
                var outSlices = Mathf.Max(1, Mathf.CeilToInt(packedQkvShape.c / 4f));
                var outRt = owner.RentTempArray(packedQkvShape.w, packedQkvShape.h, outSlices, outFormat);
                var srcIsStrictLinear = AexisGraphSession.IsStrictLinearMatTexture(srcTex);
                var srcIsPack4Linear = AexisGraphSession.IsPack4LinearMatTexture(srcTex, srcShape);
                if (srcIsStrictLinear)
                {
                    owner.Ops.Gemm2DAttentionQkvLinearTextureA(
                        srcTex.texture,
                        gp.TextureWeightBinding,
                        useC ? gp.cData : null,
                        m,
                        n,
                        k,
                        gp.transB,
                        gp.alpha,
                        gp.beta,
                        useC,
                        gp.broadcastTypeC,
                        packedQkvShape.w,
                        packedQkvShape.c,
                        outRt);
                }
                else if (srcIsPack4Linear)
                {
                    owner.Ops.Gemm2DAttentionQkvPack4LinearTextureA(
                        srcTex.texture,
                        gp.TextureWeightBinding,
                        useC ? gp.cData : null,
                        m,
                        n,
                        k,
                        gp.transB,
                        gp.alpha,
                        gp.beta,
                        useC,
                        gp.broadcastTypeC,
                        packedQkvShape.w,
                        packedQkvShape.c,
                        outRt);
                }
                else if (srcTex.width == srcShape.w && srcTex.height == srcShape.h && srcTex.packs == 1)
                {
                    owner.Ops.Gemm2DAttentionQkvTextureA(
                        srcTex.texture,
                        gp.TextureWeightBinding,
                        useC ? gp.cData : null,
                        m,
                        n,
                        k,
                        gp.transB,
                        gp.alpha,
                        gp.beta,
                        useC,
                        gp.broadcastTypeC,
                        packedQkvShape.w,
                        packedQkvShape.c,
                        outRt);
                }
                else
                {
                    owner.ReturnTempArray(outRt);
                    return false;
                }

                AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], outRt, logicalOutShape, packedQkvShape);
                owner.DebugLog?.Invoke(
                    "[AttentionQkv][Gemm]"
                    + " | layer=" + layer.name
                    + " | linear=" + (srcIsStrictLinear ? "1" : "0")
                    + " | pack4Linear=" + (srcIsPack4Linear ? "1" : "0")
                    + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | packed=d" + packedQkvShape.dims + ":" + packedQkvShape.w + "x" + packedQkvShape.h + "x" + packedQkvShape.d + "x" + packedQkvShape.c);
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

            // A pack4-linear matrix is also the normal representation of FFN activations.
            // Only the SDPA context projection uses the attention-specific physical layout.
            if (!IsAttentionContextOutputProjection(owner, layer)
                || !TryResolveAttentionPackedLinearMatInput(srcTex, srcShape, out var packedAttentionInputShape))
                return false;

            var usePack4LinearOut = n % 4 == 0;
            var outStorageShape = usePack4LinearOut
                ? AexisGraphSession.ResolvePack4LinearMatStorageShape(logicalOutShape)
                : AexisGraphSession.ResolveLinearMatStorageShape(logicalOutShape);
            var outMat = usePack4LinearOut
                ? owner.RentTempArray(outStorageShape.w, outStorageShape.h, 1, outFormat)
                : owner.RentTempMat(outStorageShape.w, outStorageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
            if (usePack4LinearOut)
            {
                owner.Ops.Gemm2DAttentionPack4ToPack4LinearTextureA(
                    srcTex.texture,
                    gp.TextureWeightBinding,
                    useC ? gp.cData : null,
                    m,
                    n,
                    k,
                    gp.transB,
                    gp.alpha,
                    gp.beta,
                    useC,
                    gp.broadcastTypeC,
                    packedAttentionInputShape.w,
                    packedAttentionInputShape.c,
                    outMat);
            }
            else
            {
                owner.Ops.Gemm2DAttentionPack4ToLinearTextureA(
                    srcTex.texture,
                    gp.TextureWeightBinding,
                    useC ? gp.cData : null,
                    m,
                    n,
                    k,
                    gp.transB,
                    gp.alpha,
                    gp.beta,
                    useC,
                    gp.broadcastTypeC,
                    packedAttentionInputShape.w,
                    packedAttentionInputShape.c,
                    outMat);
            }
            AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], outMat, logicalOutShape, outStorageShape);
            owner.DebugLog?.Invoke(
                "[AttentionOutProj][Gemm]"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | packed=d" + packedAttentionInputShape.dims + ":" + packedAttentionInputShape.w + "x" + packedAttentionInputShape.h + "x" + packedAttentionInputShape.d + "x" + packedAttentionInputShape.c
                + " | out=d" + logicalOutShape.dims + ":" + logicalOutShape.w + "x" + logicalOutShape.h + "x" + logicalOutShape.d + "x" + logicalOutShape.c
                + " | outStorage=d" + outStorageShape.dims + ":" + outStorageShape.w + "x" + outStorageShape.h + "x" + outStorageShape.d + "x" + outStorageShape.c
                + " | pack4Linear=" + (usePack4LinearOut ? "1" : "0"));
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

        private static bool TryExecuteCommandBufferAttentionProjectionTexturePath(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisLayerCommandBufferContext context,
            AexisGraphSession.GemmPack gp)
        {
            if (owner == null || layer == null || context == null || gp == null || !gp.constantB)
                return false;
            if (layer.bottomNames == null || layer.bottomNames.Length < 1 || layer.topNames == null || layer.topNames.Length < 1)
                return false;
            if (!context.blobs.TryGetValue(layer.bottomNames[0], out var srcTex) || srcTex == null || srcTex.texture == null)
                return false;

            var srcShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            if (srcShape.dims != 2 || srcShape.w <= 0 || srcShape.h <= 0)
                return false;

            var bRows = gp.transB ? gp.constantN : gp.constantK;
            var bCols = gp.transB ? gp.constantK : gp.constantN;
            var kFromB = gp.transB ? bCols : bRows;
            var m = srcShape.h;
            var k = srcShape.w;
            var n = gp.transB ? bRows : bCols;
            if (gp.constantK > 0 && k != gp.constantK)
                return false;
            if (k != kFromB || n <= 0)
                return false;

            var logicalOutShape = new AexisGraphSession.BufferShape(2, Mathf.Max(1, n), Mathf.Max(1, m), 1, 1);
            var useC = gp.constantC && gp.broadcastTypeC != -1 && gp.cData != null;
            var outFormat = ShouldPromoteAttentionGemmOutputTexture(owner, layer)
                ? RenderTextureFormat.ARGBFloat
                : owner.ResolveActivationTextureFormat(layer, logicalOutShape.dims);

            if (TryResolveAttentionQkvProjectionSpec(owner, layer, logicalOutShape, out var packedQkvShape))
            {
                var outSlices = Mathf.Max(1, Mathf.CeilToInt(packedQkvShape.c / 4f));
                var outArr = owner.RentTempArray(context.commandBuffer, packedQkvShape.w, packedQkvShape.h, outSlices, outFormat);
                var srcIsStrictLinear = AexisGraphSession.IsStrictLinearMatTexture(srcTex);
                var srcIsPack4Linear = AexisGraphSession.IsPack4LinearMatTexture(srcTex, srcShape);
                if (srcIsStrictLinear)
                {
                    owner.Ops.Gemm2DAttentionQkvLinearTextureA(
                        context.commandBuffer,
                        srcTex.texture,
                        gp.TextureWeightBinding,
                        useC ? gp.cData : null,
                        m,
                        n,
                        k,
                        gp.transB,
                        gp.alpha,
                        gp.beta,
                        useC,
                        gp.broadcastTypeC,
                        packedQkvShape.w,
                        packedQkvShape.c,
                        outArr);
                }
                else if (srcIsPack4Linear)
                {
                    owner.Ops.Gemm2DAttentionQkvPack4LinearTextureA(
                        context.commandBuffer,
                        srcTex.texture,
                        gp.TextureWeightBinding,
                        useC ? gp.cData : null,
                        m,
                        n,
                        k,
                        gp.transB,
                        gp.alpha,
                        gp.beta,
                        useC,
                        gp.broadcastTypeC,
                        packedQkvShape.w,
                        packedQkvShape.c,
                        outArr);
                }
                else if (srcTex.width == srcShape.w && srcTex.height == srcShape.h && srcTex.packs == 1)
                {
                    owner.Ops.Gemm2DAttentionQkvTextureA(
                        context.commandBuffer,
                        srcTex.texture,
                        gp.TextureWeightBinding,
                        useC ? gp.cData : null,
                        m,
                        n,
                        k,
                        gp.transB,
                        gp.alpha,
                        gp.beta,
                        useC,
                        gp.broadcastTypeC,
                        packedQkvShape.w,
                        packedQkvShape.c,
                        outArr);
                }
                else
                {
                    owner.ReturnTempArray(context.commandBuffer, outArr);
                    return false;
                }

                context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outArr, logicalOutShape, packedQkvShape, owned: true);
                context.shapes[layer.topNames[0]] = logicalOutShape;
                owner.DebugLog?.Invoke(
                    "[CmdAttentionQkv][Gemm]"
                    + " | layer=" + layer.name
                    + " | linear=" + (srcIsStrictLinear ? "1" : "0")
                    + " | pack4Linear=" + (srcIsPack4Linear ? "1" : "0")
                    + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | packed=d" + packedQkvShape.dims + ":" + packedQkvShape.w + "x" + packedQkvShape.h + "x" + packedQkvShape.d + "x" + packedQkvShape.c);
                owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
                return true;
            }

            // Keep the command-buffer route in lockstep with the immediate pack4 RT route.
            if (!IsAttentionContextOutputProjection(owner, layer)
                || !TryResolveAttentionPackedLinearMatInput(srcTex, srcShape, out var packedAttentionInputShape))
                return false;

            var usePack4LinearOut = n % 4 == 0;
            var outStorageShape = usePack4LinearOut
                ? AexisGraphSession.ResolvePack4LinearMatStorageShape(logicalOutShape)
                : AexisGraphSession.ResolveLinearMatStorageShape(logicalOutShape);
            var outMat = usePack4LinearOut
                ? owner.RentTempArray(context.commandBuffer, outStorageShape.w, outStorageShape.h, 1, outFormat)
                : owner.RentTempMat(context.commandBuffer, outStorageShape.w, outStorageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
            if (usePack4LinearOut)
            {
                owner.Ops.Gemm2DAttentionPack4ToPack4LinearTextureA(
                    context.commandBuffer,
                    srcTex.texture,
                    gp.TextureWeightBinding,
                    useC ? gp.cData : null,
                    m,
                    n,
                    k,
                    gp.transB,
                    gp.alpha,
                    gp.beta,
                    useC,
                    gp.broadcastTypeC,
                    packedAttentionInputShape.w,
                    packedAttentionInputShape.c,
                    outMat);
            }
            else
            {
                owner.Ops.Gemm2DAttentionPack4ToLinearTextureA(
                    context.commandBuffer,
                    srcTex.texture,
                    gp.TextureWeightBinding,
                    useC ? gp.cData : null,
                    m,
                    n,
                    k,
                    gp.transB,
                    gp.alpha,
                    gp.beta,
                    useC,
                    gp.broadcastTypeC,
                    packedAttentionInputShape.w,
                    packedAttentionInputShape.c,
                    outMat);
            }
            context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outMat, logicalOutShape, outStorageShape, owned: true);
            context.shapes[layer.topNames[0]] = logicalOutShape;
            owner.DebugLog?.Invoke(
                "[CmdAttentionOutProj][Gemm]"
                + " | layer=" + layer.name
                + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                + " | packed=d" + packedAttentionInputShape.dims + ":" + packedAttentionInputShape.w + "x" + packedAttentionInputShape.h + "x" + packedAttentionInputShape.d + "x" + packedAttentionInputShape.c
                + " | out=d" + logicalOutShape.dims + ":" + logicalOutShape.w + "x" + logicalOutShape.h + "x" + logicalOutShape.d + "x" + logicalOutShape.c
                + " | outStorage=d" + outStorageShape.dims + ":" + outStorageShape.w + "x" + outStorageShape.h + "x" + outStorageShape.d + "x" + outStorageShape.c
                + " | pack4Linear=" + (usePack4LinearOut ? "1" : "0"));
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
            return true;
        }

        private static bool TryResolveAttentionQkvProjectionSpec(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisGraphSession.BufferShape logicalOutShape,
            out AexisGraphSession.BufferShape packedShape)
        {
            packedShape = default;
            if (owner?.Model?.layers == null || layer?.topNames == null || layer.topNames.Length == 0)
                return false;

            var reshape = FindSingleConsumer(owner.Model, layer.topNames[0]);
            if (reshape == null || reshape.type != AexisLayerTypes.Reshape || reshape.topNames == null || reshape.topNames.Length == 0)
                return false;

            var reshapeOut = AexisGraphSession.ResolveReshapeShape(logicalOutShape, reshape);
            if (reshapeOut.dims != 3 || reshapeOut.d != 1 || reshapeOut.w <= 0 || reshapeOut.h <= 0 || reshapeOut.c <= 0)
                return false;

            var permute = FindSingleConsumer(owner.Model, reshape.topNames[0]);
            if (permute == null || permute.type != AexisLayerTypes.Permute || permute.topNames == null || permute.topNames.Length == 0 || permute.GetInt(0, -1) != 2)
                return false;

            var permuteOut = AexisGraphSession.ResolvePermuteShape(reshapeOut, 3, AexisGraphSession.ResolvePermuteAxes(3, 2, permute.name));
            if (permuteOut.dims != 3
                || permuteOut.d != 1
                || permuteOut.w != reshapeOut.w
                || permuteOut.h != reshapeOut.c
                || permuteOut.c != reshapeOut.h)
            {
                return false;
            }

            var sdpa = FindSingleConsumer(owner.Model, permute.topNames[0]);
            if (sdpa == null || sdpa.type != AexisLayerTypes.SDPA)
                return false;
            if (logicalOutShape.w != permuteOut.w * permuteOut.c || logicalOutShape.h != permuteOut.h)
                return false;

            packedShape = permuteOut;
            return true;
        }

        private static bool TryResolveAttentionPackedLinearMatInput(
            AexisGraphSession.TensorRef source,
            AexisGraphSession.BufferShape logicalShape,
            out AexisGraphSession.BufferShape packedShape)
        {
            packedShape = default;
            if (source == null || source.texture == null || logicalShape.dims != 2 || AexisGraphSession.IsStrictLinearMatTexture(source))
                return false;

            var storageShape = AexisGraphSession.GetTextureStorageShape(source, logicalShape);
            if (storageShape.dims != 3 || storageShape.d != 1 || storageShape.w <= 0 || storageShape.h <= 0 || storageShape.c <= 1)
                return false;
            if (source.width != storageShape.w || source.height != storageShape.h)
                return false;
            if (source.packs != Mathf.Max(1, Mathf.CeilToInt(storageShape.c / 4f)))
                return false;
            if (logicalShape.w != storageShape.w * storageShape.c || logicalShape.h != storageShape.h)
                return false;

            packedShape = storageShape;
            return true;
        }

        private static bool TryResolveAttentionPackedLinearMatInput(
            AexisGraphSession.CmdTensorRef source,
            AexisGraphSession.BufferShape logicalShape,
            out AexisGraphSession.BufferShape packedShape)
        {
            packedShape = default;
            if (source == null || source.texture == null || logicalShape.dims != 2 || AexisGraphSession.IsStrictLinearMatTexture(source))
                return false;

            var storageShape = AexisGraphSession.GetCmdStorageShape(source, logicalShape);
            if (storageShape.dims != 3 || storageShape.d != 1 || storageShape.w <= 0 || storageShape.h <= 0 || storageShape.c <= 1)
                return false;
            if (source.width != storageShape.w || source.height != storageShape.h)
                return false;
            if (source.packs != Mathf.Max(1, Mathf.CeilToInt(storageShape.c / 4f)))
                return false;
            if (logicalShape.w != storageShape.w * storageShape.c || logicalShape.h != storageShape.h)
                return false;

            packedShape = storageShape;
            return true;
        }

        private static bool TryExecuteCommandBufferTextureMatMulPath(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisLayerCommandBufferContext context,
            AexisGraphSession.GemmPack gp)
        {
            if (owner == null || layer == null || context == null || gp == null)
                return false;
            if (gp.constantB || gp.constantC || gp.broadcastTypeC != 0)
                return false;
            if (layer.bottomNames == null || layer.bottomNames.Length < 2 || layer.topNames == null || layer.topNames.Length < 1)
                return false;
            if (!TryGetScalar2DCmdTexture(context.blobs, context.shapes, layer.bottomNames[0], out var aTex, out var aShape)
                || !TryGetScalar2DCmdTexture(context.blobs, context.shapes, layer.bottomNames[1], out var bTex, out var bShape))
            {
                return false;
            }

            var m = aShape.h;
            var k = aShape.w;
            var bRows = bShape.h;
            var bCols = bShape.w;
            var kFromB = gp.transB ? bCols : bRows;
            var n = gp.transB ? bRows : bCols;
            if (gp.constantK > 0 && k != gp.constantK)
                return false;
            if (k != kFromB || m <= 0 || n <= 0)
                return false;

            var outShape = new AexisGraphSession.BufferShape(2, Mathf.Max(1, n), Mathf.Max(1, m), 1, 1);
            var useStrictLinearOut = AexisGraphSession.IsStrictLinearMatTexture(aTex) || AexisGraphSession.IsStrictLinearMatTexture(bTex);
            var outStorageShape = useStrictLinearOut
                ? AexisGraphSession.ResolveLinearMatStorageShape(outShape)
                : new AexisGraphSession.BufferShape(3, outShape.w, outShape.h, 1, 1);
            var outFormat = ShouldPromoteAttentionGemmOutputTexture(owner, layer)
                ? RenderTextureFormat.ARGBFloat
                : owner.ResolveActivationTextureFormat(layer, outShape.dims);
            ComputeTexture aMaterialized = null;
            ComputeTexture bMaterialized = null;
            ComputeTexture outRt = null;
            try
            {
                var aInput = MaterializeScalar2DInput(owner, context.commandBuffer, aTex, aShape, outFormat, ref aMaterialized);
                var bInput = MaterializeScalar2DInput(owner, context.commandBuffer, bTex, bShape, outFormat, ref bMaterialized);
                outRt = owner.RentTempArray(context.commandBuffer, outShape.w, outShape.h, 1, outFormat);
                owner.Ops.MatMulPack4Cdhw(
                    context.commandBuffer,
                    aInput,
                    m,
                    k,
                    1,
                    1,
                    bInput,
                    bRows,
                    bCols,
                    1,
                    1,
                    gp.transB,
                    1,
                    1,
                    outRt);

                if (!Mathf.Approximately(gp.alpha, 1f))
                {
                    var scaledRt = owner.RentTempArray(context.commandBuffer, outShape.w, outShape.h, 1, outFormat);
                    owner.Ops.BinaryOpScalarPack4(context.commandBuffer, outRt, gp.alpha, 1, 2, scaledRt);
                    owner.ReturnTempArray(context.commandBuffer, outRt);
                    outRt = scaledRt;
                }

                if (useStrictLinearOut)
                {
                    var outMat = owner.RentTempMat(context.commandBuffer, outStorageShape.w, outStorageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                    owner.Ops.ReshapePack4ToLinearMat(context.commandBuffer, outRt, outShape.w, outShape.h, outShape.d, outShape.c, outShape.dims, outMat);
                    context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outMat, outShape, outStorageShape, owned: true);
                }
                else
                {
                    context.blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
                    {
                        texture = outRt,
                        width = outShape.w,
                        height = outShape.h,
                        packs = 1,
                        refs = 1,
                        owned = true,
                        hasLogicalShape = true,
                        logicalShape = outShape,
                        hasStorageShape = true,
                        storageShape = outStorageShape
                    };
                    outRt = null;
                }
            }
            finally
            {
                ReturnTempUnique(owner, context.commandBuffer, ref bMaterialized, aMaterialized);
                if (aMaterialized != null)
                    owner.ReturnTempArray(context.commandBuffer, aMaterialized);
                if (outRt != null)
                    owner.ReturnTempArray(context.commandBuffer, outRt);
            }
            context.shapes[layer.topNames[0]] = outShape;
            owner.DebugLog?.Invoke(
                "[CmdTexture2D][Gemm]"
                + " | layer=" + layer.name
                + " | a=d" + aShape.dims + ":" + aShape.w + "x" + aShape.h + "x" + aShape.d + "x" + aShape.c
                + " | b=d" + bShape.dims + ":" + bShape.w + "x" + bShape.h + "x" + bShape.d + "x" + bShape.c
                + " | transB=" + (gp.transB ? "1" : "0")
                + " | alpha=" + gp.alpha.ToString(CultureInfo.InvariantCulture)
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c
                + " | linear=" + (useStrictLinearOut ? "1" : "0"));
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
            return true;
        }

        private static bool TryExecuteRenderTextureTextureMatMulPath(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisLayerBufferContext context,
            AexisGraphSession.GemmPack gp)
        {
            if (owner == null || layer == null || context == null || gp == null)
                return false;
            if (gp.constantB || gp.constantC || gp.broadcastTypeC != 0)
                return false;
            if (layer.bottomNames == null || layer.bottomNames.Length < 2 || layer.topNames == null || layer.topNames.Length < 1)
                return false;
            if (!TryGetScalar2DTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var aTex, out var aShape)
                || !TryGetScalar2DTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[1], out var bTex, out var bShape))
            {
                return false;
            }

            var m = aShape.h;
            var k = aShape.w;
            var bRows = bShape.h;
            var bCols = bShape.w;
            var kFromB = gp.transB ? bCols : bRows;
            var n = gp.transB ? bRows : bCols;
            if (gp.constantK > 0 && k != gp.constantK)
                return false;
            if (k != kFromB || m <= 0 || n <= 0)
                return false;

            var outShape = new AexisGraphSession.BufferShape(2, Mathf.Max(1, n), Mathf.Max(1, m), 1, 1);
            var useStrictLinearOut = AexisGraphSession.IsStrictLinearMatTexture(aTex) || AexisGraphSession.IsStrictLinearMatTexture(bTex);
            var outStorageShape = useStrictLinearOut
                ? AexisGraphSession.ResolveLinearMatStorageShape(outShape)
                : new AexisGraphSession.BufferShape(3, outShape.w, outShape.h, 1, 1);
            var outFormat = ShouldPromoteAttentionGemmOutputTexture(owner, layer)
                ? RenderTextureFormat.ARGBFloat
                : owner.ResolveActivationTextureFormat(layer, outShape.dims);
            RenderTexture aMaterialized = null;
            RenderTexture bMaterialized = null;
            RenderTexture outRt = null;
            try
            {
                var aInput = MaterializeScalar2DInput(owner, aTex, aShape, outFormat, ref aMaterialized);
                var bInput = MaterializeScalar2DInput(owner, bTex, bShape, outFormat, ref bMaterialized);
                outRt = owner.RentTempArray(outShape.w, outShape.h, 1, outFormat);
                owner.Ops.MatMulPack4Cdhw(
                    aInput,
                    m,
                    k,
                    1,
                    1,
                    bInput,
                    bRows,
                    bCols,
                    1,
                    1,
                    gp.transB,
                    1,
                    1,
                    outRt);

                if (!Mathf.Approximately(gp.alpha, 1f))
                {
                    var scaledRt = owner.RentTempArray(outShape.w, outShape.h, 1, outFormat);
                    owner.Ops.BinaryOpScalarPack4(outRt, gp.alpha, 1, 2, scaledRt);
                    owner.ReturnTempArray(outRt);
                    outRt = scaledRt;
                }

                if (useStrictLinearOut)
                {
                    var outMat = owner.RentTempMat(outStorageShape.w, outStorageShape.h, AexisGraphSession.ResolveLinearMatTextureFormat());
                    owner.Ops.ReshapePack4ToLinearMat(outRt, outShape.w, outShape.h, outShape.d, outShape.c, outShape.dims, outMat);
                    AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], outMat, outShape, outStorageShape);
                }
                else
                {
                    AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], outRt, outShape, outStorageShape);
                    outRt = null;
                }
            }
            finally
            {
                ReturnTempUnique(owner, ref bMaterialized, aMaterialized);
                if (aMaterialized != null)
                    owner.ReturnTempArray(aMaterialized);
                if (outRt != null)
                    owner.ReturnTempArray(outRt);
            }
            owner.DebugLog?.Invoke(
                "[Texture2D][Gemm]"
                + " | layer=" + layer.name
                + " | a=d" + aShape.dims + ":" + aShape.w + "x" + aShape.h + "x" + aShape.d + "x" + aShape.c
                + " | b=d" + bShape.dims + ":" + bShape.w + "x" + bShape.h + "x" + bShape.d + "x" + bShape.c
                + " | transB=" + (gp.transB ? "1" : "0")
                + " | alpha=" + gp.alpha.ToString(CultureInfo.InvariantCulture)
                + " | out=d" + outShape.dims + ":" + outShape.w + "x" + outShape.h + "x" + outShape.d + "x" + outShape.c
                + " | linear=" + (useStrictLinearOut ? "1" : "0"));
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

        private static bool TryGetScalar2DTexture(
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            string blobName,
            out AexisGraphSession.TensorRef texture,
            out AexisGraphSession.BufferShape shape)
        {
            texture = null;
            shape = default;
            if (!AexisGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, blobName, out texture, out shape))
                return false;
            return texture != null
                && texture.texture != null
                && shape.dims == 2
                && shape.w > 0
                && shape.h > 0
                && texture.width == shape.w
                && texture.height == shape.h
                && texture.packs == 1;
        }

        private static bool TryGetScalar2DCmdTexture(
            Dictionary<string, AexisGraphSession.CmdTensorRef> blobs,
            Dictionary<string, AexisGraphSession.BufferShape> shapes,
            string blobName,
            out AexisGraphSession.CmdTensorRef texture,
            out AexisGraphSession.BufferShape shape)
        {
            texture = null;
            shape = default;
            if (blobs == null || shapes == null || string.IsNullOrWhiteSpace(blobName))
                return false;
            if (!blobs.TryGetValue(blobName, out texture) || texture == null || texture.texture == null)
            {
                texture = null;
                return false;
            }

            shape = AexisGraphSession.GetCmdShape(shapes, blobs, blobName);
            return shape.dims == 2
                && shape.w > 0
                && shape.h > 0
                && texture.width == shape.w
                && texture.height == shape.h
                && texture.packs == 1;
        }

        private static RenderTexture MaterializeScalar2DInput(
            AexisGraphSession owner,
            AexisGraphSession.TensorRef source,
            AexisGraphSession.BufferShape logicalShape,
            RenderTextureFormat pack4Format,
            ref RenderTexture materialized)
        {
            if (owner == null)
                throw new ArgumentNullException(nameof(owner));
            if (source == null || source.texture == null)
                throw new ArgumentNullException(nameof(source));

            if (!AexisGraphSession.IsStrictLinearMatTexture(source))
                return source.texture;

            var storageShape = AexisGraphSession.GetTextureStorageShape(source, logicalShape);
            materialized = owner.RentTempArray(logicalShape.w, logicalShape.h, 1, pack4Format);
            owner.Ops.ReshapeLinearMatToPack4(
                source.texture,
                storageShape.w,
                storageShape.h,
                logicalShape.w,
                logicalShape.h,
                1,
                1,
                2,
                materialized);
            return materialized;
        }

        private static ComputeTexture MaterializeScalar2DInput(
            AexisGraphSession owner,
            CommandBuffer cmd,
            AexisGraphSession.CmdTensorRef source,
            AexisGraphSession.BufferShape logicalShape,
            RenderTextureFormat pack4Format,
            ref ComputeTexture materialized)
        {
            if (owner == null)
                throw new ArgumentNullException(nameof(owner));
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if (source == null || source.texture == null)
                throw new ArgumentNullException(nameof(source));

            if (!AexisGraphSession.IsStrictLinearMatTexture(source))
                return source.texture;

            var storageShape = AexisGraphSession.GetCmdStorageShape(source, logicalShape);
            materialized = owner.RentTempArray(cmd, logicalShape.w, logicalShape.h, 1, pack4Format);
            owner.Ops.ReshapeLinearMatToPack4(
                cmd,
                source.texture,
                storageShape.w,
                storageShape.h,
                logicalShape.w,
                logicalShape.h,
                1,
                1,
                2,
                materialized);
            return materialized;
        }

        private static void ReturnTempUnique(AexisGraphSession owner, ref RenderTexture texture, RenderTexture alias0)
        {
            if (texture == null || ReferenceEquals(texture, alias0))
            {
                texture = null;
                return;
            }

            owner.ReturnTempArray(texture);
            texture = null;
        }

        private static void ReturnTempUnique(AexisGraphSession owner, CommandBuffer cmd, ref ComputeTexture texture, ComputeTexture alias0)
        {
            if (texture == null)
                return;
            if (alias0 != null && texture.nameID == alias0.nameID)
            {
                texture = null;
                return;
            }

            owner.ReturnTempArray(cmd, texture);
            texture = null;
        }

        private static bool ShouldPromoteAttentionGemmOutputTexture(AexisGraphSession owner, AexisGraphModel.Layer layer)
        {
            if (owner?.Model?.layers == null || layer?.topNames == null || layer.topNames.Length == 0)
                return false;

            var reshape = FindSingleConsumer(owner.Model, layer.topNames[0]);
            if (reshape == null || reshape.type != AexisLayerTypes.Reshape || reshape.topNames == null || reshape.topNames.Length == 0)
                return false;

            var next = FindSingleConsumer(owner.Model, reshape.topNames[0]);
            return next != null
                && next.type == AexisLayerTypes.Reshape
                && IsParamlessReshape(next);
        }

        private static bool IsParamlessReshape(AexisGraphModel.Layer layer)
        {
            if (layer == null)
                return false;
            if (!string.IsNullOrWhiteSpace(layer.GetString(6, null)))
                return false;
            return layer.GetInt(0, -233) == -233
                && layer.GetInt(1, -233) == -233
                && layer.GetInt(11, -233) == -233
                && layer.GetInt(2, -233) == -233;
        }

        private static AexisGraphModel.Layer FindSingleConsumer(AexisGraphModel model, string blobName)
        {
            if (model?.layers == null || string.IsNullOrWhiteSpace(blobName))
                return null;

            AexisGraphModel.Layer found = null;
            for (var i = 0; i < model.layers.Count; i++)
            {
                var candidate = model.layers[i];
                if (candidate?.bottomNames == null)
                    continue;
                for (var j = 0; j < candidate.bottomNames.Length; j++)
                {
                    if (!string.Equals(candidate.bottomNames[j], blobName, StringComparison.Ordinal))
                        continue;
                    if (found != null)
                        return null;
                    found = candidate;
                    break;
                }
            }

            return found;
        }

        private static bool IsAttentionContextOutputProjection(AexisGraphSession owner, AexisGraphModel.Layer layer)
        {
            if (owner?.Model?.layers == null || layer?.bottomNames == null || layer.bottomNames.Length == 0)
                return false;

            var producer = FindSingleProducer(owner.Model, layer.bottomNames[0]);
            // pnnx aten::to carries dtype/device constants after its data input.
            if (producer?.type == AexisLayerTypes.AtenTo && producer.bottomNames != null && producer.bottomNames.Length >= 1)
                producer = FindSingleProducer(owner.Model, producer.bottomNames[0]);
            if (producer?.type != AexisLayerTypes.Reshape || producer.bottomNames == null || producer.bottomNames.Length != 1)
                return false;

            var contextPermute = FindSingleProducer(owner.Model, producer.bottomNames[0]);
            return contextPermute != null
                && contextPermute.type == AexisLayerTypes.Permute
                && contextPermute.GetInt(0, -1) == 2;
        }

        private static AexisGraphModel.Layer FindSingleProducer(AexisGraphModel model, string blobName)
        {
            if (model?.layers == null || string.IsNullOrWhiteSpace(blobName))
                return null;

            AexisGraphModel.Layer found = null;
            for (var i = 0; i < model.layers.Count; i++)
            {
                var candidate = model.layers[i];
                if (candidate?.topNames == null)
                    continue;
                for (var j = 0; j < candidate.topNames.Length; j++)
                {
                    if (!string.Equals(candidate.topNames[j], blobName, StringComparison.Ordinal))
                        continue;
                    if (found != null)
                        return null;
                    found = candidate;
                    break;
                }
            }

            return found;
        }
    }
}
