using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnGemmLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnGemmLayerRepro() : base(NcnnLayerTypes.Gemm, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var gp = new NcnnRepro.GemmPack
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
                                            throw new InvalidOperationException("Gemm constantA is not supported in NcnnRepro: " + layer.name);
                                        if (gp.constantB)
                                        {
                                            var bw = gp.transB ? gp.constantK : gp.constantN;
                                            var bh = gp.transB ? gp.constantN : gp.constantK;

                                            phaseSw.Restart();
                                            var b = NcnnRepro.ReadClipMatAsFloat32(br, bw, bh, 0, 0, 0);
                                            gp.bDataCpu = b;
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

                                                var c = NcnnRepro.ReadClipMatAsFloat32(br, cw, ch, 0, 0, 0);
                                                gp.cDataCpu = c;
                                            }
                                            phaseSw.Stop();
                                            readMs += phaseSw.ElapsedMilliseconds;

                                            phaseSw.Restart();
                                            gp.bData = NcnnRepro.NewBuffer(gp.bDataCpu);
                                            if (owner.UsesFp16WeightStorage)
                                                gp.bDataFp16 = NcnnRepro.NewFp16Buffer(gp.bDataCpu, "NcnnRepro.GemmWeightFp16:" + layer.name);
                                            if (gp.cDataCpu != null)
                                                gp.cData = NcnnRepro.NewBuffer(gp.cDataCpu);
                                            phaseSw.Stop();
                                            uploadMs += phaseSw.ElapsedMilliseconds;
                                        }

                                        owner._gemm[layer.name] = gp;
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
                                                if (!owner._gemm.TryGetValue(layer.name, out var gp))
                                                    throw new InvalidOperationException("Gemm not found: " + layer.name);

                                                var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                if (srcBuf == null || srcView == null)
                                                    throw new InvalidOperationException("Gemm source not found: " + layer.name);
                                                if (gp.transA)
                                                    throw new InvalidOperationException("Gemm transA is not supported in NcnnRepro: " + layer.name);
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
                                                    var bView = NcnnRepro.TryGetBufferView(layer.bottomNames[1], bufferBlobs, bufferViews);
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
                                                        cCpu = NcnnRepro.ReadFloatBuffer(cBuf);
                                                }

                                                var outTensor = m == 1 && srcView.dims == 1
                                                    ? owner.RentTempTensorBuffer(1, n)
                                                    : owner.RentTempTensorBuffer(2, n, m);
                                                if (owner.ForceCpuGemmAll)
                                                {
                                                    if (!gp.constantB)
                                                        throw new InvalidOperationException("ForceCpuGemmAll currently requires constantB=1: " + layer.name);

                                                    var cpuOut = NcnnRepro.RunGemmCpu(srcBuf, srcView, gp, cCpu);
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

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            owner.Ops.SetFp16GemmWeights(owner.UsesFp16WeightStorage && owner._gemm.TryGetValue(layer.name, out var fp16Gemm) ? fp16Gemm.bDataFp16 : null);
            if (TryExecuteRenderTexturePath(owner, layer, context))
                return;

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            owner.Ops.SetFp16GemmWeights(owner.UsesFp16WeightStorage && owner._gemm.TryGetValue(layer.name, out var fp16Gemm) ? fp16Gemm.bDataFp16 : null);
            if (TryExecuteCommandBufferTexturePath(owner, layer, context))
                return;

            if (!owner._gemm.TryGetValue(layer.name, out var gp))
                throw new InvalidOperationException("Gemm not found: " + layer.name);

            var aShape = NcnnRepro.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            throw new NotSupportedException("CommandBuffer Gemm requires non-transposed A and a verified texture-native constant-B profile"
                + " | layer=" + layer.name
                + " | input=d" + aShape.dims + ":" + aShape.w + "x" + aShape.h + "x" + aShape.d + "x" + aShape.c
                + " | constantB=" + (gp.constantB ? "1" : "0")
                + " | transA=" + (gp.transA ? "1" : "0")
                + " | transB=" + (gp.transB ? "1" : "0")
                + " | rejectedFallback=placeholder-or-buffer-materialization");
        }

        private static bool TryExecuteCommandBufferTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
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

            var srcTex = NcnnRepro.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var srcShape = NcnnRepro.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            if (srcTex == null || srcTex.texture == null)
                return false;
            if (srcShape.dims != 2)
                return false;
            var srcIsStrictLinear = NcnnRepro.IsStrictLinearMatTexture(srcTex);
            var srcIsPack4Linear = NcnnRepro.IsPack4LinearMatTexture(srcTex, srcShape);
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
            var outShape = new NcnnRepro.BufferShape(2, Mathf.Max(1, n), Mathf.Max(1, m), 1, 1);
            var usePack4LinearMat = (srcIsStrictLinear || srcIsPack4Linear) && n % 4 == 0;
            var useStrictLinearMat = srcIsStrictLinear && !usePack4LinearMat;
            var outStorageShape = usePack4LinearMat
                ? NcnnRepro.ResolvePack4LinearMatStorageShape(outShape)
                : useStrictLinearMat
                ? NcnnRepro.ResolveLinearMatStorageShape(outShape)
                : new NcnnRepro.BufferShape(3, outShape.w, outShape.h, 1, 1);
            var outRt = usePack4LinearMat
                ? owner.RentTempArray(
                    context.commandBuffer,
                    outStorageShape.w,
                    outStorageShape.h,
                    1,
                    ShouldPromoteAttentionGemmOutputTexture(owner, layer) ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGBHalf)
                : useStrictLinearMat
                ? owner.RentTempMat(context.commandBuffer, outStorageShape.w, outStorageShape.h, NcnnRepro.ResolveLinearMatTextureFormat())
                : owner.RentTempArray(
                    context.commandBuffer,
                    outShape.w,
                    outShape.h,
                    1,
                    ShouldPromoteAttentionGemmOutputTexture(owner, layer) ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGBHalf);
            if (usePack4LinearMat)
            {
                owner.Ops.Gemm2DPack4LinearTextureA(
                    context.commandBuffer,
                    srcTex.texture,
                    srcIsPack4Linear,
                    gp.bData,
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
            else if (useStrictLinearMat)
            {
                owner.Ops.Gemm2DLinearTextureA(
                    context.commandBuffer,
                    srcTex.texture,
                    gp.bData,
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
                    gp.bData,
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

            context.blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
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
            context.blobs[layer.topNames[0]].layoutKind = NcnnRepro.ResolveRepoVkLayoutKind(outShape, outStorageShape, 1);
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

        private static bool TryExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
            if (!NcnnRepro.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var srcTex, out var srcShape))
                return false;
            if (srcTex == null || srcTex.texture == null)
                return false;
            if (srcShape.dims != 2)
                return false;
            var srcIsStrictLinear = NcnnRepro.IsStrictLinearMatTexture(srcTex);
            var srcIsPack4Linear = NcnnRepro.IsPack4LinearMatTexture(srcTex, srcShape);
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
            var outShape = new NcnnRepro.BufferShape(2, Mathf.Max(1, n), Mathf.Max(1, m), 1, 1);
            var usePack4LinearMat = (srcIsStrictLinear || srcIsPack4Linear) && n % 4 == 0;
            var useStrictLinearMat = srcIsStrictLinear && !usePack4LinearMat;
            var outStorageShape = usePack4LinearMat
                ? NcnnRepro.ResolvePack4LinearMatStorageShape(outShape)
                : useStrictLinearMat
                ? NcnnRepro.ResolveLinearMatStorageShape(outShape)
                : new NcnnRepro.BufferShape(3, outShape.w, outShape.h, 1, 1);
            var outRt = usePack4LinearMat
                ? owner.RentTempArray(
                    outStorageShape.w,
                    outStorageShape.h,
                    1,
                    ShouldPromoteAttentionGemmOutputTexture(owner, layer) ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGBHalf)
                : useStrictLinearMat
                ? owner.RentTempMat(outStorageShape.w, outStorageShape.h, NcnnRepro.ResolveLinearMatTextureFormat())
                : owner.RentTempArray(
                    outShape.w,
                    outShape.h,
                    1,
                    ShouldPromoteAttentionGemmOutputTexture(owner, layer) ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGBHalf);
            if (usePack4LinearMat)
            {
                owner.Ops.Gemm2DPack4LinearTextureA(
                    srcTex.texture,
                    srcIsPack4Linear,
                    gp.bData,
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
            else if (useStrictLinearMat)
            {
                owner.Ops.Gemm2DLinearTextureA(
                    srcTex.texture,
                    gp.bData,
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
                    gp.bData,
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
            NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], outRt, outShape, outStorageShape);
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

        private static bool TryExecuteRenderTextureAttentionProjectionTexturePath(
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnLayerBufferContext context,
            NcnnRepro.GemmPack gp)
        {
            if (owner == null || layer == null || context == null || gp == null || !gp.constantB)
                return false;
            if (layer.bottomNames == null || layer.bottomNames.Length < 1 || layer.topNames == null || layer.topNames.Length < 1)
                return false;
            if (!NcnnRepro.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var srcTex, out var srcShape))
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

            var logicalOutShape = new NcnnRepro.BufferShape(2, Mathf.Max(1, n), Mathf.Max(1, m), 1, 1);
            var useC = gp.constantC && gp.broadcastTypeC != -1 && gp.cData != null;
            var outFormat = ShouldPromoteAttentionGemmOutputTexture(owner, layer)
                ? RenderTextureFormat.ARGBFloat
                : RenderTextureFormat.ARGBHalf;

            if (TryResolveAttentionQkvProjectionSpec(owner, layer, logicalOutShape, out var packedQkvShape))
            {
                var outSlices = Mathf.Max(1, Mathf.CeilToInt(packedQkvShape.c / 4f));
                var outRt = owner.RentTempArray(packedQkvShape.w, packedQkvShape.h, outSlices, outFormat);
                var srcIsStrictLinear = NcnnRepro.IsStrictLinearMatTexture(srcTex);
                var srcIsPack4Linear = NcnnRepro.IsPack4LinearMatTexture(srcTex, srcShape);
                if (srcIsStrictLinear)
                {
                    owner.Ops.Gemm2DAttentionQkvLinearTextureA(
                        srcTex.texture,
                        gp.bData,
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
                        gp.bData,
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
                        gp.bData,
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

                NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], outRt, logicalOutShape, packedQkvShape);
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
                ? NcnnRepro.ResolvePack4LinearMatStorageShape(logicalOutShape)
                : NcnnRepro.ResolveLinearMatStorageShape(logicalOutShape);
            var outMat = usePack4LinearOut
                ? owner.RentTempArray(outStorageShape.w, outStorageShape.h, 1, outFormat)
                : owner.RentTempMat(outStorageShape.w, outStorageShape.h, NcnnRepro.ResolveLinearMatTextureFormat());
            if (usePack4LinearOut)
            {
                owner.Ops.Gemm2DAttentionPack4ToPack4LinearTextureA(
                    srcTex.texture,
                    gp.bData,
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
                    gp.bData,
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
            NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], outMat, logicalOutShape, outStorageShape);
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
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnLayerCommandBufferContext context,
            NcnnRepro.GemmPack gp)
        {
            if (owner == null || layer == null || context == null || gp == null || !gp.constantB)
                return false;
            if (layer.bottomNames == null || layer.bottomNames.Length < 1 || layer.topNames == null || layer.topNames.Length < 1)
                return false;
            if (!context.blobs.TryGetValue(layer.bottomNames[0], out var srcTex) || srcTex == null || srcTex.texture == null)
                return false;

            var srcShape = NcnnRepro.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
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

            var logicalOutShape = new NcnnRepro.BufferShape(2, Mathf.Max(1, n), Mathf.Max(1, m), 1, 1);
            var useC = gp.constantC && gp.broadcastTypeC != -1 && gp.cData != null;
            var outFormat = ShouldPromoteAttentionGemmOutputTexture(owner, layer)
                ? RenderTextureFormat.ARGBFloat
                : RenderTextureFormat.ARGBHalf;

            if (TryResolveAttentionQkvProjectionSpec(owner, layer, logicalOutShape, out var packedQkvShape))
            {
                var outSlices = Mathf.Max(1, Mathf.CeilToInt(packedQkvShape.c / 4f));
                var outArr = owner.RentTempArray(context.commandBuffer, packedQkvShape.w, packedQkvShape.h, outSlices, outFormat);
                var srcIsStrictLinear = NcnnRepro.IsStrictLinearMatTexture(srcTex);
                var srcIsPack4Linear = NcnnRepro.IsPack4LinearMatTexture(srcTex, srcShape);
                if (srcIsStrictLinear)
                {
                    owner.Ops.Gemm2DAttentionQkvLinearTextureA(
                        context.commandBuffer,
                        srcTex.texture,
                        gp.bData,
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
                        gp.bData,
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
                        gp.bData,
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

                context.blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(outArr, logicalOutShape, packedQkvShape, owned: true);
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
                ? NcnnRepro.ResolvePack4LinearMatStorageShape(logicalOutShape)
                : NcnnRepro.ResolveLinearMatStorageShape(logicalOutShape);
            var outMat = usePack4LinearOut
                ? owner.RentTempArray(context.commandBuffer, outStorageShape.w, outStorageShape.h, 1, outFormat)
                : owner.RentTempMat(context.commandBuffer, outStorageShape.w, outStorageShape.h, NcnnRepro.ResolveLinearMatTextureFormat());
            if (usePack4LinearOut)
            {
                owner.Ops.Gemm2DAttentionPack4ToPack4LinearTextureA(
                    context.commandBuffer,
                    srcTex.texture,
                    gp.bData,
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
                    gp.bData,
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
            context.blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(outMat, logicalOutShape, outStorageShape, owned: true);
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
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnRepro.BufferShape logicalOutShape,
            out NcnnRepro.BufferShape packedShape)
        {
            packedShape = default;
            if (owner?.Model?.layers == null || layer?.topNames == null || layer.topNames.Length == 0)
                return false;

            var reshape = FindSingleConsumer(owner.Model, layer.topNames[0]);
            if (reshape == null || reshape.type != NcnnLayerTypes.Reshape || reshape.topNames == null || reshape.topNames.Length == 0)
                return false;

            var reshapeOut = NcnnRepro.ResolveReshapeShape(logicalOutShape, reshape);
            if (reshapeOut.dims != 3 || reshapeOut.d != 1 || reshapeOut.w <= 0 || reshapeOut.h <= 0 || reshapeOut.c <= 0)
                return false;

            var permute = FindSingleConsumer(owner.Model, reshape.topNames[0]);
            if (permute == null || permute.type != NcnnLayerTypes.Permute || permute.topNames == null || permute.topNames.Length == 0 || permute.GetInt(0, -1) != 2)
                return false;

            var permuteOut = NcnnRepro.ResolvePermuteShape(reshapeOut, 3, NcnnRepro.ResolvePermuteAxes(3, 2, permute.name));
            if (permuteOut.dims != 3
                || permuteOut.d != 1
                || permuteOut.w != reshapeOut.w
                || permuteOut.h != reshapeOut.c
                || permuteOut.c != reshapeOut.h)
            {
                return false;
            }

            var sdpa = FindSingleConsumer(owner.Model, permute.topNames[0]);
            if (sdpa == null || sdpa.type != NcnnLayerTypes.SDPA)
                return false;
            if (logicalOutShape.w != permuteOut.w * permuteOut.c || logicalOutShape.h != permuteOut.h)
                return false;

            packedShape = permuteOut;
            return true;
        }

        private static bool TryResolveAttentionPackedLinearMatInput(
            NcnnRepro.TensorRef source,
            NcnnRepro.BufferShape logicalShape,
            out NcnnRepro.BufferShape packedShape)
        {
            packedShape = default;
            if (source == null || source.texture == null || logicalShape.dims != 2 || NcnnRepro.IsStrictLinearMatTexture(source))
                return false;

            var storageShape = NcnnRepro.GetTextureStorageShape(source, logicalShape);
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
            NcnnRepro.CmdTensorRef source,
            NcnnRepro.BufferShape logicalShape,
            out NcnnRepro.BufferShape packedShape)
        {
            packedShape = default;
            if (source == null || source.texture == null || logicalShape.dims != 2 || NcnnRepro.IsStrictLinearMatTexture(source))
                return false;

            var storageShape = NcnnRepro.GetCmdStorageShape(source, logicalShape);
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
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnLayerCommandBufferContext context,
            NcnnRepro.GemmPack gp)
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

            var outShape = new NcnnRepro.BufferShape(2, Mathf.Max(1, n), Mathf.Max(1, m), 1, 1);
            var useStrictLinearOut = NcnnRepro.IsStrictLinearMatTexture(aTex) || NcnnRepro.IsStrictLinearMatTexture(bTex);
            var outStorageShape = useStrictLinearOut
                ? NcnnRepro.ResolveLinearMatStorageShape(outShape)
                : new NcnnRepro.BufferShape(3, outShape.w, outShape.h, 1, 1);
            var outFormat = ShouldPromoteAttentionGemmOutputTexture(owner, layer)
                ? RenderTextureFormat.ARGBFloat
                : RenderTextureFormat.ARGBHalf;
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
                    var outMat = owner.RentTempMat(context.commandBuffer, outStorageShape.w, outStorageShape.h, NcnnRepro.ResolveLinearMatTextureFormat());
                    owner.Ops.ReshapePack4ToLinearMat(context.commandBuffer, outRt, outShape.w, outShape.h, outShape.d, outShape.c, outShape.dims, outMat);
                    context.blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(outMat, outShape, outStorageShape, owned: true);
                }
                else
                {
                    context.blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef
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
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnLayerBufferContext context,
            NcnnRepro.GemmPack gp)
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

            var outShape = new NcnnRepro.BufferShape(2, Mathf.Max(1, n), Mathf.Max(1, m), 1, 1);
            var useStrictLinearOut = NcnnRepro.IsStrictLinearMatTexture(aTex) || NcnnRepro.IsStrictLinearMatTexture(bTex);
            var outStorageShape = useStrictLinearOut
                ? NcnnRepro.ResolveLinearMatStorageShape(outShape)
                : new NcnnRepro.BufferShape(3, outShape.w, outShape.h, 1, 1);
            var outFormat = ShouldPromoteAttentionGemmOutputTexture(owner, layer)
                ? RenderTextureFormat.ARGBFloat
                : RenderTextureFormat.ARGBHalf;
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
                    var outMat = owner.RentTempMat(outStorageShape.w, outStorageShape.h, NcnnRepro.ResolveLinearMatTextureFormat());
                    owner.Ops.ReshapePack4ToLinearMat(outRt, outShape.w, outShape.h, outShape.d, outShape.c, outShape.dims, outMat);
                    NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], outMat, outShape, outStorageShape);
                }
                else
                {
                    NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], outRt, outShape, outStorageShape);
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
            Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            Dictionary<string, NcnnRepro.BufferShape> textureShapes,
            string blobName,
            out NcnnRepro.TensorRef texture,
            out NcnnRepro.BufferShape shape)
        {
            texture = null;
            shape = default;
            if (!NcnnRepro.TryGetExistingTexture(textureBlobs, textureShapes, blobName, out texture, out shape))
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
            Dictionary<string, NcnnRepro.CmdTensorRef> blobs,
            Dictionary<string, NcnnRepro.BufferShape> shapes,
            string blobName,
            out NcnnRepro.CmdTensorRef texture,
            out NcnnRepro.BufferShape shape)
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

            shape = NcnnRepro.GetCmdShape(shapes, blobs, blobName);
            return shape.dims == 2
                && shape.w > 0
                && shape.h > 0
                && texture.width == shape.w
                && texture.height == shape.h
                && texture.packs == 1;
        }

        private static RenderTexture MaterializeScalar2DInput(
            NcnnRepro owner,
            NcnnRepro.TensorRef source,
            NcnnRepro.BufferShape logicalShape,
            RenderTextureFormat pack4Format,
            ref RenderTexture materialized)
        {
            if (owner == null)
                throw new ArgumentNullException(nameof(owner));
            if (source == null || source.texture == null)
                throw new ArgumentNullException(nameof(source));

            if (!NcnnRepro.IsStrictLinearMatTexture(source))
                return source.texture;

            var storageShape = NcnnRepro.GetTextureStorageShape(source, logicalShape);
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
            NcnnRepro owner,
            CommandBuffer cmd,
            NcnnRepro.CmdTensorRef source,
            NcnnRepro.BufferShape logicalShape,
            RenderTextureFormat pack4Format,
            ref ComputeTexture materialized)
        {
            if (owner == null)
                throw new ArgumentNullException(nameof(owner));
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if (source == null || source.texture == null)
                throw new ArgumentNullException(nameof(source));

            if (!NcnnRepro.IsStrictLinearMatTexture(source))
                return source.texture;

            var storageShape = NcnnRepro.GetCmdStorageShape(source, logicalShape);
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

        private static void ReturnTempUnique(NcnnRepro owner, ref RenderTexture texture, RenderTexture alias0)
        {
            if (texture == null || ReferenceEquals(texture, alias0))
            {
                texture = null;
                return;
            }

            owner.ReturnTempArray(texture);
            texture = null;
        }

        private static void ReturnTempUnique(NcnnRepro owner, CommandBuffer cmd, ref ComputeTexture texture, ComputeTexture alias0)
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

        private static bool ShouldPromoteAttentionGemmOutputTexture(NcnnRepro owner, NcnnParamModel.Layer layer)
        {
            if (owner?.Model?.layers == null || layer?.topNames == null || layer.topNames.Length == 0)
                return false;

            var reshape = FindSingleConsumer(owner.Model, layer.topNames[0]);
            if (reshape == null || reshape.type != NcnnLayerTypes.Reshape || reshape.topNames == null || reshape.topNames.Length == 0)
                return false;

            var next = FindSingleConsumer(owner.Model, reshape.topNames[0]);
            return next != null
                && next.type == NcnnLayerTypes.Reshape
                && IsParamlessReshape(next);
        }

        private static bool IsParamlessReshape(NcnnParamModel.Layer layer)
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

        private static NcnnParamModel.Layer FindSingleConsumer(NcnnParamModel model, string blobName)
        {
            if (model?.layers == null || string.IsNullOrWhiteSpace(blobName))
                return null;

            NcnnParamModel.Layer found = null;
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

        private static bool IsAttentionContextOutputProjection(NcnnRepro owner, NcnnParamModel.Layer layer)
        {
            if (owner?.Model?.layers == null || layer?.bottomNames == null || layer.bottomNames.Length == 0)
                return false;

            var producer = FindSingleProducer(owner.Model, layer.bottomNames[0]);
            // pnnx aten::to carries dtype/device constants after its data input.
            if (producer?.type == NcnnLayerTypes.AtenTo && producer.bottomNames != null && producer.bottomNames.Length >= 1)
                producer = FindSingleProducer(owner.Model, producer.bottomNames[0]);
            if (producer?.type != NcnnLayerTypes.Reshape || producer.bottomNames == null || producer.bottomNames.Length != 1)
                return false;

            var contextPermute = FindSingleProducer(owner.Model, producer.bottomNames[0]);
            return contextPermute != null
                && contextPermute.type == NcnnLayerTypes.Permute
                && contextPermute.GetInt(0, -1) == 2;
        }

        private static NcnnParamModel.Layer FindSingleProducer(NcnnParamModel model, string blobName)
        {
            if (model?.layers == null || string.IsNullOrWhiteSpace(blobName))
                return null;

            NcnnParamModel.Layer found = null;
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
