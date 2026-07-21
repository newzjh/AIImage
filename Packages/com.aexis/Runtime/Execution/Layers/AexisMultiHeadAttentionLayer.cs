using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisMultiHeadAttentionLayer : AexisBaseLayer
    {
        public AexisMultiHeadAttentionLayer() : base(AexisLayerTypes.MultiHeadAttention, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var mp = new AexisGraphSession.MultiHeadAttentionPack();
                                        mp.embedDim = layer.GetInt(0, 0);
                                        mp.numHeads = layer.GetInt(1, 1);
                                        mp.weightDataSize = layer.GetInt(2, 0);
                                        mp.kdim = layer.GetInt(3, mp.embedDim);
                                        mp.vdim = layer.GetInt(4, mp.embedDim);
                                        mp.attnMask = layer.GetInt(5, 0) != 0;
                                        mp.kvCache = layer.GetInt(7, 0) != 0;
                                        mp.scale = layer.GetFloat(6, 1f / Mathf.Sqrt(Mathf.Max(1, mp.embedDim / Mathf.Max(1, mp.numHeads))));
                                        mp.qdim = mp.embedDim > 0 ? mp.weightDataSize / Mathf.Max(1, mp.embedDim) : 0;

                                        phaseSw.Restart();
                                        var qW = AexisGraphSession.ReadClipMatAsFloat32(br, mp.embedDim * mp.qdim, 0, 0, 0, 0);
                                        var qB = br.ReadTensorAsFloat32(mp.embedDim, 0, 0, 0, 1);
                                        var kW = AexisGraphSession.ReadClipMatAsFloat32(br, mp.embedDim * mp.kdim, 0, 0, 0, 0);
                                        var kB = br.ReadTensorAsFloat32(mp.embedDim, 0, 0, 0, 1);
                                        var vW = AexisGraphSession.ReadClipMatAsFloat32(br, mp.embedDim * mp.vdim, 0, 0, 0, 0);
                                        var vB = br.ReadTensorAsFloat32(mp.embedDim, 0, 0, 0, 1);
                                        var oW = AexisGraphSession.ReadClipMatAsFloat32(br, mp.qdim * mp.embedDim, 0, 0, 0, 0);
                                        var oB = br.ReadTensorAsFloat32(mp.qdim, 0, 0, 0, 1);
                                        phaseSw.Stop();
                                        readMs += phaseSw.ElapsedMilliseconds;

                                        phaseSw.Restart();
                                        mp.qW = AexisGraphSession.NewBuffer(qW);
                                        mp.qB = AexisGraphSession.NewBuffer(qB);
                                        mp.kW = AexisGraphSession.NewBuffer(kW);
                                        mp.kB = AexisGraphSession.NewBuffer(kB);
                                        mp.vW = AexisGraphSession.NewBuffer(vW);
                                        mp.vB = AexisGraphSession.NewBuffer(vB);
                                        mp.oW = AexisGraphSession.NewBuffer(oW);
                                        mp.oB = AexisGraphSession.NewBuffer(oB);
                                        phaseSw.Stop();
                                        uploadMs += phaseSw.ElapsedMilliseconds;

                                        owner._multiHeadAttention[layer.name] = mp;
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
                                                if (!owner._multiHeadAttention.TryGetValue(layer.name, out var mp))
                                                    throw new InvalidOperationException("MultiHeadAttention not found: " + layer.name);

                                                ResolveBottomBlobIndices(layer.bottomNames?.Length ?? 0, mp.attnMask, mp.kvCache, out var qBlobIndex, out var kBlobIndex, out var vBlobIndex, out var attnMaskIndex);

                                                using var qTensor = owner.GetReadableTensorInput(layer.bottomNames[qBlobIndex], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                using var kTensor = owner.GetReadableTensorInput(layer.bottomNames[kBlobIndex], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                using var vTensor = owner.GetReadableTensorInput(layer.bottomNames[vBlobIndex], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                var qBuf = qTensor?.buffer;
                                                var kBuf = kTensor?.buffer;
                                                var vBuf = vTensor?.buffer;
                                                ComputeBuffer attnMaskBuf = null;
                                                AexisTensorBuffer attnMaskView = null;
                                                if (attnMaskIndex >= 0)
                                                {
                                                    attnMaskBuf = owner.GetOrConvertToBuffer(layer.bottomNames[attnMaskIndex], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                    attnMaskView = AexisGraphSession.TryGetBufferView(layer.bottomNames[attnMaskIndex], bufferBlobs, bufferViews);
                                                    if (attnMaskBuf == null || attnMaskView == null)
                                                        throw new InvalidOperationException("MultiHeadAttention attention mask input not found: " + layer.name);
                                                }
                                                if (qBuf == null)
                                                    throw new InvalidOperationException("MultiHeadAttention q input not found: " + layer.name);
                                                if (kBuf == null)
                                                    throw new InvalidOperationException("MultiHeadAttention k input not found: " + layer.name);
                                                if (vBuf == null)
                                                    throw new InvalidOperationException("MultiHeadAttention v input not found: " + layer.name);

                                                var srcLen = qBuf.count / Mathf.Max(1, mp.qdim);
                                                var dstLen = kBuf.count / Mathf.Max(1, mp.kdim);
                                                var ctx = owner.RentTempBuffer(srcLen * mp.embedDim, sizeof(float));

                                                var canFuseSelfAttentionQkv = owner.EnableMhaQkvFusion
                                                                              && owner.EnableMhaParallelSoftmax
                                                                              && attnMaskBuf == null
                                                                              && ReferenceEquals(qBuf, kBuf)
                                                                              && ReferenceEquals(qBuf, vBuf)
                                                                              && mp.qdim == mp.kdim
                                                                              && mp.qdim == mp.vdim
                                                                              && srcLen == dstLen;

                                                if (canFuseSelfAttentionQkv)
                                                {
                                                    var qkv = owner.RentTempBuffer(srcLen * mp.embedDim * 3, sizeof(float));
                                                    owner.Ops.MhaProjectQkv2D(qBuf, srcLen, mp.qdim, mp.qW, mp.qB, mp.kW, mp.kB, mp.vW, mp.vB, mp.embedDim, qkv);
                                                    owner.Ops.MhaAttentionQkv(qkv, srcLen, mp.embedDim, mp.numHeads, mp.scale, ctx);
                                                    tempOwned.Add(qkv);
                                                }
                                                else
                                                {
                                                    var qAff = owner.RentTempBuffer(srcLen * mp.embedDim, sizeof(float));
                                                    var kAff = owner.RentTempBuffer(dstLen * mp.embedDim, sizeof(float));
                                                    var vAff = owner.RentTempBuffer(dstLen * mp.embedDim, sizeof(float));
                                                    owner.Ops.InnerProduct2D(qBuf, srcLen, mp.qdim, mp.qW, mp.qB, mp.embedDim, qAff);
                                                    owner.Ops.InnerProduct2D(kBuf, dstLen, mp.kdim, mp.kW, mp.kB, mp.embedDim, kAff);
                                                    owner.Ops.InnerProduct2D(vBuf, dstLen, mp.vdim, mp.vW, mp.vB, mp.embedDim, vAff);

                                                    var qScaled = owner.RentTempBuffer(srcLen * mp.embedDim, sizeof(float));
                                                    owner.Ops.BinaryOpScalarBuf(qAff, mp.scale, qAff.count, 2, qScaled);
                                                    owner.Ops.MhaAttention(
                                                        qScaled,
                                                        kAff,
                                                        vAff,
                                                        attnMaskBuf,
                                                        srcLen,
                                                        dstLen,
                                                        mp.embedDim,
                                                        mp.numHeads,
                                                        1f,
                                                        attnMaskView?.dims ?? 0,
                                                        attnMaskView?.w ?? 0,
                                                        attnMaskView?.h ?? 0,
                                                        attnMaskView?.c ?? 0,
                                                        ctx,
                                                        owner.EnableMhaParallelSoftmax);

                                                    tempOwned.Add(qAff);
                                                    tempOwned.Add(kAff);
                                                    tempOwned.Add(vAff);
                                                    tempOwned.Add(qScaled);
                                                }

                                                var outTensor = owner.RentTempTensorBuffer(2, mp.qdim, srcLen);
                                                owner.Ops.InnerProduct2D(ctx, srcLen, mp.embedDim, mp.oW, mp.oB, mp.qdim, outTensor.buffer);

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
                                                tempOwned.Add(ctx);
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

            if (!owner._multiHeadAttention.TryGetValue(layer.name, out var mp))
                throw new InvalidOperationException("MultiHeadAttention not found: " + layer.name);

            var qShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            if (mp.kvCache)
                throw new NotSupportedException("CommandBuffer MultiHeadAttention kv-cache is not implemented"
                    + " | layer=" + layer.name
                    + " | rejectedFallback=buffer-materialization");
            throw new NotSupportedException("CommandBuffer MultiHeadAttention requires verified LinearMat Q/K/V and an optional Pack4 scalar mask"
                + " | layer=" + layer.name
                + " | query=d" + qShape.dims + ":" + qShape.w + "x" + qShape.h + "x" + qShape.d + "x" + qShape.c
                + " | rejectedFallback=placeholder-or-buffer-materialization");
        }

        private static void ResolveBottomBlobIndices(int bottomBlobCount, bool attnMask, bool kvCache, out int qBlobIndex, out int kBlobIndex, out int vBlobIndex, out int attnMaskIndex)
        {
            qBlobIndex = 0;
            kBlobIndex = 0;
            vBlobIndex = 0;
            attnMaskIndex = -1;

            if (kvCache)
                throw new InvalidOperationException("MultiHeadAttention kv_cache path is not implemented in repro.");

            if (attnMask)
            {
                switch (bottomBlobCount)
                {
                    case 2:
                        attnMaskIndex = 1;
                        return;
                    case 3:
                        kBlobIndex = 1;
                        vBlobIndex = 1;
                        attnMaskIndex = 2;
                        return;
                    case 4:
                        kBlobIndex = 1;
                        vBlobIndex = 2;
                        attnMaskIndex = 3;
                        return;
                    default:
                        throw new InvalidOperationException("Unsupported MultiHeadAttention bottom count with attn_mask: " + bottomBlobCount);
                }
            }

            switch (bottomBlobCount)
            {
                case 1:
                    return;
                case 2:
                    kBlobIndex = 1;
                    vBlobIndex = 1;
                    return;
                case 3:
                    kBlobIndex = 1;
                    vBlobIndex = 2;
                    return;
                default:
                    throw new InvalidOperationException("Unsupported MultiHeadAttention bottom count: " + bottomBlobCount);
            }
        }

        private static bool TryExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (owner == null || layer == null || context == null)
                return false;
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;

            // MHA weights stay FP32. Do not inherit the preceding InnerProduct/Gemm FP16 upload.
            owner.Ops.SetFp16GemmWeights(null);
            owner.Ops.SetInt8GemmWeights(null, null);
            owner.Ops.SetInt4GemmWeights(null, null);
            if (!TryResolveRtOnlyPlan(owner, layer, context.textureBlobs, context.textureShapes, out var plan))
                return false;

            RenderTexture qProj = null;
            RenderTexture kProj = null;
            RenderTexture vProj = null;
            RenderTexture qScaled = null;
            RenderTexture qHead = null;
            RenderTexture kHead = null;
            RenderTexture vHead = null;
            RenderTexture kHeadT = null;
            RenderTexture scores = null;
            RenderTexture attnMaskTiled = null;
            RenderTexture scoresBiased = null;
            RenderTexture weights = null;
            RenderTexture contextHeads = null;
            RenderTexture contextPermuted = null;
            RenderTexture contextFlat = null;
            RenderTexture output = null;
            RenderTexture outputPacked = null;

            try
            {
                RunRtOnlyPass(
                    owner,
                    plan,
                    ref qProj,
                    ref kProj,
                    ref vProj,
                    ref qScaled,
                    ref qHead,
                    ref kHead,
                    ref vHead,
                    ref kHeadT,
                    ref scores,
                    ref attnMaskTiled,
                    ref scoresBiased,
                    ref weights,
                    ref contextHeads,
                    ref contextPermuted,
                    ref contextFlat,
                    ref output,
                    ref outputPacked);

                if (owner.UseLegacyPack4AttentionLayout || owner.PreserveLegacyFp32Execution)
                {
                    AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, plan.outputLogicalShape, plan.outputScalarStorageShape);
                    output = null;
                }
                else
                {
                    AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], outputPacked, plan.outputLogicalShape, plan.outputStorageShape);
                    outputPacked = null;
                }
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
            finally
            {
                ReturnTemp(owner, ref qProj);
                ReturnTemp(owner, ref kProj);
                ReturnTemp(owner, ref vProj);
                ReturnTemp(owner, ref qScaled);
                ReturnTemp(owner, ref qHead);
                ReturnTemp(owner, ref kHead);
                ReturnTemp(owner, ref vHead);
                ReturnTemp(owner, ref kHeadT);
                ReturnTemp(owner, ref scores);
                ReturnTemp(owner, ref attnMaskTiled);
                ReturnTemp(owner, ref scoresBiased);
                ReturnTemp(owner, ref weights);
                ReturnTemp(owner, ref contextHeads);
                ReturnTemp(owner, ref contextPermuted);
                ReturnTemp(owner, ref contextFlat);
                ReturnTemp(owner, ref output);
                ReturnTemp(owner, ref outputPacked);
            }
        }

        private static bool TryExecuteCommandBufferTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            if (owner == null || layer == null || context == null)
                return false;
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;

            // MHA weights stay FP32. Do not inherit the preceding InnerProduct/Gemm FP16 upload.
            owner.Ops.SetFp16GemmWeights(null);
            owner.Ops.SetInt8GemmWeights(null, null);
            owner.Ops.SetInt4GemmWeights(null, null);
            if (!TryResolveCmdRtOnlyPlan(owner, layer, context.blobs, context.shapes, out var plan))
                return false;

            var cmd = context.commandBuffer;
            ComputeTexture qProj = null;
            ComputeTexture kProj = null;
            ComputeTexture vProj = null;
            ComputeTexture qScaled = null;
            ComputeTexture qHead = null;
            ComputeTexture kHead = null;
            ComputeTexture vHead = null;
            ComputeTexture kHeadT = null;
            ComputeTexture scores = null;
            ComputeTexture attnMaskTiled = null;
            ComputeTexture scoresBiased = null;
            ComputeTexture weights = null;
            ComputeTexture contextHeads = null;
            ComputeTexture contextPermuted = null;
            ComputeTexture contextFlat = null;
            ComputeTexture output = null;
            ComputeTexture outputPacked = null;

            try
            {
                qProj = owner.RentTempArray(cmd, plan.scalarStorageShape.w, plan.scalarStorageShape.h, 1, plan.scalarTextureFormat);
                kProj = owner.RentTempArray(cmd, plan.scalarStorageShape.w, plan.scalarStorageShape.h, 1, plan.scalarTextureFormat);
                vProj = owner.RentTempArray(cmd, plan.scalarStorageShape.w, plan.scalarStorageShape.h, 1, plan.scalarTextureFormat);
                qScaled = owner.RentTempArray(cmd, plan.scalarStorageShape.w, plan.scalarStorageShape.h, 1, plan.scalarTextureFormat);
                qHead = owner.RentTempArray(cmd, plan.headStorageShape.w, plan.headStorageShape.h, plan.headSlices, plan.pack4TextureFormat);
                kHead = owner.RentTempArray(cmd, plan.headStorageShape.w, plan.headStorageShape.h, plan.headSlices, plan.pack4TextureFormat);
                vHead = owner.RentTempArray(cmd, plan.headStorageShape.w, plan.headStorageShape.h, plan.headSlices, plan.pack4TextureFormat);
                kHeadT = owner.RentTempArray(cmd, plan.keyTransposedStorageShape.w, plan.keyTransposedStorageShape.h, plan.keyTransposedSlices, plan.pack4TextureFormat);
                scores = owner.RentTempArray(cmd, plan.scoresStorageShape.w, plan.scoresStorageShape.h, plan.scoresSlices, plan.pack4TextureFormat);
                if (plan.hasAttnMask)
                {
                    attnMaskTiled = owner.RentTempArray(cmd, plan.scoresStorageShape.w, plan.scoresStorageShape.h, plan.scoresSlices, plan.pack4TextureFormat);
                    scoresBiased = owner.RentTempArray(cmd, plan.scoresStorageShape.w, plan.scoresStorageShape.h, plan.scoresSlices, plan.pack4TextureFormat);
                }
                weights = owner.RentTempArray(cmd, plan.scoresStorageShape.w, plan.scoresStorageShape.h, plan.scoresSlices, plan.pack4TextureFormat);
                contextHeads = owner.RentTempArray(cmd, plan.headStorageShape.w, plan.headStorageShape.h, plan.headSlices, plan.pack4TextureFormat);
                contextPermuted = owner.RentTempArray(cmd, plan.contextPermutedStorageShape.w, plan.contextPermutedStorageShape.h, plan.contextPermutedSlices, plan.pack4TextureFormat);
                contextFlat = owner.RentTempArray(cmd, plan.scalarStorageShape.w, plan.scalarStorageShape.h, 1, plan.scalarTextureFormat);
                output = owner.RentTempArray(cmd, plan.outputScalarStorageShape.w, plan.outputScalarStorageShape.h, 1, plan.scalarTextureFormat);
                outputPacked = owner.RentTempArray(cmd, plan.outputStorageShape.w, plan.outputStorageShape.h, 1, plan.scalarTextureFormat);

                ComputeTexture qScalarInput = null;
                ComputeTexture kScalarInput = null;
                ComputeTexture vScalarInput = null;
                ComputeTexture qScalarMaterialized = null;
                ComputeTexture kScalarMaterialized = null;
                ComputeTexture vScalarMaterialized = null;
                try
                {
                    qScalarInput = MaterializeScalar2DArrayInput(owner, cmd, plan.qTex, plan.qShape, plan.scalarTextureFormat, ref qScalarMaterialized);
                    if (plan.kTex.texture != null
                        && plan.qTex.texture != null
                        && plan.kTex.texture.nameID == plan.qTex.texture.nameID
                        && RequiresScalar2DMaterialization(plan.kTex, plan.kShape))
                    {
                        kScalarInput = qScalarInput;
                        kScalarMaterialized = qScalarMaterialized;
                    }
                    else
                    {
                        kScalarInput = MaterializeScalar2DArrayInput(owner, cmd, plan.kTex, plan.kShape, plan.scalarTextureFormat, ref kScalarMaterialized);
                    }

                    if (plan.vTex.texture != null
                        && plan.qTex.texture != null
                        && plan.vTex.texture.nameID == plan.qTex.texture.nameID
                        && RequiresScalar2DMaterialization(plan.vTex, plan.vShape))
                    {
                        vScalarInput = qScalarInput;
                        vScalarMaterialized = qScalarMaterialized;
                    }
                    else if (plan.vTex.texture != null
                             && plan.kTex.texture != null
                             && plan.vTex.texture.nameID == plan.kTex.texture.nameID
                             && RequiresScalar2DMaterialization(plan.vTex, plan.vShape))
                    {
                        vScalarInput = kScalarInput;
                        vScalarMaterialized = kScalarMaterialized;
                    }
                    else
                    {
                        vScalarInput = MaterializeScalar2DArrayInput(owner, cmd, plan.vTex, plan.vShape, plan.scalarTextureFormat, ref vScalarMaterialized);
                    }

                    owner.Ops.Gemm2DTextureA(cmd, qScalarInput, plan.pack.qW, plan.pack.qB, plan.rows, plan.embedDim, plan.qShape.w, transB: true, alpha: 1f, beta: 1f, useC: true, broadcastTypeC: 4, qProj);
                    owner.Ops.Gemm2DTextureA(cmd, kScalarInput, plan.pack.kW, plan.pack.kB, plan.rows, plan.embedDim, plan.kShape.w, transB: true, alpha: 1f, beta: 1f, useC: true, broadcastTypeC: 4, kProj);
                    owner.Ops.Gemm2DTextureA(cmd, vScalarInput, plan.pack.vW, plan.pack.vB, plan.rows, plan.embedDim, plan.vShape.w, transB: true, alpha: 1f, beta: 1f, useC: true, broadcastTypeC: 4, vProj);
                }
                finally
                {
                    ReturnTempUnique(owner, cmd, ref vScalarMaterialized, qScalarMaterialized, kScalarMaterialized);
                    ReturnTempUnique(owner, cmd, ref kScalarMaterialized, qScalarMaterialized, null);
                    ReturnTemp(owner, cmd, ref qScalarMaterialized);
                }

                owner.Ops.BinaryOpScalarPack4(cmd, qProj, plan.pack.scale, 1, 2, qScaled);

                owner.Ops.AttentionReshapePack4(cmd, qScaled, plan.embedDim, plan.rows, 1, plan.headDim, 1, plan.numHeads, qHead);
                owner.Ops.AttentionReshapePack4(cmd, kProj, plan.embedDim, plan.rows, 1, plan.headDim, 1, plan.numHeads, kHead);
                owner.Ops.AttentionReshapePack4(cmd, vProj, plan.embedDim, plan.rows, 1, plan.headDim, 1, plan.numHeads, vHead);

                owner.Ops.PermutePack4Cdhw(
                    cmd,
                    kHead,
                    plan.headShape.w,
                    plan.headShape.h,
                    plan.headShape.d,
                    plan.headShape.c,
                    new Vector4Int(1, 0, 2, 3),
                    plan.keyTransposedShape.w,
                    plan.keyTransposedShape.h,
                    plan.keyTransposedShape.d,
                    plan.keyTransposedShape.c,
                    kHeadT);

                owner.Ops.MatMulPack4Cdhw(
                    cmd,
                    qHead,
                    plan.headShape.h,
                    plan.headShape.w,
                    plan.headShape.d,
                    plan.headShape.c,
                    kHeadT,
                    plan.keyTransposedShape.h,
                    plan.keyTransposedShape.w,
                    plan.keyTransposedShape.d,
                    plan.keyTransposedShape.c,
                    false,
                    plan.scoresShape.d,
                    plan.scoresShape.c,
                    scores);

                if (plan.hasAttnMask)
                {
                    ComputeTexture attnMaskScalarInput = null;
                    ComputeTexture attnMaskScalarMaterialized = null;
                    try
                    {
                        attnMaskScalarInput = MaterializeScalar2DArrayInput(owner, cmd, plan.attnMaskTex, plan.attnMaskShape, plan.scalarTextureFormat, ref attnMaskScalarMaterialized);
                        owner.Ops.TilePack4(cmd, attnMaskScalarInput, plan.attnMaskShape, plan.scoresShape, ResolveAttentionMaskTileRepeats(plan.scoresShape), attnMaskTiled);
                        owner.Ops.BinaryOpPack4(cmd, scores, attnMaskTiled, plan.scoresSlices, 0, scoresBiased);
                    }
                    finally
                    {
                        ReturnTemp(owner, cmd, ref attnMaskScalarMaterialized);
                    }

                    owner.Ops.SoftmaxPack4Cdhw(cmd, scoresBiased, plan.scoresShape.w, plan.scoresShape.h, plan.scoresShape.d, plan.scoresShape.c, weights);
                }
                else
                {
                    owner.Ops.SoftmaxPack4Cdhw(cmd, scores, plan.scoresShape.w, plan.scoresShape.h, plan.scoresShape.d, plan.scoresShape.c, weights);
                }

                owner.Ops.MatMulPack4Cdhw(
                    cmd,
                    weights,
                    plan.scoresShape.h,
                    plan.scoresShape.w,
                    plan.scoresShape.d,
                    plan.scoresShape.c,
                    vHead,
                    plan.headShape.h,
                    plan.headShape.w,
                    plan.headShape.d,
                    plan.headShape.c,
                    false,
                    plan.headShape.d,
                    plan.headShape.c,
                    contextHeads);

                owner.Ops.PermutePack4Cdhw(
                    cmd,
                    contextHeads,
                    plan.headShape.w,
                    plan.headShape.h,
                    plan.headShape.d,
                    plan.headShape.c,
                    new Vector4Int(3, 1, 2, 0),
                    plan.contextPermutedShape.w,
                    plan.contextPermutedShape.h,
                    plan.contextPermutedShape.d,
                    plan.contextPermutedShape.c,
                    contextPermuted);

                owner.Ops.AttentionContextFlattenPack4(
                    cmd,
                    contextPermuted,
                    plan.contextPermutedShape.w,
                    plan.contextPermutedShape.h,
                    plan.contextPermutedShape.d,
                    plan.contextPermutedShape.c,
                    plan.contextFlattenOutChannels,
                    2,
                    contextFlat);

                owner.Ops.Gemm2DTextureA(
                    cmd,
                    contextFlat,
                    plan.pack.oW,
                    plan.pack.oB,
                    plan.rows,
                    plan.pack.qdim,
                    plan.embedDim,
                    transB: true,
                    alpha: 1f,
                    beta: 1f,
                    useC: true,
                    broadcastTypeC: 4,
                    output);
                owner.Ops.Pack4LinearFromScalar2D(cmd, output, plan.outputLogicalShape.w, plan.outputLogicalShape.h, outputPacked);

                var useLegacyAttentionLayout = owner.UseLegacyPack4AttentionLayout || owner.PreserveLegacyFp32Execution;
                var outputTexture = useLegacyAttentionLayout ? output : outputPacked;
                var outputStorageShape = useLegacyAttentionLayout ? plan.outputScalarStorageShape : plan.outputStorageShape;
                context.blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
                {
                    texture = outputTexture,
                    width = outputStorageShape.w,
                    height = outputStorageShape.h,
                    packs = 1,
                    refs = 1,
                    owned = true,
                    hasLogicalShape = true,
                    logicalShape = plan.outputLogicalShape,
                    hasStorageShape = true,
                    storageShape = outputStorageShape
                };
                context.shapes[layer.topNames[0]] = plan.outputLogicalShape;
                if (useLegacyAttentionLayout)
                    output = null;
                else
                    outputPacked = null;
                owner.ConsumeCmd(cmd, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
                return true;
            }
            finally
            {
                ReturnTemp(owner, cmd, ref qProj);
                ReturnTemp(owner, cmd, ref kProj);
                ReturnTemp(owner, cmd, ref vProj);
                ReturnTemp(owner, cmd, ref qScaled);
                ReturnTemp(owner, cmd, ref qHead);
                ReturnTemp(owner, cmd, ref kHead);
                ReturnTemp(owner, cmd, ref vHead);
                ReturnTemp(owner, cmd, ref kHeadT);
                ReturnTemp(owner, cmd, ref scores);
                ReturnTemp(owner, cmd, ref attnMaskTiled);
                ReturnTemp(owner, cmd, ref scoresBiased);
                ReturnTemp(owner, cmd, ref weights);
                ReturnTemp(owner, cmd, ref contextHeads);
                ReturnTemp(owner, cmd, ref contextPermuted);
                ReturnTemp(owner, cmd, ref contextFlat);
                ReturnTemp(owner, cmd, ref output);
                ReturnTemp(owner, cmd, ref outputPacked);
            }
        }

        private static bool TryResolveRtOnlyPlan(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            out MultiHeadAttentionRtPlan plan)
        {
            plan = default;
            if (owner == null || layer == null || textureBlobs == null || textureShapes == null)
                return false;
            if (!owner._multiHeadAttention.TryGetValue(layer.name, out var pack) || pack == null)
                return false;
            if (pack.kvCache)
                return false;
            if ((pack.embedDim % Mathf.Max(1, pack.numHeads)) != 0)
                return false;

            ResolveBottomBlobIndices(layer.bottomNames?.Length ?? 0, pack.attnMask, pack.kvCache, out var qBlobIndex, out var kBlobIndex, out var vBlobIndex, out var attnMaskIndex);
            if (!TryGetScalar2DTexture(textureBlobs, textureShapes, layer.bottomNames[qBlobIndex], out var qTex, out var qShape)
                || !TryGetScalar2DTexture(textureBlobs, textureShapes, layer.bottomNames[kBlobIndex], out var kTex, out var kShape)
                || !TryGetScalar2DTexture(textureBlobs, textureShapes, layer.bottomNames[vBlobIndex], out var vTex, out var vShape))
            {
                return false;
            }

            var rows = qShape.h;
            if (rows <= 0 || qShape.w != pack.qdim || kShape.w != pack.kdim || vShape.w != pack.vdim)
                return false;
            if (kShape.h != rows || vShape.h != rows)
                return false;
            AexisGraphSession.TensorRef attnMaskTex = null;
            AexisGraphSession.BufferShape attnMaskShape = default;
            if (attnMaskIndex >= 0)
            {
                if (!TryGetScalar2DTexture(textureBlobs, textureShapes, layer.bottomNames[attnMaskIndex], out attnMaskTex, out attnMaskShape))
                    return false;
                if (attnMaskShape.w != kShape.h || attnMaskShape.h != rows)
                    return false;
            }
            if (pack.qW == null || pack.qB == null || pack.kW == null || pack.kB == null || pack.vW == null || pack.vB == null || pack.oW == null || pack.oB == null)
                return false;

            plan = new MultiHeadAttentionRtPlan(
                pack,
                qTex,
                qShape,
                kTex,
                kShape,
                vTex,
                vShape,
                rows,
                attnMaskTex,
                attnMaskShape,
                owner.ResolveSensitiveOutputTextureFormat());
            return true;
        }

        private static bool TryResolveCmdRtOnlyPlan(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            Dictionary<string, AexisGraphSession.CmdTensorRef> blobs,
            Dictionary<string, AexisGraphSession.BufferShape> shapes,
            out MultiHeadAttentionCmdRtPlan plan)
        {
            plan = default;
            if (owner == null || layer == null || blobs == null || shapes == null)
                return false;
            if (!owner._multiHeadAttention.TryGetValue(layer.name, out var pack) || pack == null)
                return false;
            if (pack.kvCache)
                return false;
            if ((pack.embedDim % Mathf.Max(1, pack.numHeads)) != 0)
                return false;

            ResolveBottomBlobIndices(layer.bottomNames?.Length ?? 0, pack.attnMask, pack.kvCache, out var qBlobIndex, out var kBlobIndex, out var vBlobIndex, out var attnMaskIndex);
            if (!TryGetScalar2DTexture(blobs, shapes, layer.bottomNames[qBlobIndex], out var qTex, out var qShape)
                || !TryGetScalar2DTexture(blobs, shapes, layer.bottomNames[kBlobIndex], out var kTex, out var kShape)
                || !TryGetScalar2DTexture(blobs, shapes, layer.bottomNames[vBlobIndex], out var vTex, out var vShape))
            {
                return false;
            }

            var rows = qShape.h;
            if (rows <= 0 || qShape.w != pack.qdim || kShape.w != pack.kdim || vShape.w != pack.vdim)
                return false;
            if (kShape.h != rows || vShape.h != rows)
                return false;
            AexisGraphSession.CmdTensorRef attnMaskTex = null;
            AexisGraphSession.BufferShape attnMaskShape = default;
            if (attnMaskIndex >= 0)
            {
                if (!TryGetScalar2DTexture(blobs, shapes, layer.bottomNames[attnMaskIndex], out attnMaskTex, out attnMaskShape))
                    return false;
                if (attnMaskShape.w != kShape.h || attnMaskShape.h != rows)
                    return false;
            }
            if (pack.qW == null || pack.qB == null || pack.kW == null || pack.kB == null || pack.vW == null || pack.vB == null || pack.oW == null || pack.oB == null)
                return false;

            plan = new MultiHeadAttentionCmdRtPlan(
                pack,
                qTex,
                qShape,
                kTex,
                kShape,
                vTex,
                vShape,
                rows,
                attnMaskTex,
                attnMaskShape,
                owner.ResolveSensitiveOutputTextureFormat());
            return true;
        }

        private static void RunRtOnlyPass(
            AexisGraphSession owner,
            in MultiHeadAttentionRtPlan plan,
            ref RenderTexture qProj,
            ref RenderTexture kProj,
            ref RenderTexture vProj,
            ref RenderTexture qScaled,
            ref RenderTexture qHead,
            ref RenderTexture kHead,
            ref RenderTexture vHead,
            ref RenderTexture kHeadT,
            ref RenderTexture scores,
            ref RenderTexture attnMaskTiled,
            ref RenderTexture scoresBiased,
            ref RenderTexture weights,
            ref RenderTexture contextHeads,
            ref RenderTexture contextPermuted,
            ref RenderTexture contextFlat,
            ref RenderTexture output,
            ref RenderTexture outputPacked)
        {
            qProj = owner.RentTempArray(plan.scalarStorageShape.w, plan.scalarStorageShape.h, 1, plan.scalarTextureFormat);
            kProj = owner.RentTempArray(plan.scalarStorageShape.w, plan.scalarStorageShape.h, 1, plan.scalarTextureFormat);
            vProj = owner.RentTempArray(plan.scalarStorageShape.w, plan.scalarStorageShape.h, 1, plan.scalarTextureFormat);
            qScaled = owner.RentTempArray(plan.scalarStorageShape.w, plan.scalarStorageShape.h, 1, plan.scalarTextureFormat);
            qHead = owner.RentTempArray(plan.headStorageShape.w, plan.headStorageShape.h, plan.headSlices, plan.pack4TextureFormat);
            kHead = owner.RentTempArray(plan.headStorageShape.w, plan.headStorageShape.h, plan.headSlices, plan.pack4TextureFormat);
            vHead = owner.RentTempArray(plan.headStorageShape.w, plan.headStorageShape.h, plan.headSlices, plan.pack4TextureFormat);
            kHeadT = owner.RentTempArray(plan.keyTransposedStorageShape.w, plan.keyTransposedStorageShape.h, plan.keyTransposedSlices, plan.pack4TextureFormat);
            scores = owner.RentTempArray(plan.scoresStorageShape.w, plan.scoresStorageShape.h, plan.scoresSlices, plan.pack4TextureFormat);
            if (plan.hasAttnMask)
            {
                attnMaskTiled = owner.RentTempArray(plan.scoresStorageShape.w, plan.scoresStorageShape.h, plan.scoresSlices, plan.pack4TextureFormat);
                scoresBiased = owner.RentTempArray(plan.scoresStorageShape.w, plan.scoresStorageShape.h, plan.scoresSlices, plan.pack4TextureFormat);
            }
            weights = owner.RentTempArray(plan.scoresStorageShape.w, plan.scoresStorageShape.h, plan.scoresSlices, plan.pack4TextureFormat);
            contextHeads = owner.RentTempArray(plan.headStorageShape.w, plan.headStorageShape.h, plan.headSlices, plan.pack4TextureFormat);
            contextPermuted = owner.RentTempArray(plan.contextPermutedStorageShape.w, plan.contextPermutedStorageShape.h, plan.contextPermutedSlices, plan.pack4TextureFormat);
            contextFlat = owner.RentTempArray(plan.scalarStorageShape.w, plan.scalarStorageShape.h, 1, plan.scalarTextureFormat);
            output = owner.RentTempArray(plan.outputScalarStorageShape.w, plan.outputScalarStorageShape.h, 1, plan.scalarTextureFormat);
            outputPacked = owner.RentTempArray(plan.outputStorageShape.w, plan.outputStorageShape.h, 1, plan.scalarTextureFormat);

            RenderTexture qScalarInput = null;
            RenderTexture kScalarInput = null;
            RenderTexture vScalarInput = null;
            RenderTexture qScalarMaterialized = null;
            RenderTexture kScalarMaterialized = null;
            RenderTexture vScalarMaterialized = null;
            try
            {
                qScalarInput = MaterializeScalar2DArrayInput(owner, plan.qTex, plan.qShape, plan.scalarTextureFormat, ref qScalarMaterialized);
                if (ReferenceEquals(plan.kTex.texture, plan.qTex.texture)
                    && RequiresScalar2DMaterialization(plan.kTex, plan.kShape))
                {
                    kScalarInput = qScalarInput;
                    kScalarMaterialized = qScalarMaterialized;
                }
                else
                {
                    kScalarInput = MaterializeScalar2DArrayInput(owner, plan.kTex, plan.kShape, plan.scalarTextureFormat, ref kScalarMaterialized);
                }

                if (ReferenceEquals(plan.vTex.texture, plan.qTex.texture)
                    && RequiresScalar2DMaterialization(plan.vTex, plan.vShape))
                {
                    vScalarInput = qScalarInput;
                    vScalarMaterialized = qScalarMaterialized;
                }
                else if (ReferenceEquals(plan.vTex.texture, plan.kTex.texture)
                         && RequiresScalar2DMaterialization(plan.vTex, plan.vShape))
                {
                    vScalarInput = kScalarInput;
                    vScalarMaterialized = kScalarMaterialized;
                }
                else
                {
                    vScalarInput = MaterializeScalar2DArrayInput(owner, plan.vTex, plan.vShape, plan.scalarTextureFormat, ref vScalarMaterialized);
                }

                owner.Ops.Gemm2DTextureA(qScalarInput, plan.pack.qW, plan.pack.qB, plan.rows, plan.embedDim, plan.qShape.w, transB: true, alpha: 1f, beta: 1f, useC: true, broadcastTypeC: 4, qProj);
                owner.Ops.Gemm2DTextureA(kScalarInput, plan.pack.kW, plan.pack.kB, plan.rows, plan.embedDim, plan.kShape.w, transB: true, alpha: 1f, beta: 1f, useC: true, broadcastTypeC: 4, kProj);
                owner.Ops.Gemm2DTextureA(vScalarInput, plan.pack.vW, plan.pack.vB, plan.rows, plan.embedDim, plan.vShape.w, transB: true, alpha: 1f, beta: 1f, useC: true, broadcastTypeC: 4, vProj);
            }
            finally
            {
                ReturnTempUnique(owner, ref vScalarMaterialized, qScalarMaterialized, kScalarMaterialized);
                ReturnTempUnique(owner, ref kScalarMaterialized, qScalarMaterialized, null);
                ReturnTemp(owner, ref qScalarMaterialized);
            }

            owner.Ops.BinaryOpScalarPack4(qProj, plan.pack.scale, 1, 2, qScaled);

            owner.Ops.AttentionReshapePack4(qScaled, plan.embedDim, plan.rows, 1, plan.headDim, 1, plan.numHeads, qHead);
            owner.Ops.AttentionReshapePack4(kProj, plan.embedDim, plan.rows, 1, plan.headDim, 1, plan.numHeads, kHead);
            owner.Ops.AttentionReshapePack4(vProj, plan.embedDim, plan.rows, 1, plan.headDim, 1, plan.numHeads, vHead);

            owner.Ops.PermutePack4Cdhw(
                kHead,
                plan.headShape.w,
                plan.headShape.h,
                plan.headShape.d,
                plan.headShape.c,
                new Vector4Int(1, 0, 2, 3),
                plan.keyTransposedShape.w,
                plan.keyTransposedShape.h,
                plan.keyTransposedShape.d,
                plan.keyTransposedShape.c,
                kHeadT);

            owner.Ops.MatMulPack4Cdhw(
                qHead,
                plan.headShape.h,
                plan.headShape.w,
                plan.headShape.d,
                plan.headShape.c,
                kHeadT,
                plan.keyTransposedShape.h,
                plan.keyTransposedShape.w,
                plan.keyTransposedShape.d,
                plan.keyTransposedShape.c,
                false,
                plan.scoresShape.d,
                plan.scoresShape.c,
                scores);

            if (plan.hasAttnMask)
            {
                RenderTexture attnMaskScalarInput = null;
                RenderTexture attnMaskScalarMaterialized = null;
                try
                {
                    attnMaskScalarInput = MaterializeScalar2DArrayInput(owner, plan.attnMaskTex, plan.attnMaskShape, plan.scalarTextureFormat, ref attnMaskScalarMaterialized);
                    owner.Ops.TilePack4(attnMaskScalarInput, plan.attnMaskShape, plan.scoresShape, ResolveAttentionMaskTileRepeats(plan.scoresShape), attnMaskTiled);
                    owner.Ops.BinaryOpPack4(scores, attnMaskTiled, plan.scoresSlices, 0, scoresBiased);
                }
                finally
                {
                    ReturnTemp(owner, ref attnMaskScalarMaterialized);
                }

                owner.Ops.SoftmaxPack4Cdhw(scoresBiased, plan.scoresShape.w, plan.scoresShape.h, plan.scoresShape.d, plan.scoresShape.c, weights);
            }
            else
            {
                owner.Ops.SoftmaxPack4Cdhw(scores, plan.scoresShape.w, plan.scoresShape.h, plan.scoresShape.d, plan.scoresShape.c, weights);
            }

            owner.Ops.MatMulPack4Cdhw(
                weights,
                plan.scoresShape.h,
                plan.scoresShape.w,
                plan.scoresShape.d,
                plan.scoresShape.c,
                vHead,
                plan.headShape.h,
                plan.headShape.w,
                plan.headShape.d,
                plan.headShape.c,
                false,
                plan.headShape.d,
                plan.headShape.c,
                contextHeads);

            owner.Ops.PermutePack4Cdhw(
                contextHeads,
                plan.headShape.w,
                plan.headShape.h,
                plan.headShape.d,
                plan.headShape.c,
                new Vector4Int(3, 1, 2, 0),
                plan.contextPermutedShape.w,
                plan.contextPermutedShape.h,
                plan.contextPermutedShape.d,
                plan.contextPermutedShape.c,
                contextPermuted);

            owner.Ops.AttentionContextFlattenPack4(
                contextPermuted,
                plan.contextPermutedShape.w,
                plan.contextPermutedShape.h,
                plan.contextPermutedShape.d,
                plan.contextPermutedShape.c,
                plan.contextFlattenOutChannels,
                2,
                contextFlat);

            owner.Ops.Gemm2DTextureA(
                contextFlat,
                plan.pack.oW,
                plan.pack.oB,
                plan.rows,
                plan.pack.qdim,
                plan.embedDim,
                transB: true,
                alpha: 1f,
                beta: 1f,
                useC: true,
                broadcastTypeC: 4,
                output);
            owner.Ops.Pack4LinearFromScalar2D(output, plan.outputLogicalShape.w, plan.outputLogicalShape.h, outputPacked);
        }

        private static bool TryGetScalar2DTexture(
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            string blobName,
            out AexisGraphSession.TensorRef texture,
            out AexisGraphSession.BufferShape shape)
        {
            if (!AexisGraphSession.TryGetExistingTexture(textureBlobs, textureShapes, blobName, out texture, out shape))
                return false;
            return IsScalarOrPack4Linear2DTexture(texture, shape);
        }

        private static Vector4Int ResolveAttentionMaskTileRepeats(AexisGraphSession.BufferShape scoresShape)
        {
            return new Vector4Int(
                1,
                1,
                Mathf.Max(1, scoresShape.d),
                Mathf.Max(1, scoresShape.c));
        }

        private static bool TryGetScalar2DTexture(
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
            return IsScalarOrPack4Linear2DTexture(texture, shape);
        }

        private static bool IsScalarOrPack4Linear2DTexture(
            AexisGraphSession.TensorRef texture,
            AexisGraphSession.BufferShape shape)
        {
            return texture != null
                && texture.texture != null
                && shape.dims == 2
                && shape.w > 0
                && shape.h > 0
                && ((texture.width == shape.w && texture.height == shape.h && texture.packs == 1)
                    || AexisGraphSession.IsPack4LinearMatTexture(texture, shape));
        }

        private static bool IsScalarOrPack4Linear2DTexture(
            AexisGraphSession.CmdTensorRef texture,
            AexisGraphSession.BufferShape shape)
        {
            return texture != null
                && texture.texture != null
                && shape.dims == 2
                && shape.w > 0
                && shape.h > 0
                && ((texture.width == shape.w && texture.height == shape.h && texture.packs == 1)
                    || AexisGraphSession.IsPack4LinearMatTexture(texture, shape));
        }

        private static bool RequiresScalar2DMaterialization(
            AexisGraphSession.TensorRef texture,
            AexisGraphSession.BufferShape shape)
        {
            return AexisGraphSession.IsStrictLinearMatTexture(texture)
                || AexisGraphSession.IsPack4LinearMatTexture(texture, shape);
        }

        private static bool RequiresScalar2DMaterialization(
            AexisGraphSession.CmdTensorRef texture,
            AexisGraphSession.BufferShape shape)
        {
            return AexisGraphSession.IsStrictLinearMatTexture(texture)
                || AexisGraphSession.IsPack4LinearMatTexture(texture, shape);
        }

        private static RenderTexture MaterializeScalar2DArrayInput(
            AexisGraphSession owner,
            AexisGraphSession.TensorRef source,
            AexisGraphSession.BufferShape shape,
            RenderTextureFormat outputFormat,
            ref RenderTexture materialized)
        {
            if (owner == null)
                throw new ArgumentNullException(nameof(owner));
            if (source == null || source.texture == null)
                throw new ArgumentNullException(nameof(source));
            if (!RequiresScalar2DMaterialization(source, shape))
                return source.texture;

            materialized = owner.RentTempArray(Mathf.Max(1, shape.w), Mathf.Max(1, shape.h), 1, outputFormat);
            if (AexisGraphSession.IsPack4LinearMatTexture(source, shape))
            {
                owner.Ops.ReshapePack4ToScalar2D(
                    source.texture,
                    shape.w,
                    shape.h,
                    1,
                    1,
                    2,
                    materialized);
            }
            else if (AexisGraphSession.IsStrictLinearMatTexture(source))
            {
                owner.Ops.ReshapeLinearMatToPack4(
                    source.texture,
                    shape.w,
                    shape.h,
                    shape.w,
                    shape.h,
                    1,
                    1,
                    2,
                    materialized);
            }
            return materialized;
        }

        private static ComputeTexture MaterializeScalar2DArrayInput(
            AexisGraphSession owner,
            CommandBuffer cmd,
            AexisGraphSession.CmdTensorRef source,
            AexisGraphSession.BufferShape shape,
            RenderTextureFormat outputFormat,
            ref ComputeTexture materialized)
        {
            if (owner == null)
                throw new ArgumentNullException(nameof(owner));
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if (source == null || source.texture == null)
                throw new ArgumentNullException(nameof(source));
            if (!RequiresScalar2DMaterialization(source, shape))
                return source.texture;

            materialized = owner.RentTempArray(cmd, Mathf.Max(1, shape.w), Mathf.Max(1, shape.h), 1, outputFormat);
            if (AexisGraphSession.IsPack4LinearMatTexture(source, shape))
            {
                owner.Ops.ReshapePack4ToScalar2D(
                    cmd,
                    source.texture,
                    shape.w,
                    shape.h,
                    1,
                    1,
                    2,
                    materialized);
            }
            else if (AexisGraphSession.IsStrictLinearMatTexture(source))
            {
                owner.Ops.ReshapeLinearMatToPack4(
                    cmd,
                    source.texture,
                    shape.w,
                    shape.h,
                    shape.w,
                    shape.h,
                    1,
                    1,
                    2,
                    materialized);
            }
            return materialized;
        }

        private static void ReturnTemp(AexisGraphSession owner, ref RenderTexture texture)
        {
            if (owner == null || texture == null)
                return;
            owner.ReturnTempArray(texture);
            texture = null;
        }

        private static void ReturnTempUnique(AexisGraphSession owner, ref RenderTexture texture, RenderTexture alias0, RenderTexture alias1)
        {
            if (texture == null || ReferenceEquals(texture, alias0) || ReferenceEquals(texture, alias1))
            {
                texture = null;
                return;
            }

            ReturnTemp(owner, ref texture);
        }

        private static void ReturnTemp(AexisGraphSession owner, CommandBuffer cmd, ref ComputeTexture texture)
        {
            if (owner == null || cmd == null || texture == null)
                return;
            owner.ReturnTempArray(cmd, texture);
            texture = null;
        }

        private static void ReturnTempUnique(AexisGraphSession owner, CommandBuffer cmd, ref ComputeTexture texture, ComputeTexture alias0, ComputeTexture alias1)
        {
            if (texture == null)
                return;
            if ((alias0 != null && texture.nameID == alias0.nameID) || (alias1 != null && texture.nameID == alias1.nameID))
            {
                texture = null;
                return;
            }

            ReturnTemp(owner, cmd, ref texture);
        }

        private readonly struct MultiHeadAttentionRtPlan
        {
            public readonly AexisGraphSession.MultiHeadAttentionPack pack;
            public readonly AexisGraphSession.TensorRef qTex;
            public readonly AexisGraphSession.BufferShape qShape;
            public readonly AexisGraphSession.TensorRef kTex;
            public readonly AexisGraphSession.BufferShape kShape;
            public readonly AexisGraphSession.TensorRef vTex;
            public readonly AexisGraphSession.BufferShape vShape;
            public readonly bool hasAttnMask;
            public readonly AexisGraphSession.TensorRef attnMaskTex;
            public readonly AexisGraphSession.BufferShape attnMaskShape;
            public readonly int rows;
            public readonly int embedDim;
            public readonly int numHeads;
            public readonly int headDim;
            public readonly AexisGraphSession.BufferShape scalarStorageShape;
            public readonly AexisGraphSession.BufferShape headShape;
            public readonly AexisGraphSession.BufferShape headStorageShape;
            public readonly int headSlices;
            public readonly AexisGraphSession.BufferShape keyTransposedShape;
            public readonly AexisGraphSession.BufferShape keyTransposedStorageShape;
            public readonly int keyTransposedSlices;
            public readonly AexisGraphSession.BufferShape scoresShape;
            public readonly AexisGraphSession.BufferShape scoresStorageShape;
            public readonly int scoresSlices;
            public readonly AexisGraphSession.BufferShape contextPermutedShape;
            public readonly AexisGraphSession.BufferShape contextPermutedStorageShape;
            public readonly int contextPermutedSlices;
            public readonly int contextFlattenOutChannels;
            public readonly AexisGraphSession.BufferShape outputLogicalShape;
            public readonly AexisGraphSession.BufferShape outputScalarStorageShape;
            public readonly AexisGraphSession.BufferShape outputStorageShape;
            public readonly RenderTextureFormat scalarTextureFormat;
            public readonly RenderTextureFormat pack4TextureFormat;

            public MultiHeadAttentionRtPlan(
                AexisGraphSession.MultiHeadAttentionPack pack,
                AexisGraphSession.TensorRef qTex,
                AexisGraphSession.BufferShape qShape,
                AexisGraphSession.TensorRef kTex,
                AexisGraphSession.BufferShape kShape,
                AexisGraphSession.TensorRef vTex,
                AexisGraphSession.BufferShape vShape,
                int rows,
                AexisGraphSession.TensorRef attnMaskTex,
                AexisGraphSession.BufferShape attnMaskShape,
                RenderTextureFormat sensitiveTextureFormat)
            {
                this.pack = pack;
                this.qTex = qTex;
                this.qShape = qShape;
                this.kTex = kTex;
                this.kShape = kShape;
                this.vTex = vTex;
                this.vShape = vShape;
                hasAttnMask = attnMaskTex != null && attnMaskTex.texture != null;
                this.attnMaskTex = attnMaskTex;
                this.attnMaskShape = hasAttnMask ? attnMaskShape : default;
                this.rows = rows;
                embedDim = pack.embedDim;
                numHeads = Mathf.Max(1, pack.numHeads);
                headDim = Mathf.Max(1, pack.embedDim / numHeads);
                scalarStorageShape = new AexisGraphSession.BufferShape(3, Mathf.Max(1, embedDim), Mathf.Max(1, rows), 1, 1);
                headShape = new AexisGraphSession.BufferShape(4, Mathf.Max(1, headDim), Mathf.Max(1, rows), 1, Mathf.Max(1, numHeads));
                headStorageShape = headShape;
                headSlices = Mathf.Max(1, headShape.d * Mathf.CeilToInt(headShape.c / 4f));
                keyTransposedShape = new AexisGraphSession.BufferShape(4, Mathf.Max(1, rows), Mathf.Max(1, headDim), 1, Mathf.Max(1, numHeads));
                keyTransposedStorageShape = keyTransposedShape;
                keyTransposedSlices = Mathf.Max(1, keyTransposedShape.d * Mathf.CeilToInt(keyTransposedShape.c / 4f));
                scoresShape = new AexisGraphSession.BufferShape(4, Mathf.Max(1, rows), Mathf.Max(1, rows), 1, Mathf.Max(1, numHeads));
                scoresStorageShape = scoresShape;
                scoresSlices = Mathf.Max(1, scoresShape.d * Mathf.CeilToInt(scoresShape.c / 4f));
                contextPermutedShape = new AexisGraphSession.BufferShape(4, Mathf.Max(1, numHeads), Mathf.Max(1, rows), 1, Mathf.Max(1, headDim));
                contextPermutedStorageShape = contextPermutedShape;
                contextPermutedSlices = Mathf.Max(1, contextPermutedShape.d * Mathf.CeilToInt(contextPermutedShape.c / 4f));
                contextFlattenOutChannels = Mathf.Max(1, headDim);
                outputLogicalShape = new AexisGraphSession.BufferShape(2, Mathf.Max(1, pack.qdim), Mathf.Max(1, rows), 1, 1);
                outputScalarStorageShape = new AexisGraphSession.BufferShape(3, outputLogicalShape.w, outputLogicalShape.h, 1, 1);
                outputStorageShape = AexisGraphSession.ResolvePack4LinearMatStorageShape(outputLogicalShape);
                scalarTextureFormat = sensitiveTextureFormat;
                pack4TextureFormat = AexisGraphSession.ResolveTensorTextureFormat(4);
            }
        }

        private readonly struct MultiHeadAttentionCmdRtPlan
        {
            public readonly AexisGraphSession.MultiHeadAttentionPack pack;
            public readonly AexisGraphSession.CmdTensorRef qTex;
            public readonly AexisGraphSession.BufferShape qShape;
            public readonly AexisGraphSession.CmdTensorRef kTex;
            public readonly AexisGraphSession.BufferShape kShape;
            public readonly AexisGraphSession.CmdTensorRef vTex;
            public readonly AexisGraphSession.BufferShape vShape;
            public readonly bool hasAttnMask;
            public readonly AexisGraphSession.CmdTensorRef attnMaskTex;
            public readonly AexisGraphSession.BufferShape attnMaskShape;
            public readonly int rows;
            public readonly int embedDim;
            public readonly int numHeads;
            public readonly int headDim;
            public readonly AexisGraphSession.BufferShape scalarStorageShape;
            public readonly AexisGraphSession.BufferShape headShape;
            public readonly AexisGraphSession.BufferShape headStorageShape;
            public readonly int headSlices;
            public readonly AexisGraphSession.BufferShape keyTransposedShape;
            public readonly AexisGraphSession.BufferShape keyTransposedStorageShape;
            public readonly int keyTransposedSlices;
            public readonly AexisGraphSession.BufferShape scoresShape;
            public readonly AexisGraphSession.BufferShape scoresStorageShape;
            public readonly int scoresSlices;
            public readonly AexisGraphSession.BufferShape contextPermutedShape;
            public readonly AexisGraphSession.BufferShape contextPermutedStorageShape;
            public readonly int contextPermutedSlices;
            public readonly int contextFlattenOutChannels;
            public readonly AexisGraphSession.BufferShape outputLogicalShape;
            public readonly AexisGraphSession.BufferShape outputScalarStorageShape;
            public readonly AexisGraphSession.BufferShape outputStorageShape;
            public readonly RenderTextureFormat scalarTextureFormat;
            public readonly RenderTextureFormat pack4TextureFormat;

            public MultiHeadAttentionCmdRtPlan(
                AexisGraphSession.MultiHeadAttentionPack pack,
                AexisGraphSession.CmdTensorRef qTex,
                AexisGraphSession.BufferShape qShape,
                AexisGraphSession.CmdTensorRef kTex,
                AexisGraphSession.BufferShape kShape,
                AexisGraphSession.CmdTensorRef vTex,
                AexisGraphSession.BufferShape vShape,
                int rows,
                AexisGraphSession.CmdTensorRef attnMaskTex,
                AexisGraphSession.BufferShape attnMaskShape,
                RenderTextureFormat sensitiveTextureFormat)
            {
                this.pack = pack;
                this.qTex = qTex;
                this.qShape = qShape;
                this.kTex = kTex;
                this.kShape = kShape;
                this.vTex = vTex;
                this.vShape = vShape;
                hasAttnMask = attnMaskTex != null && attnMaskTex.texture != null;
                this.attnMaskTex = attnMaskTex;
                this.attnMaskShape = hasAttnMask ? attnMaskShape : default;
                this.rows = rows;
                embedDim = pack.embedDim;
                numHeads = Mathf.Max(1, pack.numHeads);
                headDim = Mathf.Max(1, pack.embedDim / numHeads);
                scalarStorageShape = new AexisGraphSession.BufferShape(3, Mathf.Max(1, embedDim), Mathf.Max(1, rows), 1, 1);
                headShape = new AexisGraphSession.BufferShape(4, Mathf.Max(1, headDim), Mathf.Max(1, rows), 1, Mathf.Max(1, numHeads));
                headStorageShape = headShape;
                headSlices = Mathf.Max(1, headShape.d * Mathf.CeilToInt(headShape.c / 4f));
                keyTransposedShape = new AexisGraphSession.BufferShape(4, Mathf.Max(1, rows), Mathf.Max(1, headDim), 1, Mathf.Max(1, numHeads));
                keyTransposedStorageShape = keyTransposedShape;
                keyTransposedSlices = Mathf.Max(1, keyTransposedShape.d * Mathf.CeilToInt(keyTransposedShape.c / 4f));
                scoresShape = new AexisGraphSession.BufferShape(4, Mathf.Max(1, rows), Mathf.Max(1, rows), 1, Mathf.Max(1, numHeads));
                scoresStorageShape = scoresShape;
                scoresSlices = Mathf.Max(1, scoresShape.d * Mathf.CeilToInt(scoresShape.c / 4f));
                contextPermutedShape = new AexisGraphSession.BufferShape(4, Mathf.Max(1, numHeads), Mathf.Max(1, rows), 1, Mathf.Max(1, headDim));
                contextPermutedStorageShape = contextPermutedShape;
                contextPermutedSlices = Mathf.Max(1, contextPermutedShape.d * Mathf.CeilToInt(contextPermutedShape.c / 4f));
                contextFlattenOutChannels = Mathf.Max(1, headDim);
                outputLogicalShape = new AexisGraphSession.BufferShape(2, Mathf.Max(1, pack.qdim), Mathf.Max(1, rows), 1, 1);
                outputScalarStorageShape = new AexisGraphSession.BufferShape(3, outputLogicalShape.w, outputLogicalShape.h, 1, 1);
                outputStorageShape = AexisGraphSession.ResolvePack4LinearMatStorageShape(outputLogicalShape);
                scalarTextureFormat = sensitiveTextureFormat;
                pack4TextureFormat = AexisGraphSession.ResolveTensorTextureFormat(4);
            }
        }
    }
}
