using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnMultiHeadAttentionLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnMultiHeadAttentionLayerRepro() : base(NcnnLayerTypes.MultiHeadAttention, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var mp = new NcnnRepro.MultiHeadAttentionPack();
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
                                        var qW = NcnnRepro.ReadClipMatAsFloat32(br, mp.embedDim * mp.qdim, 0, 0, 0, 0);
                                        var qB = br.ReadNcnnMatAsFloat32(mp.embedDim, 0, 0, 0, 1);
                                        var kW = NcnnRepro.ReadClipMatAsFloat32(br, mp.embedDim * mp.kdim, 0, 0, 0, 0);
                                        var kB = br.ReadNcnnMatAsFloat32(mp.embedDim, 0, 0, 0, 1);
                                        var vW = NcnnRepro.ReadClipMatAsFloat32(br, mp.embedDim * mp.vdim, 0, 0, 0, 0);
                                        var vB = br.ReadNcnnMatAsFloat32(mp.embedDim, 0, 0, 0, 1);
                                        var oW = NcnnRepro.ReadClipMatAsFloat32(br, mp.qdim * mp.embedDim, 0, 0, 0, 0);
                                        var oB = br.ReadNcnnMatAsFloat32(mp.qdim, 0, 0, 0, 1);
                                        phaseSw.Stop();
                                        readMs += phaseSw.ElapsedMilliseconds;

                                        phaseSw.Restart();
                                        mp.qW = NcnnRepro.NewBuffer(qW);
                                        mp.qB = NcnnRepro.NewBuffer(qB);
                                        mp.kW = NcnnRepro.NewBuffer(kW);
                                        mp.kB = NcnnRepro.NewBuffer(kB);
                                        mp.vW = NcnnRepro.NewBuffer(vW);
                                        mp.vB = NcnnRepro.NewBuffer(vB);
                                        mp.oW = NcnnRepro.NewBuffer(oW);
                                        mp.oB = NcnnRepro.NewBuffer(oB);
                                        phaseSw.Stop();
                                        uploadMs += phaseSw.ElapsedMilliseconds;

                                        owner._multiHeadAttention[layer.name] = mp;
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
                                                NcnnTensorBuffer attnMaskView = null;
                                                if (attnMaskIndex >= 0)
                                                {
                                                    attnMaskBuf = owner.GetOrConvertToBuffer(layer.bottomNames[attnMaskIndex], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                    attnMaskView = NcnnRepro.TryGetBufferView(layer.bottomNames[attnMaskIndex], bufferBlobs, bufferViews);
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

            if (!owner._multiHeadAttention.TryGetValue(layer.name, out var mp))
                throw new InvalidOperationException("MultiHeadAttention not found: " + layer.name);

            var qShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var totalElems = Mathf.Max(1, qShape.w * qShape.h * qShape.d * qShape.c);
            var srcLen = mp.qdim > 0 ? Mathf.Max(1, totalElems / mp.qdim) : Mathf.Max(1, qShape.h);
            var outShape = new NcnnRepro.BufferShape(2, Mathf.Max(1, mp.qdim), srcLen, 1, 1);
            owner.PublishCmdPlaceholder(cmd, layer.topNames[0], outShape, blobs, shapes);
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
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
    }
}
