using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
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
        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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

                                                if (!bufferBlobs.TryGetValue(layer.bottomNames[0], out var qBuf) || qBuf == null)
                                                    throw new InvalidOperationException("MultiHeadAttention q input not found: " + layer.name);
                                                if (!bufferBlobs.TryGetValue(layer.bottomNames.Length > 1 ? layer.bottomNames[1] : layer.bottomNames[0], out var kBuf) || kBuf == null)
                                                    throw new InvalidOperationException("MultiHeadAttention k input not found: " + layer.name);
                                                if (!bufferBlobs.TryGetValue(layer.bottomNames.Length > 2 ? layer.bottomNames[2] : layer.bottomNames[0], out var vBuf) || vBuf == null)
                                                    throw new InvalidOperationException("MultiHeadAttention v input not found: " + layer.name);

                                                var srcLen = qBuf.count / Mathf.Max(1, mp.qdim);
                                                var dstLen = kBuf.count / Mathf.Max(1, mp.kdim);
                                                var ctx = owner.RentTempBuffer(srcLen * mp.embedDim, sizeof(float));

                                                var canFuseSelfAttentionQkv = owner.EnableMhaQkvFusion
                                                                              && owner.EnableMhaParallelSoftmax
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
                                                    owner.Ops.MhaAttention(qScaled, kAff, vAff, srcLen, dstLen, mp.embedDim, mp.numHeads, 1f, ctx, owner.EnableMhaParallelSoftmax);

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

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            owner.PublishCmdTensorLikeInput(cmd, layer.topNames[0], Mathf.Max(1, src.width), Mathf.Max(1, src.height), Mathf.Max(1, src.packs), blobs);
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
        }
    }
}
