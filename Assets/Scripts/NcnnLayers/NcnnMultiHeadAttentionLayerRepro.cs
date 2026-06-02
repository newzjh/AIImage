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
        public NcnnMultiHeadAttentionLayerRepro() : base(NcnnLayerTypes.MultiHeadAttention, supportsBufferPath: true, supportsCommandBufferPath: false) { }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br) => owner.LoadMultiHeadAttentionLayer(layer, br);
        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context) => owner.ExecuteMultiHeadAttentionBufferLayer(layer, context);
    }

    public partial class NcnnRepro
    {
        internal LayerLoadMetrics LoadMultiHeadAttentionLayer(NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

                            var mp = new MultiHeadAttentionPack();
                            mp.embedDim = layer.GetInt(0, 0);
                            mp.numHeads = layer.GetInt(1, 1);
                            mp.weightDataSize = layer.GetInt(2, 0);
                            mp.kdim = layer.GetInt(3, mp.embedDim);
                            mp.vdim = layer.GetInt(4, mp.embedDim);
                            mp.scale = layer.GetFloat(6, 1f / Mathf.Sqrt(Mathf.Max(1, mp.embedDim / Mathf.Max(1, mp.numHeads))));
                            mp.qdim = mp.embedDim > 0 ? mp.weightDataSize / Mathf.Max(1, mp.embedDim) : 0;

                            phaseSw.Restart();
                            var qW = ReadClipMatAsFloat32(br, mp.embedDim * mp.qdim, 0, 0, 0, 0);
                            var qB = br.ReadNcnnMatAsFloat32(mp.embedDim, 0, 0, 0, 1);
                            var kW = ReadClipMatAsFloat32(br, mp.embedDim * mp.kdim, 0, 0, 0, 0);
                            var kB = br.ReadNcnnMatAsFloat32(mp.embedDim, 0, 0, 0, 1);
                            var vW = ReadClipMatAsFloat32(br, mp.embedDim * mp.vdim, 0, 0, 0, 0);
                            var vB = br.ReadNcnnMatAsFloat32(mp.embedDim, 0, 0, 0, 1);
                            var oW = ReadClipMatAsFloat32(br, mp.qdim * mp.embedDim, 0, 0, 0, 0);
                            var oB = br.ReadNcnnMatAsFloat32(mp.qdim, 0, 0, 0, 1);
                            phaseSw.Stop();
                            readMs += phaseSw.ElapsedMilliseconds;

                            phaseSw.Restart();
                            mp.qW = NewBuffer(qW);
                            mp.qB = NewBuffer(qB);
                            mp.kW = NewBuffer(kW);
                            mp.kB = NewBuffer(kB);
                            mp.vW = NewBuffer(vW);
                            mp.vB = NewBuffer(vB);
                            mp.oW = NewBuffer(oW);
                            mp.oB = NewBuffer(oB);
                            phaseSw.Stop();
                            uploadMs += phaseSw.ElapsedMilliseconds;

                            _multiHeadAttention[layer.name] = mp;
                            return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        internal void ExecuteMultiHeadAttentionBufferLayer(NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                                    if (!_multiHeadAttention.TryGetValue(layer.name, out var mp))
                                        throw new InvalidOperationException("MultiHeadAttention not found: " + layer.name);

                                    if (!bufferBlobs.TryGetValue(layer.bottomNames[0], out var qBuf) || qBuf == null)
                                        throw new InvalidOperationException("MultiHeadAttention q input not found: " + layer.name);
                                    if (!bufferBlobs.TryGetValue(layer.bottomNames.Length > 1 ? layer.bottomNames[1] : layer.bottomNames[0], out var kBuf) || kBuf == null)
                                        throw new InvalidOperationException("MultiHeadAttention k input not found: " + layer.name);
                                    if (!bufferBlobs.TryGetValue(layer.bottomNames.Length > 2 ? layer.bottomNames[2] : layer.bottomNames[0], out var vBuf) || vBuf == null)
                                        throw new InvalidOperationException("MultiHeadAttention v input not found: " + layer.name);

                                    var srcLen = qBuf.count / Mathf.Max(1, mp.qdim);
                                    var dstLen = kBuf.count / Mathf.Max(1, mp.kdim);
                                    var qAff = RentTempBuffer(srcLen * mp.embedDim, sizeof(float));
                                    var kAff = RentTempBuffer(dstLen * mp.embedDim, sizeof(float));
                                    var vAff = RentTempBuffer(dstLen * mp.embedDim, sizeof(float));
                                    _ops.InnerProduct2D(qBuf, srcLen, mp.qdim, mp.qW, mp.qB, mp.embedDim, qAff);
                                    _ops.InnerProduct2D(kBuf, dstLen, mp.kdim, mp.kW, mp.kB, mp.embedDim, kAff);
                                    _ops.InnerProduct2D(vBuf, dstLen, mp.vdim, mp.vW, mp.vB, mp.embedDim, vAff);

                                    var qScaled = RentTempBuffer(srcLen * mp.embedDim, sizeof(float));
                                    _ops.BinaryOpScalarBuf(qAff, mp.scale, qAff.count, 2, qScaled);

                                    var ctx = RentTempBuffer(srcLen * mp.embedDim, sizeof(float));
                                    _ops.MhaAttention(qScaled, kAff, vAff, srcLen, dstLen, mp.embedDim, mp.numHeads, 1f, ctx);

                                    var outBuf = RentTempBuffer(srcLen * mp.qdim, sizeof(float));
                                    _ops.InnerProduct2D(ctx, srcLen, mp.embedDim, mp.oW, mp.oB, mp.qdim, outBuf);

                                    bufferBlobs[layer.topNames[0]] = outBuf;
                                    bufferViews[layer.topNames[0]] = new NcnnTensorBuffer(outBuf, 2, mp.qdim, srcLen, 1, 1, false);
                                    tempOwned.Add(qAff);
                                    tempOwned.Add(kAff);
                                    tempOwned.Add(vAff);
                                    tempOwned.Add(qScaled);
                                    tempOwned.Add(ctx);
                                    tempOwned.Add(outBuf);
                                    Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }
    }
}
