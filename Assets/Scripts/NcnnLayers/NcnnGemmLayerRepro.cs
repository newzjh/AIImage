using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnGemmLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnGemmLayerRepro() : base(NcnnLayerTypes.Gemm, supportsBufferPath: true, supportsCommandBufferPath: false) { }

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
                                            if (gp.cDataCpu != null)
                                                gp.cData = NcnnRepro.NewBuffer(gp.cDataCpu);
                                            phaseSw.Stop();
                                            uploadMs += phaseSw.ElapsedMilliseconds;
                                        }

                                        owner._gemm[layer.name] = gp;
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
                                                if (gp.constantC && gp.broadcastTypeC != -1 && gp.cData != null)
                                                {
                                                    useC = true;
                                                    cBuf = gp.cData;
                                                }
                                                else if (!gp.constantC && layer.bottomNames.Length > 2)
                                                {
                                                    cBuf = owner.GetOrConvertToBuffer(layer.bottomNames[2], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                    useC = cBuf != null;
                                                }

                                                var outBuf = owner.RentTempBuffer(m * n, sizeof(float));
                                                owner.Ops.Gemm2D(srcBuf, bBuf, cBuf, m, n, k, gp.transB, gp.alpha, gp.beta, useC, gp.broadcastTypeC, outBuf);
                                                bufferBlobs[layer.topNames[0]] = outBuf;
                                                bufferRefs[layer.topNames[0]] = owner.NewOwnedBufferRef(layer.topNames[0], outBuf);
                                                bufferViews[layer.topNames[0]] = m == 1 && srcView.dims == 1
                                                    ? new NcnnTensorBuffer(outBuf, 1, n, 1, 1, 1, false)
                                                    : new NcnnTensorBuffer(outBuf, 2, n, m, 1, 1, false);
                                                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }
    }
}
