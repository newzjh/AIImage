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

            if (!owner._gemm.TryGetValue(layer.name, out var gp))
                throw new InvalidOperationException("Gemm not found: " + layer.name);

            var aShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var aRows = aShape.dims == 1 ? 1 : aShape.h;
            var aCols = aShape.w;
            var m = aShape.dims == 1 ? 1 : aRows;
            var k = aCols;
            var bRows = 0;
            var bCols = 0;

            if (gp.constantB)
            {
                bRows = gp.transB ? gp.constantN : gp.constantK;
                bCols = gp.transB ? gp.constantK : gp.constantN;
            }
            else
            {
                if (layer.bottomNames.Length < 2)
                    throw new InvalidOperationException("Gemm B input missing: " + layer.name);
                var bShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[1]);
                bRows = bShape.dims == 1 ? 1 : bShape.h;
                bCols = bShape.w;
            }

            var kFromB = gp.transB ? bCols : bRows;
            var n = gp.transB ? bRows : bCols;
            if (gp.constantK > 0)
                k = gp.constantK;
            if (kFromB > 0)
                k = Mathf.Min(k, kFromB) == 0 ? kFromB : k;

            var outShape = m == 1 && aShape.dims == 1
                ? new NcnnRepro.BufferShape(1, Mathf.Max(1, n), 1, 1, 1)
                : new NcnnRepro.BufferShape(2, Mathf.Max(1, n), Mathf.Max(1, m), 1, 1);
            owner.PublishCmdPlaceholder(cmd, layer.topNames[0], outShape, blobs, shapes);
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool TryExecuteCommandBufferTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            if (owner == null || layer == null || context == null)
                return false;
            if (owner.ShouldForceCurrentLayerBufferPath())
                return false;
            if (!owner._gemm.TryGetValue(layer.name, out var gp))
                return false;
            if (gp.transA || !gp.constantB)
                return false;

            var srcTex = NcnnRepro.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var srcShape = NcnnRepro.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            if (srcTex == null || srcTex.texture == null)
                return false;
            if (srcShape.dims != 2)
                return false;
            if (srcTex.width != srcShape.w || srcTex.height != srcShape.h || srcTex.packs != 1)
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
            var outFormat = ShouldPromoteAttentionGemmOutputTexture(owner, layer)
                ? RenderTextureFormat.ARGBFloat
                : RenderTextureFormat.ARGBHalf;
            var outRt = owner.RentTempArray(context.commandBuffer, outShape.w, outShape.h, 1, outFormat);
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
                storageShape = new NcnnRepro.BufferShape(3, outShape.w, outShape.h, 1, 1)
            };
            context.shapes[layer.topNames[0]] = outShape;
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
            if (gp.transA || !gp.constantB)
                return false;
            if (!NcnnRepro.TryGetExistingTexture(context.textureBlobs, context.textureShapes, layer.bottomNames[0], out var srcTex, out var srcShape))
                return false;
            if (srcTex == null || srcTex.texture == null)
                return false;
            if (srcShape.dims != 2)
                return false;
            if (srcTex.width != srcShape.w || srcTex.height != srcShape.h || srcTex.packs != 1)
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
            var outFormat = ShouldPromoteAttentionGemmOutputTexture(owner, layer)
                ? RenderTextureFormat.ARGBFloat
                : RenderTextureFormat.ARGBHalf;
            var outRt = owner.RentTempArray(outShape.w, outShape.h, 1, outFormat);
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
            NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], outRt, outShape, outShape);
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
    }
}
