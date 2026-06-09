using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnLayerNormLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnLayerNormLayerRepro() : base(NcnnLayerTypes.LayerNorm, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var lp = new NcnnRepro.LayerNormPack();
                                        lp.affineSize = layer.GetInt(0, 0);
                                        lp.eps = layer.GetFloat(1, 1e-5f);
                                        lp.affine = layer.GetInt(2, 1) != 0;

                                        float[] gamma = null;
                                        float[] beta = null;
                                        if (lp.affine && lp.affineSize > 0)
                                        {
                                            phaseSw.Restart();
                                            gamma = br.ReadNcnnMatAsFloat32(lp.affineSize, 0, 0, 0, 1);
                                            beta = br.ReadNcnnMatAsFloat32(lp.affineSize, 0, 0, 0, 1);
                                            phaseSw.Stop();
                                            readMs += phaseSw.ElapsedMilliseconds;

                                            phaseSw.Restart();
                                            lp.gamma = new ComputeBuffer(gamma.Length, sizeof(float), ComputeBufferType.Structured);
                                            lp.beta = new ComputeBuffer(beta.Length, sizeof(float), ComputeBufferType.Structured);
                                            lp.gamma.SetData(gamma);
                                            lp.beta.SetData(beta);
                                            phaseSw.Stop();
                                            uploadMs += phaseSw.ElapsedMilliseconds;
                                        }

                                        owner._layerNorm[layer.name] = lp;
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
                                                if (!owner._layerNorm.TryGetValue(layer.name, out var lp))
                                                    throw new InvalidOperationException("LayerNorm not found: " + layer.name);
                                                using var srcView = owner.GetReadableTensorInput(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                if (srcView == null || srcView.buffer == null)
                                                    throw new InvalidOperationException("LayerNorm source not found: " + layer.name);
                                                ResolveLayerNormRowsCols(srcView, lp.affineSize, layer.name, out var rows, out var cols);
                                                var outTensor = owner.RentTempTensorBuffer(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c);
                                                owner.Ops.CopyBuf(srcView.buffer, outTensor.buffer, srcView.buffer.count);
                                                owner.Ops.LayerNorm2DInplace(outTensor.buffer, rows, cols, lp.eps, lp.affine, lp.gamma, lp.beta);
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
#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        private static void ResolveLayerNormRowsCols(NcnnTensorBuffer srcView, int affineSize, string layerName, out int rows, out int cols)
        {
            if (srcView == null)
                throw new ArgumentNullException(nameof(srcView));

            var w = srcView.w;
            var h = Mathf.Max(1, srcView.h);
            var d = Mathf.Max(1, srcView.d);
            var c = Mathf.Max(1, srcView.c);
            rows = 0;
            cols = 0;

            switch (srcView.dims)
            {
                case 1:
                {
                    rows = 1;
                    cols = w;
                    return;
                }
                case 2:
                {
                    rows = h;
                    cols = w;
                    return;
                }
                case 3:
                {
                    if (affineSize <= 0 || affineSize == w)
                    {
                        rows = h * c;
                        cols = w;
                        return;
                    }

                    if (affineSize == w * h)
                    {
                        rows = c;
                        cols = w * h;
                        return;
                    }

                    break;
                }
                case 4:
                {
                    if (affineSize <= 0 || affineSize == w)
                    {
                        rows = h * d * c;
                        cols = w;
                        return;
                    }

                    if (affineSize == w * h)
                    {
                        rows = d * c;
                        cols = w * h;
                        return;
                    }

                    if (affineSize == w * h * d)
                    {
                        rows = c;
                        cols = w * h * d;
                        return;
                    }

                    break;
                }
            }

            throw new InvalidOperationException(
                "Unsupported LayerNorm shape"
                + " | layer=" + layerName
                + " | dims=" + srcView.dims
                + " | w=" + srcView.w
                + " | h=" + srcView.h
                + " | d=" + srcView.d
                + " | c=" + srcView.c
                + " | affineSize=" + affineSize);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            owner.PublishCmdPlaceholder(cmd, layer.topNames[0], srcShape, blobs, shapes);
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }
    }
}
