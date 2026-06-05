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
                                                if (!owner._layerNorm.TryGetValue(layer.name, out var lp))
                                                    throw new InvalidOperationException("LayerNorm not found: " + layer.name);
                                                var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                var srcView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                if (srcBuf == null || srcView == null || srcView.dims != 2)
                                                    throw new InvalidOperationException("LayerNorm expects dims=2 buffer input: " + layer.name);
                                                var outTensor = owner.RentTempTensorBuffer(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c);
                                                owner.Ops.CopyBuf(srcBuf, outTensor.buffer, srcBuf.count);
                                                owner.Ops.LayerNorm2DInplace(outTensor.buffer, srcView.h, srcView.w, lp.eps, lp.affine, lp.gamma, lp.beta);
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
