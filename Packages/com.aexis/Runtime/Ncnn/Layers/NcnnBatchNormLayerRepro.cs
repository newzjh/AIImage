using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Ncnn
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnBatchNormLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnBatchNormLayerRepro() : base(NcnnLayerTypes.BatchNorm, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override NcnnGraphSession.LayerLoadMetrics LoadLayer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var bp = new NcnnGraphSession.BatchNormPack();
                                        bp.channels = layer.GetInt(0, 0);

                                        phaseSw.Restart();
                                        var slope = br.ReadNcnnMatAsFloat32(bp.channels, 0, 0, 0, 1);
                                        var mean = br.ReadNcnnMatAsFloat32(bp.channels, 0, 0, 0, 1);
                                        var variance = br.ReadNcnnMatAsFloat32(bp.channels, 0, 0, 0, 1);
                                        var bias = br.ReadNcnnMatAsFloat32(bp.channels, 0, 0, 0, 1);
                                        phaseSw.Stop();
                                        readMs += phaseSw.ElapsedMilliseconds;

                                        var eps = layer.GetFloat(1, 0f);
                                        var a = new float[bp.channels];
                                        var b = new float[bp.channels];
                                        for (var i = 0; i < bp.channels; i++)
                                        {
                                            var sqrtVar = Mathf.Sqrt(variance[i] + eps);
                                            if (Mathf.Abs(sqrtVar) < 1e-8f)
                                                sqrtVar = 1e-4f;
                                            b[i] = slope[i] / sqrtVar;
                                            a[i] = bias[i] - slope[i] * mean[i] / sqrtVar;
                                        }

                                        phaseSw.Restart();
                                        var packs = (bp.channels + 3) / 4;
                                        var a4 = NcnnGraphSession.PackBiasToO4(a, bp.channels, packs);
                                        var b4 = NcnnGraphSession.PackBiasToO4(b, bp.channels, packs);
                                        bp.biasA = NcnnGraphSession.NewBuffer(a);
                                        bp.scaleB = NcnnGraphSession.NewBuffer(b);
                                        bp.biasA4 = new ComputeBuffer(a4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                        bp.scaleB4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                        bp.biasA4.SetData(a4);
                                        bp.scaleB4.SetData(b4);
                                        phaseSw.Stop();
                                        uploadMs += phaseSw.ElapsedMilliseconds;
                                        packMs += phaseSw.ElapsedMilliseconds;

                                        owner._batchNorm[layer.name] = bp;
                                        return new NcnnGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }
        public override void ExecuteBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out _,
                    out _))
            {
                ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            if (!owner._batchNorm.TryGetValue(layer.name, out var bp) || bp.biasA == null || bp.scaleB == null)
                throw new InvalidOperationException("BatchNorm not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = NcnnGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("BatchNorm source not found: " + layer.name);

            var outTensor = owner.RentTempTensorBuffer(srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c);
            owner.Ops.ScaleBuf(srcBuf, srcView, bp.scaleB, bp.channels, true, bp.biasA, outTensor.buffer);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: srcView.dims <= 3,
                textureBlobs,
                textureShapes,
                bufferBlobs,
                bufferRefs,
                bufferViews,
                tempOwned);
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteRenderTexturePath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferViews = context.bufferViews;

            if (!owner._batchNorm.TryGetValue(layer.name, out var bp) || bp.biasA4 == null || bp.scaleB4 == null)
                throw new InvalidOperationException("BatchNorm not found: " + layer.name);
            if (!owner.TryGetPack4Texture(layer.bottomNames[0], textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape))
                throw new InvalidOperationException("BatchNorm render-texture path requires pack4 texture input: " + layer.name);
            if (srcShape.dims != 3)
                throw new InvalidOperationException("BatchNorm expects dims=3 tensor input: " + layer.name);

            var outRt = owner.RentTempArray(srcTex.width, srcTex.height, srcTex.packs, RenderTextureFormat.ARGBHalf);
            owner.Ops.BatchNormPack4(srcTex.texture, bp.biasA4, bp.scaleB4, srcTex.packs, outRt);
            NcnnGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, srcShape);
            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
                        var cmd = context.commandBuffer;
                        var blobs = context.blobs;
                        var shapes = context.shapes;
                        var remaining = context.remaining;
                        var pinnedNames = context.pinnedNames;

                        do
                        {
                                                var src = NcnnGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
                                                var srcShape = NcnnGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
                                                if (!owner._batchNorm.TryGetValue(layer.name, out var bp) || bp.biasA4 == null || bp.scaleB4 == null)
                                                    throw new InvalidOperationException("BatchNorm not found: " + layer.name);

                                                var outArr = owner.RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                                                owner.Ops.BatchNormPack4(cmd, src.texture, bp.biasA4, bp.scaleB4, src.packs, outArr);
                                                blobs[layer.topNames[0]] = new NcnnGraphSession.CmdTensorRef
                                                {
                                                    texture = outArr,
                                                    width = src.width,
                                                    height = src.height,
                                                    packs = src.packs,
                                                    refs = 1,
                                                    owned = true
                                                };
                                                if (shapes != null)
                                                    shapes[layer.topNames[0]] = srcShape;
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                                                continue;
                        } while (false);
        }
    }
}
