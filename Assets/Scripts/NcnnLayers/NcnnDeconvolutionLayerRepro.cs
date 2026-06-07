using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnDeconvolutionLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnDeconvolutionLayerRepro() : base(NcnnLayerTypes.Deconvolution, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var pack = new NcnnRepro.DeconvPack();
                                        pack.outC = layer.GetInt(0, 0);
                                        pack.group = Mathf.Max(1, layer.GetInt(7, 1));
                                        pack.kernelW = layer.GetInt(1, 0);
                                        pack.kernelH = layer.GetInt(11, pack.kernelW);
                                        pack.dilationW = layer.GetInt(2, 1);
                                        pack.dilationH = layer.GetInt(12, pack.dilationW);
                                        pack.strideW = layer.GetInt(3, 1);
                                        pack.strideH = layer.GetInt(13, pack.strideW);
                                        pack.padLeft = layer.GetInt(4, 0);
                                        pack.padRight = layer.GetInt(15, pack.padLeft);
                                        pack.padTop = layer.GetInt(14, pack.padLeft);
                                        pack.padBottom = layer.GetInt(16, pack.padTop);
                                        pack.outputPadRight = layer.GetInt(18, 0);
                                        pack.outputPadBottom = layer.GetInt(19, pack.outputPadRight);
                                        pack.biasTerm = layer.GetInt(5, 0);
                                        pack.weightSize = layer.GetInt(6, 0);
                                        pack.activationType = layer.GetInt(9, 0);
                                        pack.activationSlope = NcnnRepro.ParseLeakySlope(layer);

                                        var kernelArea = Mathf.Max(1, pack.kernelW * pack.kernelH);
                                        pack.inC = Mathf.Max(1, (pack.weightSize * pack.group) / Mathf.Max(1, pack.outC * kernelArea));
                                        pack.inPacks = (pack.inC + 3) / 4;
                                        pack.outPacks = (pack.outC + 3) / 4;

                                        phaseSw.Restart();
                                        var w = NcnnRepro.ReadPackedOrRawWeightArray(br, pack.weightSize, layer.name);
                                        var b = pack.biasTerm != 0 ? br.ReadFloat32Array(pack.outC) : new float[pack.outC];
                                        phaseSw.Stop();
                                        readMs += phaseSw.ElapsedMilliseconds;

                                        phaseSw.Restart();
                                        pack.rawWeight = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                                        pack.rawBias = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                                        pack.rawWeight.SetData(w);
                                        pack.rawBias.SetData(b);
                                        phaseSw.Stop();
                                        uploadMs += phaseSw.ElapsedMilliseconds;

                                        var canUseGeneralTexturePack = owner.EnableGeneralTextureConvolution
                                                                       && pack.group == 1
                                                                       && pack.kernelW > 0
                                                                       && pack.kernelH == pack.kernelW;
                                        if (canUseGeneralTexturePack)
                                        {
                                            phaseSw.Restart();
                                            var w4 = NcnnRepro.PackWeightsToO4I4K(w, pack.outC, pack.inC, pack.kernelW, pack.outPacks, pack.inPacks);
                                            var b4 = NcnnRepro.PackBiasToO4(b, pack.outC, pack.outPacks);
                                            pack.packedWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                            pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                            pack.packedWeight4.SetData(w4);
                                            pack.packedBias4.SetData(b4);
                                            phaseSw.Stop();
                                            packMs += phaseSw.ElapsedMilliseconds;
                                        }

                                        owner._deconv[layer.name] = pack;
                                        return new NcnnRepro.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }
        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (!owner._deconv.TryGetValue(layer.name, out var deconv))
                throw new InvalidOperationException("Deconvolution not found: " + layer.name);

            var canUseGeneralTexturePath = !owner.ShouldForceCurrentLayerBufferPath()
                                           && owner.EnableGeneralTextureConvolution
                                           && deconv.group == 1
                                           && deconv.packedWeight4 != null
                                           && deconv.packedBias4 != null
                                           && deconv.kernelW > 0
                                           && deconv.kernelH == deconv.kernelW;

            if (canUseGeneralTexturePath)
            {
                ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            if (!owner._deconv.TryGetValue(layer.name, out var deconv))
                throw new InvalidOperationException("Deconvolution not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcTensor = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcTensor == null || srcTensor.dims != 3)
                throw new InvalidOperationException("Deconvolution expects dims=3 tensor input: " + layer.name);

            var outW = NcnnRepro.ComputeDeconvOut(srcTensor.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
            var outH = NcnnRepro.ComputeDeconvOut(srcTensor.h, deconv.kernelH, deconv.dilationH, deconv.strideH, deconv.padTop, deconv.padBottom, deconv.outputPadBottom);
            var outTensor = owner.RentTempTensorBuffer(3, outW, outH, 1, deconv.outC);
            owner.Ops.Deconvolution(
                srcTensor,
                deconv.rawWeight,
                deconv.rawBias,
                deconv.outC,
                deconv.group,
                deconv.kernelW,
                deconv.kernelH,
                deconv.strideW,
                deconv.strideH,
                deconv.padLeft,
                deconv.padRight,
                deconv.padTop,
                deconv.padBottom,
                deconv.outputPadRight,
                deconv.outputPadBottom,
                deconv.dilationW,
                deconv.dilationH,
                deconv.activationType,
                deconv.activationSlope,
                outTensor);
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
        }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferViews = context.bufferViews;

            if (!owner._deconv.TryGetValue(layer.name, out var deconv))
                throw new InvalidOperationException("Deconvolution not found: " + layer.name);

            NcnnRepro.TensorRef src;
            RenderTexture tempInputTex = null;
            if (!textureBlobs.TryGetValue(layer.bottomNames[0], out src) || src == null || src.texture == null)
            {
                if (!bufferBlobs.TryGetValue(layer.bottomNames[0], out var deconvInputBuf) || deconvInputBuf == null)
                    throw new InvalidOperationException("Deconvolution source not found: " + layer.name);
                var deconvInputView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                if (deconvInputView == null || deconvInputView.dims != 3)
                    throw new InvalidOperationException("Deconvolution texture path expects dims=3 buffer input: " + layer.name);

                tempInputTex = owner.RentTempArray(deconvInputView.w, deconvInputView.h, deconv.inPacks, RenderTextureFormat.ARGBHalf);
                owner.Ops.FillPack4FromBufferCHW(deconvInputBuf, deconvInputView.w, deconvInputView.h, deconvInputView.c, tempInputTex);
                src = new NcnnRepro.TensorRef
                {
                    texture = tempInputTex,
                    width = deconvInputView.w,
                    height = deconvInputView.h,
                    packs = deconv.inPacks,
                    refs = 1,
                    owned = false
                };
            }

            var outWTex = NcnnRepro.ComputeDeconvOut(src.width, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
            var outHTex = NcnnRepro.ComputeDeconvOut(src.height, deconv.kernelH, deconv.dilationH, deconv.strideH, deconv.padTop, deconv.padBottom, deconv.outputPadBottom);
            var outRt = owner.RentTempArray(outWTex, outHTex, deconv.outPacks, RenderTextureFormat.ARGBHalf);
            owner.Ops.DeconvolutionPack4General(
                src.texture,
                deconv.inPacks,
                deconv.packedWeight4,
                deconv.packedBias4,
                deconv.outPacks,
                deconv.kernelW,
                deconv.kernelH,
                deconv.strideW,
                deconv.strideH,
                deconv.padLeft,
                deconv.padTop,
                deconv.dilationW,
                deconv.dilationH,
                deconv.activationType,
                deconv.activationSlope,
                outRt);

            textureBlobs[layer.topNames[0]] = new NcnnRepro.TensorRef
            {
                texture = outRt,
                width = outWTex,
                height = outHTex,
                packs = deconv.outPacks,
                refs = 1,
                owned = true
            };
            textureShapes[layer.topNames[0]] = new NcnnRepro.BufferShape(3, outWTex, outHTex, 1, deconv.outC);
            if (tempInputTex != null)
                owner.ReturnTempArray(tempInputTex);
            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var outW = NcnnRepro.ComputeDeconvOut(srcShape.w, layer.GetInt(1, 0), layer.GetInt(2, 1), layer.GetInt(3, 1), layer.GetInt(4, 0), layer.GetInt(15, layer.GetInt(4, 0)), layer.GetInt(18, 0));
            var outH = NcnnRepro.ComputeDeconvOut(srcShape.h, layer.GetInt(11, layer.GetInt(1, 0)), layer.GetInt(12, layer.GetInt(2, 1)), layer.GetInt(13, layer.GetInt(3, 1)), layer.GetInt(14, layer.GetInt(4, 0)), layer.GetInt(16, layer.GetInt(14, layer.GetInt(4, 0))), layer.GetInt(19, layer.GetInt(18, 0)));
            owner.PublishCmdPlaceholder(
                cmd,
                layer.topNames[0],
                new NcnnRepro.BufferShape(3, outW, outH, 1, Mathf.Max(1, layer.GetInt(0, 0))),
                blobs,
                shapes);
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }
    }
}
