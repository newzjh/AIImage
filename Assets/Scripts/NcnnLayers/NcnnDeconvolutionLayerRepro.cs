using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnDeconvolutionLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnDeconvolutionLayerRepro() : base(NcnnLayerTypes.Deconvolution, supportsBufferPath: true, supportsCommandBufferPath: false) { }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br) => owner.LoadDeconvolutionLayer(layer, br);
        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context) => owner.ExecuteDeconvolutionBufferLayer(layer, context);
    }

    public partial class NcnnRepro
    {
        internal LayerLoadMetrics LoadDeconvolutionLayer(NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

                            var pack = new DeconvPack();
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
                            pack.activationSlope = ParseLeakySlope(layer);

                            var kernelArea = Mathf.Max(1, pack.kernelW * pack.kernelH);
                            pack.inC = Mathf.Max(1, (pack.weightSize * pack.group) / Mathf.Max(1, pack.outC * kernelArea));

                            phaseSw.Restart();
                            var w = ReadPackedOrRawWeightArray(br, pack.weightSize, layer.name);
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

                            _deconv[layer.name] = pack;
                            return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        internal void ExecuteDeconvolutionBufferLayer(NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                                    if (!_deconv.TryGetValue(layer.name, out var deconv))
                                        throw new InvalidOperationException("Deconvolution not found: " + layer.name);

                                    var srcBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                    var srcTensor = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                    if (srcBuf == null || srcTensor == null || srcTensor.dims != 3)
                                        throw new InvalidOperationException("Deconvolution expects dims=3 tensor input: " + layer.name);

                                    var outW = ComputeDeconvOut(srcTensor.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
                                    var outH = ComputeDeconvOut(srcTensor.h, deconv.kernelH, deconv.dilationH, deconv.strideH, deconv.padTop, deconv.padBottom, deconv.outputPadBottom);
                                    var outTensor = RentTempTensorBuffer(3, outW, outH, 1, deconv.outC);
                                    _ops.Deconvolution(
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

                                    bufferBlobs[layer.topNames[0]] = outTensor.buffer;
                                    bufferViews[layer.topNames[0]] = outTensor;
                                    tempOwned.Add(outTensor);
                                    Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }
    }
}
