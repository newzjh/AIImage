using System;
using System.Diagnostics;
using UnityEngine;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnDeconvolutionDepthWiseLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnDeconvolutionDepthWiseLayerRepro()
            : base(NcnnLayerTypes.DeconvolutionDepthWise, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
            PreferredPath = NcnnLayerPathPreference.Buffer;
        }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            var pack = new NcnnRepro.DeconvPack
            {
                outC = layer.GetInt(0, 0),
                group = Mathf.Max(1, layer.GetInt(7, 1)),
                kernelW = layer.GetInt(1, 0),
                kernelH = layer.GetInt(11, layer.GetInt(1, 0)),
                dilationW = layer.GetInt(2, 1),
                dilationH = layer.GetInt(12, layer.GetInt(2, 1)),
                strideW = layer.GetInt(3, 1),
                strideH = layer.GetInt(13, layer.GetInt(3, 1)),
                padLeft = layer.GetInt(4, 0),
                padRight = layer.GetInt(15, layer.GetInt(4, 0)),
                padTop = layer.GetInt(14, layer.GetInt(4, 0)),
                padBottom = layer.GetInt(16, layer.GetInt(14, layer.GetInt(4, 0))),
                outputPadRight = layer.GetInt(18, 0),
                outputPadBottom = layer.GetInt(19, layer.GetInt(18, 0)),
                biasTerm = layer.GetInt(5, 0),
                weightSize = layer.GetInt(6, 0),
                activationType = layer.GetInt(9, 0),
                activationSlope = NcnnRepro.ParseLeakySlope(layer)
            };

            var kernelArea = Mathf.Max(1, pack.kernelW * pack.kernelH);
            pack.inC = Mathf.Max(1, (pack.weightSize * pack.group) / Mathf.Max(1, pack.outC * kernelArea));

            phaseSw.Restart();
            var w = NcnnRepro.ReadPackedOrRawWeightArray(br, pack.weightSize, layer.name);
            var b = pack.biasTerm != 0 ? br.ReadFloat32Array(pack.outC) : new float[pack.outC];
            phaseSw.Stop();
            readMs += phaseSw.ElapsedMilliseconds;

            phaseSw.Restart();
            pack.rawWeight = NcnnRepro.NewBuffer(w);
            pack.rawBias = NcnnRepro.NewBuffer(b);
            phaseSw.Stop();
            uploadMs += phaseSw.ElapsedMilliseconds;

            owner._deconv[layer.name] = pack;
            return new NcnnRepro.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                throw new InvalidOperationException("DeconvolutionDepthWise not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcTensor = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcTensor == null || srcTensor.dims != 3)
                throw new InvalidOperationException("DeconvolutionDepthWise expects dims=3 tensor input: " + layer.name);

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
