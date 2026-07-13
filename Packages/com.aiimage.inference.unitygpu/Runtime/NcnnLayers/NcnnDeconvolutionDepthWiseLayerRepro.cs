using System;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnDeconvolutionDepthWiseLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnDeconvolutionDepthWiseLayerRepro()
            : base(NcnnLayerTypes.DeconvolutionDepthWise, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
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
            pack.inPacks = Mathf.Max(1, Mathf.CeilToInt(pack.inC / 4f));
            pack.outPacks = Mathf.Max(1, Mathf.CeilToInt(pack.outC / 4f));

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

            if (CanUsePack4DepthWiseDeconvolution(pack))
            {
                phaseSw.Restart();
                var w4 = NcnnRepro.PackDepthWiseWeightsToP4KhKw(w, pack.outC, pack.kernelW, pack.kernelH, pack.outPacks);
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
            if (!owner.ShouldForceCurrentLayerBufferPath()
                && context.textureBlobs != null
                && context.textureBlobs.TryGetValue(layer.bottomNames[0], out var src)
                && src != null
                && src.texture != null)
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

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._deconv.TryGetValue(layer.name, out var deconv))
                throw new InvalidOperationException("DeconvolutionDepthWise not found: " + layer.name);
            if (!textureBlobs.TryGetValue(layer.bottomNames[0], out var src) || src == null || src.texture == null)
                throw new InvalidOperationException("DeconvolutionDepthWise render-texture path requires texture input: " + layer.name);

            var srcShape = NcnnRepro.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
            ValidatePack4DepthWiseTexture(layer.name, deconv, src, srcShape, isCommandBuffer: false);

            var outW = NcnnRepro.ComputeDeconvOut(srcShape.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
            var outH = NcnnRepro.ComputeDeconvOut(srcShape.h, deconv.kernelH, deconv.dilationH, deconv.strideH, deconv.padTop, deconv.padBottom, deconv.outputPadBottom);
            var outShape = new NcnnRepro.BufferShape(3, outW, outH, 1, deconv.outC);
            RenderTexture output = null;
            try
            {
                output = owner.RentTempArray(outW, outH, deconv.outPacks, NcnnRepro.ResolveTensorTextureFormat(outShape.dims));
                owner.Ops.DeconvolutionDepthWisePack4(
                    src.texture,
                    deconv.packedWeight4,
                    deconv.packedBias4,
                    deconv.inC,
                    deconv.outC,
                    deconv.group,
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
                    output);
                NcnnRepro.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], output, outShape, outShape);
                output = null;
            }
            finally
            {
                if (output != null)
                    owner.ReturnTempArray(output);
            }

            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var srcShape = NcnnRepro.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
            if (!owner._deconv.TryGetValue(layer.name, out var deconv))
                throw new InvalidOperationException("DeconvolutionDepthWise not found: " + layer.name);
            ValidatePack4DepthWiseTexture(layer.name, deconv, src, srcShape, isCommandBuffer: true);

            var outW = NcnnRepro.ComputeDeconvOut(srcShape.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
            var outH = NcnnRepro.ComputeDeconvOut(srcShape.h, deconv.kernelH, deconv.dilationH, deconv.strideH, deconv.padTop, deconv.padBottom, deconv.outputPadBottom);
            var outShape = new NcnnRepro.BufferShape(3, outW, outH, 1, deconv.outC);
            ComputeTexture output = null;
            try
            {
                output = owner.RentTempArray(cmd, outW, outH, deconv.outPacks, NcnnRepro.ResolveTensorTextureFormat(outShape.dims));
                owner.Ops.DeconvolutionDepthWisePack4(
                    cmd,
                    src.texture,
                    deconv.packedWeight4,
                    deconv.packedBias4,
                    deconv.inC,
                    deconv.outC,
                    deconv.group,
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
                    output);
                blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(output, outShape, outShape, owned: true);
                output = null;
                if (shapes != null)
                    shapes[layer.topNames[0]] = outShape;
            }
            finally
            {
                if (output != null)
                    owner.ReturnTempArray(cmd, output);
            }

            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool CanUsePack4DepthWiseDeconvolution(NcnnRepro.DeconvPack deconv)
        {
            return deconv != null
                && deconv.inC > 0
                && deconv.outC > 0
                && deconv.group > 0
                && deconv.group == deconv.inC
                && deconv.outC % deconv.group == 0
                && deconv.kernelW > 0
                && deconv.kernelH > 0
                && deconv.strideW > 0
                && deconv.strideH > 0
                && deconv.dilationW > 0
                && deconv.dilationH > 0;
        }

        private static void ValidatePack4DepthWiseTexture(
            string layerName,
            NcnnRepro.DeconvPack deconv,
            NcnnRepro.TensorRef src,
            NcnnRepro.BufferShape srcShape,
            bool isCommandBuffer)
        {
            if (src == null || src.texture == null)
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "source texture missing"));
            if (src.texture.dimension != TextureDimension.Tex2DArray || NcnnRepro.IsStrictLinearMatTexture(src))
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "source must be pack4 texture-array"));
            ValidatePack4DepthWiseShape(layerName, deconv, srcShape, src.packs, isCommandBuffer);
            var storageShape = NcnnRepro.GetTextureStorageShape(src, srcShape);
            if (!NcnnRepro.BufferShapeEquals(srcShape, storageShape))
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "logical/storage shape mismatch"));
        }

        private static void ValidatePack4DepthWiseTexture(
            string layerName,
            NcnnRepro.DeconvPack deconv,
            NcnnRepro.CmdTensorRef src,
            NcnnRepro.BufferShape srcShape,
            bool isCommandBuffer)
        {
            if (src == null || src.texture == null)
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "source texture missing"));
            if (src.texture.dimension != TextureDimension.Tex2DArray || NcnnRepro.IsStrictLinearMatTexture(src))
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "source must be pack4 texture-array"));
            ValidatePack4DepthWiseShape(layerName, deconv, srcShape, src.packs, isCommandBuffer);
            var storageShape = NcnnRepro.GetCmdStorageShape(src, srcShape);
            if (!NcnnRepro.BufferShapeEquals(srcShape, storageShape))
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "logical/storage shape mismatch"));
        }

        private static void ValidatePack4DepthWiseShape(
            string layerName,
            NcnnRepro.DeconvPack deconv,
            NcnnRepro.BufferShape srcShape,
            int packs,
            bool isCommandBuffer)
        {
            if (!CanUsePack4DepthWiseDeconvolution(deconv) || deconv.packedWeight4 == null || deconv.packedBias4 == null)
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "unsupported depthwise/group parameters"));
            if (srcShape.dims != 3 || srcShape.w <= 0 || srcShape.h <= 0 || srcShape.c != deconv.inC)
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "source shape must be dims=3 and match input channels"));
            if (packs != Mathf.Max(1, Mathf.CeilToInt(srcShape.c / 4f)))
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "source pack count mismatch"));
        }

        private static string BuildUnsupportedMessage(string layerName, NcnnRepro.BufferShape shape, bool isCommandBuffer, string reason)
        {
            return "DeconvolutionDepthWise " + (isCommandBuffer ? "command-buffer" : "render-texture")
                + " pack4 path unsupported: " + layerName
                + " | reason=" + reason
                + " | shape=d" + shape.dims + ":" + shape.w + "x" + shape.h + "x" + shape.d + "x" + shape.c;
        }
    }
}
