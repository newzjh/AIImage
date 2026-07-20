using System;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Ncnn
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnDeconvolutionDepthWiseLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnDeconvolutionDepthWiseLayerRepro()
            : base(NcnnLayerTypes.DeconvolutionDepthWise, supportsBufferPath: true, supportsCommandBufferPath: true)
        {
        }

        public override NcnnGraphSession.LayerLoadMetrics LoadLayer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            var pack = new NcnnGraphSession.DeconvPack
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
                activationSlope = NcnnGraphSession.ParseLeakySlope(layer)
            };

            var kernelArea = Mathf.Max(1, pack.kernelW * pack.kernelH);
            pack.inC = Mathf.Max(1, (pack.weightSize * pack.group) / Mathf.Max(1, pack.outC * kernelArea));
            pack.inPacks = Mathf.Max(1, Mathf.CeilToInt(pack.inC / 4f));
            pack.outPacks = Mathf.Max(1, Mathf.CeilToInt(pack.outC / 4f));

            phaseSw.Restart();
            var w = NcnnGraphSession.ReadPackedOrRawWeightArray(br, pack.weightSize, layer.name);
            var b = pack.biasTerm != 0 ? br.ReadFloat32Array(pack.outC) : new float[pack.outC];
            phaseSw.Stop();
            readMs += phaseSw.ElapsedMilliseconds;

            phaseSw.Restart();
            pack.rawWeight = NcnnGraphSession.NewBuffer(w);
            pack.rawBias = NcnnGraphSession.NewBuffer(b);
            phaseSw.Stop();
            uploadMs += phaseSw.ElapsedMilliseconds;

            if (CanUsePack4DepthWiseDeconvolution(pack))
            {
                phaseSw.Restart();
                var w4 = NcnnGraphSession.PackDepthWiseWeightsToP4KhKw(w, pack.outC, pack.kernelW, pack.kernelH, pack.outPacks);
                var b4 = NcnnGraphSession.PackBiasToO4(b, pack.outC, pack.outPacks);
                pack.packedWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                pack.packedWeight4.SetData(w4);
                pack.packedBias4.SetData(b4);
                phaseSw.Stop();
                packMs += phaseSw.ElapsedMilliseconds;
            }

            owner._deconv[layer.name] = pack;
            return new NcnnGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        public override void ExecuteBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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

            if (!owner._deconv.TryGetValue(layer.name, out var deconv))
                throw new InvalidOperationException("DeconvolutionDepthWise not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcTensor = NcnnGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcTensor == null || srcTensor.dims != 3)
                throw new InvalidOperationException("DeconvolutionDepthWise expects dims=3 tensor input: " + layer.name);

            var outW = NcnnGraphSession.ComputeDeconvOut(srcTensor.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
            var outH = NcnnGraphSession.ComputeDeconvOut(srcTensor.h, deconv.kernelH, deconv.dilationH, deconv.strideH, deconv.padTop, deconv.padBottom, deconv.outputPadBottom);
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

        public override void ExecuteRenderTexturePath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._deconv.TryGetValue(layer.name, out var deconv))
                throw new InvalidOperationException("DeconvolutionDepthWise not found: " + layer.name);
            if (!textureBlobs.TryGetValue(layer.bottomNames[0], out var src) || src == null || src.texture == null)
                throw new InvalidOperationException("DeconvolutionDepthWise render-texture path requires texture input: " + layer.name);

            var srcShape = NcnnGraphSession.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
            ValidatePack4DepthWiseTexture(layer.name, deconv, src, srcShape, isCommandBuffer: false);

            var outW = NcnnGraphSession.ComputeDeconvOut(srcShape.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
            var outH = NcnnGraphSession.ComputeDeconvOut(srcShape.h, deconv.kernelH, deconv.dilationH, deconv.strideH, deconv.padTop, deconv.padBottom, deconv.outputPadBottom);
            var outShape = new NcnnGraphSession.BufferShape(3, outW, outH, 1, deconv.outC);
            RenderTexture output = null;
            try
            {
                output = owner.RentTempArray(outW, outH, deconv.outPacks, NcnnGraphSession.ResolveTensorTextureFormat(outShape.dims));
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
                NcnnGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], output, outShape, outShape);
                output = null;
            }
            finally
            {
                if (output != null)
                    owner.ReturnTempArray(output);
            }

            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var srcShape = NcnnGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            var src = NcnnGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            if (!owner._deconv.TryGetValue(layer.name, out var deconv))
                throw new InvalidOperationException("DeconvolutionDepthWise not found: " + layer.name);
            if (!SupportsCommandBufferPack4(deconv, out var reason))
                throw new InvalidOperationException(BuildCmdUnsupportedMessage(layer, src, srcShape, reason));
            if (!MatchesCommandBufferPack4Source(src, srcShape, deconv))
                throw new InvalidOperationException(BuildCmdUnsupportedMessage(layer, src, srcShape, "source is not a matching Pack4 texture-array"));

            var outW = NcnnGraphSession.ComputeDeconvOut(srcShape.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
            var outH = NcnnGraphSession.ComputeDeconvOut(srcShape.h, deconv.kernelH, deconv.dilationH, deconv.strideH, deconv.padTop, deconv.padBottom, deconv.outputPadBottom);
            var outShape = new NcnnGraphSession.BufferShape(3, outW, outH, 1, deconv.outC);
            ComputeTexture output = null;
            try
            {
                output = owner.RentTempArray(cmd, outW, outH, deconv.outPacks, NcnnGraphSession.ResolveTensorTextureFormat(outShape.dims));
                if (CanUsePack4DepthWiseDeconvolution(deconv) && (deconv.outC & 3) == 0 && deconv.packedWeight4 != null && deconv.packedBias4 != null)
                {
                    owner.Ops.DeconvolutionDepthWisePack4(cmd, src.texture, deconv.packedWeight4, deconv.packedBias4, deconv.inC, deconv.outC, deconv.group, deconv.outPacks, deconv.kernelW, deconv.kernelH, deconv.strideW, deconv.strideH, deconv.padLeft, deconv.padTop, deconv.dilationW, deconv.dilationH, deconv.activationType, deconv.activationSlope, output);
                }
                else
                {
                    owner.Ops.Deconvolution2dGroupPack4(
                        cmd, src.texture, deconv.rawWeight, deconv.rawBias, deconv.inC, deconv.outC, deconv.group,
                        deconv.kernelW, deconv.kernelH, deconv.strideW, deconv.strideH, deconv.padLeft, deconv.padTop,
                        deconv.dilationW, deconv.dilationH, deconv.activationType, deconv.activationSlope, output);
                }
                blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorRef(output, outShape, outShape, owned: true);
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

        private static bool SupportsCommandBufferPack4(NcnnGraphSession.DeconvPack deconv, out string reason)
        {
            reason = null;
            if (deconv == null || deconv.rawWeight == null || deconv.rawBias == null)
                reason = "immutable scalar weights/bias are unavailable";
            else if (deconv.inC <= 0 || deconv.outC <= 0 || deconv.group <= 0 || deconv.inC % deconv.group != 0 || deconv.outC % deconv.group != 0)
                reason = "group must divide positive input and output channels";
            else if (deconv.kernelW <= 0 || deconv.kernelH <= 0 || deconv.strideW <= 0 || deconv.strideH <= 0 || deconv.dilationW <= 0 || deconv.dilationH <= 0)
                reason = "kernel, stride, and dilation must be positive";
            else if (deconv.padLeft < 0 || deconv.padRight < 0 || deconv.padTop < 0 || deconv.padBottom < 0 || deconv.outputPadRight < 0 || deconv.outputPadBottom < 0)
                reason = "negative/auto padding or output padding is not implemented";
            else if (deconv.activationType != 0 && deconv.activationType != 1 && deconv.activationType != 2 && deconv.activationType != 4)
                reason = "activation supports only none, ReLU, LeakyReLU, or Sigmoid";
            else if (deconv.weightSize != deconv.outC * (deconv.inC / deconv.group) * deconv.kernelW * deconv.kernelH)
                reason = "weight_data_size does not match the grouped OIHW profile";
            return reason == null;
        }

        private static bool MatchesCommandBufferPack4Source(NcnnGraphSession.CmdTensorRef src, NcnnGraphSession.BufferShape shape, NcnnGraphSession.DeconvPack deconv)
        {
            return src != null
                && src.texture != null
                && src.texture.dimension == TextureDimension.Tex2DArray
                && !NcnnGraphSession.IsStrictLinearMatTexture(src)
                && shape.dims == 3
                && shape.d == 1
                && shape.c == deconv.inC
                && src.width == shape.w
                && src.height == shape.h
                && src.packs == deconv.inPacks
                && NcnnGraphSession.BufferShapeEquals(shape, NcnnGraphSession.GetCmdStorageShape(src, shape));
        }

        private static string BuildCmdUnsupportedMessage(NcnnParamModel.Layer layer, NcnnGraphSession.CmdTensorRef src, NcnnGraphSession.BufferShape logicalShape, string reason)
        {
            var storageShape = src != null ? NcnnGraphSession.GetCmdStorageShape(src, logicalShape) : default;
            return "DeconvolutionDepthWise CommandBuffer Pack4 rejected"
                + " | layer=" + (layer?.name ?? string.Empty)
                + " | blob=" + (layer?.bottomNames != null && layer.bottomNames.Length > 0 ? layer.bottomNames[0] : string.Empty)
                + " | logical_shape=" + logicalShape
                + " | storage_shape=" + storageShape
                + " | layout=Packed4"
                + " | dtype=" + (src?.texture != null ? src.texture.format.ToString() : "unknown")
                + " | reason=" + reason
                + " | rejected_fallback=Buffer/materialize-from-buffer/placeholder";
        }

        private static bool CanUsePack4DepthWiseDeconvolution(NcnnGraphSession.DeconvPack deconv)
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
            NcnnGraphSession.DeconvPack deconv,
            NcnnGraphSession.TensorRef src,
            NcnnGraphSession.BufferShape srcShape,
            bool isCommandBuffer)
        {
            if (src == null || src.texture == null)
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "source texture missing"));
            if (src.texture.dimension != TextureDimension.Tex2DArray || NcnnGraphSession.IsStrictLinearMatTexture(src))
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "source must be pack4 texture-array"));
            ValidatePack4DepthWiseShape(layerName, deconv, srcShape, src.packs, isCommandBuffer);
            var storageShape = NcnnGraphSession.GetTextureStorageShape(src, srcShape);
            if (!NcnnGraphSession.BufferShapeEquals(srcShape, storageShape))
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "logical/storage shape mismatch"));
        }

        private static void ValidatePack4DepthWiseTexture(
            string layerName,
            NcnnGraphSession.DeconvPack deconv,
            NcnnGraphSession.CmdTensorRef src,
            NcnnGraphSession.BufferShape srcShape,
            bool isCommandBuffer)
        {
            if (src == null || src.texture == null)
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "source texture missing"));
            if (src.texture.dimension != TextureDimension.Tex2DArray || NcnnGraphSession.IsStrictLinearMatTexture(src))
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "source must be pack4 texture-array"));
            ValidatePack4DepthWiseShape(layerName, deconv, srcShape, src.packs, isCommandBuffer);
            var storageShape = NcnnGraphSession.GetCmdStorageShape(src, srcShape);
            if (!NcnnGraphSession.BufferShapeEquals(srcShape, storageShape))
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "logical/storage shape mismatch"));
        }

        private static void ValidatePack4DepthWiseShape(
            string layerName,
            NcnnGraphSession.DeconvPack deconv,
            NcnnGraphSession.BufferShape srcShape,
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

        private static string BuildUnsupportedMessage(string layerName, NcnnGraphSession.BufferShape shape, bool isCommandBuffer, string reason)
        {
            return "DeconvolutionDepthWise " + (isCommandBuffer ? "command-buffer" : "render-texture")
                + " pack4 path unsupported: " + layerName
                + " | reason=" + reason
                + " | shape=d" + shape.dims + ":" + shape.w + "x" + shape.h + "x" + shape.d + "x" + shape.c;
        }
    }
}
