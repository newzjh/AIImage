using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisDeconvolutionLayer : AexisBaseLayer
    {
        public AexisDeconvolutionLayer() : base(AexisLayerTypes.Deconvolution, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var pack = new AexisGraphSession.DeconvPack();
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
                                        pack.activationSlope = AexisGraphSession.ParseLeakySlope(layer);

                                        var kernelArea = Mathf.Max(1, pack.kernelW * pack.kernelH);
                                        pack.inC = Mathf.Max(1, (pack.weightSize * pack.group) / Mathf.Max(1, pack.outC * kernelArea));
                                        pack.inPacks = (pack.inC + 3) / 4;
                                        pack.outPacks = (pack.outC + 3) / 4;

                                        phaseSw.Restart();
                                        var w = AexisGraphSession.ReadPackedOrRawWeightArray(br, pack.weightSize, layer.name);
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
                                                                       && pack.kernelH > 0;
                                        if (canUseGeneralTexturePack)
                                        {
                                            phaseSw.Restart();
                                            var w4 = AexisGraphSession.PackWeightsToO4I4K2D(w, pack.outC, pack.inC, pack.kernelW, pack.kernelH, pack.outPacks, pack.inPacks);
                                            var b4 = AexisGraphSession.PackBiasToO4(b, pack.outC, pack.outPacks);
                                            pack.packedWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                            pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                            pack.packedWeight4.SetData(w4);
                                            pack.packedBias4.SetData(b4);
                                            if (owner.UsesFp16WeightStorage)
                                                pack.packedWeight4Fp16 = AexisGraphSession.NewFp16Vector4Buffer(w4, "AexisGraphSession.DeconvPackedWeight4Fp16:" + layer.name);
                                            phaseSw.Stop();
                                            packMs += phaseSw.ElapsedMilliseconds;
                                        }

                                        owner._deconv[layer.name] = pack;
                                        return new AexisGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }
        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (!owner._deconv.TryGetValue(layer.name, out var deconv))
                throw new InvalidOperationException("Deconvolution not found: " + layer.name);

            var canUseGeneralTexturePath = !owner.ShouldForceCurrentLayerBufferPath()
                                           && owner.EnableGeneralTextureConvolution
                                           && deconv.group == 1
                                           && deconv.packedWeight4 != null
                                           && deconv.packedBias4 != null
                                           && deconv.kernelW > 0
                                           && deconv.kernelH > 0
                                           && context.textureBlobs != null
                                           && context.textureBlobs.TryGetValue(layer.bottomNames[0], out var src)
                                           && src != null
                                           && src.texture != null;

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
        public override void ExecuteComputeBufferPath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
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
            var srcTensor = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcTensor == null || srcTensor.dims != 3)
                throw new InvalidOperationException("Deconvolution expects dims=3 tensor input: " + layer.name);

            var outW = AexisGraphSession.ComputeDeconvOut(srcTensor.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
            var outH = AexisGraphSession.ComputeDeconvOut(srcTensor.h, deconv.kernelH, deconv.dilationH, deconv.strideH, deconv.padTop, deconv.padBottom, deconv.outputPadBottom);
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

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;

            if (!owner._deconv.TryGetValue(layer.name, out var deconv))
                throw new InvalidOperationException("Deconvolution not found: " + layer.name);
            if (!textureBlobs.TryGetValue(layer.bottomNames[0], out var srcTex) || srcTex == null || srcTex.texture == null)
                throw new InvalidOperationException("Deconvolution render-texture path requires texture input: " + layer.name);

            var srcShape = AexisGraphSession.GetTextureShape(textureShapes, srcTex, layer.bottomNames[0]);
            ValidatePack4DeconvolutionTexture(layer.name, deconv, srcTex, srcShape, isCommandBuffer: false);

            var outWTex = AexisGraphSession.ComputeDeconvOut(srcShape.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
            var outHTex = AexisGraphSession.ComputeDeconvOut(srcShape.h, deconv.kernelH, deconv.dilationH, deconv.strideH, deconv.padTop, deconv.padBottom, deconv.outputPadBottom);
            var outShape = new AexisGraphSession.BufferShape(3, outWTex, outHTex, 1, deconv.outC);
            RenderTexture outRt = null;
            try
            {
                outRt = owner.RentTempArray(outWTex, outHTex, deconv.outPacks, RenderTextureFormat.ARGBHalf);
                owner.Ops.SetFp16ConvWeights(owner.UsesFp16WeightStorage ? deconv.packedWeight4Fp16 : null);
                owner.Ops.DeconvolutionPack4General(
                    srcTex.texture,
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

                AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape, outShape);
                outRt = null;
            }
            finally
            {
                if (outRt != null)
                    owner.ReturnTempArray(outRt);
            }
            owner.Consume(textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            var srcShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            if (!owner._deconv.TryGetValue(layer.name, out var deconv))
                throw new InvalidOperationException("Deconvolution not found: " + layer.name);
            var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            if (!SupportsCommandBufferPack4(deconv, out var reason))
                throw new InvalidOperationException(BuildCmdUnsupportedMessage(layer, src, srcShape, reason));
            if (!MatchesCommandBufferPack4Source(src, srcShape, deconv))
                throw new InvalidOperationException(BuildCmdUnsupportedMessage(layer, src, srcShape, "source is not a matching Pack4 texture-array"));

            var outW = AexisGraphSession.ComputeDeconvOut(srcShape.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
            var outH = AexisGraphSession.ComputeDeconvOut(srcShape.h, deconv.kernelH, deconv.dilationH, deconv.strideH, deconv.padTop, deconv.padBottom, deconv.outputPadBottom);
            var outShape = new AexisGraphSession.BufferShape(3, outW, outH, 1, Mathf.Max(1, deconv.outC));

            ComputeTexture outArr = null;
            try
            {
                outArr = owner.RentTempArray(cmd, outW, outH, deconv.outPacks, AexisGraphSession.ResolveTensorTextureFormat(outShape.dims));
                if (deconv.group == 1
                    && (deconv.outC & 3) == 0
                    && deconv.packedWeight4 != null
                    && deconv.packedBias4 != null)
                {
                    owner.Ops.SetFp16ConvWeights(owner.UsesFp16WeightStorage ? deconv.packedWeight4Fp16 : null);
                    owner.Ops.DeconvolutionPack4General(cmd, src.texture, deconv.inPacks, deconv.packedWeight4, deconv.packedBias4, deconv.outPacks, deconv.kernelW, deconv.kernelH, deconv.strideW, deconv.strideH, deconv.padLeft, deconv.padTop, deconv.dilationW, deconv.dilationH, deconv.activationType, deconv.activationSlope, outArr);
                }
                else
                {
                    owner.Ops.SetFp16ConvWeights(null);
                    owner.Ops.Deconvolution2dGroupPack4(
                        cmd, src.texture, deconv.rawWeight, deconv.rawBias, deconv.inC, deconv.outC, deconv.group,
                        deconv.kernelW, deconv.kernelH, deconv.strideW, deconv.strideH, deconv.padLeft, deconv.padTop,
                        deconv.dilationW, deconv.dilationH, deconv.activationType, deconv.activationSlope, outArr);
                }
                blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(outArr, outShape, outShape, owned: true);
                outArr = null;
                if (shapes != null)
                    shapes[layer.topNames[0]] = outShape;
            }
            finally
            {
                if (outArr != null)
                    owner.ReturnTempArray(cmd, outArr);
            }
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool SupportsCommandBufferPack4(AexisGraphSession.DeconvPack deconv, out string reason)
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

        private static bool MatchesCommandBufferPack4Source(AexisGraphSession.CmdTensorRef src, AexisGraphSession.BufferShape shape, AexisGraphSession.DeconvPack deconv)
        {
            return src != null
                && src.texture != null
                && src.texture.dimension == TextureDimension.Tex2DArray
                && !AexisGraphSession.IsStrictLinearMatTexture(src)
                && shape.dims == 3
                && shape.d == 1
                && shape.c == deconv.inC
                && src.width == shape.w
                && src.height == shape.h
                && src.packs == deconv.inPacks
                && AexisGraphSession.BufferShapeEquals(shape, AexisGraphSession.GetCmdStorageShape(src, shape));
        }

        private static string BuildCmdUnsupportedMessage(AexisGraphModel.Layer layer, AexisGraphSession.CmdTensorRef src, AexisGraphSession.BufferShape logicalShape, string reason)
        {
            var storageShape = src != null ? AexisGraphSession.GetCmdStorageShape(src, logicalShape) : default;
            return "Deconvolution CommandBuffer Pack4 rejected"
                + " | layer=" + (layer?.name ?? string.Empty)
                + " | blob=" + (layer?.bottomNames != null && layer.bottomNames.Length > 0 ? layer.bottomNames[0] : string.Empty)
                + " | logical_shape=" + logicalShape
                + " | storage_shape=" + storageShape
                + " | layout=Packed4"
                + " | dtype=" + (src?.texture != null ? src.texture.format.ToString() : "unknown")
                + " | reason=" + reason
                + " | rejected_fallback=Buffer/materialize-from-buffer/placeholder";
        }

        private static bool CanUsePack4Deconvolution(AexisGraphSession.DeconvPack deconv)
        {
            return deconv != null
                && deconv.group == 1
                && deconv.packedWeight4 != null
                && deconv.packedBias4 != null
                && deconv.inC > 0
                && deconv.outC > 0
                && deconv.inPacks > 0
                && deconv.outPacks > 0
                && deconv.kernelW > 0
                && deconv.kernelH > 0
                && deconv.strideW > 0
                && deconv.strideH > 0
                && deconv.dilationW > 0
                && deconv.dilationH > 0;
        }

        private static void ValidatePack4DeconvolutionTexture(
            string layerName,
            AexisGraphSession.DeconvPack deconv,
            AexisGraphSession.TensorRef src,
            AexisGraphSession.BufferShape srcShape,
            bool isCommandBuffer)
        {
            if (src == null || src.texture == null)
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "source texture missing"));
            if (src.texture.dimension != TextureDimension.Tex2DArray || AexisGraphSession.IsStrictLinearMatTexture(src))
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "source must be pack4 texture-array"));
            ValidatePack4DeconvolutionShape(layerName, deconv, srcShape, src.width, src.height, src.packs, isCommandBuffer);
            var storageShape = AexisGraphSession.GetTextureStorageShape(src, srcShape);
            if (!AexisGraphSession.BufferShapeEquals(srcShape, storageShape))
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "logical/storage shape mismatch"));
        }

        private static void ValidatePack4DeconvolutionTexture(
            string layerName,
            AexisGraphSession.DeconvPack deconv,
            AexisGraphSession.CmdTensorRef src,
            AexisGraphSession.BufferShape srcShape,
            bool isCommandBuffer)
        {
            if (src == null || src.texture == null)
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "source texture missing"));
            if (src.texture.dimension != TextureDimension.Tex2DArray || AexisGraphSession.IsStrictLinearMatTexture(src))
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "source must be pack4 texture-array"));
            ValidatePack4DeconvolutionShape(layerName, deconv, srcShape, src.width, src.height, src.packs, isCommandBuffer);
            var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
            if (!AexisGraphSession.BufferShapeEquals(srcShape, storageShape))
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "logical/storage shape mismatch"));
        }

        private static void ValidatePack4DeconvolutionShape(
            string layerName,
            AexisGraphSession.DeconvPack deconv,
            AexisGraphSession.BufferShape srcShape,
            int textureWidth,
            int textureHeight,
            int packs,
            bool isCommandBuffer)
        {
            if (!CanUsePack4Deconvolution(deconv))
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "unsupported deconvolution parameters"));
            if (srcShape.dims != 3 || srcShape.w <= 0 || srcShape.h <= 0 || srcShape.c != deconv.inC)
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "source shape must be dims=3 and match input channels"));
            if (textureWidth != srcShape.w || textureHeight != srcShape.h)
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "texture extent mismatch"));
            if (packs != deconv.inPacks)
                throw new InvalidOperationException(BuildUnsupportedMessage(layerName, srcShape, isCommandBuffer, "source pack count mismatch"));
        }

        private static string BuildUnsupportedMessage(string layerName, AexisGraphSession.BufferShape shape, bool isCommandBuffer, string reason)
        {
            return "Deconvolution " + (isCommandBuffer ? "command-buffer" : "render-texture")
                + " pack4 path unsupported: " + layerName
                + " | reason=" + reason
                + " | shape=d" + shape.dims + ":" + shape.w + "x" + shape.h + "x" + shape.d + "x" + shape.c;
        }
    }
}
