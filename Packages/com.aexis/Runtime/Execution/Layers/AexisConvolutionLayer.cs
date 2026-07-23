using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisConvolutionLayer : AexisBaseLayer
    {
        public AexisConvolutionLayer() : base(AexisLayerTypes.Convolution, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var pack = new AexisGraphSession.ConvPack();
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
                                        pack.biasTerm = layer.GetInt(5, 0);
                                        pack.weightSize = layer.GetInt(6, 0);
                                        pack.activationType = layer.GetInt(9, 0);
                                        pack.activationSlope = AexisGraphSession.ParseLeakySlope(layer);
                                        pack.isDepthWise = layer.type == AexisLayerTypes.ConvolutionDepthWise;

                                        var kernelArea = Mathf.Max(1, pack.kernelW * pack.kernelH);
                                        if (pack.isDepthWise)
                                        {
                                            pack.inC = Mathf.Max(1, (pack.weightSize * pack.group) / Mathf.Max(1, pack.outC * kernelArea));
                                            pack.useBufferPath = true;
                                        }
                                        else
                                        {
                                            pack.inC = Mathf.Max(1, (pack.weightSize * pack.group) / Mathf.Max(1, pack.outC * kernelArea));
                                            pack.useBufferPath = pack.strideW != 1
                                                                 || pack.strideH != 1
                                                                 || pack.kernelW != 1 && pack.kernelW != 3
                                                                 || pack.kernelH != pack.kernelW
                                                                 || pack.dilationW != 1
                                                                 || pack.dilationH != 1
                                                                 || pack.padLeft != pack.padRight
                                                                 || pack.padTop != pack.padBottom
                                                                 || pack.kernelW != 3 && pack.kernelW != 1;
                                        }
                                        pack.inPacks = (pack.inC + 3) / 4;
                                        pack.outPacks = (pack.outC + 3) / 4;

                                        phaseSw.Restart();
                                        var w = AexisGraphSession.ReadPackedOrRawWeightArray(br, pack.weightSize, layer.name);
                                        var b = pack.biasTerm != 0 ? br.ReadFloat32Array(pack.outC) : new float[pack.outC];
                                        phaseSw.Stop();
                                        readMs += phaseSw.ElapsedMilliseconds;

                                        var canUseSpecializedTexturePack = !pack.useBufferPath
                                                                          && !pack.isDepthWise
                                                                          && pack.group == 1
                                                                          && !(pack.kernelW == 1 && pack.kernelH == 1 && !owner.EnableConv1x1TextureConvolution);
                                        var canUseGeneralTexturePack = owner.EnableGeneralTextureConvolution
                                                                       && !pack.isDepthWise
                                                                       && pack.group == 1
                                                                       && pack.kernelW > 0
                                                                       && pack.kernelH == pack.kernelW
                                                                       && !(pack.kernelW == 1 && pack.kernelH == 1 && !owner.EnableConv1x1TextureConvolution);
                                        var needGeneralTexturePack = !owner.ForceBufferConvolutionAll
                                                                     && (canUseSpecializedTexturePack || canUseGeneralTexturePack);
                                        var needDepthWiseTexturePack = !owner.ForceBufferConvolutionAll
                                                                       && owner.EnableDepthWiseTextureConvolution
                                                                       && pack.isDepthWise
                                                                       && pack.group == pack.inC
                                                                       && pack.outC == pack.inC
                                                                       && pack.kernelW == 3
                                                                       && pack.kernelH == 3
                                                                       && pack.dilationW == 1
                                                                       && pack.dilationH == 1
                                                                       && pack.padLeft == pack.padRight
                                                                       && pack.padTop == pack.padBottom
                                                                       && pack.padLeft == pack.padTop;

                                        // Cmd Pack4 group/tail kernels consume this immutable scalar upload.
                                        // It is never used for activation or intermediate storage.
                                        phaseSw.Restart();
                                        if (owner.UsesInt4WeightOnlyForLayer(layer))
                                            AexisGraphSession.UploadInt4WeightOnlyConvWeights(pack, w, b, layer.name);
                                        else if (owner.UsesInt8WeightOnlyForLayer(layer))
                                            AexisGraphSession.UploadInt8WeightOnlyConvWeights(pack, w, b, layer.name);
                                        else
                                            AexisGraphSession.UploadRawConvWeights(pack, w, b);
                                        phaseSw.Stop();
                                        uploadMs += phaseSw.ElapsedMilliseconds;

                                        if (needGeneralTexturePack && !owner.UsesQuantizedWeightsForLayer(layer))
                                        {
                                            phaseSw.Restart();
                                            var w4 = AexisGraphSession.PackWeightsToO4I4K(w, pack.outC, pack.inC, pack.kernelW, pack.outPacks, pack.inPacks);
                                            var b4 = AexisGraphSession.PackBiasToO4(b, pack.outC, pack.outPacks);
                                            pack.packedWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                            pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                            pack.packedWeight4.SetData(w4);
                                            pack.packedBias4.SetData(b4);
                                            if (owner.UsesFp16WeightsForLayer(layer))
                                                pack.packedWeight4Fp16 = AexisGraphSession.NewFp16Vector4Buffer(w4, "AexisGraphSession.ConvPackedWeight4Fp16:" + layer.name);

                                            if (AexisGraphSession.EnableWinograd23
                                                && pack.kernelW == 3
                                                && pack.kernelH == 3
                                                && pack.strideW == 1
                                                && pack.strideH == 1
                                                && pack.padLeft == 1
                                                && pack.padRight == 1
                                                && pack.padTop == 1
                                                && pack.padBottom == 1
                                                && AexisWinograd23.CanUse(pack.kernelW, pack.padLeft, pack.inPacks, pack.outPacks))
                                            {
                                                pack.useWinograd23 = true;
                                                var wTm = AexisWinograd23.PackWeightTm23(w, pack.outC, pack.inC, pack.outPacks, pack.inPacks);
                                                pack.packedWeightTm23 = new ComputeBuffer(wTm.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                                pack.packedWeightTm23.SetData(wTm);
                                            }
                                            phaseSw.Stop();
                                            packMs += phaseSw.ElapsedMilliseconds;
                                        }
                                        else if (needDepthWiseTexturePack && !owner.UsesQuantizedWeightsForLayer(layer))
                                        {
                                            phaseSw.Restart();
                                            var w4 = AexisGraphSession.PackDepthWiseWeightsToP4KhKw(w, pack.outC, pack.kernelW, pack.kernelH, pack.outPacks);
                                            var b4 = AexisGraphSession.PackBiasToO4(b, pack.outC, pack.outPacks);
                                            pack.packedDepthWiseWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                            pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                            pack.packedDepthWiseWeight4.SetData(w4);
                                            pack.packedBias4.SetData(b4);
                                            phaseSw.Stop();
                                            packMs += phaseSw.ElapsedMilliseconds;
                                        }

                                        owner._conv[layer.name] = pack;
                                        return new AexisGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }
        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (!owner._conv.TryGetValue(layer.name, out var conv))
                throw new InvalidOperationException("Convolution not found: " + layer.name);

            if (!owner.ShouldForceCurrentLayerBufferPath() && CanExecuteRenderTexturePath(owner, layer, context, conv))
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
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            if (!owner._conv.TryGetValue(layer.name, out var conv))
                throw new InvalidOperationException("Convolution not found: " + layer.name);

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcTensor = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcTensor == null || srcTensor.dims != 3)
                throw new InvalidOperationException("Buffer convolution expects dims=3 tensor input: " + layer.name);

            var outW = AexisGraphSession.ComputeConvOut(srcTensor.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
            var outH = AexisGraphSession.ComputeConvOut(srcTensor.h, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom);
            var outTensor = owner.RentTempTensorBuffer(3, outW, outH, 1, conv.outC);
            if (conv.isDepthWise || conv.group > 1 || conv.kernelW != 3 || conv.kernelH != 3 || conv.strideW != conv.strideH || conv.padLeft != conv.padTop)
            {
                owner.Ops.ConvDepthWise(
                    srcTensor,
                    conv.rawWeight,
                    conv.rawBias,
                    conv.outC,
                    conv.group,
                    conv.kernelW,
                    conv.kernelH,
                    conv.strideW,
                    conv.strideH,
                    conv.padLeft,
                    conv.padTop,
                    conv.dilationW,
                    conv.dilationH,
                    conv.activationType,
                    conv.activationSlope,
                    outTensor);
            }
            else
            {
                owner.Ops.Conv3x3(srcTensor, conv.rawWeight, conv.rawBias, conv.outC, conv.strideW, conv.padLeft, conv.activationType, conv.activationSlope, outTensor);
            }

            bufferBlobs[layer.topNames[0]] = outTensor.buffer;
            bufferViews[layer.topNames[0]] = outTensor;
            tempOwned.Add(outTensor);
            owner.Consume(textureBlobs, bufferBlobs, context.bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            if (!owner._conv.TryGetValue(layer.name, out var conv))
                throw new InvalidOperationException("Convolution not found: " + layer.name);

            var canUseDepthWiseTexturePath = owner.EnableDepthWiseTextureConvolution
                                            && conv.isDepthWise
                                            && conv.group == conv.inC
                                            && conv.outC == conv.inC
                                            && conv.packedDepthWiseWeight4 != null
                                            && conv.packedBias4 != null;
            var canUseGeneralTexturePath = owner.EnableGeneralTextureConvolution
                                           && !conv.isDepthWise
                                           && conv.group == 1
                                           && conv.packedWeight4 != null
                                           && conv.packedBias4 != null
                                           && conv.kernelW > 0
                                           && conv.kernelH == conv.kernelW
                                           && !(conv.kernelW == 1 && conv.kernelH == 1 && !owner.EnableConv1x1TextureConvolution);

            AexisGraphSession.TensorRef src;
            AexisGraphSession.BufferShape srcShape;
            RenderTexture tempInputTex = null;
            if (textureBlobs.TryGetValue(layer.bottomNames[0], out src) && src != null && src.texture != null)
            {
                srcShape = AexisGraphSession.GetTextureShape(textureShapes, src, layer.bottomNames[0]);
            }
            else
            {
                if (!bufferBlobs.TryGetValue(layer.bottomNames[0], out var convInputBuf) || convInputBuf == null)
                    throw new InvalidOperationException("Convolution source not found: " + layer.name);
                var convInputView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                if (convInputView == null || convInputView.dims != 3)
                    throw new InvalidOperationException("Convolution texture path expects dims=3 buffer input: " + layer.name);

                var inPacks = (convInputView.c + 3) / 4;
                tempInputTex = owner.RentTempArray(convInputView.w, convInputView.h, inPacks, RenderTextureFormat.ARGBHalf);
                owner.Ops.FillPack4FromBufferCHW(convInputBuf, convInputView.w, convInputView.h, convInputView.c, tempInputTex);
                srcShape = new AexisGraphSession.BufferShape(3, convInputView.w, convInputView.h, 1, convInputView.c);
                src = AexisGraphSession.CreateTextureRef(tempInputTex, srcShape, srcShape, owned: false);
            }

            if (srcShape.dims == 2 && conv.inC == 1)
            {
                var promotedShape = new AexisGraphSession.BufferShape(3, srcShape.w, srcShape.h, 1, 1);
                if (!AexisGraphSession.IsStrictLinearMatTexture(src))
                    src = AexisGraphSession.CreateTextureRef(src.texture, promotedShape, promotedShape, owned: false, sharedTextureOwner: src, blobName: layer.bottomNames[0]);
                srcShape = promotedShape;
            }

            if (srcShape.dims != 3 || srcShape.d != 1 || srcShape.c != conv.inC)
            {
                throw new InvalidOperationException(
                    "Convolution texture path expects dims=3 pack4 input"
                    + " | layer=" + layer.name
                    + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | convInC=" + conv.inC);
            }

            if (AexisGraphSession.IsStrictLinearMatTexture(src))
            {
                var storageShape = AexisGraphSession.GetTextureStorageShape(src, srcShape);
                var materialized = owner.RentTempArray(srcShape.w, srcShape.h, conv.inPacks, RenderTextureFormat.ARGBHalf);
                owner.Ops.ReshapeLinearMatToPack4(
                    src.texture,
                    storageShape.w,
                    storageShape.h,
                    srcShape.w,
                    srcShape.h,
                    srcShape.d,
                    srcShape.c,
                    srcShape.dims,
                    materialized);
                if (tempInputTex != null)
                    owner.ReturnTempArray(tempInputTex);
                tempInputTex = materialized;
                src = AexisGraphSession.CreateTextureRef(tempInputTex, srcShape, srcShape, owned: false);
            }
            else if (!AexisGraphSession.MatchesPack4TextureStorage(src, srcShape) || src.packs != conv.inPacks)
            {
                throw new InvalidOperationException(
                    "Convolution texture path requires pack4 storage"
                    + " | layer=" + layer.name
                    + " | src=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | texture=" + src.width + "x" + src.height + "x" + src.packs
                    + " | storage=" + AexisGraphSession.GetTextureStorageShape(src, srcShape));
            }

            var outWTex = AexisGraphSession.ComputeConvOut(srcShape.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
            var outHTex = AexisGraphSession.ComputeConvOut(srcShape.h, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom);
            var outRt = owner.RentTempArray(outWTex, outHTex, conv.outPacks, owner.ResolveActivationTextureFormat(layer, 3));
            var canUseSpecialized3x3TexturePath = conv.kernelW == 3
                                                 && conv.kernelH == 3
                                                 && conv.strideW == 1
                                                 && conv.strideH == 1
                                                 && conv.padLeft == conv.padRight
                                                 && conv.padTop == conv.padBottom
                                                 && conv.padLeft == conv.padTop
                                                 && (conv.inC & 3) == 0
                                                 && (conv.outC & 3) == 0;
            var forceGeneralTexturePath = conv.kernelW == 1
                                          && conv.kernelH == 1
                                          && owner.ShouldCompareTextureConvLayer(layer.name)
                                          && canUseGeneralTexturePath;

            var useFp16GeneralWeights = owner.UsesFp16WeightsForLayer(layer)
                                        && conv.packedWeight4Fp16 != null
                                        && conv.group == 1
                                        && !conv.isDepthWise
                                        && conv.kernelW == conv.kernelH;
            var useInt8WeightOnly = owner.UsesInt8WeightsForLayer(layer);
            var useInt4WeightOnly = owner.UsesInt4WeightsForLayer(layer);
            owner.Ops.SetFp16ConvWeights(useFp16GeneralWeights ? conv.packedWeight4Fp16 : null);
            owner.Ops.SetInt8ConvWeights(
                useInt8WeightOnly ? conv.rawWeightInt8Packed : null,
                useInt8WeightOnly ? conv.rawWeightInt8Scales : null);
            owner.Ops.SetInt4ConvWeights(
                useInt4WeightOnly ? conv.rawWeightInt4Packed : null,
                useInt4WeightOnly ? conv.rawWeightInt4Scales : null);
            owner.ConfigureInt8ActivationQuantization(layer);
            var preferRawGroupPack4 = !useInt8WeightOnly
                                      && !useInt4WeightOnly
                                      && !conv.isDepthWise
                                      && conv.group == 1
                                      && (conv.dilationW != 1 || conv.dilationH != 1)
                                      && conv.rawWeight != null
                                      && conv.rawBias != null;
            if (useInt8WeightOnly || useInt4WeightOnly)
            {
                owner.Ops.Conv2dGroupPack4(
                    src.texture,
                    useInt4WeightOnly ? conv.rawWeightInt4Packed : conv.rawWeightInt8Packed,
                    conv.rawBias,
                    conv.inC,
                    conv.outC,
                    conv.group,
                    conv.kernelW,
                    conv.kernelH,
                    conv.strideW,
                    conv.strideH,
                    conv.padLeft,
                    conv.padTop,
                    conv.dilationW,
                    conv.dilationH,
                    conv.activationType,
                    conv.activationSlope,
                    outRt);
            }
            else if (preferRawGroupPack4)
            {
                owner.Ops.Conv2dGroupPack4(
                    src.texture,
                    conv.rawWeight,
                    conv.rawBias,
                    conv.inC,
                    conv.outC,
                    conv.group,
                    conv.kernelW,
                    conv.kernelH,
                    conv.strideW,
                    conv.strideH,
                    conv.padLeft,
                    conv.padTop,
                    conv.dilationW,
                    conv.dilationH,
                    conv.activationType,
                    conv.activationSlope,
                    outRt);
            }
            else if (useFp16GeneralWeights)
            {
                owner.Ops.ConvPack4General(src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.outC, conv.kernelW, conv.kernelH, conv.strideW, conv.strideH, conv.padLeft, conv.padTop, conv.dilationW, conv.dilationH, conv.activationType, conv.activationSlope, outRt);
            }
            else if (conv.kernelW == 1
                && conv.kernelH == 1
                && owner.EnableConv1x1TextureConvolution
                && !forceGeneralTexturePath
                && src.width == outWTex
                && src.height == outHTex)
            {
                owner.Ops.Conv1x1Pack4(src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.activationType, conv.activationSlope, outRt);
                if (owner.ShouldCompareTextureConvLayer(layer.name) && !forceGeneralTexturePath)
                    owner.CompareTextureConvPath(layer.name, layer.bottomNames[0], conv, outWTex, outHTex, outRt, textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            }
            else if (canUseDepthWiseTexturePath)
            {
                owner.Ops.ConvDepthWisePack4(src.texture, conv.packedDepthWiseWeight4, conv.packedBias4, conv.inC, conv.outC, conv.group, conv.outPacks, conv.kernelW, conv.kernelH, conv.strideW, conv.strideH, conv.padLeft, conv.padTop, conv.dilationW, conv.dilationH, conv.activationType, conv.activationSlope, outRt);
                if (owner.ShouldCompareTextureConvLayer(layer.name))
                    owner.CompareTextureConvPath(layer.name, layer.bottomNames[0], conv, outWTex, outHTex, outRt, textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            }
            else if (canUseSpecialized3x3TexturePath)
            {
                if (AexisGraphSession.EnableWinograd23 && conv.useWinograd23)
                    owner.Ops.Conv3x3Pack4Winograd23(src.texture, conv.inPacks, conv.packedWeightTm23, conv.packedBias4, conv.outPacks, conv.biasTerm, conv.activationType, conv.activationSlope, outRt);
                else
                    owner.Ops.Conv3x3Pack4(src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.padLeft, conv.activationType, conv.activationSlope, outRt);
                if (owner.ShouldCompareTextureConvLayer(layer.name))
                    owner.CompareTextureConvPath(layer.name, layer.bottomNames[0], conv, outWTex, outHTex, outRt, textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            }
            else if (canUseGeneralTexturePath)
            {
                owner.Ops.ConvPack4General(src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.outC, conv.kernelW, conv.kernelH, conv.strideW, conv.strideH, conv.padLeft, conv.padTop, conv.dilationW, conv.dilationH, conv.activationType, conv.activationSlope, outRt);
                if (owner.ShouldCompareTextureConvLayer(layer.name))
                    owner.CompareTextureConvPath(layer.name, layer.bottomNames[0], conv, outWTex, outHTex, outRt, textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            }
            else
            {
                throw new InvalidOperationException("Texture convolution path unsupported config: " + layer.name);
            }

            var outShape = new AexisGraphSession.BufferShape(3, outWTex, outHTex, 1, conv.outC);
            AexisGraphSession.SetTextureBlob(textureBlobs, textureShapes, layer.topNames[0], outRt, outShape, outShape);
            if (tempInputTex != null)
                owner.ReturnTempArray(tempInputTex);
            owner.Consume(textureBlobs, bufferBlobs, context.bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }
        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            if (!owner._conv.TryGetValue(layer.name, out var conv))
                throw new InvalidOperationException("Convolution not found: " + layer.name);
            if (!SupportsCommandBufferPack4(conv, out var reason))
                throw new InvalidOperationException(BuildUnsupportedMessage(layer, src, srcShape, reason));

            ComputeTexture tempInput = null;
            ComputeTexture output = null;
            try
            {
                if (srcShape.dims == 2 && conv.inC == 1)
                {
                    var promotedShape = new AexisGraphSession.BufferShape(3, srcShape.w, srcShape.h, 1, 1);
                    if (!AexisGraphSession.IsStrictLinearMatTexture(src))
                        src = AexisGraphSession.CreateCmdTensorRef(src.texture, promotedShape, promotedShape, owned: false, sharedTextureOwner: src, blobName: layer.bottomNames[0]);
                    srcShape = promotedShape;
                }

                if (AexisGraphSession.IsStrictLinearMatTexture(src)
                    && srcShape.dims == 3
                    && srcShape.d == 1
                    && srcShape.c == conv.inC)
                {
                    var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
                    tempInput = owner.RentTempArray(cmd, srcShape.w, srcShape.h, conv.inPacks, owner.ResolveActivationTextureFormat(layer, 3));
                    owner.Ops.ReshapeLinearMatToPack4(cmd, src.texture, storageShape.w, storageShape.h, srcShape.w, srcShape.h, srcShape.d, srcShape.c, srcShape.dims, tempInput);
                    src = AexisGraphSession.CreateCmdTensorRef(tempInput, srcShape, srcShape, owned: false);
                }

                if (!CanUsePack4CmdPath(src, srcShape, conv))
                    throw new InvalidOperationException(BuildUnsupportedMessage(layer, src, srcShape, "source is not a matching Pack4 texture-array"));

                var outShape = ResolveCmdOutputShape(srcShape, conv);
                output = owner.RentTempArray(cmd, outShape.w, outShape.h, conv.outPacks, owner.ResolveActivationTextureFormat(layer, 3));
                var useFp16GeneralWeights = owner.UsesFp16WeightsForLayer(layer)
                                            && conv.packedWeight4Fp16 != null
                                            && conv.group == 1
                                            && !conv.isDepthWise
                                            && conv.kernelW == conv.kernelH;
                owner.Ops.SetFp16ConvWeights(useFp16GeneralWeights ? conv.packedWeight4Fp16 : null);
                var useInt8WeightOnly = owner.UsesInt8WeightsForLayer(layer);
                var useInt4WeightOnly = owner.UsesInt4WeightsForLayer(layer);
                owner.Ops.SetInt8ConvWeights(
                    useInt8WeightOnly ? conv.rawWeightInt8Packed : null,
                    useInt8WeightOnly ? conv.rawWeightInt8Scales : null);
                owner.Ops.SetInt4ConvWeights(
                    useInt4WeightOnly ? conv.rawWeightInt4Packed : null,
                    useInt4WeightOnly ? conv.rawWeightInt4Scales : null);
                owner.ConfigureInt8ActivationQuantization(layer);
                if (useInt8WeightOnly || useInt4WeightOnly)
                {
                    owner.Ops.Conv2dGroupPack4(
                        cmd, src.texture, useInt4WeightOnly ? conv.rawWeightInt4Packed : conv.rawWeightInt8Packed, conv.rawBias, conv.inC, conv.outC, conv.group,
                        conv.kernelW, conv.kernelH, conv.strideW, conv.strideH, conv.padLeft, conv.padTop,
                        conv.dilationW, conv.dilationH, conv.activationType, conv.activationSlope, output);
                }
                else if (useFp16GeneralWeights)
                {
                    owner.Ops.ConvPack4General(cmd, src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.outC, conv.kernelW, conv.kernelH, conv.strideW, conv.strideH, conv.padLeft, conv.padTop, conv.dilationW, conv.dilationH, conv.activationType, conv.activationSlope, output);
                }
                else if (conv.group == 1
                    && !conv.isDepthWise
                    && conv.kernelW == 1
                    && conv.kernelH == 1
                    && conv.strideW == 1
                    && conv.strideH == 1
                    && conv.dilationW == 1
                    && conv.dilationH == 1
                    && conv.padLeft == 0
                    && conv.padRight == 0
                    && conv.padTop == 0
                    && conv.padBottom == 0
                    && (conv.outC & 3) == 0
                    && conv.packedWeight4 != null
                    && conv.packedBias4 != null)
                {
                    owner.Ops.Conv1x1Pack4(cmd, src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.activationType, conv.activationSlope, output);
                }
                else if (conv.group == 1
                         && !conv.isDepthWise
                         && conv.kernelW == 3
                         && conv.kernelH == 3
                         && conv.strideW == 1
                         && conv.strideH == 1
                         && conv.dilationW == 1
                         && conv.dilationH == 1
                         && conv.padLeft == conv.padRight
                         && conv.padLeft == conv.padTop
                         && conv.padTop == conv.padBottom
                         && (conv.inC & 3) == 0
                         && (conv.outC & 3) == 0
                         && conv.packedWeight4 != null
                         && conv.packedBias4 != null)
                {
                    owner.Ops.Conv3x3Pack4(cmd, src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.padLeft, conv.activationType, conv.activationSlope, output);
                }
                else if (conv.group == 1
                         && !conv.isDepthWise
                         && conv.kernelW == conv.kernelH
                         && (conv.outC & 3) == 0
                         && conv.packedWeight4 != null
                         && conv.packedBias4 != null)
                {
                    owner.Ops.ConvPack4General(cmd, src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.outC, conv.kernelW, conv.kernelH, conv.strideW, conv.strideH, conv.padLeft, conv.padTop, conv.dilationW, conv.dilationH, conv.activationType, conv.activationSlope, output);
                }
                else
                {
                    owner.Ops.Conv2dGroupPack4(
                        cmd, src.texture, conv.rawWeight, conv.rawBias, conv.inC, conv.outC, conv.group,
                        conv.kernelW, conv.kernelH, conv.strideW, conv.strideH, conv.padLeft, conv.padTop,
                        conv.dilationW, conv.dilationH, conv.activationType, conv.activationSlope, output);
                }
                blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(output, outShape, outShape, owned: true);
                output = null;
                if (shapes != null)
                    shapes[layer.topNames[0]] = outShape;
            }
            finally
            {
                if (output != null)
                    owner.ReturnTempArray(cmd, output);
                if (tempInput != null)
                    owner.ReturnTempArray(cmd, tempInput);
            }

            owner.ConsumeCmd(cmd, blobs, context.remaining, layer.bottomNames, context.pinnedNames, shapes);
        }

        private static bool CanUsePack4CmdPath(AexisGraphSession.CmdTensorRef src, AexisGraphSession.BufferShape srcShape, AexisGraphSession.ConvPack conv)
        {
            return src != null
                && src.texture != null
                && conv != null
                && srcShape.dims == 3
                && srcShape.d == 1
                && srcShape.c == conv.inC
                && AexisGraphSession.MatchesPack4TextureStorage(src, srcShape)
                && src.packs == conv.inPacks;
        }

        private static bool SupportsCommandBufferPack4(AexisGraphSession.ConvPack conv, out string reason)
        {
            reason = null;
            if (conv == null || (conv.rawWeight == null && conv.rawWeightInt8Packed == null && conv.rawWeightInt4Packed == null) || conv.rawBias == null)
                reason = "immutable scalar weights/bias are unavailable";
            else if (conv.inC <= 0 || conv.outC <= 0 || conv.group <= 0 || conv.inC % conv.group != 0 || conv.outC % conv.group != 0)
                reason = "group must divide positive input and output channels";
            else if (conv.kernelW <= 0 || conv.kernelH <= 0 || conv.strideW <= 0 || conv.strideH <= 0 || conv.dilationW <= 0 || conv.dilationH <= 0)
                reason = "kernel, stride, and dilation must be positive";
            else if (conv.padLeft < 0 || conv.padRight < 0 || conv.padTop < 0 || conv.padBottom < 0)
                reason = "negative/auto padding is not implemented";
            else if (conv.activationType != 0 && conv.activationType != 1 && conv.activationType != 2 && conv.activationType != 4)
                reason = "activation supports only none, ReLU, LeakyReLU, or Sigmoid";
            else if (conv.weightSize != conv.outC * (conv.inC / conv.group) * conv.kernelW * conv.kernelH)
                reason = "weight_data_size does not match the grouped OIHW profile";
            return reason == null;
        }

        private static string BuildUnsupportedMessage(AexisGraphModel.Layer layer, AexisGraphSession.CmdTensorRef src, AexisGraphSession.BufferShape logicalShape, string reason)
        {
            var storageShape = src != null ? AexisGraphSession.GetCmdStorageShape(src, logicalShape) : default;
            return "Convolution CommandBuffer Pack4 rejected"
                + " | layer=" + (layer?.name ?? string.Empty)
                + " | blob=" + (layer?.bottomNames != null && layer.bottomNames.Length > 0 ? layer.bottomNames[0] : string.Empty)
                + " | logical_shape=" + logicalShape
                + " | storage_shape=" + storageShape
                + " | layout=Packed4"
                + " | dtype=" + (src?.texture != null ? src.texture.format.ToString() : "unknown")
                + " | reason=" + reason
                + " | rejected_fallback=Buffer/materialize-from-buffer/placeholder";
        }

        private static bool CanExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context, AexisGraphSession.ConvPack conv)
        {
            var canUseConv1x1TexturePath = !conv.isDepthWise
                                           && conv.group == 1
                                           && conv.kernelW == 1
                                           && conv.kernelH == 1
                                           && owner.EnableConv1x1TextureConvolution
                                           && conv.packedWeight4 != null
                                           && conv.packedBias4 != null;
            var canUseDepthWiseTexturePath = owner.EnableDepthWiseTextureConvolution
                                            && conv.isDepthWise
                                            && conv.group == conv.inC
                                            && conv.outC == conv.inC
                                            && conv.packedDepthWiseWeight4 != null
                                            && conv.packedBias4 != null;
            var canUseSpecialized3x3TexturePath = !conv.isDepthWise
                                                 && conv.group == 1
                                                 && conv.kernelW == 3
                                                 && conv.kernelH == 3
                                                 && conv.strideW == 1
                                                 && conv.strideH == 1
                                                 && conv.padLeft == conv.padRight
                                                 && conv.padTop == conv.padBottom
                                                 && conv.padLeft == conv.padTop
                                                 && (conv.inC & 3) == 0
                                                 && (conv.outC & 3) == 0
                                                 && conv.packedWeight4 != null
                                                 && conv.packedBias4 != null;
            var canUseGeneralTexturePath = owner.EnableGeneralTextureConvolution
                                           && !conv.isDepthWise
                                           && conv.group == 1
                                           && conv.packedWeight4 != null
                                           && conv.packedBias4 != null
                                           && conv.kernelW > 0
                                           && conv.kernelH == conv.kernelW
                                           && !(conv.kernelW == 1 && conv.kernelH == 1 && !owner.EnableConv1x1TextureConvolution);
            var canUseQuantizedTexturePath = owner.UsesQuantizedWeightsForLayer(layer)
                                             && SupportsCommandBufferPack4(conv, out _);
            var hasSupportedTexturePath = canUseConv1x1TexturePath
                                          || canUseDepthWiseTexturePath
                                          || canUseSpecialized3x3TexturePath
                                          || canUseGeneralTexturePath
                                          || canUseQuantizedTexturePath;

            if (!hasSupportedTexturePath)
                return false;

            var forceBufferThisConv = owner.ForceBufferConvolutionAll
                                      || (conv.useBufferPath && !canUseDepthWiseTexturePath && !canUseGeneralTexturePath && !canUseQuantizedTexturePath)
                                      || (conv.kernelW == 1 && conv.kernelH == 1 && !owner.EnableConv1x1TextureConvolution);

            if (forceBufferThisConv)
            {
                return owner.PreferTexturePathForFaceDetector
                    && string.Equals(layer.bottomNames[0], "images", StringComparison.Ordinal) == false
                    && layer.name.StartsWith("Conv_", StringComparison.Ordinal)
                    && conv.kernelW == 3
                    && conv.kernelH == 3
                    && conv.dilationW == 1
                    && conv.dilationH == 1
                    && conv.group == conv.inC
                    && conv.outC == conv.inC;
            }

            return true;
        }

        private static AexisGraphSession.BufferShape ResolveCmdOutputShape(AexisGraphSession.BufferShape srcShape, AexisGraphSession.ConvPack conv)
        {
            var outW = Mathf.Max(1, AexisGraphSession.ComputeConvOut(srcShape.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight));
            var outH = Mathf.Max(1, AexisGraphSession.ComputeConvOut(srcShape.h, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom));
            return new AexisGraphSession.BufferShape(3, outW, outH, 1, Mathf.Max(1, conv.outC));
        }
    }
}
