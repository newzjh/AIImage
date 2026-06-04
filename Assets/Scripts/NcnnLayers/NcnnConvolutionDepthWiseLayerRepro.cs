using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnConvolutionDepthWiseLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnConvolutionDepthWiseLayerRepro() : base(NcnnLayerTypes.ConvolutionDepthWise, supportsBufferPath: true, supportsCommandBufferPath: false) { }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var pack = new NcnnRepro.ConvPack();
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
                                        pack.activationSlope = NcnnRepro.ParseLeakySlope(layer);
                                        pack.isDepthWise = layer.type == NcnnLayerTypes.ConvolutionDepthWise;

                                        var kernelArea = Mathf.Max(1, pack.kernelW * pack.kernelH);
                                        if (pack.isDepthWise)
                                        {
                                            pack.inC = Mathf.Max(1, (pack.weightSize * pack.group) / Mathf.Max(1, pack.outC * kernelArea));
                                            pack.useBufferPath = true;
                                        }
                                        else
                                        {
                                            pack.inC = Mathf.Max(1, pack.weightSize / Mathf.Max(1, pack.outC * kernelArea));
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
                                        var w = NcnnRepro.ReadPackedOrRawWeightArray(br, pack.weightSize, layer.name);
                                        var b = pack.biasTerm != 0 ? br.ReadFloat32Array(pack.outC) : new float[pack.outC];
                                        phaseSw.Stop();
                                        readMs += phaseSw.ElapsedMilliseconds;

                                        var needGeneralTexturePack = !owner.ForceBufferConvolutionAll
                                                                     && !pack.useBufferPath
                                                                     && !pack.isDepthWise
                                                                     && pack.group == 1
                                                                     && !(pack.kernelW == 1 && pack.kernelH == 1 && !owner.EnableConv1x1TextureConvolution);
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

                                        if (owner.ShouldKeepRawConvWeightsForTexturePath(layer.name, pack, needGeneralTexturePack, needDepthWiseTexturePack))
                                        {
                                            phaseSw.Restart();
                                            NcnnRepro.UploadRawConvWeights(pack, w, b);
                                            phaseSw.Stop();
                                            uploadMs += phaseSw.ElapsedMilliseconds;
                                        }

                                        if (needGeneralTexturePack)
                                        {
                                            phaseSw.Restart();
                                            var w4 = NcnnRepro.PackWeightsToO4I4K(w, pack.outC, pack.inC, pack.kernelW, pack.outPacks, pack.inPacks);
                                            var b4 = NcnnRepro.PackBiasToO4(b, pack.outC, pack.outPacks);
                                            pack.packedWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                            pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                            pack.packedWeight4.SetData(w4);
                                            pack.packedBias4.SetData(b4);

                                            if (NcnnRepro.EnableWinograd23
                                                && pack.kernelW == 3
                                                && pack.kernelH == 3
                                                && pack.strideW == 1
                                                && pack.strideH == 1
                                                && pack.padLeft == 1
                                                && pack.padRight == 1
                                                && pack.padTop == 1
                                                && pack.padBottom == 1
                                                && NcnnWinograd23.CanUse(pack.kernelW, pack.padLeft, pack.inPacks, pack.outPacks))
                                            {
                                                pack.useWinograd23 = true;
                                                var wTm = NcnnWinograd23.PackWeightTm23(w, pack.outC, pack.inC, pack.outPacks, pack.inPacks);
                                                pack.packedWeightTm23 = new ComputeBuffer(wTm.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                                pack.packedWeightTm23.SetData(wTm);
                                            }
                                            phaseSw.Stop();
                                            packMs += phaseSw.ElapsedMilliseconds;
                                        }
                                        else if (needDepthWiseTexturePack)
                                        {
                                            phaseSw.Restart();
                                            var w4 = NcnnRepro.PackDepthWiseWeightsToP4K4(w, pack.outC, pack.kernelW, pack.outPacks);
                                            var b4 = NcnnRepro.PackBiasToO4(b, pack.outC, pack.outPacks);
                                            pack.packedDepthWiseWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                            pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                            pack.packedDepthWiseWeight4.SetData(w4);
                                            pack.packedBias4.SetData(b4);
                                            phaseSw.Stop();
                                            packMs += phaseSw.ElapsedMilliseconds;
                                        }

                                        owner._conv[layer.name] = pack;
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
                                                if (!owner._conv.TryGetValue(layer.name, out var conv))
                                                    throw new InvalidOperationException("Convolution not found: " + layer.name);

                                                var canUseDepthWiseTexturePath = owner.EnableDepthWiseTextureConvolution
                                                                                && conv.isDepthWise
                                                                                && conv.group == conv.inC
                                                                                && conv.outC == conv.inC
                                                                                && conv.packedDepthWiseWeight4 != null
                                                                                && conv.packedBias4 != null;

                                                var forceBufferThisConv = owner.ForceBufferConvolutionAll
                                                                          || (conv.useBufferPath && !canUseDepthWiseTexturePath)
                                                                          || (conv.kernelW == 1 && conv.kernelH == 1 && !owner.EnableConv1x1TextureConvolution);

                                                if (forceBufferThisConv)
                                                {
                                                    if (owner.PreferTexturePathForFaceDetector
                                                        && string.Equals(layer.bottomNames[0], "images", StringComparison.Ordinal) == false
                                                        && layer.name.StartsWith("Conv_", StringComparison.Ordinal)
                                                        && conv.kernelW == 3
                                                        && conv.kernelH == 3
                                                        && conv.dilationW == 1
                                                        && conv.dilationH == 1
                                                        && conv.group == conv.inC
                                                        && conv.outC == conv.inC)
                                                    {
                                                        goto TextureConvPath;
                                                    }

                                                    var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                    var srcTensor = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                    if (srcBuf == null || srcTensor == null || srcTensor.dims != 3)
                                                        throw new InvalidOperationException("Buffer convolution expects dims=3 tensor input: " + layer.name);

                                                    var outW = NcnnRepro.ComputeConvOut(srcTensor.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
                                                    var outH = NcnnRepro.ComputeConvOut(srcTensor.h, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom);
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
                                                    owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                    continue;
                                                }

                            TextureConvPath:
                                                NcnnRepro.TensorRef src;
                                                RenderTexture tempInputTex = null;
                                                    if (textureBlobs.TryGetValue(layer.bottomNames[0], out src) && src != null && src.texture != null)
                                                    {
                                                    }
                                                    else
                                                {
                                                    if (!bufferBlobs.TryGetValue(layer.bottomNames[0], out var convInputBuf) || convInputBuf == null)
                                                        throw new InvalidOperationException("Convolution source not found: " + layer.name);
                                                    var convInputView = NcnnRepro.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                                    if (convInputView == null || convInputView.dims != 3)
                                                        throw new InvalidOperationException("Convolution texture path expects dims=3 buffer input: " + layer.name);

                                                    var inPacks = (convInputView.c + 3) / 4;
                                                    tempInputTex = owner.RentTempArray(convInputView.w, convInputView.h, inPacks, RenderTextureFormat.ARGBHalf);
                                                    owner.Ops.FillPack4FromBufferCHW(convInputBuf, convInputView.w, convInputView.h, convInputView.c, tempInputTex);
                                                    src = new NcnnRepro.TensorRef
                                                    {
                                                        texture = tempInputTex,
                                                        width = convInputView.w,
                                                        height = convInputView.h,
                                                        packs = inPacks,
                                                        refs = 1,
                                                        owned = false
                                                    };
                                                }
                                                var outWTex = NcnnRepro.ComputeConvOut(src.width, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
                                                var outHTex = NcnnRepro.ComputeConvOut(src.height, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom);
                                                var outRt = owner.RentTempArray(outWTex, outHTex, conv.outPacks, RenderTextureFormat.ARGBHalf);

                                                if (conv.kernelW == 1 && conv.kernelH == 1 && owner.EnableConv1x1TextureConvolution)
                                                {
                                                    if (src.width != outWTex || src.height != outHTex)
                                                        throw new InvalidOperationException("Conv1x1 texture path does not support spatial resize: " + layer.name);
                                                    owner.Ops.Conv1x1Pack4(src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.activationType, conv.activationSlope, outRt);
                                                    if (owner.ShouldCompareTextureConvLayer(layer.name))
                                                    {
                                                        owner.CompareTextureConvPath(layer.name, layer.bottomNames[0], conv, outWTex, outHTex, outRt, textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                    }
                                                }
                                                else if (canUseDepthWiseTexturePath)
                                                {
                                                    owner.Ops.ConvDepthWisePack4(src.texture, conv.packedDepthWiseWeight4, conv.packedBias4, conv.outPacks, conv.kernelW, conv.kernelH, conv.strideW, conv.strideH, conv.padLeft, conv.padTop, conv.dilationW, conv.dilationH, conv.activationType, conv.activationSlope, outRt);
                                                    if (owner.ShouldCompareTextureConvLayer(layer.name))
                                                    {
                                                        owner.CompareTextureConvPath(layer.name, layer.bottomNames[0], conv, outWTex, outHTex, outRt, textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                    }
                                                }
                                                else if (conv.kernelW == 3
                                                         && conv.kernelH == 3
                                                         && conv.strideW == 1
                                                         && conv.strideH == 1
                                                         && conv.padLeft == conv.padRight
                                                         && conv.padTop == conv.padBottom
                                                         && conv.padLeft == conv.padTop)
                                                {
                                                    if (NcnnRepro.EnableWinograd23 && conv.useWinograd23)
                                                        owner.Ops.Conv3x3Pack4Winograd23(src.texture, conv.inPacks, conv.packedWeightTm23, conv.packedBias4, conv.outPacks, conv.biasTerm, conv.activationType, conv.activationSlope, outRt);
                                                    else
                                                        owner.Ops.Conv3x3Pack4(src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.padLeft, conv.activationType, conv.activationSlope, outRt);
                                                    if (owner.ShouldCompareTextureConvLayer(layer.name))
                                                    {
                                                        owner.CompareTextureConvPath(layer.name, layer.bottomNames[0], conv, outWTex, outHTex, outRt, textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                                    }
                                                }
                                                else
                                                {
                                                    throw new InvalidOperationException("Texture convolution path unsupported config: " + layer.name);
                                                }

                                                textureBlobs[layer.topNames[0]] = new NcnnRepro.TensorRef
                                                {
                                                    texture = outRt,
                                                    width = outWTex,
                                                    height = outHTex,
                                                    packs = conv.outPacks,
                                                    refs = 1,
                                                    owned = true
                                                };
                                                textureShapes[layer.topNames[0]] = new NcnnRepro.BufferShape(3, outWTex, outHTex, 1, conv.outC);
                                                if (tempInputTex != null)
                                                    owner.ReturnTempArray(tempInputTex);
                                                owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }
        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
                        var cmd = context.commandBuffer;
                        var blobs = context.blobs;
                        var remaining = context.remaining;
                        var pinnedNames = context.pinnedNames;

                        do
                        {
                                                var src = NcnnRepro.GetCmdTensor(blobs, layer.bottomNames[0]);
                                                if (!owner._conv.TryGetValue(layer.name, out var conv))
                                                    throw new InvalidOperationException("Convolution not found: " + layer.name);
                                                if (src.packs != conv.inPacks)
                                                    throw new InvalidOperationException("unexpected in packs for " + layer.name + ": " + src.packs + " vs " + conv.inPacks);
                                                if (conv.isDepthWise || conv.group != 1)
                                                    throw new InvalidOperationException("CommandBuffer convolution does not support depthwise/group conv: " + layer.name);
                                                if (conv.strideW != 1 || conv.strideH != 1 || conv.dilationW != 1 || conv.dilationH != 1)
                                                    throw new InvalidOperationException("CommandBuffer convolution only supports stride=1 dilation=1: " + layer.name);

                                                var outW = NcnnRepro.ComputeConvOut(src.width, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
                                                var outH = NcnnRepro.ComputeConvOut(src.height, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom);
                                                var outArr = owner.RentTempArray(cmd, outW, outH, conv.outPacks, RenderTextureFormat.ARGBHalf);

                                                if (conv.kernelW == 1 && conv.kernelH == 1)
                                                {
                                                    owner.Ops.Conv1x1Pack4(cmd, src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.activationType, conv.activationSlope, outArr);
                                                }
                                                else if (conv.kernelW == 3 && conv.kernelH == 3 && conv.padLeft == conv.padRight && conv.padLeft == conv.padTop && conv.padTop == conv.padBottom)
                                                {
                                                    var useWinograd = NcnnRepro.EnableWinograd23
                                                        && conv.packedWeightTm23 != null
                                                        && conv.strideW == 1
                                                        && conv.strideH == 1
                                                        && conv.padLeft == 1
                                                        && conv.padTop == 1;
                                                    if (useWinograd)
                                                    {
                                                        owner.Ops.Conv3x3Pack4Winograd23(cmd, src.texture, conv.inPacks, conv.packedWeightTm23, conv.packedBias4, conv.outPacks, conv.biasTerm, conv.activationType, conv.activationSlope, outArr);
                                                    }
                                                    else
                                                    {
                                                        owner.Ops.Conv3x3Pack4(cmd, src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.padLeft, conv.activationType, conv.activationSlope, outArr);
                                                    }
                                                }
                                                else
                                                {
                                                    throw new InvalidOperationException("CommandBuffer convolution only supports 1x1/3x3 symmetric conv: " + layer.name);
                                                }

                                                blobs[layer.topNames[0]] = new NcnnRepro.CmdTensorRef { texture = outArr, width = outW, height = outH, packs = conv.outPacks, refs = 1, owned = true };
                                                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames);
                                                continue;
                        } while (false);
        }
    }
}
