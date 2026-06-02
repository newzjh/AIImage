using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public sealed class NcnnConvolutionLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnConvolutionLayerRepro() : base(NcnnLayerTypes.Convolution, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br) => owner.LoadConvolutionFamilyLayer(layer, br);
        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context) => owner.ExecuteConvolutionFamilyBufferLayer(layer, context);
        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context) => owner.ExecuteConvolutionFamilyCommandBufferLayer(layer, context);
    }

    public partial class NcnnRepro
    {
        internal LayerLoadMetrics LoadConvolutionFamilyLayer(NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

                            var pack = new ConvPack();
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
                            pack.activationSlope = ParseLeakySlope(layer);
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

                            var needGeneralTexturePack = !ForceBufferConvolutionAll
                                                         && !pack.useBufferPath
                                                         && !pack.isDepthWise
                                                         && pack.group == 1
                                                         && !(pack.kernelW == 1 && pack.kernelH == 1 && !EnableConv1x1TextureConvolution);
                            var needDepthWiseTexturePack = !ForceBufferConvolutionAll
                                                           && EnableDepthWiseTextureConvolution
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

                            if (needGeneralTexturePack)
                            {
                                phaseSw.Restart();
                                var w4 = PackWeightsToO4I4K(w, pack.outC, pack.inC, pack.kernelW, pack.outPacks, pack.inPacks);
                                var b4 = PackBiasToO4(b, pack.outC, pack.outPacks);
                                pack.packedWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                pack.packedWeight4.SetData(w4);
                                pack.packedBias4.SetData(b4);

                                if (EnableWinograd23
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
                                var w4 = PackDepthWiseWeightsToP4K4(w, pack.outC, pack.kernelW, pack.outPacks);
                                var b4 = PackBiasToO4(b, pack.outC, pack.outPacks);
                                pack.packedDepthWiseWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                                pack.packedDepthWiseWeight4.SetData(w4);
                                pack.packedBias4.SetData(b4);
                                phaseSw.Stop();
                                packMs += phaseSw.ElapsedMilliseconds;
                            }

                            _conv[layer.name] = pack;
                            return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        internal void ExecuteConvolutionFamilyBufferLayer(NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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
                                    if (!_conv.TryGetValue(layer.name, out var conv))
                                        throw new InvalidOperationException("Convolution not found: " + layer.name);

                                    var canUseDepthWiseTexturePath = EnableDepthWiseTextureConvolution
                                                                    && conv.isDepthWise
                                                                    && conv.group == conv.inC
                                                                    && conv.outC == conv.inC
                                                                    && conv.packedDepthWiseWeight4 != null
                                                                    && conv.packedBias4 != null;

                                    var forceBufferThisConv = ForceBufferConvolutionAll
                                                              || (conv.useBufferPath && !canUseDepthWiseTexturePath)
                                                              || (conv.kernelW == 1 && conv.kernelH == 1 && !EnableConv1x1TextureConvolution);

                                    if (forceBufferThisConv)
                                    {
                                        if (PreferTexturePathForFaceDetector
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

                                        var srcBuf = GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                        var srcTensor = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                        if (srcBuf == null || srcTensor == null || srcTensor.dims != 3)
                                            throw new InvalidOperationException("Buffer convolution expects dims=3 tensor input: " + layer.name);

                                        var outW = ComputeConvOut(srcTensor.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
                                        var outH = ComputeConvOut(srcTensor.h, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom);
                                        var outTensor = RentTempTensorBuffer(3, outW, outH, 1, conv.outC);
                                        if (conv.isDepthWise || conv.group > 1 || conv.kernelW != 3 || conv.kernelH != 3 || conv.strideW != conv.strideH || conv.padLeft != conv.padTop)
                                        {
                                            _ops.ConvDepthWise(
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
                                            _ops.Conv3x3(srcTensor, conv.rawWeight, conv.rawBias, conv.outC, conv.strideW, conv.padLeft, conv.activationType, conv.activationSlope, outTensor);
                                        }

                                        bufferBlobs[layer.topNames[0]] = outTensor.buffer;
                                        bufferViews[layer.topNames[0]] = outTensor;
                                        tempOwned.Add(outTensor);
                                        Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                        continue;
                                    }

                TextureConvPath:
                                    TensorRef src;
                                    RenderTexture tempInputTex = null;
                                        if (textureBlobs.TryGetValue(layer.bottomNames[0], out src) && src != null && src.texture != null)
                                        {
                                        }
                                        else
                                    {
                                        if (!bufferBlobs.TryGetValue(layer.bottomNames[0], out var convInputBuf) || convInputBuf == null)
                                            throw new InvalidOperationException("Convolution source not found: " + layer.name);
                                        var convInputView = TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
                                        if (convInputView == null || convInputView.dims != 3)
                                            throw new InvalidOperationException("Convolution texture path expects dims=3 buffer input: " + layer.name);

                                        var inPacks = (convInputView.c + 3) / 4;
                                        tempInputTex = RentTempArray(convInputView.w, convInputView.h, inPacks, RenderTextureFormat.ARGBHalf);
                                        _ops.FillPack4FromBufferCHW(convInputBuf, convInputView.w, convInputView.h, convInputView.c, tempInputTex);
                                        src = new TensorRef
                                        {
                                            texture = tempInputTex,
                                            width = convInputView.w,
                                            height = convInputView.h,
                                            packs = inPacks,
                                            refs = 1,
                                            owned = false
                                        };
                                    }
                                    var outWTex = ComputeConvOut(src.width, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
                                    var outHTex = ComputeConvOut(src.height, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom);
                                    var outRt = RentTempArray(outWTex, outHTex, conv.outPacks, RenderTextureFormat.ARGBHalf);

                                    if (conv.kernelW == 1 && conv.kernelH == 1 && EnableConv1x1TextureConvolution)
                                    {
                                        if (src.width != outWTex || src.height != outHTex)
                                            throw new InvalidOperationException("Conv1x1 texture path does not support spatial resize: " + layer.name);
                                        _ops.Conv1x1Pack4(src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.activationType, conv.activationSlope, outRt);
                                        if (ShouldCompareTextureConvLayer(layer.name))
                                        {
                                            CompareTextureConvPath(layer.name, layer.bottomNames[0], conv, outWTex, outHTex, outRt, textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                        }
                                    }
                                    else if (canUseDepthWiseTexturePath)
                                    {
                                        _ops.ConvDepthWisePack4(src.texture, conv.packedDepthWiseWeight4, conv.packedBias4, conv.outPacks, conv.kernelW, conv.kernelH, conv.strideW, conv.strideH, conv.padLeft, conv.padTop, conv.dilationW, conv.dilationH, conv.activationType, conv.activationSlope, outRt);
                                        if (ShouldCompareTextureConvLayer(layer.name))
                                        {
                                            CompareTextureConvPath(layer.name, layer.bottomNames[0], conv, outWTex, outHTex, outRt, textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
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
                                        if (EnableWinograd23 && conv.useWinograd23)
                                            _ops.Conv3x3Pack4Winograd23(src.texture, conv.inPacks, conv.packedWeightTm23, conv.packedBias4, conv.outPacks, conv.biasTerm, conv.activationType, conv.activationSlope, outRt);
                                        else
                                            _ops.Conv3x3Pack4(src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.padLeft, conv.activationType, conv.activationSlope, outRt);
                                        if (ShouldCompareTextureConvLayer(layer.name))
                                        {
                                            CompareTextureConvPath(layer.name, layer.bottomNames[0], conv, outWTex, outHTex, outRt, textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                                        }
                                    }
                                    else
                                    {
                                        throw new InvalidOperationException("Texture convolution path unsupported config: " + layer.name);
                                    }

                                    textureBlobs[layer.topNames[0]] = new TensorRef
                                    {
                                        texture = outRt,
                                        width = outWTex,
                                        height = outHTex,
                                        packs = conv.outPacks,
                                        refs = 1,
                                        owned = true
                                    };
                                    textureShapes[layer.topNames[0]] = new BufferShape(3, outWTex, outHTex, 1, conv.outC);
                                    if (tempInputTex != null)
                                        ReturnTempArray(tempInputTex);
                                    Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }

        internal void ExecuteConvolutionFamilyCommandBufferLayer(NcnnParamModel.Layer l, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            do
            {
                                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                                    if (!_conv.TryGetValue(l.name, out var conv))
                                        throw new InvalidOperationException("Convolution not found: " + l.name);
                                    if (src.packs != conv.inPacks)
                                        throw new InvalidOperationException("unexpected in packs for " + l.name + ": " + src.packs + " vs " + conv.inPacks);
                                    if (conv.isDepthWise || conv.group != 1)
                                        throw new InvalidOperationException("CommandBuffer convolution does not support depthwise/group conv: " + l.name);
                                    if (conv.strideW != 1 || conv.strideH != 1 || conv.dilationW != 1 || conv.dilationH != 1)
                                        throw new InvalidOperationException("CommandBuffer convolution only supports stride=1 dilation=1: " + l.name);

                                    var outW = ComputeConvOut(src.width, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
                                    var outH = ComputeConvOut(src.height, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom);
                                    var outArr = RentTempArray(cmd, outW, outH, conv.outPacks, RenderTextureFormat.ARGBHalf);

                                    if (conv.kernelW == 1 && conv.kernelH == 1)
                                    {
                                        _ops.Conv1x1Pack4(cmd, src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.activationType, conv.activationSlope, outArr);
                                    }
                                    else if (conv.kernelW == 3 && conv.kernelH == 3 && conv.padLeft == conv.padRight && conv.padLeft == conv.padTop && conv.padTop == conv.padBottom)
                                    {
                                        var useWinograd = EnableWinograd23
                                            && conv.packedWeightTm23 != null
                                            && conv.strideW == 1
                                            && conv.strideH == 1
                                            && conv.padLeft == 1
                                            && conv.padTop == 1;
                                        if (useWinograd)
                                        {
                                            _ops.Conv3x3Pack4Winograd23(cmd, src.texture, conv.inPacks, conv.packedWeightTm23, conv.packedBias4, conv.outPacks, conv.biasTerm, conv.activationType, conv.activationSlope, outArr);
                                        }
                                        else
                                        {
                                            _ops.Conv3x3Pack4(cmd, src.texture, conv.inPacks, conv.packedWeight4, conv.packedBias4, conv.outPacks, conv.padLeft, conv.activationType, conv.activationSlope, outArr);
                                        }
                                    }
                                    else
                                    {
                                        throw new InvalidOperationException("CommandBuffer convolution only supports 1x1/3x3 symmetric conv: " + l.name);
                                    }

                                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = outW, height = outH, packs = conv.outPacks, refs = 1, owned = true };
                                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                                    continue;
            } while (false);
        }
    }
}
