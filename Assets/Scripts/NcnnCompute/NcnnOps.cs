using System;
using UnityEngine;

namespace NcnnCompute
{
    public sealed class NcnnOps
    {
        private readonly ComputeShader _cs;
        private readonly int _kConv3x3;
        private readonly int _kTexToBuf3;
        private readonly int _kBufToTex3;
        private readonly int _kLeakyReluBuf;
        private readonly int _kAddWeighted;
        private readonly int _kCopyC;
        private readonly int _kInterp2x;
        private readonly int _kBlitTileToDst;
        private readonly int _kPackRgbToPack4;
        private readonly int _kConv3x3Pack4;
        private readonly int _kConv1x1Pack4;
        private readonly int _kAddPack4;
        private readonly int _kCopyPack4;
        private readonly int _kInterp2xPack4;
        private readonly int _kInterp2xNearestPack4;
        private readonly int _kInterpDown2Pack4;
        private readonly int _kInterpDown2NearestPack4;
        private readonly int _kPack4ToBufferChw;
        private readonly int _kInnerProduct;
        private readonly int _kPackRgbToPack4Gfpgan;
        private readonly int _kFillPack4FromBufferChw;
        private readonly int _kScalePack4;
        private readonly int _kAddBiasPack4;
        private readonly int _kLeakyReluPack4;
        private readonly int _kAddNoiseBroadcastPack4;
        private readonly int _kClipPack4;
        private readonly int _kSftPack4;
        private readonly int _kPack4ToRgb01;
        private readonly int _kProbeTilePack4;
        private readonly int _kProbeSeams;
        private readonly int _kPaddingPack4;
        private readonly int _kPoolingPack4;
        private readonly int _kSoftmaxChannelPack4;
        private readonly int _kUnaryOpPack4;
        private readonly int _kBinaryOpPack4;
        private readonly int _kSwishPack4;
        private readonly int _kSigmoidPack4;
        private readonly int _kGeluPack4;

        public NcnnOps()
        {
            _cs = Resources.Load<ComputeShader>("NcnnCompute");
            if (_cs == null)
                throw new InvalidOperationException("ComputeShader not found: Resources/NcnnCompute.compute");
            _kConv3x3 = _cs.FindKernel("NcnnConv3x3");
            _kTexToBuf3 = _cs.FindKernel("NcnnTexToBuf3");
            _kBufToTex3 = _cs.FindKernel("NcnnBufToTex3");
            _kLeakyReluBuf = _cs.FindKernel("NcnnLeakyReluBuf");
            _kAddWeighted = _cs.FindKernel("NcnnAddWeighted");
            _kCopyC = _cs.FindKernel("NcnnCopyC");
            _kInterp2x = _cs.FindKernel("NcnnInterp2x");
            _kBlitTileToDst = _cs.FindKernel("NcnnBlitTileToDst");
            _kPackRgbToPack4 = _cs.FindKernel("NcnnPackRgbToPack4");
            _kConv3x3Pack4 = _cs.FindKernel("NcnnConv3x3Pack4");
            _kConv1x1Pack4 = _cs.FindKernel("NcnnConv1x1Pack4");
            _kAddPack4 = _cs.FindKernel("NcnnAddPack4");
            _kCopyPack4 = _cs.FindKernel("NcnnCopyPack4");
            _kInterp2xPack4 = _cs.FindKernel("NcnnInterp2xPack4");
            _kInterp2xNearestPack4 = _cs.FindKernel("NcnnInterp2xNearestPack4");
            _kInterpDown2Pack4 = _cs.FindKernel("NcnnInterpDown2Pack4");
            _kInterpDown2NearestPack4 = _cs.FindKernel("NcnnInterpDown2NearestPack4");
            _kPack4ToBufferChw = _cs.FindKernel("NcnnPack4ToBufferCHW");
            _kInnerProduct = _cs.FindKernel("NcnnInnerProduct");
            _kPackRgbToPack4Gfpgan = _cs.FindKernel("NcnnPackRgbToPack4Gfpgan");
            _kFillPack4FromBufferChw = _cs.FindKernel("NcnnFillPack4FromBufferCHW");
            _kScalePack4 = _cs.FindKernel("NcnnScalePack4");
            _kAddBiasPack4 = _cs.FindKernel("NcnnAddBiasPack4");
            _kLeakyReluPack4 = _cs.FindKernel("NcnnLeakyReluPack4");
            _kAddNoiseBroadcastPack4 = _cs.FindKernel("NcnnAddNoiseBroadcastPack4");
            _kClipPack4 = _cs.FindKernel("NcnnClipPack4");
            _kSftPack4 = _cs.FindKernel("NcnnSftPack4");
            _kPack4ToRgb01 = _cs.FindKernel("NcnnPack4ToRgb01");
            _kProbeTilePack4 = _cs.FindKernel("NcnnProbeTilePack4");
            _kProbeSeams = _cs.FindKernel("NcnnProbeSeams");
            _kPaddingPack4 = _cs.FindKernel("NcnnPaddingPack4");
            _kPoolingPack4 = _cs.FindKernel("NcnnPoolingPack4");
            _kSoftmaxChannelPack4 = _cs.FindKernel("NcnnSoftmaxChannelPack4");
            _kUnaryOpPack4 = _cs.FindKernel("NcnnUnaryOpPack4");
            _kBinaryOpPack4 = _cs.FindKernel("NcnnBinaryOpPack4");
            _kSwishPack4 = _cs.FindKernel("NcnnSwishPack4");
            _kSigmoidPack4 = _cs.FindKernel("NcnnSigmoidPack4");
            _kGeluPack4 = _cs.FindKernel("NcnnGeluPack4");
        }

        public void TextureToBuffer3(Texture src, int offsetX, int offsetY, NcnnTensorBuffer output)
        {
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (output.c != 3) throw new ArgumentOutOfRangeException(nameof(output), "output.c must be 3");

            _cs.SetInt("_InW", output.w);
            _cs.SetInt("_InH", output.h);
            _cs.SetInt("_OffsetX", offsetX);
            _cs.SetInt("_OffsetY", offsetY);
            _cs.SetTexture(_kTexToBuf3, "_NcnnIn", src);
            _cs.SetBuffer(_kTexToBuf3, "_BufOut", output.buffer);

            var n = output.w * output.h;
            Dispatch1D(_kTexToBuf3, n, 256);
        }

        public void BufferToTexture3(NcnnTensorBuffer input, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (input.c != 3) throw new ArgumentOutOfRangeException(nameof(input), "input.c must be 3");
            if (input.w != output.width || input.h != output.height)
                throw new ArgumentOutOfRangeException(nameof(output), "output dimensions must match input");

            _cs.SetBuffer(_kBufToTex3, "_BufA", input.buffer);
            _cs.SetTexture(_kBufToTex3, "_NcnnOut", output);
            Dispatch2D(_kBufToTex3, output.width, output.height, 8, 8);
        }

        public void BlitTileToDst(RenderTexture tileOut, RenderTexture dst, int dstX, int dstY, int tileOutOriginX, int tileOutOriginY, int w, int h, float dstToSrcScale)
        {
            if (tileOut == null) throw new ArgumentNullException(nameof(tileOut));
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (w <= 0 || h <= 0) return;
            if (dstX < 0 || dstY < 0 || dstX + w > dst.width || dstY + h > dst.height)
                throw new ArgumentOutOfRangeException(nameof(dstX), "dst rect out of range");

            _cs.SetInt("_BlitW", w);
            _cs.SetInt("_BlitH", h);
            _cs.SetInt("_DstX", dstX);
            _cs.SetInt("_DstY", dstY);
            _cs.SetInt("_TileOutX", tileOutOriginX);
            _cs.SetInt("_TileOutY", tileOutOriginY);
            _cs.SetFloat("_BlitScale", dstToSrcScale);
            _cs.SetTexture(_kBlitTileToDst, "_NcnnInArr", tileOut);
            _cs.SetTexture(_kBlitTileToDst, "_NcnnOut", dst);
            Dispatch2D(_kBlitTileToDst, w, h, 32, 32);
        }

        public void PackRgbToPack4(Texture src, int offsetX, int offsetY, RenderTexture dstPack4)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (dstPack4 == null) throw new ArgumentNullException(nameof(dstPack4));
            _cs.SetInt("_OffsetX", offsetX);
            _cs.SetInt("_OffsetY", offsetY);
            _cs.SetTexture(_kPackRgbToPack4, "_NcnnIn", src);
            _cs.SetTexture(_kPackRgbToPack4, "_NcnnOutArr", dstPack4);
            Dispatch2D(_kPackRgbToPack4, dstPack4.width, dstPack4.height, 32, 32);
        }

        public void PackRgbToPack4Gfpgan(Texture src, int offsetX, int offsetY, RenderTexture dstPack4)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (dstPack4 == null) throw new ArgumentNullException(nameof(dstPack4));
            _cs.SetInt("_OffsetX", offsetX);
            _cs.SetInt("_OffsetY", offsetY);
            _cs.SetTexture(_kPackRgbToPack4Gfpgan, "_NcnnIn", src);
            _cs.SetTexture(_kPackRgbToPack4Gfpgan, "_NcnnOutArr", dstPack4);
            Dispatch2D(_kPackRgbToPack4Gfpgan, dstPack4.width, dstPack4.height, 32, 32);
        }

        public void FillPack4FromBufferCHW(ComputeBuffer input, int w, int h, int c, RenderTexture outputPack4)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (outputPack4 == null) throw new ArgumentNullException(nameof(outputPack4));
            _cs.SetInt("_FillW", w);
            _cs.SetInt("_FillH", h);
            _cs.SetInt("_FillC", c);
            _cs.SetBuffer(_kFillPack4FromBufferChw, "_FillIn", input);
            _cs.SetTexture(_kFillPack4FromBufferChw, "_FillOutArr", outputPack4);
            Dispatch3D(_kFillPack4FromBufferChw, w, h, outputPack4.volumeDepth, 8, 8);
        }

        public void ScalePack4(RenderTexture input, float k, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetFloat("_ScaleK", k);
            _cs.SetTexture(_kScalePack4, "_ScaleInArr", input);
            _cs.SetTexture(_kScalePack4, "_ScaleOutArr", output);
            Dispatch3D(_kScalePack4, output.width, output.height, packs, 8, 8);
        }

        public void AddBiasPack4(RenderTexture input, ComputeBuffer bias4, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (bias4 == null) throw new ArgumentNullException(nameof(bias4));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetBuffer(_kAddBiasPack4, "_Bias4", bias4);
            _cs.SetTexture(_kAddBiasPack4, "_BiasInArr", input);
            _cs.SetTexture(_kAddBiasPack4, "_BiasOutArr", output);
            Dispatch3D(_kAddBiasPack4, output.width, output.height, packs, 8, 8);
        }

        public void LeakyReluPack4(RenderTexture input, float slope, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetFloat("_LreluSlope", slope);
            _cs.SetTexture(_kLeakyReluPack4, "_LreluInArr", input);
            _cs.SetTexture(_kLeakyReluPack4, "_LreluOutArr", output);
            Dispatch3D(_kLeakyReluPack4, output.width, output.height, packs, 8, 8);
        }

        public void AddNoiseBroadcastPack4(RenderTexture inOut, ComputeBuffer noise, float weight, int packs)
        {
            if (inOut == null) throw new ArgumentNullException(nameof(inOut));
            if (noise == null) throw new ArgumentNullException(nameof(noise));
            _cs.SetFloat("_NoiseWeight", weight);
            _cs.SetBuffer(_kAddNoiseBroadcastPack4, "_Noise", noise);
            _cs.SetTexture(_kAddNoiseBroadcastPack4, "_NoiseInOutArr", inOut);
            Dispatch3D(_kAddNoiseBroadcastPack4, inOut.width, inOut.height, packs, 8, 8);
        }

        public void ClipPack4(RenderTexture input, float min, float max, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetFloat("_ClipMin", min);
            _cs.SetFloat("_ClipMax", max);
            _cs.SetTexture(_kClipPack4, "_ClipInArr", input);
            _cs.SetTexture(_kClipPack4, "_ClipOutArr", output);
            Dispatch3D(_kClipPack4, output.width, output.height, packs, 8, 8);
        }

        public void SftPack4(RenderTexture input, RenderTexture condMul, RenderTexture condAdd, int outPacks, int halfPacks, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (condMul == null) throw new ArgumentNullException(nameof(condMul));
            if (condAdd == null) throw new ArgumentNullException(nameof(condAdd));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_SftHalfPacks", halfPacks);
            _cs.SetTexture(_kSftPack4, "_SftInArr", input);
            _cs.SetTexture(_kSftPack4, "_SftCondMulArr", condMul);
            _cs.SetTexture(_kSftPack4, "_SftCondAddArr", condAdd);
            _cs.SetTexture(_kSftPack4, "_SftOutArr", output);
            Dispatch3D(_kSftPack4, output.width, output.height, outPacks, 8, 8);
        }

        public void Pack4ToRgb01(RenderTexture inputPack4, RenderTexture outputRgb)
        {
            if (inputPack4 == null) throw new ArgumentNullException(nameof(inputPack4));
            if (outputRgb == null) throw new ArgumentNullException(nameof(outputRgb));
            _cs.SetTexture(_kPack4ToRgb01, "_RgbInArr", inputPack4);
            _cs.SetTexture(_kPack4ToRgb01, "_RgbOut", outputRgb);
            Dispatch2D(_kPack4ToRgb01, outputRgb.width, outputRgb.height, 32, 32);
        }

        public void ProbeTilePack4(RenderTexture tileOutPack4, int probeIndex, int pad, int coreW, int coreH, ComputeBuffer probeOut)
        {
            if (tileOutPack4 == null) throw new ArgumentNullException(nameof(tileOutPack4));
            if (probeOut == null) throw new ArgumentNullException(nameof(probeOut));
            _cs.SetInt("_ProbeIndex", probeIndex);
            _cs.SetInt("_ProbePad", pad);
            _cs.SetInt("_ProbeCoreW", coreW);
            _cs.SetInt("_ProbeCoreH", coreH);
            _cs.SetTexture(_kProbeTilePack4, "_ProbeInArr", tileOutPack4);
            _cs.SetBuffer(_kProbeTilePack4, "_ProbeOut", probeOut);
            _cs.Dispatch(_kProbeTilePack4, 1, 1, 1);
        }

        public void ProbeSeams(RenderTexture tex, int tilesX, int tilesY, int stepX, int stepY, int samples, ComputeBuffer seamOut4)
        {
            if (tex == null) throw new ArgumentNullException(nameof(tex));
            if (seamOut4 == null) throw new ArgumentNullException(nameof(seamOut4));
            _cs.SetTexture(_kProbeSeams, "_SeamTex", tex);
            _cs.SetBuffer(_kProbeSeams, "_SeamOut4", seamOut4);
            _cs.SetInt("_SeamW", tex.width);
            _cs.SetInt("_SeamH", tex.height);
            _cs.SetInt("_SeamTilesX", tilesX);
            _cs.SetInt("_SeamTilesY", tilesY);
            _cs.SetInt("_SeamStepX", stepX);
            _cs.SetInt("_SeamStepY", stepY);
            _cs.SetInt("_SeamSamples", samples);
            var seamCount = Mathf.Max(0, tilesX - 1) + Mathf.Max(0, tilesY - 1);
            if (seamCount <= 0) return;
            var groups = Mathf.CeilToInt(seamCount / 64f);
            _cs.Dispatch(_kProbeSeams, groups, 1, 1);
        }

        public void PaddingPack4(RenderTexture input, int packs, int padLeft, int padRight, int padTop, int padBottom, int padType, Vector4 padValue, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_PadLeft", padLeft);
            _cs.SetInt("_PadRight", padRight);
            _cs.SetInt("_PadTop", padTop);
            _cs.SetInt("_PadBottom", padBottom);
            _cs.SetInt("_PadType", padType);
            _cs.SetVector("_PadValue4", padValue);
            _cs.SetTexture(_kPaddingPack4, "_PadInArr", input);
            _cs.SetTexture(_kPaddingPack4, "_PadOutArr", output);
            Dispatch3D(_kPaddingPack4, output.width, output.height, packs, 8, 8);
        }

        public void PoolingPack4(RenderTexture input, int packs, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padTop, int poolType, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_PoolKernelW", kernelW);
            _cs.SetInt("_PoolKernelH", kernelH);
            _cs.SetInt("_PoolStrideW", strideW);
            _cs.SetInt("_PoolStrideH", strideH);
            _cs.SetInt("_PoolPadLeft", padLeft);
            _cs.SetInt("_PoolPadTop", padTop);
            _cs.SetInt("_PoolType", poolType);
            _cs.SetTexture(_kPoolingPack4, "_PoolInArr", input);
            _cs.SetTexture(_kPoolingPack4, "_PoolOutArr", output);
            Dispatch3D(_kPoolingPack4, output.width, output.height, packs, 8, 8);
        }

        public void SoftmaxChannelPack4(RenderTexture input, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_SoftmaxPacks", packs);
            _cs.SetTexture(_kSoftmaxChannelPack4, "_SoftmaxInArr", input);
            _cs.SetTexture(_kSoftmaxChannelPack4, "_SoftmaxOutArr", output);
            Dispatch3D(_kSoftmaxChannelPack4, output.width, output.height, packs, 8, 8);
        }

        public void UnaryOpPack4(RenderTexture input, int packs, int opType, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_UnaryOpType", opType);
            _cs.SetTexture(_kUnaryOpPack4, "_UnaryInArr", input);
            _cs.SetTexture(_kUnaryOpPack4, "_UnaryOutArr", output);
            Dispatch3D(_kUnaryOpPack4, output.width, output.height, packs, 8, 8);
        }

        public void BinaryOpPack4(RenderTexture a, RenderTexture b, int packs, int opType, RenderTexture output)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_BinaryOpType", opType);
            _cs.SetInt("_BinaryWithScalar", 0);
            _cs.SetFloat("_BinaryScalar", 0f);
            _cs.SetTexture(_kBinaryOpPack4, "_BinaryA", a);
            _cs.SetTexture(_kBinaryOpPack4, "_BinaryB", b);
            _cs.SetTexture(_kBinaryOpPack4, "_BinaryOutArr", output);
            Dispatch3D(_kBinaryOpPack4, output.width, output.height, packs, 8, 8);
        }

        public void BinaryOpScalarPack4(RenderTexture a, float b, int packs, int opType, RenderTexture output)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_BinaryOpType", opType);
            _cs.SetInt("_BinaryWithScalar", 1);
            _cs.SetFloat("_BinaryScalar", b);
            _cs.SetTexture(_kBinaryOpPack4, "_BinaryA", a);
            _cs.SetTexture(_kBinaryOpPack4, "_BinaryB", a);
            _cs.SetTexture(_kBinaryOpPack4, "_BinaryOutArr", output);
            Dispatch3D(_kBinaryOpPack4, output.width, output.height, packs, 8, 8);
        }

        public void SwishPack4(RenderTexture input, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetTexture(_kSwishPack4, "_ActInArr", input);
            _cs.SetTexture(_kSwishPack4, "_ActOutArr", output);
            Dispatch3D(_kSwishPack4, output.width, output.height, packs, 8, 8);
        }

        public void SigmoidPack4(RenderTexture input, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetTexture(_kSigmoidPack4, "_ActInArr", input);
            _cs.SetTexture(_kSigmoidPack4, "_ActOutArr", output);
            Dispatch3D(_kSigmoidPack4, output.width, output.height, packs, 8, 8);
        }

        public void GeluPack4(RenderTexture input, int packs, bool fastGelu, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_GeluFast", fastGelu ? 1 : 0);
            _cs.SetTexture(_kGeluPack4, "_ActInArr", input);
            _cs.SetTexture(_kGeluPack4, "_ActOutArr", output);
            Dispatch3D(_kGeluPack4, output.width, output.height, packs, 8, 8);
        }

        public void Conv3x3Pack4(RenderTexture srcPack4, int inPacks, ComputeBuffer w4, ComputeBuffer b4, int outPacks, int pad, int activationType, float activationParam, RenderTexture dstPack4)
        {
            if (srcPack4 == null) throw new ArgumentNullException(nameof(srcPack4));
            if (dstPack4 == null) throw new ArgumentNullException(nameof(dstPack4));
            if (w4 == null) throw new ArgumentNullException(nameof(w4));
            if (b4 == null) throw new ArgumentNullException(nameof(b4));
            if (inPacks <= 0) throw new ArgumentOutOfRangeException(nameof(inPacks));
            if (outPacks <= 0) throw new ArgumentOutOfRangeException(nameof(outPacks));

            _cs.SetInt("_InPacks", inPacks);
            _cs.SetInt("_OutPacks", outPacks);
            _cs.SetInt("_Pad", Mathf.Max(0, pad));
            _cs.SetInt("_ActType", activationType);
            _cs.SetFloat("_ActParam", activationParam);
            _cs.SetBuffer(_kConv3x3Pack4, "_ConvW4", w4);
            _cs.SetBuffer(_kConv3x3Pack4, "_ConvB4", b4);
            _cs.SetTexture(_kConv3x3Pack4, "_ConvInArr", srcPack4);
            _cs.SetTexture(_kConv3x3Pack4, "_ConvOutArr", dstPack4);
            Dispatch3D(_kConv3x3Pack4, dstPack4.width, dstPack4.height, outPacks, 8, 8);
        }

        public void Conv1x1Pack4(RenderTexture srcPack4, int inPacks, ComputeBuffer w4, ComputeBuffer b4, int outPacks, int activationType, float activationParam, RenderTexture dstPack4)
        {
            if (srcPack4 == null) throw new ArgumentNullException(nameof(srcPack4));
            if (dstPack4 == null) throw new ArgumentNullException(nameof(dstPack4));
            if (w4 == null) throw new ArgumentNullException(nameof(w4));
            if (b4 == null) throw new ArgumentNullException(nameof(b4));
            if (inPacks <= 0) throw new ArgumentOutOfRangeException(nameof(inPacks));
            if (outPacks <= 0) throw new ArgumentOutOfRangeException(nameof(outPacks));

            _cs.SetInt("_InPacks", inPacks);
            _cs.SetInt("_OutPacks", outPacks);
            _cs.SetInt("_ActType", activationType);
            _cs.SetFloat("_ActParam", activationParam);
            _cs.SetBuffer(_kConv1x1Pack4, "_ConvW4", w4);
            _cs.SetBuffer(_kConv1x1Pack4, "_ConvB4", b4);
            _cs.SetTexture(_kConv1x1Pack4, "_ConvInArr", srcPack4);
            _cs.SetTexture(_kConv1x1Pack4, "_ConvOutArr", dstPack4);
            Dispatch3D(_kConv1x1Pack4, dstPack4.width, dstPack4.height, outPacks, 8, 8);
        }

        public void AddPack4(RenderTexture a, RenderTexture b, float coeffA, float coeffB, int packs, RenderTexture output)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetFloat("_CoeffA", coeffA);
            _cs.SetFloat("_CoeffB", coeffB);
            _cs.SetTexture(_kAddPack4, "_AddA", a);
            _cs.SetTexture(_kAddPack4, "_AddB", b);
            _cs.SetTexture(_kAddPack4, "_AddOutArr", output);
            Dispatch3D(_kAddPack4, output.width, output.height, packs, 8, 8);
        }

        public void CopyPack4(RenderTexture src, int srcPackOffset, RenderTexture dst, int dstPackOffset, int packs)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (packs <= 0) return;
            if (src.width != dst.width || src.height != dst.height)
                throw new InvalidOperationException("CopyPack4 requires same width/height");

            for (var p = 0; p < packs; p++)
            {
                var sp = srcPackOffset + p;
                var dp = dstPackOffset + p;
                Graphics.CopyTexture(src, sp, 0, 0, 0, src.width, src.height, dst, dp, 0, 0, 0);
            }
        }

        public void Interp2xPack4(RenderTexture input, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetTexture(_kInterp2xPack4, "_InterpInArr", input);
            _cs.SetTexture(_kInterp2xPack4, "_InterpOutArr", output);
            Dispatch3D(_kInterp2xPack4, output.width, output.height, packs, 8, 8);
        }

        public void Interp2xNearestPack4(RenderTexture input, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetTexture(_kInterp2xNearestPack4, "_InterpNnInArr", input);
            _cs.SetTexture(_kInterp2xNearestPack4, "_InterpNnOutArr", output);
            Dispatch3D(_kInterp2xNearestPack4, output.width, output.height, packs, 8, 8);
        }

        public void InterpDown2Pack4(RenderTexture input, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetTexture(_kInterpDown2Pack4, "_InterpDownInArr", input);
            _cs.SetTexture(_kInterpDown2Pack4, "_InterpDownOutArr", output);
            Dispatch3D(_kInterpDown2Pack4, output.width, output.height, packs, 8, 8);
        }

        public void InterpDown2NearestPack4(RenderTexture input, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetTexture(_kInterpDown2NearestPack4, "_InterpDownNnInArr", input);
            _cs.SetTexture(_kInterpDown2NearestPack4, "_InterpDownNnOutArr", output);
            Dispatch3D(_kInterpDown2NearestPack4, output.width, output.height, packs, 8, 8);
        }

        public void Pack4ToBufferCHW(RenderTexture input, int w, int h, int c, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            var total = w * h * c;
            if (total <= 0) return;
            _cs.SetInt("_Pack4W", w);
            _cs.SetInt("_Pack4H", h);
            _cs.SetInt("_Pack4C", c);
            _cs.SetTexture(_kPack4ToBufferChw, "_Pack4InArr", input);
            _cs.SetBuffer(_kPack4ToBufferChw, "_Pack4Out", output);
            Dispatch1D(_kPack4ToBufferChw, total, 256);
        }

        public void InnerProduct(ComputeBuffer input, int inFeatures, ComputeBuffer weights, ComputeBuffer bias, int outFeatures, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (bias == null) throw new ArgumentNullException(nameof(bias));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (inFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(inFeatures));
            if (outFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(outFeatures));

            _cs.SetInt("_IPInFeatures", inFeatures);
            _cs.SetInt("_IPOutFeatures", outFeatures);
            _cs.SetBuffer(_kInnerProduct, "_IPIn", input);
            _cs.SetBuffer(_kInnerProduct, "_IPW", weights);
            _cs.SetBuffer(_kInnerProduct, "_IPB", bias);
            _cs.SetBuffer(_kInnerProduct, "_IPOut", output);
            Dispatch1D(_kInnerProduct, outFeatures, 64);
        }

        public void Conv3x3(NcnnTensorBuffer input, ComputeBuffer weightsOihw, ComputeBuffer biasO, int outC, int stride, int pad, int activationType, float activationParam, NcnnTensorBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (weightsOihw == null) throw new ArgumentNullException(nameof(weightsOihw));
            if (biasO == null) throw new ArgumentNullException(nameof(biasO));
            if (outC <= 0) throw new ArgumentOutOfRangeException(nameof(outC));

            _cs.SetInt("_InW", input.w);
            _cs.SetInt("_InH", input.h);
            _cs.SetInt("_InC", input.c);
            _cs.SetInt("_OutC", outC);
            _cs.SetInt("_Stride", Mathf.Max(1, stride));
            _cs.SetInt("_Pad", Mathf.Max(0, pad));
            _cs.SetInt("_ActType", activationType);
            _cs.SetFloat("_ActParam", activationParam);
            _cs.SetBuffer(_kConv3x3, "_ConvIn", input.buffer);
            _cs.SetBuffer(_kConv3x3, "_ConvW", weightsOihw);
            _cs.SetBuffer(_kConv3x3, "_ConvB", biasO);
            _cs.SetBuffer(_kConv3x3, "_ConvOut", output.buffer);

            var total = input.w * input.h * outC;
            Dispatch1D(_kConv3x3, total, 64);
        }

        public void LeakyReluInplace(NcnnTensorBuffer t, float slope)
        {
            if (t == null) throw new ArgumentNullException(nameof(t));
            var total = t.w * t.h * t.c;
            _cs.SetInt("_Total", total);
            _cs.SetFloat("_CoeffA", slope);
            _cs.SetBuffer(_kLeakyReluBuf, "_BufOut", t.buffer);
            Dispatch1D(_kLeakyReluBuf, total, 256);
        }

        public void AddWeighted(NcnnTensorBuffer a, NcnnTensorBuffer b, float coeffA, float coeffB, NcnnTensorBuffer output)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (a.w != b.w || a.h != b.h || a.c != b.c || a.w != output.w || a.h != output.h || a.c != output.c)
                throw new ArgumentOutOfRangeException(nameof(output), "shape mismatch");

            var total = output.w * output.h * output.c;
            _cs.SetInt("_Total", total);
            _cs.SetFloat("_CoeffA", coeffA);
            _cs.SetFloat("_CoeffB", coeffB);
            _cs.SetBuffer(_kAddWeighted, "_BufA", a.buffer);
            _cs.SetBuffer(_kAddWeighted, "_BufB", b.buffer);
            _cs.SetBuffer(_kAddWeighted, "_BufOut", output.buffer);
            Dispatch1D(_kAddWeighted, total, 256);
        }

        public void CopyToConcat(NcnnTensorBuffer src, NcnnTensorBuffer dst, int dstChannelOffset)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (src.w != dst.w || src.h != dst.h) throw new ArgumentOutOfRangeException(nameof(dst), "shape mismatch");
            if (dstChannelOffset < 0 || dstChannelOffset + src.c > dst.c) throw new ArgumentOutOfRangeException(nameof(dstChannelOffset));

            var total = src.w * src.h * src.c;
            _cs.SetInt("_InW", src.w);
            _cs.SetInt("_InH", src.h);
            _cs.SetInt("_ChanOffset", dstChannelOffset);
            _cs.SetInt("_CopyTotal", total);
            _cs.SetBuffer(_kCopyC, "_BufA", src.buffer);
            _cs.SetBuffer(_kCopyC, "_BufOut", dst.buffer);
            Dispatch1D(_kCopyC, total, 256);
        }

        public void Interp2x(NcnnTensorBuffer input, NcnnTensorBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (output.w != input.w * 2 || output.h != input.h * 2 || output.c != input.c)
                throw new ArgumentOutOfRangeException(nameof(output), "output shape must be 2x");

            _cs.SetInt("_InW", input.w);
            _cs.SetInt("_InH", input.h);
            _cs.SetInt("_InC", input.c);
            _cs.SetBuffer(_kInterp2x, "_BufA", input.buffer);
            _cs.SetBuffer(_kInterp2x, "_BufOut", output.buffer);
            var total = output.w * output.h * output.c;
            Dispatch1D(_kInterp2x, total, 64);
        }

        private static void Dispatch2D(ComputeShader cs, int kernel, int w, int h, int tx, int ty)
        {
            var gx = Mathf.CeilToInt(w / (float)tx);
            var gy = Mathf.CeilToInt(h / (float)ty);
            cs.Dispatch(kernel, Mathf.Max(1, gx), Mathf.Max(1, gy), 1);
        }

        private static void Dispatch3D(ComputeShader cs, int kernel, int w, int h, int z, int tx, int ty)
        {
            var gx = Mathf.CeilToInt(w / (float)tx);
            var gy = Mathf.CeilToInt(h / (float)ty);
            cs.Dispatch(kernel, Mathf.Max(1, gx), Mathf.Max(1, gy), Mathf.Max(1, z));
        }

        private void Dispatch2D(int kernel, int w, int h, int tx, int ty)
        {
            Dispatch2D(_cs, kernel, w, h, tx, ty);
        }

        private void Dispatch3D(int kernel, int w, int h, int z, int tx, int ty)
        {
            Dispatch3D(_cs, kernel, w, h, z, tx, ty);
        }

        private void Dispatch1D(int kernel, int total, int threadsPerGroup)
        {
            if (total <= 0)
                return;

            const int maxGroups = 65535;
            var maxThreads = threadsPerGroup * maxGroups;
            var baseIndex = 0;
            while (baseIndex < total)
            {
                var remaining = total - baseIndex;
                var dispatchCount = Mathf.Min(remaining, maxThreads);
                var groups = Mathf.CeilToInt(dispatchCount / (float)threadsPerGroup);
                _cs.SetInt("_BaseIndex", baseIndex);
                _cs.Dispatch(kernel, Mathf.Max(1, groups), 1, 1);
                baseIndex += groups * threadsPerGroup;
            }
        }
    }
}
