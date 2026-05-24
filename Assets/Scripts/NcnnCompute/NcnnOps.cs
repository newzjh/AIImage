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
        private readonly int _kAddPack4;
        private readonly int _kCopyPack4;
        private readonly int _kInterp2xPack4;

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
            _kAddPack4 = _cs.FindKernel("NcnnAddPack4");
            _kCopyPack4 = _cs.FindKernel("NcnnCopyPack4");
            _kInterp2xPack4 = _cs.FindKernel("NcnnInterp2xPack4");
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
            Dispatch2D(_kBlitTileToDst, w, h, 8, 8);
        }

        public void PackRgbToPack4(Texture src, int offsetX, int offsetY, RenderTexture dstPack4)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (dstPack4 == null) throw new ArgumentNullException(nameof(dstPack4));
            _cs.SetInt("_OffsetX", offsetX);
            _cs.SetInt("_OffsetY", offsetY);
            _cs.SetTexture(_kPackRgbToPack4, "_NcnnIn", src);
            _cs.SetTexture(_kPackRgbToPack4, "_NcnnOutArr", dstPack4);
            Dispatch2D(_kPackRgbToPack4, dstPack4.width, dstPack4.height, 8, 8);
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
            _cs.SetInt("_CopyInOffset", srcPackOffset);
            _cs.SetInt("_CopyOutOffset", dstPackOffset);
            _cs.SetInt("_CopyPacks", packs);
            _cs.SetTexture(_kCopyPack4, "_CopyInArr", src);
            _cs.SetTexture(_kCopyPack4, "_CopyOutArr", dst);
            Dispatch3D(_kCopyPack4, dst.width, dst.height, packs, 8, 8);
        }

        public void Interp2xPack4(RenderTexture input, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetTexture(_kInterp2xPack4, "_InterpInArr", input);
            _cs.SetTexture(_kInterp2xPack4, "_InterpOutArr", output);
            Dispatch3D(_kInterp2xPack4, output.width, output.height, packs, 8, 8);
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
