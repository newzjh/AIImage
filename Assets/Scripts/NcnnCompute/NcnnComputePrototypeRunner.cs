using System;
using System.IO;
using Cysharp.Threading.Tasks;
using UnityEngine;

namespace NcnnCompute
{
    public sealed class NcnnComputePrototypeRunner : MonoBehaviour
    {
        public string paramRelativePath = "RealESRGAN/models/realesrgan-x4plus.param";
        public Texture inputTexture;
        public RenderTexture outputTexture;
        public bool runBufferOpsSelfTest;

        public async UniTask<NcnnParamModel> LoadParamAsync()
        {
            var path = Path.Combine(Application.streamingAssetsPath, paramRelativePath);
            string txt;
            try
            {
                txt = await File.ReadAllTextAsync(path);
            }
            catch (Exception e)
            {
                throw new InvalidOperationException("read param failed: " + path + " " + e.Message);
            }
            return NcnnParamParser.Parse(txt);
        }

        private async void Start()
        {
            try
            {
                await LoadParamAsync();
                if (inputTexture == null)
                {
                    if (runBufferOpsSelfTest)
                        RunBufferOpsSelfTest();
                    return;
                }
                var backend = new NcnnComputeBackend();
                using var t = backend.Passthrough(inputTexture);
                outputTexture = t.rt;
                if (runBufferOpsSelfTest)
                    RunBufferOpsSelfTest();
            }
            catch
            {
            }
        }

        private void RunBufferOpsSelfTest()
        {
            var ops = new NcnnOps();
            SelfTestMatMul(ops);
            SelfTestGemm(ops);
            SelfTestLayerNorm(ops);
            SelfTestSoftmax(ops);
            SelfTestEmbed(ops);
            SelfTestPermute(ops);
            SelfTestSlice(ops);
        }

        private static void SelfTestMatMul(NcnnOps ops)
        {
            const int m = 3;
            const int n = 4;
            const int k = 5;

            var a = new float[m * k];
            var b = new float[k * n];
            var refOut = new float[m * n];
            for (var i = 0; i < a.Length; i++) a[i] = (i * 13 % 17) * 0.1f - 0.7f;
            for (var i = 0; i < b.Length; i++) b[i] = (i * 7 % 19) * 0.05f - 0.4f;

            for (var i = 0; i < m; i++)
                for (var j = 0; j < n; j++)
                {
                    var sum = 0f;
                    for (var kk = 0; kk < k; kk++)
                        sum += a[i * k + kk] * b[kk * n + j];
                    refOut[i * n + j] = sum;
                }

            using var bufA = new ComputeBuffer(a.Length, sizeof(float), ComputeBufferType.Structured);
            using var bufB = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
            using var bufOut = new ComputeBuffer(refOut.Length, sizeof(float), ComputeBufferType.Structured);
            bufA.SetData(a);
            bufB.SetData(b);
            ops.MatMul2D(bufA, bufB, m, n, k, false, bufOut);
            var got = new float[refOut.Length];
            bufOut.GetData(got);

            var maxErr = 0f;
            for (var i = 0; i < got.Length; i++)
                maxErr = Mathf.Max(maxErr, Mathf.Abs(got[i] - refOut[i]));
            UnityEngine.Debug.Log("[SELFTEST] MatMul2D maxErr=" + maxErr);
        }

        private static void SelfTestGemm(NcnnOps ops)
        {
            const int m = 3;
            const int n = 4;
            const int k = 5;
            const float alpha = 0.9f;
            const float beta = 1.1f;

            var a = new float[m * k];
            var b = new float[k * n];
            for (var i = 0; i < a.Length; i++) a[i] = (i * 3 % 23) * 0.03f - 0.2f;
            for (var i = 0; i < b.Length; i++) b[i] = (i * 5 % 29) * 0.02f - 0.25f;

            using var bufA = new ComputeBuffer(a.Length, sizeof(float), ComputeBufferType.Structured);
            using var bufB = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
            bufA.SetData(a);
            bufB.SetData(b);

            var refOut = new float[m * n];
            var got = new float[m * n];

            {
                var c = new float[1] { -0.4f };
                using var bufC = new ComputeBuffer(1, sizeof(float), ComputeBufferType.Structured);
                using var bufOut = new ComputeBuffer(refOut.Length, sizeof(float), ComputeBufferType.Structured);
                bufC.SetData(c);

                for (var i = 0; i < m; i++)
                    for (var j = 0; j < n; j++)
                    {
                        var sum = beta * c[0];
                        var acc = 0f;
                        for (var kk = 0; kk < k; kk++) acc += a[i * k + kk] * b[kk * n + j];
                        sum += alpha * acc;
                        refOut[i * n + j] = sum;
                    }

                ops.Gemm2D(bufA, bufB, bufC, m, n, k, false, alpha, beta, true, 0, bufOut);
                bufOut.GetData(got);
                var maxErr = 0f;
                for (var t = 0; t < got.Length; t++) maxErr = Mathf.Max(maxErr, Mathf.Abs(got[t] - refOut[t]));
                UnityEngine.Debug.Log("[SELFTEST] Gemm2D C=scalar maxErr=" + maxErr);
            }

            {
                var c = new float[m];
                for (var i = 0; i < c.Length; i++) c[i] = (i - 1) * 0.17f;
                using var bufC = new ComputeBuffer(c.Length, sizeof(float), ComputeBufferType.Structured);
                using var bufOut = new ComputeBuffer(refOut.Length, sizeof(float), ComputeBufferType.Structured);
                bufC.SetData(c);

                for (var i = 0; i < m; i++)
                    for (var j = 0; j < n; j++)
                    {
                        var sum = beta * c[i];
                        var acc = 0f;
                        for (var kk = 0; kk < k; kk++) acc += a[i * k + kk] * b[kk * n + j];
                        sum += alpha * acc;
                        refOut[i * n + j] = sum;
                    }

                ops.Gemm2D(bufA, bufB, bufC, m, n, k, false, alpha, beta, true, 1, bufOut);
                bufOut.GetData(got);
                var maxErr = 0f;
                for (var t = 0; t < got.Length; t++) maxErr = Mathf.Max(maxErr, Mathf.Abs(got[t] - refOut[t]));
                UnityEngine.Debug.Log("[SELFTEST] Gemm2D C=row maxErr=" + maxErr);
            }

            {
                var c = new float[n];
                for (var i = 0; i < c.Length; i++) c[i] = (i - 2) * -0.11f;
                using var bufC = new ComputeBuffer(c.Length, sizeof(float), ComputeBufferType.Structured);
                using var bufOut = new ComputeBuffer(refOut.Length, sizeof(float), ComputeBufferType.Structured);
                bufC.SetData(c);

                for (var i = 0; i < m; i++)
                    for (var j = 0; j < n; j++)
                    {
                        var sum = beta * c[j];
                        var acc = 0f;
                        for (var kk = 0; kk < k; kk++) acc += a[i * k + kk] * b[kk * n + j];
                        sum += alpha * acc;
                        refOut[i * n + j] = sum;
                    }

                ops.Gemm2D(bufA, bufB, bufC, m, n, k, false, alpha, beta, true, 4, bufOut);
                bufOut.GetData(got);
                var maxErr = 0f;
                for (var t = 0; t < got.Length; t++) maxErr = Mathf.Max(maxErr, Mathf.Abs(got[t] - refOut[t]));
                UnityEngine.Debug.Log("[SELFTEST] Gemm2D C=col maxErr=" + maxErr);
            }

            {
                var c = new float[m * n];
                for (var i = 0; i < c.Length; i++) c[i] = (i % 7) * 0.07f - 0.2f;
                using var bufC = new ComputeBuffer(c.Length, sizeof(float), ComputeBufferType.Structured);
                using var bufOut = new ComputeBuffer(refOut.Length, sizeof(float), ComputeBufferType.Structured);
                bufC.SetData(c);

                for (var i = 0; i < m; i++)
                    for (var j = 0; j < n; j++)
                    {
                        var sum = beta * c[i * n + j];
                        var acc = 0f;
                        for (var kk = 0; kk < k; kk++) acc += a[i * k + kk] * b[kk * n + j];
                        sum += alpha * acc;
                        refOut[i * n + j] = sum;
                    }

                ops.Gemm2D(bufA, bufB, bufC, m, n, k, false, alpha, beta, true, 3, bufOut);
                bufOut.GetData(got);
                var maxErr = 0f;
                for (var t = 0; t < got.Length; t++) maxErr = Mathf.Max(maxErr, Mathf.Abs(got[t] - refOut[t]));
                UnityEngine.Debug.Log("[SELFTEST] Gemm2D C=full maxErr=" + maxErr);
            }
        }

        private static void SelfTestLayerNorm(NcnnOps ops)
        {
            const int rows = 2;
            const int cols = 8;
            const float eps = 0.001f;

            var x = new float[rows * cols];
            var refY = new float[rows * cols];
            var gamma = new float[cols];
            var beta = new float[cols];
            for (var i = 0; i < x.Length; i++) x[i] = (i * 11 % 31) * 0.02f - 0.3f;
            for (var i = 0; i < cols; i++) gamma[i] = 0.7f + i * 0.01f;
            for (var i = 0; i < cols; i++) beta[i] = -0.1f + i * 0.005f;

            for (var r = 0; r < rows; r++)
            {
                var sum = 0f;
                var sqsum = 0f;
                for (var c = 0; c < cols; c++)
                {
                    var v = x[r * cols + c];
                    sum += v;
                    sqsum += v * v;
                }
                var mean = sum / cols;
                var var = sqsum / cols - mean * mean;
                var invstd = 1f / Mathf.Sqrt(var + eps);
                for (var c = 0; c < cols; c++)
                {
                    var v = (x[r * cols + c] - mean) * invstd;
                    v = v * gamma[c] + beta[c];
                    refY[r * cols + c] = v;
                }
            }

            using var buf = new ComputeBuffer(x.Length, sizeof(float), ComputeBufferType.Structured);
            using var bufGamma = new ComputeBuffer(gamma.Length, sizeof(float), ComputeBufferType.Structured);
            using var bufBeta = new ComputeBuffer(beta.Length, sizeof(float), ComputeBufferType.Structured);
            buf.SetData(x);
            bufGamma.SetData(gamma);
            bufBeta.SetData(beta);
            ops.LayerNorm2DInplace(buf, rows, cols, eps, true, bufGamma, bufBeta);
            var got = new float[refY.Length];
            buf.GetData(got);
            var maxErr = 0f;
            for (var i = 0; i < got.Length; i++) maxErr = Mathf.Max(maxErr, Mathf.Abs(got[i] - refY[i]));
            UnityEngine.Debug.Log("[SELFTEST] LayerNorm2D maxErr=" + maxErr);
        }

        private static void SelfTestSoftmax(NcnnOps ops)
        {
            const int rows = 2;
            const int cols = 8;

            var x = new float[rows * cols];
            var refY = new float[rows * cols];
            for (var i = 0; i < x.Length; i++) x[i] = (i * 9 % 37) * 0.03f - 0.5f;

            for (var r = 0; r < rows; r++)
            {
                var maxv = float.NegativeInfinity;
                for (var c = 0; c < cols; c++) maxv = Mathf.Max(maxv, x[r * cols + c]);
                var sum = 0f;
                for (var c = 0; c < cols; c++)
                {
                    var e = Mathf.Exp(x[r * cols + c] - maxv);
                    refY[r * cols + c] = e;
                    sum += e;
                }
                var inv = 1f / sum;
                for (var c = 0; c < cols; c++) refY[r * cols + c] *= inv;
            }

            using var bufIn = new ComputeBuffer(x.Length, sizeof(float), ComputeBufferType.Structured);
            using var bufOut = new ComputeBuffer(x.Length, sizeof(float), ComputeBufferType.Structured);
            bufIn.SetData(x);
            ops.Softmax2D(bufIn, bufOut, rows, cols);
            var got = new float[refY.Length];
            bufOut.GetData(got);
            var maxErr = 0f;
            for (var i = 0; i < got.Length; i++) maxErr = Mathf.Max(maxErr, Mathf.Abs(got[i] - refY[i]));
            UnityEngine.Debug.Log("[SELFTEST] Softmax2D maxErr=" + maxErr);
        }

        private static void SelfTestEmbed(NcnnOps ops)
        {
            const int words = 3;
            const int numOutput = 4;
            const int inputDim = 6;

            var idx = new int[words] { 0, 3, 5 };
            var w = new float[inputDim * numOutput];
            var b = new float[numOutput];
            var refY = new float[words * numOutput];

            for (var i = 0; i < w.Length; i++) w[i] = (i * 7 % 41) * 0.01f - 0.2f;
            for (var i = 0; i < b.Length; i++) b[i] = (i - 1) * 0.03f;

            for (var q = 0; q < words; q++)
            {
                var wi = Mathf.Clamp(idx[q], 0, inputDim - 1);
                for (var p = 0; p < numOutput; p++)
                    refY[q * numOutput + p] = w[wi * numOutput + p] + b[p];
            }

            using var bufIdx = new ComputeBuffer(idx.Length, sizeof(int), ComputeBufferType.Structured);
            using var bufW = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
            using var bufB = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
            using var bufOut = new ComputeBuffer(refY.Length, sizeof(float), ComputeBufferType.Structured);
            bufIdx.SetData(idx);
            bufW.SetData(w);
            bufB.SetData(b);
            ops.Embed(bufIdx, words, bufW, bufB, numOutput, inputDim, true, bufOut);
            var got = new float[refY.Length];
            bufOut.GetData(got);
            var maxErr = 0f;
            for (var i = 0; i < got.Length; i++) maxErr = Mathf.Max(maxErr, Mathf.Abs(got[i] - refY[i]));
            UnityEngine.Debug.Log("[SELFTEST] Embed maxErr=" + maxErr);
        }

        private static void SelfTestPermute(NcnnOps ops)
        {
            {
                const int dims = 3;
                const int inW = 2;
                const int inH = 3;
                const int inC = 4;
                const int inD = 1;
                const int orderType = 2;

                var inCount = inW * inH * inC;
                var input = new float[inCount];
                for (var i = 0; i < input.Length; i++) input[i] = i;

                var outW = inW;
                var outH = inC;
                var outC = inH;
                var outCount = outW * outH * outC;
                var refY = new float[outCount];
                for (var oc = 0; oc < outC; oc++)
                    for (var oh = 0; oh < outH; oh++)
                        for (var ow = 0; ow < outW; ow++)
                        {
                            var iw = ow;
                            var ih = oc;
                            var ic = oh;
                            refY[(oc * outH + oh) * outW + ow] = input[(ic * inH + ih) * inW + iw];
                        }

                using var bufIn = new ComputeBuffer(inCount, sizeof(float), ComputeBufferType.Structured);
                using var bufOut = new ComputeBuffer(outCount, sizeof(float), ComputeBufferType.Structured);
                bufIn.SetData(input);
                ops.Permute(bufIn, dims, inW, inH, inD, inC, orderType, bufOut);
                var got = new float[outCount];
                bufOut.GetData(got);
                var maxErr = 0f;
                for (var i = 0; i < got.Length; i++) maxErr = Mathf.Max(maxErr, Mathf.Abs(got[i] - refY[i]));
                UnityEngine.Debug.Log("[SELFTEST] Permute dims3 ot2 maxErr=" + maxErr);
            }

            {
                const int dims = 4;
                const int inW = 2;
                const int inH = 3;
                const int inD = 4;
                const int inC = 2;
                const int orderType = 3;

                var inCount = inW * inH * inD * inC;
                var input = new float[inCount];
                for (var i = 0; i < input.Length; i++) input[i] = i * 0.5f;

                var outW = inD;
                var outH = inW;
                var outD = inH;
                var outC = inC;
                var outCount = outW * outH * outD * outC;
                var refY = new float[outCount];

                for (var oc = 0; oc < outC; oc++)
                    for (var od = 0; od < outD; od++)
                        for (var oh = 0; oh < outH; oh++)
                            for (var ow = 0; ow < outW; ow++)
                            {
                                var ic = oc;
                                var iw = oh;
                                var ih = od;
                                var idd = ow;
                                var inIdx = (((ic * inD + idd) * inH + ih) * inW + iw);
                                var outIdx = (((oc * outD + od) * outH + oh) * outW + ow);
                                refY[outIdx] = input[inIdx];
                            }

                using var bufIn = new ComputeBuffer(inCount, sizeof(float), ComputeBufferType.Structured);
                using var bufOut = new ComputeBuffer(outCount, sizeof(float), ComputeBufferType.Structured);
                bufIn.SetData(input);
                ops.Permute(bufIn, dims, inW, inH, inD, inC, orderType, bufOut);
                var got = new float[outCount];
                bufOut.GetData(got);
                var maxErr = 0f;
                for (var i = 0; i < got.Length; i++) maxErr = Mathf.Max(maxErr, Mathf.Abs(got[i] - refY[i]));
                UnityEngine.Debug.Log("[SELFTEST] Permute dims4 ot3 maxErr=" + maxErr);
            }
        }

        private static void SelfTestSlice(NcnnOps ops)
        {
            const int dims = 3;
            const int inW = 2;
            const int inH = 3;
            const int inC = 5;
            const int inD = 1;
            const int axis = 2;
            const int begin = 1;
            const int outC = 3;
            const int outW = inW;
            const int outH = inH;
            const int outD = 1;

            var inCount = inW * inH * inC;
            var input = new float[inCount];
            for (var i = 0; i < input.Length; i++) input[i] = i * 0.25f - 0.7f;

            var outCount = outW * outH * outC;
            var refY = new float[outCount];
            for (var oc = 0; oc < outC; oc++)
                for (var y = 0; y < outH; y++)
                    for (var x = 0; x < outW; x++)
                    {
                        var ic = oc + begin;
                        refY[(oc * outH + y) * outW + x] = input[(ic * inH + y) * inW + x];
                    }

            using var bufIn = new ComputeBuffer(inCount, sizeof(float), ComputeBufferType.Structured);
            using var bufOut = new ComputeBuffer(outCount, sizeof(float), ComputeBufferType.Structured);
            bufIn.SetData(input);
            ops.Slice(bufIn, dims, inW, inH, inD, inC, axis, begin, outW, outH, outD, outC, bufOut);
            var got = new float[outCount];
            bufOut.GetData(got);
            var maxErr = 0f;
            for (var i = 0; i < got.Length; i++) maxErr = Mathf.Max(maxErr, Mathf.Abs(got[i] - refY[i]));
            UnityEngine.Debug.Log("[SELFTEST] Slice dims3 axis2 maxErr=" + maxErr);
        }
    }
}
