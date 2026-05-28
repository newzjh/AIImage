using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
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

        public static void RunSelfTestsFromUI()
        {
            var runner = FindAnyObjectByType<NcnnComputePrototypeRunner>();
            if (runner == null)
            {
                var go = new GameObject("NcnnComputePrototypeRunner");
                runner = go.AddComponent<NcnnComputePrototypeRunner>();
            }
            runner.RunSelfTests();
        }

        public static void RunSdParamSupportReportFromUI()
        {
            try
            {
                var root = Directory.GetParent(Application.dataPath)?.FullName ?? "";
                var supported = new HashSet<string>(StringComparer.Ordinal)
                {
                    "Input","Split","Concat",
                    "Reshape","Permute","Slice","ExpandDims",
                    "Tile",
                    "Convolution","Interp","Eltwise",
                    "BinaryOp","UnaryOp","Swish","Sigmoid","GELU","Softmax",
                    "Padding","Pooling",
                    "InnerProduct",
                    "MatMul","Gemm",
                    "MultiHeadAttention",
                    "LayerNorm","GroupNorm",
                    "Embed",
                    "Reduction",
                    "MemoryData"
                };

                var baseDir = Path.Combine(root, "ref", "Stable-Diffusion-NCNN-main", "Windows", "Binary", "x64", "assets");
                var paramFiles = new[]
                {
                    "FrozenCLIPEmbedder-fp16.param",
                    "UNetModel-base-MHA-fp16.param",
                    "AutoencoderKL-base-fp16.param"
                };

                string ResolveBinPath(string dir, string paramFileName)
                {
                    var baseName0 = Path.GetFileNameWithoutExtension(paramFileName);
                    var candidates = new List<string>(8) { baseName0 };
                    if (baseName0.Contains("-base-", StringComparison.Ordinal))
                        candidates.Add(baseName0.Replace("-base-", "-", StringComparison.Ordinal));
                    if (baseName0.Contains("-base", StringComparison.Ordinal))
                        candidates.Add(baseName0.Replace("-base", "", StringComparison.Ordinal));
                    if (baseName0.Contains("UNetModel-base-", StringComparison.Ordinal))
                        candidates.Add(baseName0.Replace("UNetModel-base-", "UNetModel-", StringComparison.Ordinal));
                    if (baseName0.Contains("AutoencoderKL-base-", StringComparison.Ordinal))
                        candidates.Add(baseName0.Replace("AutoencoderKL-base-", "AutoencoderKL-", StringComparison.Ordinal));
                    candidates.Add(baseName0.Replace("-fp16", "", StringComparison.Ordinal));
                    candidates.Add(baseName0.Replace("-fp16", "", StringComparison.Ordinal).Replace("-base", "", StringComparison.Ordinal));

                    for (var i = 0; i < candidates.Count; i++)
                    {
                        var p = Path.Combine(dir, candidates[i] + ".bin");
                        if (File.Exists(p))
                            return p;
                    }
                    return Path.Combine(dir, baseName0 + ".bin");
                }

                foreach (var fn in paramFiles)
                {
                    var path = Path.Combine(baseDir, fn);
                    if (!File.Exists(path))
                    {
                        Debug.Log("[SD] param not found: " + path);
                        continue;
                    }

                    var txt = File.ReadAllText(path);
                    var model = NcnnParamParser.Parse(txt);
                    var counts = new Dictionary<string, int>(StringComparer.Ordinal);
                    foreach (var l in model.layers)
                    {
                        var key = l.type ?? "";
                        if (!counts.TryGetValue(key, out var c)) c = 0;
                        counts[key] = c + 1;
                    }

                    var missing = counts.Keys.Where(t => !supported.Contains(t)).OrderBy(t => t).ToArray();
                    var top = counts.OrderByDescending(kv => kv.Value).Take(32).Select(kv => kv.Key + ":" + kv.Value).ToArray();
                    Debug.Log("[SD] " + fn + " layers=" + model.layers.Count + " types=" + counts.Count + " | top=" + string.Join(",", top));
                    if (missing.Length > 0)
                        Debug.Log("[SD] " + fn + " unsupported: " + string.Join(",", missing));

                    var baseName = Path.GetFileNameWithoutExtension(fn);
                    var directBin = ResolveBinPath(baseDir, fn);
                    var streamingHit = Array.Empty<string>();
                    try
                    {
                        streamingHit = Directory.GetFiles(Application.streamingAssetsPath, baseName + ".bin", SearchOption.AllDirectories);
                    }
                    catch
                    {
                    }
                    var haveDirect = File.Exists(directBin);
                    var haveStreaming = streamingHit.Length > 0;
                    Debug.Log("[SD] " + baseName + ".bin present: direct=" + (haveDirect ? 1 : 0) + " streaming=" + (haveStreaming ? streamingHit.Length : 0) + (haveDirect ? (" | directPath=" + directBin) : ""));
                }
            }
            catch (Exception e)
            {
                Debug.Log("[SD] support report failed: " + e);
            }
        }

        public static void RunSdMemoryDataDumpFromUI()
        {
            try
            {
                var root = Directory.GetParent(Application.dataPath)?.FullName ?? "";
                var baseDir = Path.Combine(root, "ref", "Stable-Diffusion-NCNN-main", "Windows", "Binary", "x64", "assets");
                var paramPath = Path.Combine(baseDir, "FrozenCLIPEmbedder-fp16.param");
                var binPath = Path.Combine(baseDir, "FrozenCLIPEmbedder-fp16.bin");
                if (!File.Exists(paramPath) || !File.Exists(binPath))
                {
                    Debug.Log("[SD] MemoryData dump missing files: param=" + (File.Exists(paramPath) ? 1 : 0) + " bin=" + (File.Exists(binPath) ? 1 : 0));
                    Debug.Log("[SD] paramPath=" + paramPath);
                    Debug.Log("[SD] binPath=" + binPath);
                    return;
                }

                var model = NcnnParamParser.Parse(File.ReadAllText(paramPath));
                using var fs = File.OpenRead(binPath);
                using var br = new NcnnBinReader(fs);

                var buffers = new List<ComputeBuffer>();
                try
                {
                    var count = 0;
                    foreach (var l in model.layers)
                    {
                        if (!string.Equals(l.type, "MemoryData", StringComparison.Ordinal))
                            continue;

                        var w = l.GetInt(0, 0);
                        var h = l.GetInt(1, 0);
                        var d = l.GetInt(11, 0);
                        var c = l.GetInt(2, 0);
                        var loadType = l.GetInt(21, 1);

                        var a = br.ReadNcnnMatAsFloat32(w, h, d, c, loadType);
                        var buf = new ComputeBuffer(a.Length, sizeof(float), ComputeBufferType.Structured);
                        buf.SetData(a);
                        buffers.Add(buf);
                        count++;

                        var shape = d != 0 ? (w + "x" + h + "x" + d + "x" + c)
                            : (c != 0 ? (w + "x" + h + "x" + c)
                            : (h != 0 ? (w + "x" + h) : w.ToString()));
                        var v0 = a.Length > 0 ? a[0].ToString("0.######") : "n/a";
                        Debug.Log("[SD] MemoryData " + l.name + " shape=" + shape + " loadType=" + loadType + " count=" + a.Length + " v0=" + v0);
                    }
                    Debug.Log("[SD] MemoryData dump done, count=" + count + " binPos=" + br.Position);
                }
                finally
                {
                    foreach (var b in buffers)
                    {
                        try { b?.Dispose(); } catch { }
                    }
                }
            }
            catch (Exception e)
            {
                Debug.Log("[SD] MemoryData dump failed: " + e);
            }
        }

        public static void RunSdClipSmokeFromUI()
        {
            try
            {
                var root = Directory.GetParent(Application.dataPath)?.FullName ?? "";
                var baseDir = Path.Combine(root, "ref", "Stable-Diffusion-NCNN-main", "Windows", "Binary", "x64", "assets");
                var paramPath = Path.Combine(baseDir, "FrozenCLIPEmbedder-fp16.param");
                var binPath = Path.Combine(baseDir, "FrozenCLIPEmbedder-fp16.bin");
                if (!File.Exists(paramPath) || !File.Exists(binPath))
                {
                    Debug.Log("[SD] CLIP smoke missing files: param=" + (File.Exists(paramPath) ? 1 : 0) + " bin=" + (File.Exists(binPath) ? 1 : 0));
                    Debug.Log("[SD] paramPath=" + paramPath);
                    Debug.Log("[SD] binPath=" + binPath);
                    return;
                }

                using var repro = new NcnnRepro(new NcnnOps());
                using (var fs = File.OpenRead(binPath))
                using (var br = new NcnnBinReader(fs))
                {
                    repro.LoadModel(paramPath, br);
                }

                const int words = 77;
                const int startTok = 49406;
                const int endTok = 49407;
                var tokenIds = new int[words];
                tokenIds[0] = startTok;
                tokenIds[words - 1] = endTok;
                using var tokBuf = new ComputeBuffer(words, sizeof(int), ComputeBufferType.Structured);
                tokBuf.SetData(tokenIds);

                var input = new Dictionary<string, ComputeBuffer>(StringComparer.Ordinal)
                {
                    { "token", tokBuf }
                };

                const string firstLayerNormOutput = "19";
                using (var result = repro.InferFromBuffers(input, firstLayerNormOutput))
                {
                    var outBuf = result.GetBuffer(firstLayerNormOutput);
                    if (outBuf != null)
                    {
                        var peek = new float[Math.Min(16, outBuf.count)];
                        outBuf.GetData(peek, 0, 0, peek.Length);
                        var maxAbs = 0f;
                        for (var k = 0; k < peek.Length; k++)
                            maxAbs = Mathf.Max(maxAbs, Mathf.Abs(peek[k]));
                        Debug.Log("[SD] CLIP smoke ok: firstLayerNorm=" + firstLayerNormOutput + " count=" + outBuf.count + " peekMaxAbs=" + maxAbs.ToString("0.######", System.Globalization.CultureInfo.InvariantCulture));
                    }
                }
            }
            catch (Exception e)
            {
                Debug.Log("[SD] CLIP smoke failed: " + e);
            }
        }

        public static void RunSdWeightScanFromUI()
        {
            try
            {
                var root = Directory.GetParent(Application.dataPath)?.FullName ?? "";
                var baseDir = Path.Combine(root, "ref", "Stable-Diffusion-NCNN-main", "Windows", "Binary", "x64", "assets");

                var paramFiles = new[]
                {
                    "FrozenCLIPEmbedder-fp16.param",
                    "UNetModel-base-MHA-fp16.param",
                    "AutoencoderKL-base-fp16.param"
                };

                string ResolveBinPath(string dir, string paramFileName)
                {
                    var baseName0 = Path.GetFileNameWithoutExtension(paramFileName);
                    var candidates = new List<string>(8) { baseName0 };
                    if (baseName0.Contains("-base-", StringComparison.Ordinal))
                        candidates.Add(baseName0.Replace("-base-", "-", StringComparison.Ordinal));
                    if (baseName0.Contains("-base", StringComparison.Ordinal))
                        candidates.Add(baseName0.Replace("-base", "", StringComparison.Ordinal));
                    if (baseName0.Contains("UNetModel-base-", StringComparison.Ordinal))
                        candidates.Add(baseName0.Replace("UNetModel-base-", "UNetModel-", StringComparison.Ordinal));
                    if (baseName0.Contains("AutoencoderKL-base-", StringComparison.Ordinal))
                        candidates.Add(baseName0.Replace("AutoencoderKL-base-", "AutoencoderKL-", StringComparison.Ordinal));

                    for (var i = 0; i < candidates.Count; i++)
                    {
                        var p = Path.Combine(dir, candidates[i] + ".bin");
                        if (File.Exists(p))
                            return p;
                    }
                    return Path.Combine(dir, baseName0 + ".bin");
                }

                foreach (var pf in paramFiles)
                {
                    var paramPath = Path.Combine(baseDir, pf);
                    if (!File.Exists(paramPath))
                    {
                        Debug.Log("[SD] WeightScan missing param: " + paramPath);
                        continue;
                    }

                    var binPath = ResolveBinPath(baseDir, pf);
                    if (!File.Exists(binPath))
                    {
                        Debug.Log("[SD] WeightScan missing bin for " + pf + " -> " + binPath);
                        continue;
                    }

                    var model = NcnnParamParser.Parse(File.ReadAllText(paramPath));
                    using var fs = File.OpenRead(binPath);
                    using var br = new NcnnBinReader(fs);

                    int wLayers = 0;
                    for (var i = 0; i < model.layers.Count; i++)
                    {
                        var l = model.layers[i];

                        if (string.Equals(l.type, "Convolution", StringComparison.Ordinal))
                        {
                            var numOut = l.GetInt(0, 0);
                            var biasTerm = l.GetInt(5, 0) != 0;
                            var weightSize = l.GetInt(6, 0);
                            br.SkipNcnnArray(weightSize, 0);
                            if (biasTerm)
                                br.SkipNcnnArray(numOut, 1);
                            wLayers++;
                            continue;
                        }

                        if (string.Equals(l.type, "InnerProduct", StringComparison.Ordinal))
                        {
                            var numOut = l.GetInt(0, 0);
                            var biasTerm = l.GetInt(1, 0) != 0;
                            var weightSize = l.GetInt(2, 0);
                            br.SkipNcnnArray(weightSize, 0);
                            if (biasTerm)
                                br.SkipNcnnArray(numOut, 1);
                            wLayers++;
                            continue;
                        }

                        if (string.Equals(l.type, "LayerNorm", StringComparison.Ordinal))
                        {
                            var affineSize = l.GetInt(0, 0);
                            var affine = l.GetInt(2, 1) != 0;
                            if (affine)
                            {
                                br.SkipNcnnArray(affineSize, 1);
                                br.SkipNcnnArray(affineSize, 1);
                                wLayers++;
                            }
                            continue;
                        }

                        if (string.Equals(l.type, "GroupNorm", StringComparison.Ordinal))
                        {
                            var channels = l.GetInt(0, 0);
                            var affine = l.GetInt(3, 1) != 0;
                            if (affine)
                            {
                                br.SkipNcnnArray(channels, 1);
                                br.SkipNcnnArray(channels, 1);
                                wLayers++;
                            }
                            continue;
                        }

                        if (string.Equals(l.type, "Embed", StringComparison.Ordinal))
                        {
                            var numOut = l.GetInt(0, 0);
                            var biasTerm = l.GetInt(2, 0) != 0;
                            var weightSize = l.GetInt(3, 0);
                            br.SkipNcnnArray(weightSize, 0);
                            if (biasTerm)
                                br.SkipNcnnArray(numOut, 1);
                            wLayers++;
                            continue;
                        }

                        if (string.Equals(l.type, "MultiHeadAttention", StringComparison.Ordinal))
                        {
                            var embedDim = l.GetInt(0, 0);
                            var weightSize = l.GetInt(2, 0);
                            var kdim = l.GetInt(3, embedDim);
                            var vdim = l.GetInt(4, embedDim);
                            var qdim = embedDim > 0 ? (weightSize / Math.Max(1, embedDim)) : 0;

                            br.SkipNcnnArray(embedDim * qdim, 0);
                            br.SkipNcnnArray(embedDim, 1);
                            br.SkipNcnnArray(embedDim * kdim, 0);
                            br.SkipNcnnArray(embedDim, 1);
                            br.SkipNcnnArray(embedDim * vdim, 0);
                            br.SkipNcnnArray(embedDim, 1);
                            br.SkipNcnnArray(qdim * embedDim, 0);
                            br.SkipNcnnArray(qdim, 1);
                            wLayers++;
                            continue;
                        }
                    }

                    Debug.Log("[SD] WeightScan ok: " + Path.GetFileName(paramPath) + " | weightedLayers=" + wLayers + " | binPos=" + br.Position + " / " + fs.Length + " | bin=" + Path.GetFileName(binPath));
                }
            }
            catch (Exception e)
            {
                Debug.Log("[SD] WeightScan failed: " + e);
            }
        }

        public static void RunSdUnetMhaFromUI()
        {
            try
            {
                var root = Directory.GetParent(Application.dataPath)?.FullName ?? "";
                var baseDir = Path.Combine(root, "ref", "Stable-Diffusion-NCNN-main", "Windows", "Binary", "x64", "assets");
                var paramPath = Path.Combine(baseDir, "UNetModel-base-MHA-fp16.param");
                var binPath = ResolveSdBinPath(baseDir, "UNetModel-base-MHA-fp16.param");
                if (!File.Exists(paramPath) || !File.Exists(binPath))
                {
                    Debug.Log("[SD] UNet MHA missing files: param=" + (File.Exists(paramPath) ? 1 : 0) + " bin=" + (File.Exists(binPath) ? 1 : 0));
                    Debug.Log("[SD] paramPath=" + paramPath);
                    Debug.Log("[SD] binPath=" + binPath);
                    return;
                }

                var model = NcnnParamParser.Parse(File.ReadAllText(paramPath));
                var mhaLayer = model.layers.FirstOrDefault(l => string.Equals(l.type, "MultiHeadAttention", StringComparison.Ordinal));
                if (mhaLayer == null)
                {
                    Debug.Log("[SD] UNet MHA not found in param");
                    return;
                }

                using var fs = File.OpenRead(binPath);
                using var br = new NcnnBinReader(fs);

                var ops = new NcnnOps();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var found = false;

                for (var i = 0; i < model.layers.Count; i++)
                {
                    var l = model.layers[i];

                    if (ReferenceEquals(l, mhaLayer))
                    {
                        var embedDim = l.GetInt(0, 0);
                        var numHeads = l.GetInt(1, 1);
                        var weightDataSize = l.GetInt(2, 0);
                        var kdim = l.GetInt(3, embedDim);
                        var vdim = l.GetInt(4, embedDim);
                        var scale = l.GetFloat(6, 1f / Mathf.Sqrt(Mathf.Max(1, embedDim / Mathf.Max(1, numHeads))));
                        var qdim = embedDim > 0 ? (weightDataSize / Math.Max(1, embedDim)) : 0;

                        var qW = br.ReadNcnnMatAsFloat32(embedDim * qdim, 0, 0, 0, 0);
                        var qB = br.ReadNcnnMatAsFloat32(embedDim, 0, 0, 0, 1);
                        var kW = br.ReadNcnnMatAsFloat32(embedDim * kdim, 0, 0, 0, 0);
                        var kB = br.ReadNcnnMatAsFloat32(embedDim, 0, 0, 0, 1);
                        var vW = br.ReadNcnnMatAsFloat32(embedDim * vdim, 0, 0, 0, 0);
                        var vB = br.ReadNcnnMatAsFloat32(embedDim, 0, 0, 0, 1);
                        var oW = br.ReadNcnnMatAsFloat32(qdim * embedDim, 0, 0, 0, 0);
                        var oB = br.ReadNcnnMatAsFloat32(qdim, 0, 0, 0, 1);

                        const int srcLen = 16;
                        const int dstLen = 16;
                        var q = new float[srcLen * qdim];
                        var k = new float[dstLen * kdim];
                        var v = new float[dstLen * vdim];
                        for (var t = 0; t < q.Length; t++) q[t] = (t * 13 % 17) * 0.07f - 0.5f;
                        for (var t = 0; t < k.Length; t++) k[t] = (t * 7 % 19) * 0.05f - 0.4f;
                        for (var t = 0; t < v.Length; t++) v[t] = (t * 11 % 23) * 0.03f - 0.3f;

                        var refOut = CpuMhaFull(q, k, v, srcLen, dstLen, embedDim, numHeads, qdim, kdim, vdim, scale, qW, qB, kW, kB, vW, vB, oW, oB);

                        using var bufQ = new ComputeBuffer(q.Length, sizeof(float), ComputeBufferType.Structured);
                        using var bufK = new ComputeBuffer(k.Length, sizeof(float), ComputeBufferType.Structured);
                        using var bufV = new ComputeBuffer(v.Length, sizeof(float), ComputeBufferType.Structured);
                        bufQ.SetData(q);
                        bufK.SetData(k);
                        bufV.SetData(v);

                        using var bufQW = new ComputeBuffer(qW.Length, sizeof(float), ComputeBufferType.Structured);
                        using var bufQB = new ComputeBuffer(qB.Length, sizeof(float), ComputeBufferType.Structured);
                        using var bufKW = new ComputeBuffer(kW.Length, sizeof(float), ComputeBufferType.Structured);
                        using var bufKB = new ComputeBuffer(kB.Length, sizeof(float), ComputeBufferType.Structured);
                        using var bufVW = new ComputeBuffer(vW.Length, sizeof(float), ComputeBufferType.Structured);
                        using var bufVB = new ComputeBuffer(vB.Length, sizeof(float), ComputeBufferType.Structured);
                        using var bufOW = new ComputeBuffer(oW.Length, sizeof(float), ComputeBufferType.Structured);
                        using var bufOB = new ComputeBuffer(oB.Length, sizeof(float), ComputeBufferType.Structured);
                        bufQW.SetData(qW);
                        bufQB.SetData(qB);
                        bufKW.SetData(kW);
                        bufKB.SetData(kB);
                        bufVW.SetData(vW);
                        bufVB.SetData(vB);
                        bufOW.SetData(oW);
                        bufOB.SetData(oB);

                        using var qAff = new ComputeBuffer(srcLen * embedDim, sizeof(float), ComputeBufferType.Structured);
                        using var kAff = new ComputeBuffer(dstLen * embedDim, sizeof(float), ComputeBufferType.Structured);
                        using var vAff = new ComputeBuffer(dstLen * embedDim, sizeof(float), ComputeBufferType.Structured);
                        ops.InnerProduct2D(bufQ, srcLen, qdim, bufQW, bufQB, embedDim, qAff);
                        ops.InnerProduct2D(bufK, dstLen, kdim, bufKW, bufKB, embedDim, kAff);
                        ops.InnerProduct2D(bufV, dstLen, vdim, bufVW, bufVB, embedDim, vAff);

                        using var qAffScaled = new ComputeBuffer(srcLen * embedDim, sizeof(float), ComputeBufferType.Structured);
                        ops.BinaryOpScalarBuf(qAff, scale, qAff.count, 2, qAffScaled);

                        using var ctx = new ComputeBuffer(srcLen * embedDim, sizeof(float), ComputeBufferType.Structured);
                        ops.MhaAttention(qAffScaled, kAff, vAff, srcLen, dstLen, embedDim, numHeads, 1f, ctx);

                        using var outBuf = new ComputeBuffer(srcLen * qdim, sizeof(float), ComputeBufferType.Structured);
                        ops.InnerProduct2D(ctx, srcLen, embedDim, bufOW, bufOB, qdim, outBuf);

                        var got = new float[refOut.Length];
                        outBuf.GetData(got);

                        var maxErr = 0f;
                        for (var t = 0; t < got.Length; t++)
                            maxErr = Mathf.Max(maxErr, Mathf.Abs(got[t] - refOut[t]));

                        Debug.Log("[SD] UNet MHA ok: layer=" + l.name + " embedDim=" + embedDim + " heads=" + numHeads + " qdim=" + qdim + " kdim=" + kdim + " vdim=" + vdim + " srcLen=" + srcLen + " dstLen=" + dstLen + " maxErr=" + maxErr.ToString("0.########", System.Globalization.CultureInfo.InvariantCulture) + " ms=" + sw.ElapsedMilliseconds);
                        found = true;
                        break;
                    }

                    SkipLayerWeights(br, l);
                }

                if (!found)
                    Debug.Log("[SD] UNet MHA failed: could not locate weights at runtime");
            }
            catch (Exception e)
            {
                Debug.Log("[SD] UNet MHA failed: " + e);
            }
        }

        private static string ResolveSdBinPath(string baseDir, string paramFileName)
        {
            var baseName0 = Path.GetFileNameWithoutExtension(paramFileName);
            var candidates = new List<string>(8) { baseName0 };
            if (baseName0.Contains("-base-", StringComparison.Ordinal))
                candidates.Add(baseName0.Replace("-base-", "-", StringComparison.Ordinal));
            if (baseName0.Contains("-base", StringComparison.Ordinal))
                candidates.Add(baseName0.Replace("-base", "", StringComparison.Ordinal));
            if (baseName0.Contains("UNetModel-base-", StringComparison.Ordinal))
                candidates.Add(baseName0.Replace("UNetModel-base-", "UNetModel-", StringComparison.Ordinal));
            if (baseName0.Contains("AutoencoderKL-base-", StringComparison.Ordinal))
                candidates.Add(baseName0.Replace("AutoencoderKL-base-", "AutoencoderKL-", StringComparison.Ordinal));

            for (var i = 0; i < candidates.Count; i++)
            {
                var p = Path.Combine(baseDir, candidates[i] + ".bin");
                if (File.Exists(p))
                    return p;
            }
            return Path.Combine(baseDir, baseName0 + ".bin");
        }

        private static void SkipLayerWeights(NcnnBinReader br, NcnnParamModel.Layer l)
        {
            if (string.Equals(l.type, "Convolution", StringComparison.Ordinal))
            {
                var numOut = l.GetInt(0, 0);
                var biasTerm = l.GetInt(5, 0) != 0;
                var weightSize = l.GetInt(6, 0);
                br.SkipNcnnArray(weightSize, 0);
                if (biasTerm)
                    br.SkipNcnnArray(numOut, 1);
                return;
            }

            if (string.Equals(l.type, "InnerProduct", StringComparison.Ordinal))
            {
                var numOut = l.GetInt(0, 0);
                var biasTerm = l.GetInt(1, 0) != 0;
                var weightSize = l.GetInt(2, 0);
                br.SkipNcnnArray(weightSize, 0);
                if (biasTerm)
                    br.SkipNcnnArray(numOut, 1);
                return;
            }

            if (string.Equals(l.type, "LayerNorm", StringComparison.Ordinal))
            {
                var affineSize = l.GetInt(0, 0);
                var affine = l.GetInt(2, 1) != 0;
                if (affine)
                {
                    br.SkipNcnnArray(affineSize, 1);
                    br.SkipNcnnArray(affineSize, 1);
                }
                return;
            }

            if (string.Equals(l.type, "GroupNorm", StringComparison.Ordinal))
            {
                var channels = l.GetInt(1, 0);
                var affine = l.GetInt(3, 1) != 0;
                if (affine)
                {
                    br.SkipNcnnArray(channels, 1);
                    br.SkipNcnnArray(channels, 1);
                }
                return;
            }

            if (string.Equals(l.type, "Embed", StringComparison.Ordinal))
            {
                var numOut = l.GetInt(0, 0);
                var biasTerm = l.GetInt(2, 0) != 0;
                var weightSize = l.GetInt(3, 0);
                br.SkipNcnnArray(weightSize, 0);
                if (biasTerm)
                    br.SkipNcnnArray(numOut, 1);
                return;
            }

            if (string.Equals(l.type, "MultiHeadAttention", StringComparison.Ordinal))
            {
                var embedDim = l.GetInt(0, 0);
                var weightSize = l.GetInt(2, 0);
                var kdim = l.GetInt(3, embedDim);
                var vdim = l.GetInt(4, embedDim);
                var qdim = embedDim > 0 ? (weightSize / Math.Max(1, embedDim)) : 0;

                br.SkipNcnnArray(embedDim * qdim, 0);
                br.SkipNcnnArray(embedDim, 1);
                br.SkipNcnnArray(embedDim * kdim, 0);
                br.SkipNcnnArray(embedDim, 1);
                br.SkipNcnnArray(embedDim * vdim, 0);
                br.SkipNcnnArray(embedDim, 1);
                br.SkipNcnnArray(qdim * embedDim, 0);
                br.SkipNcnnArray(qdim, 1);
            }
        }

        private static float[] CpuMhaFull(
            float[] q, float[] k, float[] v,
            int srcLen, int dstLen,
            int embedDim, int numHeads,
            int qdim, int kdim, int vdim,
            float scale,
            float[] qW, float[] qB,
            float[] kW, float[] kB,
            float[] vW, float[] vB,
            float[] oW, float[] oB)
        {
            var qAff = new float[srcLen * embedDim];
            var kAff = new float[dstLen * embedDim];
            var vAff = new float[dstLen * embedDim];

            for (var r = 0; r < srcLen; r++)
                for (var o = 0; o < embedDim; o++)
                {
                    var sum = qB[o];
                    var wBase = o * qdim;
                    var inBase = r * qdim;
                    for (var i = 0; i < qdim; i++) sum += q[inBase + i] * qW[wBase + i];
                    qAff[r * embedDim + o] = sum * scale;
                }

            for (var r = 0; r < dstLen; r++)
                for (var o = 0; o < embedDim; o++)
                {
                    var sum = kB[o];
                    var wBase = o * kdim;
                    var inBase = r * kdim;
                    for (var i = 0; i < kdim; i++) sum += k[inBase + i] * kW[wBase + i];
                    kAff[r * embedDim + o] = sum;
                }

            for (var r = 0; r < dstLen; r++)
                for (var o = 0; o < embedDim; o++)
                {
                    var sum = vB[o];
                    var wBase = o * vdim;
                    var inBase = r * vdim;
                    for (var i = 0; i < vdim; i++) sum += v[inBase + i] * vW[wBase + i];
                    vAff[r * embedDim + o] = sum;
                }

            var headDim = embedDim / numHeads;
            var ctx = new float[srcLen * embedDim];
            var scores = new float[dstLen];

            for (var qi = 0; qi < srcLen; qi++)
            {
                for (var h = 0; h < numHeads; h++)
                {
                    var maxv = float.NegativeInfinity;
                    for (var j = 0; j < dstLen; j++)
                    {
                        var s = 0f;
                        var qBase = qi * embedDim + h * headDim;
                        var kBase = j * embedDim + h * headDim;
                        for (var d = 0; d < headDim; d++)
                            s += qAff[qBase + d] * kAff[kBase + d];
                        scores[j] = s;
                        if (s > maxv) maxv = s;
                    }

                    var sum = 0f;
                    for (var j = 0; j < dstLen; j++)
                    {
                        var e = Mathf.Exp(scores[j] - maxv);
                        scores[j] = e;
                        sum += e;
                    }
                    var invSum = 1f / Mathf.Max(sum, 1e-20f);

                    for (var d = 0; d < headDim; d++)
                    {
                        var acc = 0f;
                        for (var j = 0; j < dstLen; j++)
                        {
                            var w = scores[j] * invSum;
                            acc += w * vAff[j * embedDim + h * headDim + d];
                        }
                        ctx[qi * embedDim + h * headDim + d] = acc;
                    }
                }
            }

            var out0 = new float[srcLen * qdim];
            for (var r = 0; r < srcLen; r++)
                for (var o = 0; o < qdim; o++)
                {
                    var sum = oB[o];
                    var wBase = o * embedDim;
                    var inBase = r * embedDim;
                    for (var i = 0; i < embedDim; i++) sum += ctx[inBase + i] * oW[wBase + i];
                    out0[r * qdim + o] = sum;
                }

            return out0;
        }

        public void RunSelfTests()
        {
            RunBufferOpsSelfTest();
        }

        public void RunBufferOpsSelfTest()
        {
            var ops = new NcnnOps();
            SelfTestMatMul(ops);
            SelfTestGemm(ops);
            SelfTestLayerNorm(ops);
            SelfTestSoftmax(ops);
            SelfTestEmbed(ops);
            SelfTestPermute(ops);
            SelfTestSlice(ops);
            SelfTestTile(ops);
            SelfTestReduceAll(ops);
            SelfTestGroupNorm(ops);
            SelfTestMhaAttention(ops);
            SelfTestSwishGeluBuf(ops);
            SelfTestReshapeExpandDims();
        }

        private static void SelfTestSwishGeluBuf(NcnnOps ops)
        {
            const int n = 256;
            var x = new float[n];
            for (var i = 0; i < n; i++) x[i] = (i - 128) * 0.01f;

            var refSwish = new float[n];
            var refGelu = new float[n];
            for (var i = 0; i < n; i++)
            {
                var v = x[i];
                var s = 1f / (1f + Mathf.Exp(-v));
                refSwish[i] = v * s;
                var t = 0.7978845608f * (v + 0.044715f * v * v * v);
                refGelu[i] = 0.5f * v * (1f + (float)Math.Tanh(t));
            }

            using var bufIn = new ComputeBuffer(n, sizeof(float), ComputeBufferType.Structured);
            using var bufOut = new ComputeBuffer(n, sizeof(float), ComputeBufferType.Structured);
            bufIn.SetData(x);

            ops.SwishBuf(bufIn, n, bufOut);
            var got = new float[n];
            bufOut.GetData(got);
            var maxErr = 0f;
            for (var i = 0; i < n; i++) maxErr = Mathf.Max(maxErr, Mathf.Abs(got[i] - refSwish[i]));
            Debug.Log("[SELFTEST] SwishBuf maxErr=" + maxErr);

            ops.GeluBuf(bufIn, n, bufOut);
            bufOut.GetData(got);
            maxErr = 0f;
            for (var i = 0; i < n; i++) maxErr = Mathf.Max(maxErr, Mathf.Abs(got[i] - refGelu[i]));
            Debug.Log("[SELFTEST] GeluBuf maxErr=" + maxErr);
        }

        private static void SelfTestMhaAttention(NcnnOps ops)
        {
            const int srcLen = 3;
            const int dstLen = 4;
            const int embedDim = 8;
            const int numHeads = 2;
            const float scale = 0.125f;
            const int headDim = embedDim / numHeads;

            var q = new float[srcLen * embedDim];
            var k = new float[dstLen * embedDim];
            var v = new float[dstLen * embedDim];
            for (var i = 0; i < q.Length; i++) q[i] = (i * 13 % 17) * 0.07f - 0.5f;
            for (var i = 0; i < k.Length; i++) k[i] = (i * 7 % 19) * 0.05f - 0.4f;
            for (var i = 0; i < v.Length; i++) v[i] = (i * 11 % 23) * 0.03f - 0.3f;

            var refOut = new float[srcLen * embedDim];
            for (var qi = 0; qi < srcLen; qi++)
            {
                for (var h = 0; h < numHeads; h++)
                {
                    var scores = new float[dstLen];
                    var maxv = float.NegativeInfinity;
                    for (var j = 0; j < dstLen; j++)
                    {
                        var s = 0f;
                        var qBase = qi * embedDim + h * headDim;
                        var kBase = j * embedDim + h * headDim;
                        for (var d = 0; d < headDim; d++)
                            s += q[qBase + d] * k[kBase + d];
                        s *= scale;
                        scores[j] = s;
                        if (s > maxv) maxv = s;
                    }
                    var sum = 0f;
                    for (var j = 0; j < dstLen; j++)
                    {
                        var e = Mathf.Exp(scores[j] - maxv);
                        scores[j] = e;
                        sum += e;
                    }
                    var invSum = 1f / Mathf.Max(sum, 1e-20f);

                    for (var d = 0; d < headDim; d++)
                    {
                        var acc = 0f;
                        for (var j = 0; j < dstLen; j++)
                        {
                            var w = scores[j] * invSum;
                            acc += w * v[j * embedDim + h * headDim + d];
                        }
                        refOut[qi * embedDim + h * headDim + d] = acc;
                    }
                }
            }

            using var bufQ = new ComputeBuffer(q.Length, sizeof(float), ComputeBufferType.Structured);
            using var bufK = new ComputeBuffer(k.Length, sizeof(float), ComputeBufferType.Structured);
            using var bufV = new ComputeBuffer(v.Length, sizeof(float), ComputeBufferType.Structured);
            using var bufOut = new ComputeBuffer(refOut.Length, sizeof(float), ComputeBufferType.Structured);
            bufQ.SetData(q);
            bufK.SetData(k);
            bufV.SetData(v);
            ops.MhaAttention(bufQ, bufK, bufV, srcLen, dstLen, embedDim, numHeads, scale, bufOut);
            var got = new float[refOut.Length];
            bufOut.GetData(got);

            var maxErr = 0f;
            for (var i = 0; i < got.Length; i++)
                maxErr = Mathf.Max(maxErr, Mathf.Abs(got[i] - refOut[i]));
            Debug.Log("[SELFTEST] MhaAttention maxErr=" + maxErr);
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

            var bt = new float[n * k];
            for (var kk = 0; kk < k; kk++)
                for (var j = 0; j < n; j++)
                    bt[j * k + kk] = b[kk * n + j];
            using var bufBt = new ComputeBuffer(bt.Length, sizeof(float), ComputeBufferType.Structured);
            bufBt.SetData(bt);
            ops.MatMul2D(bufA, bufBt, m, n, k, true, bufOut);
            bufOut.GetData(got);
            maxErr = 0f;
            for (var i = 0; i < got.Length; i++)
                maxErr = Mathf.Max(maxErr, Mathf.Abs(got[i] - refOut[i]));
            UnityEngine.Debug.Log("[SELFTEST] MatMul2D transB maxErr=" + maxErr);
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

        private static void SelfTestReduceAll(NcnnOps ops)
        {
            var x = new float[1024];
            var sum = 0f;
            for (var i = 0; i < x.Length; i++)
            {
                x[i] = (i % 17) * 0.1f - 0.8f;
                sum += x[i];
            }
            var mean = sum / x.Length;

            using var bufIn = new ComputeBuffer(x.Length, sizeof(float), ComputeBufferType.Structured);
            using var bufOut = new ComputeBuffer(1, sizeof(float), ComputeBufferType.Structured);
            bufIn.SetData(x);
            ops.ReduceAllSumOrMean(bufIn, x.Length, false, bufOut);
            var gotSum = new float[1];
            bufOut.GetData(gotSum);
            UnityEngine.Debug.Log("[SELFTEST] ReduceAllSum absErr=" + Mathf.Abs(gotSum[0] - sum));

            ops.ReduceAllSumOrMean(bufIn, x.Length, true, bufOut);
            var gotMean = new float[1];
            bufOut.GetData(gotMean);
            UnityEngine.Debug.Log("[SELFTEST] ReduceAllMean absErr=" + Mathf.Abs(gotMean[0] - mean));
        }

        private static void SelfTestGroupNorm(NcnnOps ops)
        {
            const int w = 4;
            const int h = 3;
            const int c = 8;
            const int group = 2;
            const float eps = 0.001f;

            var size = w * h;
            var x = new float[c * size];
            for (var i = 0; i < x.Length; i++) x[i] = (i % 23) * 0.03f - 0.4f;
            var gamma = new float[c];
            var beta = new float[c];
            for (var i = 0; i < c; i++)
            {
                gamma[i] = 0.9f + i * 0.01f;
                beta[i] = -0.05f + i * 0.005f;
            }

            var refY = (float[])x.Clone();
            var channelsG = c / group;
            for (var g = 0; g < group; g++)
            {
                var chBase = g * channelsG;
                var total = channelsG * size;
                var sum = 0f;
                var sqsum = 0f;
                for (var cc = 0; cc < channelsG; cc++)
                {
                    var ch = chBase + cc;
                    for (var s = 0; s < size; s++)
                    {
                        var v = refY[ch * size + s];
                        sum += v;
                        sqsum += v * v;
                    }
                }
                var mean = sum / total;
                var var = sqsum / total - mean * mean;
                var invstd = 1f / Mathf.Sqrt(var + eps);
                for (var cc = 0; cc < channelsG; cc++)
                {
                    var ch = chBase + cc;
                    for (var s = 0; s < size; s++)
                    {
                        var v = (refY[ch * size + s] - mean) * invstd;
                        v = v * gamma[ch] + beta[ch];
                        refY[ch * size + s] = v;
                    }
                }
            }

            using var buf = new ComputeBuffer(x.Length, sizeof(float), ComputeBufferType.Structured);
            using var bufGamma = new ComputeBuffer(gamma.Length, sizeof(float), ComputeBufferType.Structured);
            using var bufBeta = new ComputeBuffer(beta.Length, sizeof(float), ComputeBufferType.Structured);
            buf.SetData(x);
            bufGamma.SetData(gamma);
            bufBeta.SetData(beta);
            ops.GroupNormInplace(buf, w, h, c, group, eps, true, bufGamma, bufBeta);
            var got = new float[x.Length];
            buf.GetData(got);
            var maxErr = 0f;
            for (var i = 0; i < got.Length; i++) maxErr = Mathf.Max(maxErr, Mathf.Abs(got[i] - refY[i]));
            UnityEngine.Debug.Log("[SELFTEST] GroupNorm maxErr=" + maxErr);
        }

        private static void SelfTestReshapeExpandDims()
        {
            using var t = new NcnnTensorBuffer(2, 3, 4);
            var v2 = t.Reshape(2, 6, 4);
            UnityEngine.Debug.Log("[SELFTEST] Reshape dims2 ok=" + (v2.elementCount == t.elementCount && v2.dims == 2));
            var v4 = t.ExpandDims(2);
            UnityEngine.Debug.Log("[SELFTEST] ExpandDims ok=" + (v4.elementCount == t.elementCount && v4.dims == 4));
        }

        private static void SelfTestTile(NcnnOps ops)
        {
            const int dims = 1;
            const int inW = 4;
            const int tiles = 3;
            var x = new float[inW];
            for (var i = 0; i < x.Length; i++) x[i] = i * 0.25f - 0.3f;
            var outW = inW * tiles;
            var refY = new float[outW];
            for (var i = 0; i < outW; i++) refY[i] = x[i % inW];

            using var bufIn = new ComputeBuffer(inW, sizeof(float), ComputeBufferType.Structured);
            using var bufOut = new ComputeBuffer(outW, sizeof(float), ComputeBufferType.Structured);
            bufIn.SetData(x);
            ops.Tile(bufIn, dims, inW, 1, 1, 1, 0, tiles, outW, 1, 1, 1, bufOut);
            var got = new float[outW];
            bufOut.GetData(got);
            var maxErr = 0f;
            for (var i = 0; i < outW; i++) maxErr = Mathf.Max(maxErr, Mathf.Abs(got[i] - refY[i]));
            UnityEngine.Debug.Log("[SELFTEST] Tile dims1 maxErr=" + maxErr);
        }
    }
}
