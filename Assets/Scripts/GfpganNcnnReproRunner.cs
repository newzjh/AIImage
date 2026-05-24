using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Threading;
using Cysharp.Threading.Tasks;
using NcnnCompute;
using UnityEngine;
using UnityEngine.Rendering;

public sealed class GfpganNcnnReproRunner : MonoBehaviour
{
    public bool enableGfpganRepro = true;
    public string paramRelativePath = "GFPGAN/models/encoder.param";
    public string binRelativePath = "GFPGAN/models/encoder.bin";
    public int maxInputLongSide = 2048;
    public bool enableTempPool = false;
    public bool clearTempPoolAfterRun = true;
    public int maxPooledPerShape = 2;

    public event Action<float, string> ProgressChanged;

    private sealed class ConvPack : IDisposable
    {
        public int outC;
        public int inC;
        public int kernel;
        public int pad;
        public int weightSize;
        public int activationType;
        public float activationSlope;
        public int inPacks;
        public int outPacks;
        public ComputeBuffer w4;
        public ComputeBuffer b4;

        public void Dispose()
        {
            try { w4?.Dispose(); } catch { }
            try { b4?.Dispose(); } catch { }
        }
    }

    private sealed class TensorRef
    {
        public RenderTexture t;
        public int w;
        public int h;
        public int packs;
        public int refs;
        public bool owned;
    }

    private sealed class IpPack : IDisposable
    {
        public int inFeatures;
        public int outFeatures;
        public int weightSize;
        public int biasTerm;
        public ComputeBuffer w;
        public ComputeBuffer b;

        public void Dispose()
        {
            try { w?.Dispose(); } catch { }
            try { b?.Dispose(); } catch { }
        }
    }

    private NcnnParamModel _model;
    private readonly Dictionary<string, ConvPack> _conv = new Dictionary<string, ConvPack>(StringComparer.Ordinal);
    private readonly Dictionary<string, IpPack> _ip = new Dictionary<string, IpPack>(StringComparer.Ordinal);
    private Dictionary<string, int> _blobUseCount;
    private NcnnOps _ops;
    private bool _loaded;
    private readonly Dictionary<RtKey, Stack<PooledRt>> _rtPool = new Dictionary<RtKey, Stack<PooledRt>>();
    private bool _useTempPoolThisRun;

    private readonly struct RtKey : IEquatable<RtKey>
    {
        public readonly int w;
        public readonly int h;
        public readonly int d;
        public readonly RenderTextureFormat format;

        public RtKey(int w, int h, int d, RenderTextureFormat format)
        {
            this.w = w;
            this.h = h;
            this.d = d;
            this.format = format;
        }

        public bool Equals(RtKey other) => w == other.w && h == other.h && d == other.d && format == other.format;
        public override bool Equals(object obj) => obj is RtKey other && Equals(other);
        public override int GetHashCode()
        {
            unchecked
            {
                var hash = w;
                hash = (hash * 397) ^ h;
                hash = (hash * 397) ^ d;
                hash = (hash * 397) ^ (int)format;
                return hash;
            }
        }
    }

    private readonly struct PooledRt
    {
        public readonly RenderTexture rt;
        public readonly GraphicsFence fence;

        public PooledRt(RenderTexture rt, GraphicsFence fence)
        {
            this.rt = rt;
            this.fence = fence;
        }
    }

    private void Awake()
    {
        _ops = new NcnnOps();
    }

    private void OnDestroy()
    {
        foreach (var kv in _conv)
            kv.Value?.Dispose();
        _conv.Clear();
        foreach (var kv in _ip)
            kv.Value?.Dispose();
        _ip.Clear();
        ClearTempPool();
        _blobUseCount = null;
        _model = null;
        _loaded = false;
    }

    public async UniTask<GfpganResult> ProcessAsync(Texture2D src, CancellationToken ct)
    {
        if (!enableGfpganRepro)
            return new GfpganResult { error = "GFPGAN(复刻) disabled" };
        if (src == null)
            return default;

        var totalSw = Stopwatch.StartNew();
        var originalW0 = src.width;
        var originalH0 = src.height;
        var isVulkan = SystemInfo.graphicsDeviceType == GraphicsDeviceType.Vulkan;
        _useTempPoolThisRun = enableTempPool || isVulkan;

        GfpganResult Finish(GfpganResult r)
        {
            r.elapsedMs = totalSw.ElapsedMilliseconds;
            try
            {
                UnityEngine.Debug.Log("[TIMING] GFPGAN(repro) " + r.elapsedMs + " ms | in=" + originalW0 + "x" + originalH0 + " | err=" + (r.error ?? ""));
            }
            catch
            {
            }
            return r;
        }

        EnsureLoaded();
        if (_model == null)
            return Finish(new GfpganResult { error = "GFPGAN(复刻) 模型不可用" });

        var originalW = src.width;
        var originalH = src.height;
        var maxSide = Mathf.Max(originalW, originalH);
        var limit = Mathf.Max(256, maxInputLongSide);
        var runInW = originalW;
        var runInH = originalH;
        if (maxSide > limit)
        {
            var s = (float)limit / maxSide;
            runInW = Mathf.Max(1, Mathf.RoundToInt(originalW * s));
            runInH = Mathf.Max(1, Mathf.RoundToInt(originalH * s));
        }

        ReportProgress(0f, "准备输入");
        await UniTask.Yield();

        Texture2D scaled = null;
        Texture2D face = null;
        RenderTexture inArr = null;
        RenderTexture outArr = null;
        try
        {
            var inputTex = src;
            if (runInW != originalW || runInH != originalH)
            {
                scaled = ResizeTextureBilinear(src, runInW, runInH);
                if (scaled == null)
                    return Finish(new GfpganResult { error = "缩放输入失败" });
                inputTex = scaled;
            }

            var side = Mathf.Min(inputTex.width, inputTex.height);
            var crop = new RectInt((inputTex.width - side) / 2, (inputTex.height - side) / 2, side, side);
            var cropped = CropTexture(inputTex, crop);
            if (cropped == null)
                return Finish(new GfpganResult { error = "裁剪失败" });
            face = ResizeTextureBilinear(cropped, 512, 512);
            Destroy(cropped);
            if (face == null)
                return Finish(new GfpganResult { error = "resize到512失败" });

            inArr = RentTempArray(512, 512, 1, RenderTextureFormat.ARGBHalf);
            _ops.PackRgbToPack4(face, 0, 0, inArr);

            ReportProgress(0.05f, "推理中…");
            try
            {
                outArr = ForwardPack4(inArr, 1);
            }
            catch (Exception e)
            {
                return Finish(new GfpganResult { error = e.Message });
            }

            ReportProgress(1f, "完成");
            return Finish(new GfpganResult { error = "GFPGAN(复刻) 已跑通到当前支持的层类型。下一步补齐 Reshape/InnerProduct 等算子后即可继续推进 encoder 全链路。", texture = null });
        }
        catch (Exception e)
        {
            if (IsLikelyVulkanOom(e))
                return Finish(new GfpganResult { error = "Vulkan - Out of device memory" });
            return Finish(new GfpganResult { error = e.Message });
        }
        finally
        {
            if (scaled != null) Destroy(scaled);
            if (face != null) Destroy(face);
            if (inArr != null) ReturnTempArray(inArr);
            if (outArr != null) ReturnTempArray(outArr);
            if (_useTempPoolThisRun && clearTempPoolAfterRun)
                ClearTempPool();
            _useTempPoolThisRun = false;
        }
    }

    private RenderTexture ForwardPack4(RenderTexture inputPack4, int inputPacks)
    {
        var remaining = new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);
        var blobs = new Dictionary<string, TensorRef>(StringComparer.Ordinal);
        var ownedBuffers = new List<ComputeBuffer>();
        var bufferBlobs = new Dictionary<string, ComputeBuffer>(StringComparer.Ordinal);

        var inputName = _model.layers.Count > 0 && _model.layers[0].topNames != null && _model.layers[0].topNames.Length > 0
            ? _model.layers[0].topNames[0]
            : "input.1";
        blobs[inputName] = new TensorRef { t = inputPack4, w = inputPack4.width, h = inputPack4.height, packs = inputPacks, refs = 1, owned = false };

        try
        {
            for (var li = 0; li < _model.layers.Count; li++)
            {
                var l = _model.layers[li];
                if (string.Equals(l.type, "Input", StringComparison.Ordinal))
                    continue;

                if (string.Equals(l.type, "Split", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    for (var i = 0; i < l.topNames.Length; i++)
                    {
                        blobs[l.topNames[i]] = src;
                        src.refs++;
                    }
                    Consume(blobs, remaining, l.bottomNames);
                    continue;
                }

                if (string.Equals(l.type, "Reshape", StringComparison.Ordinal))
                {
                    if (blobs.TryGetValue(l.bottomNames[0], out var srcTex) && srcTex != null)
                    {
                        blobs[l.topNames[0]] = srcTex;
                        srcTex.refs++;
                        Consume(blobs, remaining, l.bottomNames);
                        continue;
                    }
                    if (bufferBlobs.TryGetValue(l.bottomNames[0], out var srcBuf) && srcBuf != null)
                    {
                        bufferBlobs[l.topNames[0]] = srcBuf;
                        Consume(blobs, remaining, l.bottomNames);
                        continue;
                    }
                    throw new InvalidOperationException("blob not found for Reshape " + l.name);
                }

                if (string.Equals(l.type, "InnerProduct", StringComparison.Ordinal))
                {
                    var ip = _ip[l.name];
                    if (!blobs.TryGetValue(l.bottomNames[0], out var src) || src == null || src.t == null)
                        throw new InvalidOperationException("InnerProduct expects texture blob: " + l.bottomNames[0]);
                    if (src.w * src.h * src.packs * 4 != ip.inFeatures)
                        throw new InvalidOperationException("InnerProduct inFeatures mismatch for " + l.name + ": got " + (src.w * src.h * src.packs * 4) + " expected " + ip.inFeatures);

                    var inBuf = new ComputeBuffer(ip.inFeatures, sizeof(float), ComputeBufferType.Structured);
                    ownedBuffers.Add(inBuf);
                    _ops.Pack4ToBufferCHW(src.t, src.w, src.h, src.packs * 4, inBuf);

                    var outBuf = new ComputeBuffer(ip.outFeatures, sizeof(float), ComputeBufferType.Structured);
                    ownedBuffers.Add(outBuf);
                    _ops.InnerProduct(inBuf, ip.inFeatures, ip.w, ip.b, ip.outFeatures, outBuf);
                    bufferBlobs[l.topNames[0]] = outBuf;
                    Consume(blobs, remaining, l.bottomNames);
                    continue;
                }

                if (string.Equals(l.type, "Convolution", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var pack = _conv[l.name];
                    if (src.packs != pack.inPacks)
                        throw new InvalidOperationException("unexpected in packs for " + l.name + ": " + src.packs + " vs " + pack.inPacks);

                    var outArr = RentTempArray(src.w, src.h, pack.outPacks, RenderTextureFormat.ARGBHalf);
                    if (pack.kernel == 1)
                        _ops.Conv1x1Pack4(src.t, pack.inPacks, pack.w4, pack.b4, pack.outPacks, pack.activationType, pack.activationSlope, outArr);
                    else if (pack.kernel == 3)
                        _ops.Conv3x3Pack4(src.t, pack.inPacks, pack.w4, pack.b4, pack.outPacks, pack.pad, pack.activationType, pack.activationSlope, outArr);
                    else
                        throw new InvalidOperationException("unsupported kernel size: " + pack.kernel);

                    blobs[l.topNames[0]] = new TensorRef { t = outArr, w = src.w, h = src.h, packs = pack.outPacks, refs = 1, owned = true };
                    Consume(blobs, remaining, l.bottomNames);
                    continue;
                }

                if (string.Equals(l.type, "BinaryOp", StringComparison.Ordinal))
                {
                    var a = Get(blobs, l.bottomNames[0]);
                    var b = Get(blobs, l.bottomNames[1]);
                    if (a.w != b.w || a.h != b.h || a.packs != b.packs)
                        throw new InvalidOperationException("shape mismatch for BinaryOp " + l.name);
                    var outArr = RentTempArray(a.w, a.h, a.packs, RenderTextureFormat.ARGBHalf);
                    _ops.AddPack4(a.t, b.t, 1f, 1f, a.packs, outArr);
                    blobs[l.topNames[0]] = new TensorRef { t = outArr, w = a.w, h = a.h, packs = a.packs, refs = 1, owned = true };
                    Consume(blobs, remaining, l.bottomNames);
                    continue;
                }

                if (string.Equals(l.type, "Interp", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var resizeType = l.GetInt(0, 2);
                    var sx = l.GetFloat(1, 1f);
                    var sy = l.GetFloat(2, 1f);
                    if (Mathf.Abs(sx - sy) > 1e-3f)
                        throw new InvalidOperationException("unsupported interp scale: " + sx.ToString("0.###", CultureInfo.InvariantCulture) + "," + sy.ToString("0.###", CultureInfo.InvariantCulture));

                    if (Mathf.Abs(sx - 2f) < 1e-3f)
                    {
                        var outArr = RentTempArray(src.w * 2, src.h * 2, src.packs, RenderTextureFormat.ARGBHalf);
                        if (resizeType == 1)
                            _ops.Interp2xNearestPack4(src.t, src.packs, outArr);
                        else
                            _ops.Interp2xPack4(src.t, src.packs, outArr);
                        blobs[l.topNames[0]] = new TensorRef { t = outArr, w = src.w * 2, h = src.h * 2, packs = src.packs, refs = 1, owned = true };
                        Consume(blobs, remaining, l.bottomNames);
                        continue;
                    }
                    if (Mathf.Abs(sx - 0.5f) < 1e-3f)
                    {
                        var outArr = RentTempArray(src.w / 2, src.h / 2, src.packs, RenderTextureFormat.ARGBHalf);
                        if (resizeType == 1)
                            _ops.InterpDown2NearestPack4(src.t, src.packs, outArr);
                        else
                            _ops.InterpDown2Pack4(src.t, src.packs, outArr);
                        blobs[l.topNames[0]] = new TensorRef { t = outArr, w = src.w / 2, h = src.h / 2, packs = src.packs, refs = 1, owned = true };
                        Consume(blobs, remaining, l.bottomNames);
                        continue;
                    }

                    throw new InvalidOperationException("unsupported interp scale: " + sx.ToString("0.###", CultureInfo.InvariantCulture));
                }

                throw new InvalidOperationException("unsupported layer type: " + l.type + " (" + l.name + ")");
            }
        }
        finally
        {
            var visited = new HashSet<TensorRef>();
            foreach (var kv in blobs)
            {
                var tr = kv.Value;
                if (tr == null || !visited.Add(tr))
                    continue;
                if (tr.owned && tr.t != null)
                    ReturnTempArray(tr.t);
            }

            for (var i = 0; i < ownedBuffers.Count; i++)
            {
                try { ownedBuffers[i].Dispose(); } catch { }
            }
        }

        return null;
    }

    private void EnsureLoaded()
    {
        if (_loaded)
            return;

        var paramPath = Path.Combine(Application.streamingAssetsPath, paramRelativePath);
        var binPath = Path.Combine(Application.streamingAssetsPath, binRelativePath);
        if (!File.Exists(paramPath))
            throw new InvalidOperationException("GFPGAN(复刻) param 不存在: " + paramPath);
        if (!File.Exists(binPath))
            throw new InvalidOperationException("GFPGAN(复刻) bin 不存在: " + binPath);

        var paramText = File.ReadAllText(paramPath);
        _model = NcnnParamParser.Parse(paramText);
        _blobUseCount = BuildBlobUseCount(_model);

        using (var fs = File.OpenRead(binPath))
        using (var br = new NcnnBinReader(fs))
        {
            foreach (var layer in _model.layers)
            {
                if (string.Equals(layer.type, "Convolution", StringComparison.Ordinal))
                {
                    var pack = new ConvPack();
                    pack.outC = layer.GetInt(0, 0);
                    pack.kernel = layer.GetInt(1, 3);
                    pack.pad = layer.GetInt(4, 0);
                    var biasTerm = layer.GetInt(5, 0);
                    pack.weightSize = layer.GetInt(6, 0);
                    pack.activationType = layer.GetInt(9, 0);
                    pack.activationSlope = ParseLeakySlope(layer);
                    pack.inC = Mathf.Max(1, pack.weightSize / Mathf.Max(1, pack.outC * pack.kernel * pack.kernel));
                    pack.inPacks = (pack.inC + 3) / 4;
                    pack.outPacks = (pack.outC + 3) / 4;

                    var tag = br.ReadInt32();
                    if (tag != 0x01306B47)
                        throw new InvalidOperationException("unexpected weight tag at " + br.Position + ": 0x" + tag.ToString("X8", CultureInfo.InvariantCulture));

                    var w = br.ReadFp16ArrayAsFloat32(pack.weightSize);
                    var b = biasTerm != 0 ? br.ReadFloat32Array(pack.outC) : new float[pack.outC];
                    var w4 = PackWeightsToO4I4K(w, pack.outC, pack.inC, pack.kernel, pack.outPacks, pack.inPacks);
                    var b4 = PackBiasToO4(b, pack.outC, pack.outPacks);
                    pack.w4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                    pack.b4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                    pack.w4.SetData(w4);
                    pack.b4.SetData(b4);
                    _conv[layer.name] = pack;
                    continue;
                }

                if (string.Equals(layer.type, "InnerProduct", StringComparison.Ordinal))
                {
                    var ip = new IpPack();
                    ip.outFeatures = layer.GetInt(0, 0);
                    ip.biasTerm = layer.GetInt(1, 0);
                    ip.weightSize = layer.GetInt(2, 0);
                    ip.inFeatures = ip.outFeatures > 0 ? (ip.weightSize / ip.outFeatures) : 0;
                    if (ip.outFeatures <= 0 || ip.inFeatures <= 0)
                        throw new InvalidOperationException("invalid InnerProduct shape for " + layer.name);

                    var tag = br.ReadInt32();
                    if (tag != 0x01306B47)
                        throw new InvalidOperationException("unexpected weight tag at " + br.Position + ": 0x" + tag.ToString("X8", CultureInfo.InvariantCulture));

                    var w = br.ReadFp16ArrayAsFloat32(ip.weightSize);
                    var b = ip.biasTerm != 0 ? br.ReadFloat32Array(ip.outFeatures) : new float[ip.outFeatures];
                    ip.w = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                    ip.b = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                    ip.w.SetData(w);
                    ip.b.SetData(b);
                    _ip[layer.name] = ip;
                    continue;
                }
            }
        }

        _loaded = true;
    }

    private void ReportProgress(float progress01, string text)
    {
        progress01 = Mathf.Clamp01(progress01);
        try { ProgressChanged?.Invoke(progress01, text ?? ""); } catch { }
    }

    private static TensorRef Get(Dictionary<string, TensorRef> blobs, string name)
    {
        if (!blobs.TryGetValue(name, out var tr) || tr == null || tr.t == null)
            throw new InvalidOperationException("blob not found: " + name);
        return tr;
    }

    private static void Consume(Dictionary<string, TensorRef> blobs, Dictionary<string, int> remaining, string[] bottomNames)
    {
        for (var i = 0; i < bottomNames.Length; i++)
        {
            var b = bottomNames[i];
            if (!remaining.TryGetValue(b, out var c))
                continue;
            c--;
            remaining[b] = c;
            if (c > 0)
                continue;

            if (blobs.TryGetValue(b, out var tr) && tr != null)
            {
                tr.refs--;
                if (tr.refs <= 0)
                {
                    if (tr.owned && tr.t != null)
                    {
                        var v = GameObject.FindFirstObjectByType<GfpganNcnnReproRunner>();
                        try { v.ReturnTempArray(tr.t); } catch { }
                    }
                    tr.t = null;
                    tr.owned = false;
                }
            }
            blobs.Remove(b);
        }
    }

    private static Dictionary<string, int> BuildBlobUseCount(NcnnParamModel model)
    {
        var use = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var i = 0; i < model.layers.Count; i++)
        {
            var l = model.layers[i];
            if (l.bottomNames == null)
                continue;
            for (var b = 0; b < l.bottomNames.Length; b++)
            {
                var n = l.bottomNames[b];
                if (string.IsNullOrEmpty(n))
                    continue;
                use.TryGetValue(n, out var c);
                use[n] = c + 1;
            }
        }
        return use;
    }

    private static float ParseLeakySlope(NcnnParamModel.Layer layer)
    {
        if (layer.intParams == null || !layer.intParams.TryGetValue(-23310, out var s) || string.IsNullOrWhiteSpace(s))
            return 0.2f;
        var parts = s.Split(',');
        if (parts.Length >= 2 && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
            return v;
        return 0.2f;
    }

    private static Vector4[] PackBiasToO4(float[] b, int outC, int outPacks)
    {
        var r = new Vector4[outPacks];
        for (var op = 0; op < outPacks; op++)
        {
            var oc0 = op * 4 + 0;
            var oc1 = op * 4 + 1;
            var oc2 = op * 4 + 2;
            var oc3 = op * 4 + 3;
            r[op] = new Vector4(
                oc0 < outC ? b[oc0] : 0f,
                oc1 < outC ? b[oc1] : 0f,
                oc2 < outC ? b[oc2] : 0f,
                oc3 < outC ? b[oc3] : 0f);
        }
        return r;
    }

    private static Vector4[] PackWeightsToO4I4K(float[] w, int outC, int inC, int k, int outPacks, int inPacks)
    {
        var r = new Vector4[outPacks * inPacks * k * k * 4];
        var idx = 0;
        for (var op = 0; op < outPacks; op++)
        {
            for (var ip = 0; ip < inPacks; ip++)
            {
                for (var ky = 0; ky < k; ky++)
                {
                    for (var kx = 0; kx < k; kx++)
                    {
                        for (var ol = 0; ol < 4; ol++)
                        {
                            var oc = op * 4 + ol;
                            var il0 = ip * 4 + 0;
                            var il1 = ip * 4 + 1;
                            var il2 = ip * 4 + 2;
                            var il3 = ip * 4 + 3;
                            var kk = ky * k + kx;

                            float GetW(int ic)
                            {
                                if (oc >= outC || ic >= inC)
                                    return 0f;
                                return w[(oc * inC + ic) * (k * k) + kk];
                            }

                            r[idx++] = new Vector4(GetW(il0), GetW(il1), GetW(il2), GetW(il3));
                        }
                    }
                }
            }
        }
        return r;
    }

    private RenderTexture RentTempArray(int w, int h, int depth, RenderTextureFormat format)
    {
        if (!_useTempPoolThisRun)
            return CreateTempArray(w, h, depth, format);

        var key = new RtKey(w, h, Mathf.Max(1, depth), format);
        if (_rtPool.TryGetValue(key, out var stack) && stack.Count > 0)
        {
            var keep = new Stack<PooledRt>(stack.Count);
            RenderTexture hit = null;
            while (stack.Count > 0)
            {
                var p = stack.Pop();
                if (hit == null && p.rt != null && p.fence.passed)
                {
                    hit = p.rt;
                    break;
                }
                keep.Push(p);
            }
            while (keep.Count > 0)
                stack.Push(keep.Pop());
            if (hit != null)
                return hit;
        }
        return CreateTempArray(w, h, depth, format);
    }

    private static RenderTexture CreateTempArray(int w, int h, int depth, RenderTextureFormat format)
    {
        var desc = new RenderTextureDescriptor(w, h, format, 0)
        {
            dimension = TextureDimension.Tex2DArray,
            volumeDepth = Mathf.Max(1, depth),
            msaaSamples = 1,
            sRGB = false,
            enableRandomWrite = true
        };
        var rt = RenderTexture.GetTemporary(desc);
        rt.wrapMode = TextureWrapMode.Clamp;
        rt.filterMode = FilterMode.Point;
        rt.Create();
        if (!rt.IsCreated())
            throw new InvalidOperationException("failed to create temp array " + w + "x" + h + "x" + depth + " " + format);
        return rt;
    }

    private void ReturnTempArray(RenderTexture rt)
    {
        if (rt == null)
            return;
        if (!_useTempPoolThisRun)
        {
            RenderTexture.ReleaseTemporary(rt);
            return;
        }

        var key = new RtKey(rt.width, rt.height, rt.volumeDepth, rt.format);
        if (!_rtPool.TryGetValue(key, out var stack))
        {
            stack = new Stack<PooledRt>();
            _rtPool[key] = stack;
        }
        var cap = Mathf.Max(0, maxPooledPerShape);
        if (stack.Count >= cap)
        {
            RenderTexture.ReleaseTemporary(rt);
            return;
        }
        try
        {
            var fence = Graphics.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation, SynchronisationStageFlags.ComputeProcessing);
            stack.Push(new PooledRt(rt, fence));
        }
        catch
        {
            RenderTexture.ReleaseTemporary(rt);
        }
    }

    private void ClearTempPool()
    {
        foreach (var kv in _rtPool)
        {
            var stack = kv.Value;
            while (stack.Count > 0)
            {
                var rt = stack.Pop().rt;
                try { RenderTexture.ReleaseTemporary(rt); } catch { }
            }
        }
        _rtPool.Clear();
    }

    private static bool IsLikelyVulkanOom(Exception e)
    {
        if (e == null) return false;
        var msg = e.Message ?? "";
        if (msg.IndexOf("Out of device memory", StringComparison.OrdinalIgnoreCase) >= 0) return true;
        if (msg.IndexOf("out of memory", StringComparison.OrdinalIgnoreCase) >= 0) return true;
        if (msg.IndexOf("failed to create", StringComparison.OrdinalIgnoreCase) >= 0) return true;
        return false;
    }

    private static RectInt ClampRect(RectInt r, int w, int h)
    {
        var x0 = Mathf.Clamp(r.x, 0, Mathf.Max(0, w));
        var y0 = Mathf.Clamp(r.y, 0, Mathf.Max(0, h));
        var x1 = Mathf.Clamp(r.x + r.width, 0, Mathf.Max(0, w));
        var y1 = Mathf.Clamp(r.y + r.height, 0, Mathf.Max(0, h));
        return new RectInt(x0, y0, Mathf.Max(0, x1 - x0), Mathf.Max(0, y1 - y0));
    }

    private static Texture2D CropTexture(Texture2D src, RectInt rect)
    {
        rect = ClampRect(rect, src.width, src.height);
        if (rect.width <= 0 || rect.height <= 0)
            return null;
        var dst = new Texture2D(rect.width, rect.height, TextureFormat.RGBA32, false, true);
        var srcPixels = src.GetPixels32();
        var dstPixels = new Color32[rect.width * rect.height];
        var sw = src.width;
        for (var y = 0; y < rect.height; y++)
        {
            var srcRow = (rect.y + y) * sw + rect.x;
            var dstRow = y * rect.width;
            Array.Copy(srcPixels, srcRow, dstPixels, dstRow, rect.width);
        }
        dst.SetPixels32(dstPixels);
        dst.Apply(false, false);
        dst.wrapMode = TextureWrapMode.Clamp;
        dst.filterMode = FilterMode.Bilinear;
        return dst;
    }

    private static Texture2D ResizeTextureBilinear(Texture2D src, int w, int h)
    {
        if (src == null)
            return null;
        if (w <= 0 || h <= 0)
            return null;

        var dst = new Texture2D(w, h, TextureFormat.RGBA32, false, true);
        dst.wrapMode = TextureWrapMode.Clamp;
        dst.filterMode = FilterMode.Bilinear;

        var srcPixels = src.GetPixels32();
        var dstPixels = new Color32[w * h];
        var sw = src.width;
        var sh = src.height;
        var invW = sw > 1 ? 1f / (sw - 1f) : 0f;
        var invH = sh > 1 ? 1f / (sh - 1f) : 0f;

        for (var y = 0; y < h; y++)
        {
            var v = h > 1 ? y / (h - 1f) : 0f;
            var sy = v / Mathf.Max(1e-6f, invH);
            var y0 = Mathf.Clamp((int)sy, 0, sh - 1);
            var y1 = Mathf.Clamp(y0 + 1, 0, sh - 1);
            var ty = sy - y0;
            for (var x = 0; x < w; x++)
            {
                var u = w > 1 ? x / (w - 1f) : 0f;
                var sx = u / Mathf.Max(1e-6f, invW);
                var x0 = Mathf.Clamp((int)sx, 0, sw - 1);
                var x1 = Mathf.Clamp(x0 + 1, 0, sw - 1);
                var tx = sx - x0;

                var c00 = srcPixels[y0 * sw + x0];
                var c10 = srcPixels[y0 * sw + x1];
                var c01 = srcPixels[y1 * sw + x0];
                var c11 = srcPixels[y1 * sw + x1];

                var r0 = Mathf.Lerp(c00.r, c10.r, tx);
                var g0 = Mathf.Lerp(c00.g, c10.g, tx);
                var b0 = Mathf.Lerp(c00.b, c10.b, tx);
                var a0 = Mathf.Lerp(c00.a, c10.a, tx);

                var r1 = Mathf.Lerp(c01.r, c11.r, tx);
                var g1 = Mathf.Lerp(c01.g, c11.g, tx);
                var b1 = Mathf.Lerp(c01.b, c11.b, tx);
                var a1 = Mathf.Lerp(c01.a, c11.a, tx);

                var r2 = Mathf.Lerp(r0, r1, ty);
                var g2 = Mathf.Lerp(g0, g1, ty);
                var b2 = Mathf.Lerp(b0, b1, ty);
                var a2 = Mathf.Lerp(a0, a1, ty);

                dstPixels[y * w + x] = new Color32((byte)r2, (byte)g2, (byte)b2, (byte)a2);
            }
        }

        dst.SetPixels32(dstPixels);
        dst.Apply(false, false);
        return dst;
    }
}
