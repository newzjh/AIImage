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
    public string styleRelativePath = "GFPGAN/models/style.bin";
    public int maxInputLongSide = 2048;
    public float faceMaskThreshold = 0.2f;
    public float faceBoxExpand = 0.35f;
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

    private sealed class StyleConvWeights : IDisposable
    {
        public int inc;
        public int hidDim;
        public int numOutput;
        public float[] selfWeight;
        public float[] modulationW;
        public float[] modulationB;
        public float noiseWeight;
        public float[] bias;
        public ComputeBuffer bias4;

        public void Dispose()
        {
            try { bias4?.Dispose(); } catch { }
        }
    }

    private sealed class ToRgbWeights : IDisposable
    {
        public int inc;
        public int hidDim;
        public int numOutput;
        public float[] selfWeight;
        public float[] modulationW;
        public float[] modulationB;
        public float[] bias;
        public ComputeBuffer bias4;

        public void Dispose()
        {
            try { bias4?.Dispose(); } catch { }
        }
    }

    private NcnnParamModel _model;
    private readonly Dictionary<string, ConvPack> _conv = new Dictionary<string, ConvPack>(StringComparer.Ordinal);
    private readonly Dictionary<string, IpPack> _ip = new Dictionary<string, IpPack>(StringComparer.Ordinal);
    private StyleConvWeights[] _styleConv;
    private ToRgbWeights[] _toRgb;
    private float[] _constInput;
    private ComputeBuffer _constInputBuf;
    private readonly Dictionary<int, ComputeBuffer> _zeroBias4 = new Dictionary<int, ComputeBuffer>();
    private ComputeBuffer _dynW4;
    private int _dynW4Count;
    private Vector4[] _dynW4Host;
    private float[] _demodTmp;
    private float[] _styleOutTmp;
    private readonly Dictionary<int, ComputeBuffer> _noiseBuf = new Dictionary<int, ComputeBuffer>();
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
        public readonly int frame;

        public PooledRt(RenderTexture rt, GraphicsFence fence, int frame)
        {
            this.rt = rt;
            this.fence = fence;
            this.frame = frame;
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
        if (_styleConv != null)
        {
            for (var i = 0; i < _styleConv.Length; i++)
                _styleConv[i]?.Dispose();
        }
        if (_toRgb != null)
        {
            for (var i = 0; i < _toRgb.Length; i++)
                _toRgb[i]?.Dispose();
        }
        _styleConv = null;
        _toRgb = null;
        try { _constInputBuf?.Dispose(); } catch { }
        _constInputBuf = null;
        _constInput = null;
        foreach (var kv in _zeroBias4)
        {
            try { kv.Value?.Dispose(); } catch { }
        }
        _zeroBias4.Clear();
        try { _dynW4?.Dispose(); } catch { }
        _dynW4 = null;
        _dynW4Count = 0;
        foreach (var kv in _noiseBuf)
        {
            try { kv.Value?.Dispose(); } catch { }
        }
        _noiseBuf.Clear();
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

        ReportProgress(0f, "准备输入");
        await UniTask.Yield();

        Texture2D scaled = null;
        Texture2D faceMask = null;
        Texture2D faceCrop = null;
        Texture2D face512 = null;
        Texture2D restored512 = null;
        Texture2D restoredCrop = null;
        RenderTexture composed = null;
        try
        {
            var inputTex = src;
            float scaleDown = 1f;
            if (maxSide > limit)
            {
                ReportProgress(0.02f, "缩小到2k以内");
                scaleDown = (float)limit / maxSide;
                var sw = Mathf.Max(1, Mathf.RoundToInt(originalW * scaleDown));
                var sh = Mathf.Max(1, Mathf.RoundToInt(originalH * scaleDown));
                scaled = ResizeTextureBilinear(src, sw, sh);
                if (scaled == null)
                    return Finish(new GfpganResult { error = "缩放输入失败" });
                inputTex = scaled;
            }

            ReportProgress(0.06f, "生成脸部区域");
            var fm = GetComponent<FaceMaskGenerator>();
            if (fm != null)
            {
                var mr = await fm.GenerateForCurrentAsync(inputTex, false, ct);
                if (string.IsNullOrWhiteSpace(mr.error) && mr.mask != null)
                    faceMask = mr.mask;
            }

            var rect = FindFaceRect(faceMask, inputTex.width, inputTex.height, faceMaskThreshold);
            rect = ExpandRect(rect, inputTex.width, inputTex.height, faceBoxExpand);
            if (rect.width <= 8 || rect.height <= 8)
                rect = new RectInt(inputTex.width / 4, inputTex.height / 4, inputTex.width / 2, inputTex.height / 2);

            ReportProgress(0.10f, "裁剪脸部");
            faceCrop = CropTexture(inputTex, rect);
            if (faceCrop == null)
                return Finish(new GfpganResult { error = "裁剪失败" });

            face512 = ResizeTextureBilinear(faceCrop, 512, 512);
            if (face512 == null)
                return Finish(new GfpganResult { error = "resize到512失败" });

            ReportProgress(0.15f, "推理中…");
            restored512 = await RunGfpgan512Async(face512, ct);
            if (restored512 == null)
                return Finish(new GfpganResult { error = "GFPGAN(复刻) 推理失败" });

            ReportProgress(0.85f, "回贴到原图");
            restoredCrop = ResizeTextureBilinear(restored512, rect.width, rect.height);
            if (restoredCrop == null)
                return Finish(new GfpganResult { error = "回贴缩放失败" });

            composed = ComposeWithMask(inputTex, restoredCrop, faceMask, rect);
            if (composed == null)
                return Finish(new GfpganResult { error = "合成失败" });

            var composedTex = RenderTextureToTexture2D(composed, inputTex.width, inputTex.height);
            if (composedTex == null)
                return Finish(new GfpganResult { error = "合成回读失败" });

            Texture2D finalTex = composedTex;
            if (Mathf.Abs(scaleDown - 1f) > 1e-6f)
            {
                ReportProgress(0.95f, "回缩放到原分辨率");
                var resized = ResizeTextureBilinear(finalTex, originalW, originalH);
                Destroy(finalTex);
                finalTex = resized;
                if (finalTex == null)
                    return Finish(new GfpganResult { error = "回缩放失败" });
            }

            finalTex.wrapMode = TextureWrapMode.Clamp;
            finalTex.filterMode = FilterMode.Bilinear;
            finalTex.name = "GFPGAN_Repro";
            ReportProgress(1f, "完成");
            return Finish(new GfpganResult { texture = finalTex });
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
            if (faceMask != null) Destroy(faceMask);
            if (faceCrop != null) Destroy(faceCrop);
            if (face512 != null) Destroy(face512);
            if (restored512 != null) Destroy(restored512);
            if (restoredCrop != null) Destroy(restoredCrop);
            if (composed != null) { composed.Release(); Destroy(composed); }
            if (_useTempPoolThisRun && clearTempPoolAfterRun)
                ClearTempPool();
            _useTempPoolThisRun = false;
        }
    }

    private async UniTask<Texture2D> RunGfpgan512Async(Texture2D face512, CancellationToken ct)
    {
        if (face512 == null || face512.width != 512 || face512.height != 512)
            return null;

        RenderTexture inArr = null;
        RenderTexture[] cond = null;
        RenderTexture constIn = null;
        RenderTexture outFeat = null;
        RenderTexture skip = null;
        RenderTexture tmp = null;
        RenderTexture skipClip = null;
        RenderTexture rgb = null;
        Texture2D outTex = null;

        try
        {
            await UniTask.Yield();

            inArr = RentTempArray(512, 512, 1, RenderTextureFormat.ARGBHalf);
            _ops.PackRgbToPack4Gfpgan(face512, 0, 0, inArr);

            var styles = RunEncoderForGfpgan(inArr, out cond);
            if (styles == null || styles.Length < 512)
                return null;
            if (cond == null || cond.Length != 14)
                return null;

            constIn = RentTempArray(4, 4, 128, RenderTextureFormat.ARGBHalf);
            _ops.FillPack4FromBufferCHW(_constInputBuf, 4, 4, 512, constIn);

            outFeat = RunStyleConv(constIn, styles, 0, _styleConv[14], 0, true);

            skip = RunToRgb(outFeat, styles, 1, _toRgb[7], null);

            var j = 0;
            for (var i = 1; i < 14; i += 2)
            {
                outFeat = RunStyleConv(outFeat, styles, i, _styleConv[i - 1], 1, true);
                tmp = RentTempArray(outFeat.width, outFeat.height, outFeat.volumeDepth, RenderTextureFormat.ARGBHalf);
                _ops.SftPack4(outFeat, cond[i - 1], cond[i], outFeat.volumeDepth, outFeat.volumeDepth / 2, tmp);
                ReturnTempArray(outFeat);
                outFeat = tmp;
                tmp = null;

                outFeat = RunStyleConv(outFeat, styles, i + 1, _styleConv[i], 0, true);

                skip = RunToRgb(outFeat, styles, i + 2, _toRgb[j], skip);
                j++;
            }

            skipClip = RentTempArray(512, 512, 1, RenderTextureFormat.ARGBHalf);
            _ops.ClipPack4(skip, -1f, 1f, 1, skipClip);

            rgb = new RenderTexture(512, 512, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
            rgb.enableRandomWrite = true;
            rgb.Create();
            _ops.Pack4ToRgb01(skipClip, rgb);

            outTex = RenderTextureToTexture2D(rgb, 512, 512);
            if (outTex == null)
                return null;
            outTex.wrapMode = TextureWrapMode.Clamp;
            outTex.filterMode = FilterMode.Bilinear;
            return outTex;
        }
        catch
        {
            if (outTex != null) Destroy(outTex);
            return null;
        }
        finally
        {
            if (inArr != null) ReturnTempArray(inArr);
            if (cond != null)
            {
                for (var i = 0; i < cond.Length; i++)
                    if (cond[i] != null) ReturnTempArray(cond[i]);
            }
            if (constIn != null) ReturnTempArray(constIn);
            if (outFeat != null) ReturnTempArray(outFeat);
            if (skip != null) ReturnTempArray(skip);
            if (tmp != null) ReturnTempArray(tmp);
            if (skipClip != null) ReturnTempArray(skipClip);
            if (rgb != null) { rgb.Release(); Destroy(rgb); }
        }
    }

    private float[] RunEncoderForGfpgan(RenderTexture inputPack4, out RenderTexture[] conditions)
    {
        var condNames = new[]
        {
            "440","443","463","466","486","489","509","512","532","535","555","558","578","581"
        };

        var pinned = new HashSet<string>(condNames, StringComparer.Ordinal);
        var remaining = new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);
        var blobs = new Dictionary<string, TensorRef>(StringComparer.Ordinal);
        var ownedBuffers = new List<ComputeBuffer>();
        var bufferBlobs = new Dictionary<string, ComputeBuffer>(StringComparer.Ordinal);

        var inputName = _model.layers.Count > 0 && _model.layers[0].topNames != null && _model.layers[0].topNames.Length > 0
            ? _model.layers[0].topNames[0]
            : "input.1";
        blobs[inputName] = new TensorRef { t = inputPack4, w = inputPack4.width, h = inputPack4.height, packs = 1, refs = 1, owned = false };

        conditions = null;
        float[] styles = null;

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
                    Consume(blobs, remaining, l.bottomNames, pinned);
                    continue;
                }

                if (string.Equals(l.type, "Reshape", StringComparison.Ordinal))
                {
                    if (blobs.TryGetValue(l.bottomNames[0], out var srcTex) && srcTex != null)
                    {
                        blobs[l.topNames[0]] = srcTex;
                        srcTex.refs++;
                        Consume(blobs, remaining, l.bottomNames, pinned);
                        continue;
                    }
                    if (bufferBlobs.TryGetValue(l.bottomNames[0], out var srcBuf) && srcBuf != null)
                    {
                        bufferBlobs[l.topNames[0]] = srcBuf;
                        Consume(blobs, remaining, l.bottomNames, pinned);
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
                        throw new InvalidOperationException("InnerProduct inFeatures mismatch for " + l.name);

                    var inBuf = new ComputeBuffer(ip.inFeatures, sizeof(float), ComputeBufferType.Structured);
                    ownedBuffers.Add(inBuf);
                    _ops.Pack4ToBufferCHW(src.t, src.w, src.h, src.packs * 4, inBuf);

                    var outBuf = new ComputeBuffer(ip.outFeatures, sizeof(float), ComputeBufferType.Structured);
                    ownedBuffers.Add(outBuf);
                    _ops.InnerProduct(inBuf, ip.inFeatures, ip.w, ip.b, ip.outFeatures, outBuf);
                    bufferBlobs[l.topNames[0]] = outBuf;
                    Consume(blobs, remaining, l.bottomNames, pinned);
                    continue;
                }

                if (string.Equals(l.type, "Convolution", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var pack = _conv[l.name];
                    var outArr = RentTempArray(src.w, src.h, pack.outPacks, RenderTextureFormat.ARGBHalf);
                    if (pack.kernel == 1)
                        _ops.Conv1x1Pack4(src.t, pack.inPacks, pack.w4, pack.b4, pack.outPacks, pack.activationType, pack.activationSlope, outArr);
                    else if (pack.kernel == 3)
                        _ops.Conv3x3Pack4(src.t, pack.inPacks, pack.w4, pack.b4, pack.outPacks, pack.pad, pack.activationType, pack.activationSlope, outArr);
                    else
                        throw new InvalidOperationException("unsupported kernel size: " + pack.kernel);

                    blobs[l.topNames[0]] = new TensorRef { t = outArr, w = src.w, h = src.h, packs = pack.outPacks, refs = 1, owned = true };
                    Consume(blobs, remaining, l.bottomNames, pinned);
                    continue;
                }

                if (string.Equals(l.type, "BinaryOp", StringComparison.Ordinal))
                {
                    var a = Get(blobs, l.bottomNames[0]);
                    var b = Get(blobs, l.bottomNames[1]);
                    var outArr = RentTempArray(a.w, a.h, a.packs, RenderTextureFormat.ARGBHalf);
                    _ops.AddPack4(a.t, b.t, 1f, 1f, a.packs, outArr);
                    blobs[l.topNames[0]] = new TensorRef { t = outArr, w = a.w, h = a.h, packs = a.packs, refs = 1, owned = true };
                    Consume(blobs, remaining, l.bottomNames, pinned);
                    continue;
                }

                if (string.Equals(l.type, "Interp", StringComparison.Ordinal))
                {
                    var src = Get(blobs, l.bottomNames[0]);
                    var resizeType = l.GetInt(0, 2);
                    var sx = l.GetFloat(1, 1f);
                    if (Mathf.Abs(sx - 2f) < 1e-3f)
                    {
                        var outArr = RentTempArray(src.w * 2, src.h * 2, src.packs, RenderTextureFormat.ARGBHalf);
                        if (resizeType == 1)
                            _ops.Interp2xNearestPack4(src.t, src.packs, outArr);
                        else
                            _ops.Interp2xPack4(src.t, src.packs, outArr);
                        blobs[l.topNames[0]] = new TensorRef { t = outArr, w = src.w * 2, h = src.h * 2, packs = src.packs, refs = 1, owned = true };
                        Consume(blobs, remaining, l.bottomNames, pinned);
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
                        Consume(blobs, remaining, l.bottomNames, pinned);
                        continue;
                    }
                    throw new InvalidOperationException("unsupported interp scale");
                }

                throw new InvalidOperationException("unsupported layer type: " + l.type);
            }

            if (!bufferBlobs.TryGetValue("420", out var stylesBuf) || stylesBuf == null)
                return null;

            styles = new float[stylesBuf.count];
            stylesBuf.GetData(styles);

            conditions = new RenderTexture[condNames.Length];
            for (var i = 0; i < condNames.Length; i++)
            {
                var n = condNames[i];
                if (!blobs.TryGetValue(n, out var tr) || tr == null || tr.t == null)
                    throw new InvalidOperationException("condition blob missing: " + n);
                tr.owned = false;
                conditions[i] = tr.t;
            }

            return styles;
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
    }

    private RenderTexture RunStyleConv(RenderTexture x, float[] styles, int styleRow, StyleConvWeights w, int sampleMode, bool demodulate)
    {
        var inArr = x;
        if (sampleMode == 1)
        {
            var up = RentTempArray(inArr.width * 2, inArr.height * 2, inArr.volumeDepth, RenderTextureFormat.ARGBHalf);
            _ops.Interp2xPack4(inArr, inArr.volumeDepth, up);
            if (inArr != x)
                ReturnTempArray(inArr);
            inArr = up;
        }

        var outPacks = (w.numOutput + 3) / 4;
        var outArr = RentTempArray(inArr.width, inArr.height, outPacks, RenderTextureFormat.ARGBHalf);

        var w4Count = outPacks * (w.hidDim / 4) * 9 * 4;
        EnsureDynW4(w4Count);
        EnsureStyleTmp(w.hidDim, w.numOutput);

        ComputeStyleOut(styles, styleRow * 512, w.modulationW, w.modulationB, _styleOutTmp, w.hidDim);
        BuildDynW4_3x3(_dynW4Host, _styleOutTmp, w.selfWeight, w.hidDim, w.numOutput, demodulate ? _demodTmp : null);

        _dynW4.SetData(_dynW4Host, 0, 0, w4Count);
        var zeroB4 = GetZeroBias4(outPacks);
        _ops.Conv3x3Pack4(inArr, w.hidDim / 4, _dynW4, zeroB4, outPacks, 1, 0, 0f, outArr);

        var scaled = RentTempArray(outArr.width, outArr.height, outArr.volumeDepth, RenderTextureFormat.ARGBHalf);
        _ops.ScalePack4(outArr, 1.4142135381698608f, outArr.volumeDepth, scaled);
        ReturnTempArray(outArr);
        outArr = scaled;

        var noiseWh = outArr.width * outArr.height;
        var noise = GetNoiseBuffer(noiseWh);
        FillNoise(noise, noiseWh);
        _ops.AddNoiseBroadcastPack4(outArr, noise, w.noiseWeight, outArr.volumeDepth);

        var biased = RentTempArray(outArr.width, outArr.height, outArr.volumeDepth, RenderTextureFormat.ARGBHalf);
        _ops.AddBiasPack4(outArr, w.bias4, outArr.volumeDepth, biased);
        ReturnTempArray(outArr);
        outArr = biased;

        var act = RentTempArray(outArr.width, outArr.height, outArr.volumeDepth, RenderTextureFormat.ARGBHalf);
        _ops.LeakyReluPack4(outArr, 0.2f, outArr.volumeDepth, act);
        ReturnTempArray(outArr);
        outArr = act;

        if (inArr != x)
            ReturnTempArray(inArr);
        return outArr;
    }

    private RenderTexture RunToRgb(RenderTexture feat, float[] styles, int styleRow, ToRgbWeights w, RenderTexture skip)
    {
        var outArr = RentTempArray(feat.width, feat.height, 1, RenderTextureFormat.ARGBHalf);

        var w4Count = 1 * (w.hidDim / 4) * 4;
        EnsureDynW4(w4Count);
        EnsureStyleTmp(w.hidDim, w.numOutput);
        ComputeStyleOut(styles, styleRow * 512, w.modulationW, w.modulationB, _styleOutTmp, w.hidDim);
        BuildDynW4_1x1(_dynW4Host, _styleOutTmp, w.selfWeight, w.hidDim, w.numOutput);
        _dynW4.SetData(_dynW4Host, 0, 0, w4Count);

        var zeroB4 = GetZeroBias4(1);
        _ops.Conv1x1Pack4(feat, w.hidDim / 4, _dynW4, zeroB4, 1, 0, 0f, outArr);

        var biased = RentTempArray(outArr.width, outArr.height, 1, RenderTextureFormat.ARGBHalf);
        _ops.AddBiasPack4(outArr, w.bias4, 1, biased);
        ReturnTempArray(outArr);
        outArr = biased;

        if (skip == null)
            return outArr;

        var up = RentTempArray(skip.width * 2, skip.height * 2, 1, RenderTextureFormat.ARGBHalf);
        _ops.Interp2xPack4(skip, 1, up);
        ReturnTempArray(skip);

        var sum = RentTempArray(up.width, up.height, 1, RenderTextureFormat.ARGBHalf);
        _ops.AddPack4(outArr, up, 1f, 1f, 1, sum);
        ReturnTempArray(outArr);
        ReturnTempArray(up);
        return sum;
    }

    private void LoadStyleBin(string stylePath)
    {
        if (_styleConv != null || _toRgb != null)
            return;

        _styleConv = new StyleConvWeights[15];
        _toRgb = new ToRgbWeights[8];

        using (var fs = File.OpenRead(stylePath))
        using (var br = new BinaryReader(fs))
        {
            var styleHidDim = new[] { 512,512,512,512,512,512,512,512,512,256,256,128,128,64,512 };
            var styleOutC = new[] { 512,512,512,512,512,512,512,512,256,256,128,128,64,64,512 };
            for (var i = 0; i < 15; i++)
            {
                var w = new StyleConvWeights();
                w.inc = 512;
                w.hidDim = styleHidDim[i];
                w.numOutput = styleOutC[i];

                var selfCount = w.numOutput * w.hidDim * 3 * 3;
                w.selfWeight = ReadFloatArray(br, selfCount);
                w.modulationW = ReadFloatArray(br, w.inc * w.hidDim);
                w.modulationB = ReadFloatArray(br, w.hidDim);
                w.noiseWeight = br.ReadSingle();
                w.bias = ReadFloatArray(br, w.numOutput);
                w.bias4 = CreateBias4Buffer(w.bias, w.numOutput);
                _styleConv[i] = w;
            }

            var rgbHidDim = new[] { 512,512,512,512,256,128,64,512 };
            for (var i = 0; i < 8; i++)
            {
                var w = new ToRgbWeights();
                w.inc = 512;
                w.hidDim = rgbHidDim[i];
                w.numOutput = 3;
                w.selfWeight = ReadFloatArray(br, w.numOutput * w.hidDim);
                w.modulationW = ReadFloatArray(br, w.inc * w.hidDim);
                w.modulationB = ReadFloatArray(br, w.hidDim);
                w.bias = ReadFloatArray(br, 3);
                w.bias4 = CreateBias4Buffer(w.bias, 3);
                _toRgb[i] = w;
            }

            _constInput = ReadFloatArray(br, 4 * 4 * 512);
        }

        _constInputBuf = new ComputeBuffer(_constInput.Length, sizeof(float), ComputeBufferType.Structured);
        _constInputBuf.SetData(_constInput);
    }

    private static float[] ReadFloatArray(BinaryReader br, int count)
    {
        var a = new float[count];
        for (var i = 0; i < count; i++)
            a[i] = br.ReadSingle();
        return a;
    }

    private ComputeBuffer CreateBias4Buffer(float[] bias, int outC)
    {
        var outPacks = (outC + 3) / 4;
        var v4 = new Vector4[outPacks];
        for (var p = 0; p < outPacks; p++)
        {
            var c0 = p * 4 + 0;
            var c1 = p * 4 + 1;
            var c2 = p * 4 + 2;
            var c3 = p * 4 + 3;
            v4[p] = new Vector4(
                c0 < outC ? bias[c0] : 0f,
                c1 < outC ? bias[c1] : 0f,
                c2 < outC ? bias[c2] : 0f,
                c3 < outC ? bias[c3] : 0f);
        }
        var cb = new ComputeBuffer(outPacks, sizeof(float) * 4, ComputeBufferType.Structured);
        cb.SetData(v4);
        return cb;
    }

    private ComputeBuffer GetZeroBias4(int outPacks)
    {
        if (_zeroBias4.TryGetValue(outPacks, out var cb) && cb != null)
            return cb;
        var v = new Vector4[outPacks];
        cb = new ComputeBuffer(outPacks, sizeof(float) * 4, ComputeBufferType.Structured);
        cb.SetData(v);
        _zeroBias4[outPacks] = cb;
        return cb;
    }

    private void EnsureDynW4(int count)
    {
        if (_dynW4 == null || _dynW4Count < count)
        {
            try { _dynW4?.Dispose(); } catch { }
            _dynW4 = new ComputeBuffer(count, sizeof(float) * 4, ComputeBufferType.Structured);
            _dynW4Count = count;
        }
        if (_dynW4Host == null || _dynW4Host.Length < count)
            _dynW4Host = new Vector4[count];
    }

    private void EnsureStyleTmp(int hidDim, int outC)
    {
        if (_styleOutTmp == null || _styleOutTmp.Length < hidDim)
            _styleOutTmp = new float[hidDim];
        if (_demodTmp == null || _demodTmp.Length < outC)
            _demodTmp = new float[outC];
    }

    private static void ComputeStyleOut(float[] styles, int styleOffset, float[] modW, float[] modB, float[] outVec, int hidDim)
    {
        for (var o = 0; o < hidDim; o++)
        {
            var sum = modB[o];
            var wbase = o * 512;
            for (var i = 0; i < 512; i++)
                sum += modW[wbase + i] * styles[styleOffset + i];
            outVec[o] = sum;
        }
    }

    private static void BuildDynW4_3x3(Vector4[] dst, float[] styleOut, float[] selfWeight, int hidDim, int outC, float[] demodTmp)
    {
        var inPacks = hidDim / 4;
        var outPacks = (outC + 3) / 4;
        var k = 9;
        if (demodTmp != null)
        {
            for (var oc = 0; oc < outC; oc++)
            {
                double sum = 0.0;
                var base0 = oc * hidDim * k;
                for (var ic = 0; ic < hidDim; ic++)
                {
                    var s = styleOut[ic];
                    var base1 = base0 + ic * k;
                    for (var kk = 0; kk < k; kk++)
                    {
                        var v = selfWeight[base1 + kk] * s;
                        sum += v * v;
                    }
                }
                demodTmp[oc] = (float)(1.0 / Math.Sqrt(sum + 1e-8));
            }
        }

        var idx = 0;
        for (var op = 0; op < outPacks; op++)
        {
            for (var ip = 0; ip < inPacks; ip++)
            {
                for (var kk = 0; kk < k; kk++)
                {
                    for (var ol = 0; ol < 4; ol++)
                    {
                        var oc = op * 4 + ol;
                        if (oc >= outC)
                        {
                            dst[idx++] = Vector4.zero;
                            continue;
                        }
                        var dm = demodTmp != null ? demodTmp[oc] : 1f;
                        var il0 = ip * 4 + 0;
                        var il1 = ip * 4 + 1;
                        var il2 = ip * 4 + 2;
                        var il3 = ip * 4 + 3;
                        var b0 = (oc * hidDim + il0) * k + kk;
                        var b1 = (oc * hidDim + il1) * k + kk;
                        var b2 = (oc * hidDim + il2) * k + kk;
                        var b3 = (oc * hidDim + il3) * k + kk;
                        var x0 = selfWeight[b0] * styleOut[il0] * dm;
                        var x1 = selfWeight[b1] * styleOut[il1] * dm;
                        var x2 = selfWeight[b2] * styleOut[il2] * dm;
                        var x3 = selfWeight[b3] * styleOut[il3] * dm;
                        dst[idx++] = new Vector4(x0, x1, x2, x3);
                    }
                }
            }
        }
    }

    private static void BuildDynW4_1x1(Vector4[] dst, float[] styleOut, float[] selfWeight, int hidDim, int outC)
    {
        var inPacks = hidDim / 4;
        var outPacks = (outC + 3) / 4;
        var idx = 0;
        for (var op = 0; op < outPacks; op++)
        {
            for (var ip = 0; ip < inPacks; ip++)
            {
                for (var ol = 0; ol < 4; ol++)
                {
                    var oc = op * 4 + ol;
                    if (oc >= outC)
                    {
                        dst[idx++] = Vector4.zero;
                        continue;
                    }
                    var il0 = ip * 4 + 0;
                    var il1 = ip * 4 + 1;
                    var il2 = ip * 4 + 2;
                    var il3 = ip * 4 + 3;
                    var b0 = oc * hidDim + il0;
                    var b1 = oc * hidDim + il1;
                    var b2 = oc * hidDim + il2;
                    var b3 = oc * hidDim + il3;
                    var x0 = selfWeight[b0] * styleOut[il0];
                    var x1 = selfWeight[b1] * styleOut[il1];
                    var x2 = selfWeight[b2] * styleOut[il2];
                    var x3 = selfWeight[b3] * styleOut[il3];
                    dst[idx++] = new Vector4(x0, x1, x2, x3);
                }
            }
        }
    }

    private ComputeBuffer GetNoiseBuffer(int wh)
    {
        if (_noiseBuf.TryGetValue(wh, out var cb) && cb != null)
            return cb;
        cb = new ComputeBuffer(wh, sizeof(float), ComputeBufferType.Structured);
        _noiseBuf[wh] = cb;
        return cb;
    }

    private static void FillNoise(ComputeBuffer noise, int wh)
    {
        var r = new System.Random(unchecked((int)DateTime.UtcNow.Ticks));
        var a = new float[wh];
        var i = 0;
        while (i < wh)
        {
            var u1 = Math.Max(1e-12, r.NextDouble());
            var u2 = Math.Max(1e-12, r.NextDouble());
            var mag = Math.Sqrt(-2.0 * Math.Log(u1));
            var z0 = (float)(mag * Math.Cos(2.0 * Math.PI * u2));
            var z1 = (float)(mag * Math.Sin(2.0 * Math.PI * u2));
            a[i++] = z0;
            if (i < wh) a[i++] = z1;
        }
        noise.SetData(a);
    }

    private static RectInt FindFaceRect(Texture2D mask, int w, int h, float threshold01)
    {
        if (mask == null || mask.width <= 0 || mask.height <= 0)
            return new RectInt(w / 4, h / 4, w / 2, h / 2);
        var pixels = mask.GetPixels();
        var minX = w;
        var minY = h;
        var maxX = -1;
        var maxY = -1;
        for (var y = 0; y < mask.height; y++)
        {
            for (var x = 0; x < mask.width; x++)
            {
                var v = pixels[y * mask.width + x].r;
                if (v < threshold01)
                    continue;
                minX = Math.Min(minX, x);
                minY = Math.Min(minY, y);
                maxX = Math.Max(maxX, x);
                maxY = Math.Max(maxY, y);
            }
        }
        if (maxX < 0 || maxY < 0)
            return new RectInt(w / 4, h / 4, w / 2, h / 2);
        return new RectInt(minX, minY, maxX - minX + 1, maxY - minY + 1);
    }

    private static RectInt ExpandRect(RectInt r, int w, int h, float expand01)
    {
        if (r.width <= 0 || r.height <= 0)
            return r;
        var cx = r.x + r.width * 0.5f;
        var cy = r.y + r.height * 0.5f;
        var ex = r.width * (1f + Mathf.Max(0f, expand01));
        var ey = r.height * (1f + Mathf.Max(0f, expand01));
        var x0 = Mathf.Clamp(Mathf.FloorToInt(cx - ex * 0.5f), 0, Mathf.Max(0, w - 1));
        var y0 = Mathf.Clamp(Mathf.FloorToInt(cy - ey * 0.5f), 0, Mathf.Max(0, h - 1));
        var x1 = Mathf.Clamp(Mathf.CeilToInt(cx + ex * 0.5f), 0, w);
        var y1 = Mathf.Clamp(Mathf.CeilToInt(cy + ey * 0.5f), 0, h);
        return new RectInt(x0, y0, Mathf.Max(1, x1 - x0), Mathf.Max(1, y1 - y0));
    }

    private static RenderTexture ComposeWithMask(Texture2D baseTex, Texture2D overlayCrop, Texture2D mask, RectInt rect)
    {
        if (baseTex == null || overlayCrop == null)
            return null;

        var cs = Resources.Load<ComputeShader>("ImageProcessing");
        if (cs == null)
            return null;

        int k;
        try { k = cs.FindKernel("PasteRectWithMask"); } catch { return null; }
        if (k < 0)
            return null;

        var rt = new RenderTexture(baseTex.width, baseTex.height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        rt.enableRandomWrite = true;
        rt.Create();

        cs.SetTexture(k, "_Source", baseTex);
        cs.SetTexture(k, "_Overlay", overlayCrop);
        cs.SetTexture(k, "_Result", rt);
        cs.SetInts("_CropRect", rect.x, rect.y, rect.width, rect.height);

        if (mask != null && mask.width == baseTex.width && mask.height == baseTex.height && mask.format == TextureFormat.RHalf)
            cs.SetTexture(k, "_FaceMaskIn", mask);
        else
            cs.SetTexture(k, "_FaceMaskIn", Texture2D.blackTexture);

        cs.Dispatch(k, Mathf.CeilToInt(baseTex.width / 8f), Mathf.CeilToInt(baseTex.height / 8f), 1);
        return rt;
    }

    private static Texture2D RenderTextureToTexture2D(RenderTexture rt, int w, int h)
    {
        if (rt == null || w <= 0 || h <= 0)
            return null;
        var prev = RenderTexture.active;
        try
        {
            RenderTexture.active = rt;
            var tex = new Texture2D(w, h, TextureFormat.RGBA32, false, false);
            tex.ReadPixels(new Rect(0, 0, w, h), 0, 0, false);
            tex.Apply(false, false);
            tex.wrapMode = TextureWrapMode.Clamp;
            tex.filterMode = FilterMode.Bilinear;
            return tex;
        }
        catch
        {
            return null;
        }
        finally
        {
            RenderTexture.active = prev;
        }
    }

    private RenderTexture ForwardPack4(RenderTexture inputPack4, int inputPacks)
    {
        var remaining = new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);
        HashSet<string> pinned = null;
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
                    Consume(blobs, remaining, l.bottomNames, pinned);
                    continue;
                }

                if (string.Equals(l.type, "Reshape", StringComparison.Ordinal))
                {
                    if (blobs.TryGetValue(l.bottomNames[0], out var srcTex) && srcTex != null)
                    {
                        blobs[l.topNames[0]] = srcTex;
                        srcTex.refs++;
                        Consume(blobs, remaining, l.bottomNames, pinned);
                        continue;
                    }
                    if (bufferBlobs.TryGetValue(l.bottomNames[0], out var srcBuf) && srcBuf != null)
                    {
                        bufferBlobs[l.topNames[0]] = srcBuf;
                        Consume(blobs, remaining, l.bottomNames, pinned);
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
                    Consume(blobs, remaining, l.bottomNames, pinned);
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
                    Consume(blobs, remaining, l.bottomNames, pinned);
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
                    Consume(blobs, remaining, l.bottomNames, pinned);
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
                        Consume(blobs, remaining, l.bottomNames, pinned);
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
                        Consume(blobs, remaining, l.bottomNames, pinned);
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
        var stylePath = Path.Combine(Application.streamingAssetsPath, styleRelativePath);
        if (!File.Exists(paramPath))
            throw new InvalidOperationException("GFPGAN(复刻) param 不存在: " + paramPath);
        if (!File.Exists(binPath))
            throw new InvalidOperationException("GFPGAN(复刻) bin 不存在: " + binPath);
        if (!File.Exists(stylePath))
            throw new InvalidOperationException("GFPGAN(复刻) style 不存在: " + stylePath);

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

        LoadStyleBin(stylePath);
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

    private void Consume(Dictionary<string, TensorRef> blobs, Dictionary<string, int> remaining, string[] bottomNames, HashSet<string> pinned)
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
            if (pinned != null && pinned.Contains(b))
                continue;

            if (blobs.TryGetValue(b, out var tr) && tr != null)
            {
                tr.refs--;
                if (tr.refs <= 0)
                {
                    if (tr.owned && tr.t != null)
                    {
                        try { ReturnTempArray(tr.t); } catch { }
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
                if (hit == null && p.rt != null && IsFencePassedOrAged(p))
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
            stack.Push(new PooledRt(rt, fence, Time.frameCount));
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

    private static bool IsFencePassedOrAged(PooledRt p)
    {
        try
        {
            return p.fence.passed;
        }
        catch
        {
            return (Time.frameCount - p.frame) >= 2;
        }
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
