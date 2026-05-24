using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Threading;
using Cysharp.Threading.Tasks;
using NcnnCompute;
using UnityEngine;
using UnityEngine.Rendering;

public sealed class RealEsrganNcnnReproRunner : MonoBehaviour
{
    public bool enableRepro = true;
    public string modelName = "realesrgan-x4plus";
    public int tileSize = 128;
    public int tilePad = 10;
    public int maxInputLongSide = 2048;

    public event Action<float, string> ProgressChanged;

    private NcnnParamModel _model;
    private readonly Dictionary<string, ConvPack> _conv = new Dictionary<string, ConvPack>(StringComparer.Ordinal);
    private Dictionary<string, int> _blobUseCount;
    private NcnnOps _ops;
    private bool _loaded;

    private sealed class ConvPack : IDisposable
    {
        public int outC;
        public int kernel;
        public int pad;
        public int biasTerm;
        public int weightSize;
        public int activationType;
        public float activationSlope;
        public ComputeBuffer w;
        public ComputeBuffer b;

        public void Dispose()
        {
            try { w?.Dispose(); } catch { }
            try { b?.Dispose(); } catch { }
        }
    }

    private sealed class TensorRef
    {
        public NcnnTensorBuffer t;
        public int refs;
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
        _loaded = false;
        _model = null;
        _blobUseCount = null;
    }

    public async UniTask<RealEsrganResult> ProcessAsync(Texture2D src, CancellationToken ct)
    {
        if (!enableRepro)
            return new RealEsrganResult { error = "ESRGAN(复刻) disabled" };
        if (src == null)
            return default;

        EnsureLoaded();

        var originalW = src.width;
        var originalH = src.height;
        var runFactor = 4;
        var limit = Mathf.Max(256, maxInputLongSide);
        var maxSide = Mathf.Max(originalW, originalH);
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

        Texture2D runInput = null;
        var ownsRunInput = false;
        RenderTexture outRt = null;
        try
        {
            if (runInW != originalW || runInH != originalH)
            {
                runInput = ResizeTextureBilinear(src, runInW, runInH);
                ownsRunInput = true;
                if (runInput == null)
                    return new RealEsrganResult { error = "resize input failed" };
            }
            else
            {
                runInput = src;
            }

            outRt = new RenderTexture(runInW, runInH, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
            outRt.enableRandomWrite = true;
            outRt.wrapMode = TextureWrapMode.Clamp;
            outRt.filterMode = FilterMode.Bilinear;
            outRt.Create();

            var tilesX = Mathf.CeilToInt(runInW / (float)Mathf.Max(1, tileSize));
            var tilesY = Mathf.CeilToInt(runInH / (float)Mathf.Max(1, tileSize));
            var tileCount = Mathf.Max(1, tilesX * tilesY);
            var tileIndex = 0;

            for (var ty = 0; ty < runInH; ty += Mathf.Max(1, tileSize))
            {
                for (var tx = 0; tx < runInW; tx += Mathf.Max(1, tileSize))
                {
                    ct.ThrowIfCancellationRequested();

                    var tw = Mathf.Min(tileSize, runInW - tx);
                    var th = Mathf.Min(tileSize, runInH - ty);
                    var cw = tw + tilePad * 2;
                    var ch = th + tilePad * 2;
                    var ox = tx - tilePad;
                    var oy = ty - tilePad;
                    var tileProgress = (float)tileIndex / tileCount;
                    ReportProgress(tileProgress, "推理分块 " + (tileIndex + 1) + "/" + tileCount);
                    await UniTask.Yield();

                    using var inBuf = new NcnnTensorBuffer(cw, ch, 3);
                    _ops.TextureToBuffer3(runInput, ox, oy, inBuf);
                    using var outBuf = Forward(inBuf);

                    var tileRt = RenderTexture.GetTemporary(cw * runFactor, ch * runFactor, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
                    tileRt.enableRandomWrite = true;
                    tileRt.wrapMode = TextureWrapMode.Clamp;
                    tileRt.filterMode = FilterMode.Bilinear;
                    tileRt.Create();
                    _ops.BufferToTexture3(outBuf, tileRt);

                    var srcX = tilePad * runFactor;
                    var srcY = tilePad * runFactor;
                    _ops.BlitCropDown4(tileRt, outRt, tx, ty, srcX, srcY, tw, th);

                    RenderTexture.ReleaseTemporary(tileRt);

                    tileIndex++;
                }
            }

            ReportProgress(0.98f, "读取结果");
            var scaledTex = await ReadbackTextureAsync(outRt, outRt.width, outRt.height, ct);
            if (scaledTex == null)
                return new RealEsrganResult { error = "readback failed" };

            Texture2D finalTex = scaledTex;
            if (finalTex.width != originalW || finalTex.height != originalH)
            {
                var resized = ResizeTextureBilinear(finalTex, originalW, originalH);
                Destroy(finalTex);
                finalTex = resized;
                if (finalTex == null)
                    return new RealEsrganResult { error = "resize output failed" };
            }
            if (finalTex == null)
                return new RealEsrganResult { error = "resize output failed" };

            ReportProgress(1f, "完成");
            return new RealEsrganResult { texture = finalTex, error = null };
        }
        catch (OperationCanceledException)
        {
            return new RealEsrganResult { error = "Cancelled" };
        }
        catch (Exception e)
        {
            return new RealEsrganResult { error = e.Message };
        }
        finally
        {
            if (ownsRunInput && runInput != null) Destroy(runInput);
            if (outRt != null)
            {
                outRt.Release();
                Destroy(outRt);
            }
        }
    }

    private void EnsureLoaded()
    {
        if (_loaded)
            return;

        var model = string.IsNullOrWhiteSpace(modelName) ? "realesrgan-x4plus" : modelName.Trim();
        var paramPath = Path.Combine(Application.streamingAssetsPath, "RealESRGAN", "models", model + ".param");
        var binPath = Path.Combine(Application.streamingAssetsPath, "RealESRGAN", "models", model + ".bin");

        var paramText = File.ReadAllText(paramPath);
        _model = NcnnParamParser.Parse(paramText);
        _blobUseCount = BuildBlobUseCount(_model);

        using (var fs = File.OpenRead(binPath))
        using (var br = new NcnnBinReader(fs))
        {
            foreach (var layer in _model.layers)
            {
                if (!string.Equals(layer.type, "Convolution", StringComparison.Ordinal))
                    continue;

                var pack = new ConvPack();
                pack.outC = layer.GetInt(0, 0);
                pack.kernel = layer.GetInt(1, 3);
                pack.pad = layer.GetInt(4, 0);
                pack.biasTerm = layer.GetInt(5, 0);
                pack.weightSize = layer.GetInt(6, 0);
                pack.activationType = layer.GetInt(9, 0);
                pack.activationSlope = ParseLeakySlope(layer);

                var tag = br.ReadInt32();
                if (tag != 0x01306B47)
                    throw new InvalidOperationException("unexpected weight tag at " + br.Position + ": 0x" + tag.ToString("X8", CultureInfo.InvariantCulture));

                var w = br.ReadFp16ArrayAsFloat32(pack.weightSize);
                var b = pack.biasTerm != 0 ? br.ReadFloat32Array(pack.outC) : new float[pack.outC];
                pack.w = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                pack.b = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                pack.w.SetData(w);
                pack.b.SetData(b);

                _conv[layer.name] = pack;
            }
        }

        _loaded = true;
    }

    private NcnnTensorBuffer Forward(NcnnTensorBuffer input)
    {
        var remaining = new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);
        var blobs = new Dictionary<string, TensorRef>(StringComparer.Ordinal);

        var inputRef = new TensorRef { t = input, refs = 1 };
        blobs["data"] = inputRef;

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

            if (string.Equals(l.type, "Concat", StringComparison.Ordinal))
            {
                var parts = new TensorRef[l.bottomNames.Length];
                var w = 0;
                var h = 0;
                var sumC = 0;
                for (var i = 0; i < l.bottomNames.Length; i++)
                {
                    var tr = Get(blobs, l.bottomNames[i]);
                    parts[i] = tr;
                    w = tr.t.w;
                    h = tr.t.h;
                    sumC += tr.t.c;
                }

                var outBuf = new NcnnTensorBuffer(w, h, sumC);
                var off = 0;
                for (var i = 0; i < parts.Length; i++)
                {
                    _ops.CopyToConcat(parts[i].t, outBuf, off);
                    off += parts[i].t.c;
                }

                blobs[l.topNames[0]] = new TensorRef { t = outBuf, refs = 1 };
                Consume(blobs, remaining, l.bottomNames);
                continue;
            }

            if (string.Equals(l.type, "Convolution", StringComparison.Ordinal))
            {
                var src = Get(blobs, l.bottomNames[0]);
                var pack = _conv[l.name];
                if (pack.kernel != 3)
                    throw new InvalidOperationException("unsupported kernel size: " + pack.kernel);
                var outBuf = new NcnnTensorBuffer(src.t.w, src.t.h, pack.outC);
                _ops.Conv3x3(src.t, pack.w, pack.b, pack.outC, 1, pack.pad, outBuf);
                if (pack.activationType == 2)
                    _ops.LeakyReluInplace(outBuf, pack.activationSlope);
                blobs[l.topNames[0]] = new TensorRef { t = outBuf, refs = 1 };
                Consume(blobs, remaining, l.bottomNames);
                continue;
            }

            if (string.Equals(l.type, "Eltwise", StringComparison.Ordinal))
            {
                var a = Get(blobs, l.bottomNames[0]);
                var b = Get(blobs, l.bottomNames[1]);
                var coeff = ParseEltwiseCoeff(l);
                var outBuf = new NcnnTensorBuffer(a.t.w, a.t.h, a.t.c);
                _ops.AddWeighted(a.t, b.t, coeff.coeffA, coeff.coeffB, outBuf);
                blobs[l.topNames[0]] = new TensorRef { t = outBuf, refs = 1 };
                Consume(blobs, remaining, l.bottomNames);
                continue;
            }

            if (string.Equals(l.type, "BinaryOp", StringComparison.Ordinal))
            {
                var a = Get(blobs, l.bottomNames[0]);
                var b = Get(blobs, l.bottomNames[1]);
                var outBuf = new NcnnTensorBuffer(a.t.w, a.t.h, a.t.c);
                _ops.AddWeighted(a.t, b.t, 1f, 1f, outBuf);
                blobs[l.topNames[0]] = new TensorRef { t = outBuf, refs = 1 };
                Consume(blobs, remaining, l.bottomNames);
                continue;
            }

            if (string.Equals(l.type, "Interp", StringComparison.Ordinal))
            {
                var src = Get(blobs, l.bottomNames[0]);
                var sx = l.GetFloat(1, 1f);
                var sy = l.GetFloat(2, 1f);
                if (Mathf.Abs(sx - 2f) > 1e-3f || Mathf.Abs(sy - 2f) > 1e-3f)
                    throw new InvalidOperationException("unsupported interp scale: " + sx.ToString("0.###", CultureInfo.InvariantCulture) + "," + sy.ToString("0.###", CultureInfo.InvariantCulture));
                var outBuf = new NcnnTensorBuffer(src.t.w * 2, src.t.h * 2, src.t.c);
                _ops.Interp2x(src.t, outBuf);
                blobs[l.topNames[0]] = new TensorRef { t = outBuf, refs = 1 };
                Consume(blobs, remaining, l.bottomNames);
                continue;
            }

            throw new InvalidOperationException("unsupported layer type: " + l.type);
        }

        var outRef = Get(blobs, "output");
        var keep = outRef.t;
        outRef.t = null;

        var visited = new HashSet<TensorRef>();
        foreach (var kv in blobs)
        {
            var tr = kv.Value;
            if (tr == null || !visited.Add(tr))
                continue;
            if (tr.t != null)
                tr.t.Dispose();
        }

        return keep;
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
                    try { tr.t?.Dispose(); } catch { }
                    tr.t = null;
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

    private static (float coeffA, float coeffB) ParseEltwiseCoeff(NcnnParamModel.Layer layer)
    {
        if (layer.intParams == null || !layer.intParams.TryGetValue(-23301, out var s) || string.IsNullOrWhiteSpace(s))
            return (1f, 1f);
        var parts = s.Split(',');
        if (parts.Length >= 3
            && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var a)
            && float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out var b))
            return (a, b);
        return (1f, 1f);
    }

    private async UniTask<Texture2D> ReadbackTextureAsync(RenderTexture rt, int w, int h, CancellationToken ct)
    {
        var tcs = new UniTaskCompletionSource<AsyncGPUReadbackRequest>();
        AsyncGPUReadback.Request(rt, 0, TextureFormat.RGBA32, req => tcs.TrySetResult(req));
        var r = await tcs.Task;
        ct.ThrowIfCancellationRequested();
        if (r.hasError)
            return null;
        var data = r.GetData<byte>();
        var tex = new Texture2D(w, h, TextureFormat.RGBA32, false, true);
        tex.LoadRawTextureData(data);
        tex.Apply(false, false);
        tex.wrapMode = TextureWrapMode.Clamp;
        tex.filterMode = FilterMode.Bilinear;
        return tex;
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

    private void ReportProgress(float p, string t)
    {
        try { ProgressChanged?.Invoke(Mathf.Clamp01(p), t ?? ""); } catch { }
    }
}
