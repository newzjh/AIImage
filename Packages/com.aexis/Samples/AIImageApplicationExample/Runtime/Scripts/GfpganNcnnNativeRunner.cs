using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using Aexis.Samples.Async;
using UnityEngine;
using UnityEngine.Networking;

[Obsolete("native plugin mode is not support any more")]
public sealed class GfpganNcnnNativeRunner : MonoBehaviour
{
    public bool enableGfpganNative = true;
        public string modelName = "encoder";
    public int gpuId = -1;
    public int maxInputLongSide = 2048;
    public float faceMaskThreshold = 0.2f;
    public float faceBoxExpand = 0.35f;

    public event Action<float, string> ProgressChanged;

    private IntPtr _ctx = IntPtr.Zero;

    public async UniTask<GfpganResult> ProcessAsync(Texture2D src, CancellationToken ct)
    {
        if (!enableGfpganNative)
            return new GfpganResult { error = "GFPGAN(原生) disabled" };
        if (src == null)
            return default;

        var totalSw = Stopwatch.StartNew();

        GfpganResult Finish(GfpganResult r, int w, int h)
        {
            r.elapsedMs = totalSw.ElapsedMilliseconds;
            try
            {
                UnityEngine.Debug.Log("[TIMING] GFPGAN(native) " + r.elapsedMs + " ms | in=" + w + "x" + h + " | model=" + (modelName ?? "") + " | err=" + (r.error ?? ""));
            }
            catch
            {
            }
            return r;
        }

        var modelDir = await PrepareModelDirAsync(modelName, ct);
        if (string.IsNullOrWhiteSpace(modelDir) || !Directory.Exists(modelDir))
            return Finish(new GfpganResult { error = "models目录不可用: " + (modelDir ?? "") }, src.width, src.height);

        try
        {
            await ReportDbgAsync("A", "gfpgan.native.ensureInit.begin", "[DEBUG] EnsureInit begin",
                "{\"modelDir\":\"" + EscapeJson(modelDir) + "\",\"modelName\":\"" + EscapeJson(modelName) + "\",\"gpuId\":" + gpuId + ",\"streamingAssetsPath\":\"" + EscapeJson(Application.streamingAssetsPath) + "\"}",
                "", ct);
            EnsureInit(modelDir);
            await ReportDbgAsync("A", "gfpgan.native.ensureInit.ok", "[DEBUG] EnsureInit ok",
                "{\"ctx\":\"" + (_ctx != IntPtr.Zero ? "nonzero" : "zero") + "\"}", "", ct);
        }
        catch (Exception e)
        {
            await ReportDbgAsync("B", "gfpgan.native.ensureInit.exception", "[DEBUG] EnsureInit exception",
                "{\"msg\":\"" + EscapeJson(e.Message) + "\"}", "", ct);
            return Finish(new GfpganResult { error = e.Message }, src.width, src.height);
        }

        var originalW = src.width;
        var originalH = src.height;

        ReportProgress(0f, "准备输入");
        await UniTask.Yield();

        Texture2D scaledInput = null;
        Texture2D faceMask = null;
        Texture2D faceCrop = null;
        Texture2D face512 = null;
        Texture2D restored512 = null;
        Texture2D restoredCrop = null;
        RenderTexture composed = null;

        try
        {
            var inputTex = src;
            var maxSide = Mathf.Max(originalW, originalH);
            var limit = Mathf.Max(256, maxInputLongSide);
            float scaleDown = 1f;
            if (maxSide > limit)
            {
                ReportProgress(0.02f, "缩小到2k以内");
                scaleDown = (float)limit / maxSide;
                var sw = Mathf.Max(1, Mathf.RoundToInt(originalW * scaleDown));
                var sh = Mathf.Max(1, Mathf.RoundToInt(originalH * scaleDown));
                scaledInput = ResizeTextureBilinear(src, sw, sh);
                if (scaledInput == null)
                    return Finish(new GfpganResult { error = "缩小输入失败" }, originalW, originalH);
                inputTex = scaledInput;
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
                return Finish(new GfpganResult { error = "裁剪失败" }, originalW, originalH);

            face512 = ResizeTextureBilinear(faceCrop, 512, 512);
            if (face512 == null)
                return Finish(new GfpganResult { error = "resize到512失败" }, originalW, originalH);

            var rgba = face512.GetRawTextureData<byte>().ToArray();
            var outRgba = new byte[512 * 512 * 4];

            ReportProgress(0.15f, "推理中…");
            await ReportDbgAsync("A", "gfpgan.native.call.begin", "[DEBUG] calling native Gfpgan_ProcessRgba",
                "{\"ctx\":\"" + (_ctx != IntPtr.Zero ? "nonzero" : "zero") + "\",\"w\":512,\"h\":512,\"persistentDataPath\":\"" + EscapeJson(Application.persistentDataPath) + "\"}",
                "", ct);
            var errPtr = await UniTask.RunOnThreadPool(() =>
            {
                ct.ThrowIfCancellationRequested();
                return GfpganNcnnNative.Gfpgan_ProcessRgba(_ctx, rgba, 512, 512, outRgba, 512, 512);
            }, cancellationToken: ct);

            var err = GfpganNcnnNative.Utf8ToString(errPtr);
            if (!string.IsNullOrWhiteSpace(err))
            {
                await ReportDbgAsync("C", "gfpgan.native.call.error", "[DEBUG] native returned error",
                    "{\"err\":\"" + EscapeJson(err) + "\"}", "", ct);
                return Finish(new GfpganResult { error = err }, originalW, originalH);
            }
            await ReportDbgAsync("A", "gfpgan.native.call.ok", "[DEBUG] native returned ok", "{}", "", ct);

            await UniTask.SwitchToMainThread();
            restored512 = new Texture2D(512, 512, TextureFormat.RGBA32, false, false);
            restored512.LoadRawTextureData(outRgba);
            restored512.Apply(false, false);
            restored512.wrapMode = TextureWrapMode.Clamp;
            restored512.filterMode = FilterMode.Bilinear;

            ReportProgress(0.85f, "回贴到原图");
            restoredCrop = ResizeTextureBilinear(restored512, rect.width, rect.height);
            if (restoredCrop == null)
                return Finish(new GfpganResult { error = "回贴缩放失败" }, originalW, originalH);

            composed = ComposeWithMask(inputTex, restoredCrop, faceMask, rect);
            if (composed == null)
                return Finish(new GfpganResult { error = "合成失败" }, originalW, originalH);

            var composedTex = RenderTextureToTexture2D(composed, inputTex.width, inputTex.height);
            if (composedTex == null)
                return Finish(new GfpganResult { error = "合成回读失败" }, originalW, originalH);

            Texture2D finalTex = composedTex;
            if (finalTex.width != originalW || finalTex.height != originalH)
            {
                ReportProgress(0.95f, "回缩放到原分辨率");
                var resized = ResizeTextureBilinear(finalTex, originalW, originalH);
                Destroy(finalTex);
                finalTex = resized;
                if (finalTex == null)
                    return Finish(new GfpganResult { error = "回缩放失败" }, originalW, originalH);
            }

            finalTex.wrapMode = TextureWrapMode.Clamp;
            finalTex.filterMode = FilterMode.Bilinear;
            finalTex.name = "GFPGAN_NCNN";
            ReportProgress(1f, "完成");
            return Finish(new GfpganResult { texture = finalTex }, originalW, originalH);
        }
        finally
        {
            if (scaledInput != null) Destroy(scaledInput);
            if (faceCrop != null) Destroy(faceCrop);
            if (face512 != null) Destroy(face512);
            if (restored512 != null) Destroy(restored512);
            if (restoredCrop != null) Destroy(restoredCrop);
            if (composed != null) { composed.Release(); Destroy(composed); }
        }
    }

    private void EnsureInit(string modelDir)
    {
        if (_ctx != IntPtr.Zero)
            return;

#if UNITY_IOS && !UNITY_EDITOR
        throw new PlatformNotSupportedException(
            "iOS release packages do not include the optional GFPGAN native NCNN bridge.");
#else
        try
        {
            var errPtr = GfpganNcnnNative.Gfpgan_Create(modelDir, modelName, gpuId, out _ctx);
            var err = GfpganNcnnNative.Utf8ToString(errPtr);
            if (!string.IsNullOrWhiteSpace(err))
                throw new InvalidOperationException(err);
            if (_ctx == IntPtr.Zero)
                throw new InvalidOperationException("创建GFPGAN(原生)上下文失败");
        }
        catch (DllNotFoundException e)
        {
            throw new InvalidOperationException("GFPGAN(原生) DLL加载失败: " + e.Message);
        }
        catch (EntryPointNotFoundException e)
        {
            throw new InvalidOperationException("GFPGAN(原生) 导出函数缺失: " + e.Message);
        }
#endif
    }

    private void OnDestroy()
    {
        if (_ctx != IntPtr.Zero)
        {
            try { GfpganNcnnNative.Gfpgan_Destroy(_ctx); } catch { }
            _ctx = IntPtr.Zero;
        }
    }

    private void ReportProgress(float progress01, string text)
    {
        progress01 = Mathf.Clamp01(progress01);
        try { ProgressChanged?.Invoke(progress01, text ?? ""); } catch { }
    }

    #region debug-point windows-player-native-crash-report
    private static string _dbgUrl;
    private static string _dbgSessionId;
    private static bool _dbgLoaded;

    private static void LoadDbgEnv()
    {
        if (_dbgLoaded) return;
        _dbgLoaded = true;
        _dbgUrl = "http://127.0.0.1:7778/event";
        _dbgSessionId = "windows-player-native-crash";
        try
        {
            var envName = "windows-player-native-crash.env";
            var candidates = new[]
            {
                Path.Combine(Application.persistentDataPath, ".dbg", envName),
                Path.Combine(Environment.CurrentDirectory, ".dbg", envName),
                Path.Combine(Application.dataPath, "..", ".dbg", envName)
            };
            for (var i = 0; i < candidates.Length; i++)
            {
                var envPath = candidates[i];
                if (!File.Exists(envPath))
                    continue;
                var lines = File.ReadAllLines(envPath);
                foreach (var raw in lines)
                {
                    var line = raw?.Trim();
                    if (string.IsNullOrEmpty(line)) continue;
                    if (line.StartsWith("DEBUG_SERVER_URL=", StringComparison.Ordinal))
                        _dbgUrl = line.Substring("DEBUG_SERVER_URL=".Length).Trim();
                    else if (line.StartsWith("DEBUG_SESSION_ID=", StringComparison.Ordinal))
                        _dbgSessionId = line.Substring("DEBUG_SESSION_ID=".Length).Trim();
                }
                break;
            }
        }
        catch
        {
        }
    }

    private static async UniTask ReportDbgAsync(string hypothesisId, string type, string msg, string dataJson, string traceId, CancellationToken ct)
    {
        try
        {
            LoadDbgEnv();
            if (string.IsNullOrWhiteSpace(hypothesisId)) hypothesisId = "A";
            if (string.IsNullOrWhiteSpace(dataJson)) dataJson = "{}";
            var payload =
                "{\"sessionId\":\"" + EscapeJson(_dbgSessionId) + "\"" +
                ",\"runId\":\"pre-fix\"" +
                ",\"hypothesisId\":\"" + EscapeJson(hypothesisId) + "\"" +
                ",\"location\":\"GfpganNcnnNativeRunner\"" +
                ",\"msg\":\"" + EscapeJson(msg) + "\"" +
                ",\"type\":\"" + EscapeJson(type) + "\"" +
                ",\"traceId\":\"" + EscapeJson(traceId ?? "") + "\"" +
                ",\"ts\":" + DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() +
                ",\"data\":" + dataJson +
                "}";
            var line = payload + "\n";

            var logPath = Path.Combine(Application.persistentDataPath, "trae-debug-log-windows-player-native-crash.ndjson");
            try { File.AppendAllText(logPath, line); } catch { }
            try { await File.AppendAllTextAsync(logPath, line, ct); } catch { }

            if (!string.IsNullOrWhiteSpace(_dbgUrl))
            {
                try
                {
                    await UniTask.SwitchToMainThread();
                    using var req = new UnityWebRequest(_dbgUrl, "POST");
                    var body = System.Text.Encoding.UTF8.GetBytes(payload);
                    req.uploadHandler = new UploadHandlerRaw(body);
                    req.downloadHandler = new DownloadHandlerBuffer();
                    req.SetRequestHeader("Content-Type", "application/json");
                    var op = req.SendWebRequest();
                    while (!op.isDone)
                        await UniTask.Delay(15, cancellationToken: ct);
                }
                catch
                {
                }
            }
        }
        catch
        {
        }
    }

    private static string EscapeJson(string s)
    {
        if (string.IsNullOrEmpty(s)) return "";
        return s.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\r", "\\r").Replace("\n", "\\n");
    }
    #endregion

    private static RectInt FindFaceRect(Texture2D mask, int w, int h, float threshold)
    {
        if (mask == null || mask.width != w || mask.height != h)
            return default;
        if (mask.format != TextureFormat.RHalf)
            return default;

        var data = mask.GetRawTextureData<byte>();
        if (data.Length < w * h * 2)
            return default;

        int minX = w, minY = h, maxX = -1, maxY = -1;
        var t = threshold;
        for (int y = 0; y < h; y++)
        {
            var row = y * w * 2;
            for (int x = 0; x < w; x++)
            {
                var i = row + x * 2;
                ushort halfBits = (ushort)(data[i] | (data[i + 1] << 8));
                float v = HalfToFloat(halfBits);
                if (v > t)
                {
                    if (x < minX) minX = x;
                    if (y < minY) minY = y;
                    if (x > maxX) maxX = x;
                    if (y > maxY) maxY = y;
                }
            }
        }

        if (maxX < minX || maxY < minY)
            return default;
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

    private static Texture2D CropTexture(Texture2D src, RectInt r)
    {
        if (src == null || r.width <= 0 || r.height <= 0)
            return null;
        var prev = RenderTexture.active;
        var rt = new RenderTexture(src.width, src.height, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        rt.Create();
        try
        {
            Graphics.Blit(src, rt);
            RenderTexture.active = rt;
            var tex = new Texture2D(r.width, r.height, TextureFormat.RGBA32, false, false);
            tex.ReadPixels(new Rect(r.x, r.y, r.width, r.height), 0, 0, false);
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
            try { rt.Release(); } catch { }
            Destroy(rt);
        }
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

    private static Texture2D ResizeTextureBilinear(Texture2D src, int w, int h)
    {
        if (src == null || w <= 0 || h <= 0)
            return null;
        var prev = RenderTexture.active;
        var rt = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        rt.enableRandomWrite = false;
        rt.Create();
        try
        {
            var prevFilter = src.filterMode;
            src.filterMode = FilterMode.Bilinear;
            Graphics.Blit(src, rt);
            src.filterMode = prevFilter;

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
            try { rt.Release(); } catch { }
            Destroy(rt);
        }
    }

    private static float HalfToFloat(ushort h)
    {
        uint sign = (uint)(h >> 15) & 1u;
        uint exp = (uint)(h >> 10) & 0x1Fu;
        uint mant = (uint)h & 0x3FFu;
        if (exp == 0)
        {
            if (mant == 0) return sign == 0 ? 0f : -0f;
            float v = mant / 1024f;
            v *= Mathf.Pow(2f, -14f);
            return sign == 0 ? v : -v;
        }
        if (exp == 31)
        {
            if (mant == 0) return sign == 0 ? float.PositiveInfinity : float.NegativeInfinity;
            return float.NaN;
        }
        float value = 1f + mant / 1024f;
        value *= Mathf.Pow(2f, (int)exp - 15);
        return sign == 0 ? value : -value;
    }

    private static async UniTask<string> PrepareModelDirAsync(string modelName, CancellationToken ct)
    {
        Aexis.Samples.AexisSampleStreamingAssets.TryResolveDirectoryPath(Path.Combine("GFPGAN", "models"), out var streamingDir);
        var pEncoderParam = string.IsNullOrWhiteSpace(streamingDir) ? null : Path.Combine(streamingDir, "encoder.param");
        var pEncoderBin = string.IsNullOrWhiteSpace(streamingDir) ? null : Path.Combine(streamingDir, "encoder.bin");
        var pStyleBin = string.IsNullOrWhiteSpace(streamingDir) ? null : Path.Combine(streamingDir, "style.bin");

        if (Directory.Exists(streamingDir) && File.Exists(pEncoderParam) && File.Exists(pEncoderBin) && File.Exists(pStyleBin))
            return streamingDir;

        var cacheDir = Path.Combine(Application.persistentDataPath, "GFPGAN", "models");
        try { Directory.CreateDirectory(cacheDir); } catch { }

        await EnsureFileAsync(cacheDir, "encoder.param", ct);
        await EnsureFileAsync(cacheDir, "encoder.bin", ct);
        await EnsureFileAsync(cacheDir, "style.bin", ct);

        return cacheDir;
    }

    private static async UniTask EnsureFileAsync(string cacheDir, string fileName, CancellationToken ct)
    {
        var dst = Path.Combine(cacheDir, fileName);
        if (File.Exists(dst))
            return;

        var bytes = await LoadStreamingAssetBytesAsync(Path.Combine("GFPGAN", "models", fileName), ct);
        if (bytes == null || bytes.Length == 0)
            return;
        try { await File.WriteAllBytesAsync(dst, bytes, ct); } catch { }
    }

    private static async UniTask<byte[]> LoadStreamingAssetBytesAsync(string relativePath, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(relativePath))
            return null;
        try { return await Aexis.Samples.AexisSampleStreamingAssets.ReadBytesAsync(relativePath, ct); }
        catch { return null; }
    }
}

public struct GfpganResult
{
    public Texture2D texture;
    public string error;
    public long elapsedMs;
}

internal static class GfpganNcnnNative
{
#if !(UNITY_IOS && !UNITY_EDITOR)
    private const string DllName = "realesrgan_unity";

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr Gfpgan_Create(
        [MarshalAs(UnmanagedType.LPStr)] string modelDir,
        [MarshalAs(UnmanagedType.LPStr)] string modelName,
        int gpuId,
        out IntPtr ctx);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void Gfpgan_Destroy(IntPtr ctx);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr Gfpgan_ProcessRgba(
        IntPtr ctx,
        byte[] rgba,
        int w,
        int h,
        byte[] outRgba,
        int outW,
        int outH);
#else
    public static IntPtr Gfpgan_Create(string modelDir, string modelName, int gpuId, out IntPtr ctx)
    {
        ctx = IntPtr.Zero;
        return IntPtr.Zero;
    }

    public static void Gfpgan_Destroy(IntPtr ctx)
    {
    }

    public static IntPtr Gfpgan_ProcessRgba(
        IntPtr ctx,
        byte[] rgba,
        int w,
        int h,
        byte[] outRgba,
        int outW,
        int outH)
    {
        return IntPtr.Zero;
    }
#endif

    public static string Utf8ToString(IntPtr utf8)
    {
        if (utf8 == IntPtr.Zero)
            return "";
        try
        {
            return Marshal.PtrToStringUTF8(utf8) ?? "";
        }
        catch
        {
            return "";
        }
    }
}
