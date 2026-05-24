using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using Cysharp.Threading.Tasks;
using UnityEngine;
using UnityEngine.Networking;

public sealed class RealEsrganNcnnNativeRunner : MonoBehaviour
{
    public bool enableRealEsrganNative = true;
    public string modelName = "realesrgan-x4plus";
    public int scale = 2;
    public int tileSize = 0;
    public int gpuId = -1;
    public int prepadding = 10;
    public bool ttaMode = false;
    public bool allowCpuFallback = false;

    public event Action<float, string> ProgressChanged;

    private IntPtr _ctx = IntPtr.Zero;
    private GCHandle _progressHandle;
    private RealEsrganNcnnNative.ProgressCallback _progressCb;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.BeforeSceneLoad)]
    private static void BootstrapNativeDllSearchPath()
    {
#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
        try
        {
            var flags = LOAD_LIBRARY_SEARCH_APPLICATION_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32 | LOAD_LIBRARY_SEARCH_USER_DIRS;
            SetDefaultDllDirectories(flags);

            var exeDir = Environment.CurrentDirectory;
            if (!string.IsNullOrWhiteSpace(exeDir))
                AddDllDirectory(exeDir);

            var pluginsDir = Path.Combine(Application.dataPath, "Plugins");
            if (Directory.Exists(pluginsDir))
                AddDllDirectory(pluginsDir);

            var system32 = Environment.GetFolderPath(Environment.SpecialFolder.System);
            if (!string.IsNullOrWhiteSpace(system32))
            {
                LoadLibraryW(Path.Combine(system32, "VCRUNTIME140.dll"));
                LoadLibraryW(Path.Combine(system32, "VCRUNTIME140_1.dll"));
                LoadLibraryW(Path.Combine(system32, "MSVCP140.dll"));
                LoadLibraryW(Path.Combine(system32, "CONCRT140.dll"));
            }
        }
        catch
        {
        }
#endif
    }

    public async UniTask<RealEsrganResult> ProcessAsync(Texture2D src, CancellationToken ct)
    {
        if (!enableRealEsrganNative)
            return new RealEsrganResult { error = "Real-ESRGAN(原生) disabled" };
        if (src == null)
            return default;

        var traceId = Guid.NewGuid().ToString("N");
        var originalW = src.width;
        var originalH = src.height;
        var modelDir = await PrepareModelDirAsync(modelName, ct);
        if (string.IsNullOrWhiteSpace(modelDir) || !Directory.Exists(modelDir))
            return new RealEsrganResult { error = "models目录不可用: " + (modelDir ?? "") };

        try
        {
            EnsureInit(modelDir);
        }
        catch (Exception e)
        {
            await ReportDbgAsync(
                "B",
                "realesrgan.native.init.exception",
                "[DEBUG] EnsureInit exception",
                "{\"msg\":\"" + EscapeJson(e.Message) + "\",\"modelDir\":\"" + EscapeJson(modelDir) + "\",\"modelName\":\"" + EscapeJson(modelName) + "\"}",
                traceId,
                ct);
            return new RealEsrganResult { error = e.Message };
        }

        ReportProgress(0f, "准备输入");
        await UniTask.Yield();

        var desiredScale = Mathf.Clamp(scale, 2, 4);
        var modelFactor = InferModelFactor(modelName);
        var runFactor = modelFactor;
        const int maxInputLongSide = 2048;

        Texture2D scaledInput = null;
        Texture2D rgbaInput = null;
        try
        {
            var inputTex = src;
            var maxSide = Mathf.Max(originalW, originalH);
            if (maxSide > maxInputLongSide)
            {
                ReportProgress(0.02f, "缩小到2k以内");
                var scaleDown = (float)maxInputLongSide / maxSide;
                var sw = Mathf.Max(1, Mathf.RoundToInt(originalW * scaleDown));
                var sh = Mathf.Max(1, Mathf.RoundToInt(originalH * scaleDown));
                scaledInput = ResizeTextureBilinear(src, sw, sh);
                if (scaledInput == null)
                    return new RealEsrganResult { error = "缩小输入失败" };
                inputTex = scaledInput;
            }

            var w = inputTex.width;
            var h = inputTex.height;
            var rgba = GetRgbaBytes(inputTex, out rgbaInput);
            if (rgba == null || rgba.Length != w * h * 4)
                return new RealEsrganResult { error = "读取RGBA像素失败" };

            var outW = w * runFactor;
            var outH = h * runFactor;
            var outRgba = new byte[outW * outH * 4];

            await ReportDbgAsync(
                "A",
                "realesrgan.native.process.enter",
                "[DEBUG] ProcessAsync enter",
                "{\"w\":" + w + ",\"h\":" + h + ",\"outW\":" + outW + ",\"outH\":" + outH + ",\"originalW\":" + originalW + ",\"originalH\":" + originalH + ",\"desiredScale\":" + desiredScale + ",\"runFactor\":" + runFactor + ",\"tileSize\":" + tileSize + ",\"gpuId\":" + gpuId + ",\"prepadding\":" + prepadding + ",\"ttaMode\":" + (ttaMode ? 1 : 0) + ",\"modelDir\":\"" + EscapeJson(modelDir) + "\",\"modelName\":\"" + EscapeJson(modelName) + "\"}",
                traceId,
                ct);

            await ReportDbgAsync(
                "A",
                "realesrgan.native.call.begin",
                "[DEBUG] calling native Realesrgan_ProcessRgba",
                "{\"ctx\":\"" + (_ctx != IntPtr.Zero ? "nonzero" : "zero") + "\"}",
                traceId,
                ct);

            var errPtr = await UniTask.RunOnThreadPool(() =>
            {
                ct.ThrowIfCancellationRequested();
                ReportProgress(0.05f, "推理中…");

                var r = RealEsrganNcnnNative.Realesrgan_ProcessRgba(
                    _ctx,
                    rgba, w, h,
                    outRgba, outW, outH,
                    tileSize,
                    runFactor);

                return r;
            }, cancellationToken: ct);

            var err = RealEsrganNcnnNative.Utf8ToString(errPtr);
            if (!string.IsNullOrWhiteSpace(err))
            {
                await ReportDbgAsync(
                    "C",
                    "realesrgan.native.error",
                    "[DEBUG] native returned error",
                    "{\"err\":\"" + EscapeJson(err) + "\",\"w\":" + w + ",\"h\":" + h + ",\"outW\":" + outW + ",\"outH\":" + outH + ",\"originalW\":" + originalW + ",\"originalH\":" + originalH + ",\"runFactor\":" + runFactor + ",\"tileSize\":" + tileSize + ",\"gpuId\":" + gpuId + ",\"prepadding\":" + prepadding + ",\"ttaMode\":" + (ttaMode ? 1 : 0) + ",\"modelName\":\"" + EscapeJson(modelName) + "\"}",
                    traceId,
                    ct);
                ResetContext();
                return new RealEsrganResult { error = err };
            }

            await ReportDbgAsync(
                "A",
                "realesrgan.native.call.ok",
                "[DEBUG] native returned ok",
                "{\"ctx\":\"" + (_ctx != IntPtr.Zero ? "nonzero" : "zero") + "\"}",
                traceId,
                ct);

            ReportProgress(0.92f, "生成纹理");
            await UniTask.SwitchToMainThread();

            var tex = new Texture2D(outW, outH, TextureFormat.RGBA32, false, false);
            tex.LoadRawTextureData(outRgba);
            tex.Apply(false, false);
            tex.wrapMode = TextureWrapMode.Clamp;
            tex.filterMode = FilterMode.Bilinear;
            tex.name = "RealESRGAN_NCNN_" + runFactor + "x";

            Texture2D finalTex = tex;
            if (desiredScale != runFactor)
            {
                ReportProgress(0.94f, "回缩放到目标倍率");
                var resized = ResizeTextureBilinear(finalTex, w * desiredScale, h * desiredScale);
                Destroy(finalTex);
                finalTex = resized;
                if (finalTex == null)
                    return new RealEsrganResult { error = "回缩放失败" };
            }

            if (finalTex.width != originalW * desiredScale || finalTex.height != originalH * desiredScale)
            {
                ReportProgress(0.98f, "回缩放到原分辨率");
                var resized = ResizeTextureBilinear(finalTex, originalW * desiredScale, originalH * desiredScale);
                Destroy(finalTex);
                finalTex = resized;
                if (finalTex == null)
                    return new RealEsrganResult { error = "回缩放失败" };
            }

            finalTex.name = "RealESRGAN_NCNN_" + desiredScale + "x";
            ReportProgress(1f, "完成");
            return new RealEsrganResult { texture = finalTex };
        }
        finally
        {
            if (scaledInput != null)
                Destroy(scaledInput);
            if (rgbaInput != null)
                Destroy(rgbaInput);
        }
    }

    private static byte[] GetRgbaBytes(Texture2D src, out Texture2D readableCopy)
    {
        readableCopy = null;
        if (src == null)
            return null;

        Texture2D tex = src;
        if (!src.isReadable)
        {
            readableCopy = RenderToReadableRgba32(src, src.width, src.height);
            tex = readableCopy;
        }
        else if (src.format != TextureFormat.RGBA32)
        {
            readableCopy = new Texture2D(src.width, src.height, TextureFormat.RGBA32, false, false);
            readableCopy.SetPixels32(src.GetPixels32());
            readableCopy.Apply(false, false);
            tex = readableCopy;
        }

        if (tex == null)
            return null;

        var pixels = tex.GetPixels32();
        var bytes = new byte[pixels.Length * 4];
        var bi = 0;
        for (var i = 0; i < pixels.Length; i++)
        {
            var p = pixels[i];
            bytes[bi++] = p.r;
            bytes[bi++] = p.g;
            bytes[bi++] = p.b;
            bytes[bi++] = p.a;
        }
        return bytes;
    }

    private static Texture2D RenderToReadableRgba32(Texture src, int w, int h)
    {
        var prev = RenderTexture.active;
        var rt = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        rt.enableRandomWrite = false;
        rt.Create();
        try
        {
            Graphics.Blit(src, rt);
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
            UnityEngine.Object.Destroy(rt);
        }
    }

    private void EnsureInit(string modelDir)
    {
        if (_ctx != IntPtr.Zero)
            return;

#if !UNITY_EDITOR
        allowCpuFallback = false;
#endif

        _progressCb = OnNativeProgress;
        _progressHandle = GCHandle.Alloc(this);
        var user = GCHandle.ToIntPtr(_progressHandle);

        var modelFactor = InferModelFactor(modelName);
        ReportDbgAsync(
            "B",
            "realesrgan.native.create.begin",
            "[DEBUG] Realesrgan_Create begin",
            "{\"modelDir\":\"" + EscapeJson(modelDir) + "\",\"modelName\":\"" + EscapeJson(modelName) + "\",\"modelFactor\":" + modelFactor + ",\"gpuId\":" + gpuId + ",\"prepadding\":" + Mathf.Max(1, prepadding) + ",\"ttaMode\":" + (ttaMode ? 1 : 0) + "}",
            "",
            CancellationToken.None).Forget();

        var err = TryCreate(modelDir, modelFactor, gpuId, user, out _ctx);
        if (!string.IsNullOrWhiteSpace(err) && allowCpuFallback)
        {
            var shouldFallback =
                err.IndexOf("native exception in Realesrgan_Create", StringComparison.OrdinalIgnoreCase) >= 0 ||
                err.IndexOf("vulkan", StringComparison.OrdinalIgnoreCase) >= 0;

            if (shouldFallback)
            {
                ReportDbgAsync(
                    "B",
                    "realesrgan.native.create.fallback_cpu",
                    "[DEBUG] fallback to CPU mode (gpuId=-2)",
                    "{\"prevErr\":\"" + EscapeJson(err) + "\"}",
                    "",
                    CancellationToken.None).Forget();

                ReleaseProgressHandle();
                _progressHandle = GCHandle.Alloc(this);
                user = GCHandle.ToIntPtr(_progressHandle);
                err = TryCreate(modelDir, modelFactor, -2, user, out _ctx);
            }
        }

        if (!string.IsNullOrWhiteSpace(err))
        {
            ReleaseProgressHandle();
            throw new InvalidOperationException(err);
        }
        if (_ctx == IntPtr.Zero)
        {
            ReleaseProgressHandle();
            throw new InvalidOperationException("创建Real-ESRGAN(原生)上下文失败");
        }
    }

    private string TryCreate(string modelDir, int modelFactor, int createGpuId, IntPtr user, out IntPtr ctx)
    {
        ctx = IntPtr.Zero;
        var errPtr = RealEsrganNcnnNative.Realesrgan_Create(
            modelDir,
            modelName,
            modelFactor,
            createGpuId,
            Mathf.Max(1, prepadding),
            ttaMode ? 1 : 0,
            user,
            _progressCb,
            out ctx);

        var err = RealEsrganNcnnNative.Utf8ToString(errPtr);
        ReportDbgAsync(
            "B",
            "realesrgan.native.create.end",
            "[DEBUG] Realesrgan_Create end",
            "{\"err\":\"" + EscapeJson(err) + "\",\"ctx\":\"" + (ctx != IntPtr.Zero ? "nonzero" : "zero") + "\",\"gpuId\":" + createGpuId + "}",
            "",
            CancellationToken.None).Forget();
        return err;
    }

    private void ReleaseProgressHandle()
    {
        try
        {
            if (_progressHandle.IsAllocated)
                _progressHandle.Free();
        }
        catch
        {
        }
        _progressHandle = default;
    }

#if UNITY_STANDALONE_WIN && !UNITY_EDITOR
    private const uint LOAD_LIBRARY_SEARCH_APPLICATION_DIR = 0x00000200;
    private const uint LOAD_LIBRARY_SEARCH_SYSTEM32 = 0x00000800;
    private const uint LOAD_LIBRARY_SEARCH_USER_DIRS = 0x00000400;

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr LoadLibraryW(string lpFileName);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool SetDefaultDllDirectories(uint DirectoryFlags);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr AddDllDirectory(string NewDirectory);
#endif

    [AOT.MonoPInvokeCallback(typeof(RealEsrganNcnnNative.ProgressCallback))]
    private static void OnNativeProgress(IntPtr user, float progress01, IntPtr utf8Message)
    {
        try
        {
            if (user == IntPtr.Zero)
                return;
            var handle = GCHandle.FromIntPtr(user);
            if (!(handle.Target is RealEsrganNcnnNativeRunner runner))
                return;
            var msg = RealEsrganNcnnNative.Utf8ToString(utf8Message);
            runner.ReportProgress(progress01, msg);
        }
        catch
        {
        }
    }

    private void ReportProgress(float progress01, string text)
    {
        progress01 = Mathf.Clamp01(progress01);
        try { ProgressChanged?.Invoke(progress01, text ?? ""); } catch { }
    }

    private void OnDestroy()
    {
        ReportDbgAsync(
            "E",
            "realesrgan.native.runner.destroy",
            "[DEBUG] runner OnDestroy",
            "{\"ctx\":\"" + (_ctx != IntPtr.Zero ? "nonzero" : "zero") + "\"}",
            "",
            CancellationToken.None).Forget();

        if (_ctx != IntPtr.Zero)
        {
            try { RealEsrganNcnnNative.Realesrgan_Destroy(_ctx); } catch { }
            _ctx = IntPtr.Zero;
        }
        ReleaseProgressHandle();
    }

    private void ResetContext()
    {
        if (_ctx != IntPtr.Zero)
        {
            ReportDbgAsync(
                "D",
                "realesrgan.native.ctx.reset",
                "[DEBUG] ResetContext destroying native ctx",
                "{\"ctx\":\"nonzero\"}",
                "",
                CancellationToken.None).Forget();
            try { RealEsrganNcnnNative.Realesrgan_Destroy(_ctx); } catch { }
            _ctx = IntPtr.Zero;
        }
        ReleaseProgressHandle();
    }

    private static int InferModelFactor(string name)
    {
        if (string.IsNullOrWhiteSpace(name))
            return 4;
        var s = name;
        for (var i = 0; i < s.Length - 1; i++)
        {
            if ((s[i] == 'x' || s[i] == 'X') && char.IsDigit(s[i + 1]))
                return Mathf.Clamp(s[i + 1] - '0', 2, 4);
            if (char.IsDigit(s[i]) && (s[i + 1] == 'x' || s[i + 1] == 'X'))
                return Mathf.Clamp(s[i] - '0', 2, 4);
        }
        return 4;
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

    private static async UniTask<string> PrepareModelDirAsync(string modelName, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(modelName))
            return null;

        var streamingDir = Path.Combine(Application.streamingAssetsPath, "RealESRGAN", "models");
        var paramName = modelName + ".param";
        var binName = modelName + ".bin";
        var pParam = Path.Combine(streamingDir, paramName);
        var pBin = Path.Combine(streamingDir, binName);

        if (Directory.Exists(streamingDir) && File.Exists(pParam) && File.Exists(pBin))
            return streamingDir;

        var cacheDir = Path.Combine(Application.persistentDataPath, "RealESRGAN", "models");
        try { Directory.CreateDirectory(cacheDir); } catch { }

        var dstParam = Path.Combine(cacheDir, paramName);
        var dstBin = Path.Combine(cacheDir, binName);

        if (!File.Exists(dstParam))
        {
            var bytes = await LoadStreamingAssetBytesAsync(Path.Combine("RealESRGAN", "models", paramName), ct);
            if (bytes == null || bytes.Length == 0) return cacheDir;
            try { await File.WriteAllBytesAsync(dstParam, bytes, ct); } catch { return cacheDir; }
        }

        if (!File.Exists(dstBin))
        {
            var bytes = await LoadStreamingAssetBytesAsync(Path.Combine("RealESRGAN", "models", binName), ct);
            if (bytes == null || bytes.Length == 0) return cacheDir;
            try { await File.WriteAllBytesAsync(dstBin, bytes, ct); } catch { return cacheDir; }
        }

        return cacheDir;
    }

    private static async UniTask<byte[]> LoadStreamingAssetBytesAsync(string relativePath, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(relativePath))
            return null;
        var url = Path.Combine(Application.streamingAssetsPath, relativePath).Replace("\\", "/");

        // Android: streamingAssetsPath can be jar:file:///...!/assets
        // Desktop/iOS: file path works with file:///
        if (url.IndexOf("://", StringComparison.Ordinal) < 0)
            url = "file:///" + url;

        using var req = UnityWebRequest.Get(url);
        req.downloadHandler = new DownloadHandlerBuffer();
        var op = req.SendWebRequest();
        while (!op.isDone)
        {
            ct.ThrowIfCancellationRequested();
            await UniTask.Delay(20, cancellationToken: ct);
        }

#if UNITY_2020_1_OR_NEWER
        if (req.result != UnityWebRequest.Result.Success)
            return null;
#else
        if (req.isNetworkError || req.isHttpError)
            return null;
#endif
        return req.downloadHandler.data;
    }

    #region debug-point realesrgan-extract-crash-report
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

            var logPath = Path.Combine(Application.persistentDataPath, "trae-debug-log-windows-player-native-crash.ndjson");
            var dir = Path.GetDirectoryName(logPath);
            if (!string.IsNullOrWhiteSpace(dir))
            {
                try { Directory.CreateDirectory(dir); } catch { }
            }
            if (string.IsNullOrWhiteSpace(hypothesisId)) hypothesisId = "A";
            if (string.IsNullOrWhiteSpace(dataJson)) dataJson = "{}";
            var payload =
                "{\"sessionId\":\"" + EscapeJson(_dbgSessionId) + "\"" +
                ",\"runId\":\"pre-fix\"" +
                ",\"hypothesisId\":\"" + EscapeJson(hypothesisId) + "\"" +
                ",\"location\":\"RealEsrganNcnnNativeRunner\"" +
                ",\"msg\":\"" + EscapeJson(msg) + "\"" +
                ",\"type\":\"" + EscapeJson(type) + "\"" +
                ",\"traceId\":\"" + EscapeJson(traceId ?? "") + "\"" +
                ",\"ts\":" + DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() +
                ",\"data\":" + dataJson +
                "}";

            var line = payload + "\n";
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
}

internal static class RealEsrganNcnnNative
{
#if UNITY_IOS && !UNITY_EDITOR
    private const string DllName = "__Internal";
#else
    private const string DllName = "realesrgan_unity";
#endif

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void ProgressCallback(IntPtr user, float progress01, IntPtr utf8Message);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr Realesrgan_Create(
        [MarshalAs(UnmanagedType.LPStr)] string modelDir,
        [MarshalAs(UnmanagedType.LPStr)] string modelName,
        int modelFactor,
        int gpuId,
        int prepadding,
        int ttaMode,
        IntPtr user,
        ProgressCallback progress,
        out IntPtr ctx);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern void Realesrgan_Destroy(IntPtr ctx);

    [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
    public static extern IntPtr Realesrgan_ProcessRgba(
        IntPtr ctx,
        byte[] rgba,
        int w,
        int h,
        byte[] outRgba,
        int outW,
        int outH,
        int tileSize,
        int scale);

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
