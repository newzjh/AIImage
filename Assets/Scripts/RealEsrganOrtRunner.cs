using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using Cysharp.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using UnityEngine;
using UnityEngine.Networking;

public sealed class RealEsrganOrtRunner : MonoBehaviour
{
    public bool enableRealEsrganOrt = true;
    public string modelOnnxFileName = "realesrgan-x4plus.onnx";
    public int maxInputLongSide = 2048;

    public event Action<float, string> ProgressChanged;

    private readonly object _sessionLock = new object();
    private InferenceSession _session;
    private string _loadedModelPath;

    public async UniTask<RealEsrganResult> ProcessAsync(Texture2D src, CancellationToken ct)
    {
        if (!enableRealEsrganOrt)
            return new RealEsrganResult { error = "Real-ESRGAN(ORT) disabled" };
        if (src == null)
            return default;

        var originalW = src.width;
        var originalH = src.height;

        ReportProgress(0f, "准备输入");
        await UniTask.Yield();

        var maxSide = Mathf.Max(originalW, originalH);
        var limit = Mathf.Max(256, maxInputLongSide);

        Texture2D scaledInput = null;
        try
        {
            var inputTex = src;
            if (maxSide > limit)
            {
                var scaleDown = (float)limit / maxSide;
                var sw = Mathf.Max(1, Mathf.RoundToInt(originalW * scaleDown));
                var sh = Mathf.Max(1, Mathf.RoundToInt(originalH * scaleDown));
                scaledInput = ResizeTextureBilinear(src, sw, sh);
                if (scaledInput == null)
                    return new RealEsrganResult { error = "Failed to scale down input image" };
                inputTex = scaledInput;
            }

            await EnsureSessionAsync(ct);
            if (_session == null)
                return new RealEsrganResult { error = "Real-ESRGAN(ORT) session init failed" };

            ReportProgress(0.08f, "预处理");
            await UniTask.Yield();

            var rgba = inputTex.GetRawTextureData<byte>().ToArray();
            var w = inputTex.width;
            var h = inputTex.height;

            byte[] outRgba = null;
            int outW = 0;
            int outH = 0;

            ReportProgress(0.12f, "推理中…");
            using var pseudoCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            var pseudo = PseudoProgressAsync(pseudoCts.Token);

            await UniTask.SwitchToThreadPool();
            try
            {
                (outRgba, outW, outH) = RunModelRgbToRgb(_session, rgba, w, h);
            }
            finally
            {
                try { pseudoCts.Cancel(); } catch { }
                await UniTask.SwitchToMainThread();
                try { await pseudo; } catch { }
            }

            if (outRgba == null || outRgba.Length == 0 || outW <= 0 || outH <= 0)
                return new RealEsrganResult { error = "Real-ESRGAN(ORT) output is empty" };

            ReportProgress(0.92f, "后处理");

            var outTex = new Texture2D(outW, outH, TextureFormat.RGBA32, false, false);
            outTex.LoadRawTextureData(outRgba);
            outTex.Apply(false, false);

            Texture2D finalTex = outTex;
            if (finalTex.width != originalW || finalTex.height != originalH)
            {
                ReportProgress(0.96f, "回缩放到原分辨率");
                var resized = ResizeTextureBilinear(finalTex, originalW, originalH);
                Destroy(finalTex);
                finalTex = resized;
                if (finalTex == null)
                    return new RealEsrganResult { error = "Failed to resize output back to original resolution" };
            }

            finalTex.wrapMode = TextureWrapMode.Clamp;
            finalTex.filterMode = FilterMode.Bilinear;
            finalTex.name = "RealESRGAN_ORT";

            ReportProgress(1f, "完成");
            return new RealEsrganResult { texture = finalTex };
        }
        finally
        {
            if (scaledInput != null)
                Destroy(scaledInput);
        }
    }

    private async UniTask EnsureSessionAsync(CancellationToken ct)
    {
        var modelPath = ResolveModelPath();
        if (string.IsNullOrWhiteSpace(modelPath))
            return;

        lock (_sessionLock)
        {
            if (_session != null && string.Equals(_loadedModelPath, modelPath, StringComparison.OrdinalIgnoreCase))
                return;
        }

        ReportProgress(0.02f, "加载模型");
        var bytes = await LoadModelBytesAsync(modelPath, ct);
        if (bytes == null || bytes.Length == 0)
        {
            ReportProgress(0.02f, "模型缺失");
            return;
        }

        InferenceSession newSession = null;
        try
        {
            var so = new SessionOptions();
            newSession = new InferenceSession(bytes, so);
        }
        catch
        {
            newSession?.Dispose();
            newSession = null;
        }

        lock (_sessionLock)
        {
            _session?.Dispose();
            _session = newSession;
            _loadedModelPath = modelPath;
        }
    }

    private string ResolveModelPath()
    {
        if (string.IsNullOrWhiteSpace(modelOnnxFileName))
            return null;
        var p = Path.Combine(Application.streamingAssetsPath, "RealESRGAN", "models", modelOnnxFileName);
        return p;
    }

    private static async UniTask<byte[]> LoadModelBytesAsync(string path, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(path))
            return null;

        var isLikelyFile = path.IndexOf("://", StringComparison.Ordinal) < 0 && !path.StartsWith("jar:", StringComparison.OrdinalIgnoreCase);
        if (isLikelyFile && File.Exists(path))
        {
            try { return await File.ReadAllBytesAsync(path, ct); } catch { return null; }
        }

        var url = path;
        if (isLikelyFile)
            url = "file:///" + path.Replace("\\", "/");

        using var req = UnityWebRequest.Get(url);
        req.downloadHandler = new DownloadHandlerBuffer();
        var op = req.SendWebRequest();
        while (!op.isDone)
        {
            ct.ThrowIfCancellationRequested();
            await UniTask.Delay(25, cancellationToken: ct);
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

    private static (byte[] rgba, int w, int h) RunModelRgbToRgb(InferenceSession session, byte[] rgba, int w, int h)
    {
        if (session == null || rgba == null || rgba.Length < w * h * 4)
            return default;

        var hw = w * h;
        var input = new float[3 * hw];
        for (var i = 0; i < hw; i++)
        {
            var bi = i * 4;
            input[i] = rgba[bi] / 255f;
            input[hw + i] = rgba[bi + 1] / 255f;
            input[2 * hw + i] = rgba[bi + 2] / 255f;
        }

        using var inputOrt = OrtValue.CreateTensorValueFromMemory(input, new long[] { 1, 3, h, w });
        using var ro = new RunOptions();

        var inputs = new List<OrtValue>(1) { inputOrt };
        using var outs = session.Run(ro, session.InputNames, inputs, session.OutputNames);
        if (outs == null || outs.Count == 0)
            return default;

        var shapeInfo = outs[0].GetTensorTypeAndShape();
        var dims = shapeInfo.Shape;
        if (dims.Length != 4)
            return default;
        var oh = (int)dims[2];
        var ow = (int)dims[3];
        if (ow <= 0 || oh <= 0)
            return default;

        var outSpan = outs[0].GetTensorDataAsSpan<float>();
        var outHw = ow * oh;
        if (outSpan.Length < 3 * outHw)
            return default;

        var outBytes = new byte[outHw * 4];
        for (var i = 0; i < outHw; i++)
        {
            var r = Mathf.Clamp01(outSpan[i]);
            var g = Mathf.Clamp01(outSpan[outHw + i]);
            var b = Mathf.Clamp01(outSpan[2 * outHw + i]);
            var bi = i * 4;
            outBytes[bi] = (byte)Mathf.RoundToInt(r * 255f);
            outBytes[bi + 1] = (byte)Mathf.RoundToInt(g * 255f);
            outBytes[bi + 2] = (byte)Mathf.RoundToInt(b * 255f);
            outBytes[bi + 3] = 255;
        }

        return (outBytes, ow, oh);
    }

    private async UniTask PseudoProgressAsync(CancellationToken ct)
    {
        var p = 0.12f;
        while (!ct.IsCancellationRequested)
        {
            p = Mathf.Min(0.90f, p + 0.004f);
            ReportProgress(p, "推理中…");
            await UniTask.Delay(120, cancellationToken: ct);
        }
    }

    private void ReportProgress(float progress01, string text)
    {
        progress01 = Mathf.Clamp01(progress01);
        try { ProgressChanged?.Invoke(progress01, text ?? ""); } catch { }
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

    private void OnDestroy()
    {
        lock (_sessionLock)
        {
            _session?.Dispose();
            _session = null;
            _loadedModelPath = null;
        }
    }
}

