using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using Aexis.Samples.Async;
using UnityEngine;
using UnityEngine.Rendering;
using Debug = UnityEngine.Debug;

public sealed class FaceMaskGenerator : MonoBehaviour
{
    public Texture2D currentImageFaceMask;
    public Texture2D maleFaceMask;
    public Texture2D femaleFaceMask;

    private ComputeShader _cs;
    private readonly Dictionary<string, int> _kernelIds = new Dictionary<string, int>(StringComparer.Ordinal);

    public ComputeShader GetComputeShader()
    {
        if (_cs == null)
            _cs = Resources.Load<ComputeShader>("FaceMask");
        return _cs;
    }

    public int GetKernel(string kernelName)
    {
        if (string.IsNullOrWhiteSpace(kernelName))
            return -1;
        if (_kernelIds.TryGetValue(kernelName, out var id))
            return id;
        var cs = GetComputeShader();
        if (cs == null)
            return -1;
        try
        {
            id = cs.FindKernel(kernelName);
        }
        catch
        {
            id = -1;
        }
        _kernelIds[kernelName] = id;
        return id;
    }

    public void ClearCurrentMask()
    {
        if (currentImageFaceMask != null)
            Destroy(currentImageFaceMask);
        currentImageFaceMask = null;
    }

    public void ClearMaleMask()
    {
        if (maleFaceMask != null)
            Destroy(maleFaceMask);
        maleFaceMask = null;
    }

    public void ClearFemaleMask()
    {
        if (femaleFaceMask != null)
            Destroy(femaleFaceMask);
        femaleFaceMask = null;
    }

    public void ClearAllMasks()
    {
        ClearCurrentMask();
        ClearMaleMask();
        ClearFemaleMask();
    }

    public async UniTask<FaceMaskTextureResult> GenerateForCurrentAsync(Texture2D src, bool dumpDebug, CancellationToken ct)
    {
        ClearCurrentMask();
        var r = await GenerateMaskTextureAsync(src, dumpDebug, ct);
        if (!string.IsNullOrWhiteSpace(r.error) || r.mask == null)
            return r;
        currentImageFaceMask = r.mask;
        return r;
    }

    public async UniTask<FaceMaskTextureResult> GenerateForMaleAsync(Texture2D src, bool dumpDebug, CancellationToken ct)
    {
        ClearMaleMask();
        var r = await GenerateMaskTextureAsync(src, dumpDebug, ct);
        if (!string.IsNullOrWhiteSpace(r.error) || r.mask == null)
            return r;
        maleFaceMask = r.mask;
        return r;
    }

    public async UniTask<FaceMaskTextureResult> GenerateForFemaleAsync(Texture2D src, bool dumpDebug, CancellationToken ct)
    {
        ClearFemaleMask();
        var r = await GenerateMaskTextureAsync(src, dumpDebug, ct);
        if (!string.IsNullOrWhiteSpace(r.error) || r.mask == null)
            return r;
        femaleFaceMask = r.mask;
        return r;
    }

    private async UniTask<FaceMaskTextureResult> GenerateMaskTextureAsync(Texture2D src, bool dumpDebug, CancellationToken ct)
    {
        if (src == null)
            return default;

        var cs = GetComputeShader();
        if (cs == null)
            return new FaceMaskTextureResult { error = "FaceMask.compute not found" };

        var kRaw = GetKernel("FaceMaskRaw");
        var kDown2 = GetKernel("MaskDown2");
        var kReset = GetKernel("ResetFaceStats");
        var kStatsToParams = GetKernel("FaceStatsToParams");
        var kStats = GetKernel("FaceMaskStats");
        var kApply = GetKernel("FaceMaskApply");
        var kBoxH = GetKernel("BoxFilterH");
        var kBoxV = GetKernel("BoxFilterV");
        var kDebug = GetKernel("DebugVisScalar");
        if (kRaw < 0 || kDown2 < 0 || kReset < 0 || kStatsToParams < 0 || kStats < 0 || kApply < 0 || kBoxH < 0 || kBoxV < 0 || kDebug < 0)
            return new FaceMaskTextureResult { error = "FaceMask kernels not found" };

        RenderTexture faceMaskRaw = null;
        RenderTexture faceMask = null;
        RenderTexture faceMaskD1 = null;
        RenderTexture faceMaskD2 = null;
        RenderTexture faceMaskD3 = null;
        RenderTexture faceMaskD4 = null;
        RenderTexture faceMaskD5 = null;
        RenderTexture boxTmp = null;
        ComputeBuffer faceStats = null;
        ComputeBuffer faceParams = null;
        string dumpDir = null;
        Texture2D maskTex = null;

        try
        {
            ct.ThrowIfCancellationRequested();

            var w = src.width;
            var h = src.height;
            var gx = Mathf.CeilToInt(w / 8f);
            var gy = Mathf.CeilToInt(h / 8f);

            faceStats = new ComputeBuffer(7, sizeof(int), ComputeBufferType.Structured);
            faceParams = new ComputeBuffer(1, sizeof(float) * 4, ComputeBufferType.Structured);

            faceMaskRaw = NewRT(w, h, RenderTextureFormat.RHalf);
            faceMask = NewRT(w, h, RenderTextureFormat.RHalf);
            faceMaskD1 = NewRT(Mathf.Max(1, (w + 1) / 2), Mathf.Max(1, (h + 1) / 2), RenderTextureFormat.RHalf);
            faceMaskD2 = NewRT(Mathf.Max(1, (faceMaskD1.width + 1) / 2), Mathf.Max(1, (faceMaskD1.height + 1) / 2), RenderTextureFormat.RHalf);
            faceMaskD3 = NewRT(Mathf.Max(1, (faceMaskD2.width + 1) / 2), Mathf.Max(1, (faceMaskD2.height + 1) / 2), RenderTextureFormat.RHalf);
            faceMaskD4 = NewRT(Mathf.Max(1, (faceMaskD3.width + 1) / 2), Mathf.Max(1, (faceMaskD3.height + 1) / 2), RenderTextureFormat.RHalf);
            faceMaskD5 = NewRT(Mathf.Max(1, (faceMaskD4.width + 1) / 2), Mathf.Max(1, (faceMaskD4.height + 1) / 2), RenderTextureFormat.RHalf);
            boxTmp = NewRT(w, h, RenderTextureFormat.RHalf);

            cs.SetTexture(kRaw, "_Source", src);
            cs.SetTexture(kRaw, "_MaskOut", faceMaskRaw);
            cs.Dispatch(kRaw, gx, gy, 1);

            Downsample(cs, kDown2, faceMaskRaw, faceMaskD1);
            Downsample(cs, kDown2, faceMaskD1, faceMaskD2);
            Downsample(cs, kDown2, faceMaskD2, faceMaskD3);
            Downsample(cs, kDown2, faceMaskD3, faceMaskD4);
            Downsample(cs, kDown2, faceMaskD4, faceMaskD5);

            var statsScale = 32;
            cs.SetInt("_StatsScale", statsScale);

            cs.SetBuffer(kReset, "_FaceStats", faceStats);
            cs.Dispatch(kReset, 1, 1, 1);

            cs.SetInt("_UseGaussian", 0);
            cs.SetTexture(kStats, "_MaskIn", faceMaskD5);
            cs.SetBuffer(kStats, "_FaceStats", faceStats);
            cs.SetBuffer(kStats, "_FaceParams", faceParams);
            cs.Dispatch(kStats, Mathf.CeilToInt(faceMaskD5.width / 8f), Mathf.CeilToInt(faceMaskD5.height / 8f), 1);

            cs.SetBuffer(kStatsToParams, "_FaceStats", faceStats);
            cs.SetBuffer(kStatsToParams, "_FaceParams", faceParams);
            cs.Dispatch(kStatsToParams, 1, 1, 1);

            cs.SetBuffer(kReset, "_FaceStats", faceStats);
            cs.Dispatch(kReset, 1, 1, 1);

            cs.SetInt("_UseGaussian", 1);
            cs.SetTexture(kStats, "_MaskIn", faceMaskD5);
            cs.SetBuffer(kStats, "_FaceStats", faceStats);
            cs.SetBuffer(kStats, "_FaceParams", faceParams);
            cs.Dispatch(kStats, Mathf.CeilToInt(faceMaskD5.width / 8f), Mathf.CeilToInt(faceMaskD5.height / 8f), 1);

            cs.SetInt("_UseGaussian", 0);
            cs.SetTexture(kApply, "_MaskIn", faceMaskRaw);
            cs.SetBuffer(kApply, "_FaceStats", faceStats);
            cs.SetBuffer(kApply, "_FaceParams", faceParams);
            cs.SetInt("_StatsScale", statsScale);
            cs.SetTexture(kApply, "_Source", src);
            cs.SetTexture(kApply, "_MaskOut", faceMask);
            cs.Dispatch(kApply, gx, gy, 1);

            cs.SetInt("_BoxRadius", 2);
            BoxBlur(cs, kBoxH, kBoxV, faceMask, boxTmp, faceMask, gx, gy);

            if (dumpDebug)
            {
                dumpDir ??= CreateDumpDir();
                await DumpScalarAsync(cs, kDebug, dumpDir, faceMaskRaw, w, h, "faceMask_raw.png", ct);
                await DumpScalarAsync(cs, kDebug, dumpDir, faceMaskD1, faceMaskD1.width, faceMaskD1.height, "faceMask_d1.png", ct);
                await DumpScalarAsync(cs, kDebug, dumpDir, faceMaskD2, faceMaskD2.width, faceMaskD2.height, "faceMask_d2.png", ct);
                await DumpScalarAsync(cs, kDebug, dumpDir, faceMaskD3, faceMaskD3.width, faceMaskD3.height, "faceMask_d3.png", ct);
                await DumpScalarAsync(cs, kDebug, dumpDir, faceMaskD4, faceMaskD4.width, faceMaskD4.height, "faceMask_d4.png", ct);
                await DumpScalarAsync(cs, kDebug, dumpDir, faceMaskD5, faceMaskD5.width, faceMaskD5.height, "faceMask_d5.png", ct);
                await DumpScalarAsync(cs, kDebug, dumpDir, faceMask, w, h, "faceMask.png", ct);
                if (!string.IsNullOrWhiteSpace(dumpDir))
                    OpenFolderInShell(dumpDir);
            }

            maskTex = await ReadbackScalarTextureAsync(faceMask, w, h, ct);
            if (maskTex == null)
                return new FaceMaskTextureResult { error = "FaceMask readback failed", dumpDir = dumpDir };

            maskTex.wrapMode = TextureWrapMode.Clamp;
            maskTex.filterMode = FilterMode.Bilinear;
            maskTex.name = "FaceMask";

            return new FaceMaskTextureResult { mask = maskTex, dumpDir = dumpDir };
        }
        catch (OperationCanceledException)
        {
            if (maskTex != null)
                Destroy(maskTex);
            return default;
        }
        catch (Exception e)
        {
            Debug.LogException(e);
            if (maskTex != null)
                Destroy(maskTex);
            return new FaceMaskTextureResult { error = e.Message, dumpDir = dumpDir };
        }
        finally
        {
            SafeReleaseRT(faceMaskRaw);
            SafeReleaseRT(faceMask);
            SafeReleaseRT(faceMaskD1);
            SafeReleaseRT(faceMaskD2);
            SafeReleaseRT(faceMaskD3);
            SafeReleaseRT(faceMaskD4);
            SafeReleaseRT(faceMaskD5);
            SafeReleaseRT(boxTmp);
            try { faceStats?.Dispose(); } catch { }
            try { faceParams?.Dispose(); } catch { }
        }
    }

    private static RenderTexture NewRT(int w, int h, RenderTextureFormat fmt)
    {
        var rt = new RenderTexture(w, h, 0, fmt, RenderTextureReadWrite.Linear) { enableRandomWrite = true };
        rt.Create();
        return rt;
    }

    private static void SafeReleaseRT(RenderTexture rt)
    {
        if (rt == null)
            return;
        try { rt.Release(); } catch { }
        Destroy(rt);
    }

    private static void Downsample(ComputeShader cs, int kernel, RenderTexture input, RenderTexture output)
    {
        cs.SetTexture(kernel, "_ScalarIn", input);
        cs.SetTexture(kernel, "_ScalarOut", output);
        cs.Dispatch(kernel, Mathf.CeilToInt(output.width / 8f), Mathf.CeilToInt(output.height / 8f), 1);
    }

    private static void BoxBlur(ComputeShader cs, int kBoxH, int kBoxV, Texture input, RenderTexture tmp, RenderTexture output, int gx, int gy)
    {
        cs.SetTexture(kBoxH, "_ScalarIn", input);
        cs.SetTexture(kBoxH, "_BoxOut", tmp);
        cs.Dispatch(kBoxH, gx, gy, 1);
        cs.SetTexture(kBoxV, "_ScalarIn", tmp);
        cs.SetTexture(kBoxV, "_BoxOut", output);
        cs.Dispatch(kBoxV, gx, gy, 1);
    }

    private static async UniTask DumpScalarAsync(ComputeShader cs, int debugKernel, string dir, RenderTexture scalar, int w, int h, string fileName, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(dir) || scalar == null || cs == null || debugKernel < 0)
            return;

        var vis = NewRT(w, h, RenderTextureFormat.ARGB32);
        try
        {
            ct.ThrowIfCancellationRequested();
            cs.SetTexture(debugKernel, "_ScalarIn", scalar);
            cs.SetTexture(debugKernel, "_DebugOut", vis);
            cs.Dispatch(debugKernel, Mathf.CeilToInt(w / 8f), Mathf.CeilToInt(h / 8f), 1);
            var tex = await ReadbackTextureAsync(vis, w, h, ct);
            if (tex == null)
                return;
            try
            {
                var bytes = tex.EncodeToPNG();
                var path = Path.Combine(dir, fileName);
                await File.WriteAllBytesAsync(path, bytes, ct);
            }
            catch
            {
            }
            finally
            {
                Destroy(tex);
            }
        }
        finally
        {
            SafeReleaseRT(vis);
        }
    }

    private static async UniTask<Texture2D> ReadbackTextureAsync(RenderTexture rt, int w, int h, CancellationToken ct)
    {
        if (ShouldUseSynchronousTextureReadback())
            return ReadbackTextureSync(rt, w, h, TextureFormat.RGBA32, ct);

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

    private static async UniTask<Texture2D> ReadbackScalarTextureAsync(RenderTexture rt, int w, int h, CancellationToken ct)
    {
        if (ShouldUseSynchronousTextureReadback())
            return ReadbackTextureSync(rt, w, h, TextureFormat.RHalf, ct);

        var tcs = new UniTaskCompletionSource<AsyncGPUReadbackRequest>();
        AsyncGPUReadback.Request(rt, 0, TextureFormat.RHalf, req => tcs.TrySetResult(req));
        var r = await tcs.Task.AttachExternalCancellation(ct);
        if (r.hasError)
            return null;

        var data = r.GetData<byte>();
        var tex = new Texture2D(w, h, TextureFormat.RHalf, false, true);
        tex.LoadRawTextureData(data);
        tex.Apply(false, false);
        return tex;
    }

    private static bool ShouldUseSynchronousTextureReadback()
    {
        return Application.isBatchMode || !SystemInfo.supportsAsyncGPUReadback;
    }

    private static Texture2D ReadbackTextureSync(
        RenderTexture rt,
        int w,
        int h,
        TextureFormat format,
        CancellationToken ct)
    {
        if (rt == null || w <= 0 || h <= 0)
            return null;

        ct.ThrowIfCancellationRequested();
        var previous = RenderTexture.active;
        try
        {
            RenderTexture.active = rt;
            var tex = new Texture2D(w, h, format, false, true);
            tex.ReadPixels(new Rect(0, 0, w, h), 0, 0, false);
            tex.Apply(false, false);
            tex.wrapMode = TextureWrapMode.Clamp;
            tex.filterMode = FilterMode.Bilinear;
            return tex;
        }
        finally
        {
            RenderTexture.active = previous;
        }
    }

    private static string CreateDumpDir()
    {
        var root = Application.temporaryCachePath;
        if (string.IsNullOrWhiteSpace(root))
            root = Path.GetTempPath();
        var dir = Path.Combine(root, "AIImage_FaceMask_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
        try { Directory.CreateDirectory(dir); } catch { }
        return dir;
    }

#if !UNITY_WEBGL
    private static void OpenFolderInShell(string directoryPath)
    {
        if (string.IsNullOrWhiteSpace(directoryPath))
            return;
        try
        {
            try { Directory.CreateDirectory(directoryPath); } catch { }
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
            Process.Start(new ProcessStartInfo { FileName = directoryPath, UseShellExecute = true });
#elif UNITY_STANDALONE_OSX || UNITY_EDITOR_OSX
            Process.Start(new ProcessStartInfo("open", directoryPath) { UseShellExecute = false });
#elif UNITY_STANDALONE_LINUX
            Process.Start(new ProcessStartInfo("xdg-open", directoryPath) { UseShellExecute = false });
#else
            var url = "file://" + directoryPath.Replace('\\', '/');
            Application.OpenURL(url);
#endif
        }
        catch
        {
        }
    }
#endif
}

public struct FaceMaskTextureResult
{
    public Texture2D mask;
    public string dumpDir;
    public string error;
}
