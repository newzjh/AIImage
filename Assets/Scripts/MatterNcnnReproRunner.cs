using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using Cysharp.Threading.Tasks;
using NcnnCompute;
using UnityEngine;

public struct MattingResult
{
    public Texture2D texture;
    public Texture2D matte;
    public string error;
    public long elapsedMs;
}

public sealed class MatterNcnnReproRunner : MonoBehaviour
{
    private const string OutputBlobName = "local";

    public string modelParamRelativePath = "Matting/matting.param";
    public string modelBinRelativePath = "Matting/matting.bin";
    public int refSize = 512;
    public bool preserveAspectRatioInput = false;
    public bool useArgbFloatTensor = true;
    public bool forceBufferConvolution = true;
    public bool enableTempPool = true;
    public int maxPooledPerShape = 4;
    public bool enableWinograd23 = true;
    public Color32 compositeBackgroundColor = new Color32(120, 255, 155, 255);
    public bool enableDebugDump = false;
    public string[] debugBlobNames =
    {
        "500", "523", "554", "599", "623", "632",
        "736", "757", "786", "815", "841",
        "501", "502", "524", "555", "600",
        "637", "651", "677", "703", "722",
        "741", "762", "791", "820", "846",
        "local"
    };

    public event Action<float, string> ProgressChanged;

    private NcnnOps _ops;
    private NcnnRepro3 _repro;
    private bool _loaded;
    private string _lastDumpDir;
    public string LastDumpDir => _lastDumpDir;

    private void Awake()
    {
        EnsureRuntimeObjects();
    }

    private void OnDestroy()
    {
        Release();
    }

    public async UniTask<MattingResult> ProcessAsync(Texture2D src, CancellationToken ct)
    {
        if (src == null)
            return default;

        var totalSw = Stopwatch.StartNew();
        var originalW = src.width;
        var originalH = src.height;

        MattingResult Finish(MattingResult result)
        {
            result.elapsedMs = totalSw.ElapsedMilliseconds;
            return result;
        }

        RenderTexture resizedInput = null;
        RenderTexture inputPack4 = null;
        RenderTexture mattePack4 = null;
        RenderTexture matteFullResPack4 = null;
        Texture2D readableSrc = null;
        _lastDumpDir = null;

        try
        {
            EnsureRuntimeObjects();
            ApplyReproOptions();
            await EnsureLoaded();
            if (_repro?.Model == null)
                return Finish(new MattingResult { error = "Matting model unavailable" });

            ct.ThrowIfCancellationRequested();
            ReportProgress(0.05f, "Prepare input");
            await UniTask.Yield();

            var (inputW, inputH) = ComputeModelInputSize(originalW, originalH, refSize, preserveAspectRatioInput);
            resizedInput = ResizeTextureBilinear(src, inputW, inputH);
            if (resizedInput == null)
                return Finish(new MattingResult { error = "Resize input failed" });

            inputPack4 = _repro.RentTempArray(inputW, inputH, 1, RenderTextureFormat.ARGBHalf);
            _ops.PackRgbToPack4Gfpgan(resizedInput, 0, 0, 1f, 1f, inputPack4, false);

            ReportProgress(0.30f, "Run matting");
            await UniTask.Yield();

            HashSet<string> pinned = null;
            if (enableDebugDump && debugBlobNames != null && debugBlobNames.Length > 0)
            {
                pinned = new HashSet<string>(debugBlobNames.Where(s => !string.IsNullOrWhiteSpace(s)), StringComparer.Ordinal);
                pinned.Add(OutputBlobName);
            }

            using (var infer = _repro.Infer(inputPack4, 1, "input", pinned))
            {
                if (enableDebugDump && pinned != null && pinned.Count > 0)
                {
                    _lastDumpDir = CreateDumpDir();
                    DumpPinnedBlobStats(infer, _lastDumpDir, pinned);
                }
                mattePack4 = infer.ExtractTexture(OutputBlobName);
            }

            if (mattePack4 == null)
                return Finish(new MattingResult { error = "Matting output missing" });

            ReportProgress(0.65f, "Resize alpha");
            await UniTask.Yield();

            matteFullResPack4 = mattePack4;
            mattePack4 = null;
            if (matteFullResPack4.width != originalW || matteFullResPack4.height != originalH)
            {
                var scaleX = (float)originalW / matteFullResPack4.width;
                var scaleY = (float)originalH / matteFullResPack4.height;
                var upsampled = _repro.RentTempArray(originalW, originalH, matteFullResPack4.volumeDepth > 0 ? matteFullResPack4.volumeDepth : 1, RenderTextureFormat.ARGBHalf);
                _ops.InterpPack4(matteFullResPack4, upsampled.volumeDepth > 0 ? upsampled.volumeDepth : 1, scaleX, scaleY, upsampled);
                _repro.ReturnTempArray(matteFullResPack4);
                matteFullResPack4 = upsampled;
            }

            ReportProgress(0.82f, "Read back alpha");
            await UniTask.Yield();

            var alpha = ReadbackSingleChannel(matteFullResPack4, originalW, originalH);
            if (alpha == null || alpha.Length != originalW * originalH)
                return Finish(new MattingResult { error = "Read alpha failed" });

            readableSrc = EnsureReadable(src);
            if (readableSrc == null)
                return Finish(new MattingResult { error = "Prepare source pixels failed" });

            var matte = BuildMatteTexture(alpha, originalW, originalH);
            var composite = BuildCompositeTexture(readableSrc, alpha, compositeBackgroundColor);
            return Finish(new MattingResult { texture = composite, matte = matte });
        }
        catch (OperationCanceledException)
        {
            return Finish(new MattingResult { error = "Cancelled" });
        }
        catch (Exception e)
        {
            return Finish(new MattingResult { error = e.Message });
        }
        finally
        {
            if (resizedInput != null)
                ReleaseTemporaryRt(resizedInput);
            if (inputPack4 != null)
                _repro?.ReturnTempArray(inputPack4);
            if (mattePack4 != null)
                _repro?.ReturnTempArray(mattePack4);
            if (matteFullResPack4 != null)
                _repro?.ReturnTempArray(matteFullResPack4);
            if (readableSrc != null && readableSrc != src)
                Destroy(readableSrc);
            ReportProgress(1f, string.Empty);
        }
    }

    private void EnsureRuntimeObjects()
    {
        if (_ops == null)
            _ops = new NcnnOps();
        if (_repro == null)
            _repro = new NcnnRepro3(_ops);
    }

    private void ApplyReproOptions()
    {
        if (_repro == null)
            return;
        _repro.EnableTempPool = enableTempPool;
        _repro.MaxPooledPerShape = maxPooledPerShape;
        _repro.EnableWinograd23 = enableWinograd23;
        _repro.ForceBufferConvolution = forceBufferConvolution;
        _repro.TensorTextureFormat = useArgbFloatTensor ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGBHalf;
    }

    private async UniTask EnsureLoaded()
    {
        if (_loaded && _repro?.Model != null)
            return;

        var paramPath = Path.Combine(Application.streamingAssetsPath, modelParamRelativePath);
        var binPath = Path.Combine(Application.streamingAssetsPath, modelBinRelativePath);
        if (!File.Exists(paramPath))
            throw new FileNotFoundException("Matting param not found", paramPath);
        if (!File.Exists(binPath))
            throw new FileNotFoundException("Matting bin not found", binPath);

        var paramText = await File.ReadAllTextAsync(paramPath);
        var binBytes = await File.ReadAllBytesAsync(binPath);
        using var ms = new MemoryStream(binBytes, false);
        using var br = new NcnnBinReader(ms);
        _repro.LoadModel(paramText, br);
        _loaded = true;
    }

    private void Release()
    {
        _loaded = false;
        try { _repro?.Release(); } catch { }
        try { _repro?.Dispose(); } catch { }
        _repro = null;
        _ops = null;
    }

    private static (int width, int height) ComputeModelInputSize(int srcW, int srcH, int refSize, bool preserveAspectRatio)
    {
        if (!preserveAspectRatio)
        {
            var fixedSize = Mathf.Max(32, refSize);
            return (fixedSize, fixedSize);
        }

        var rw = srcW;
        var rh = srcH;
        if (Mathf.Max(srcH, srcW) < refSize || Mathf.Min(srcH, srcW) > refSize)
        {
            if (srcW >= srcH)
            {
                rh = refSize;
                rw = Mathf.RoundToInt((float)srcW / Mathf.Max(1, srcH) * refSize);
            }
            else
            {
                rw = refSize;
                rh = Mathf.RoundToInt((float)srcH / Mathf.Max(1, srcW) * refSize);
            }
        }

        rw -= rw % 32;
        rh -= rh % 32;
        rw = Mathf.Max(32, rw);
        rh = Mathf.Max(32, rh);
        return (rw, rh);
    }

    private static RenderTexture ResizeTextureBilinear(Texture src, int width, int height)
    {
        if (src == null)
            return null;
        var rt = GetTemporaryRt(width, height, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Default, false);
        Graphics.Blit(src, rt);
        return rt;
    }

    private static Texture2D EnsureReadable(Texture2D src)
    {
        if (src == null)
            return null;
        try
        {
            var _ = src.GetPixel(0, 0);
            return src;
        }
        catch
        {
            var rt = ResizeTextureBilinear(src, src.width, src.height);
            if (rt == null)
                return null;
            try
            {
                return RenderTextureToTexture2D(rt, src.width, src.height);
            }
            finally
            {
                ReleaseTemporaryRt(rt);
            }
        }
    }

    private float[] ReadbackSingleChannel(RenderTexture pack4, int width, int height)
    {
        if (pack4 == null)
            return null;
        using var buffer = new ComputeBuffer(width * height, sizeof(float), ComputeBufferType.Structured);
        var data = new float[width * height];
        _ops.Pack4ToBufferCHW(pack4, width, height, 1, buffer);
        buffer.GetData(data);
        return data;
    }

    private static Texture2D BuildMatteTexture(float[] alpha, int width, int height)
    {
        var pixels = new Color32[width * height];
        for (var i = 0; i < pixels.Length; i++)
        {
            var a = (byte)Mathf.Clamp(Mathf.RoundToInt(Mathf.Clamp01(alpha[i]) * 255f), 0, 255);
            pixels[i] = new Color32(a, a, a, 255);
        }

        var texture = new Texture2D(width, height, TextureFormat.RGBA32, false, false);
        texture.SetPixels32(pixels);
        texture.Apply(false, false);
        return texture;
    }

    private static Texture2D BuildCompositeTexture(Texture2D src, float[] alpha, Color32 bg)
    {
        var srcPixels = src.GetPixels32();
        var outPixels = new Color32[srcPixels.Length];
        for (var i = 0; i < srcPixels.Length; i++)
        {
            var a = Mathf.Clamp01(alpha[i]);
            var inv = 1f - a;
            var s = srcPixels[i];
            outPixels[i] = new Color32(
                (byte)Mathf.Clamp(Mathf.RoundToInt(s.r * a + bg.r * inv), 0, 255),
                (byte)Mathf.Clamp(Mathf.RoundToInt(s.g * a + bg.g * inv), 0, 255),
                (byte)Mathf.Clamp(Mathf.RoundToInt(s.b * a + bg.b * inv), 0, 255),
                255);
        }

        var texture = new Texture2D(src.width, src.height, TextureFormat.RGBA32, false, false);
        texture.SetPixels32(outPixels);
        texture.Apply(false, false);
        return texture;
    }

    private static Texture2D RenderTextureToTexture2D(RenderTexture rt, int width, int height)
    {
        var prev = RenderTexture.active;
        try
        {
            RenderTexture.active = rt;
            var texture = new Texture2D(width, height, TextureFormat.RGBA32, false, false);
            texture.ReadPixels(new Rect(0, 0, width, height), 0, 0, false);
            texture.Apply(false, false);
            return texture;
        }
        finally
        {
            RenderTexture.active = prev;
        }
    }

    private void ReportProgress(float progress01, string text)
    {
        try { ProgressChanged?.Invoke(Mathf.Clamp01(progress01), text ?? string.Empty); } catch { }
    }

    private void DumpPinnedBlobStats(NcnnRepro3.InferResult infer, string dir, ICollection<string> blobNames)
    {
        if (infer == null || string.IsNullOrWhiteSpace(dir) || blobNames == null || blobNames.Count == 0)
            return;

        Directory.CreateDirectory(dir);
        var sb = new StringBuilder();
        foreach (var name in blobNames)
        {
            if (string.IsNullOrWhiteSpace(name))
                continue;

            RenderTexture tex = null;
            try
            {
                tex = infer.GetTexture(name);
            }
            catch
            {
                continue;
            }

            if (tex == null)
                continue;

            var physicalChannels = Mathf.Max(1, (tex.volumeDepth > 0 ? tex.volumeDepth : 1) * 4);
            using var buffer = new ComputeBuffer(tex.width * tex.height * physicalChannels, sizeof(float), ComputeBufferType.Structured);
            var data = new float[buffer.count];
            _ops.Pack4ToBufferCHW(tex, tex.width, tex.height, physicalChannels, buffer);
            buffer.GetData(data);

            var min = float.PositiveInfinity;
            var max = float.NegativeInfinity;
            double sum = 0d;
            var finiteCount = 0;
            var nanCount = 0;
            var infCount = 0;
            foreach (var value in data)
            {
                if (float.IsNaN(value))
                {
                    nanCount++;
                    continue;
                }
                if (float.IsInfinity(value))
                {
                    infCount++;
                    continue;
                }
                if (value < min) min = value;
                if (value > max) max = value;
                sum += value;
                finiteCount++;
            }

            var mean = finiteCount > 0 ? (float)(sum / finiteCount) : float.NaN;
            sb.AppendLine(name + " | size=" + tex.width + "x" + tex.height + "x" + physicalChannels + " | min=" + min.ToString("G9") + " | max=" + max.ToString("G9") + " | mean=" + mean.ToString("G9") + " | nan=" + nanCount + " | inf=" + infCount);

            var previewPath = Path.Combine(dir, name + "_c0.png");
            TryWriteFirstChannelPreview(data, tex.width, tex.height, physicalChannels, previewPath);
        }

        File.WriteAllText(Path.Combine(dir, "blob_stats.txt"), sb.ToString());
    }

    private static void TryWriteFirstChannelPreview(float[] chw, int width, int height, int channels, string path)
    {
        if (chw == null || chw.Length == 0 || width <= 0 || height <= 0 || channels <= 0 || string.IsNullOrWhiteSpace(path))
            return;

        var planeSize = width * height;
        if (chw.Length < planeSize)
            return;

        var min = float.PositiveInfinity;
        var max = float.NegativeInfinity;
        for (var i = 0; i < planeSize; i++)
        {
            var v = chw[i];
            if (float.IsNaN(v) || float.IsInfinity(v))
                continue;
            if (v < min) min = v;
            if (v > max) max = v;
        }

        var scale = max > min ? 1f / (max - min) : 0f;
        var pixels = new Color32[planeSize];
        for (var i = 0; i < planeSize; i++)
        {
            var v = chw[i];
            var n = scale > 0f ? Mathf.Clamp01((v - min) * scale) : 0f;
            var b = (byte)Mathf.Clamp(Mathf.RoundToInt(n * 255f), 0, 255);
            pixels[i] = new Color32(b, b, b, 255);
        }

        var tex = new Texture2D(width, height, TextureFormat.RGBA32, false, false);
        tex.SetPixels32(pixels);
        tex.Apply(false, false);
        try
        {
            File.WriteAllBytes(path, tex.EncodeToPNG());
        }
        finally
        {
            DestroyImmediate(tex);
        }
    }

    private static string CreateDumpDir()
    {
        var root = Path.Combine(Path.GetTempPath(), "YanQi", "AIImage");
        Directory.CreateDirectory(root);
        var dir = Path.Combine(root, "AIImage_MattingBlobDump_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static RenderTexture GetTemporaryRt(int width, int height, RenderTextureFormat format, RenderTextureReadWrite readWrite, bool randomWrite)
    {
        var desc = new RenderTextureDescriptor(Mathf.Max(1, width), Mathf.Max(1, height), format, 0)
        {
            enableRandomWrite = randomWrite,
            msaaSamples = 1,
            sRGB = readWrite != RenderTextureReadWrite.Linear
        };
        var rt = RenderTexture.GetTemporary(desc);
        rt.wrapMode = TextureWrapMode.Clamp;
        rt.filterMode = FilterMode.Bilinear;
        return rt;
    }

    private static void ReleaseTemporaryRt(RenderTexture rt)
    {
        if (rt == null)
            return;
        try { RenderTexture.ReleaseTemporary(rt); } catch { }
    }
}
