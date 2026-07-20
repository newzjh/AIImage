using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using Aexis.Ncnn;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Samples.Runners
{

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
    public NcnnPrecisionMode precisionMode = NcnnPrecisionMode.Auto;
    public bool preserveAspectRatioInput = false;
    public bool useArgbFloatTensor = true;
    public bool useCommandBuffer = false;
    public bool useAsyncComputeCommandBuffer = false;
    public bool forceBufferConvolution = false;
    public bool useTextureMaxPoolingInd = true;
    public bool disallowBufferAccess = false;
    public bool disallowBufferOutputs = false;
    public bool disallowBufferToTextureMaterialization = false;
    public bool enableForegroundCleanup = true;
    [Range(0f, 1f)] public float foregroundCleanupThreshold = 0.05f;
    [Range(0, 4)] public int foregroundCleanupCloseRadius = 2;
    public Color32 compositeBackgroundColor = new Color32(120, 255, 155, 255);
    public bool enableDebugDump = false;
    public bool enableTextureConvCompare = false;
    public bool enableMaxPoolingCompare = false;
    public string[] debugBlobNames =
    {
        "500", "523", "554", "599", "623", "632",
        "736", "757", "786", "815", "841",
        "501", "502", "524", "555", "600",
        "637", "651", "677", "703", "722",
        "741", "762", "791", "820", "846",
        "local"
    };
    public string[] debugCompareTextureConvLayers =
    {
        "*"
    };
    public string[] debugCompareMaxPoolingLayers =
    {
        "MaxPool_19", "MaxPool_41"
    };

    public event Action<float, string> ProgressChanged;

    private NcnnOps _ops;
    private NcnnGraphSession _repro;
    private bool _loaded;
    private bool _hasAppliedPrecisionMode;
    private NcnnPrecisionMode _appliedPrecisionMode;
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

    public async Awaitable<MattingResult> ProcessAsync(Texture2D src, CancellationToken ct)
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
        List<string> textureConvCompareLines = null;

        try
        {
            EnsureRuntimeObjects();
            ApplyReproOptions();
            EnsureLoaded();
            if (_repro?.Model == null)
                return Finish(new MattingResult { error = "Matting model unavailable" });

            ct.ThrowIfCancellationRequested();
            ReportProgress(0.05f, "Prepare input");
            await Awaitable.NextFrameAsync();

            var (inputW, inputH) = ComputeModelInputSize(originalW, originalH, refSize, preserveAspectRatioInput);
            resizedInput = ResizeTextureBilinear(src, inputW, inputH);
            if (resizedInput == null)
                return Finish(new MattingResult { error = "Resize input failed" });

            if (!useCommandBuffer)
            {
                inputPack4 = _repro.RentTempArray(inputW, inputH, 1, RenderTextureFormat.ARGBHalf);
                _ops.PackRgbToPack4Gfpgan(resizedInput, 0, 0, 1f, 1f, inputPack4, false);
            }

            ReportProgress(0.30f, "Run matting");
            await Awaitable.NextFrameAsync();

            HashSet<string> pinned = null;
            if (enableDebugDump && debugBlobNames != null && debugBlobNames.Length > 0)
            {
                pinned = new HashSet<string>(debugBlobNames.Where(s => !string.IsNullOrWhiteSpace(s)), StringComparer.Ordinal);
                pinned.Add(OutputBlobName);
            }

            if (enableDebugDump)
            {
                _lastDumpDir = CreateDumpDir();
            }

            if (enableTextureConvCompare && !forceBufferConvolution && !useCommandBuffer)
            {
                textureConvCompareLines = new List<string>();
                _repro.DebugCompareTextureConvLayers = new HashSet<string>(debugCompareTextureConvLayers ?? Array.Empty<string>(), StringComparer.Ordinal);
                _repro.DebugLog = line =>
                {
                    if (!string.IsNullOrWhiteSpace(line))
                        textureConvCompareLines.Add(line);
                };
            }
            else
            {
                _repro.DebugCompareTextureConvLayers = null;
                _repro.DebugLog = null;
            }

            if (enableMaxPoolingCompare && !useCommandBuffer)
            {
                _repro.DebugCompareMaxPoolingLayers = new HashSet<string>(debugCompareMaxPoolingLayers ?? Array.Empty<string>(), StringComparer.Ordinal);
                if (_repro.DebugLog == null)
                {
                    textureConvCompareLines ??= new List<string>();
                    _repro.DebugLog = line =>
                    {
                        if (!string.IsNullOrWhiteSpace(line))
                            textureConvCompareLines.Add(line);
                    };
                }
            }
            else
            {
                _repro.DebugCompareMaxPoolingLayers = null;
            }

            if (useCommandBuffer)
            {
                mattePack4 = await ForwardMattingWithCommandBufferAsync(resizedInput, inputW, inputH, ct);
            }
            else
            {
                using var infer = _repro.Infer(inputPack4, 1, "input", pinned);
                if (enableDebugDump && pinned != null && pinned.Count > 0)
                {
                    DumpPinnedBlobStats(infer, _lastDumpDir, pinned);
                }
                mattePack4 = infer.ExtractTexture(OutputBlobName);
            }

            if (enableDebugDump && textureConvCompareLines != null && textureConvCompareLines.Count > 0 && !string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                File.WriteAllLines(Path.Combine(_lastDumpDir, "conv_compare.txt"), textureConvCompareLines);
            }

            if (mattePack4 == null)
                return Finish(new MattingResult { error = "Matting output missing" });

            ReportProgress(0.65f, "Resize alpha");
            await Awaitable.NextFrameAsync();

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
            await Awaitable.NextFrameAsync();

            var alpha = ReadbackSingleChannel(matteFullResPack4, originalW, originalH);
            if (alpha == null || alpha.Length != originalW * originalH)
                return Finish(new MattingResult { error = "Read alpha failed" });

            if (enableForegroundCleanup)
            {
                ApplyLargestForegroundCleanup(alpha, originalW, originalH, foregroundCleanupThreshold, foregroundCleanupCloseRadius);
            }

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
            if (_repro != null)
            {
                _repro.DebugCompareTextureConvLayers = null;
                _repro.DebugCompareMaxPoolingLayers = null;
                _repro.DebugLog = null;
            }
            ReportProgress(1f, string.Empty);
        }
    }

    private Awaitable<RenderTexture> ForwardMattingWithCommandBufferAsync(Texture resizedInput, int width, int height, CancellationToken ct)
    {
        if (resizedInput == null)
            throw new ArgumentNullException(nameof(resizedInput));
        if (_repro == null || _ops == null)
            throw new InvalidOperationException("Matting CommandBuffer path is not initialized.");

        ct.ThrowIfCancellationRequested();
        using var cmd = new CommandBuffer { name = "MattingPack4" };
        if (useAsyncComputeCommandBuffer)
            cmd.SetExecutionFlags(CommandBufferExecutionFlags.AsyncCompute);

        var inputCmd = _repro.RentTempArray(cmd, width, height, 1, RenderTextureFormat.ARGBHalf);
        ComputeTexture outputCmd = null;
        RenderTexture output = null;
        try
        {
            _ops.PackRgbToPack4Gfpgan(cmd, resizedInput, 0, 0, 1f, 1f, inputCmd, false);
            outputCmd = _repro.ForwardPack4(
                cmd,
                inputCmd,
                new NcnnGraphSession.BufferShape(3, width, height, 1, 3),
                out _,
                "input");
            if (outputCmd == null)
                throw new InvalidOperationException("Matting CommandBuffer path produced no output texture.");

            output = outputCmd.dimension == TextureDimension.Tex2D
                ? _repro.RentTempMat(outputCmd.width, outputCmd.height, outputCmd.format)
                : _repro.RentTempArray(outputCmd.width, outputCmd.height, outputCmd.depth, outputCmd.format);
            var outputDepth = outputCmd.dimension == TextureDimension.Tex2D ? 1 : Mathf.Max(1, outputCmd.depth);
            for (var slice = 0; slice < outputDepth; slice++)
                cmd.CopyTexture(outputCmd.nameID, slice, 0, output, slice, 0);

            _repro.ReturnTempArray(cmd, outputCmd);
            outputCmd = null;
            _repro.ReturnTempArray(cmd, inputCmd);
            inputCmd = null;

            if (useAsyncComputeCommandBuffer)
                Graphics.ExecuteCommandBufferAsync(cmd, ComputeQueueType.Default);
            else
                Graphics.ExecuteCommandBuffer(cmd);
            _ops.DebugSyncGpu();
            ct.ThrowIfCancellationRequested();
            return AexisSampleAwaitable.FromResult(output);
        }
        catch
        {
            if (output != null)
            {
                _repro.ReturnTempArray(output);
                output = null;
            }
            throw;
        }
        finally
        {
            if (outputCmd != null)
                _repro.ReturnTempArray(cmd, outputCmd);
            if (inputCmd != null)
                _repro.ReturnTempArray(cmd, inputCmd);
        }
    }

    private void EnsureRuntimeObjects()
    {
        if (_repro != null && _hasAppliedPrecisionMode && _appliedPrecisionMode != precisionMode)
        {
            UnityEngine.Debug.Log("[NcnnPrecision] Matting recreating session | from=" + _appliedPrecisionMode + " | to=" + precisionMode);
            Release();
        }
        if (_ops == null)
            _ops = new NcnnOps();
        if (_repro == null)
        {
            _repro = NcnnInferenceSessionFactory.Create(_ops, "matting.ncnn", precisionMode);
            _appliedPrecisionMode = precisionMode;
            _hasAppliedPrecisionMode = true;
        }
    }

    private void ApplyReproOptions()
    {
        if (_repro == null)
            return;
        _repro.ForceBufferConvolution = forceBufferConvolution;
        _repro.UseTextureMaxPoolingInd = useTextureMaxPoolingInd;
        // FP32 retains the existing ARGBFloat path. Do not override an explicitly
        // selected FP16 session back to float activations.
        _repro.TensorTextureFormat = _repro.AppliedPrecisionMode == NcnnPrecisionMode.FP16
            ? RenderTextureFormat.ARGBHalf
            : useArgbFloatTensor ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGBHalf;
        _repro.EnableGeneralTextureConvolution = true;
        _repro.DisallowBufferAccess = disallowBufferAccess;
        _repro.DisallowBufferOutputs = disallowBufferOutputs;
        _repro.DisallowBufferToTextureMaterialization = disallowBufferToTextureMaterialization;
        _repro.DisallowInferenceTempComputeBuffers = disallowBufferAccess
            || disallowBufferOutputs
            || disallowBufferToTextureMaterialization;
    }

    private void EnsureLoaded()
    {
        if (_loaded && _repro?.Model != null)
            return;

        var paramPath = Path.Combine(Application.streamingAssetsPath, modelParamRelativePath);
        var binPath = Path.Combine(Application.streamingAssetsPath, modelBinRelativePath);
        if (!File.Exists(paramPath))
            throw new FileNotFoundException("Matting param not found", paramPath);
        if (!File.Exists(binPath))
            throw new FileNotFoundException("Matting bin not found", binPath);

        var paramText = File.ReadAllText(paramPath);
        var binBytes = File.ReadAllBytes(binPath);
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
        try { _ops?.Dispose(); } catch { }
        _repro = null;
        _ops = null;
        _hasAppliedPrecisionMode = false;
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

    private void DumpPinnedBlobStats(NcnnGraphSession.InferResult infer, string dir, ICollection<string> blobNames)
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
        var root = Path.Combine(Path.GetTempPath(), "YanQi", "Aexis");
        Directory.CreateDirectory(root);
        var dir = Path.Combine(root, "Aexis_MattingBlobDump_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void ApplyLargestForegroundCleanup(float[] alpha, int width, int height, float threshold, int closeRadius)
    {
        if (alpha == null || alpha.Length != width * height || width <= 0 || height <= 0)
            return;

        if (width * height < 120000)
        {
            ApplyLegacyForegroundCleanup(alpha, width, height, threshold, closeRadius);
            return;
        }

        var weak = new bool[alpha.Length];
        for (var i = 0; i < alpha.Length; i++)
            weak[i] = alpha[i] >= threshold;

        var strongThreshold = Mathf.Clamp01(Mathf.Max(threshold + 0.20f, 0.35f));
        var strong = new bool[alpha.Length];
        for (var i = 0; i < alpha.Length; i++)
            strong[i] = alpha[i] >= strongThreshold;

        var minSeedArea = Mathf.Max(32, Mathf.RoundToInt(width * height * 0.0005f));
        var seeds = KeepConnectedComponents(strong, width, height, minSeedArea);
        if (!HasAny(seeds))
        {
            seeds = KeepLargestConnectedComponent(weak, width, height);
        }

        if (!HasAny(seeds))
            return;

        var keep = GrowWithinMask(seeds, weak, width, height);

        if (closeRadius > 0)
        {
            keep = MorphClose(keep, width, height, closeRadius);
            keep = GrowWithinMask(seeds, keep, width, height);
        }

        for (var i = 0; i < alpha.Length; i++)
        {
            if (!keep[i])
                alpha[i] = 0f;
        }

        if (closeRadius > 0)
            ApplyGrayCloseInMask(alpha, keep, width, height, closeRadius);

        var coreThreshold = Mathf.Clamp01(Mathf.Max(threshold + 0.55f, 0.75f));
        var core = new bool[alpha.Length];
        for (var i = 0; i < alpha.Length; i++)
            core[i] = keep[i] && alpha[i] >= coreThreshold;

        if (HasAny(core))
        {
            var interiorRadius = Mathf.Clamp(Mathf.Max(closeRadius, Mathf.RoundToInt(Mathf.Min(width, height) * 0.01f)), 4, 8);
            var interiorSupport = MorphDilate(core, width, height, interiorRadius);
            for (var i = 0; i < interiorSupport.Length; i++)
                interiorSupport[i] = interiorSupport[i] && keep[i];
            ApplyGrayCloseInMask(alpha, interiorSupport, width, height, interiorRadius);
        }

        RemoveSmallForegroundIslands(alpha, width, height, Mathf.Clamp01(threshold + 0.03f), 8);
        FillSmallInteriorAlphaDips(alpha, width, height, 0.65f, 320, 16, 3);
    }

    private static void ApplyLegacyForegroundCleanup(float[] alpha, int width, int height, float threshold, int closeRadius)
    {
        var binary = new bool[alpha.Length];
        for (var i = 0; i < alpha.Length; i++)
            binary[i] = alpha[i] >= threshold;

        var keep = KeepLargestConnectedComponent(binary, width, height);
        if (!HasAny(keep))
            return;

        if (closeRadius > 0)
            keep = MorphClose(keep, width, height, closeRadius);

        for (var i = 0; i < alpha.Length; i++)
        {
            if (!keep[i])
                alpha[i] = 0f;
        }

        if (closeRadius > 0)
            ApplyGrayCloseInMask(alpha, keep, width, height, closeRadius);

        RemoveSmallForegroundIslands(alpha, width, height, Mathf.Clamp01(threshold + 0.03f), 8);
        FillSmallInteriorAlphaDips(alpha, width, height, 0.65f, 120, 12, 2);
    }

    private static bool[] KeepLargestConnectedComponent(bool[] mask, int width, int height)
    {
        var labels = new int[mask.Length];
        var queue = new int[mask.Length];
        var bestLabel = 0;
        var bestArea = 0;
        var nextLabel = 1;

        for (var start = 0; start < mask.Length; start++)
        {
            if (!mask[start] || labels[start] != 0)
                continue;

            var area = FloodLabel(mask, width, height, start, nextLabel, labels, queue);
            if (area > bestArea)
            {
                bestArea = area;
                bestLabel = nextLabel;
            }

            nextLabel++;
        }

        if (bestLabel == 0)
            return new bool[mask.Length];

        var keep = new bool[mask.Length];
        for (var i = 0; i < keep.Length; i++)
            keep[i] = labels[i] == bestLabel;
        return keep;
    }

    private static bool[] KeepConnectedComponents(bool[] mask, int width, int height, int minArea)
    {
        var labels = new int[mask.Length];
        var queue = new int[mask.Length];
        var keepLabels = new HashSet<int>();
        var nextLabel = 1;

        for (var start = 0; start < mask.Length; start++)
        {
            if (!mask[start] || labels[start] != 0)
                continue;

            var area = FloodLabel(mask, width, height, start, nextLabel, labels, queue);
            if (area >= minArea)
                keepLabels.Add(nextLabel);
            nextLabel++;
        }

        var keep = new bool[mask.Length];
        if (keepLabels.Count == 0)
            return keep;

        for (var i = 0; i < keep.Length; i++)
            keep[i] = keepLabels.Contains(labels[i]);
        return keep;
    }

    private static bool[] GrowWithinMask(bool[] seeds, bool[] mask, int width, int height)
    {
        var keep = new bool[mask.Length];
        var queue = new int[mask.Length];
        var head = 0;
        var tail = 0;

        for (var i = 0; i < mask.Length; i++)
        {
            if (!seeds[i] || !mask[i])
                continue;
            keep[i] = true;
            queue[tail++] = i;
        }

        while (head < tail)
        {
            var index = queue[head++];
            var x = index % width;
            var y = index / width;

            for (var ny = Mathf.Max(0, y - 1); ny <= Mathf.Min(height - 1, y + 1); ny++)
            {
                var row = ny * width;
                for (var nx = Mathf.Max(0, x - 1); nx <= Mathf.Min(width - 1, x + 1); nx++)
                {
                    var ni = row + nx;
                    if (!mask[ni] || keep[ni])
                        continue;
                    keep[ni] = true;
                    queue[tail++] = ni;
                }
            }
        }

        return keep;
    }

    private static int FloodLabel(bool[] mask, int width, int height, int start, int label, int[] labels, int[] queue)
    {
        var head = 0;
        var tail = 0;
        queue[tail++] = start;
        labels[start] = label;
        var area = 0;

        while (head < tail)
        {
            var index = queue[head++];
            area++;
            var x = index % width;
            var y = index / width;

            for (var ny = Mathf.Max(0, y - 1); ny <= Mathf.Min(height - 1, y + 1); ny++)
            {
                var row = ny * width;
                for (var nx = Mathf.Max(0, x - 1); nx <= Mathf.Min(width - 1, x + 1); nx++)
                {
                    var ni = row + nx;
                    if (!mask[ni] || labels[ni] != 0)
                        continue;
                    labels[ni] = label;
                    queue[tail++] = ni;
                }
            }
        }

        return area;
    }

    private static bool HasAny(bool[] mask)
    {
        if (mask == null)
            return false;

        for (var i = 0; i < mask.Length; i++)
        {
            if (mask[i])
                return true;
        }

        return false;
    }

    private static void RemoveSmallForegroundIslands(float[] alpha, int width, int height, float threshold, int maxArea)
    {
        if (alpha == null || alpha.Length != width * height || maxArea <= 0)
            return;

        var mask = new bool[alpha.Length];
        for (var i = 0; i < alpha.Length; i++)
            mask[i] = alpha[i] >= threshold;

        var labels = new int[mask.Length];
        var queue = new int[mask.Length];
        var areas = new Dictionary<int, int>();
        var nextLabel = 1;

        for (var start = 0; start < mask.Length; start++)
        {
            if (!mask[start] || labels[start] != 0)
                continue;

            var area = FloodLabel(mask, width, height, start, nextLabel, labels, queue);
            areas[nextLabel] = area;
            nextLabel++;
        }

        if (areas.Count <= 1)
            return;

        for (var i = 0; i < alpha.Length; i++)
        {
            var label = labels[i];
            if (label == 0)
                continue;
            if (areas.TryGetValue(label, out var area) && area <= maxArea)
                alpha[i] = 0f;
        }
    }

    private static void FillSmallInteriorAlphaDips(float[] alpha, int width, int height, float lowAlphaThreshold, int maxArea, int boundaryBoxMax, int radius)
    {
        if (alpha == null || alpha.Length != width * height || width <= 0 || height <= 0 || maxArea <= 0 || radius <= 0)
            return;

        var support = new bool[alpha.Length];
        for (var i = 0; i < alpha.Length; i++)
            support[i] = alpha[i] > 0.02f;

        var boundary = new bool[alpha.Length];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var index = y * width + x;
                if (!support[index])
                    continue;

                if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
                {
                    boundary[index] = true;
                    continue;
                }

                var touchesOutside = false;
                for (var ny = y - 1; ny <= y + 1 && !touchesOutside; ny++)
                {
                    var row = ny * width;
                    for (var nx = x - 1; nx <= x + 1; nx++)
                    {
                        if (!support[row + nx])
                        {
                            touchesOutside = true;
                            break;
                        }
                    }
                }

                boundary[index] = touchesOutside;
            }
        }

        var lowMask = new bool[alpha.Length];
        for (var i = 0; i < alpha.Length; i++)
            lowMask[i] = support[i] && alpha[i] < lowAlphaThreshold;

        var labels = new int[alpha.Length];
        var queue = new int[alpha.Length];
        var nextLabel = 1;
        var labelArea = new Dictionary<int, int>();
        var labelMinX = new Dictionary<int, int>();
        var labelMaxX = new Dictionary<int, int>();
        var labelMinY = new Dictionary<int, int>();
        var labelMaxY = new Dictionary<int, int>();
        var labelTouchesBoundary = new Dictionary<int, bool>();

        for (var start = 0; start < lowMask.Length; start++)
        {
            if (!lowMask[start] || labels[start] != 0)
                continue;

            var head = 0;
            var tail = 0;
            queue[tail++] = start;
            labels[start] = nextLabel;
            var area = 0;
            var minX = width;
            var maxX = -1;
            var minY = height;
            var maxY = -1;
            var touchesBoundary = false;

            while (head < tail)
            {
                var index = queue[head++];
                area++;
                var x = index % width;
                var y = index / width;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                if (boundary[index])
                    touchesBoundary = true;

                for (var ny = Mathf.Max(0, y - 1); ny <= Mathf.Min(height - 1, y + 1); ny++)
                {
                    var row = ny * width;
                    for (var nx = Mathf.Max(0, x - 1); nx <= Mathf.Min(width - 1, x + 1); nx++)
                    {
                        var ni = row + nx;
                        if (!lowMask[ni] || labels[ni] != 0)
                            continue;
                        labels[ni] = nextLabel;
                        queue[tail++] = ni;
                    }
                }
            }

            labelArea[nextLabel] = area;
            labelMinX[nextLabel] = minX;
            labelMaxX[nextLabel] = maxX;
            labelMinY[nextLabel] = minY;
            labelMaxY[nextLabel] = maxY;
            labelTouchesBoundary[nextLabel] = touchesBoundary;
            nextLabel++;
        }

        for (var i = 0; i < alpha.Length; i++)
        {
            var label = labels[i];
            if (label == 0)
                continue;
            if (!labelArea.TryGetValue(label, out var area) || area > maxArea)
                continue;

            var boxW = labelMaxX[label] - labelMinX[label] + 1;
            var boxH = labelMaxY[label] - labelMinY[label] + 1;
            if (labelTouchesBoundary[label] && (boxW > boundaryBoxMax || boxH > boundaryBoxMax))
                continue;

            var x = i % width;
            var y = i / width;
            var best = alpha[i];
            for (var ny = Mathf.Max(0, y - radius); ny <= Mathf.Min(height - 1, y + radius); ny++)
            {
                var row = ny * width;
                for (var nx = Mathf.Max(0, x - radius); nx <= Mathf.Min(width - 1, x + radius); nx++)
                {
                    var v = alpha[row + nx];
                    if (v > best)
                        best = v;
                }
            }

            alpha[i] = best;
        }
    }

    private static bool[] MorphClose(bool[] mask, int width, int height, int radius)
    {
        var dilated = MorphDilate(mask, width, height, radius);
        return MorphErode(dilated, width, height, radius);
    }

    private static bool[] MorphDilate(bool[] mask, int width, int height, int radius)
    {
        var result = new bool[mask.Length];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var set = false;
                for (var ny = Mathf.Max(0, y - radius); ny <= Mathf.Min(height - 1, y + radius) && !set; ny++)
                {
                    var row = ny * width;
                    for (var nx = Mathf.Max(0, x - radius); nx <= Mathf.Min(width - 1, x + radius); nx++)
                    {
                        if (mask[row + nx])
                        {
                            set = true;
                            break;
                        }
                    }
                }

                result[y * width + x] = set;
            }
        }

        return result;
    }

    private static bool[] MorphErode(bool[] mask, int width, int height, int radius)
    {
        var result = new bool[mask.Length];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var keep = true;
                for (var ny = Mathf.Max(0, y - radius); ny <= Mathf.Min(height - 1, y + radius) && keep; ny++)
                {
                    var row = ny * width;
                    for (var nx = Mathf.Max(0, x - radius); nx <= Mathf.Min(width - 1, x + radius); nx++)
                    {
                        if (!mask[row + nx])
                        {
                            keep = false;
                            break;
                        }
                    }
                }

                result[y * width + x] = keep;
            }
        }

        return result;
    }

    private static void ApplyGrayCloseInMask(float[] alpha, bool[] keep, int width, int height, int radius)
    {
        var dilated = new float[alpha.Length];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var index = y * width + x;
                if (!keep[index])
                    continue;

                var best = 0f;
                for (var ny = Mathf.Max(0, y - radius); ny <= Mathf.Min(height - 1, y + radius); ny++)
                {
                    var row = ny * width;
                    for (var nx = Mathf.Max(0, x - radius); nx <= Mathf.Min(width - 1, x + radius); nx++)
                    {
                        var ni = row + nx;
                        if (!keep[ni])
                            continue;
                        if (alpha[ni] > best)
                            best = alpha[ni];
                    }
                }

                dilated[index] = best;
            }
        }

        var closed = new float[alpha.Length];
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var index = y * width + x;
                if (!keep[index])
                    continue;

                var best = 1f;
                var hasValue = false;
                for (var ny = Mathf.Max(0, y - radius); ny <= Mathf.Min(height - 1, y + radius); ny++)
                {
                    var row = ny * width;
                    for (var nx = Mathf.Max(0, x - radius); nx <= Mathf.Min(width - 1, x + radius); nx++)
                    {
                        var ni = row + nx;
                        if (!keep[ni])
                            continue;
                        if (!hasValue || dilated[ni] < best)
                            best = dilated[ni];
                        hasValue = true;
                    }
                }

                closed[index] = hasValue ? best : alpha[index];
            }
        }

        for (var i = 0; i < alpha.Length; i++)
        {
            if (keep[i])
                alpha[i] = Mathf.Clamp01(closed[i]);
        }
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

}
