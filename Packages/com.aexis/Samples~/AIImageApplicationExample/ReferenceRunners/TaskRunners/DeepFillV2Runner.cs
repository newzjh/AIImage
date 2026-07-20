using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Threading;
using Aexis.Ncnn;
using UnityEngine;

namespace Aexis.Samples.Runners
{

public enum DeepFillV2Backend
{
    OnnxDirect = 0,
    NcnnBin = 1
}

public struct DeepFillV2Result
{
    public Texture2D texture;
    public string error;
    public long elapsedMs;
    public long loadElapsedMs;
    public long inferenceElapsedMs;
    public DeepFillV2Backend backend;
    public string dumpDir;
    public string modelReport;
}

public sealed class DeepFillV2Runner : MonoBehaviour
{
    private const string MaskInputName = "mask";
    private const string ImageInputName = "image";
    private const string OutputName = "out0";
    private const int InputWidth = 400;
    private const int InputHeight = 512;

    public DeepFillV2Backend backend = DeepFillV2Backend.OnnxDirect;
    public string sourceOnnxRelativePath = "DeepFileV2/deepfillv2_case1.source.onnx";
    public string ncnnParamRelativePath = "DeepFileV2/deepfillv2_case1.ncnn.param";
    public string ncnnBinRelativePath = "DeepFileV2/deepfillv2_case1.ncnn.bin";
    public NcnnPrecisionMode precisionMode = NcnnPrecisionMode.Auto;
    public bool flipYInput = true;
    public bool flipYOutput = true;
    public bool preserveUnmaskedPixels = true;
    public bool enableDebugDump = true;
    public bool useArgbFloatTensor = true;
    public bool enableGeneralTextureConvolution = true;
    public bool enableDepthWiseTextureConvolution = true;
    public bool enableConv1x1TextureConvolution = true;
    public bool enableLayerPathDebugLog = false;
    public string debugTensorBlobName = string.Empty;

    private NcnnOps _ops;
    private NcnnGraphSession _repro;
    private DeepFillV2Backend _loadedBackend;
    private string _loadedSignature = string.Empty;
    private bool _hasLoaded;
    private bool _hasAppliedPrecisionMode;
    private NcnnPrecisionMode _appliedPrecisionMode;
    private DeepFillV2OnnxNcnnImportReport _lastOnnxReport;
    private string _lastDumpDir;

    public async Awaitable<DeepFillV2Result> ProcessAsync(Texture sourceImage, Texture maskImage, CancellationToken ct)
    {
        var sw = Stopwatch.StartNew();
        var loadMs = 0L;
        var inferMs = 0L;
        Texture2D readableSource = null;
        Texture2D readableMask = null;
        RenderTexture source512 = null;
        RenderTexture mask512 = null;
        RenderTexture sourcePack4 = null;
        RenderTexture maskPack4 = null;
        RenderTexture rgb512 = null;
        Texture2D generated512 = null;
        Texture2D generatedFull = null;

        DeepFillV2Result Finish(DeepFillV2Result result)
        {
            sw.Stop();
            result.elapsedMs = sw.ElapsedMilliseconds;
            result.loadElapsedMs = loadMs;
            result.inferenceElapsedMs = inferMs;
            result.backend = backend;
            result.dumpDir = _lastDumpDir;
            if (string.IsNullOrEmpty(result.modelReport))
                result.modelReport = BuildModelReport();
            return result;
        }

        try
        {
            if (sourceImage == null || maskImage == null)
                return Finish(new DeepFillV2Result { error = "DeepFillV2 source image or mask image is null." });

            _lastDumpDir = enableDebugDump ? CreateDumpDir() : null;
            EnsureRuntimeObjects();
            ApplyReproOptions();

            var loadSw = Stopwatch.StartNew();
            await EnsureLoadedAsync(ct);
            loadSw.Stop();
            loadMs = loadSw.ElapsedMilliseconds;
            ct.ThrowIfCancellationRequested();

            readableSource = EnsureReadable(sourceImage);
            readableMask = EnsureReadable(maskImage);
            if (readableSource == null || readableMask == null)
                return Finish(new DeepFillV2Result { error = "DeepFillV2 failed to read source or mask texture." });

            source512 = PrepareModelInput(readableSource);
            mask512 = PrepareModelInput(readableMask);
            if (source512 == null || mask512 == null)
                return Finish(new DeepFillV2Result { error = "DeepFillV2 failed to prepare the 400x512 source or mask." });

            var inputFormat = useArgbFloatTensor ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGBHalf;
            sourcePack4 = _repro.RentTempArray(InputWidth, InputHeight, 1, inputFormat);
            maskPack4 = _repro.RentTempArray(InputWidth, InputHeight, 1, inputFormat);
            _ops.PackRgbToPack4(source512, 0, 0, 1f, 1f, sourcePack4, flipYInput, 1f);
            _ops.PackRgbToPack4(mask512, 0, 0, 1f, 1f, maskPack4, flipYInput, 1f);

            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                TryWriteTexturePng(source512, Path.Combine(_lastDumpDir, "00_source_400x512.png"));
                TryWriteTexturePng(mask512, Path.Combine(_lastDumpDir, "01_mask_400x512.png"));
            }

            var textureInputs = new Dictionary<string, RenderTexture>(StringComparer.Ordinal)
            {
                [MaskInputName] = maskPack4,
                [ImageInputName] = sourcePack4
            };
            var textureShapes = new Dictionary<string, NcnnGraphSession.BufferShape>(StringComparer.Ordinal)
            {
                [MaskInputName] = new NcnnGraphSession.BufferShape(3, InputWidth, InputHeight, 1, 1),
                [ImageInputName] = new NcnnGraphSession.BufferShape(3, InputWidth, InputHeight, 1, 3)
            };
            var pinned = new HashSet<string>(StringComparer.Ordinal) { OutputName };
            var debugTensorBlobs = ParseDebugTensorBlobNames(debugTensorBlobName);
            for (var i = 0; i < debugTensorBlobs.Count; i++)
                pinned.Add(debugTensorBlobs[i]);

            var inferSw = Stopwatch.StartNew();
            using (var infer = _repro.InferWithMultiInputs(textureInputs, null, pinned, textureShapes))
            {
                if (debugTensorBlobs.Count > 0 && enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
                {
                    for (var i = 0; i < debugTensorBlobs.Count; i++)
                        TryWriteDebugTensorBlob(infer, debugTensorBlobs[i], _lastDumpDir);
                }

                var outPack4 = infer.GetTexture(OutputName);
                rgb512 = NewRenderTexture(InputWidth, InputHeight, RenderTextureFormat.ARGB32, true, "DeepFillV2.Rgb400x512");
                _ops.Pack4ToRgbScaled(outPack4, rgb512, 0.5f, 0.5f, flipYOutput);
                generated512 = ReadRenderTexture(rgb512, TextureFormat.RGBA32, false);
            }
            inferSw.Stop();
            inferMs = inferSw.ElapsedMilliseconds;

            if (generated512 == null)
                return Finish(new DeepFillV2Result { error = "DeepFillV2 output readback failed." });

            generatedFull = RestoreModelOutput(generated512, readableSource);
            if (generatedFull == null)
                return Finish(new DeepFillV2Result { error = "DeepFillV2 failed to resize output to source size." });

            var finalTexture = preserveUnmaskedPixels
                ? CompositeMasked(readableSource, generatedFull, readableMask)
                : generatedFull;
            if (preserveUnmaskedPixels)
                DestroyRuntimeObject(generatedFull);
            if (finalTexture == null)
                return Finish(new DeepFillV2Result { error = "DeepFillV2 final composite failed." });

            if (enableDebugDump && !string.IsNullOrWhiteSpace(_lastDumpDir))
            {
                TryWriteTexturePng(generated512, Path.Combine(_lastDumpDir, "02_deepfillv2_400x512.png"));
                TryWriteTexturePng(finalTexture, Path.Combine(_lastDumpDir, "03_deepfillv2_final.png"));
                WriteSummary(Path.Combine(_lastDumpDir, "summary.txt"), readableSource, readableMask, finalTexture, loadMs, inferMs);
            }

            return Finish(new DeepFillV2Result { texture = finalTexture });
        }
        catch (OperationCanceledException)
        {
            return Finish(new DeepFillV2Result { error = "Cancelled" });
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogError("[DeepFillV2] ProcessAsync failed: " + e);
            return Finish(new DeepFillV2Result { error = e.ToString() });
        }
        finally
        {
            if (sourcePack4 != null) _repro?.ReturnTempArray(sourcePack4);
            if (maskPack4 != null) _repro?.ReturnTempArray(maskPack4);
            ReleaseRenderTexture(source512);
            ReleaseRenderTexture(mask512);
            ReleaseRenderTexture(rgb512);
            if (generated512 != null) DestroyRuntimeObject(generated512);
            if (readableSource != null && !ReferenceEquals(readableSource, sourceImage)) DestroyRuntimeObject(readableSource);
            if (readableMask != null && !ReferenceEquals(readableMask, maskImage)) DestroyRuntimeObject(readableMask);
        }
    }

    private void EnsureRuntimeObjects()
    {
        if (_repro != null && _hasAppliedPrecisionMode && _appliedPrecisionMode != precisionMode)
            Release();
        _ops ??= new NcnnOps();
        if (_repro == null)
        {
            _repro = NcnnInferenceSessionFactory.Create(_ops, "deepfillv2", precisionMode);
            _appliedPrecisionMode = precisionMode;
            _hasAppliedPrecisionMode = true;
        }
    }

    private void ApplyReproOptions()
    {
        if (_repro == null)
            return;
        _repro.EnableGeneralTextureConvolution = enableGeneralTextureConvolution;
        _repro.EnableDepthWiseTextureConvolution = enableDepthWiseTextureConvolution;
        _repro.EnableConv1x1TextureConvolution = enableConv1x1TextureConvolution;
        _repro.TensorTextureFormat = useArgbFloatTensor ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGBHalf;
        _repro.DisallowBufferAccess = true;
        _repro.DisallowBufferOutputs = true;
        _repro.DisallowBufferToTextureMaterialization = true;
        _repro.DisallowInferenceTempComputeBuffers = true;
        _repro.DebugLog = enableLayerPathDebugLog ? line => UnityEngine.Debug.Log("[DeepFillV2][Layer] " + line) : null;
    }

    private async Awaitable EnsureLoadedAsync(CancellationToken ct)
    {
        var paramPath = ResolveStreamingAssetPath(ncnnParamRelativePath);
        var signature = BuildLoadSignature(paramPath);
        if (backend == DeepFillV2Backend.OnnxDirect)
            signature += "|" + BuildLoadSignature(ResolveStreamingAssetPath(sourceOnnxRelativePath));
        else
            signature += "|" + BuildLoadSignature(ResolveStreamingAssetPath(ncnnBinRelativePath));

        if (_hasLoaded && _loadedBackend == backend && string.Equals(_loadedSignature, signature, StringComparison.Ordinal))
            return;

        if (_repro == null)
            EnsureRuntimeObjects();
        _repro.Release();
        _lastOnnxReport = null;

        if (backend == DeepFillV2Backend.OnnxDirect)
        {
            var onnxPath = ResolveStreamingAssetPath(sourceOnnxRelativePath);
            var imported = DeepFillV2OnnxNcnnImporter.Import(onnxPath, paramPath);
            _lastOnnxReport = imported.report;
            using (var ms = new MemoryStream(imported.ncnnBinBytes, false))
            using (var br = new NcnnBinReader(ms))
                await _repro.LoadModelAsync(imported.paramText, br, progress => LogLoadProgress("onnx-direct", progress), ct);
        }
        else
        {
            var paramText = File.ReadAllText(paramPath);
            var binPath = ResolveStreamingAssetPath(ncnnBinRelativePath);
            using (var fs = File.OpenRead(binPath))
            using (var br = new NcnnBinReader(fs))
                await _repro.LoadModelAsync(paramText, br, progress => LogLoadProgress("ncnn-bin", progress), ct);
        }

        _loadedBackend = backend;
        _loadedSignature = signature;
        _hasLoaded = true;
    }

    private static string ResolveStreamingAssetPath(string relativePath)
    {
        if (string.IsNullOrWhiteSpace(relativePath))
            return Application.streamingAssetsPath;
        if (Path.IsPathRooted(relativePath))
            return relativePath;
        var rel = relativePath.Replace('\\', '/').TrimStart('/');
        return Path.Combine(Application.streamingAssetsPath, rel);
    }

    private static string BuildLoadSignature(string path)
    {
        if (string.IsNullOrWhiteSpace(path) || !File.Exists(path))
            return path ?? string.Empty;
        var info = new FileInfo(path);
        return info.FullName + "|" + info.Length.ToString(CultureInfo.InvariantCulture) + "|" + info.LastWriteTimeUtc.Ticks.ToString(CultureInfo.InvariantCulture);
    }

    private static void TryWriteDebugTensorBlob(NcnnGraphSession.InferResult infer, string blobName, string dumpDir)
    {
        if (infer == null || string.IsNullOrWhiteSpace(blobName) || string.IsNullOrWhiteSpace(dumpDir))
            return;
        try
        {
            var data = infer.ReadTextureDataForOutput(blobName);
            if (data == null)
                return;

            infer.TryGetLogicalShape(blobName, out var dims, out var w, out var h, out var d, out var c);
            var safeName = MakeSafeFileName(blobName);
            var tensorPath = Path.Combine(dumpDir, "tensor_" + safeName + ".f32");
            using (var fs = File.Create(tensorPath))
            using (var bw = new BinaryWriter(fs))
            {
                for (var i = 0; i < data.Length; i++)
                    bw.Write(data[i]);
            }

            var shapePath = Path.Combine(dumpDir, "tensor_" + safeName + ".shape.txt");
            File.WriteAllText(
                shapePath,
                "blob=" + blobName + Environment.NewLine
                + "dims=" + dims.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                + "w=" + w.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                + "h=" + h.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                + "d=" + d.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                + "c=" + c.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
                + "count=" + data.Length.ToString(CultureInfo.InvariantCulture) + Environment.NewLine,
                System.Text.Encoding.UTF8);
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogWarning("[DeepFillV2] Failed to dump debug tensor blob " + blobName + ": " + e.Message);
        }
    }

    private static List<string> ParseDebugTensorBlobNames(string value)
    {
        var result = new List<string>();
        if (string.IsNullOrWhiteSpace(value))
            return result;
        var tokens = value.Split(new[] { ',', ';', ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
        for (var i = 0; i < tokens.Length; i++)
        {
            var token = tokens[i].Trim();
            if (token.Length == 0 || result.Contains(token))
                continue;
            result.Add(token);
        }
        return result;
    }

    private static string MakeSafeFileName(string value)
    {
        if (string.IsNullOrWhiteSpace(value))
            return "blob";
        var chars = value.ToCharArray();
        var invalid = Path.GetInvalidFileNameChars();
        for (var i = 0; i < chars.Length; i++)
        {
            if (Array.IndexOf(invalid, chars[i]) >= 0 || chars[i] == ':' || chars[i] == '/' || chars[i] == '\\')
                chars[i] = '_';
        }
        return new string(chars);
    }

    private void LogLoadProgress(string label, NcnnGraphSession.LoadProgress progress)
    {
        if (progress.stage == "complete")
            UnityEngine.Debug.Log("[DeepFillV2] LoadModel complete | backend=" + label + " | layers=" + progress.layerCount.ToString(CultureInfo.InvariantCulture));
    }

    private string BuildModelReport()
    {
        if (backend != DeepFillV2Backend.OnnxDirect || _lastOnnxReport == null)
            return "backend=" + backend;
        return "backend=OnnxDirect"
               + " onnxBytes=" + _lastOnnxReport.onnxBytes.ToString(CultureInfo.InvariantCulture)
               + " generatedBinBytes=" + _lastOnnxReport.generatedBinBytes.ToString(CultureInfo.InvariantCulture)
               + " conv=" + _lastOnnxReport.convNodeCount.ToString(CultureInfo.InvariantCulture)
               + " extractImagePatches=" + _lastOnnxReport.extractImagePatchesNodeCount.ToString(CultureInfo.InvariantCulture)
               + " sha256=" + _lastOnnxReport.generatedBinSha256;
    }

    private static Texture2D EnsureReadable(Texture texture)
    {
        if (texture == null)
            return null;
        if (texture is Texture2D tex2d && tex2d.isReadable)
            return tex2d;
        var rt = RenderTexture.GetTemporary(Mathf.Max(1, texture.width), Mathf.Max(1, texture.height), 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Default);
        try
        {
            Graphics.Blit(texture, rt);
            return ReadRenderTexture(rt, TextureFormat.RGBA32, false);
        }
        finally
        {
            if (RenderTexture.active == rt)
                RenderTexture.active = null;
            RenderTexture.ReleaseTemporary(rt);
        }
    }

    private static RenderTexture PrepareModelInput(Texture source)
    {
        if (source == null)
            return null;
        return ResizeTexture(source, InputWidth, InputHeight, RenderTextureFormat.ARGB32);
    }

    private static Texture2D RestoreModelOutput(Texture2D modelOutput, Texture2D original)
    {
        if (modelOutput == null || original == null)
            return null;
        return ResizeTextureToTexture2D(modelOutput, original.width, original.height);
    }

    private static RenderTexture ResizeTexture(Texture source, int width, int height, RenderTextureFormat format)
    {
        if (source == null)
            return null;
        var rt = NewRenderTexture(width, height, format, false, "DeepFillV2.Resize");
        Graphics.Blit(source, rt);
        return rt;
    }

    private static Texture2D ResizeTextureToTexture2D(Texture source, int width, int height)
    {
        var rt = ResizeTexture(source, width, height, RenderTextureFormat.ARGB32);
        if (rt == null)
            return null;
        try
        {
            return ReadRenderTexture(rt, TextureFormat.RGBA32, false);
        }
        finally
        {
            ReleaseRenderTexture(rt);
        }
    }

    private static RenderTexture NewRenderTexture(int width, int height, RenderTextureFormat format, bool randomWrite, string name)
    {
        var rt = new RenderTexture(Mathf.Max(1, width), Mathf.Max(1, height), 0, format, RenderTextureReadWrite.Default)
        {
            enableRandomWrite = randomWrite,
            useMipMap = false,
            autoGenerateMips = false,
            name = name ?? "DeepFillV2.RT"
        };
        rt.Create();
        return rt;
    }

    private static Texture2D ReadRenderTexture(RenderTexture rt, TextureFormat format, bool linear)
    {
        if (rt == null)
            return null;
        var prev = RenderTexture.active;
        try
        {
            RenderTexture.active = rt;
            var tex = new Texture2D(rt.width, rt.height, format, false, linear);
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0, false);
            tex.Apply(false, false);
            return tex;
        }
        finally
        {
            RenderTexture.active = prev;
        }
    }

    private static Texture2D CompositeMasked(Texture2D original, Texture2D candidate, Texture2D mask)
    {
        if (original == null || candidate == null || mask == null)
            return null;
        var originalSizedMask = mask.width == original.width && mask.height == original.height
            ? mask
            : ResizeTextureToTexture2D(mask, original.width, original.height);
        try
        {
            var src = original.GetPixels32();
            var gen = candidate.GetPixels32();
            var maskPixels = originalSizedMask.GetPixels32();
            var count = Mathf.Min(src.Length, Mathf.Min(gen.Length, maskPixels.Length));
            var dst = new Color32[src.Length];
            for (var i = 0; i < count; i++)
            {
                var m = Mathf.Max(maskPixels[i].r, Mathf.Max(maskPixels[i].g, maskPixels[i].b)) / 255f;
                dst[i] = Lerp(src[i], gen[i], m);
            }
            for (var i = count; i < dst.Length; i++)
                dst[i] = src[i];
            var tex = new Texture2D(original.width, original.height, TextureFormat.RGBA32, false);
            tex.SetPixels32(dst);
            tex.Apply(false, false);
            return tex;
        }
        finally
        {
            if (originalSizedMask != null && !ReferenceEquals(originalSizedMask, mask))
                DestroyRuntimeObject(originalSizedMask);
        }
    }

    private static Color32 Lerp(Color32 a, Color32 b, float t)
    {
        t = Mathf.Clamp01(t);
        return new Color32(
            (byte)Mathf.RoundToInt(a.r + (b.r - a.r) * t),
            (byte)Mathf.RoundToInt(a.g + (b.g - a.g) * t),
            (byte)Mathf.RoundToInt(a.b + (b.b - a.b) * t),
            255);
    }

    private static string CreateDumpDir()
    {
        var root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "YanQi", "Aexis");
        var dir = Path.Combine(root, "Aexis_DeepFillV2_" + DateTime.Now.ToString("yyyyMMdd-HHmmss-fff", CultureInfo.InvariantCulture));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private void WriteSummary(string path, Texture2D source, Texture2D mask, Texture2D result, long loadMs, long inferMs)
    {
        var coverage = ComputeMaskCoverage(mask);
        var diff = ComputeMaskedMeanAbsDiff(source, result, mask, out var maskedPixels);
        File.WriteAllText(
            path,
            "backend=" + backend + Environment.NewLine
            + "load_ms=" + loadMs.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
            + "inference_ms=" + inferMs.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
            + "mask_coverage=" + coverage.ToString("0.000000", CultureInfo.InvariantCulture) + Environment.NewLine
            + "masked_pixels=" + maskedPixels.ToString(CultureInfo.InvariantCulture) + Environment.NewLine
            + "masked_mean_abs_diff_rgb=" + diff.ToString("0.000000", CultureInfo.InvariantCulture) + Environment.NewLine
            + "model_report=" + BuildModelReport() + Environment.NewLine);
    }

    public static float ComputeMaskedMeanAbsDiff(Texture2D source, Texture2D candidate, Texture2D mask, out int maskedPixels)
    {
        maskedPixels = 0;
        if (source == null || candidate == null || mask == null)
            return 0f;
        var s = source.GetPixels32();
        var c = candidate.GetPixels32();
        var m = mask.width == source.width && mask.height == source.height
            ? mask.GetPixels32()
            : ResizeTextureToTexture2D(mask, source.width, source.height)?.GetPixels32();
        if (m == null)
            return 0f;
        var count = Mathf.Min(s.Length, Mathf.Min(c.Length, m.Length));
        double sum = 0;
        for (var i = 0; i < count; i++)
        {
            var include = Mathf.Max(m[i].r, Mathf.Max(m[i].g, m[i].b)) >= 128;
            if (!include)
                continue;
            sum += (Math.Abs(s[i].r - c[i].r) + Math.Abs(s[i].g - c[i].g) + Math.Abs(s[i].b - c[i].b)) / (3.0 * 255.0);
            maskedPixels++;
        }
        return maskedPixels > 0 ? (float)(sum / maskedPixels) : 0f;
    }

    public static float ComputeMaskCoverage(Texture2D mask)
    {
        if (mask == null)
            return 0f;
        var pixels = mask.GetPixels32();
        if (pixels == null || pixels.Length == 0)
            return 0f;
        var count = 0;
        for (var i = 0; i < pixels.Length; i++)
        {
            if (Mathf.Max(pixels[i].r, Mathf.Max(pixels[i].g, pixels[i].b)) >= 128)
                count++;
        }
        return count / (float)pixels.Length;
    }

    private static void TryWriteTexturePng(Texture texture, string path)
    {
        if (texture == null || string.IsNullOrWhiteSpace(path))
            return;
        Texture2D tex = null;
        try
        {
            tex = texture as Texture2D ?? ResizeTextureToTexture2D(texture, texture.width, texture.height);
            if (tex == null)
                return;
            Directory.CreateDirectory(Path.GetDirectoryName(path) ?? ".");
            File.WriteAllBytes(path, tex.EncodeToPNG());
        }
        catch (Exception e)
        {
            UnityEngine.Debug.LogWarning("[DeepFillV2] Failed to write debug PNG: " + e.Message);
        }
        finally
        {
            if (tex != null && !ReferenceEquals(tex, texture))
                DestroyRuntimeObject(tex);
        }
    }

    private static void ReleaseRenderTexture(RenderTexture rt)
    {
        if (rt == null)
            return;
        if (RenderTexture.active == rt)
            RenderTexture.active = null;
        try { rt.Release(); } catch { }
        DestroyRuntimeObject(rt);
    }

    private static void DestroyRuntimeObject(UnityEngine.Object obj)
    {
        if (obj == null)
            return;
        if (Application.isPlaying)
            Destroy(obj);
        else
            DestroyImmediate(obj);
    }

    private void OnDestroy()
    {
        Release();
    }

    public void Release()
    {
        try { _repro?.Release(); } catch { }
        _repro = null;
        _hasLoaded = false;
    }
}

}
