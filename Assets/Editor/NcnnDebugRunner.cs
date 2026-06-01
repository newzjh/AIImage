#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using Cysharp.Threading.Tasks;
using UnityEditor;
using UnityEngine;
using UnityEngine.Profiling;
using Debug = UnityEngine.Debug;

public static class NcnnDebugRunner
{
    private const string DebugInputEnvVar = "AIIMAGE_DEBUG_INPUT";
    private const string FaceBufferPathEnvVar = "AIIMAGE_FACE_BUFFER_PATH";
    private const string FaceProbThresholdEnvVar = "AIIMAGE_FACE_PROB_THRESHOLD";
    private const string FaceNmsThresholdEnvVar = "AIIMAGE_FACE_NMS_THRESHOLD";
    private const string StressCountEnvVar = "AIIMAGE_STRESS_COUNT";
    private const string StressInputDirEnvVar = "AIIMAGE_STRESS_INPUT_DIR";
    private static readonly string DefaultFaceDebugImagePath = Path.Combine(Directory.GetCurrentDirectory(), "ref", "Pa070111a.jpg");
    private static readonly string DefaultCodeFormerDebugImagePath = Path.Combine(Directory.GetCurrentDirectory(), "ref", "Pa070111a.jpg");
    private static readonly string DefaultMattingDebugImagePath = Path.Combine(Directory.GetCurrentDirectory(), "ref", "ncnn_matting-main", "test_img.jpg");
    private static readonly string DefaultMattingReferencePath = Path.Combine(Directory.GetCurrentDirectory(), "ref", "ncnn_matting-main", "test_result.jpg");

    [MenuItem("Tools/AIImage/Run NCNN Face Debug")]
    public static void RunFaceDebugMenu()
    {
        RunFaceDebug().Forget();
    }

    [MenuItem("Tools/AIImage/Run NCNN Internal Self Test")]
    public static void RunNcnnSelfTestMenu()
    {
        NcnnCompute.NcnnComputePrototypeRunner.RunSelfTestsFromUI();
    }

    [MenuItem("Tools/AIImage/Run CodeFormer Debug")]
    public static void RunCodeFormerDebugMenu()
    {
        RunCodeFormerDebug().Forget();
    }

    [MenuItem("Tools/AIImage/Run GFPGAN Debug")]
    public static void RunGfpganDebugMenu()
    {
        RunGfpganDebug().Forget();
    }

    [MenuItem("Tools/AIImage/Run Matting Debug")]
    public static void RunMattingDebugMenu()
    {
        RunMattingDebug().Forget();
    }

    [MenuItem("Tools/AIImage/Run CodeFormer Stress (60x)")]
    public static void RunCodeFormerStressMenu()
    {
        RunCodeFormerStressBatch();
    }

    public static async UniTaskVoid RunFaceDebug()
    {
        var inputPath = ResolveInputPath(DefaultFaceDebugImagePath);
        var tex = LoadTexture(inputPath);
        if (tex == null)
        {
            Debug.LogError("Failed to load debug input: " + inputPath);
            return;
        }

        var go = new GameObject("NcnnFaceDebugRunner");
        try
        {
            var face = go.AddComponent<NcnnFaceRegionGenerator>();
            face.enableNcnnFaceRegion = true;
            face.preferTexturePathForFaceDetector = ResolveFacePreferTexturePath();
            ApplyFaceThresholdOverrides(face);
            face.enableDetailedProposalDump = true;
            face.autoOpenDumpDir = false;
            var result = await face.GenerateAsync(tex, true, CancellationToken.None);
            Debug.Log("NCNN Face Debug result | error=" + (result.error ?? "") + " | dump=" + (result.dumpDir ?? ""));
            if (result.mask != null)
                UnityEngine.Object.DestroyImmediate(result.mask);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(tex);
        }
    }

    public static async void RunFaceDebugBatch()
    {
        try
        {
            Debug.Log("[NcnnDebugRunner] RunFaceDebugBatch start");
            await RunFaceDebugInternal();
            Debug.Log("[NcnnDebugRunner] RunFaceDebugBatch done");
            EditorApplication.Exit(0);
        }
        catch (Exception e)
        {
            Debug.Log("[NcnnDebugRunner] RunFaceDebugBatch failed: " + e.Message);
            Debug.LogException(e);
            EditorApplication.Exit(1);
        }
    }

    public static async UniTaskVoid RunCodeFormerDebug()
    {
        var inputPath = ResolveInputPath(DefaultCodeFormerDebugImagePath);
        var tex = LoadTexture(inputPath);
        if (tex == null)
        {
            Debug.LogError("Failed to load debug input: " + inputPath);
            return;
        }

        var go = new GameObject("CodeFormerDebugRunner");
        try
        {
            var runner = go.AddComponent<CodeFormerNcnnReproRunner2>();
            runner.enableDebugDump = true;
            runner.enableFaceRegionDebugDump = true;
            var result = await runner.ProcessAsync(tex, CancellationToken.None);
            Debug.Log("CodeFormer Debug result | error=" + (result.error ?? "") + " | elapsedMs=" + result.elapsedMs + " | dump=" + (runner.LastDumpDir ?? ""));
            if (result.texture != null)
            {
                TryWriteTexturePng(result.texture, runner.LastDumpDir, "17_full_output.png");
                UnityEngine.Object.DestroyImmediate(result.texture);
            }
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(tex);
        }
    }

    public static async UniTaskVoid RunGfpganDebug()
    {
        var inputPath = ResolveInputPath(DefaultCodeFormerDebugImagePath);
        var tex = LoadTexture(inputPath);
        if (tex == null)
        {
            Debug.LogError("Failed to load debug input: " + inputPath);
            return;
        }

        var go = new GameObject("GfpganDebugRunner");
        try
        {
            var runner = go.AddComponent<GfpganNcnnReproRunner>();
            runner.enableFaceRegionDebugDump = true;
            var result = await runner.ProcessAsync(tex, CancellationToken.None);
            Debug.Log("GFPGAN Debug result | error=" + (result.error ?? "") + " | elapsedMs=" + result.elapsedMs);
            if (result.texture != null)
            {
                var dir = CreateGenericDumpDir("AIImage_GfpganRepro");
                TryWriteTexturePng(result.texture, dir, "17_full_output.png");
                UnityEngine.Object.DestroyImmediate(result.texture);
            }
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(tex);
        }
    }

    public static async UniTaskVoid RunMattingDebug()
    {
        await RunMattingDebugInternal();
    }

    public static async void RunMattingDebugBatch()
    {
        try
        {
            Debug.Log("[NcnnDebugRunner] RunMattingDebugBatch start");
            await RunMattingDebugInternal();
            Debug.Log("[NcnnDebugRunner] RunMattingDebugBatch done");
            EditorApplication.Exit(0);
        }
        catch (Exception e)
        {
            Debug.Log("[NcnnDebugRunner] RunMattingDebugBatch failed: " + e.Message);
            Debug.LogException(e);
            EditorApplication.Exit(1);
        }
    }

    private static async UniTask RunMattingDebugInternal()
    {
        var inputPath = ResolveInputPath(DefaultMattingDebugImagePath);
        var tex = LoadTexture(inputPath);
        if (tex == null)
        {
            Debug.LogError("Failed to load debug input: " + inputPath);
            return;
        }

        var go = new GameObject("MattingDebugRunner");
        try
        {
            var runner = go.AddComponent<MatterNcnnReproRunner>();
            runner.enableDebugDump = true;
            runner.forceBufferConvolution = false;
            runner.enableWinograd23 = false;
            var result = await runner.ProcessAsync(tex, CancellationToken.None);
            Debug.Log("Matting Debug result | error=" + (result.error ?? "") + " | elapsedMs=" + result.elapsedMs + " | dump=" + (runner.LastDumpDir ?? ""));
            if (!string.IsNullOrWhiteSpace(result.error))
                return;

            var dir = CreateGenericDumpDir("AIImage_MattingRepro");
            TryWriteTexturePng(result.texture, dir, "17_composite.png");
            TryWriteTexturePng(result.matte, dir, "18_matte.png");
            TryCompareTextureWithReference(result.texture, DefaultMattingReferencePath);

            if (result.texture != null)
                UnityEngine.Object.DestroyImmediate(result.texture);
            if (result.matte != null)
                UnityEngine.Object.DestroyImmediate(result.matte);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(tex);
        }
    }

    public static async void RunCodeFormerDebugBatch()
    {
        try
        {
            Debug.Log("[NcnnDebugRunner] RunCodeFormerDebugBatch start");
            await RunCodeFormerDebugInternal();
            Debug.Log("[NcnnDebugRunner] RunCodeFormerDebugBatch done");
            EditorApplication.Exit(0);
        }
        catch (Exception e)
        {
            Debug.Log("[NcnnDebugRunner] RunCodeFormerDebugBatch failed: " + e.Message);
            Debug.LogException(e);
            EditorApplication.Exit(1);
        }
    }

    public static async void RunGfpganDebugBatch()
    {
        try
        {
            Debug.Log("[NcnnDebugRunner] RunGfpganDebugBatch start");
            await RunGfpganDebugInternal();
            Debug.Log("[NcnnDebugRunner] RunGfpganDebugBatch done");
            EditorApplication.Exit(0);
        }
        catch (Exception e)
        {
            Debug.Log("[NcnnDebugRunner] RunGfpganDebugBatch failed: " + e.Message);
            Debug.LogException(e);
            EditorApplication.Exit(1);
        }
    }

    public static async void RunCodeFormerStressBatch()
    {
        try
        {
            Debug.Log("[NcnnDebugRunner] RunCodeFormerStressBatch start");
            await RunCodeFormerStressInternal();
            Debug.Log("[NcnnDebugRunner] RunCodeFormerStressBatch done");
            EditorApplication.Exit(0);
        }
        catch (Exception e)
        {
            Debug.Log("[NcnnDebugRunner] RunCodeFormerStressBatch failed: " + e.Message);
            Debug.LogException(e);
            EditorApplication.Exit(1);
        }
    }

    private static async UniTask RunFaceDebugInternal()
    {
        var inputPath = ResolveInputPath(DefaultFaceDebugImagePath);
        var tex = LoadTexture(inputPath);
        if (tex == null)
            throw new InvalidOperationException("Failed to load debug input: " + inputPath);

        var go = new GameObject("NcnnFaceDebugRunner");
        try
        {
            var face = go.AddComponent<NcnnFaceRegionGenerator>();
            face.enableNcnnFaceRegion = true;
            face.preferTexturePathForFaceDetector = ResolveFacePreferTexturePath();
            ApplyFaceThresholdOverrides(face);
            face.enableDetailedProposalDump = true;
            face.autoOpenDumpDir = false;
            var result = await face.GenerateAsync(tex, true, CancellationToken.None);
            Debug.Log("NCNN Face Debug result | error=" + (result.error ?? "") + " | dump=" + (result.dumpDir ?? ""));
            if (result.mask != null)
                UnityEngine.Object.DestroyImmediate(result.mask);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(tex);
        }
    }

    private static async UniTask RunCodeFormerDebugInternal()
    {
        var inputPath = ResolveInputPath(DefaultCodeFormerDebugImagePath);
        var tex = LoadTexture(inputPath);
        if (tex == null)
            throw new InvalidOperationException("Failed to load debug input: " + inputPath);

        var go = new GameObject("CodeFormerDebugRunner");
        try
        {
            var runner = go.AddComponent<CodeFormerNcnnReproRunner2>();
            runner.enableDebugDump = true;
            runner.enableFaceRegionDebugDump = true;
            var result = await runner.ProcessAsync(tex, CancellationToken.None);
            Debug.Log("CodeFormer Debug result | error=" + (result.error ?? "") + " | elapsedMs=" + result.elapsedMs + " | dump=" + (runner.LastDumpDir ?? ""));
            if (result.texture != null)
            {
                TryWriteTexturePng(result.texture, runner.LastDumpDir, "17_full_output.png");
                UnityEngine.Object.DestroyImmediate(result.texture);
            }
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(tex);
        }
    }

    private static async UniTask RunGfpganDebugInternal()
    {
        var inputPath = ResolveInputPath(DefaultCodeFormerDebugImagePath);
        var tex = LoadTexture(inputPath);
        if (tex == null)
            throw new InvalidOperationException("Failed to load debug input: " + inputPath);

        var go = new GameObject("GfpganDebugRunner");
        try
        {
            var runner = go.AddComponent<GfpganNcnnReproRunner>();
            runner.enableFaceRegionDebugDump = true;
            var result = await runner.ProcessAsync(tex, CancellationToken.None);
            Debug.Log("GFPGAN Debug result | error=" + (result.error ?? "") + " | elapsedMs=" + result.elapsedMs);
            if (result.texture != null)
            {
                var dir = CreateGenericDumpDir("AIImage_GfpganRepro");
                TryWriteTexturePng(result.texture, dir, "17_full_output.png");
                UnityEngine.Object.DestroyImmediate(result.texture);
            }
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(go);
            UnityEngine.Object.DestroyImmediate(tex);
        }
    }

    private static async UniTask RunCodeFormerStressInternal()
    {
        var inputPaths = ResolveStressInputPaths(DefaultCodeFormerDebugImagePath);
        if (inputPaths.Count == 0)
            throw new InvalidOperationException("No stress inputs resolved");

        var iterations = ResolveStressCount(inputPaths.Count);
        var dumpDir = CreateGenericDumpDir("AIImage_CodeFormerStress");
        var logPath = Path.Combine(dumpDir, "stress_summary.txt");
        var lines = new List<string>(iterations + 8)
        {
            "iterations=" + iterations.ToString(CultureInfo.InvariantCulture),
            "inputs=" + string.Join(" | ", inputPaths)
        };

        NcnnCompute.NcnnGpuResourceTracker.Enabled = true;
        NcnnCompute.NcnnGpuResourceTracker.Reset("CodeFormerStress");

        var go = new GameObject("CodeFormerStressRunner");
        try
        {
            var runner = go.AddComponent<CodeFormerNcnnReproRunner2>();
            runner.enableDebugDump = false;
            runner.enableFaceRegionDebugDump = false;

            for (var i = 0; i < iterations; i++)
            {
                var inputPath = inputPaths[i % inputPaths.Count];
                var tex = LoadTexture(inputPath);
                if (tex == null)
                    throw new InvalidOperationException("Failed to load stress input: " + inputPath);

                try
                {
                    var sw = Stopwatch.StartNew();
                    var result = await runner.ProcessAsync(tex, CancellationToken.None);
                    sw.Stop();

                    var privateMb = Process.GetCurrentProcess().PrivateMemorySize64 / (1024.0 * 1024.0);
                    var managedMb = GC.GetTotalMemory(false) / (1024.0 * 1024.0);
                    var gfxMb = Profiler.GetAllocatedMemoryForGraphicsDriver() / (1024.0 * 1024.0);
                    lines.Add(
                        "iter=" + (i + 1).ToString(CultureInfo.InvariantCulture)
                        + " | file=" + Path.GetFileName(inputPath)
                        + " | elapsed_ms=" + sw.ElapsedMilliseconds.ToString(CultureInfo.InvariantCulture)
                        + " | err=" + (result.error ?? "")
                        + " | private_mb=" + privateMb.ToString("F3", CultureInfo.InvariantCulture)
                        + " | managed_mb=" + managedMb.ToString("F3", CultureInfo.InvariantCulture)
                        + " | gfx_mb=" + gfxMb.ToString("F3", CultureInfo.InvariantCulture)
                        + " | " + NcnnCompute.NcnnGpuResourceTracker.BuildSummary());

                    if (result.texture != null)
                        UnityEngine.Object.DestroyImmediate(result.texture);
                }
                finally
                {
                    UnityEngine.Object.DestroyImmediate(tex);
                }
            }
        }
        finally
        {
            try
            {
                NcnnCompute.NcnnGpuResourceTracker.WriteReport(dumpDir, "stress_gpu_resources.txt");
            }
            catch
            {
            }

            try
            {
                File.WriteAllLines(logPath, lines);
            }
            catch
            {
            }

            NcnnCompute.NcnnGpuResourceTracker.Enabled = false;
            UnityEngine.Object.DestroyImmediate(go);
        }
    }

    private static string ResolveInputPath(string fallbackPath)
    {
        try
        {
            var env = Environment.GetEnvironmentVariable(DebugInputEnvVar);
            if (!string.IsNullOrWhiteSpace(env) && File.Exists(env))
                return env;
        }
        catch
        {
        }
        return fallbackPath;
    }

    private static List<string> ResolveStressInputPaths(string fallbackPath)
    {
        try
        {
            var envDir = Environment.GetEnvironmentVariable(StressInputDirEnvVar);
            if (!string.IsNullOrWhiteSpace(envDir) && Directory.Exists(envDir))
            {
                var files = Directory.GetFiles(envDir)
                    .Where(IsImagePath)
                    .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
                    .ToList();
                if (files.Count > 0)
                    return files;
            }
        }
        catch
        {
        }

        return new List<string> { ResolveInputPath(fallbackPath) };
    }

    private static int ResolveStressCount(int inputCount)
    {
        try
        {
            var env = Environment.GetEnvironmentVariable(StressCountEnvVar);
            if (!string.IsNullOrWhiteSpace(env)
                && int.TryParse(env.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed)
                && parsed > 0)
                return parsed;
        }
        catch
        {
        }

        return Mathf.Max(60, inputCount);
    }

    private static bool IsImagePath(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return false;

        var ext = Path.GetExtension(path);
        return string.Equals(ext, ".png", StringComparison.OrdinalIgnoreCase)
            || string.Equals(ext, ".jpg", StringComparison.OrdinalIgnoreCase)
            || string.Equals(ext, ".jpeg", StringComparison.OrdinalIgnoreCase)
            || string.Equals(ext, ".bmp", StringComparison.OrdinalIgnoreCase)
            || string.Equals(ext, ".webp", StringComparison.OrdinalIgnoreCase);
    }

    private static bool ResolveFacePreferTexturePath()
    {
        try
        {
            var env = Environment.GetEnvironmentVariable(FaceBufferPathEnvVar);
            if (string.IsNullOrWhiteSpace(env))
                return true;
            env = env.Trim();
            if (string.Equals(env, "1", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "true", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "buffer", StringComparison.OrdinalIgnoreCase))
                return false;
            if (string.Equals(env, "0", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "false", StringComparison.OrdinalIgnoreCase)
                || string.Equals(env, "texture", StringComparison.OrdinalIgnoreCase))
                return true;
            return true;
        }
        catch
        {
            return true;
        }
    }

    private static void ApplyFaceThresholdOverrides(NcnnFaceRegionGenerator face)
    {
        if (face == null)
            return;

        if (TryReadFloatEnv(FaceProbThresholdEnvVar, out var prob))
            face.probThreshold = Mathf.Clamp(prob, 0.01f, 0.99f);
        if (TryReadFloatEnv(FaceNmsThresholdEnvVar, out var nms))
            face.nmsThreshold = Mathf.Clamp01(nms);
    }

    private static bool TryReadFloatEnv(string envName, out float value)
    {
        value = 0f;
        try
        {
            var env = Environment.GetEnvironmentVariable(envName);
            if (string.IsNullOrWhiteSpace(env))
                return false;
            return float.TryParse(env.Trim(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out value);
        }
        catch
        {
            return false;
        }
    }

    private static void TryWriteTexturePng(Texture2D texture, string dir, string fileName)
    {
        if (texture == null || string.IsNullOrWhiteSpace(dir) || string.IsNullOrWhiteSpace(fileName))
            return;
        try
        {
            Directory.CreateDirectory(dir);
            File.WriteAllBytes(Path.Combine(dir, fileName), texture.EncodeToPNG());
        }
        catch (Exception e)
        {
            Debug.LogWarning("Failed to write debug texture: " + e.Message);
        }
    }

    private static void TryCompareTextureWithReference(Texture2D candidate, string referencePath)
    {
        if (candidate == null || string.IsNullOrWhiteSpace(referencePath) || !File.Exists(referencePath))
            return;

        var reference = LoadTexture(referencePath);
        if (reference == null)
            return;

        try
        {
            if (candidate.width != reference.width || candidate.height != reference.height)
            {
                Debug.Log("Matting Debug compare skipped | size mismatch " + candidate.width + "x" + candidate.height + " vs " + reference.width + "x" + reference.height);
                return;
            }

            var a = candidate.GetPixels32();
            var b = reference.GetPixels32();
            double sumAbs = 0d;
            var maxAbs = 0;
            var count = a.Length * 3;
            for (var i = 0; i < a.Length; i++)
            {
                var dr = Mathf.Abs(a[i].r - b[i].r);
                var dg = Mathf.Abs(a[i].g - b[i].g);
                var db = Mathf.Abs(a[i].b - b[i].b);
                sumAbs += dr + dg + db;
                maxAbs = Mathf.Max(maxAbs, Mathf.Max(dr, Mathf.Max(dg, db)));
            }

            var meanAbs = count > 0 ? sumAbs / count : 0d;
            Debug.Log("Matting Debug compare | ref=" + referencePath + " | mean_abs_rgb=" + meanAbs.ToString("F4", CultureInfo.InvariantCulture) + " | max_abs_rgb=" + maxAbs.ToString(CultureInfo.InvariantCulture));
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(reference);
        }
    }

    private static string CreateGenericDumpDir(string prefix)
    {
        var root = Path.Combine(Path.GetTempPath(), "YanQi", "AIImage");
        Directory.CreateDirectory(root);
        var dir = Path.Combine(root, prefix + "_" + DateTime.Now.ToString("yyyyMMdd_HHmmss"));
        Directory.CreateDirectory(dir);
        return dir;
    }
    private static Texture2D LoadTexture(string path)
    {
        if (!File.Exists(path))
            return null;

        try
        {
            var bytes = File.ReadAllBytes(path);
            var tex = new Texture2D(2, 2, TextureFormat.RGBA32, false, true);
            if (!ImageConversion.LoadImage(tex, bytes, false))
            {
                UnityEngine.Object.DestroyImmediate(tex);
                return null;
            }
            tex.wrapMode = TextureWrapMode.Clamp;
            tex.filterMode = FilterMode.Bilinear;
            tex.name = Path.GetFileNameWithoutExtension(path);
            return tex;
        }
        catch (Exception e)
        {
            Debug.LogException(e);
            return null;
        }
    }
}
#endif
