#if UNITY_EDITOR
using System;
using System.IO;
using System.Threading;
using Cysharp.Threading.Tasks;
using UnityEditor;
using UnityEngine;

public static class NcnnDebugRunner
{
    private const string DebugInputEnvVar = "AIIMAGE_DEBUG_INPUT";
    private const string FaceBufferPathEnvVar = "AIIMAGE_FACE_BUFFER_PATH";
    private static readonly string DefaultFaceDebugImagePath = Path.Combine(Directory.GetCurrentDirectory(), "ref", "Pa070111a.jpg");
    private static readonly string DefaultCodeFormerDebugImagePath = Path.Combine(Directory.GetCurrentDirectory(), "ref", "Pa070111a.jpg");

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
