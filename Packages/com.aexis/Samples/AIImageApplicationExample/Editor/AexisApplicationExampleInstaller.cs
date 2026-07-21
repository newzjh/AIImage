#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

internal static class AexisApplicationExampleInstaller
{
    private const string Destination = "Assets/StreamingAssets";

    [MenuItem("Aexis/Examples/Install Main2 Application StreamingAssets")]
    private static void InstallStreamingAssets()
    {
        var root = ResolveSampleRoot();
        CopyDirectory(Path.Combine(root, "StreamingAssets"), Path.GetFullPath(Destination), overwrite: true);
        AssetDatabase.Refresh();
        Debug.Log("Installed AIImage application configuration and permitted default models to " + Destination + ".");
    }

    [MenuItem("Aexis/Examples/Open Main2 Application Scene")]
    private static void OpenMain2Scene()
    {
        EditorSceneManager.OpenScene(AexisApplicationExamplePaths.Main2SceneAssetPath, OpenSceneMode.Single);
    }

    internal static void EnsureStreamingAssetsInstalled()
    {
        var root = ResolveSampleRoot();
        var source = Path.Combine(root, "StreamingAssets");
        var destination = Path.GetFullPath(Destination);
        if (!Directory.Exists(source))
            throw new DirectoryNotFoundException("AIImage application sample StreamingAssets are missing: " + source);
        if (HasDefaultModelPayload(destination))
            return;

        CopyDirectory(source, destination, overwrite: false);
        AssetDatabase.Refresh();
        Debug.Log("Installed missing AIImage application configuration and permitted default models to " + Destination + ".");
    }

    private static string ResolveSampleRoot()
    {
        return AexisApplicationExamplePaths.SampleRootAbsolutePath;
    }

    private static bool HasDefaultModelPayload(string root)
    {
        return File.Exists(Path.Combine(root, "Clip", "mobileclip_s0_export", "image_encoder.ncnn.bin"))
            && File.Exists(Path.Combine(root, "CodeFormer", "models", "generator.bin"))
            && File.Exists(Path.Combine(root, "DeepFileV2", "deepfillv2_case1.ncnn.bin"))
            && File.Exists(Path.Combine(root, "Matting", "matting.bin"))
            && File.Exists(Path.Combine(root, "RealESRGAN", "models", "realesrgan-x4plus.bin"))
            && File.Exists(Path.Combine(root, "Yolo", "yolov8n_seg.ncnn.bin"));
    }

    private static void CopyDirectory(string source, string destination, bool overwrite)
    {
        if (!Directory.Exists(source))
            return;
        Directory.CreateDirectory(destination);
        foreach (var directory in Directory.GetDirectories(source, "*", SearchOption.AllDirectories))
            Directory.CreateDirectory(directory.Replace(source, destination));
        foreach (var file in Directory.GetFiles(source, "*", SearchOption.AllDirectories))
        {
            if (string.Equals(Path.GetExtension(file), ".meta", StringComparison.OrdinalIgnoreCase))
                continue;
            var target = file.Replace(source, destination);
            if (overwrite || !File.Exists(target))
                File.Copy(file, target, true);
        }
    }
}

internal static class AexisApplicationExamplePaths
{
    private const string InstallerScriptName = "AexisApplicationExampleInstaller";

    internal static string SampleRootAbsolutePath
    {
        get
        {
            var guids = AssetDatabase.FindAssets(InstallerScriptName);
            if (guids.Length != 1)
                throw new InvalidOperationException("Could not resolve the AIImage application sample root.");
            var scriptPath = AssetDatabase.GUIDToAssetPath(guids[0]);
            return Path.GetFullPath(Path.Combine(Path.GetDirectoryName(scriptPath), ".."));
        }
    }

    internal static string Main2SceneAssetPath => ToAssetPath(Path.Combine(SampleRootAbsolutePath, "Scenes", "Main2.unity"));

    internal static string SampleTextureAbsolutePath(string fileName)
    {
        if (string.IsNullOrWhiteSpace(fileName))
            throw new ArgumentException("A sample texture file name is required.", nameof(fileName));
        return Path.Combine(SampleRootAbsolutePath, "Textures", fileName);
    }

    internal static string SampleTextureDirectory => Path.Combine(SampleRootAbsolutePath, "Textures");

    internal static string ResolveStreamingAssetFilePath(params string[] pathSegments)
    {
        return Aexis.Samples.AexisSampleStreamingAssets.ResolveFilePath(Path.Combine(pathSegments));
    }

    internal static string ResolveStreamingAssetDirectoryPath(params string[] pathSegments)
    {
        return Aexis.Samples.AexisSampleStreamingAssets.ResolveDirectoryPath(pathSegments == null || pathSegments.Length == 0 ? null : Path.Combine(pathSegments));
    }

    internal static string ToAbsolutePath(string assetPath)
    {
        if (string.IsNullOrWhiteSpace(assetPath))
            throw new ArgumentException("An asset path is required.", nameof(assetPath));
        if (Path.IsPathRooted(assetPath))
            return Path.GetFullPath(assetPath);
        var projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        return Path.GetFullPath(Path.Combine(projectRoot, assetPath));
    }

    private static string ToAssetPath(string absolutePath)
    {
        var projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
        return absolutePath.Substring(projectRoot.Length + 1).Replace('\\', '/');
    }
}
#endif
