#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

public static class AexisApplicationExampleInstaller
{
    private const string Destination = "Assets/StreamingAssets";
    private const string WindowsBuildOutputEnvironmentVariable = "AEXIS_MAIN2_WINDOWS_BUILD_OUTPUT";

    [MenuItem("Aexis/Examples/Install Main2 Application StreamingAssets")]
    private static void InstallStreamingAssets()
    {
        var root = ResolveSampleRoot();
        var copiedFiles = CopyDirectory(Path.Combine(root, "StreamingAssets"), Path.GetFullPath(Destination), overwrite: true);
        AssetDatabase.Refresh();
        Debug.Log("Installed " + copiedFiles + " AIImage application StreamingAssets files to " + Destination + ".");
    }

    [MenuItem("Aexis/Examples/Open Main2 Application Scene")]
    private static void OpenMain2Scene()
    {
        EditorSceneManager.OpenScene(AexisApplicationExamplePaths.Main2SceneAssetPath, OpenSceneMode.Single);
    }

    [MenuItem("Aexis/Examples/Build Main2 Windows Player")]
    public static void BuildMain2Windows64()
    {
        BuildMain2Windows64Internal();
    }

    public static void BuildMain2Windows64Batch()
    {
        try
        {
            BuildMain2Windows64Internal();
            EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            EditorApplication.Exit(1);
        }
    }

    internal static void EnsureStreamingAssetsInstalled()
    {
        var root = ResolveSampleRoot();
        var source = Path.Combine(root, "StreamingAssets");
        var destination = Path.GetFullPath(Destination);
        if (!Directory.Exists(source))
            throw new DirectoryNotFoundException("AIImage application sample StreamingAssets are missing: " + source);

        // Player builds can only read Assets/StreamingAssets. The editor resolver may
        // fall back to the package payload, so copy every missing file instead of
        // checking a small sentinel subset of model binaries.
        var copiedFiles = CopyDirectory(source, destination, overwrite: false);
        if (copiedFiles == 0)
            return;

        AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
        Debug.Log("Installed " + copiedFiles + " missing AIImage application StreamingAssets files to " + Destination + ".");
    }

    private static void BuildMain2Windows64Internal()
    {
        const BuildTarget target = BuildTarget.StandaloneWindows64;
        var output = Environment.GetEnvironmentVariable(WindowsBuildOutputEnvironmentVariable);
        if (string.IsNullOrWhiteSpace(output))
        {
            var projectRoot = Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
            output = Path.Combine(projectRoot, "Builds", "AexisMain2", "AexisMain2.exe");
        }
        output = Path.GetFullPath(output);
        Directory.CreateDirectory(Path.GetDirectoryName(output));

        var report = AIImageReducedModelBuild.BuildMain2(target, output, BuildOptions.None);
        if (report.summary.result != BuildResult.Succeeded || report.summary.totalErrors != 0)
        {
            throw new InvalidOperationException(
                "Main2 Windows Player build failed: result=" + report.summary.result
                + " errors=" + report.summary.totalErrors
                + " output=" + output);
        }

        Debug.Log("Built Main2 Windows Player: " + output);
    }

    private static string ResolveSampleRoot()
    {
        return AexisApplicationExamplePaths.SampleRootAbsolutePath;
    }

    private static int CopyDirectory(string source, string destination, bool overwrite)
    {
        if (!Directory.Exists(source))
            return 0;

        var copiedFiles = 0;
        Directory.CreateDirectory(destination);
        foreach (var directory in Directory.GetDirectories(source, "*", SearchOption.AllDirectories))
            Directory.CreateDirectory(directory.Replace(source, destination));
        foreach (var file in Directory.GetFiles(source, "*", SearchOption.AllDirectories))
        {
            if (string.Equals(Path.GetExtension(file), ".meta", StringComparison.OrdinalIgnoreCase))
                continue;
            var target = file.Replace(source, destination);
            if (overwrite || !File.Exists(target))
            {
                File.Copy(file, target, true);
                copiedFiles++;
            }
        }

        return copiedFiles;
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
