#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Security.Cryptography;
using Aexis.Editor;
using UnityEditor;
using UnityEngine;

public static class AIImageModelReleasePackager
{
    private const string OutputEnvironmentVariable = "AEXIS_MODEL_RELEASE_OUTPUT";

    [MenuItem("Aexis/Release/Build Reduced/Prepare Model Release Assets")]
    public static void PrepareModelReleaseAssets()
    {
        var output = GetOutputDirectory();
        Directory.CreateDirectory(output);
        var manifest = new ModelReleaseManifest
        {
            schema = "aiimage.model-release/v1",
            release_tag = Environment.GetEnvironmentVariable("AIIMAGE_MODEL_RELEASE_TAG") ?? AIImageModelDelivery.DefaultReleaseTag,
            generated_utc = DateTime.UtcNow.ToString("O")
        };

        foreach (var group in AIImageModelDelivery.AllGroups)
        {
            var files = ResolveReleaseFiles(group).ToArray();
            if (files.Length == 0)
            {
                if (group.BundledByDefault)
                    throw new InvalidOperationException("Default release model group has no source files: " + group.DisplayName);
                Debug.Log("Skipping unavailable optional release model group: " + group.DisplayName);
                continue;
            }

            var archivePath = Path.Combine(output, group.ArchiveName);
            CreateArchive(archivePath, files);
            manifest.assets.Add(new ModelReleaseAsset
            {
                id = group.Id.ToString(),
                display_name = group.DisplayName,
                bundled_by_default = group.BundledByDefault,
                archive = group.ArchiveName,
                bytes = new FileInfo(archivePath).Length,
                sha256 = ComputeSha256(archivePath),
                files = files.Select(file => file.RelativePath).ToArray()
            });
        }

        var manifestPath = Path.Combine(output, "AIImageModelReleaseManifest.json");
        File.WriteAllText(manifestPath, JsonUtility.ToJson(manifest, true));
        File.WriteAllText(
            Path.Combine(output, "UPLOAD_TO_GITHUB_RELEASE.txt"),
            "Upload every AIImageModels.*.zip and AIImageModelReleaseManifest.json to the "
            + manifest.release_tag + " release of " + AIImageModelDelivery.ReleaseOwner + "/"
            + AIImageModelDelivery.ReleaseRepository + ".\n"
            + "Runtime download URLs use the release asset names exactly as generated.\n");
        Debug.Log("Prepared " + manifest.assets.Count + " AIImage model release archives: " + output);
    }

    public static void PrepareModelReleaseAssetsBatch()
    {
        try
        {
            PrepareModelReleaseAssets();
            EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            EditorApplication.Exit(1);
        }
    }

    [MenuItem("Aexis/Release/Build Reduced/Export Complete UnityPackage")]
    public static void ExportCompleteUnityPackage()
    {
        var output = Path.Combine(GetOutputDirectory(), "Aexis-Complete.unitypackage");
        AexisUnityPackageExporter.ExportCompleteUnityPackage(output);
    }

    public static void ExportReducedMain2UnityPackage()
    {
        ExportCompleteUnityPackage();
    }

    public static void ExportReducedMain2UnityPackageBatch()
    {
        try
        {
            ExportCompleteUnityPackage();
            EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            EditorApplication.Exit(1);
        }
    }

    private static IEnumerable<ReleaseFile> ResolveReleaseFiles(AIImageModelGroup group)
    {
        var result = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (var relative in group.Files)
        {
            var source = AIImageReducedModelBuild.ResolveModelSourceFile(relative);
            if (source == null)
            {
                if (group.BundledByDefault)
                    throw new FileNotFoundException("Missing default model release source.", relative);
                return Array.Empty<ReleaseFile>();
            }
            result[relative] = source;
        }

        foreach (var prefix in group.Prefixes)
        {
            foreach (var root in EnumerateSourceRoots())
            {
                var directory = Path.Combine(root, prefix.Replace('/', Path.DirectorySeparatorChar));
                if (!Directory.Exists(directory))
                    continue;
                foreach (var source in Directory.GetFiles(directory, "*", SearchOption.AllDirectories))
                {
                    if (source.EndsWith(".meta", StringComparison.OrdinalIgnoreCase))
                        continue;
                    var relative = AIImageModelDelivery.NormalizeRelativePath(source.Substring(root.Length));
                    result[relative] = source;
                }
                break;
            }
        }

        return result
            .OrderBy(pair => pair.Key, StringComparer.OrdinalIgnoreCase)
            .Select(pair => new ReleaseFile(pair.Key, pair.Value));
    }

    private static IEnumerable<string> EnumerateSourceRoots()
    {
        yield return Path.Combine(ProjectRoot, "Assets", "StreamingAssets");
        yield return Path.Combine(AexisApplicationExamplePaths.SampleRootAbsolutePath, "StreamingAssets");
        yield return AIImageModelDelivery.PersistentRoot;
    }

    private static void CreateArchive(string archivePath, IReadOnlyList<ReleaseFile> files)
    {
        if (File.Exists(archivePath))
            File.Delete(archivePath);
        using (var archive = ZipFile.Open(archivePath, ZipArchiveMode.Create))
        {
            for (var index = 0; index < files.Count; index++)
            {
                var file = files[index];
                archive.CreateEntryFromFile(
                    file.SourcePath,
                    file.RelativePath,
                    System.IO.Compression.CompressionLevel.Optimal);
            }
        }
    }

    private static string GetOutputDirectory()
    {
        var configured = Environment.GetEnvironmentVariable(OutputEnvironmentVariable);
        return Path.GetFullPath(string.IsNullOrWhiteSpace(configured)
            ? Path.Combine(ProjectRoot, "Builds", "ReducedModelRelease")
            : configured);
    }

    private static string ComputeSha256(string path)
    {
        using (var sha = SHA256.Create())
        using (var stream = File.OpenRead(path))
            return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", string.Empty).ToLowerInvariant();
    }

    private static string ProjectRoot => Path.GetFullPath(Path.Combine(Application.dataPath, ".."));

    private readonly struct ReleaseFile
    {
        public ReleaseFile(string relativePath, string sourcePath)
        {
            RelativePath = relativePath;
            SourcePath = sourcePath;
        }

        public string RelativePath { get; }
        public string SourcePath { get; }
    }

    [Serializable]
    private sealed class ModelReleaseManifest
    {
        public string schema;
        public string release_tag;
        public string generated_utc;
        public List<ModelReleaseAsset> assets = new List<ModelReleaseAsset>();
    }

    [Serializable]
    private sealed class ModelReleaseAsset
    {
        public string id;
        public string display_name;
        public bool bundled_by_default;
        public string archive;
        public long bytes;
        public string sha256;
        public string[] files;
    }
}
#endif
