using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

internal static class AexisApplicationExampleExporter
{
    private const string ScenePath = "Assets/Scenes/Main2.unity";
    private const string OutputRoot = "Packages/com.aexis/Samples~/AIImageApplicationExample";

    [MenuItem("Aexis/Examples/Export Main2 Application Sample")]
    public static void Export()
    {
        if (!File.Exists(ScenePath))
            throw new FileNotFoundException("Main2 scene was not found.", ScenePath);

        Directory.CreateDirectory(OutputRoot);
        RemoveStaleScriptMirror();
        RemoveGeneratedDirectory("ThirdParty");
        RemoveGeneratedDirectory("ThirdParty~");
        var copied = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var skipped = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

        foreach (var dependency in AssetDatabase.GetDependencies(ScenePath, true))
            CopyAsset(dependency, copied, skipped);

        CopyDirectory("Assets/Scripts", "Runtime/Scripts", copied);
        CopyDirectory("Assets/Editor", "Editor", copied);
        RemoveGeneratedFile("Editor/AexisApplicationExampleExporter.cs.meta");
        CopyDirectory("Assets/Resources", "Resources", copied);
        CopyDirectory("Assets/Textures", "Textures", copied);
        CopyDirectory("Assets/UI Toolkit", "UI Toolkit", copied);
        CopyDirectory("Assets/Settings", "Settings", copied);
        CopyShadedSourceDirectory(
            "Assets/Packages/SharpZip",
            "ThirdParty/AexisSampleSharpZip",
            "ICSharpCode.SharpZipLib",
            "Aexis.Samples.SharpZipLib");
        CopyShadedUniTaskRuntime();
        WriteThirdPartyNotice();
        RewriteApplicationNamespaces("Runtime/Scripts");
        RewriteApplicationNamespaces("Editor");
        CopyStreamingConfiguration(copied);

        var manifest = new AexisApplicationExampleManifest
        {
            scene = "Scenes/Main2.unity",
            copiedAssetCount = copied.Count,
            skippedDependencies = skipped.OrderBy(path => path, StringComparer.OrdinalIgnoreCase).ToArray(),
            generatedUtc = DateTime.UtcNow.ToString("O")
        };
        File.WriteAllText(Path.Combine(OutputRoot, "export-manifest.json"), JsonUtility.ToJson(manifest, true));
        AssetDatabase.Refresh();
    }

    private static void CopyStreamingConfiguration(ISet<string> copied)
    {
        const string source = "Assets/StreamingAssets";
        if (!Directory.Exists(source))
            return;

        foreach (var path in Directory.GetFiles(source, "*", SearchOption.AllDirectories))
        {
            if (path.EndsWith(".meta", StringComparison.OrdinalIgnoreCase))
                continue;
            var extension = Path.GetExtension(path);
            if (!string.Equals(extension, ".json", StringComparison.OrdinalIgnoreCase)
                && !string.Equals(extension, ".txt", StringComparison.OrdinalIgnoreCase)
                && !string.Equals(extension, ".md", StringComparison.OrdinalIgnoreCase)
                && !string.Equals(extension, ".license", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            var relative = path.Substring(source.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            CopyFile(path, Path.Combine(OutputRoot, "StreamingAssets", relative), copied);
        }
    }

    private static void CopyDirectory(string source, string destinationRelative, ISet<string> copied)
    {
        if (!Directory.Exists(source))
            return;

        foreach (var path in Directory.GetFiles(source, "*", SearchOption.AllDirectories))
        {
            if (path.EndsWith(".meta", StringComparison.OrdinalIgnoreCase))
                continue;
            if (string.Equals(Path.GetFileName(path), nameof(AexisApplicationExampleExporter) + ".cs", StringComparison.Ordinal))
                continue;
            var relative = path.Substring(source.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            CopyFile(path, Path.Combine(OutputRoot, destinationRelative, relative), copied);
        }
    }

    private static void CopyShadedUniTaskRuntime()
    {
        const string source = "Assets/Packages/Unitask/src/UniTask/Assets/Plugins/UniTask/Runtime";
        const string destinationRelative = "ThirdParty/AexisSampleAsync";
        RemoveGeneratedDirectory(destinationRelative);
        foreach (var path in Directory.GetFiles(source, "*.cs", SearchOption.AllDirectories))
        {
            var normalized = path.Replace('\\', '/');
            if (normalized.Contains("/External/") || normalized.Contains("/Linq/"))
                continue;
            var relative = path.Substring(source.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            var destination = Path.Combine(OutputRoot, destinationRelative, relative);
            Directory.CreateDirectory(Path.GetDirectoryName(destination));
            var content = File.ReadAllText(path).Replace("Cysharp.Threading.Tasks", "Aexis.Samples.Async");
            File.WriteAllText(destination, content);
        }

        File.WriteAllText(
            Path.Combine(OutputRoot, destinationRelative, "Aexis.Sample.Async.asmdef"),
            "{\n  \"name\": \"Aexis.Sample.Async\",\n  \"rootNamespace\": \"Aexis.Samples.Async\",\n  \"references\": [],\n  \"includePlatforms\": [],\n  \"excludePlatforms\": [],\n  \"allowUnsafeCode\": false,\n  \"overrideReferences\": false,\n  \"precompiledReferences\": [],\n  \"autoReferenced\": true,\n  \"defineConstraints\": [],\n  \"versionDefines\": [\n    { \"name\": \"com.unity.modules.assetbundle\", \"expression\": \"\", \"define\": \"UNITASK_ASSETBUNDLE_SUPPORT\" },\n    { \"name\": \"com.unity.modules.physics\", \"expression\": \"\", \"define\": \"UNITASK_PHYSICS_SUPPORT\" },\n    { \"name\": \"com.unity.modules.physics2d\", \"expression\": \"\", \"define\": \"UNITASK_PHYSICS2D_SUPPORT\" },\n    { \"name\": \"com.unity.modules.particlesystem\", \"expression\": \"\", \"define\": \"UNITASK_PARTICLESYSTEM_SUPPORT\" },\n    { \"name\": \"com.unity.ugui\", \"expression\": \"\", \"define\": \"UNITASK_UGUI_SUPPORT\" },\n    { \"name\": \"com.unity.modules.unitywebrequest\", \"expression\": \"\", \"define\": \"UNITASK_WEBREQUEST_SUPPORT\" }\n  ],\n  \"noEngineReferences\": false\n}\n");
    }

    private static void CopyShadedSourceDirectory(string source, string destinationRelative, string sourceNamespace, string destinationNamespace)
    {
        RemoveGeneratedDirectory(destinationRelative);
        foreach (var path in Directory.GetFiles(source, "*.cs", SearchOption.AllDirectories))
        {
            var relative = path.Substring(source.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            var destination = Path.Combine(OutputRoot, destinationRelative, relative);
            Directory.CreateDirectory(Path.GetDirectoryName(destination));
            File.WriteAllText(destination, File.ReadAllText(path).Replace(sourceNamespace, destinationNamespace));
        }
    }

    private static void WriteThirdPartyNotice()
    {
        var path = Path.Combine(OutputRoot, "ThirdParty", "NOTICE.md");
        File.WriteAllText(path,
            "# Sample Dependency Notice\n\n"
            + "This directory is part of the application example only. It is not referenced by Aexis Runtime.\n\n"
            + "- `AexisSampleAsync` is a namespace-isolated source copy derived from UniTask. Its public namespaces are rewritten from `Cysharp.Threading.Tasks` to `Aexis.Samples.Async`.\n"
            + "- `AexisSampleSharpZip` is a namespace-isolated source copy derived from SharpZipLib. Its public namespaces are rewritten from `ICSharpCode.SharpZipLib` to `Aexis.Samples.SharpZipLib`.\n\n"
            + "The rewrite prevents import collisions but does not alter upstream licenses. Preserve source headers and complete the upstream URL, immutable revision, license, modification, and redistribution review before publishing.\n");
    }

    private static void RewriteApplicationNamespaces(string directoryRelative)
    {
        var directory = Path.Combine(OutputRoot, directoryRelative);
        foreach (var path in Directory.GetFiles(directory, "*.cs", SearchOption.AllDirectories))
        {
            var content = File.ReadAllText(path)
                .Replace("Cysharp.Threading.Tasks", "Aexis.Samples.Async")
                .Replace("ICSharpCode.SharpZipLib", "Aexis.Samples.SharpZipLib");
            File.WriteAllText(path, content);
        }
    }

    private static void RemoveGeneratedDirectory(string relativePath)
    {
        var path = Path.Combine(OutputRoot, relativePath);
        if (Directory.Exists(path))
            Directory.Delete(path, true);
    }

    private static void RemoveGeneratedFile(string relativePath)
    {
        var path = Path.Combine(OutputRoot, relativePath);
        if (File.Exists(path))
            File.Delete(path);
    }

    private static void CopyAsset(string assetPath, ISet<string> copied, ISet<string> skipped)
    {
        if (string.IsNullOrWhiteSpace(assetPath) || !assetPath.StartsWith("Assets/", StringComparison.OrdinalIgnoreCase))
            return;
        if (assetPath.StartsWith("Assets/StreamingAssets/", StringComparison.OrdinalIgnoreCase))
        {
            skipped.Add(assetPath);
            return;
        }
        if (assetPath.StartsWith("Assets/Scripts/", StringComparison.OrdinalIgnoreCase))
        {
            skipped.Add(assetPath);
            return;
        }

        var relative = assetPath.Substring("Assets/".Length);
        CopyFile(assetPath, Path.Combine(OutputRoot, relative), copied);
    }

    private static void RemoveStaleScriptMirror()
    {
        var stalePath = Path.Combine(OutputRoot, "Scripts");
        if (Directory.Exists(stalePath))
            Directory.Delete(stalePath, true);
    }

    private static void CopyFile(string source, string destination, ISet<string> copied)
    {
        var sourceFullPath = Path.GetFullPath(source);
        if (!File.Exists(sourceFullPath))
            return;

        var destinationFullPath = Path.GetFullPath(destination);
        Directory.CreateDirectory(Path.GetDirectoryName(destinationFullPath));
        File.Copy(sourceFullPath, destinationFullPath, true);

        var metaSource = sourceFullPath + ".meta";
        if (File.Exists(metaSource))
            File.Copy(metaSource, destinationFullPath + ".meta", true);
        copied.Add(source.Replace('\\', '/'));
    }

    [Serializable]
    private sealed class AexisApplicationExampleManifest
    {
        public string scene;
        public int copiedAssetCount;
        public string[] skippedDependencies;
        public string generatedUtc;
    }
}
