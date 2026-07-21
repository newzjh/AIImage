#if UNITY_EDITOR
using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using UnityEditor;
using UnityEngine;

namespace Aexis.Editor
{
    /// <summary>Exports the standalone engine and complete Main2 example for projects that cannot consume a UPM package.</summary>
    public static class AexisUnityPackageExporter
    {
        private const string ExportOutputEnvironmentVariable = "AEXIS_UNITYPACKAGE_OUTPUT";
        private const string ExportStagingDirectoryName = "AexisUnityPackageExport";

        [MenuItem("Aexis/Release/Export Complete UnityPackage")]
        public static void ExportCompleteUnityPackage()
        {
            var root = ResolveAexisRoot();
            var version = ReadPackageVersion(root);
            var outputPath = ResolveOutputPath(version);
            var stagingProject = CreateStagingProject(root, outputPath);
            var logPath = Path.Combine(stagingProject, "export.log");

            try
            {
                RunStagingExporter(stagingProject, logPath);
                if (!File.Exists(outputPath))
                    throw new FileNotFoundException("UnityPackage export did not produce an output file.", outputPath);

                UnityEngine.Debug.Log("Aexis UnityPackage exported: " + outputPath);
            }
            catch (Exception exception)
            {
                var log = File.Exists(logPath) ? File.ReadAllText(logPath) : "No staging log was written.";
                throw new InvalidOperationException("Aexis UnityPackage export failed. Staging project: " + stagingProject + Environment.NewLine + log, exception);
            }

            try
            {
                Directory.Delete(stagingProject, true);
            }
            catch (IOException)
            {
                UnityEngine.Debug.LogWarning("Aexis UnityPackage export succeeded, but the staging directory is still locked and was preserved: " + stagingProject);
            }
        }

        private static string ResolveAexisRoot()
        {
            var guids = AssetDatabase.FindAssets(nameof(AexisUnityPackageExporter));
            if (guids.Length != 1)
                throw new InvalidOperationException("Could not resolve the Aexis UnityPackage exporter asset.");

            var scriptPath = AssetDatabase.GUIDToAssetPath(guids[0]);
            var editorDirectory = Path.GetDirectoryName(Path.GetFullPath(scriptPath));
            return Directory.GetParent(editorDirectory).FullName;
        }

        private static string ReadPackageVersion(string root)
        {
            var packageJson = Path.Combine(root, "package.json");
            if (!File.Exists(packageJson))
                return "local";

            var package = JsonUtility.FromJson<PackageMetadata>(File.ReadAllText(packageJson));
            return string.IsNullOrWhiteSpace(package.version) ? "local" : package.version;
        }

        private static string ResolveOutputPath(string version)
        {
            var configured = Environment.GetEnvironmentVariable(ExportOutputEnvironmentVariable);
            if (!string.IsNullOrWhiteSpace(configured))
                return Path.GetFullPath(configured);

            var projectRoot = Directory.GetParent(Application.dataPath).FullName;
            return Path.Combine(projectRoot, "output", "Aexis-" + version + ".unitypackage");
        }

        private static string CreateStagingProject(string sourceRoot, string outputPath)
        {
            var projectRoot = Directory.GetParent(Application.dataPath).FullName;
            var stagingProject = Path.Combine(projectRoot, "Temp", ExportStagingDirectoryName, Guid.NewGuid().ToString("N"));
            var assetsRoot = Path.Combine(stagingProject, "Assets");
            var bootstrapRoot = Path.Combine(assetsRoot, "AexisPackageBootstrap");
            Directory.CreateDirectory(bootstrapRoot);
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath));

            CreatePackageArchive(sourceRoot, Path.Combine(bootstrapRoot, "com.aexis.zip"));
            WritePackageBootstrap(bootstrapRoot);

            var projectSettingsDirectory = Path.Combine(stagingProject, "ProjectSettings");
            var packagesDirectory = Path.Combine(stagingProject, "Packages");
            Directory.CreateDirectory(projectSettingsDirectory);
            Directory.CreateDirectory(packagesDirectory);
            File.WriteAllText(
                Path.Combine(projectSettingsDirectory, "ProjectVersion.txt"),
                "m_EditorVersion: " + Application.unityVersion + Environment.NewLine);
            File.WriteAllText(Path.Combine(packagesDirectory, "manifest.json"),
                "{\n"
                + "  \"dependencies\": {\n"
                + "    \"com.unity.modules.assetbundle\": \"1.0.0\",\n"
                + "    \"com.unity.modules.imageconversion\": \"1.0.0\",\n"
                + "    \"com.unity.modules.particlesystem\": \"1.0.0\",\n"
                + "    \"com.unity.modules.physics\": \"1.0.0\",\n"
                + "    \"com.unity.modules.physics2d\": \"1.0.0\",\n"
                + "    \"com.unity.modules.ui\": \"1.0.0\",\n"
                + "    \"com.unity.modules.uielements\": \"1.0.0\",\n"
                + "    \"com.unity.modules.unitywebrequest\": \"1.0.0\",\n"
                + "    \"com.unity.modules.unitywebrequesttexture\": \"1.0.0\",\n"
                + "    \"com.unity.test-framework\": \"1.1.33\",\n"
                + "    \"com.unity.ugui\": \"1.0.0\"\n"
                + "  }\n"
                + "}\n");
            WriteStagingExporter(assetsRoot, outputPath);
            return stagingProject;
        }

        private static void RunStagingExporter(string stagingProject, string logPath)
        {
            var arguments = "-batchmode -quit -projectPath \"" + stagingProject + "\" -executeMethod AexisUnityPackageStagingExporter.Export -logFile \"" + logPath + "\"";
            var startInfo = new ProcessStartInfo(EditorApplication.applicationPath, arguments)
            {
                CreateNoWindow = true,
                UseShellExecute = false
            };

            using (var process = Process.Start(startInfo))
            {
                process.WaitForExit();
                if (process.ExitCode != 0)
                    throw new InvalidOperationException("Staging Unity process exited with code " + process.ExitCode + ".");
            }
        }

        private static void WriteStagingExporter(string assetsRoot, string outputPath)
        {
            var escapedOutput = outputPath.Replace("\\", "/").Replace("\"", "\\\"");
            var exporterPath = Path.Combine(assetsRoot, "Editor", "AexisUnityPackageStagingExporter.cs");
            Directory.CreateDirectory(Path.GetDirectoryName(exporterPath));
            File.WriteAllText(exporterPath,
                "#if UNITY_EDITOR\n"
                + "using UnityEditor;\n"
                + "public static class AexisUnityPackageStagingExporter\n"
                + "{\n"
                + "    public static void Export()\n"
                + "    {\n"
                + "        AssetDatabase.ExportPackage(new[] { \"Assets/AexisPackageBootstrap\" }, \"" + escapedOutput + "\", ExportPackageOptions.Recurse);\n"
                + "    }\n"
                + "}\n"
                + "#endif\n");
        }

        private static void CreatePackageArchive(string sourceRoot, string archivePath)
        {
            using (var archiveStream = File.Create(archivePath))
            using (var archive = new ZipArchive(archiveStream, ZipArchiveMode.Create, false))
            {
                foreach (var file in Directory.GetFiles(sourceRoot, "*", SearchOption.AllDirectories))
                {
                    var relative = file.Substring(sourceRoot.Length + 1).Replace('\\', '/');
                    if (relative.StartsWith("UnityPackage~/", StringComparison.OrdinalIgnoreCase))
                        continue;

                    var entry = archive.CreateEntry(relative, System.IO.Compression.CompressionLevel.Fastest);
                    using (var input = File.OpenRead(file))
                    using (var entryStream = entry.Open())
                        input.CopyTo(entryStream);
                }
            }
        }

        private static void WritePackageBootstrap(string bootstrapRoot)
        {
            var editorDirectory = Path.Combine(bootstrapRoot, "Editor");
            Directory.CreateDirectory(editorDirectory);
            var bootstrapPath = Path.Combine(editorDirectory, "AexisEmbeddedPackageBootstrap.cs");
            File.WriteAllText(bootstrapPath,
                "#if UNITY_EDITOR\n"
                + "using System;\n"
                + "using System.IO;\n"
                + "using System.IO.Compression;\n"
                + "using UnityEditor;\n"
                + "using UnityEditor.PackageManager;\n"
                + "using UnityEngine;\n"
                + "\n"
                + "[InitializeOnLoad]\n"
                + "internal static class AexisEmbeddedPackageBootstrap\n"
                + "{\n"
                + "    private const string PackageName = \"com.aexis\";\n"
                + "    private const string BootstrapScriptName = \"AexisEmbeddedPackageBootstrap\";\n"
                + "    private const string CleanupMarkerRelativePath = \"Library/AexisPackageBootstrapCleanup.marker\";\n"
                + "    private static double _waitStarted;\n"
                + "\n"
                + "    static AexisEmbeddedPackageBootstrap()\n"
                + "    {\n"
                + "        EditorApplication.delayCall += Install;\n"
                + "    }\n"
                + "\n"
                + "    private static void Install()\n"
                + "    {\n"
                + "        try\n"
                + "        {\n"
                + "            var packageRoot = Path.Combine(ProjectRoot, \"Packages\", PackageName);\n"
                + "            if (File.Exists(CleanupMarkerPath) && UnityEditor.PackageManager.PackageInfo.FindForAssetPath(\"Packages/com.aexis/package.json\") != null)\n"
                + "            {\n"
                + "                File.Delete(CleanupMarkerPath);\n"
                + "                EditorApplication.delayCall += CleanupBootstrapAssets;\n"
                + "                return;\n"
                + "            }\n"
                + "            if (!Directory.Exists(packageRoot))\n"
                + "                ExtractPackage(packageRoot);\n"
                + "\n"
                + "            if (!File.Exists(Path.Combine(packageRoot, \"package.json\")))\n"
                + "                throw new FileNotFoundException(\"Aexis package.json was not restored.\", packageRoot);\n"
                + "\n"
                + "            _waitStarted = EditorApplication.timeSinceStartup;\n"
                + "            Client.Resolve();\n"
                + "            EditorApplication.update += AwaitEmbeddedPackage;\n"
                + "        }\n"
                + "        catch (Exception exception)\n"
                + "        {\n"
                + "            Debug.LogError(\"Aexis UnityPackage bootstrap failed: \" + exception);\n"
                + "        }\n"
                + "    }\n"
                + "\n"
                + "    private static void AwaitEmbeddedPackage()\n"
                + "    {\n"
                + "        var package = UnityEditor.PackageManager.PackageInfo.FindForAssetPath(\"Packages/com.aexis/package.json\");\n"
                + "        if (package != null && string.Equals(package.name, PackageName, StringComparison.Ordinal))\n"
                + "        {\n"
                + "            EditorApplication.update -= AwaitEmbeddedPackage;\n"
                + "            Debug.Log(\"Aexis package restored to Packages/com.aexis. Import the AIImage Main2 Application Example from Package Manager to run Main2.\");\n"
                + "            Directory.CreateDirectory(Path.GetDirectoryName(CleanupMarkerPath));\n"
                + "            File.WriteAllText(CleanupMarkerPath, \"delete bootstrap after the package scripts have reloaded\");\n"
                + "            return;\n"
                + "        }\n"
                + "\n"
                + "        if (EditorApplication.timeSinceStartup - _waitStarted > 120d)\n"
                + "        {\n"
                + "            EditorApplication.update -= AwaitEmbeddedPackage;\n"
                + "            Debug.LogError(\"Aexis package resolution timed out.\");\n"
                + "        }\n"
                + "    }\n"
                + "\n"
                + "    private static void ExtractPackage(string packageRoot)\n"
                + "    {\n"
                + "        var archivePath = Path.Combine(BootstrapRoot, \"com.aexis.zip\");\n"
                + "        if (!File.Exists(archivePath))\n"
                + "            throw new FileNotFoundException(\"Aexis package payload is missing.\", archivePath);\n"
                + "\n"
                + "        Directory.CreateDirectory(packageRoot);\n"
                + "        var packageRootWithSeparator = Path.GetFullPath(packageRoot) + Path.DirectorySeparatorChar;\n"
                + "        using (var archiveStream = File.OpenRead(archivePath))\n"
                + "        using (var archive = new ZipArchive(archiveStream, ZipArchiveMode.Read, false))\n"
                + "        {\n"
                + "            foreach (var entry in archive.Entries)\n"
                + "            {\n"
                + "                var destination = Path.GetFullPath(Path.Combine(packageRoot, entry.FullName));\n"
                + "                if (!destination.StartsWith(packageRootWithSeparator, StringComparison.OrdinalIgnoreCase))\n"
                + "                    throw new InvalidOperationException(\"Aexis package payload contains an invalid path: \" + entry.FullName);\n"
                + "                if (string.IsNullOrEmpty(entry.Name))\n"
                + "                {\n"
                + "                    Directory.CreateDirectory(destination);\n"
                + "                    continue;\n"
                + "                }\n"
                + "\n"
                + "                Directory.CreateDirectory(Path.GetDirectoryName(destination));\n"
                + "                using (var input = entry.Open())\n"
                + "                using (var output = File.Create(destination))\n"
                + "                    input.CopyTo(output);\n"
                + "            }\n"
                + "        }\n"
                + "    }\n"
                + "\n"
                + "    private static string BootstrapRoot\n"
                + "    {\n"
                + "        get\n"
                + "        {\n"
                + "            var guids = AssetDatabase.FindAssets(BootstrapScriptName);\n"
                + "            if (guids.Length != 1)\n"
                + "                throw new InvalidOperationException(\"Could not resolve the Aexis UnityPackage bootstrap asset.\");\n"
                + "            var scriptPath = AssetDatabase.GUIDToAssetPath(guids[0]);\n"
                + "            return Path.GetFullPath(Path.Combine(Path.GetDirectoryName(scriptPath), \"..\"));\n"
                + "        }\n"
                + "    }\n"
                + "\n"
                + "    private static string ProjectRoot => Path.GetFullPath(Path.Combine(Application.dataPath, \"..\"));\n"
                + "\n"
                + "    private static string CleanupMarkerPath => Path.Combine(ProjectRoot, CleanupMarkerRelativePath);\n"
                + "\n"
                + "    private static void CleanupBootstrapAssets()\n"
                + "    {\n"
                + "        AssetDatabase.DeleteAsset(\"Assets/AexisPackageBootstrap\");\n"
                + "        AssetDatabase.Refresh();\n"
                + "    }\n"
                + "}\n"
                + "#endif\n");
        }

        [Serializable]
        private sealed class PackageMetadata
        {
            public string version;
        }
    }
}
#endif
