#if UNITY_EDITOR
using System;
using System.Diagnostics;
using System.IO;
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
            var aexisRoot = Path.Combine(assetsRoot, "Aexis");
            Directory.CreateDirectory(aexisRoot);
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath));

            CopyDirectory(Path.Combine(sourceRoot, "Runtime"), Path.Combine(aexisRoot, "Runtime"));
            CopyDirectory(Path.Combine(sourceRoot, "Editor"), Path.Combine(aexisRoot, "Editor"));
            CopyDirectory(
                Path.Combine(sourceRoot, "Samples~", "AIImageApplicationExample"),
                Path.Combine(aexisRoot, "Examples", "AIImageApplicationExample"));
            CopyDirectory(
                Path.Combine(sourceRoot, "UnityPackage~", "ThirdParty", "AexisSampleNewtonsoft"),
                Path.Combine(aexisRoot, "Examples", "AIImageApplicationExample", "ThirdParty", "AexisSampleNewtonsoft"));
            CopyFileIfPresent(Path.Combine(sourceRoot, "package.json"), Path.Combine(aexisRoot, "package.json"));
            CopyFileIfPresent(Path.Combine(sourceRoot, "README.md"), Path.Combine(aexisRoot, "README.md"));
            CopyFileIfPresent(Path.Combine(sourceRoot, "CHANGELOG.md"), Path.Combine(aexisRoot, "CHANGELOG.md"));
            CopyFileIfPresent(Path.Combine(sourceRoot, "LICENSE.md"), Path.Combine(aexisRoot, "LICENSE.md"));
            CopyFileIfPresent(Path.Combine(sourceRoot, "Third Party Notices.md"), Path.Combine(aexisRoot, "Third Party Notices.md"));
            CopyDirectory(Path.Combine(sourceRoot, "Documentation~"), Path.Combine(aexisRoot, "Documentation"));

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
                + "        AssetDatabase.ExportPackage(new[] { \"Assets/Aexis\" }, \"" + escapedOutput + "\", ExportPackageOptions.Recurse);\n"
                + "    }\n"
                + "}\n"
                + "#endif\n");
        }

        private static void CopyDirectory(string source, string destination)
        {
            if (!Directory.Exists(source))
                throw new DirectoryNotFoundException("Required Aexis export source was not found: " + source);

            foreach (var directory in Directory.GetDirectories(source, "*", SearchOption.AllDirectories))
                Directory.CreateDirectory(directory.Replace(source, destination));
            foreach (var file in Directory.GetFiles(source, "*", SearchOption.AllDirectories))
            {
                var target = file.Replace(source, destination);
                Directory.CreateDirectory(Path.GetDirectoryName(target));
                File.Copy(file, target, true);
            }
        }

        private static void CopyFileIfPresent(string source, string destination)
        {
            if (!File.Exists(source))
                return;

            Directory.CreateDirectory(Path.GetDirectoryName(destination));
            File.Copy(source, destination, true);
            var metaSource = source + ".meta";
            if (File.Exists(metaSource))
                File.Copy(metaSource, destination + ".meta", true);
        }

        [Serializable]
        private sealed class PackageMetadata
        {
            public string version;
        }
    }
}
#endif
