#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;

namespace Aexis.Editor
{
    /// <summary>Stages the package-owned Main2 default payload into a Player build.</summary>
    public sealed class AexisApplicationExamplePlayerBuildPreprocessor : IPreprocessBuildWithReport
    {
        public const string SkipStagingEnvironmentVariable = "AEXIS_SKIP_SAMPLE_STREAMING_ASSETS_STAGING";

        private const string PackageName = "com.aexis";
        private const string SampleStreamingAssetsRelativePath = "Samples/AIImageApplicationExample/StreamingAssets";

        public int callbackOrder => -1000;

        public void OnPreprocessBuild(BuildReport report)
        {
            if (IsStagingDisabled())
                return;

            var source = ResolveSampleStreamingAssetsPath();
            if (source == null)
                return;

            var destination = Path.Combine(Application.dataPath, "StreamingAssets");
            var copiedFiles = CopyMissingFiles(source, destination);
            if (copiedFiles == 0)
                return;

            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            Debug.Log("[Aexis.Editor] Staged " + copiedFiles + " Main2 StreamingAssets files for " + report.summary.platform + ".");
        }

        private static bool IsStagingDisabled()
        {
            return string.Equals(
                Environment.GetEnvironmentVariable(SkipStagingEnvironmentVariable),
                "1",
                StringComparison.Ordinal);
        }

        private static string ResolveSampleStreamingAssetsPath()
        {
            var packages = UnityEditor.PackageManager.PackageInfo.GetAllRegisteredPackages();
            for (var index = 0; index < packages.Length; index++)
            {
                var package = packages[index];
                if (package == null || !string.Equals(package.name, PackageName, StringComparison.Ordinal))
                    continue;

                var source = Path.Combine(package.resolvedPath, SampleStreamingAssetsRelativePath);
                return Directory.Exists(source) ? source : null;
            }

            return null;
        }

        private static int CopyMissingFiles(string source, string destination)
        {
            var copiedFiles = 0;
            Directory.CreateDirectory(destination);
            foreach (var sourceFile in Directory.GetFiles(source, "*", SearchOption.AllDirectories))
            {
                if (string.Equals(Path.GetExtension(sourceFile), ".meta", StringComparison.OrdinalIgnoreCase))
                    continue;

                var relativePath = sourceFile.Substring(source.Length + 1);
                var destinationFile = Path.Combine(destination, relativePath);
                if (File.Exists(destinationFile))
                    continue;

                Directory.CreateDirectory(Path.GetDirectoryName(destinationFile));
                File.Copy(sourceFile, destinationFile, false);
                copiedFiles++;
            }

            return copiedFiles;
        }
    }
}
#endif
