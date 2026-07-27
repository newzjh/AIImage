#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Reflection;
using System.Text;
using UnityEditor;
using UnityEditor.Android;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEngine;
using UnityEngine.Rendering;
using Aexis.Samples.Json.Linq;
using AIImage.Qwen35;

public sealed class AIImageReducedModelBuild : IPostprocessBuildWithReport, IPostGenerateGradleAndroidProject
{
    internal const string ReducedMain2BuildSessionStateKey = "Aexis.ReducedMain2Build.Active";
    private const string OutputEnvironmentVariable = "AEXIS_REDUCED_RELEASE_OUTPUT";
    private const string WindowsOutputPathEnvironmentVariable = "AEXIS_REDUCED_WINDOWS_OUTPUT_PATH";
    private static BuildSession ActiveSession;
    private static readonly string[] ModelPayloadRootPrefixes =
    {
        "Clip/",
        "CodeFormer/",
        "DeepFileV2/",
        "GFPGAN/",
        "Matting/",
        "QWEN35/",
        "RealESRGAN/",
        "StableDiffusion/",
        "sdinpainting/",
        "Yolo/",
        "MONAI/",
        "Monai/",
        "VISTA/",
        "Vista/"
    };

    public int callbackOrder => 900;

    [InitializeOnLoadMethod]
    private static void RegisterEditorBuildHandler()
    {
        BuildPlayerWindow.RegisterBuildPlayerHandler(BuildPlayerFromEditorWindow);
    }

    private static void BuildPlayerFromEditorWindow(BuildPlayerOptions buildPlayerOptions)
    {
        if (!ShouldApplyReducedMain2Policy(buildPlayerOptions))
        {
            BuildPlayerWindow.DefaultBuildMethods.BuildPlayer(buildPlayerOptions);
            return;
        }

        Debug.Log("Applying reduced Main2 model policy to the Build window Player output: "
            + buildPlayerOptions.locationPathName);
        BuildReducedMain2(buildPlayerOptions, configureAndroidVulkan: true);
    }

    [MenuItem("Aexis/Release/Build Reduced/Main2 Windows x64")]
    public static void BuildWindowsMenu()
    {
        BuildMain2(BuildTarget.StandaloneWindows64, DefaultOutput(BuildTarget.StandaloneWindows64), BuildOptions.None);
    }

    [MenuItem("Aexis/Release/Build Reduced/Main2 Android APK")]
    public static void BuildAndroidMenu()
    {
        BuildMain2(BuildTarget.Android, DefaultOutput(BuildTarget.Android), BuildOptions.None);
    }

    [MenuItem("Aexis/Release/Build Reduced/Main2 macOS")]
    public static void BuildMacMenu()
    {
        BuildMain2(BuildTarget.StandaloneOSX, DefaultOutput(BuildTarget.StandaloneOSX), BuildOptions.None);
    }

    [MenuItem("Aexis/Release/Build Reduced/Main2 Linux x64")]
    public static void BuildLinuxMenu()
    {
        BuildMain2(BuildTarget.StandaloneLinux64, DefaultOutput(BuildTarget.StandaloneLinux64), BuildOptions.None);
    }

    [MenuItem("Aexis/Release/Build Reduced/Main2 iOS Xcode Project")]
    public static void BuildIosMenu()
    {
        BuildMain2(BuildTarget.iOS, DefaultOutput(BuildTarget.iOS), BuildOptions.None);
    }

    [MenuItem("Aexis/Release/Build Reduced/All Installed Platforms")]
    public static void BuildAllInstalledPlatformsMenu()
    {
        BuildAllInstalledPlatforms();
    }

    public static void BuildMain2Windows64Batch()
    {
        try
        {
            BuildMain2(BuildTarget.StandaloneWindows64, DefaultOutput(BuildTarget.StandaloneWindows64), BuildOptions.None);
            EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            EditorApplication.Exit(1);
        }
    }

    public static void BuildMain2Windows64AtConfiguredPathBatch()
    {
        try
        {
            var output = Environment.GetEnvironmentVariable(WindowsOutputPathEnvironmentVariable);
            if (string.IsNullOrWhiteSpace(output))
                throw new InvalidOperationException(
                    WindowsOutputPathEnvironmentVariable + " must contain the target .exe path.");
            BuildMain2(BuildTarget.StandaloneWindows64, Path.GetFullPath(output), BuildOptions.None);
            EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            EditorApplication.Exit(1);
        }
    }

    public static void BuildMain2Windows64BuildWindowBatch()
    {
        try
        {
            BuildPlayerFromEditorWindow(new BuildPlayerOptions
            {
                scenes = new[] { AexisApplicationExamplePaths.Main2SceneAssetPath },
                locationPathName = DefaultOutput(BuildTarget.StandaloneWindows64),
                target = BuildTarget.StandaloneWindows64,
                targetGroup = BuildTargetGroup.Standalone,
                options = BuildOptions.None
            });
            EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            EditorApplication.Exit(1);
        }
    }

    public static void BuildAllInstalledPlatformsBatch()
    {
        try
        {
            BuildAllInstalledPlatforms();
            EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            EditorApplication.Exit(1);
        }
    }

    public static BuildReport BuildMain2(BuildTarget target, string output, BuildOptions options)
    {
        return BuildReducedMain2(new BuildPlayerOptions
        {
            scenes = new[] { AexisApplicationExamplePaths.Main2SceneAssetPath },
            locationPathName = output,
            target = target,
            options = options
        }, configureAndroidVulkan: true);
    }

    internal static BuildReport BuildMain2WithCurrentAndroidSettings(BuildTarget target, string output, BuildOptions options)
    {
        return BuildReducedMain2(new BuildPlayerOptions
        {
            scenes = new[] { AexisApplicationExamplePaths.Main2SceneAssetPath },
            locationPathName = output,
            target = target,
            options = options
        }, configureAndroidVulkan: false);
    }

    private static BuildReport BuildReducedMain2(BuildPlayerOptions buildPlayerOptions, bool configureAndroidVulkan)
    {
        var target = buildPlayerOptions.target;
        var group = BuildPipeline.GetBuildTargetGroup(target);
        if (!BuildPipeline.IsBuildTargetSupported(group, target))
            throw new NotSupportedException(target + " build support is not installed in this Unity editor.");
        if (ActiveSession != null)
            throw new InvalidOperationException("A reduced Main2 build is already in progress.");

        var output = Path.GetFullPath(buildPlayerOptions.locationPathName);
        Directory.CreateDirectory(Path.GetDirectoryName(output));

        var session = new BuildSession(target, output);
        ActiveSession = session;
        SessionState.SetBool(ReducedMain2BuildSessionStateKey, true);
        BuildReport report = null;
        AndroidVulkanBuildSettingsOverride androidSettings = null;
        MacOsAppleDoubleBuildGuard macOsBuildGuard = null;
        try
        {
            ValidateDefaultModelSources();
            if (target == BuildTarget.StandaloneOSX)
            {
                macOsBuildGuard = new MacOsAppleDoubleBuildGuard();
                macOsBuildGuard.Apply();
            }
            if (target == BuildTarget.Android && configureAndroidVulkan)
            {
                androidSettings = new AndroidVulkanBuildSettingsOverride();
                androidSettings.Apply();
            }

            buildPlayerOptions.locationPathName = output;
            buildPlayerOptions.options |= BuildOptions.CompressWithLz4;
            report = BuildPipeline.BuildPlayer(buildPlayerOptions);
            if (report.summary.result != BuildResult.Succeeded || report.summary.totalErrors != 0)
            {
                var buildErrors = report.SummarizeErrors();
                var diagnosticPath = WriteBuildFailureDiagnostic(report, target, output);
                throw new InvalidOperationException(
                    "Reduced Main2 build failed: result=" + report.summary.result
                    + " errors=" + report.summary.totalErrors
                    + " output=" + output
                    + (string.IsNullOrWhiteSpace(buildErrors) ? string.Empty : "\nBuild errors:\n" + buildErrors)
                    + (string.IsNullOrWhiteSpace(diagnosticPath)
                        ? string.Empty
                        : "\nBuild failure diagnostic: " + diagnosticPath));
            }

            // Unity invokes post-build callbacks during BuildPlayer, but apply the
            // file policy again after the Player writer has closed every output.
            // This is the authoritative desktop/iOS pruning point.
            if (target != BuildTarget.Android)
            {
                RewriteOutputStreamingAssets(FindPlayerStreamingAssetsDirectory(target, output));
                RemoveNonShippingOutputArtifacts(output);
                session.OutputWasRewritten = true;
            }
            if (!session.OutputWasRewritten)
                throw new InvalidOperationException("Reduced model policy was not applied to " + target + " output.");
            return report;
        }
        finally
        {
            androidSettings?.Dispose();
            macOsBuildGuard?.Dispose();
            SessionState.EraseBool(ReducedMain2BuildSessionStateKey);
            ActiveSession = null;
        }
    }

    private static string WriteBuildFailureDiagnostic(BuildReport report, BuildTarget target, string output)
    {
        try
        {
            var outputDirectory = Path.GetDirectoryName(output);
            if (string.IsNullOrWhiteSpace(outputDirectory))
                outputDirectory = ProjectRoot;

            var outputName = Path.GetFileNameWithoutExtension(output);
            if (string.IsNullOrWhiteSpace(outputName))
                outputName = "AIImage_Main2";

            var diagnosticPath = Path.Combine(outputDirectory, outputName + ".build-failure.txt");
            Directory.CreateDirectory(outputDirectory);
            File.WriteAllText(
                diagnosticPath,
                CreateBuildFailureDiagnostic(report, target, output),
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
            Debug.LogError("[Aexis.Editor] Reduced Main2 build failure diagnostic: " + diagnosticPath);
            return diagnosticPath;
        }
        catch (Exception exception)
        {
            // The original BuildReport failure remains the authoritative exception.
            Debug.LogWarning("[Aexis.Editor] Could not write reduced Main2 build failure diagnostic: " + exception.Message);
            return null;
        }
    }

    private static string CreateBuildFailureDiagnostic(BuildReport report, BuildTarget target, string output)
    {
        var text = new StringBuilder(4096);
        text.AppendLine("AIImage Reduced Main2 build failure diagnostic");
        text.AppendLine("generated_utc=" + DateTime.UtcNow.ToString("O"));
        text.AppendLine("unity_version=" + Application.unityVersion);
        text.AppendLine("host_operating_system=" + SystemInfo.operatingSystem);
        text.AppendLine("requested_target=" + target);
        text.AppendLine("requested_output=" + output);
        text.AppendLine("editor_console_log=" + Application.consoleLogPath);
        text.AppendLine();

        if (report == null)
        {
            text.AppendLine("BuildReport was not returned by BuildPipeline.BuildPlayer.");
        }
        else
        {
            var summary = report.summary;
            text.AppendLine("BuildReport summary");
            text.AppendLine("result=" + summary.result);
            text.AppendLine("platform=" + summary.platform);
            text.AppendLine("platform_group=" + summary.platformGroup);
            text.AppendLine("output_path=" + summary.outputPath);
            text.AppendLine("total_errors=" + summary.totalErrors);
            text.AppendLine("total_warnings=" + summary.totalWarnings);
            text.AppendLine("total_bytes=" + summary.totalSize);
            text.AppendLine("total_time_ms=" + summary.totalTime.TotalMilliseconds);
            text.AppendLine();
            text.AppendLine("BuildReport.SummarizeErrors()");
            var summarizedErrors = report.SummarizeErrors();
            text.AppendLine(string.IsNullOrWhiteSpace(summarizedErrors) ? "<empty>" : summarizedErrors);
            text.AppendLine();
            text.AppendLine("BuildReport steps and messages");

            var stepIndex = 0;
            foreach (var step in report.steps)
            {
                text.Append('[').Append(stepIndex++).Append("] ").Append(step.name)
                    .Append(" depth=").Append(step.depth)
                    .Append(" duration_ms=").Append(step.duration.TotalMilliseconds)
                    .AppendLine();
                if (step.messages == null || !step.messages.Any())
                {
                    text.AppendLine("  <no messages>");
                    continue;
                }

                foreach (var message in step.messages)
                {
                    text.Append("  [").Append(message.type).Append("] ")
                        .AppendLine(message.content ?? string.Empty);
                }
            }
        }

        text.AppendLine();
        text.AppendLine("Editor console log tail (up to 512 KiB)");
        text.AppendLine(ReadConsoleLogTail(Application.consoleLogPath));
        return text.ToString();
    }

    private static string ReadConsoleLogTail(string consoleLogPath)
    {
        const int maxBytes = 512 * 1024;
        if (string.IsNullOrWhiteSpace(consoleLogPath))
            return "<Unity did not provide Application.consoleLogPath>";
        if (!File.Exists(consoleLogPath))
            return "<console log does not exist>";

        try
        {
            using (var stream = new FileStream(consoleLogPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            {
                if (stream.Length > maxBytes)
                    stream.Seek(-maxBytes, SeekOrigin.End);
                using (var reader = new StreamReader(stream, Encoding.UTF8, detectEncodingFromByteOrderMarks: true))
                    return reader.ReadToEnd();
            }
        }
        catch (Exception exception)
        {
            return "<could not read editor console log: " + exception.Message + ">";
        }
    }

    private static void ResetMacOsBeeCache()
    {
        var beeDirectory = Path.Combine(ProjectRoot, "Library", "Bee");
        if (!Directory.Exists(beeDirectory))
            return;

        try
        {
            Directory.Delete(beeDirectory, true);
            Debug.Log("[Aexis.Editor] Cleared generated macOS Bee cache before the reduced Main2 build: " + beeDirectory);
        }
        catch (Exception exception)
        {
            throw new InvalidOperationException(
                "Could not clear the generated macOS Bee cache at " + beeDirectory
                + ". Close other Unity processes using this project and retry.",
                exception);
        }
    }

    public void OnPostprocessBuild(BuildReport report)
    {
        var session = ActiveSession;
        if (session == null || report == null || report.summary.platform != session.Target)
            return;

        if (session.Target == BuildTarget.Android)
        {
            ValidateAndroidApk(session.OutputPath);
            session.OutputWasRewritten = true;
            return;
        }

        var streamingAssetsDirectory = FindPlayerStreamingAssetsDirectory(session.Target, session.OutputPath);
        RewriteOutputStreamingAssets(streamingAssetsDirectory);
        session.OutputWasRewritten = true;
    }

    public void OnPostGenerateGradleAndroidProject(string path)
    {
        var session = ActiveSession;
        if (session == null || session.Target != BuildTarget.Android)
            return;

        var assetsDirectory = FindGradleAssetsDirectory(path);
        RewriteOutputStreamingAssets(assetsDirectory);
        session.AndroidAssetsWereRewritten = true;
    }

    internal static IReadOnlyList<string> DefaultModelFiles => GetDefaultModelFiles();

    internal static string ResolveModelSourceFile(string relativePath)
    {
        var normalized = AIImageModelDelivery.NormalizeRelativePath(relativePath);
        if (string.IsNullOrWhiteSpace(normalized))
            return null;

        const string q4Prefix = "QWEN35/qwen3.5_0.8b_mobile_q4/";
        if (normalized.StartsWith(q4Prefix, StringComparison.OrdinalIgnoreCase))
        {
            var q4Source = Qwen35MobileQ4AssetBuilder.ResolveQ4SourceDirectory();
            if (!string.IsNullOrWhiteSpace(q4Source))
            {
                var q4Relative = normalized.Substring(q4Prefix.Length).Replace('/', Path.DirectorySeparatorChar);
                var candidate = Path.Combine(q4Source, q4Relative);
                if (File.Exists(candidate))
                    return candidate;
            }
        }

        var candidates = new[]
        {
            Path.Combine(ProjectRoot, "Assets", "StreamingAssets", normalized),
            Path.Combine(AexisApplicationExamplePaths.SampleRootAbsolutePath, "StreamingAssets", normalized),
            AIImageModelDelivery.GetPersistentPath(normalized)
        };
        for (var index = 0; index < candidates.Length; index++)
        {
            if (File.Exists(candidates[index]))
                return candidates[index];
        }
        return null;
    }

    private static bool ShouldApplyReducedMain2Policy(BuildPlayerOptions buildPlayerOptions)
    {
        return ActiveSession == null
            && IsReducedPlayerTarget(buildPlayerOptions.target)
            && !string.IsNullOrWhiteSpace(buildPlayerOptions.locationPathName)
            && IncludesMain2Scene(buildPlayerOptions.scenes);
    }

    private static bool IsReducedPlayerTarget(BuildTarget target)
    {
        switch (target)
        {
            case BuildTarget.StandaloneWindows64:
            case BuildTarget.Android:
            case BuildTarget.StandaloneOSX:
            case BuildTarget.StandaloneLinux64:
            case BuildTarget.iOS:
                return true;
            default:
                return false;
        }
    }

    private static bool IncludesMain2Scene(IEnumerable<string> scenes)
    {
        if (scenes == null)
            return false;

        var main2Path = AexisApplicationExamplePaths.Main2SceneAssetPath.Replace('\\', '/');
        return scenes.Any(scene => string.Equals(
            (scene ?? string.Empty).Replace('\\', '/'),
            main2Path,
            StringComparison.OrdinalIgnoreCase));
    }

    private static void BuildAllInstalledPlatforms()
    {
        var targets = new[]
        {
            BuildTarget.StandaloneWindows64,
            BuildTarget.Android,
            BuildTarget.StandaloneOSX,
            BuildTarget.StandaloneLinux64,
            BuildTarget.iOS
        };
        var errors = new List<string>();
        for (var index = 0; index < targets.Length; index++)
        {
            var target = targets[index];
            if (!BuildPipeline.IsBuildTargetSupported(BuildPipeline.GetBuildTargetGroup(target), target))
            {
                Debug.Log("Skipping unavailable reduced Main2 target: " + target);
                continue;
            }

            try
            {
                BuildMain2(target, DefaultOutput(target), BuildOptions.None);
            }
            catch (Exception exception)
            {
                errors.Add(target + ": " + exception.Message);
            }
        }

        if (errors.Count > 0)
            throw new InvalidOperationException("One or more reduced Main2 builds failed:\n" + string.Join("\n", errors));

        AIImageModelReleasePackager.ExportReducedMain2UnityPackage();
    }

    private static void RewriteOutputStreamingAssets(string directory)
    {
        if (string.IsNullOrWhiteSpace(directory))
            throw new ArgumentException("A player StreamingAssets directory is required.", nameof(directory));

        Directory.CreateDirectory(directory);
        foreach (var path in Directory.GetFiles(directory, "*", SearchOption.AllDirectories))
        {
            var relative = AIImageModelDelivery.NormalizeRelativePath(path.Substring(directory.Length));
            if (IsModelPayloadPath(relative))
                File.Delete(path);
        }

        var files = GetDefaultModelFiles();
        for (var index = 0; index < files.Count; index++)
        {
            var relative = files[index];
            var source = ResolveModelSourceFile(relative);
            if (source == null)
            {
                Debug.LogWarning(
                    "Reduced Main2 build skipped model file that became unavailable during staging: " + relative);
                continue;
            }
            var destination = Path.Combine(directory, relative.Replace('/', Path.DirectorySeparatorChar));
            Directory.CreateDirectory(Path.GetDirectoryName(destination));
            File.Copy(source, destination, true);
        }

        CopyRuntimeMetadata(directory);
        WriteBundledManifest(directory);
        ValidateStreamingAssetsDirectory(directory);
    }

    private static void ValidateDefaultModelSources()
    {
        GetBundledModelGroups(reportExcludedGroups: true);
    }

    private static IReadOnlyList<AIImageModelGroup> GetBundledModelGroups(bool reportExcludedGroups = false)
    {
        var result = new List<AIImageModelGroup>();
        foreach (var group in AIImageModelDelivery.DefaultGroups)
        {
            if (TryGetBundledGroupFiles(group, out _, out var reason))
            {
                result.Add(group);
                continue;
            }

            if (reportExcludedGroups)
            {
                Debug.LogWarning(
                    "Reduced Main2 build omits default model group '" + group.DisplayName + "': " + reason
                    + ". The Player will offer this group for download when it is selected at runtime.");
            }
        }

        return result;
    }

    private static void CopyRuntimeMetadata(string outputDirectory)
    {
        const string metadataDirectoryName = "InferenceManifests";
        var candidates = new[]
        {
            Path.Combine(ProjectRoot, "Assets", "StreamingAssets", metadataDirectoryName),
            Path.Combine(AexisApplicationExamplePaths.SampleRootAbsolutePath, "StreamingAssets", metadataDirectoryName)
        };

        string sourceDirectory = null;
        for (var index = 0; index < candidates.Length; index++)
        {
            if (Directory.Exists(candidates[index]))
            {
                sourceDirectory = candidates[index];
                break;
            }
        }
        if (sourceDirectory == null)
            throw new DirectoryNotFoundException("Runtime inference manifests are unavailable for the reduced Player build.");

        var destinationDirectory = Path.Combine(outputDirectory, metadataDirectoryName);
        foreach (var source in Directory.GetFiles(sourceDirectory, "*", SearchOption.AllDirectories))
        {
            if (source.EndsWith(".meta", StringComparison.OrdinalIgnoreCase))
                continue;

            var relative = source.Substring(sourceDirectory.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            var destination = Path.Combine(destinationDirectory, relative);
            Directory.CreateDirectory(Path.GetDirectoryName(destination));
            File.Copy(source, destination, true);
        }
    }

    private static void ValidateAndroidApk(string apkPath)
    {
        if (!File.Exists(apkPath))
            throw new FileNotFoundException("Android APK was not produced.", apkPath);
        if (ActiveSession == null || !ActiveSession.AndroidAssetsWereRewritten)
            throw new InvalidOperationException("Android Gradle assets were not rewritten before APK packaging.");

        var bundled = new HashSet<string>(GetDefaultModelFiles(), StringComparer.OrdinalIgnoreCase);
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        using (var archive = ZipFile.OpenRead(apkPath))
        {
            foreach (var entry in archive.Entries)
            {
                var name = entry.FullName.Replace('\\', '/');
                if (!name.StartsWith("assets/", StringComparison.OrdinalIgnoreCase))
                    continue;
                var relative = name.Substring("assets/".Length);
                if (!IsModelPayloadPath(relative))
                    continue;
                if (!bundled.Contains(relative))
                    throw new InvalidDataException("Reduced Android APK still contains an excluded model file: " + relative);
                seen.Add(relative);
            }
        }

        foreach (var required in bundled)
        {
            if (!seen.Contains(required))
                throw new InvalidDataException("Reduced Android APK is missing default model file: " + required);
        }
    }

    private static void ValidateStreamingAssetsDirectory(string directory)
    {
        var bundled = new HashSet<string>(GetDefaultModelFiles(), StringComparer.OrdinalIgnoreCase);
        foreach (var path in Directory.GetFiles(directory, "*", SearchOption.AllDirectories))
        {
            var relative = AIImageModelDelivery.NormalizeRelativePath(path.Substring(directory.Length));
            if (IsModelPayloadPath(relative) && !bundled.Contains(relative))
                throw new InvalidDataException("Reduced player output contains an excluded model file: " + relative);
        }

        foreach (var required in bundled)
        {
            if (!File.Exists(Path.Combine(directory, required.Replace('/', Path.DirectorySeparatorChar))))
                throw new InvalidDataException("Reduced player output is missing default model file: " + required);
        }
    }

    private static IReadOnlyList<string> GetDefaultModelFiles()
    {
        var files = new List<string>();
        foreach (var group in GetBundledModelGroups())
        {
            if (TryGetBundledGroupFiles(group, out var groupFiles, out _))
                files.AddRange(groupFiles);
        }
        return files
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .ToArray();
    }

    private static bool TryGetBundledGroupFiles(
        AIImageModelGroup group,
        out IReadOnlyList<string> files,
        out string reason)
    {
        var result = new List<string>(group?.Files ?? Array.Empty<string>());
        files = Array.Empty<string>();
        reason = null;
        if (group == null || result.Count == 0)
        {
            reason = "the group has no model files";
            return false;
        }

        if (group.Id == AIImageModelGroupId.Qwen35MobileQ4
            && !TryAppendQwen35Q4ShardFiles(result, out reason))
        {
            return false;
        }

        for (var index = 0; index < result.Count; index++)
        {
            var relative = result[index];
            if (ResolveModelSourceFile(relative) != null)
                continue;

            reason = "required file is unavailable: " + relative;
            return false;
        }

        files = result
            .Select(AIImageModelDelivery.NormalizeRelativePath)
            .Where(path => !string.IsNullOrWhiteSpace(path))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        return true;
    }

    private static bool TryAppendQwen35Q4ShardFiles(ICollection<string> files, out string reason)
    {
        reason = null;
        const string q4Prefix = "QWEN35/qwen3.5_0.8b_mobile_q4/";
        var sourceDirectory = Qwen35MobileQ4AssetBuilder.ResolveQ4SourceDirectory();
        if (string.IsNullOrWhiteSpace(sourceDirectory))
        {
            reason = "a complete, validated Qwen3.5 mobile Q4 asset set is unavailable";
            return false;
        }

        try
        {
            var manifestPath = Path.Combine(sourceDirectory, Qwen35MobileAssetSet.Q4ManifestFileName);
            var manifest = JObject.Parse(File.ReadAllText(manifestPath));
            var logicalFiles = manifest["logical_files"] as JObject
                ?? throw new InvalidDataException("Qwen3.5 mobile Q4 manifest has no logical_files object: " + manifestPath);
            foreach (var logical in logicalFiles.Properties())
            {
                var parts = logical.Value["parts"] as JArray
                    ?? throw new InvalidDataException("Qwen3.5 mobile Q4 logical asset has no parts array: " + logical.Name);
                foreach (var part in parts)
                {
                    var relative = AIImageModelDelivery.NormalizeRelativePath((string)part["file"]);
                    if (string.IsNullOrWhiteSpace(relative))
                        throw new InvalidDataException("Qwen3.5 mobile shard path is empty: " + logical.Name);
                    files.Add(q4Prefix + relative);
                }
            }

            return true;
        }
        catch (Exception exception)
        {
            reason = "the Qwen3.5 mobile Q4 manifest is invalid: " + exception.Message;
            return false;
        }
    }

    private static void AppendQwen35Q4ShardFiles(ICollection<string> files)
    {
        const string q4Prefix = "QWEN35/qwen3.5_0.8b_mobile_q4/";
        var sourceDirectory = Qwen35MobileQ4AssetBuilder.ResolveQ4SourceDirectory();
        if (string.IsNullOrWhiteSpace(sourceDirectory))
            return;
        var manifestPath = Path.Combine(sourceDirectory, Qwen35MobileAssetSet.Q4ManifestFileName);
        var manifest = JObject.Parse(File.ReadAllText(manifestPath));
        var logicalFiles = manifest["logical_files"] as JObject
            ?? throw new InvalidDataException("Qwen3.5 mobile Q4 manifest has no logical_files object: " + manifestPath);
        foreach (var logical in logicalFiles.Properties())
        {
            var parts = logical.Value["parts"] as JArray
                ?? throw new InvalidDataException("Qwen3.5 mobile Q4 logical asset has no parts array: " + logical.Name);
            foreach (var part in parts)
            {
                var relative = AIImageModelDelivery.NormalizeRelativePath((string)part["file"]);
                if (string.IsNullOrWhiteSpace(relative))
                    throw new InvalidDataException("Qwen3.5 mobile Q4 shard path is empty: " + logical.Name);
                files.Add(q4Prefix + relative);
            }
        }
    }

    private static bool IsModelPayloadPath(string relativePath)
    {
        var normalized = AIImageModelDelivery.NormalizeRelativePath(relativePath);
        if (string.Equals(normalized, AIImageModelDelivery.BundledManifestFileName, StringComparison.OrdinalIgnoreCase))
            return false;
        for (var index = 0; index < ModelPayloadRootPrefixes.Length; index++)
        {
            if (normalized.StartsWith(ModelPayloadRootPrefixes[index], StringComparison.OrdinalIgnoreCase))
                return true;
        }

        return false;
    }

    private static void WriteBundledManifest(string directory)
    {
        var manifest = new BundledModelManifest
        {
            groups = GetBundledModelGroups().Select(group => group.Id.ToString()).ToArray()
        };
        File.WriteAllText(
            Path.Combine(directory, AIImageModelDelivery.BundledManifestFileName),
            JsonUtility.ToJson(manifest, true));
    }

    private static string FindPlayerStreamingAssetsDirectory(BuildTarget target, string output)
    {
        var candidates = new List<string>();
        if (target == BuildTarget.iOS)
        {
            candidates.Add(Path.Combine(output, "Data", "Raw"));
            candidates.Add(Path.Combine(output, "Data", "StreamingAssets"));
        }
        else if (target == BuildTarget.StandaloneOSX)
        {
            candidates.Add(Path.Combine(output, "Contents", "Resources", "Data", "StreamingAssets"));
        }
        else
        {
            var parent = Path.GetDirectoryName(output);
            var name = Path.GetFileNameWithoutExtension(output);
            candidates.Add(Path.Combine(parent, name + "_Data", "StreamingAssets"));
            candidates.Add(Path.Combine(output, "Data", "StreamingAssets"));
        }

        for (var index = 0; index < candidates.Count; index++)
        {
            if (Directory.Exists(candidates[index]))
                return candidates[index];
        }
        return candidates[0];
    }

    private static void RemoveNonShippingOutputArtifacts(string output)
    {
        var outputDirectory = Path.GetDirectoryName(output);
        var playerName = Path.GetFileNameWithoutExtension(output);
        if (string.IsNullOrWhiteSpace(outputDirectory) || string.IsNullOrWhiteSpace(playerName))
            return;

        // IL2CPP emits generated sources and symbols here for inspection only. Unity
        // explicitly marks it as non-shipping; retaining it more than doubles the player folder.
        var backupDirectory = Path.Combine(
            outputDirectory,
            playerName + "_BackUpThisFolder_ButDontShipItWithYourGame");
        if (!Directory.Exists(backupDirectory))
            return;

        Directory.Delete(backupDirectory, true);
        Debug.Log("Removed non-shipping IL2CPP backup output: " + backupDirectory);
    }

    private static string FindGradleAssetsDirectory(string gradleProjectPath)
    {
        var candidates = new[]
        {
            Path.Combine(gradleProjectPath, "src", "main", "assets"),
            Path.Combine(gradleProjectPath, "unityLibrary", "src", "main", "assets"),
            Path.Combine(gradleProjectPath, "launcher", "src", "main", "assets")
        };
        for (var index = 0; index < candidates.Length; index++)
        {
            if (Directory.Exists(candidates[index]))
                return candidates[index];
        }
        throw new DirectoryNotFoundException("Could not locate Android Gradle assets directory below " + gradleProjectPath);
    }

    private static string DefaultOutput(BuildTarget target)
    {
        var configured = Environment.GetEnvironmentVariable(OutputEnvironmentVariable);
        var root = Path.GetFullPath(string.IsNullOrWhiteSpace(configured)
            ? Path.Combine(ProjectRoot, "Builds", "ReducedMain2")
            : configured);
        switch (target)
        {
            case BuildTarget.StandaloneWindows64: return Path.Combine(root, "Windows", "AIImageMain2.exe");
            case BuildTarget.Android: return Path.Combine(root, "Android", "AIImageMain2.apk");
            case BuildTarget.StandaloneOSX: return Path.Combine(root, "macOS", "AIImageMain2.app");
            case BuildTarget.StandaloneLinux64: return Path.Combine(root, "Linux", "AIImageMain2.x86_64");
            case BuildTarget.iOS: return Path.Combine(root, "iOS", "AIImageMain2");
            default: throw new ArgumentOutOfRangeException(nameof(target), target, "Unsupported reduced release target.");
        }
    }

    private static string ProjectRoot => Path.GetFullPath(Path.Combine(Application.dataPath, ".."));

    private sealed class BuildSession
    {
        public BuildSession(BuildTarget target, string outputPath)
        {
            Target = target;
            OutputPath = outputPath;
        }

        public BuildTarget Target { get; }
        public string OutputPath { get; }
        public bool AndroidAssetsWereRewritten { get; set; }
        public bool OutputWasRewritten { get; set; }
    }

    private sealed class AndroidVulkanBuildSettingsOverride : IDisposable
    {
        private readonly bool _useDefaultGraphicsApis;
        private readonly GraphicsDeviceType[] _graphicsApis;
        private readonly AndroidArchitecture _architectures;
        private readonly AndroidSdkVersions _minimumSdk;
        private PropertyInfo _applicationEntryProperty;
        private object _applicationEntry;
        private bool _applied;

        public AndroidVulkanBuildSettingsOverride()
        {
            _useDefaultGraphicsApis = PlayerSettings.GetUseDefaultGraphicsAPIs(BuildTarget.Android);
            _graphicsApis = PlayerSettings.GetGraphicsAPIs(BuildTarget.Android);
            _architectures = PlayerSettings.Android.targetArchitectures;
            _minimumSdk = PlayerSettings.Android.minSdkVersion;
        }

        public void Apply()
        {
            _applicationEntryProperty = typeof(PlayerSettings.Android).GetProperty(
                "applicationEntry",
                BindingFlags.Public | BindingFlags.Static);
            if (_applicationEntryProperty == null || !_applicationEntryProperty.PropertyType.IsEnum)
                throw new NotSupportedException(
                    "Android GameActivity is required for the ARM64 Vulkan release, but this Unity editor does not expose PlayerSettings.Android.applicationEntry.");

            _applicationEntry = _applicationEntryProperty.GetValue(null);
            var gameActivity = Enum.Parse(_applicationEntryProperty.PropertyType, "GameActivity", ignoreCase: false);
            PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
            PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, new[] { GraphicsDeviceType.Vulkan });
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel24;
            _applicationEntryProperty.SetValue(null, gameActivity);
            _applied = true;
        }

        public void Dispose()
        {
            if (!_applied)
                return;

            try
            {
                if (_graphicsApis != null && _graphicsApis.Length > 0)
                {
                    PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, false);
                    PlayerSettings.SetGraphicsAPIs(BuildTarget.Android, _graphicsApis);
                }
                PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android, _useDefaultGraphicsApis);
                PlayerSettings.Android.targetArchitectures = _architectures;
                PlayerSettings.Android.minSdkVersion = _minimumSdk;
                _applicationEntryProperty?.SetValue(null, _applicationEntry);
                AssetDatabase.SaveAssets();
            }
            finally
            {
                _applied = false;
            }
        }
    }

    private sealed class MacOsAppleDoubleBuildGuard : IDisposable
    {
        private const string CopyFileDisableEnvironmentVariable = "COPYFILE_DISABLE";
        private readonly string _previousValue;
        private bool _applied;

        public MacOsAppleDoubleBuildGuard()
        {
            _previousValue = Environment.GetEnvironmentVariable(CopyFileDisableEnvironmentVariable);
        }

        public void Apply()
        {
            // On non-APFS volumes macOS otherwise emits ._ sidecars for copied DLLs.
            // Unity's Bee/IL2CPP pipeline mistakes those sidecars for managed assemblies.
            Environment.SetEnvironmentVariable(CopyFileDisableEnvironmentVariable, "1");
            _applied = true;
            ResetMacOsBeeCache();
        }

        public void Dispose()
        {
            if (!_applied)
                return;

            Environment.SetEnvironmentVariable(CopyFileDisableEnvironmentVariable, _previousValue);
            _applied = false;
        }
    }

    [Serializable]
    private sealed class BundledModelManifest
    {
        public string[] groups;
    }
}
#endif
