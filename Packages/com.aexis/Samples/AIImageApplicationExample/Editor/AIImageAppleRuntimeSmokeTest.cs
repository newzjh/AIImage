#if UNITY_EDITOR
using System;
using System.IO;
using Aexis.Samples.Json.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;
using UnityEngine.Rendering;
using Stopwatch = System.Diagnostics.Stopwatch;

public static class AIImageAppleRuntimeSmokeTest
{
    private const string BuildOutputEnvironmentVariable = "AIIMAGE_APPLE_RUNTIME_SMOKE_BUILD_OUTPUT";
    private const string BuildReportEnvironmentVariable = "AIIMAGE_APPLE_RUNTIME_SMOKE_BUILD_REPORT";
    private const string IosAutorunDefine = "AEXIS_IOS_RUNTIME_SMOKE_AUTORUN";
    private const string RunnerInputDirectoryName = "aiimage-smoke";

    public static void BuildMacOSMetalRuntimeSmokeBatch()
    {
        BuildAppleRuntimeSmokePlayer(BuildTarget.StandaloneOSX, includeIosAutorunDefine: false);
    }

    public static void BuildIosMetalRuntimeSmokeBatch()
    {
        BuildAppleRuntimeSmokePlayer(BuildTarget.iOS, includeIosAutorunDefine: true);
    }

    private static void BuildAppleRuntimeSmokePlayer(BuildTarget target, bool includeIosAutorunDefine)
    {
        var stopwatch = Stopwatch.StartNew();
        var reportPath = ResolveReportPath(target);
        var report = new JObject
        {
            ["schema"] = "aiimage.apple-runtime-smoke-build/v1",
            ["target"] = target.ToString(),
            ["unity_version"] = Application.unityVersion,
            ["host_operating_system"] = SystemInfo.operatingSystem,
            ["required_graphics_api"] = GraphicsDeviceType.Metal.ToString(),
            ["ios_autorun"] = includeIosAutorunDefine,
            ["report_path"] = reportPath,
            ["started_utc"] = DateTime.UtcNow.ToString("O"),
            ["valid"] = false
        };

        var group = BuildPipeline.GetBuildTargetGroup(target);
        var useDefaultGraphicsApis = PlayerSettings.GetUseDefaultGraphicsAPIs(target);
        var graphicsApis = PlayerSettings.GetGraphicsAPIs(target);
        var originalDefines = PlayerSettings.GetScriptingDefineSymbolsForGroup(group) ?? string.Empty;
        var output = ResolveBuildOutput(target);
        var succeeded = false;
        BuildReport buildReport = null;
        try
        {
            if (!BuildPipeline.IsBuildTargetSupported(group, target))
                throw new NotSupportedException(target + " PlaybackEngine is not installed on " + SystemInfo.operatingSystem + ".");

            PlayerSettings.SetUseDefaultGraphicsAPIs(target, false);
            PlayerSettings.SetGraphicsAPIs(target, new[] { GraphicsDeviceType.Metal });
            if (includeIosAutorunDefine)
                PlayerSettings.SetScriptingDefineSymbolsForGroup(group, AddDefine(originalDefines, IosAutorunDefine));

            var configuredGraphicsApis = PlayerSettings.GetGraphicsAPIs(target);
            if (configuredGraphicsApis.Length != 1 || configuredGraphicsApis[0] != GraphicsDeviceType.Metal)
                throw new InvalidOperationException(target + " did not retain an exclusive Metal graphics API configuration.");

            Directory.CreateDirectory(target == BuildTarget.StandaloneOSX
                ? Path.GetDirectoryName(output)
                : output);
            buildReport = AIImageReducedModelBuild.BuildMain2(
                target,
                output,
                BuildOptions.Development | BuildOptions.CleanBuildCache);
            report["build"] = SerializeBuildReport(buildReport);
            if (buildReport.summary.result != BuildResult.Succeeded || buildReport.summary.totalErrors != 0)
                throw new InvalidOperationException(
                    target + " runtime-smoke Player build failed: result=" + buildReport.summary.result
                    + " errors=" + buildReport.summary.totalErrors + ".");

            CopyRunnerInputsToPlayer(target, output, report);
            report["player_output"] = output;
            report["valid"] = true;
            succeeded = true;
        }
        catch (Exception exception)
        {
            report["error"] = exception.ToString();
            Debug.LogException(exception);
        }
        finally
        {
            if (graphicsApis != null && graphicsApis.Length > 0)
            {
                PlayerSettings.SetUseDefaultGraphicsAPIs(target, false);
                PlayerSettings.SetGraphicsAPIs(target, graphicsApis);
            }
            PlayerSettings.SetUseDefaultGraphicsAPIs(target, useDefaultGraphicsApis);
            if (includeIosAutorunDefine)
                PlayerSettings.SetScriptingDefineSymbolsForGroup(group, originalDefines);

            stopwatch.Stop();
            report["ended_utc"] = DateTime.UtcNow.ToString("O");
            report["elapsed_ms"] = stopwatch.ElapsedMilliseconds;
            if (buildReport != null && report["build"] == null)
                report["build"] = SerializeBuildReport(buildReport);
            Directory.CreateDirectory(Path.GetDirectoryName(reportPath));
            File.WriteAllText(reportPath, report.ToString());
            AssetDatabase.SaveAssets();
            Debug.Log("Apple runtime-smoke build report: " + reportPath + " valid=" + succeeded);
            if (Application.isBatchMode)
                EditorApplication.Exit(succeeded ? 0 : 1);
        }
    }

    private static void CopyRunnerInputsToPlayer(BuildTarget target, string output, JObject report)
    {
        var faceInput = Path.Combine(ProjectRoot, "ref", "02.png");
        var sceneInput = Path.Combine(ProjectRoot, "ref", "03.jpg");
        if (!File.Exists(faceInput) || !File.Exists(sceneInput))
            throw new FileNotFoundException("Apple runtime-smoke inputs are missing from the repository ref directory.");

        var streamingRoot = target == BuildTarget.StandaloneOSX
            ? Path.Combine(output, "Contents", "Resources", "Data", "StreamingAssets")
            : Path.Combine(output, "Data", "Raw");
        if (!Directory.Exists(streamingRoot))
            throw new DirectoryNotFoundException("Generated Player StreamingAssets directory is missing: " + streamingRoot);

        var inputDirectory = Path.Combine(streamingRoot, RunnerInputDirectoryName);
        Directory.CreateDirectory(inputDirectory);
        var faceDestination = Path.Combine(inputDirectory, "face.png");
        var sceneDestination = Path.Combine(inputDirectory, "scene.jpg");
        File.Copy(faceInput, faceDestination, true);
        File.Copy(sceneInput, sceneDestination, true);
        report["runner_inputs"] = new JObject
        {
            ["source_face"] = faceInput,
            ["source_scene"] = sceneInput,
            ["player_directory"] = inputDirectory,
            ["face_destination"] = faceDestination,
            ["scene_destination"] = sceneDestination
        };
    }

    private static JObject SerializeBuildReport(BuildReport buildReport)
    {
        var summary = buildReport.summary;
        return new JObject
        {
            ["result"] = summary.result.ToString(),
            ["platform"] = summary.platform.ToString(),
            ["platform_group"] = summary.platformGroup.ToString(),
            ["output_path"] = summary.outputPath,
            ["total_bytes"] = summary.totalSize,
            ["total_time_ms"] = summary.totalTime.TotalMilliseconds,
            ["total_errors"] = summary.totalErrors,
            ["total_warnings"] = summary.totalWarnings
        };
    }

    private static string ResolveBuildOutput(BuildTarget target)
    {
        var configured = Environment.GetEnvironmentVariable(BuildOutputEnvironmentVariable);
        if (!string.IsNullOrWhiteSpace(configured))
            return Path.GetFullPath(configured);

        var root = Path.Combine(ProjectRoot, "Builds", "AexisRuntimeSmoke");
        return target == BuildTarget.StandaloneOSX
            ? Path.Combine(root, "macos", "AexisRuntimeSmoke.app")
            : Path.Combine(root, "ios", "AexisRuntimeSmokeXcode");
    }

    private static string ResolveReportPath(BuildTarget target)
    {
        var configured = Environment.GetEnvironmentVariable(BuildReportEnvironmentVariable);
        if (!string.IsNullOrWhiteSpace(configured))
            return Path.GetFullPath(configured);

        var name = target == BuildTarget.StandaloneOSX
            ? "AIImage_MacOS_RuntimeSmoke_Build.json"
            : "AIImage_iOS_RuntimeSmoke_Build.json";
        return Path.Combine(ProjectRoot, "Logs", name);
    }

    private static string AddDefine(string defines, string define)
    {
        var values = (defines ?? string.Empty).Split(new[] { ';' }, StringSplitOptions.RemoveEmptyEntries);
        foreach (var value in values)
        {
            if (string.Equals(value.Trim(), define, StringComparison.Ordinal))
                return string.Join(";", values);
        }

        return string.IsNullOrWhiteSpace(defines) ? define : defines.TrimEnd(';') + ";" + define;
    }

    private static string ProjectRoot => Path.GetFullPath(Path.Combine(Application.dataPath, ".."));
}
#endif
