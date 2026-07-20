using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using NcnnCompute;
using Newtonsoft.Json.Linq;
using UnityEditor;
using AIImage.Qwen35;
using UnityEditor.Build.Reporting;
using UnityEngine;
using UnityEngine.Rendering;
using Debug = UnityEngine.Debug;

public static class Qwen35MobilePlatformBuild
{
    private const string DefaultScene = "Assets/Scenes/Main2.unity";
    private const string ModelDirectoryEnvironmentVariable = "AIIMAGE_QWEN35_MODEL_DIR";
    private const string BuildOutputEnvironmentVariable = "AIIMAGE_QWEN35_BUILD_OUTPUT";
    private const string BuildReportEnvironmentVariable = "AIIMAGE_QWEN35_BUILD_REPORT";
    private const string IncludeStreamingAssetsEnvironmentVariable = "AIIMAGE_QWEN35_INCLUDE_EXISTING_STREAMING_ASSETS";

    public static void RunDeviceCompatibilityReportBatch()
    {
        var stopwatch = Stopwatch.StartNew();
        var reportPath = ResolveReportPath("qwen35_editor_device_compatibility.json");
        var root = CreateReport("Editor", reportPath);
        try
        {
            var contract = ValidateMobileAssets(root);
            var compatibility = Qwen35DeviceCompatibility.Evaluate(contract);
            root["device"] = compatibility.ToJson();
            root["valid"] = compatibility.Supported;
            if (!compatibility.Supported)
                throw new NotSupportedException(string.Join("\n", compatibility.UnsupportedReasons));
        }
        catch (Exception exception)
        {
            root["valid"] = false;
            root["error"] = exception.ToString();
            throw;
        }
        finally
        {
            FinishReport(root, reportPath, stopwatch);
        }
    }

    public static void BuildAndroidVulkanBatch()
    {
        BuildMobilePlayer(BuildTarget.Android, GraphicsDeviceType.Vulkan);
    }

    public static void BuildIosMetalBatch()
    {
        BuildMobilePlayer(BuildTarget.iOS, GraphicsDeviceType.Metal);
    }

    private static void BuildMobilePlayer(BuildTarget target, GraphicsDeviceType requiredGraphicsApi)
    {
        var stopwatch = Stopwatch.StartNew();
        var targetName = target == BuildTarget.Android ? "android_vulkan" : "ios_metal";
        var reportPath = ResolveReportPath("qwen35_" + targetName + "_build.json");
        var root = CreateReport(target.ToString(), reportPath);
        BuildReport buildReport = null;

        var useDefaultGraphicsApis = PlayerSettings.GetUseDefaultGraphicsAPIs(target);
        var graphicsApis = PlayerSettings.GetGraphicsAPIs(target);
        var androidArchitecture = PlayerSettings.Android.targetArchitectures;
        var androidMinSdk = PlayerSettings.Android.minSdkVersion;
        var androidBuildAppBundle = EditorUserBuildSettings.buildAppBundle;
        try
        {
            ValidateMobileAssets(root);
            var group = BuildPipeline.GetBuildTargetGroup(target);
            if (!BuildPipeline.IsBuildTargetSupported(group, target))
                throw new NotSupportedException(
                    target + " PlaybackEngine is not installed or is unsupported on host "
                    + SystemInfo.operatingSystem + ".");

            PlayerSettings.SetUseDefaultGraphicsAPIs(target, false);
            PlayerSettings.SetGraphicsAPIs(target, new[] { requiredGraphicsApi });
            if (target == BuildTarget.Android)
            {
                PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
                PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel24;
                EditorUserBuildSettings.buildAppBundle = false;
            }

            var configuredApis = PlayerSettings.GetGraphicsAPIs(target);
            if (configuredApis.Length != 1 || configuredApis[0] != requiredGraphicsApi)
                throw new InvalidOperationException(
                    target + " graphics API configuration did not retain exclusive " + requiredGraphicsApi + ".");

            var scene = Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_BUILD_SCENE") ?? DefaultScene;
            if (!File.Exists(Path.Combine(ProjectRoot, scene.Replace('/', Path.DirectorySeparatorChar))))
                throw new FileNotFoundException("Qwen3.5 mobile validation scene is missing.", scene);
            var output = ResolveBuildOutput(target);
            Directory.CreateDirectory(target == BuildTarget.Android ? Path.GetDirectoryName(output) : output);

            root["configuration"] = new JObject
            {
                ["scene"] = scene,
                ["output"] = output,
                ["required_graphics_api"] = requiredGraphicsApi.ToString(),
                ["configured_graphics_apis"] = new JArray(configuredApis.Select(api => api.ToString())),
                ["android_architecture"] = target == BuildTarget.Android ? PlayerSettings.Android.targetArchitectures.ToString() : null,
                ["android_min_sdk"] = target == BuildTarget.Android ? PlayerSettings.Android.minSdkVersion.ToString() : null,
                ["development_build"] = true,
                ["clean_build_cache"] = true,
                ["qwen_model_bundled_in_player"] = false
            };

            using (ExcludeUnrelatedStreamingAssets(root))
            {
                buildReport = BuildPipeline.BuildPlayer(new BuildPlayerOptions
                {
                    scenes = new[] { scene },
                    locationPathName = output,
                    target = target,
                    options = BuildOptions.Development | BuildOptions.CompressWithLz4 | BuildOptions.CleanBuildCache
                });
            }
            root["build"] = SerializeBuildReport(buildReport);
            root["valid"] = buildReport.summary.result == BuildResult.Succeeded
                && buildReport.summary.totalErrors == 0;
            if (!(bool)root["valid"])
                throw new InvalidOperationException(
                    target + " build failed or emitted build errors: result=" + buildReport.summary.result
                    + " errors=" + buildReport.summary.totalErrors);
        }
        catch (Exception exception)
        {
            root["valid"] = false;
            root["error"] = exception.ToString();
            if (buildReport != null && root["build"] == null)
                root["build"] = SerializeBuildReport(buildReport);
            throw;
        }
        finally
        {
            if (graphicsApis != null && graphicsApis.Length > 0)
            {
                PlayerSettings.SetUseDefaultGraphicsAPIs(target, false);
                PlayerSettings.SetGraphicsAPIs(target, graphicsApis);
            }
            PlayerSettings.SetUseDefaultGraphicsAPIs(target, useDefaultGraphicsApis);
            PlayerSettings.Android.targetArchitectures = androidArchitecture;
            PlayerSettings.Android.minSdkVersion = androidMinSdk;
            EditorUserBuildSettings.buildAppBundle = androidBuildAppBundle;
            AssetDatabase.SaveAssets();
            FinishReport(root, reportPath, stopwatch);
        }
    }

    private static Qwen35ModelContract ValidateMobileAssets(JObject report)
    {
        var modelDirectory = ResolveModelDirectory();
        var contract = Qwen35ModelContract.Validate(modelDirectory, requireWeights: true);
        report["model_contract_valid"] = contract.IsValid;
        report["model_contract_errors"] = new JArray(contract.Errors);
        if (!contract.IsValid)
            throw new InvalidDataException("Qwen3.5 model contract failed:\n" + string.Join("\n", contract.Errors));

        var mobileAssets = Qwen35MobileAssetSet.TryLoad(modelDirectory, verifyHashes: true);
        if (mobileAssets == null || !mobileAssets.WeightOnly)
            throw new InvalidDataException("Validated q8 mobile assets are required: " + modelDirectory);
        var memoryPolicy = Qwen35MobileMemoryPolicy.Evaluate(contract);
        report["memory_policy"] = memoryPolicy.ToJson();
        if (!memoryPolicy.DeliveryEligible)
            throw new NotSupportedException(string.Join("\n", memoryPolicy.UnsupportedReasons));

        var manifestPath = Path.Combine(modelDirectory, Qwen35MobileAssetSet.ManifestFileName);
        var manifest = JObject.Parse(File.ReadAllText(manifestPath));
        var shardCount = manifest["logical_files"]
            .Children<JProperty>()
            .Sum(property => ((JArray)property.Value["parts"]).Count);
        report["asset_delivery"] = new JObject
        {
            ["mode"] = "external-sharded-file-set",
            ["bundled_in_base_player"] = false,
            ["runtime_destination"] = "persistentDataPath/qwen3.5_0.8b_mobile_q8",
            ["source_directory"] = modelDirectory,
            ["manifest"] = manifestPath,
            ["manifest_sha256"] = ComputeSha256(manifestPath),
            ["stored_weight_bytes"] = mobileAssets.StoredWeightBytes,
            ["shard_count"] = shardCount,
            ["all_shard_hashes_verified"] = true,
            ["source_base_url"] = (string)manifest["source_base_url"]
        };
        report["shader"] = DescribeInferenceShader();
        return contract;
    }

    private static JObject DescribeInferenceShader()
    {
        var path = Path.Combine(ProjectRoot,
            "Packages/com.aiimage.inference.kernels/Runtime/Resources/NcnnCompute.compute".Replace('/', Path.DirectorySeparatorChar));
        if (!File.Exists(path)) throw new FileNotFoundException("NCNN ComputeShader is missing.", path);
        var source = File.ReadAllText(path);
        return new JObject
        {
            ["path"] = path,
            ["sha256"] = ComputeSha256(path),
            ["uses_compute_shader"] = true,
            ["contains_q8_gemm_constants"] = source.Contains("_MatBInt8Packed"),
            ["contains_q8_embed_constants"] = source.Contains("_EmbedWInt8Packed"),
            ["contains_short_conv_kernel"] = source.Contains("Qwen35ShortConvPack4"),
            ["contains_gated_delta_rule_kernel"] = source.Contains("Qwen35GatedDeltaRulePack4"),
            ["activation_storage"] = "pack4 RenderTexture/ComputeTexture",
            ["temporary_compute_buffer_fallback_allowed"] = false
        };
    }

    private static JObject SerializeBuildReport(BuildReport report)
    {
        const int maxMessages = 256;
        var summary = report.summary;
        var messages = new JArray();
        var issueCount = 0;
        foreach (var step in report.steps)
        foreach (var message in step.messages)
            if (message.type == LogType.Error || message.type == LogType.Exception || message.type == LogType.Warning)
                issueCount++;
        foreach (var step in report.steps)
        foreach (var message in step.messages)
            if ((message.type == LogType.Error || message.type == LogType.Exception) && messages.Count < maxMessages)
                messages.Add(new JObject
                {
                    ["step"] = step.name,
                    ["type"] = message.type.ToString(),
                    ["content"] = message.content
                });
        foreach (var step in report.steps)
        foreach (var message in step.messages)
            if (message.type == LogType.Warning && messages.Count < maxMessages)
                messages.Add(new JObject
                {
                    ["step"] = step.name,
                    ["type"] = message.type.ToString(),
                    ["content"] = message.content
                });
        return new JObject
        {
            ["result"] = summary.result.ToString(),
            ["platform"] = summary.platform.ToString(),
            ["platform_group"] = summary.platformGroup.ToString(),
            ["output_path"] = summary.outputPath,
            ["total_bytes"] = summary.totalSize,
            ["total_time_ms"] = summary.totalTime.TotalMilliseconds,
            ["total_errors"] = summary.totalErrors,
            ["total_warnings"] = summary.totalWarnings,
            ["issue_message_count"] = issueCount,
            ["serialized_message_count"] = messages.Count,
            ["messages_truncated"] = issueCount > messages.Count,
            ["messages"] = messages
        };
    }

    private static IDisposable ExcludeUnrelatedStreamingAssets(JObject report)
    {
        if (string.Equals(
            Environment.GetEnvironmentVariable(IncludeStreamingAssetsEnvironmentVariable),
            "1",
            StringComparison.Ordinal))
        {
            report["existing_streaming_assets"] = new JObject
            {
                ["excluded_from_qwen_build"] = false,
                ["reason"] = IncludeStreamingAssetsEnvironmentVariable + "=1"
            };
            return EmptyDisposable.Instance;
        }
        return StreamingAssetsExclusion.Begin(ProjectRoot, report);
    }

    private static JObject CreateReport(string target, string reportPath)
    {
        return new JObject
        {
            ["schema"] = "qwen35.unity-mobile-build/v1",
            ["target"] = target,
            ["command"] = Environment.CommandLine,
            ["report_path"] = reportPath,
            ["started_utc"] = DateTime.UtcNow.ToString("O"),
            ["unity_version"] = Application.unityVersion,
            ["host_operating_system"] = SystemInfo.operatingSystem,
            ["host_platform"] = Application.platform.ToString(),
            ["valid"] = false
        };
    }

    private static void FinishReport(JObject report, string path, Stopwatch stopwatch)
    {
        stopwatch.Stop();
        report["ended_utc"] = DateTime.UtcNow.ToString("O");
        report["elapsed_ms"] = stopwatch.ElapsedMilliseconds;
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, report.ToString());
        Debug.Log("Qwen3.5 mobile validation report: " + path);
    }

    private static string ResolveModelDirectory()
    {
        var configured = Environment.GetEnvironmentVariable(ModelDirectoryEnvironmentVariable);
        return Path.GetFullPath(string.IsNullOrWhiteSpace(configured)
            ? Path.Combine(ProjectRoot, "Tools", "Qwen35NcnnBaseline", "_models", "qwen3.5_0.8b_mobile_q8")
            : configured);
    }

    private static string ResolveBuildOutput(BuildTarget target)
    {
        var configured = Environment.GetEnvironmentVariable(BuildOutputEnvironmentVariable);
        if (!string.IsNullOrWhiteSpace(configured)) return Path.GetFullPath(configured);
        var root = Path.Combine(ProjectRoot, "Tools", "Qwen35NcnnBaseline", "build", "mobile");
        return target == BuildTarget.Android
            ? Path.Combine(root, "android", "Qwen35MobileValidation.apk")
            : Path.Combine(root, "ios", "Qwen35MobileValidationXcode");
    }

    private static string ResolveReportPath(string fileName)
    {
        var configured = Environment.GetEnvironmentVariable(BuildReportEnvironmentVariable);
        return Path.GetFullPath(string.IsNullOrWhiteSpace(configured)
            ? Path.Combine(ProjectRoot, "Tools", "Qwen35NcnnBaseline", "reports", fileName)
            : configured);
    }

    private static string ComputeSha256(string path)
    {
        using (var sha = SHA256.Create())
        using (var stream = File.OpenRead(path))
            return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", string.Empty).ToLowerInvariant();
    }

    private static string ProjectRoot => Path.GetFullPath(Path.Combine(Application.dataPath, ".."));

    private sealed class StreamingAssetsExclusion : IDisposable
    {
        private readonly string _sourceDirectory;
        private readonly string _sourceMeta;
        private readonly string _backupDirectory;
        private readonly string _backupMeta;
        private readonly JObject _report;
        private bool _movedDirectory;
        private bool _movedMeta;

        private StreamingAssetsExclusion(string projectRoot, JObject report)
        {
            _sourceDirectory = Path.Combine(projectRoot, "Assets", "StreamingAssets");
            _sourceMeta = _sourceDirectory + ".meta";
            var backupRoot = Path.Combine(projectRoot, "Temp", "Qwen35MobileBuildStreamingAssets");
            _backupDirectory = Path.Combine(backupRoot, "StreamingAssets");
            _backupMeta = Path.Combine(backupRoot, "StreamingAssets.meta");
            _report = report;
        }

        public static StreamingAssetsExclusion Begin(string projectRoot, JObject report)
        {
            var exclusion = new StreamingAssetsExclusion(projectRoot, report);
            exclusion.MoveOut();
            return exclusion;
        }

        private void MoveOut()
        {
            var details = new JObject
            {
                ["excluded_from_qwen_build"] = Directory.Exists(_sourceDirectory),
                ["source"] = _sourceDirectory,
                ["restored"] = false
            };
            _report["existing_streaming_assets"] = details;
            if (!Directory.Exists(_sourceDirectory))
            {
                details["file_count"] = 0;
                details["bytes"] = 0;
                details["restored"] = true;
                return;
            }
            if (Directory.Exists(_backupDirectory) || File.Exists(_backupMeta))
                throw new IOException("Qwen3.5 StreamingAssets backup already exists: " + Path.GetDirectoryName(_backupDirectory));

            long bytes = 0;
            var count = 0;
            foreach (var file in Directory.EnumerateFiles(_sourceDirectory, "*", SearchOption.AllDirectories))
            {
                bytes += new FileInfo(file).Length;
                count++;
            }
            details["file_count"] = count;
            details["bytes"] = bytes;
            Directory.CreateDirectory(Path.GetDirectoryName(_backupDirectory));
            Directory.Move(_sourceDirectory, _backupDirectory);
            _movedDirectory = true;
            if (File.Exists(_sourceMeta))
            {
                File.Move(_sourceMeta, _backupMeta);
                _movedMeta = true;
            }
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
        }

        public void Dispose()
        {
            if (_movedDirectory)
            {
                if (Directory.Exists(_sourceDirectory))
                    throw new IOException("Cannot restore Qwen3.5 build StreamingAssets because the source path was recreated: " + _sourceDirectory);
                Directory.Move(_backupDirectory, _sourceDirectory);
                _movedDirectory = false;
            }
            if (_movedMeta)
            {
                if (File.Exists(_sourceMeta))
                    throw new IOException("Cannot restore Qwen3.5 build StreamingAssets metadata because the source path was recreated: " + _sourceMeta);
                File.Move(_backupMeta, _sourceMeta);
                _movedMeta = false;
            }
            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);
            ((JObject)_report["existing_streaming_assets"])["restored"] = true;
        }
    }

    private sealed class EmptyDisposable : IDisposable
    {
        public static readonly EmptyDisposable Instance = new EmptyDisposable();
        public void Dispose() { }
    }
}
